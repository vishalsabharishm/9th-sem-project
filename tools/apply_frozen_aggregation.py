#!/usr/bin/env python3
"""
tools/apply_frozen_aggregation.py

Applies the FROZEN aggregation rule (from temporal_risk/frozen_aggregation.json,
produced by tools/select_temporal_aggregation.py on the carve split only) to the
394-clip primary split (temporal_risk/primary_window_scores.csv), and reports
clip-level metrics.

This script performs NO selection and NO tuning. It reads the frozen rule and
its parameters exactly as chosen, applies them mechanically to primary, and
reports what happens -- including if the result is worse than hoped. It also
prints a comparison against the immutable clean baseline
(docs/clean_baseline_results.md, threshold 0.16 on whole-clip inference) so
the two numbers are never confused for each other: they are different
inference regimes (whole-clip vs. sliding-window aggregation) evaluated on
the same 394 clips.

Usage:
    python tools/apply_frozen_aggregation.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_CSV = REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"
FROZEN_JSON = REPO_ROOT / "temporal_risk" / "frozen_aggregation.json"
OUT_JSON = REPO_ROOT / "temporal_risk" / "primary_aggregation_result.json"

# The immutable clean baseline (whole-clip inference, NOT window aggregation).
# See docs/clean_baseline_results.md. Reproduced here only for side-by-side
# comparison in this script's printed output -- not read from or written to.
CLEAN_BASELINE_CLIP_LEVEL = {
    "threshold": 0.16,
    "accuracy": 0.8324873096446701,
    "precision": 0.7913043478260869,
    "recall": 0.91,
    "f1": 0.8465116279069768,
    "specificity": 0.7526,
    "tp": 182, "fp": 48, "tn": 146, "fn": 18,
}


def load_split(path: Path):
    clips: dict[str, list[float]] = defaultdict(list)
    true_label: dict[str, str] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = row["clip"]
            prob = float(row["fight_probability"])
            clips[clip].append(prob)
            lbl = row["true_label"]
            if clip in true_label:
                if true_label[clip] != lbl:
                    raise ValueError(f"inconsistent true_label for clip {clip}")
            else:
                true_label[clip] = lbl
    return clips, true_label


def confusion(preds: dict[str, bool], labels: dict[str, str]):
    tp = fp = tn = fn = 0
    for clip, pred in preds.items():
        actual = labels[clip] == "Fight"
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def compute_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "fpr": fpr,
        "specificity": specificity, "accuracy": accuracy, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def apply_rule(clips, rule, params):
    preds = {}
    if rule == "any":
        thr = params["window_threshold"]
        for clip, probs in clips.items():
            preds[clip] = any(p >= thr for p in probs)
    elif rule == "k_of_n":
        thr = params["window_threshold"]
        k = params["k"]
        for clip, probs in clips.items():
            preds[clip] = sum(1 for p in probs if p >= thr) >= k
    elif rule == "max":
        thr = params["threshold"]
        for clip, probs in clips.items():
            preds[clip] = max(probs) >= thr
    elif rule == "mean":
        thr = params["threshold"]
        for clip, probs in clips.items():
            preds[clip] = (sum(probs) / len(probs)) >= thr
    else:
        raise ValueError(f"unknown rule: {rule}")
    return preds


def main():
    if not FROZEN_JSON.exists():
        print(f"ERROR: {FROZEN_JSON} not found. Run tools/select_temporal_aggregation.py first.",
              file=sys.stderr)
        sys.exit(1)
    if not PRIMARY_CSV.exists():
        print(f"ERROR: {PRIMARY_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    frozen = json.loads(FROZEN_JSON.read_text())
    selected = frozen["selected"]
    rule = selected["rule"]
    params = selected["params"]
    carve_metrics = selected["metrics"]

    clips, labels = load_split(PRIMARY_CSV)
    n_clips = len(clips)
    n_fight = sum(1 for l in labels.values() if l == "Fight")
    n_nonfight = n_clips - n_fight
    print(f"Loaded {n_clips} clips from PRIMARY split ({n_fight} Fight, {n_nonfight} NonFight)")
    print(f"Applying FROZEN rule (selected on carve, unchanged): {rule} {params}\n")

    preds = apply_rule(clips, rule, params)
    tp, fp, tn, fn = confusion(preds, labels)
    primary_metrics = compute_metrics(tp, fp, tn, fn)

    print("Primary-split window-aggregation result:")
    print(json.dumps(primary_metrics, indent=2))

    print("\nCarve-split result (for reference, same rule):")
    print(json.dumps(carve_metrics, indent=2))

    print("\nImmutable clean baseline (whole-clip inference, threshold 0.16) -- different regime, for context only:")
    print(json.dumps(CLEAN_BASELINE_CLIP_LEVEL, indent=2))

    delta_recall = primary_metrics["recall"] - carve_metrics["recall"]
    delta_precision = primary_metrics["precision"] - carve_metrics["precision"]
    print(f"\nCarve -> Primary generalization delta: "
          f"recall {delta_recall:+.4f}, precision {delta_precision:+.4f}")

    out = {
        "tool": "apply_frozen_aggregation",
        "frozen_rule_applied": selected,
        "no_tuning_performed": True,
        "primary_split": {
            "n_clips": n_clips,
            "n_fight": n_fight,
            "n_nonfight": n_nonfight,
            "metrics": primary_metrics,
        },
        "carve_split_reference": carve_metrics,
        "clean_baseline_whole_clip_reference": CLEAN_BASELINE_CLIP_LEVEL,
        "generalization_delta_carve_to_primary": {
            "recall": delta_recall,
            "precision": delta_precision,
        },
        "note": (
            "This result and the clean baseline are two different inference regimes "
            "evaluated on overlapping clip sets: whole-clip single-pass inference "
            "(clean baseline) vs. sliding-window (16-frame, stride 8, 17 windows/clip) "
            "score aggregation (this script). They are not directly substitutable and "
            "should not be quoted interchangeably in the report."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
