#!/usr/bin/env python3
"""
tools/select_temporal_aggregation.py

Selects a temporal aggregation rule (any / k-of-n / mean / max) for turning
per-window fight_probability scores into a clip-level Fight/NonFight
decision, using ONLY the 240-clip carve split
(temporal_risk/carve_window_scores.csv). The 394-clip primary split is
never read by this script -- that discipline is what keeps this selection
free of the leakage/bias problem Experiment 1 had (see
docs/temporal_baseline_results.md's own caveat on its threshold=0.14).

Selection rule: among candidates with precision >= MIN_PRECISION and
FPR <= MAX_FPR on the carve split, pick the one maximizing recall.
Ties broken by: fewer false positives, then higher F1, then rule
simplicity (any < k-of-n < max < mean).

Outputs temporal_risk/frozen_aggregation.json with the selected rule, its
parameters, its carve-split confusion matrix / metrics, and every eligible
candidate considered (for auditability).

Usage:
    python tools/select_temporal_aggregation.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CARVE_CSV = REPO_ROOT / "temporal_risk" / "carve_window_scores.csv"
OUT_JSON = REPO_ROOT / "temporal_risk" / "frozen_aggregation.json"

MIN_PRECISION = 0.80
MAX_FPR = 0.20
# The frozen clean-baseline decision threshold (docs/clean_baseline_results.md,
# temporal_risk/window_scoring_manifest.json: frozen_threshold_for_downstream_use_only).
# Used as the per-window "is this window a Fight window" vote threshold for the
# any / k-of-n rule families. mean/max aggregate the continuous scores first and
# sweep their own clip-level threshold, since averaging/maxing changes the scale.
WINDOW_VOTE_THRESHOLD = 0.16

SIMPLICITY_RANK = {"any": 0, "k_of_n": 1, "max": 2, "mean": 3}


def load_carve(path: Path):
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


def evaluate_vote_rule(clips, labels, k):
    preds = {}
    for clip, probs in clips.items():
        votes = sum(1 for p in probs if p >= WINDOW_VOTE_THRESHOLD)
        preds[clip] = votes >= k
    return confusion(preds, labels)


def evaluate_score_rule(clips, labels, agg_fn, threshold):
    preds = {}
    for clip, probs in clips.items():
        preds[clip] = agg_fn(probs) >= threshold
    return confusion(preds, labels)


def main():
    if not CARVE_CSV.exists():
        print(f"ERROR: {CARVE_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    clips, labels = load_carve(CARVE_CSV)
    n_clips = len(clips)
    window_counts = sorted(set(len(v) for v in clips.values()))
    n_fight = sum(1 for l in labels.values() if l == "Fight")
    n_nonfight = n_clips - n_fight
    print(f"Loaded {n_clips} clips from carve split ({n_fight} Fight, {n_nonfight} NonFight); "
          f"windows/clip observed: {window_counts}")

    candidates = []

    max_k = max(len(v) for v in clips.values())
    for k in range(1, max_k + 1):
        tp, fp, tn, fn = evaluate_vote_rule(clips, labels, k)
        m = compute_metrics(tp, fp, tn, fn)
        name = "any" if k == 1 else "k_of_n"
        candidates.append({
            "rule": name,
            "params": {"k": k, "n": max_k, "window_threshold": WINDOW_VOTE_THRESHOLD},
            "metrics": m,
        })

    for name, agg_fn in (("max", max), ("mean", lambda ps: sum(ps) / len(ps))):
        for t100 in range(1, 100):
            t = t100 / 100.0
            tp, fp, tn, fn = evaluate_score_rule(clips, labels, agg_fn, t)
            m = compute_metrics(tp, fp, tn, fn)
            candidates.append({
                "rule": name,
                "params": {"threshold": round(t, 2)},
                "metrics": m,
            })

    eligible = [c for c in candidates
                if c["metrics"]["precision"] >= MIN_PRECISION
                and c["metrics"]["fpr"] <= MAX_FPR]

    if not eligible:
        print("ERROR: no candidate satisfies precision>=0.80 and FPR<=0.20 on the carve split.",
              file=sys.stderr)
        sys.exit(1)

    def sort_key(c):
        m = c["metrics"]
        param_val = c["params"].get("k", c["params"].get("threshold", 0))
        return (
            -m["recall"],
            m["fp"],
            -m["f1"],
            SIMPLICITY_RANK.get(c["rule"], 99),
            param_val,
        )

    eligible.sort(key=sort_key)
    selected = eligible[0]

    frozen = {
        "tool": "select_temporal_aggregation",
        "selected_on": "carve split only (240 clips) -- primary split never read by this script",
        "selection_rule": {
            "min_precision": MIN_PRECISION,
            "max_false_positive_rate": MAX_FPR,
            "objective": "maximise recall within the alarm budget",
            "tie_breakers": [
                "fewer false positives",
                "higher F1",
                "rule simplicity (any < k_of_n < max < mean)",
            ],
        },
        "window_vote_threshold": WINDOW_VOTE_THRESHOLD,
        "selected": selected,
        "n_carve_clips": n_clips,
        "n_carve_fight_clips": n_fight,
        "n_carve_nonfight_clips": n_nonfight,
        "n_candidates_considered": len(candidates),
        "n_candidates_eligible": len(eligible),
        "top_10_eligible_candidates_sorted": eligible[:10],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(frozen, f, indent=2)

    print(f"\nSelected rule: {selected['rule']} params={selected['params']}")
    print(json.dumps(selected["metrics"], indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
