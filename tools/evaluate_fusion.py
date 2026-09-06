#!/usr/bin/env python3
"""
tools/evaluate_fusion.py

SUPERSEDED -- REFUSES TO RUN BY DEFAULT. See SUPERSEDED_NOTICE below.

Apply the pre-registered fusion protocol: calibrate on the FIT split, evaluate
once on the EVAL split, report paired statistics.

THE DISCIPLINE THIS ENFORCES
----------------------------
Thresholds are derived on the fit split only, at a false-positive budget matched
to the frozen temporal incumbent, and are then applied unchanged to the eval
split. The eval split is read once, for measurement. Nothing is re-fitted after
seeing it, no candidate is added, and the F2 weight stays at the 0.5/0.5 fixed
in the protocol. The candidate set, the spatial score, the operating point and
the endpoints were all frozen in temporal_risk/frozen_fusion_protocol.json
before any fresh-carve score existed.

The 240-clip carve is not used here: it has been spent three times already. The
394-clip primary split is not read at all.

WHAT IS COMPARED
----------------
    B0  temporal only, max >= 0.14                (frozen incumbent)
    B1  spatial only,  s >= theta_s               (baseline)
    F1  temporal OR (s >= theta_or)               (fusion)
    F2  0.5*pct(temporal) + 0.5*pct(s) >= theta_f (fusion)

Observations are the same clips decided by every candidate, so comparisons are
paired: exact McNemar on discordant pairs, never an independent-samples test.
Eval clips come from 114 source videos, so a source-clustered bootstrap is
reported next to the clip-level one.

Usage:
    python tools/evaluate_fusion.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from checkpoint_identity import file_sha256  # noqa: E402
from metric_intervals import source_video_id, wilson_interval  # noqa: E402

FUSION_DIR = REPO_ROOT / "outputs" / "fusion"
PROTOCOL = REPO_ROOT / "temporal_risk" / "frozen_fusion_protocol.json"
MANIFEST = REPO_ROOT / "temporal_risk" / "fresh_validation_carve.json"

TEMPORAL_THRESHOLD = 0.14      # frozen; never re-tuned here
F2_WEIGHT = 0.5                # fixed by protocol; not fitted
BOOTSTRAP, SEED = 2000, 42


class FusionError(RuntimeError):
    """Raised when the frozen protocol cannot be applied as written."""


def load_split(split: str) -> List[dict]:
    path = FUSION_DIR / f"fresh_carve_{split}_scores.json"
    if not path.is_file():
        raise FusionError(f"{path} not found; run tools/score_fresh_carve.py --split {split}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for record in payload["clips"]:
        if "error" in record:
            continue
        rows.append({
            "clip": record["clip"],
            "label": record["label"],
            "truth": 1 if record["label"] == "Fight" else 0,
            "temporal_max": record["temporal_max"],
            "spatial": record["speed_normalised_by_group_diagonal"],
            "source": source_video_id(record["clip"]),
        })
    return rows, payload


def confusion(rows: Sequence[dict], decisions: Dict[str, bool]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for row in rows:
        fired, positive = decisions[row["clip"]], row["truth"] == 1
        if positive and fired:
            tp += 1
        elif positive:
            fn += 1
        elif fired:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metrics(matrix: Dict[str, int]) -> dict:
    tp, fp, tn, fn = matrix["tp"], matrix["fp"], matrix["tn"], matrix["fn"]
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)
    accuracy = (tp + tn) / total if total else None
    out = {**matrix, "accuracy": accuracy, "precision": precision, "recall": recall,
           "specificity": tn / (tn + fp) if tn + fp else None,
           "fpr": fp / (fp + tn) if fp + tn else None, "f1": f1}
    if accuracy is not None:
        low, high = wilson_interval(tp + tn, total)
        out["accuracy_wilson_95"] = [low, high]
    if recall is not None:
        low, high = wilson_interval(tp, tp + fn)
        out["recall_wilson_95"] = [low, high]
    return out


def auc_scores(rows: Sequence[dict], key: str) -> Optional[float]:
    pos = [r[key] for r in rows if r["truth"] == 1 and r[key] is not None]
    neg = [r[key] for r in rows if r["truth"] == 0 and r[key] is not None]
    if len(pos) < 5 or len(neg) < 5:
        return None
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda x: x[0])
    ranks, index = [0.0] * len(merged), 0
    while index < len(merged):
        stop = index
        while stop + 1 < len(merged) and merged[stop + 1][0] == merged[index][0]:
            stop += 1
        average = (index + stop + 2) / 2.0
        for position in range(index, stop + 1):
            ranks[position] = average
        index = stop + 1
    positive_rank = sum(ranks[i] for i in range(len(merged)) if merged[i][1] == 1)
    return (positive_rank - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def average_precision(rows: Sequence[dict], key: str) -> Optional[float]:
    usable = [(r[key], r["truth"]) for r in rows if r[key] is not None]
    if not usable or all(t == 0 for _, t in usable):
        return None
    usable.sort(key=lambda x: -x[0])
    positives = sum(t for _, t in usable)
    hits = 0.0
    total = 0.0
    for index, (_, truth) in enumerate(usable, start=1):
        if truth:
            hits += 1
            total += hits / index
    return total / positives if positives else None


def mcnemar_exact(b: int, c: int) -> dict:
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "discordant": 0, "p_value": 1.0,
                "method": "exact binomial (no discordant pairs; reported as p = 1)"}
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return {"b": b, "c": c, "discordant": n, "p_value": min(1.0, 2.0 * tail),
            "method": "two-sided exact binomial on discordant pairs (McNemar)"}


def calibrate(rows: Sequence[dict], score_of, budget: int) -> dict:
    """Pick the threshold whose fit-split false-positive count matches the budget.

    Exhaustive over every distinct score present, so it cannot miss the optimum
    to grid resolution. A clip whose score is undefined never fires -- the
    protocol's abstention rule.
    """
    values = sorted({score_of(r) for r in rows if score_of(r) is not None})
    if not values:
        raise FusionError("no clip in the fit split has a defined score for this candidate.")
    best = None
    for threshold in values:
        decisions = {
            r["clip"]: (score_of(r) is not None and score_of(r) >= threshold) for r in rows
        }
        matrix = confusion(rows, decisions)
        key = (abs(matrix["fp"] - budget), -matrix["tp"], -threshold)
        if best is None or key < best[0]:
            best = (key, threshold, matrix)
    return {"threshold": best[1], "fit_matrix": best[2],
            "fit_false_positives": best[2]["fp"], "budget": budget}


def percentile_maker(reference: Sequence[float]):
    """Empirical CDF of the FIT split; eval values rank against it, not themselves."""
    ordered = sorted(v for v in reference if v is not None)

    def percentile(value: Optional[float]) -> Optional[float]:
        if value is None or not ordered:
            return None
        low, high = 0, len(ordered)
        while low < high:
            mid = (low + high) // 2
            if ordered[mid] < value:
                low = mid + 1
            else:
                high = mid
        return low / len(ordered)

    return percentile


def paired_bootstrap(rows, first, second, clusters=False, resamples=BOOTSTRAP, seed=SEED):
    """Accuracy difference (second - first), clips resampled as pairs."""
    generator = np.random.default_rng(seed)
    correct_first = np.array([first[r["clip"]] == (r["truth"] == 1) for r in rows])
    correct_second = np.array([second[r["clip"]] == (r["truth"] == 1) for r in rows])
    if clusters:
        groups = {}
        for index, row in enumerate(rows):
            groups.setdefault(row["source"], []).append(index)
        members = [np.array(v) for v in groups.values()]
    values = []
    for _ in range(resamples):
        if clusters:
            drawn = generator.integers(0, len(members), size=len(members))
            picked = np.concatenate([members[i] for i in drawn])
        else:
            picked = generator.integers(0, len(rows), size=len(rows))
        values.append(float(correct_second[picked].mean() - correct_first[picked].mean()))
    array = np.asarray(values)
    return {"difference": float(correct_second.mean() - correct_first.mean()),
            "ci_low": float(np.quantile(array, 0.025)),
            "ci_high": float(np.quantile(array, 0.975)),
            "resamples": resamples, "seed": seed,
            "method": "paired percentile bootstrap over "
                      + ("source videos" if clusters else "clips")}


SUPERSEDED_NOTICE = "\n".join([
    "REFUSED: this tool evaluates the fresh validation carve, which was proved to be",
    "R3D TRAINING data. holdout_validation is true with validation_fraction 0.15, so",
    "the 240-clip carve IS the holdout and the remainder this carve was drawn from is",
    "the training set. Measured memorization gap: +0.244 recall. Any number produced",
    "here would compare a memorized temporal branch against a non-memorized spatial",
    "one, and would be uninterpretable in either direction.",
    "",
    "Superseded by temporal_risk/frozen_confirmatory_fusion_protocol.json, which",
    "calibrates on the 240-clip carve (genuine R3D holdout) and reserves the primary",
    "394 for a single authorised confirmatory evaluation.",
    "",
    "See outputs/fusion/fusion_blocked_contamination.md. Pass",
    "--i-understand-this-carve-is-training-data only to reproduce the contamination",
    "evidence itself; it still yields no valid fusion result.",
])

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=FUSION_DIR / "fusion_evaluation.json")
    parser.add_argument("--i-understand-this-carve-is-training-data", action="store_true",
                        dest="acknowledged",
                        help="reproduce the contamination evidence; produces no valid result")
    args = parser.parse_args(argv)

    if not args.acknowledged:
        print(SUPERSEDED_NOTICE, file=sys.stderr)
        return 3

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    fit_rows, fit_payload = load_split("fit")
    eval_rows, eval_payload = load_split("eval")

    temporal = lambda r: r["temporal_max"]
    spatial = lambda r: r["spatial"]

    # B0 on the fit split fixes the alarm budget every other candidate matches.
    b0_fit = {r["clip"]: (r["temporal_max"] is not None
                          and r["temporal_max"] >= TEMPORAL_THRESHOLD) for r in fit_rows}
    b0_fit_matrix = confusion(fit_rows, b0_fit)
    budget = b0_fit_matrix["fp"]

    cal_b1 = calibrate(fit_rows, spatial, budget)

    def or_score(row, threshold):
        return (row["temporal_max"] is not None and row["temporal_max"] >= TEMPORAL_THRESHOLD) or \
               (row["spatial"] is not None and row["spatial"] >= threshold)

    # F1: sweep only the spatial arm of the OR, matching the COMBINED budget.
    candidates = sorted({r["spatial"] for r in fit_rows if r["spatial"] is not None})
    best = None
    for threshold in candidates:
        decisions = {r["clip"]: or_score(r, threshold) for r in fit_rows}
        matrix = confusion(fit_rows, decisions)
        key = (abs(matrix["fp"] - budget), -matrix["tp"], -threshold)
        if best is None or key < best[0]:
            best = (key, threshold, matrix)
    cal_f1 = {"threshold": best[1], "fit_matrix": best[2],
              "fit_false_positives": best[2]["fp"], "budget": budget}

    pct_t = percentile_maker([r["temporal_max"] for r in fit_rows])
    pct_s = percentile_maker([r["spatial"] for r in fit_rows])

    def f2_score(row):
        a, b = pct_t(row["temporal_max"]), pct_s(row["spatial"])
        if a is None:
            return None
        return F2_WEIGHT * a + (1 - F2_WEIGHT) * (b if b is not None else 0.0)

    cal_f2 = calibrate(fit_rows, f2_score, budget)

    decisions = {
        "B0_temporal_only": {r["clip"]: (r["temporal_max"] is not None
                                         and r["temporal_max"] >= TEMPORAL_THRESHOLD)
                             for r in eval_rows},
        "B1_spatial_only": {r["clip"]: (r["spatial"] is not None
                                        and r["spatial"] >= cal_b1["threshold"])
                            for r in eval_rows},
        "F1_or_gate": {r["clip"]: or_score(r, cal_f1["threshold"]) for r in eval_rows},
        "F2_rank_sum": {r["clip"]: (f2_score(r) is not None
                                    and f2_score(r) >= cal_f2["threshold"])
                        for r in eval_rows},
    }

    results = {}
    for name, decision in decisions.items():
        matrix = confusion(eval_rows, decision)
        entry = metrics(matrix)
        if name == "B0_temporal_only":
            entry["roc_auc"] = auc_scores(eval_rows, "temporal_max")
            entry["pr_auc"] = average_precision(eval_rows, "temporal_max")
        elif name == "B1_spatial_only":
            entry["roc_auc"] = auc_scores(eval_rows, "spatial")
            entry["pr_auc"] = average_precision(eval_rows, "spatial")
        results[name] = entry

    comparisons = {}
    for name in ("B1_spatial_only", "F1_or_gate", "F2_rank_sum"):
        base, other = decisions["B0_temporal_only"], decisions[name]
        overall_b = overall_c = 0
        fight_b = fight_c = 0
        nonfight_b = nonfight_c = 0
        for row in eval_rows:
            positive = row["truth"] == 1
            base_ok, other_ok = base[row["clip"]] == positive, other[row["clip"]] == positive
            if base_ok and not other_ok:
                overall_b += 1
            elif other_ok and not base_ok:
                overall_c += 1
            if positive:
                if base[row["clip"]] and not other[row["clip"]]:
                    fight_b += 1
                elif other[row["clip"]] and not base[row["clip"]]:
                    fight_c += 1
            else:
                if (not base[row["clip"]]) and other[row["clip"]]:
                    nonfight_b += 1
                elif base[row["clip"]] and not other[row["clip"]]:
                    nonfight_c += 1
        comparisons[f"{name}_vs_B0"] = {
            "overall": mcnemar_exact(overall_b, overall_c),
            "fight_detection": {
                **mcnemar_exact(fight_b, fight_c),
                "reading": "b = Fight clips B0 detects and this candidate misses; "
                           "c = Fight clips this candidate detects and B0 misses",
            },
            "nonfight_false_alarms": {
                **mcnemar_exact(nonfight_b, nonfight_c),
                "reading": "b = NonFight clips only this candidate false-alarms on; "
                           "c = NonFight clips only B0 false-alarms on",
            },
            "accuracy_difference_clip_level": paired_bootstrap(eval_rows, base, other),
            "accuracy_difference_source_level": paired_bootstrap(eval_rows, base, other, True),
            "extra_false_positive_clips": results[name]["fp"] - results["B0_temporal_only"]["fp"],
            "non_inferiority_margin_clips": 5,
        }

    undefined = [r for r in eval_rows if r["spatial"] is None]
    report = {
        "tool": "evaluate_fusion",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": file_sha256(PROTOCOL),
        "manifest_sha256": json.loads(MANIFEST.read_text(encoding="utf-8"))["manifest_sha256"],
        "fitting_split": "fit (100 clips) -- thresholds derived here only",
        "evaluation_split": "eval (240 clips) -- read once, nothing refitted",
        "primary_data_accessed": False,
        "existing_240_carve_used": False,
        "alarm_budget_false_positives": budget,
        "calibration": {"B1_spatial_only": cal_b1, "F1_or_gate": cal_f1, "F2_rank_sum": cal_f2,
                        "B0_fit_matrix": b0_fit_matrix},
        "eval_composition": {
            "clips": len(eval_rows),
            "fight": sum(1 for r in eval_rows if r["truth"] == 1),
            "nonfight": sum(1 for r in eval_rows if r["truth"] == 0),
            "sources": len({r["source"] for r in eval_rows}),
            "spatially_undefined": len(undefined),
            "spatially_undefined_by_class": {
                "Fight": sum(1 for r in undefined if r["truth"] == 1),
                "NonFight": sum(1 for r in undefined if r["truth"] == 0),
            },
        },
        "results": results,
        "paired_comparisons": comparisons,
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tool_sha256": file_sha256(Path(__file__).resolve()),
            "checkpoint_sha256": eval_payload["checkpoint_sha256"],
            "yolo_sha256": eval_payload["yolo_sha256"],
            "python": sys.version.split()[0],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("FUSION EVALUATION -- fresh validation carve")
    print("=" * 78)
    print(f"  fit budget: {budget} false positives   eval: {report['eval_composition']}")
    print(f"\n  {'candidate':<20}{'tp':>4}{'fp':>4}{'tn':>4}{'fn':>4}{'recall':>9}{'spec':>8}{'F1':>8}{'acc':>8}")
    for name, entry in results.items():
        print(f"  {name:<20}{entry['tp']:>4}{entry['fp']:>4}{entry['tn']:>4}{entry['fn']:>4}"
              f"{entry['recall']:>9.4f}{entry['specificity']:>8.4f}"
              f"{(entry['f1'] or 0):>8.4f}{entry['accuracy']:>8.4f}")
    print()
    for name, block in comparisons.items():
        fight = block["fight_detection"]
        print(f"  {name}: Fight b={fight['b']} c={fight['c']} p={fight['p_value']:.6g}  |  "
              f"overall p={block['overall']['p_value']:.6g}  |  "
              f"extra FP {block['extra_false_positive_clips']:+d}")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
