#!/usr/bin/env python3
"""
tools/verify_historical_metrics.py

Recompute the Experiment-1 evaluation metrics from the stored per-clip
predictions, using this repository's own ``temporal_metrics.binary_metrics``,
and diff the result against the historically recorded values.

WHY
---
Two separate things are checked at once, at zero cost:

1. **Are the recorded numbers internally consistent?** ``docs/temporal_baseline_results.json``
   was transcribed from a Kaggle notebook's output. ``predictions.csv`` is the
   raw per-clip output of that same run. If the recorded metrics can be
   re-derived from the raw predictions, the transcription is confirmed and the
   headline stops being an unverifiable claim.

2. **Is this project's hand-written metric code correct?** ``temporal_metrics``
   reimplements ROC-AUC (rank/Mann-Whitney) and PR-AUC (step-wise average
   precision) in numpy to avoid a scikit-learn dependency, and nothing has ever
   compared it against an independent implementation. Agreeing with a metric
   computed by scikit-learn inside a Kaggle notebook is exactly that
   comparison.

A disagreement is a real finding either way and is reported verbatim, never
rounded away.

READ-ONLY
---------
``predictions.csv`` is opened for reading only. Output goes to a new directory
that no historical artifact lives in.

Usage:
    python tools/verify_historical_metrics.py
    python tools/verify_historical_metrics.py --json outputs/step3_checkpoint_identity/historical_metric_verification.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rwf2000_config import LABEL_TO_INDEX  # noqa: E402
from temporal_metrics import binary_metrics  # noqa: E402

PREDICTIONS_CSV = REPO_ROOT / "outputs" / "temporal_violence" / "evaluation" / "predictions.csv"
RECORDED_JSON = REPO_ROOT / "docs" / "temporal_baseline_results.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "step3_checkpoint_identity"

# Experiment 1 reported at the 0.5 default; rwf2000_config documents a second,
# self-declared-biased operating point at 0.14 selected on this same set.
REPORTED_THRESHOLD = 0.5
SELECTED_THRESHOLD = 0.14

# Tolerance for "agrees". The recorded values are stored to 4 decimal places,
# so a difference below half a unit in the last recorded place is a rounding
# artifact, not a disagreement.
ROUNDING_TOLERANCE = 5e-5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(path: Path):
    """Read the stored per-clip predictions. Never writes.

    Returns the labels as indices, the scores as floats, the class counts, the
    raw score strings (needed to characterise rounding ties) and the raw label
    strings.
    """
    truths: List[int] = []
    scores: List[float] = []
    raw_scores: List[str] = []
    raw_labels: List[str] = []
    counts: Dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["true_label"]
            counts[label] = counts.get(label, 0) + 1
            truths.append(LABEL_TO_INDEX[label])
            scores.append(float(row["fight_probability"]))
            raw_scores.append(row["fight_probability"])
            raw_labels.append(label)
    return truths, scores, dict(sorted(counts.items())), raw_scores, raw_labels


def compare(recomputed: Optional[float], recorded: Optional[float],
            bounds: Optional[Tuple[float, float]] = None) -> dict:
    """Exact difference between a recomputed and a recorded value.

    ``bounds`` is supplied for the rank-based metrics (ROC-AUC, PR-AUC). Those
    depend on the *ordering* of scores, and ``predictions.csv`` stores
    probabilities rounded to 6 decimals, which collapses distinct scores into
    ties and makes the exact value unrecoverable. When bounds are given,
    agreement is judged against the achievable interval rather than against a
    point, because a point comparison would report a discrepancy that the data
    cannot actually resolve.
    """
    if recomputed is None or recorded is None:
        return {
            "recomputed": recomputed,
            "recorded": recorded,
            "difference": None,
            "agrees": recomputed is None and recorded is None,
            "note": "one side not available",
        }
    difference = recomputed - recorded
    result = {
        "recomputed": recomputed,
        "recorded": recorded,
        "difference": difference,
        "abs_difference": abs(difference),
        "agrees_within_recorded_rounding": abs(difference) <= ROUNDING_TOLERANCE,
    }
    if bounds is not None:
        low, high = bounds
        result["tie_bounds"] = {"min": low, "max": high}
        result["recorded_within_tie_bounds"] = low <= recorded <= high
        result["agrees"] = result["recorded_within_tie_bounds"]
        result["basis"] = (
            "interval implied by 6-decimal rounding in predictions.csv; the "
            "tie-averaged point estimate is the midpoint convention"
        )
    else:
        result["agrees"] = result["agrees_within_recorded_rounding"]
        result["basis"] = "exact, threshold-dependent metric"
    return result


def tie_bounds(truths, scores, metric) -> Tuple[float, float]:
    """Return the achievable range of a rank metric given 6-decimal rounding.

    Every stored score could originally have been anywhere in +/- 5e-7 of its
    printed value. Nudging positives up and negatives down (and vice versa)
    within that interval gives the best and worst orderings consistent with the
    file, and therefore the range the true value must lie in.
    """
    import numpy as np

    truth = np.asarray(truths)
    score = np.asarray(scores, dtype=np.float64)
    epsilon = 5e-7
    best = score.copy()
    worst = score.copy()
    best[truth == 1] += epsilon
    best[truth == 0] -= epsilon
    worst[truth == 1] -= epsilon
    worst[truth == 0] += epsilon
    return float(metric(truth, worst)), float(metric(truth, best))


def score_tie_report(scores_text: List[str], labels: List[str]) -> dict:
    """Describe the ties that 6-decimal rounding introduced."""
    counts: Dict[str, int] = {}
    for value in scores_text:
        counts[value] = counts.get(value, 0) + 1
    tied = {value: n for value, n in counts.items() if n > 1}
    largest = sorted(tied.items(), key=lambda kv: -kv[1])[:5]
    return {
        "decimal_places_stored": sorted({len(v.split(".")[1]) for v in scores_text if "." in v}),
        "rows": len(scores_text),
        "distinct_values": len(counts),
        "tied_values": len(tied),
        "rows_involved_in_ties": sum(tied.values()),
        "largest_tie_groups": [
            {
                "value": value,
                "count": n,
                "labels": _label_tally(
                    [labels[i] for i, v in enumerate(scores_text) if v == value]
                ),
            }
            for value, n in largest
        ],
    }


def _label_tally(labels: List[str]) -> Dict[str, int]:
    tally: Dict[str, int] = {}
    for label in labels:
        tally[label] = tally.get(label, 0) + 1
    return dict(sorted(tally.items()))


def verify(predictions_csv: Path = PREDICTIONS_CSV,
           recorded_json: Path = RECORDED_JSON) -> dict:
    truths, scores, class_counts, raw_scores, raw_labels = load_predictions(predictions_csv)
    recorded = json.loads(Path(recorded_json).read_text(encoding="utf-8"))
    evaluation = recorded["evaluation"]

    at_reported = binary_metrics(truths, scores, threshold=REPORTED_THRESHOLD)
    at_selected = binary_metrics(truths, scores, threshold=SELECTED_THRESHOLD)

    recorded_confusion = evaluation["confusion_matrix"]
    computed_confusion = at_reported.confusion_matrix.as_dict()

    from temporal_metrics import pr_auc as pr_auc_fn, roc_auc as roc_auc_fn

    roc_bounds = tie_bounds(truths, scores, roc_auc_fn)
    prc_bounds = tie_bounds(truths, scores, pr_auc_fn)
    scalar_checks = {
        "accuracy": compare(at_reported.accuracy, evaluation["accuracy"]),
        "precision_fight": compare(at_reported.precision, evaluation["precision_fight"]),
        "recall_fight": compare(at_reported.recall, evaluation["recall_fight"]),
        "f1_fight": compare(at_reported.f1, evaluation["f1_fight"]),
        "roc_auc": compare(at_reported.roc_auc, evaluation["roc_auc"], roc_bounds),
        "pr_auc": compare(at_reported.pr_auc, evaluation["pr_auc"], prc_bounds),
    }
    confusion_checks = {
        key: {
            "recomputed": computed_confusion[key],
            "recorded": recorded_confusion[key],
            "difference": computed_confusion[key] - recorded_confusion[key],
            "agrees": computed_confusion[key] == recorded_confusion[key],
        }
        for key in ("true_negative", "false_positive", "false_negative", "true_positive")
    }
    support_checks = {
        label: {
            "recomputed": at_reported.support[label],
            "recorded": evaluation["per_class"][label]["support"],
            "agrees": at_reported.support[label] == evaluation["per_class"][label]["support"],
        }
        for label in ("NonFight", "Fight")
    }

    # rwf2000_config's documented second operating point, for completeness.
    from rwf2000_config import SELECTED_DECISION_THRESHOLD, SELECTED_THRESHOLD_PROVENANCE

    return {
        "tool": "verify_historical_metrics",
        "experiment": "experiment1_primary_monitor (notebook5d98537e15 v2)",
        "predictions_csv": str(predictions_csv),
        "predictions_sha256": file_sha256(predictions_csv),
        "predictions_bytes": predictions_csv.stat().st_size,
        "recorded_json": str(recorded_json),
        "sample_count": len(truths),
        "class_distribution": class_counts,
        "recorded_clip_count": evaluation["clips"],
        "sample_count_matches_record": len(truths) == evaluation["clips"],
        "threshold_reported": REPORTED_THRESHOLD,
        "rounding_tolerance": ROUNDING_TOLERANCE,
        "scalar_metrics": scalar_checks,
        "confusion_matrix": confusion_checks,
        "support": support_checks,
        "score_ties": score_tie_report(raw_scores, raw_labels),
        "all_scalars_agree": all(check.get("agrees") for check in scalar_checks.values()),
        "threshold_dependent_metrics_agree_exactly": all(
            check.get("agrees_within_recorded_rounding")
            for name, check in scalar_checks.items()
            if name not in ("roc_auc", "pr_auc")
        ),
        "confusion_exact_match": all(check["agrees"] for check in confusion_checks.values()),
        "support_exact_match": all(check["agrees"] for check in support_checks.values()),
        "secondary_operating_point": {
            "threshold": SELECTED_DECISION_THRESHOLD,
            "note": "documented in rwf2000_config as selected on this same 394-clip "
                    "set and therefore optimistically biased",
            "provenance": SELECTED_THRESHOLD_PROVENANCE,
            "recomputed": {
                "accuracy": at_selected.accuracy,
                "precision": at_selected.precision,
                "recall": at_selected.recall,
                "f1": at_selected.f1,
                "confusion_matrix": at_selected.confusion_matrix.as_dict(),
            },
        },
        "full_recomputed_at_reported_threshold": at_reported.as_dict(),
    }


def render(report: dict) -> str:
    lines = [
        "HISTORICAL METRIC VERIFICATION -- Experiment 1",
        "=" * 74,
        f"predictions : {report['predictions_csv']}",
        f"sha256      : {report['predictions_sha256']}",
        f"samples     : {report['sample_count']}  "
        f"(record says {report['recorded_clip_count']}: "
        f"{'MATCH' if report['sample_count_matches_record'] else 'MISMATCH'})",
        f"classes     : {report['class_distribution']}",
        f"threshold   : {report['threshold_reported']}",
        "",
        f"{'metric':18s} {'recomputed':>14s} {'recorded':>12s} {'difference':>14s}  agrees",
        "-" * 74,
    ]
    for name, check in report["scalar_metrics"].items():
        lines.append(
            f"{name:18s} {check['recomputed']:14.10f} {check['recorded']:12.4f} "
            f"{check['difference']:+14.10f}  {'yes' if check.get('agrees') else 'NO'}"
        )
        if "tie_bounds" in check:
            low, high = check["tie_bounds"]["min"], check["tie_bounds"]["max"]
            lines.append(
                f"{'':18s} rank metric: rounding ties allow [{low:.10f}, {high:.10f}]"
            )
    lines += ["", f"{'confusion':18s} {'recomputed':>14s} {'recorded':>12s} {'difference':>14s}  agrees", "-" * 74]
    for name, check in report["confusion_matrix"].items():
        lines.append(
            f"{name:18s} {check['recomputed']:14d} {check['recorded']:12d} "
            f"{check['difference']:+14d}  {'yes' if check['agrees'] else 'NO'}"
        )
    lines += ["", f"{'support':18s} {'recomputed':>14s} {'recorded':>12s}", "-" * 74]
    for name, check in report["support"].items():
        lines.append(f"{name:18s} {check['recomputed']:14d} {check['recorded']:12d}  "
                     f"{'yes' if check['agrees'] else 'NO'}")

    ties = report["score_ties"]
    lines += [
        "",
        "Score precision in predictions.csv:",
        f"  stored to {ties['decimal_places_stored']} decimal places; "
        f"{ties['distinct_values']} distinct values across {ties['rows']} rows",
        f"  {ties['tied_values']} value(s) are shared by {ties['rows_involved_in_ties']} rows:",
    ]
    for group in ties["largest_tie_groups"]:
        lines.append(f"    {group['value']} x{group['count']:<4d} {group['labels']}")

    secondary = report["secondary_operating_point"]
    lines += [
        "",
        f"Secondary operating point (threshold {secondary['threshold']}, "
        "documented as biased -- recomputed for completeness):",
        f"  accuracy {secondary['recomputed']['accuracy']:.4f}  "
        f"precision {secondary['recomputed']['precision']:.4f}  "
        f"recall {secondary['recomputed']['recall']:.4f}  "
        f"f1 {secondary['recomputed']['f1']:.4f}",
        f"  confusion {secondary['recomputed']['confusion_matrix']}",
        "",
        "=" * 74,
        f"threshold-dependent metrics match exactly     : "
        f"{report['threshold_dependent_metrics_agree_exactly']}",
        f"rank metrics consistent with rounding ties     : "
        f"{report['scalar_metrics']['roc_auc']['agrees'] and report['scalar_metrics']['pr_auc']['agrees']}",
        f"all scalar metrics agree                       : {report['all_scalars_agree']}",
        f"confusion matrix exact match                  : {report['confusion_exact_match']}",
        f"class supports exact match                    : {report['support_exact_match']}",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_CSV)
    parser.add_argument("--recorded", type=Path, default=RECORDED_JSON)
    parser.add_argument("--json", type=Path, default=None, help="Write the full report as JSON.")
    args = parser.parse_args(argv)

    report = verify(args.predictions, args.recorded)
    print(render(report))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    ok = (report["all_scalars_agree"] and report["confusion_exact_match"]
          and report["support_exact_match"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
