#!/usr/bin/env python3
"""
tools/crosscheck_metrics_sklearn.py

Independent cross-check of ``src/temporal_metrics.py`` against scikit-learn,
on the committed Experiment-1 predictions.

WHY THIS IS NEEDED
------------------
``temporal_metrics`` reimplements ROC-AUC (rank / Mann-Whitney identity with
tie-averaged ranks) and PR-AUC (step-wise average precision) in numpy, to avoid
a scikit-learn dependency. Nothing has ever compared it against an independent
implementation. A hand-rolled rank statistic with tie handling is exactly the
kind of code that can be subtly wrong and still look plausible.

TWO DIFFERENT QUESTIONS -- DO NOT CONFLATE THEM
-----------------------------------------------
This is the distinction that makes the check meaningful:

1. **Is the code correct?**  scikit-learn vs ``temporal_metrics`` on the *same*
   6-decimal data. Both see identical inputs, so they must agree to floating
   -point tolerance. Any disagreement is a bug in one of them.

2. **Can the original recorded value be recovered?**  Either implementation vs
   the historically recorded 0.9432 / 0.9475. They differ by ~1.7e-4 because
   ``predictions.csv`` stores probabilities to 6 decimals, which ties 105 of
   394 rows and destroys the ordering information a rank statistic needs. That
   is a property of the *stored data*, not of either implementation, and no
   amount of cross-checking can fix it.

Step 3 established (2). This tool establishes (1). Both findings stand; neither
overwrites the other.

SCIKIT-LEARN IS OPTIONAL
------------------------
It is not in ``requirements.txt`` and is not installed in the project venv.
Without it this tool exits 0 and reports SKIPPED -- never a failure, never a
silently fabricated comparison. To run the check, install it somewhere that
does not disturb the project environment, for example:

    pip install --target .sklearn-crosscheck scikit-learn
    PYTHONPATH=.sklearn-crosscheck python tools/crosscheck_metrics_sklearn.py

Usage:
    python tools/crosscheck_metrics_sklearn.py
    python tools/crosscheck_metrics_sklearn.py --json outputs/step4_confidence_intervals/sklearn_crosscheck.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from rwf2000_config import LABEL_TO_INDEX  # noqa: E402
from temporal_metrics import binary_metrics, pr_auc, roc_auc  # noqa: E402

PREDICTIONS_CSV = REPO_ROOT / "outputs" / "temporal_violence" / "evaluation" / "predictions.csv"
RECORDED_JSON = REPO_ROOT / "docs" / "temporal_baseline_results.json"
THRESHOLD = 0.5

# Two implementations of the same formula on the same doubles should agree to
# near machine epsilon. Anything larger is a real difference in definition.
AGREEMENT_TOLERANCE = 1e-12


def load_predictions(path: Path):
    truths, scores = [], []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            truths.append(LABEL_TO_INDEX[row["true_label"]])
            scores.append(float(row["fight_probability"]))
    return np.asarray(truths, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def sklearn_available() -> Optional[str]:
    """Return the installed scikit-learn version, or None."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return None
    return sklearn.__version__


def compare(name: str, ours: Optional[float], theirs: Optional[float]) -> dict:
    if ours is None or theirs is None:
        return {"metric": name, "temporal_metrics": ours, "sklearn": theirs,
                "difference": None, "agrees": ours is None and theirs is None}
    difference = float(ours) - float(theirs)
    return {
        "metric": name,
        "temporal_metrics": float(ours),
        "sklearn": float(theirs),
        "difference": difference,
        "abs_difference": abs(difference),
        "agrees": abs(difference) <= AGREEMENT_TOLERANCE,
    }


def compute(predictions_csv: Path = PREDICTIONS_CSV) -> dict:
    version = sklearn_available()
    truth, score = load_predictions(predictions_csv)
    ours = binary_metrics(truth, score, threshold=THRESHOLD)

    base = {
        "tool": "crosscheck_metrics_sklearn",
        "predictions_csv": str(predictions_csv),
        "sample_count": int(truth.size),
        "threshold": THRESHOLD,
        "tolerance": AGREEMENT_TOLERANCE,
        "sklearn_available": version is not None,
        "sklearn_version": version,
        "temporal_metrics_values": {
            "accuracy": ours.accuracy,
            "precision": ours.precision,
            "recall": ours.recall,
            "f1": ours.f1,
            "roc_auc": ours.roc_auc,
            "pr_auc": ours.pr_auc,
            "confusion_matrix": ours.confusion_matrix.as_dict(),
        },
    }

    if version is None:
        base["status"] = "SKIPPED"
        base["reason"] = (
            "scikit-learn is not installed. It is deliberately not a project "
            "dependency; this cross-check is optional. No comparison was "
            "performed and none is claimed."
        )
        return base

    from sklearn.metrics import (  # noqa: E402
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predicted = (score >= THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()

    comparisons = [
        compare("accuracy", ours.accuracy, accuracy_score(truth, predicted)),
        compare("precision", ours.precision, precision_score(truth, predicted, zero_division=0)),
        compare("recall", ours.recall, recall_score(truth, predicted, zero_division=0)),
        compare("f1", ours.f1, f1_score(truth, predicted, zero_division=0)),
        compare("roc_auc", ours.roc_auc, roc_auc_score(truth, score)),
        compare("pr_auc", ours.pr_auc, average_precision_score(truth, score)),
    ]
    confusion_comparison = {
        key: {
            "temporal_metrics": getattr(ours.confusion_matrix, key),
            "sklearn": int(value),
            "agrees": getattr(ours.confusion_matrix, key) == int(value),
        }
        for key, value in (
            ("true_negative", tn), ("false_positive", fp),
            ("false_negative", fn), ("true_positive", tp),
        )
    }

    # Question 2: neither implementation can recover the original value.
    recorded = json.loads(RECORDED_JSON.read_text(encoding="utf-8"))["evaluation"]
    rounding = {
        "note": (
            "Both implementations read the same 6-decimal file, so their "
            "agreement tests the CODE. Their common gap to the recorded value "
            "tests the DATA and cannot be closed by cross-checking."
        ),
        "roc_auc": {
            "recorded": recorded["roc_auc"],
            "sklearn": comparisons[4]["sklearn"],
            "difference_from_recorded": comparisons[4]["sklearn"] - recorded["roc_auc"],
        },
        "pr_auc": {
            "recorded": recorded["pr_auc"],
            "sklearn": comparisons[5]["sklearn"],
            "difference_from_recorded": comparisons[5]["sklearn"] - recorded["pr_auc"],
        },
    }

    base.update(
        status="COMPARED",
        comparisons=comparisons,
        confusion_matrix_comparison=confusion_comparison,
        all_metrics_agree=all(c["agrees"] for c in comparisons),
        confusion_matrix_agrees=all(c["agrees"] for c in confusion_comparison.values()),
        rounding_limitation_still_stands=rounding,
    )
    return base


def render(report: dict) -> str:
    lines = [
        "INDEPENDENT METRIC CROSS-CHECK -- temporal_metrics vs scikit-learn",
        "=" * 78,
        f"predictions : {report['predictions_csv']}",
        f"samples     : {report['sample_count']}   threshold: {report['threshold']}",
        f"sklearn     : {report['sklearn_version'] or 'NOT INSTALLED'}",
        f"status      : {report['status']}",
        "",
    ]
    if report["status"] == "SKIPPED":
        lines += [
            report["reason"],
            "",
            "temporal_metrics values (reported without an independent check):",
        ]
        for name, value in report["temporal_metrics_values"].items():
            if name != "confusion_matrix":
                lines.append(f"  {name:12s} {value}")
        return "\n".join(lines)

    lines += [
        f"{'metric':14s} {'temporal_metrics':>20s} {'sklearn':>20s} {'difference':>16s}  agrees",
        "-" * 78,
    ]
    for comparison in report["comparisons"]:
        lines.append(
            f"{comparison['metric']:14s} {comparison['temporal_metrics']:20.16f} "
            f"{comparison['sklearn']:20.16f} {comparison['difference']:+16.2e}  "
            f"{'yes' if comparison['agrees'] else 'NO'}"
        )
    lines += ["", "confusion matrix", "-" * 78]
    for name, comparison in report["confusion_matrix_comparison"].items():
        lines.append(
            f"  {name:16s} ours={comparison['temporal_metrics']:<5} "
            f"sklearn={comparison['sklearn']:<5} "
            f"{'agrees' if comparison['agrees'] else 'DIFFERS'}"
        )
    rounding = report["rounding_limitation_still_stands"]
    lines += [
        "",
        "THE 6-DECIMAL ROUNDING LIMITATION IS UNCHANGED BY THIS CHECK",
        "-" * 78,
        f"  {rounding['note']}",
        "",
        f"  roc_auc: sklearn {rounding['roc_auc']['sklearn']:.10f} vs recorded "
        f"{rounding['roc_auc']['recorded']:.4f} "
        f"(difference {rounding['roc_auc']['difference_from_recorded']:+.10f})",
        f"  pr_auc : sklearn {rounding['pr_auc']['sklearn']:.10f} vs recorded "
        f"{rounding['pr_auc']['recorded']:.4f} "
        f"(difference {rounding['pr_auc']['difference_from_recorded']:+.10f})",
        "",
        "=" * 78,
        f"all metrics agree within {report['tolerance']:g} : {report['all_metrics_agree']}",
        f"confusion matrix agrees exactly       : {report['confusion_matrix_agrees']}",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_CSV)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--require-sklearn", action="store_true",
                        help="Exit non-zero if scikit-learn is unavailable.")
    args = parser.parse_args(argv)

    report = compute(args.predictions)
    print(render(report))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    if report["status"] == "SKIPPED":
        return 1 if args.require_sklearn else 0
    return 0 if (report["all_metrics_agree"] and report["confusion_matrix_agrees"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
