#!/usr/bin/env python3
"""
tools/metric_confidence_intervals.py

95% confidence intervals for the historical Experiment-1 evaluation, computed
from the committed per-clip predictions. Reads only; changes nothing.

WHAT THIS DOES AND DOES NOT CLAIM
---------------------------------
Three tiers, kept apart on purpose:

1. **Wilson score intervals** for metrics that are genuine binomial
   proportions -- accuracy, recall, specificity, class prevalence. Their
   denominators are fixed by the evaluation split.

2. **Wilson intervals, conditional**, for precision and NPV. Their denominators
   are counts the model produced, not design constants, so the interval is
   conditional on the observed number of predictions of that class. Reported,
   and labelled.

3. **Percentile bootstrap** for F1, ROC-AUC and PR-AUC. None of these is a
   count out of a denominator; a binomial interval would be wrong. The
   bootstrap is seeded and fully reproducible.

Nothing here gets a Wald interval, and nothing that is not a proportion gets a
Wilson interval.

THE INDEPENDENCE CAVEAT, MEASURED RATHER THAN ASSUMED AWAY
-----------------------------------------------------------
Every interval in tiers 1-3 assumes independent evaluation units. The 394
primary clips are not independent: they come from ~174 distinct source videos.
So this tool also reports a **cluster bootstrap** that resamples whole source
videos, and prints both widths side by side, so the reader can see how much the
clip-level intervals understate the uncertainty rather than take it on trust.

Usage:
    python tools/metric_confidence_intervals.py
    python tools/metric_confidence_intervals.py --json outputs/step4_confidence_intervals/intervals.json
    python tools/metric_confidence_intervals.py --resamples 2000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from metric_intervals import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    NO_INTERVAL_CLAIMED,
    bootstrap_metric_interval,
    cluster_ids,
    cluster_summary,
    confusion_intervals,
)
from rwf2000_config import LABEL_TO_INDEX  # noqa: E402
from temporal_metrics import binary_metrics, pr_auc, roc_auc  # noqa: E402

PREDICTIONS_CSV = REPO_ROOT / "outputs" / "temporal_violence" / "evaluation" / "predictions.csv"
PRIMARY_SCORES_CSV = REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"
REPORTED_THRESHOLD = 0.5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(path: Path):
    """Read labels and scores. predictions.csv is opened read-only."""
    truths, scores = [], []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            truths.append(LABEL_TO_INDEX[row["true_label"]])
            scores.append(float(row["fight_probability"]))
    return np.asarray(truths, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def primary_clip_order() -> Optional[List[str]]:
    """Return the primary split's clip names, in the order predictions.csv uses.

    ``predictions.csv`` carries only ``index``, not a clip name, so the clips
    cannot be matched to it directly. ``temporal_risk/primary_window_scores.csv``
    lists the same 394 clips, and both are sorted evaluation order over the same
    split, so the sorted clip list aligns positionally with the prediction rows.

    That alignment is an INFERENCE, and the caller must treat it as one. It is
    used for cluster structure only -- how many clips share a source video --
    which is a property of the set, not of the ordering. Even if the row order
    were permuted, the cluster sizes would be identical. It is never used to
    attach a score to a named clip.
    """
    if not PRIMARY_SCORES_CSV.is_file():
        return None
    clips = set()
    with open(PRIMARY_SCORES_CSV, newline="", encoding="utf-8",
              errors="surrogateescape") as handle:
        for row in csv.DictReader(handle):
            clips.add(row["clip"])
    return sorted(clips)


def f1_statistic(truth: np.ndarray, score: np.ndarray) -> Optional[float]:
    """F1 at the reported threshold, or None when undefined."""
    predicted = (score >= REPORTED_THRESHOLD).astype(np.int64)
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    if tp + fp == 0 or tp + fn == 0:
        return None
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def accuracy_statistic(truth: np.ndarray, score: np.ndarray) -> Optional[float]:
    predicted = (score >= REPORTED_THRESHOLD).astype(np.int64)
    return float(np.mean(predicted == truth))


def compute(
    predictions_csv: Path = PREDICTIONS_CSV,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    truth, score = load_predictions(predictions_csv)
    metrics = binary_metrics(truth, score, threshold=REPORTED_THRESHOLD)
    confusion = metrics.confusion_matrix

    wilson = confusion_intervals(
        true_positive=confusion.true_positive,
        false_positive=confusion.false_positive,
        true_negative=confusion.true_negative,
        false_negative=confusion.false_negative,
        confidence=confidence,
    )

    boot_kwargs = dict(confidence=confidence, resamples=resamples, seed=seed)
    stratified = {
        "f1": bootstrap_metric_interval("f1", truth, score, f1_statistic, **boot_kwargs),
        "roc_auc": bootstrap_metric_interval("roc_auc", truth, score, roc_auc, **boot_kwargs),
        "pr_auc": bootstrap_metric_interval("pr_auc", truth, score, pr_auc, **boot_kwargs),
        "accuracy": bootstrap_metric_interval(
            "accuracy", truth, score, accuracy_statistic, **boot_kwargs
        ),
    }

    clips = primary_clip_order()
    clustered: Dict[str, dict] = {}
    clustering: Optional[dict] = None
    if clips is not None and len(clips) == truth.size:
        clustering = cluster_summary(clips)
        groups = cluster_ids(clips)
        for name, statistic in (
            ("accuracy", accuracy_statistic),
            ("f1", f1_statistic),
            ("roc_auc", roc_auc),
            ("pr_auc", pr_auc),
        ):
            clustered[name] = bootstrap_metric_interval(
                name, truth, score, statistic, clusters=groups, **boot_kwargs
            ).as_dict()
    else:
        clustering = {
            "available": False,
            "reason": (
                f"clip list unavailable or length mismatch "
                f"({0 if clips is None else len(clips)} clips vs {truth.size} predictions); "
                "cluster bootstrap not computed"
            ),
        }

    return {
        "tool": "metric_confidence_intervals",
        "experiment": "experiment1_primary_monitor (notebook5d98537e15 v2)",
        "predictions_csv": str(predictions_csv),
        "predictions_sha256": file_sha256(predictions_csv),
        "sample_count": int(truth.size),
        "class_distribution": {
            "Fight": int((truth == 1).sum()),
            "NonFight": int((truth == 0).sum()),
        },
        "threshold": REPORTED_THRESHOLD,
        "confidence": confidence,
        "point_estimates": {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
            "pr_auc": metrics.pr_auc,
        },
        "confusion_matrix": confusion.as_dict(),
        "wilson_intervals": {name: interval.as_dict() for name, interval in wilson.items()},
        "bootstrap_intervals_stratified": {
            name: interval.as_dict() for name, interval in stratified.items()
        },
        "bootstrap_intervals_clustered": clustered,
        "cluster_structure": clustering,
        "no_interval_claimed_by_wilson": NO_INTERVAL_CLAIMED,
        "assumptions": [
            "Wilson intervals assume independent evaluation units. The 394 clips "
            "come from ~174 source videos, so this assumption is violated and the "
            "clip-level intervals are anti-conservative (too narrow).",
            "Precision and NPV intervals are conditional on the observed number of "
            "predictions of that class, which is not fixed by the design.",
            "Bootstrap intervals are percentile-method, not BCa.",
            "All intervals describe sampling variability of THIS evaluation set "
            "only. They say nothing about generalisation to other footage, and "
            "nothing about the model-selection bias that Experiment 1's protocol "
            "carries (see docs/EXPERIMENT_REPRODUCIBILITY.md).",
        ],
    }


def render(report: dict) -> str:
    lines = [
        "95% CONFIDENCE INTERVALS -- Experiment 1, primary split",
        "=" * 78,
        f"predictions : {report['predictions_csv']}",
        f"sha256      : {report['predictions_sha256'][:32]}...",
        f"samples     : {report['sample_count']}  {report['class_distribution']}",
        f"threshold   : {report['threshold']}   confidence: {report['confidence']}",
        "",
        "TIER 1 -- Wilson score intervals, denominator fixed by the split",
        "-" * 78,
        f"{'metric':22s} {'point':>9s}  {'95% CI':>20s}  {'counts':>10s}  width(pp)",
    ]
    tier1 = ("accuracy", "recall_sensitivity", "specificity", "prevalence_fight")
    for name in tier1:
        if name not in report["wilson_intervals"]:
            continue
        row = report["wilson_intervals"][name]
        width = 100 * (row["ci_high"] - row["ci_low"])
        lines.append(
            f"{name:22s} {row['point']:9.4f}  [{row['ci_low']:.4f}, {row['ci_high']:.4f}]  "
            f"{row['successes']:>4}/{row['total']:<5}  {width:8.2f}"
        )

    lines += [
        "",
        "TIER 2 -- Wilson intervals, CONDITIONAL on a model-chosen denominator",
        "-" * 78,
        f"{'metric':22s} {'point':>9s}  {'95% CI':>20s}  {'counts':>10s}  width(pp)",
    ]
    for name in ("precision_ppv", "npv"):
        if name not in report["wilson_intervals"]:
            continue
        row = report["wilson_intervals"][name]
        width = 100 * (row["ci_high"] - row["ci_low"])
        lines.append(
            f"{name:22s} {row['point']:9.4f}  [{row['ci_low']:.4f}, {row['ci_high']:.4f}]  "
            f"{row['successes']:>4}/{row['total']:<5}  {width:8.2f}"
        )

    lines += [
        "",
        "TIER 3 -- percentile bootstrap; NOT binomial proportions",
        "-" * 78,
        f"{'metric':22s} {'point':>9s}  {'95% CI':>20s}  {'resamples':>10s}",
    ]
    for name in ("f1", "roc_auc", "pr_auc"):
        row = report["bootstrap_intervals_stratified"][name]
        if row["ci_low"] is None:
            lines.append(f"{name:22s} not computable")
            continue
        lines.append(
            f"{name:22s} {row['point']:9.4f}  [{row['ci_low']:.4f}, {row['ci_high']:.4f}]  "
            f"{row['usable_resamples']:>10}"
        )

    cluster = report.get("cluster_structure") or {}
    if cluster.get("clusters"):
        lines += [
            "",
            "INDEPENDENCE CHECK -- the clips are NOT independent",
            "-" * 78,
            f"  {cluster['clips']} clips come from {cluster['clusters']} source videos "
            f"(mean {cluster['mean_cluster_size']:.2f}, max {cluster['max_cluster_size']})",
            f"  {cluster['clips_in_multi_clip_clusters']} clips share a source with "
            f"another clip; {cluster['singleton_clusters']} are singletons",
            f"  grouping {cluster['grouping']}",
            "",
            "  Effect on interval width -- clip-level vs source-video-level bootstrap:",
            f"  {'metric':16s} {'stratified (clips)':>26s} {'clustered (videos)':>26s}  ratio",
        ]
        for name in ("accuracy", "f1", "roc_auc", "pr_auc"):
            strat = report["bootstrap_intervals_stratified"].get(name)
            clus = report["bootstrap_intervals_clustered"].get(name)
            if not strat or not clus or strat["ci_low"] is None or clus["ci_low"] is None:
                continue
            ws = strat["ci_high"] - strat["ci_low"]
            wc = clus["ci_high"] - clus["ci_low"]
            lines.append(
                f"  {name:16s} [{strat['ci_low']:.4f}, {strat['ci_high']:.4f}] "
                f"{100*ws:6.2f}pp  [{clus['ci_low']:.4f}, {clus['ci_high']:.4f}] "
                f"{100*wc:6.2f}pp  {wc/ws:5.2f}x"
            )

    lines += [
        "",
        "NO WILSON INTERVAL CLAIMED FOR",
        "-" * 78,
    ]
    for name, reason in report["no_interval_claimed_by_wilson"].items():
        lines.append(f"  {name}: {reason}")
    lines += ["", "ASSUMPTIONS", "-" * 78]
    for assumption in report["assumptions"]:
        lines.append(f"  - {assumption}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_CSV)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = compute(args.predictions, args.confidence, args.resamples, args.seed)
    print(render(report))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
