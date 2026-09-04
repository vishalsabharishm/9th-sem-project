#!/usr/bin/env python3
"""
tools/sliding_window_uncertainty.py

Uncertainty quantification for the result the deployed system actually
produces: the sliding-window temporal regime, accuracy 0.7893 on the 394-clip
primary split.

WHY THIS IS A DIFFERENT JOB FROM STEP 4
---------------------------------------
Step 4 put intervals on Experiment 1's whole-clip result (0.8604), which is the
superseded baseline from a superseded checkpoint. Nothing in the running system
produces that number. The system produces 0.7893, by scoring 17 sliding windows
per clip with the clean-baseline checkpoint and applying the frozen
``max >= 0.14`` aggregation rule.

Three properties make this regime *better* evidenced than Experiment 1, not
worse:

- It is **exactly recomputable** from committed artifacts with no checkpoint:
  ``temporal_risk/primary_window_scores.csv`` plus
  ``temporal_risk/frozen_aggregation.json`` reproduce TP=160 FP=43 TN=151 FN=40
  and accuracy 0.7893401015 exactly. This tool asserts that before doing
  anything else and refuses to continue if it fails.
- Its per-clip scores carry **clip names**, so source-video grouping is applied
  directly rather than through the positional-alignment inference Step 4 had to
  make for ``predictions.csv``.
- Its per-clip continuous score is well defined: ``max`` over the 17 windows is
  precisely the quantity the frozen rule thresholds, so a rank metric over it
  describes this regime rather than some other one.

WHAT THIS TOOL WILL NOT DO
--------------------------
It will not compare 0.7893 against the clean baseline's 0.8325 with any paired
test. Those two numbers come from the same checkpoint under two inference
regimes -- which is exactly the comparison worth making -- but the clean
baseline's per-clip predictions were never recovered from Kaggle, so no clip
can be paired with itself. Descriptive intervals for both are reported; a
significance claim is refused and the reason is recorded in the output.

Usage:
    python tools/sliding_window_uncertainty.py
    python tools/sliding_window_uncertainty.py --json outputs/step5_sliding_window/uncertainty.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import numpy as np  # noqa: E402

from metric_intervals import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    bootstrap_metric_interval,
    cluster_ids,
    cluster_summary,
    confusion_intervals,
)
from temporal_event_adapter import FrozenAggregationRule, PrecomputedWindowScoreSource  # noqa: E402
from temporal_metrics import pr_auc, roc_auc  # noqa: E402

PRIMARY_CSV = REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"
FROZEN_JSON = REPO_ROOT / "temporal_risk" / "frozen_aggregation.json"
AGGREGATION_RESULT = REPO_ROOT / "temporal_risk" / "primary_aggregation_result.json"
CLEAN_BASELINE = REPO_ROOT / "temporal_risk" / "clean_baseline_final_evaluation_metrics.json"
SOURCE_FPS = 30.0


class UncertaintyError(RuntimeError):
    """Raised when the committed artifacts do not reproduce the recorded result."""


# --------------------------------------------------------------------------
# Sliding-window regime
# --------------------------------------------------------------------------


def load_sliding_window() -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return (clips, truth, clip_score, fired) for the sliding-window regime.

    ``clip_score`` is ``max`` over the 17 window probabilities -- the exact
    quantity ``FrozenAggregationRule`` thresholds when the rule is ``max``.
    ``fired`` is the rule's own decision, taken from the rule rather than
    re-derived, so a future re-selection to ``k_of_n`` or ``mean`` cannot
    silently desynchronise the two.
    """
    rule = FrozenAggregationRule.load(FROZEN_JSON)
    source = PrecomputedWindowScoreSource(PRIMARY_CSV)
    clips = source.clips()

    truth, score, fired = [], [], []
    for clip in clips:
        windows = source.get(clip) or []
        probabilities = [w.fight_probability for w in windows]
        truth.append(1 if source.true_label(clip) == "Fight" else 0)
        score.append(max(probabilities))
        fired.append(bool(rule.decide(probabilities)))
    return (
        clips,
        np.asarray(truth, dtype=np.int64),
        np.asarray(score, dtype=np.float64),
        np.asarray(fired, dtype=bool),
    )


def confusion_from(truth: np.ndarray, fired: np.ndarray) -> Dict[str, int]:
    return {
        "true_positive": int(np.sum((truth == 1) & fired)),
        "false_positive": int(np.sum((truth == 0) & fired)),
        "true_negative": int(np.sum((truth == 0) & ~fired)),
        "false_negative": int(np.sum((truth == 1) & ~fired)),
    }


def assert_reproduces_recorded(confusion: Dict[str, int]) -> dict:
    """Refuse to report intervals around a result we cannot reproduce."""
    recorded = json.loads(AGGREGATION_RESULT.read_text(encoding="utf-8"))["primary_split"]["metrics"]
    expected = {
        "true_positive": recorded["tp"], "false_positive": recorded["fp"],
        "true_negative": recorded["tn"], "false_negative": recorded["fn"],
    }
    if confusion != expected:
        raise UncertaintyError(
            f"Recomputed confusion {confusion} does not match the recorded "
            f"{expected}. Intervals around an unreproducible point estimate "
            "would be meaningless."
        )
    total = sum(confusion.values())
    accuracy = (confusion["true_positive"] + confusion["true_negative"]) / total
    return {
        "recomputed": confusion,
        "recorded": expected,
        "exact_match": True,
        "recomputed_accuracy": accuracy,
        "recorded_accuracy": recorded["accuracy"],
        "accuracy_difference": accuracy - recorded["accuracy"],
        "source": "temporal_risk/primary_window_scores.csv + frozen_aggregation.json",
        "checkpoint_required": False,
    }


def f1_at_rule(truth: np.ndarray, fired: np.ndarray) -> Optional[float]:
    tp = int(np.sum((truth == 1) & fired))
    fp = int(np.sum((truth == 0) & fired))
    fn = int(np.sum((truth == 1) & ~fired))
    if tp + fp == 0 or tp + fn == 0:
        return None
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    if precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------
# Time to alarm, censoring-aware
# --------------------------------------------------------------------------


def alarm_quantiles(detected_times: List[float], n_fight: int) -> dict:
    """Quantiles of time-to-alarm over ALL Fight clips, not only detected ones.

    Censoring here is *administrative and identical*: every clip that never
    alarms is censored at the same time, 4.8 s, the end of the last complete
    window. Nothing drops out early. That makes the Kaplan-Meier survival
    function a step function with no estimation needed below the censoring
    time, so any quantile at or under the detected fraction (160/200 = 0.80) is
    **exact**, not estimated. Above it the quantile is genuinely not reached and
    is reported as such rather than extrapolated.

    The conditional quantiles -- over the detected clips only -- are reported
    alongside because Step 2 quoted them, and the two must not be conflated:
    the conditional median answers "when the system catches a fight, how fast?",
    the unconditional one answers "how long until half of all fights are
    caught?".
    """
    ordered = sorted(detected_times)
    n_detected = len(ordered)
    detected_fraction = n_detected / n_fight if n_fight else 0.0

    def unconditional(q: float) -> Optional[float]:
        need = q * n_fight
        if need > n_detected:
            return None
        index = int(np.ceil(need)) - 1
        return ordered[max(index, 0)]

    conditional = {}
    if ordered:
        array = np.asarray(ordered)
        conditional = {
            "median_seconds": float(np.median(array)),
            "p75_seconds": float(np.quantile(array, 0.75, method="linear")),
            "min_seconds": float(array.min()),
            "max_seconds": float(array.max()),
            "mean_seconds": float(array.mean()),
        }
    return {
        "n_fight_clips": n_fight,
        "n_detected": n_detected,
        "n_right_censored": n_fight - n_detected,
        "detected_fraction": detected_fraction,
        "censoring": (
            "administrative and identical: every non-detection is censored at "
            "4.8 s, the end of the last complete window. No clip leaves the "
            "risk set early."
        ),
        "unconditional_over_all_fight_clips": {
            "note": (
                "exact below the detected fraction because all censoring occurs "
                "after it; not estimated"
            ),
            "q25_seconds": unconditional(0.25),
            "median_seconds": unconditional(0.50),
            "q75_seconds": unconditional(0.75),
            "q80_seconds": unconditional(0.80),
            "highest_reachable_quantile": detected_fraction,
            "not_reached_above": (
                "quantiles above the detected fraction are never reached; the "
                "system does not alarm on those clips at all"
            ),
        },
        "conditional_on_detection": conditional,
        "do_not_conflate": (
            "the conditional median describes latency among clips the system "
            "catches; the unconditional median describes all Fight clips. They "
            "are different quantities and the conditional one is smaller."
        ),
    }


def bootstrap_alarm_median(
    detected_times: List[float],
    n_fight: int,
    censored: int,
    confidence: float,
    resamples: int,
    seed: int,
    clusters: Optional[np.ndarray] = None,
) -> dict:
    """Percentile bootstrap for the UNCONDITIONAL median time to alarm.

    Each Fight clip is resampled as an outcome pair (alarmed?, time). A
    resample in which fewer than half the clips alarm has no median below the
    censoring time; it is discarded and counted, never imputed as 4.8 s.
    """
    times = np.asarray(
        detected_times + [np.inf] * censored, dtype=np.float64
    )  # inf marks "never alarmed"; only ordering below the median matters
    generator = np.random.default_rng(seed)
    if clusters is not None:
        unique = np.unique(clusters)
        members = [np.flatnonzero(clusters == c) for c in unique]

    values, discarded = [], 0
    for _ in range(resamples):
        if clusters is None:
            picked = generator.integers(0, times.size, size=times.size)
        else:
            drawn = generator.integers(0, len(members), size=len(members))
            picked = np.concatenate([members[i] for i in drawn])
        sample = np.sort(times[picked])
        need = int(np.ceil(0.5 * sample.size)) - 1
        if need < 0 or not np.isfinite(sample[need]):
            discarded += 1
            continue
        values.append(float(sample[need]))

    if not values:
        return {"median": None, "ci_low": None, "ci_high": None,
                "usable_resamples": 0, "discarded_resamples": discarded,
                "note": "every resample failed to reach a median before censoring"}
    tail = (1.0 - confidence) / 2.0
    array = np.asarray(values)
    ordered = np.sort(np.asarray(detected_times + [np.inf] * censored))
    point_index = int(np.ceil(0.5 * ordered.size)) - 1
    point = float(ordered[point_index]) if np.isfinite(ordered[point_index]) else None
    return {
        "median": point,
        "ci_low": float(np.quantile(array, tail)),
        "ci_high": float(np.quantile(array, 1 - tail)),
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "usable_resamples": len(values),
        "discarded_resamples": discarded,
        "method": ("percentile bootstrap over source-video clusters"
                   if clusters is not None else
                   "percentile bootstrap over Fight clips"),
        "granularity_note": (
            "alarm times are discrete -- one of 17 possible window end times -- "
            "so the bootstrap distribution is lumpy and the interval endpoints "
            "fall on those discrete values"
        ),
    }


# --------------------------------------------------------------------------
# Clean baseline
# --------------------------------------------------------------------------


def clean_baseline_intervals(confidence: float) -> dict:
    """Wilson intervals for the clean baseline, from its confusion matrix only.

    The counts ARE the sufficient statistics for these proportions, so a Wilson
    interval from them is a computation, not an invention. What cannot be done
    without per-clip predictions: any bootstrap (F1, ROC-AUC, PR-AUC), any
    source-video clustering correction, and any paired comparison. Those are
    reported as unavailable rather than approximated.
    """
    recorded = json.loads(CLEAN_BASELINE.read_text(encoding="utf-8"))
    matrix = recorded["metrics"]["confusion_matrix"]
    intervals = confusion_intervals(
        true_positive=matrix["true_positive"],
        false_positive=matrix["false_positive"],
        true_negative=matrix["true_negative"],
        false_negative=matrix["false_negative"],
        confidence=confidence,
    )
    return {
        "source": str(CLEAN_BASELINE),
        "provenance": recorded["source"],
        "regime": "whole-clip single-pass inference, threshold 0.16",
        "point_estimates": {
            key: recorded["metrics"][key]
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
        },
        "confusion_matrix": matrix,
        "wilson_intervals": {name: value.as_dict() for name, value in intervals.items()},
        "not_available": {
            "per_clip_predictions": (
                "never recovered from Kaggle notebook7bb9a86555 v3; only the "
                "aggregate metrics and confusion matrix were transcribed"
            ),
            "bootstrap_intervals": "require per-clip predictions",
            "f1_interval": "F1 is not a binomial proportion; needs the bootstrap",
            "roc_auc_and_pr_auc_intervals": "rank statistics; need per-clip scores",
            "source_video_clustering": (
                "requires knowing which clip each prediction belongs to; the "
                "clip-level Wilson intervals here are therefore uncorrected and "
                "anti-conservative, and unlike the sliding-window result the "
                "size of that effect cannot be measured"
            ),
        },
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def compute(
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    clips, truth, score, fired = load_sliding_window()
    confusion = confusion_from(truth, fired)
    reproduction = assert_reproduces_recorded(confusion)

    wilson = confusion_intervals(
        true_positive=confusion["true_positive"],
        false_positive=confusion["false_positive"],
        true_negative=confusion["true_negative"],
        false_negative=confusion["false_negative"],
        confidence=confidence,
    )

    groups = cluster_ids(clips)
    clustering = cluster_summary(clips)
    boot = dict(confidence=confidence, resamples=resamples, seed=seed)

    def accuracy_stat(t, s):
        return float(np.mean((s >= 0.14) == (t == 1)))

    def f1_stat(t, s):
        return f1_at_rule(t, s >= 0.14)

    stratified = {
        "accuracy": bootstrap_metric_interval("accuracy", truth, score, accuracy_stat, **boot),
        "f1": bootstrap_metric_interval("f1", truth, score, f1_stat, **boot),
        "roc_auc": bootstrap_metric_interval("roc_auc", truth, score, roc_auc, **boot),
        "pr_auc": bootstrap_metric_interval("pr_auc", truth, score, pr_auc, **boot),
    }
    clustered = {
        name: bootstrap_metric_interval(
            name, truth, score, statistic, clusters=groups, **boot
        )
        for name, statistic in (
            ("accuracy", accuracy_stat), ("f1", f1_stat),
            ("roc_auc", roc_auc), ("pr_auc", pr_auc),
        )
    }

    # Time to alarm, from the same committed scores.
    from time_to_alarm import compute as tta_compute

    tta = tta_compute("primary")
    fight_rows = [row for row in tta["per_clip"] if row["true_label"] == "Fight"]
    detected_times = [row["time_to_alarm_seconds"] for row in fight_rows if row["alarmed"]]
    censored = sum(1 for row in fight_rows if not row["alarmed"])
    fight_clips = [row["clip"] for row in fight_rows]
    fight_groups = cluster_ids(fight_clips)

    latency = alarm_quantiles(detected_times, len(fight_rows))
    latency["bootstrap_unconditional_median"] = {
        "clip_level": bootstrap_alarm_median(
            detected_times, len(fight_rows), censored, confidence, resamples, seed
        ),
        "source_video_level": bootstrap_alarm_median(
            detected_times, len(fight_rows), censored, confidence, resamples, seed,
            clusters=fight_groups,
        ),
    }

    return {
        "tool": "sliding_window_uncertainty",
        "regime": "sliding-window, 16 frames stride 8, 17 windows/clip, frozen max >= 0.14",
        "checkpoint": "clean-baseline best.pt (notebook7bb9a86555 v3) -- scores committed, weights absent",
        "confidence": confidence,
        "sample_count": int(truth.size),
        "class_distribution": {"Fight": int((truth == 1).sum()),
                               "NonFight": int((truth == 0).sum())},
        "reproduction_check": reproduction,
        "point_estimates": {
            "accuracy": (confusion["true_positive"] + confusion["true_negative"]) / truth.size,
            "precision": confusion["true_positive"] / (confusion["true_positive"] + confusion["false_positive"]),
            "recall": confusion["true_positive"] / (confusion["true_positive"] + confusion["false_negative"]),
            "specificity": confusion["true_negative"] / (confusion["true_negative"] + confusion["false_positive"]),
            "f1": f1_at_rule(truth, fired),
            "roc_auc": roc_auc(truth, score),
            "pr_auc": pr_auc(truth, score),
        },
        "confusion_matrix": confusion,
        "wilson_intervals": {name: value.as_dict() for name, value in wilson.items()},
        "bootstrap_intervals_stratified": {n: v.as_dict() for n, v in stratified.items()},
        "bootstrap_intervals_clustered": {n: v.as_dict() for n, v in clustered.items()},
        "cluster_structure": clustering,
        "rank_metric_note": (
            "ROC-AUC and PR-AUC here are computed over max-of-17-windows, the "
            "quantity the frozen rule thresholds. They describe THIS regime's "
            "ranking and are NOT comparable with the clean baseline's whole-clip "
            "ROC-AUC 0.9251, which ranks a different score."
        ),
        "clean_baseline": clean_baseline_intervals(confidence),
        "time_to_alarm": latency,
        "paired_comparison": paired_comparison_status(),
    }


def paired_comparison_status() -> dict:
    """Why 0.7893 vs 0.8325 cannot be tested, stated rather than glossed."""
    return {
        "requested": "sliding-window 0.7893 vs clean-baseline whole-clip 0.8325",
        "same_checkpoint": True,
        "same_clips": True,
        "difference_is": "inference regime only -- exactly the comparison worth making",
        "formal_test_possible": False,
        "blocking_reason": (
            "a paired test (McNemar) needs each clip's outcome under BOTH "
            "regimes. The sliding-window per-clip outcomes are committed; the "
            "clean baseline's per-clip predictions were never recovered from "
            "Kaggle -- only its aggregate metrics and confusion matrix were "
            "transcribed. Without per-clip pairing no discordant-pair count "
            "exists, and McNemar is undefined."
        ),
        "why_unpaired_tests_are_also_refused": (
            "an unpaired two-proportion test on the same 394 clips would be "
            "wrong: the samples are not independent, they are the identical "
            "clips scored twice. It would report a p-value that means nothing."
        ),
        "what_can_be_said": (
            "both point estimates and their descriptive intervals may be "
            "reported side by side, noting that they come from the same "
            "checkpoint on the same clips under different inference regimes. "
            "No significance claim is supported."
        ),
        "what_would_unblock_it": (
            "the clean baseline's per-clip predictions.csv from Kaggle notebook "
            "7bb9a86555 v3 (clean_experiment/final_evaluation/). With it, "
            "McNemar's exact test on the 394 paired outcomes becomes available."
        ),
        "alternative_pairing_considered_and_rejected": (
            "Experiment 1's predictions.csv IS available on the same 394 clips, "
            "but pairing against it would confound two changes at once -- a "
            "different checkpoint (notebook5d98537e15 v2) AND a different "
            "regime -- and its rows carry no clip identifier, so positional "
            "alignment could not be verified either."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "SLIDING-WINDOW SYSTEM RESULT -- UNCERTAINTY",
        "=" * 80,
        f"regime      : {report['regime']}",
        f"checkpoint  : {report['checkpoint']}",
        f"samples     : {report['sample_count']}  {report['class_distribution']}",
        "",
        "REPRODUCTION CHECK (no checkpoint needed)",
        "-" * 80,
        f"  recomputed {report['reproduction_check']['recomputed']}",
        f"  recorded   {report['reproduction_check']['recorded']}",
        f"  exact match: {report['reproduction_check']['exact_match']}   "
        f"accuracy diff {report['reproduction_check']['accuracy_difference']:+.2e}",
        "",
        "TIER 1 -- Wilson, denominator fixed by the split",
        "-" * 80,
        f"{'metric':22s} {'point':>9s}  {'95% CI':>20s}  {'counts':>11s}  width",
    ]
    for name in ("accuracy", "recall_sensitivity", "specificity", "prevalence_fight"):
        row = report["wilson_intervals"][name]
        lines.append(
            f"{name:22s} {row['point']:9.4f}  [{row['ci_low']:.4f}, {row['ci_high']:.4f}]  "
            f"{row['successes']:>5}/{row['total']:<5}  {100*(row['ci_high']-row['ci_low']):6.2f}pp"
        )
    lines += ["", "TIER 2 -- Wilson, CONDITIONAL on a model-chosen denominator", "-" * 80]
    for name in ("precision_ppv", "npv"):
        row = report["wilson_intervals"][name]
        lines.append(
            f"{name:22s} {row['point']:9.4f}  [{row['ci_low']:.4f}, {row['ci_high']:.4f}]  "
            f"{row['successes']:>5}/{row['total']:<5}  {100*(row['ci_high']-row['ci_low']):6.2f}pp"
        )
    lines += ["", "TIER 3 -- percentile bootstrap (not binomial proportions)", "-" * 80]
    for name in ("f1", "roc_auc", "pr_auc"):
        row = report["bootstrap_intervals_stratified"][name]
        lines.append(
            f"{name:22s} {row['point']:9.4f}  [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        )
    lines += ["", f"  {report['rank_metric_note']}"]

    cluster = report["cluster_structure"]
    lines += [
        "",
        "SOURCE-VIDEO DEPENDENCE (grouping INFERRED from filenames, not proven)",
        "-" * 80,
        f"  {cluster['clips']} clips from {cluster['clusters']} inferred source videos "
        f"(mean {cluster['mean_cluster_size']:.2f}, max {cluster['max_cluster_size']})",
        f"  {'metric':12s} {'clip-level':>26s} {'source-video-level':>26s}  ratio",
    ]
    for name in ("accuracy", "f1", "roc_auc", "pr_auc"):
        a = report["bootstrap_intervals_stratified"][name]
        b = report["bootstrap_intervals_clustered"][name]
        if a["ci_low"] is None or b["ci_low"] is None:
            continue
        wa, wb = a["ci_high"] - a["ci_low"], b["ci_high"] - b["ci_low"]
        lines.append(
            f"  {name:12s} [{a['ci_low']:.4f}, {a['ci_high']:.4f}] {100*wa:6.2f}pp  "
            f"[{b['ci_low']:.4f}, {b['ci_high']:.4f}] {100*wb:6.2f}pp  {wb/wa:5.2f}x"
        )

    latency = report["time_to_alarm"]
    uncond = latency["unconditional_over_all_fight_clips"]
    cond = latency["conditional_on_detection"]
    boot = latency["bootstrap_unconditional_median"]
    lines += [
        "",
        "TIME TO ALARM -- censoring handled explicitly",
        "-" * 80,
        f"  Fight clips {latency['n_fight_clips']}   detected {latency['n_detected']}   "
        f"right-censored {latency['n_right_censored']}   "
        f"detected fraction {latency['detected_fraction']:.3f}",
        f"  {latency['censoring']}",
        "",
        "  Over ALL Fight clips (exact below the detected fraction, not estimated):",
        f"    q25 {uncond['q25_seconds']}s   median {uncond['median_seconds']}s   "
        f"q75 {uncond['q75_seconds']}s   q80 {uncond['q80_seconds']}s",
        f"    quantiles above {uncond['highest_reachable_quantile']:.2f} are never reached",
        "",
        "  Conditional on detection (what Step 2 reported):",
        f"    median {cond['median_seconds']:.4f}s   p75 {cond['p75_seconds']:.4f}s   "
        f"mean {cond['mean_seconds']:.4f}s",
        "",
        f"  {latency['do_not_conflate']}",
        "",
        f"  Unconditional median, bootstrap CI:",
        f"    clip-level        {boot['clip_level']['median']}s "
        f"[{boot['clip_level']['ci_low']}, {boot['clip_level']['ci_high']}]",
        f"    source-video-level {boot['source_video_level']['median']}s "
        f"[{boot['source_video_level']['ci_low']}, {boot['source_video_level']['ci_high']}]",
    ]

    clean = report["clean_baseline"]
    lines += [
        "",
        "CLEAN BASELINE (whole-clip, threshold 0.16) -- for reference only",
        "-" * 80,
        f"{'metric':22s} {'point':>9s}  {'95% CI':>20s}  {'counts':>11s}",
    ]
    for name in ("accuracy", "recall_sensitivity", "specificity", "precision_ppv", "npv"):
        row = clean["wilson_intervals"][name]
        lines.append(
            f"{name:22s} {row['point']:9.4f}  [{row['ci_low']:.4f}, {row['ci_high']:.4f}]  "
            f"{row['successes']:>5}/{row['total']:<5}"
        )
    lines += ["", "  NOT available for the clean baseline:"]
    for key, reason in clean["not_available"].items():
        lines.append(f"    {key}: {reason}")

    paired = report["paired_comparison"]
    lines += [
        "",
        "PAIRED COMPARISON 0.7893 vs 0.8325",
        "-" * 80,
        f"  formal test possible: {paired['formal_test_possible']}",
        f"  {paired['blocking_reason']}",
        "",
        f"  {paired['why_unpaired_tests_are_also_refused']}",
        "",
        f"  What can be said: {paired['what_can_be_said']}",
        f"  What would unblock it: {paired['what_would_unblock_it']}",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = compute(args.confidence, args.resamples, args.seed)
    print(render(report))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
