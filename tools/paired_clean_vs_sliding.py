#!/usr/bin/env python3
"""
tools/paired_clean_vs_sliding.py

Paired comparison of the two inference regimes that share the clean-baseline
checkpoint on the same 394-clip primary split:

    A) clean baseline    one whole-clip forward pass, threshold 0.16
    B) sliding window    17 windows per clip, frozen rule max >= 0.14

The observations are PAIRED -- the identical clips scored twice -- so an
independent-samples two-proportion test would be wrong and is not offered here.
McNemar's exact test is used instead, on discordant pairs.

ALIGNMENT IS A PRECONDITION, NOT AN ASSUMPTION
-----------------------------------------------
The recovered clean-baseline ``predictions.csv`` carries no clip identifier.
This script refuses to run unless
``tools/recover_clean_baseline_alignment.py`` has produced an alignment report
whose ``alignment_supported`` flag is true -- that is, unless re-running the
checkpoint locally in known clip order reproduced the recovered probability
sequence position by position.

Reads only. Writes a report, a JSON record and a disagreement table under
``outputs/temporal_risk/paired_clean_vs_sliding_window/``. No committed
evaluation artifact is modified.

Usage:
    python tools/paired_clean_vs_sliding.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import numpy as np  # noqa: E402

from metric_intervals import cluster_ids, cluster_summary, wilson_interval  # noqa: E402
from temporal_event_adapter import (  # noqa: E402
    FrozenAggregationRule,
    PrecomputedWindowScoreSource,
)

OUTPUT_DIR = REPO_ROOT / "outputs" / "temporal_risk" / "paired_clean_vs_sliding_window"
ALIGNMENT_JSON = OUTPUT_DIR / "clean_baseline_alignment.json"
PRIMARY_CSV = REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"
FROZEN_JSON = REPO_ROOT / "temporal_risk" / "frozen_aggregation.json"
AGGREGATION_JSON = REPO_ROOT / "temporal_risk" / "primary_aggregation_result.json"
WINDOW_MANIFEST = REPO_ROOT / "temporal_risk" / "window_scoring_manifest.json"

CLEAN_THRESHOLD = 0.16
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
SOURCE_FPS = 30.0
# The final window covers frames 128..143, so the latest observable alarm time
# is (143 + 1) / 30 = 4.8 s. That is also the administrative censoring time.
CENSORING_SECONDS = 4.8


class PairingError(RuntimeError):
    """Raised when a paired analysis cannot be performed safely."""


# --------------------------------------------------------------------------
# Exact tests
# --------------------------------------------------------------------------


def mcnemar_exact(b: int, c: int) -> dict:
    """Two-sided exact McNemar test on discordant counts ``b`` and ``c``.

    Conditional on the number of discordant pairs ``n = b + c``, the count ``b``
    is Binomial(n, 1/2) under the null that the two methods are equally likely
    to be the one that is right. The exact two-sided p-value doubles the smaller
    tail, capped at 1. No normal approximation and no continuity correction --
    with counts this small an approximation would be the wrong tool.
    """
    n = b + c
    if n == 0:
        return {
            "b": b, "c": c, "discordant": 0, "p_value": 1.0,
            "method": "exact binomial (no discordant pairs; test undefined, reported as p = 1)",
        }
    smaller = min(b, c)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2 ** n)
    return {
        "b": b,
        "c": c,
        "discordant": n,
        "p_value": min(1.0, 2.0 * tail),
        "one_sided_p_value": tail,
        "method": "two-sided exact binomial on discordant pairs (McNemar)",
    }


def paired_table(first_correct: np.ndarray, second_correct: np.ndarray) -> Dict[str, int]:
    """The 2x2 agreement table between two boolean outcome vectors."""
    return {
        "both_correct": int(np.sum(first_correct & second_correct)),
        "first_correct_second_wrong": int(np.sum(first_correct & ~second_correct)),
        "first_wrong_second_correct": int(np.sum(~first_correct & second_correct)),
        "both_wrong": int(np.sum(~first_correct & ~second_correct)),
    }


def confusion(truth: np.ndarray, predicted: np.ndarray) -> Dict[str, int]:
    return {
        "true_positive": int(np.sum((truth == 1) & (predicted == 1))),
        "false_positive": int(np.sum((truth == 0) & (predicted == 1))),
        "true_negative": int(np.sum((truth == 0) & (predicted == 0))),
        "false_negative": int(np.sum((truth == 1) & (predicted == 0))),
    }


def metrics_from(matrix: Dict[str, int]) -> Dict[str, Optional[float]]:
    tp, fp = matrix["true_positive"], matrix["false_positive"]
    tn, fn = matrix["true_negative"], matrix["false_negative"]
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall)
        else None
    )
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": f1,
    }


def bootstrap_accuracy_difference(
    clean_correct: np.ndarray,
    sliding_correct: np.ndarray,
    clusters: Optional[np.ndarray] = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict:
    """Percentile bootstrap for the PAIRED accuracy difference.

    Each draw resamples clips (or whole source videos), carrying both methods'
    outcomes for a clip together, so the pairing is preserved. Resampling the
    two methods independently would destroy exactly the correlation that makes
    the paired comparison informative.
    """
    generator = np.random.default_rng(seed)
    difference = float(np.mean(sliding_correct) - np.mean(clean_correct))
    size = clean_correct.size

    if clusters is not None:
        unique = np.unique(clusters)
        members = [np.flatnonzero(clusters == value) for value in unique]

    values = []
    for _ in range(resamples):
        if clusters is None:
            picked = generator.integers(0, size, size=size)
        else:
            drawn = generator.integers(0, len(members), size=len(members))
            picked = np.concatenate([members[i] for i in drawn])
        values.append(
            float(np.mean(sliding_correct[picked]) - np.mean(clean_correct[picked]))
        )

    tail = (1.0 - confidence) / 2.0
    array = np.asarray(values)
    return {
        "difference": difference,
        "ci_low": float(np.quantile(array, tail)),
        "ci_high": float(np.quantile(array, 1 - tail)),
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "method": (
            "paired percentile bootstrap over source-video clusters"
            if clusters is not None
            else "paired percentile bootstrap over clips"
        ),
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_alignment(path: Path) -> List[dict]:
    if not path.is_file():
        raise PairingError(
            f"{path} not found. Run tools/recover_clean_baseline_alignment.py "
            "first: without a proven clip alignment no paired test is valid."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("alignment_supported"):
        raise PairingError(
            "The alignment report does not support the clip mapping "
            f"(label_mismatches={report.get('label_mismatches')}, "
            f"rows_over_tolerance={report.get('rows_over_tolerance')}). "
            "Paired testing cannot be performed safely."
        )
    return report


def build_paired_records(alignment: dict) -> Tuple[List[dict], dict]:
    """Join the aligned clean predictions to the sliding-window decisions."""
    rule = FrozenAggregationRule.load(FROZEN_JSON)
    source = PrecomputedWindowScoreSource(PRIMARY_CSV)

    records: List[dict] = []
    for row in alignment["per_clip"]:
        clip = row["clip"]
        windows = source.get(clip)
        if windows is None:
            raise PairingError(
                f"{clip} is in the clean-baseline evaluation but has no "
                "sliding-window scores; the two sets are not the same clips."
            )
        sliding_label = source.true_label(clip)
        if sliding_label != row["recovered_label"]:
            raise PairingError(
                f"Label disagreement for {clip}: clean says "
                f"{row['recovered_label']!r}, sliding-window says {sliding_label!r}."
            )

        probabilities = [w.fight_probability for w in windows]
        ordered = sorted(windows, key=lambda w: w.window_index)
        alarm_frame = None
        running: List[float] = []
        for window in ordered:
            running.append(window.fight_probability)
            if rule.decide(running):
                alarm_frame = window.last_frame
                break

        truth = 1 if sliding_label == "Fight" else 0
        clean_probability = row["recovered_probability"]
        clean_predicted = int(clean_probability >= CLEAN_THRESHOLD)
        sliding_predicted = int(rule.decide(probabilities))
        peak = max(ordered, key=lambda w: w.fight_probability)

        records.append(
            {
                "index": row["index"],
                "clip": clip,
                "true_label": sliding_label,
                "truth": truth,
                "clean_probability": clean_probability,
                "clean_predicted": "Fight" if clean_predicted else "NonFight",
                "clean_correct": clean_predicted == truth,
                "sliding_max_probability": max(probabilities),
                "sliding_peak_window_index": peak.window_index,
                "sliding_predicted": "Fight" if sliding_predicted else "NonFight",
                "sliding_correct": sliding_predicted == truth,
                "first_alarm_window": (
                    next((w.window_index for w in ordered if w.last_frame == alarm_frame), None)
                    if alarm_frame is not None else None
                ),
                "first_alarm_frame": alarm_frame,
                "first_alarm_seconds": (
                    round((alarm_frame + 1) / SOURCE_FPS, 4) if alarm_frame is not None else None
                ),
                "agreement": (
                    "both_correct" if (clean_predicted == truth) and (sliding_predicted == truth)
                    else "both_wrong" if (clean_predicted != truth) and (sliding_predicted != truth)
                    else "clean_correct_sliding_wrong" if clean_predicted == truth
                    else "clean_wrong_sliding_correct"
                ),
            }
        )

    extra = set(source.clips()) - {r["clip"] for r in records}
    if extra:
        raise PairingError(
            f"{len(extra)} sliding-window clips are absent from the clean-baseline "
            f"evaluation, e.g. {sorted(extra)[:3]}."
        )

    coverage = {
        "clips": len(records),
        "clean_clips": len(alignment["per_clip"]),
        "sliding_clips": len(source.clips()),
        "identical_clip_sets": True,
        "labels_agree_for_every_clip": True,
    }
    return records, coverage


def checkpoint_provenance() -> dict:
    """Confirm the sliding-window scores came from the clean-baseline checkpoint."""
    manifest = json.loads(WINDOW_MANIFEST.read_text(encoding="utf-8"))
    checkpoint = manifest["checkpoint"]
    return {
        "window_scoring_checkpoint": checkpoint,
        "clean_experiment": manifest.get("clean_experiment"),
        "is_clean_baseline_notebook": "notebook7bb9a86555" in checkpoint,
        "is_experiment1_notebook": "notebook5d98537e15" in checkpoint,
        "note": (
            "both regimes use the same clean-baseline checkpoint, so the "
            "comparison isolates the inference regime rather than the model"
        ),
    }


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def analyse(records: List[dict], coverage: dict) -> dict:
    truth = np.array([r["truth"] for r in records])
    clean_pred = np.array([1 if r["clean_predicted"] == "Fight" else 0 for r in records])
    sliding_pred = np.array([1 if r["sliding_predicted"] == "Fight" else 0 for r in records])
    clean_correct = clean_pred == truth
    sliding_correct = sliding_pred == truth
    clips = [r["clip"] for r in records]

    clean_matrix = confusion(truth, clean_pred)
    sliding_matrix = confusion(truth, sliding_pred)

    overall_table = paired_table(clean_correct, sliding_correct)
    overall_test = mcnemar_exact(
        overall_table["first_correct_second_wrong"],
        overall_table["first_wrong_second_correct"],
    )

    # Stratum 1: among Fight clips, does each regime DETECT the fight?
    fight = truth == 1
    fight_table = paired_table(clean_pred[fight] == 1, sliding_pred[fight] == 1)
    fight_test = mcnemar_exact(
        fight_table["first_correct_second_wrong"],
        fight_table["first_wrong_second_correct"],
    )

    # Stratum 2: among NonFight clips, does each regime CORRECTLY REJECT?
    nonfight = truth == 0
    nonfight_table = paired_table(clean_pred[nonfight] == 0, sliding_pred[nonfight] == 0)
    nonfight_test = mcnemar_exact(
        nonfight_table["first_correct_second_wrong"],
        nonfight_table["first_wrong_second_correct"],
    )

    groups = cluster_ids(clips)
    accuracy_difference = {
        "clip_level": bootstrap_accuracy_difference(clean_correct, sliding_correct),
        "source_video_level": bootstrap_accuracy_difference(
            clean_correct, sliding_correct, clusters=groups
        ),
    }

    clean_metrics = metrics_from(clean_matrix)
    sliding_metrics = metrics_from(sliding_matrix)
    comparison = {}
    for name in ("accuracy", "precision", "recall", "specificity", "f1"):
        first, second = clean_metrics[name], sliding_metrics[name]
        comparison[name] = {
            "clean_baseline": first,
            "sliding_window": second,
            "difference_sliding_minus_clean": (
                (second - first) if (first is not None and second is not None) else None
            ),
        }

    disagreements = [r for r in records if r["clean_correct"] != r["sliding_correct"]]
    detected = [r for r in records if r["truth"] == 1 and r["first_alarm_frame"] is not None]
    censored = [r for r in records if r["truth"] == 1 and r["first_alarm_frame"] is None]
    false_alarms = [r for r in records if r["truth"] == 0 and r["first_alarm_frame"] is not None]
    latencies = sorted(r["first_alarm_seconds"] for r in detected)
    n_fight = int(fight.sum())

    def unconditional(quantile: float) -> Optional[float]:
        need = quantile * n_fight
        if need > len(latencies):
            return None
        return latencies[max(int(math.ceil(need)) - 1, 0)]

    early = {
        "fight_clips": n_fight,
        "detected": len(detected),
        "not_detected_right_censored": len(censored),
        "detected_on_first_window": sum(1 for r in detected if r["first_alarm_window"] == 0),
        "conditional_on_detection": {
            "median_seconds": statistics.median(latencies) if latencies else None,
            "mean_seconds": statistics.fmean(latencies) if latencies else None,
            "p75_seconds": (
                statistics.quantiles(latencies, n=4, method="inclusive")[2]
                if len(latencies) > 1 else None
            ),
            "min_seconds": latencies[0] if latencies else None,
            "max_seconds": latencies[-1] if latencies else None,
        },
        "censoring_aware_over_all_fight_clips": {
            "note": (
                "censoring is administrative and identical: every non-detection is "
                "censored at 4.8 s, which is the completion time of the FINAL window "
                "((143 + 1) / 30), not a time beyond it. Detections can therefore land "
                "on that same timestamp, and some do -- see detections_at_censoring_time. "
                "Quantiles at or below the detected fraction are still exact rather than "
                "estimated, because every one of them falls strictly below 4.8 s"
            ),
            "q25_seconds": unconditional(0.25),
            "median_seconds": unconditional(0.50),
            "q75_seconds": unconditional(0.75),
            "detected_fraction": len(detected) / n_fight if n_fight else None,
            "detections_at_censoring_time": sum(
                1 for r in detected if r["first_alarm_seconds"] == CENSORING_SECONDS
            ),
            "quantiles_strictly_below_censoring_time": all(
                value is None or value < CENSORING_SECONDS
                for value in (
                    unconditional(0.25), unconditional(0.50), unconditional(0.75)
                )
            ),
        },
        "false_alarm_timing": {
            "count": len(false_alarms),
            "median_seconds": (
                statistics.median([r["first_alarm_seconds"] for r in false_alarms])
                if false_alarms else None
            ),
            "mean_seconds": (
                statistics.fmean([r["first_alarm_seconds"] for r in false_alarms])
                if false_alarms else None
            ),
        },
        "censoring_warning": (
            "the 4.8 s censoring time is NEVER imputed as a detection time; censored "
            "clips are excluded from the latency statistics and reported separately. "
            "A clip that genuinely alarms on the final window also carries 4.8 s, but "
            "that value is an observed detection, not an imputed one"
        ),
    }

    return {
        "tool": "paired_clean_vs_sliding",
        "coverage": coverage,
        "checkpoint_provenance": checkpoint_provenance(),
        "regimes": {
            "clean_baseline": {
                "description": "one whole-clip forward pass over 16 frames sampled across 150",
                "threshold": CLEAN_THRESHOLD,
                "applied_to": "a single clip-level probability",
            },
            "sliding_window": {
                "description": "17 windows of 16 consecutive frames at stride 8",
                "rule": "max over window probabilities >= 0.14",
                "applied_to": "the maximum of 17 window probabilities",
            },
            "why_not_merely_a_threshold_comparison": (
                "the two thresholds act on different random variables: one clip-level "
                "probability versus the maximum of 17. A maximum of 17 draws is "
                "stochastically larger than a single draw, so the numeric closeness of "
                "0.16 and 0.14 does not make these comparable operating points"
            ),
        },
        "confusion_matrices": {
            "clean_baseline": clean_matrix,
            "sliding_window": sliding_matrix,
        },
        "metrics": comparison,
        "wilson_intervals": {
            "clean_accuracy": list(
                wilson_interval(
                    clean_matrix["true_positive"] + clean_matrix["true_negative"], len(records)
                )
            ),
            "sliding_accuracy": list(
                wilson_interval(
                    sliding_matrix["true_positive"] + sliding_matrix["true_negative"], len(records)
                )
            ),
        },
        "paired_overall": {
            "table": overall_table,
            "mcnemar": overall_test,
            "accuracy_difference_sliding_minus_clean": accuracy_difference,
        },
        "paired_fight_only": {
            "tested": "among the Fight clips, whether each regime detects the fight",
            "table": fight_table,
            "mcnemar": fight_test,
        },
        "paired_nonfight_only": {
            "tested": "among the NonFight clips, whether each regime correctly rejects",
            "table": nonfight_table,
            "mcnemar": nonfight_test,
        },
        "cluster_structure": cluster_summary(clips),
        "early_detection": early,
        "disagreements": {
            "total": len(disagreements),
            "clean_correct_sliding_wrong": sum(
                1 for r in disagreements if r["agreement"] == "clean_correct_sliding_wrong"
            ),
            "clean_wrong_sliding_correct": sum(
                1 for r in disagreements if r["agreement"] == "clean_wrong_sliding_correct"
            ),
            "by_true_label": {
                label: {
                    "clean_correct_sliding_wrong": sum(
                        1 for r in disagreements
                        if r["true_label"] == label
                        and r["agreement"] == "clean_correct_sliding_wrong"
                    ),
                    "clean_wrong_sliding_correct": sum(
                        1 for r in disagreements
                        if r["true_label"] == label
                        and r["agreement"] == "clean_wrong_sliding_correct"
                    ),
                }
                for label in ("Fight", "NonFight")
            },
        },
        "per_clip": records,
    }


DISAGREEMENT_FIELDS = (
    "clip", "true_label", "clean_probability", "clean_predicted",
    "sliding_max_probability", "sliding_peak_window_index", "sliding_predicted",
    "first_alarm_window", "first_alarm_frame", "first_alarm_seconds", "agreement",
)


def write_disagreement_table(records: List[dict], path: Path) -> int:
    rows = [r for r in records if r["clean_correct"] != r["sliding_correct"]]
    rows.sort(key=lambda r: (r["true_label"], r["agreement"], r["clip"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DISAGREEMENT_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in DISAGREEMENT_FIELDS})
    return len(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT_JSON)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    try:
        alignment = load_alignment(args.alignment)
        records, coverage = build_paired_records(alignment)
    except PairingError as error:
        print(f"PAIRED ANALYSIS REFUSED: {error}", file=sys.stderr)
        return 2

    report = analyse(records, coverage)
    report["alignment_evidence"] = {
        "clean_predictions_sha256": alignment["clean_predictions_sha256"],
        "checkpoint_sha256": alignment["checkpoint_sha256"],
        "hypothesis": alignment["hypothesis"],
        "basis": alignment["basis"],
        "exact_matches_at_six_decimals": alignment["exact_matches_at_six_decimals"],
        "max_abs_difference": alignment["max_abs_difference"],
        "label_mismatches": alignment["label_mismatches"],
        "alignment_supported": alignment["alignment_supported"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "paired_comparison.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output_dir / "disagreement_table.csv"
    written = write_disagreement_table(records, csv_path)

    overall = report["paired_overall"]
    accuracy = report["metrics"]["accuracy"]
    print("PAIRED COMPARISON -- clean baseline vs sliding window")
    print("=" * 72)
    print(f"  clips              : {coverage['clips']}")
    print(f"  clean confusion    : {report['confusion_matrices']['clean_baseline']}")
    print(f"  sliding confusion  : {report['confusion_matrices']['sliding_window']}")
    print(f"  accuracy           : clean {accuracy['clean_baseline']:.4f} vs sliding "
          f"{accuracy['sliding_window']:.4f} "
          f"({accuracy['difference_sliding_minus_clean']:+.4f})")
    print()
    print(f"  paired table       : {overall['table']}")
    print(f"  McNemar exact p    : {overall['mcnemar']['p_value']:.6g}  "
          f"(b={overall['mcnemar']['b']}, c={overall['mcnemar']['c']})")
    print(f"  Fight-only    p    : {report['paired_fight_only']['mcnemar']['p_value']:.6g}  "
          f"{report['paired_fight_only']['table']}")
    print(f"  NonFight-only p    : {report['paired_nonfight_only']['mcnemar']['p_value']:.6g}  "
          f"{report['paired_nonfight_only']['table']}")
    difference = overall["accuracy_difference_sliding_minus_clean"]
    print(f"  acc diff CI clip   : [{difference['clip_level']['ci_low']:+.4f}, "
          f"{difference['clip_level']['ci_high']:+.4f}]")
    print(f"  acc diff CI source : [{difference['source_video_level']['ci_low']:+.4f}, "
          f"{difference['source_video_level']['ci_high']:+.4f}]")
    print()
    print(f"  disagreements      : {report['disagreements']['total']}")
    print(f"  wrote {json_path}")
    print(f"  wrote {csv_path} ({written} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
