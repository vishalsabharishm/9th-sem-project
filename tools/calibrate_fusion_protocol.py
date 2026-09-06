#!/usr/bin/env python3
"""
tools/calibrate_fusion_protocol.py

Calibrate and FREEZE the confirmatory fusion protocol on development data, so
that a later authorised task can evaluate it on the protected primary split in
one shot with no free parameters left.

WHY THE CALIBRATION SET CHANGED
-------------------------------
The first fusion protocol calibrated on a "fresh" carve drawn from the untouched
training remainder. That carve was source-disjoint from everything and untouched
by selection, but it was NOT untouched by training: the 240-clip carve is the
R3D holdout (validation_fraction 0.15 of train), so the remainder it came from
is the R3D training set. Measured signature: recall 0.9583 there versus 0.7143
on the holdout, a +0.244 memorization gap. See
outputs/fusion/fusion_blocked_contamination.md. That carve is permanently
invalid for validating a branch trained on it and is not used here.

The 240-clip carve is used instead. It is genuine R3D HOLDOUT data -- the
property that matters for calibrating a rule that sits on top of R3D. It has
been spent three times for SELECTION (best epoch, the 215-candidate aggregation
sweep, the v2 matched-FPR protocol) and once for spatial characterization, so it
is a DEVELOPMENT resource and cannot be described as an unbiased test set. No
number produced here is a performance claim. The confirmatory estimate comes
from primary, which this script never reads.

WHAT IS FROZEN HERE
-------------------
Every quantity the primary evaluation will need: the three thresholds, the fixed
0.5/0.5 weight, and the empirical percentile reference distributions. Storing
the reference distributions explicitly is what makes it impossible for the
primary run to normalise against its own scores -- primary values are ranked
against carve values, and the carve values travel inside the frozen artifact.

WHAT IS NOT DONE HERE
---------------------
No feature is redesigned, no candidate is added, no weight is searched, and no
sweep is run beyond the pre-specified exhaustive threshold calibration that the
already-reviewed aggregation protocol uses. The primary split is not opened.

Usage:
    python tools/calibrate_fusion_protocol.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from checkpoint_identity import file_sha256  # noqa: E402
from metric_intervals import source_video_id  # noqa: E402

CARVE_WINDOWS = REPO_ROOT / "temporal_risk" / "carve_window_scores.csv"
CARVE_SPATIAL = REPO_ROOT / "outputs" / "fusion" / "carve_spatial_characterization.csv"
CHECKPOINT = REPO_ROOT / "models" / "temporal_violence" / "best.pt"
YOLO_WEIGHTS = REPO_ROOT / "models" / "yolov8s.pt"
DEFAULT_OUTPUT = REPO_ROOT / "temporal_risk" / "frozen_confirmatory_fusion_protocol.json"

PROTOCOL_VERSION = "confirmatory-fusion-v2-2026-09"
TEMPORAL_THRESHOLD = 0.14      # frozen incumbent; never re-tuned
F2_WEIGHT = 0.5                # fixed design choice, NOT fitted
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
SIGNIFICANCE_LEVEL = 0.05
NON_INFERIORITY_MARGIN_CLIPS = 10


class CalibrationError(RuntimeError):
    """Raised when the development data cannot support the frozen protocol."""


def load_development() -> List[dict]:
    """Join carve temporal scores to carve spatial features. Carve only."""
    windows: Dict[str, List[float]] = {}
    labels: Dict[str, str] = {}
    with open(CARVE_WINDOWS, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "carve":
                raise CalibrationError(
                    f"{CARVE_WINDOWS} contains split {row['split']!r}; carve only."
                )
            windows.setdefault(row["clip"], []).append(float(row["fight_probability"]))
            labels[row["clip"]] = row["true_label"]

    rows: List[dict] = []
    with open(CARVE_SPATIAL, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            clip = record["clip"]
            if clip not in windows:
                raise CalibrationError(f"{clip} has spatial features but no carve windows.")
            speed = record.get("speed_mean") or ""
            diagonal = record.get("diag_mean") or ""
            spatial = None
            if speed and diagonal and float(diagonal) > 0:
                spatial = float(speed) / float(diagonal)
            rows.append({
                "clip": clip,
                "label": labels[clip],
                "truth": 1 if labels[clip] == "Fight" else 0,
                "temporal_max": max(windows[clip]),
                "spatial": spatial,
                "source": source_video_id(clip),
            })
    return rows


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


def calibrate(rows: Sequence[dict], score_of: Callable, budget: int) -> dict:
    """Deterministic threshold calibration to a fixed false-positive budget.

    The grid is every distinct score present, so the search is exhaustive over
    distinct confusion matrices rather than sampled -- it cannot miss an optimum
    to grid resolution. Ordering: false-positive count closest to the budget,
    then most true positives (recall at matched alarm rate, the one
    pre-specified criterion), then the highest threshold purely so ties resolve
    deterministically. A clip whose score is undefined never fires.
    """
    values = sorted({score_of(r) for r in rows if score_of(r) is not None})
    if not values:
        raise CalibrationError("no development clip has a defined score for this candidate.")
    best = None
    for threshold in values:
        decisions = {
            r["clip"]: (score_of(r) is not None and score_of(r) >= threshold) for r in rows
        }
        matrix = confusion(rows, decisions)
        key = (abs(matrix["fp"] - budget), -matrix["tp"], -threshold)
        if best is None or key < best[0]:
            best = (key, threshold, matrix)
    _, threshold, matrix = best
    ties = [
        t for t in values
        if confusion(rows, {r["clip"]: (score_of(r) is not None and score_of(r) >= t)
                            for r in rows}) == matrix
    ]
    return {
        "threshold": threshold,
        "development_matrix": matrix,
        "achieved_false_positives": matrix["fp"],
        "budget": budget,
        "exact_budget_match": matrix["fp"] == budget,
        "thresholds_giving_an_identical_matrix": len(ties),
        "tie_break": "highest threshold among those with the identical confusion matrix",
    }


def percentile_reference(values: Sequence[Optional[float]]) -> List[float]:
    """The empirical CDF that primary values will be ranked against.

    Stored explicitly in the frozen artifact. This is the mechanism that makes
    it impossible for the confirmatory run to normalise primary against its own
    distribution: the reference travels with the protocol.
    """
    return sorted(float(v) for v in values if v is not None)


def percentile_of(value: Optional[float], reference: Sequence[float]) -> Optional[float]:
    if value is None or not reference:
        return None
    low, high = 0, len(reference)
    while low < high:
        middle = (low + high) // 2
        if reference[middle] < value:
            low = middle + 1
        else:
            high = middle
    return low / len(reference)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    rows = load_development()
    defined = [r for r in rows if r["spatial"] is not None]
    undefined = [r for r in rows if r["spatial"] is None]

    b0 = {r["clip"]: r["temporal_max"] >= TEMPORAL_THRESHOLD for r in rows}
    b0_matrix = confusion(rows, b0)
    budget = b0_matrix["fp"]

    spatial_reference = percentile_reference([r["spatial"] for r in rows])
    temporal_reference = percentile_reference([r["temporal_max"] for r in rows])

    cal_b1 = calibrate(rows, lambda r: r["spatial"], budget)

    def or_rule(row, threshold):
        return (row["temporal_max"] >= TEMPORAL_THRESHOLD) or \
               (row["spatial"] is not None and row["spatial"] >= threshold)

    best = None
    for threshold in sorted({r["spatial"] for r in defined}):
        matrix = confusion(rows, {r["clip"]: or_rule(r, threshold) for r in rows})
        key = (abs(matrix["fp"] - budget), -matrix["tp"], -threshold)
        if best is None or key < best[0]:
            best = (key, threshold, matrix)
    cal_f1 = {"threshold": best[1], "development_matrix": best[2],
              "achieved_false_positives": best[2]["fp"], "budget": budget,
              "exact_budget_match": best[2]["fp"] == budget}

    def f2_score(row):
        temporal = percentile_of(row["temporal_max"], temporal_reference)
        spatial = percentile_of(row["spatial"], spatial_reference)
        if temporal is None:
            return None
        return F2_WEIGHT * temporal + (1 - F2_WEIGHT) * (spatial if spatial is not None else 0.0)

    cal_f2 = calibrate(rows, f2_score, budget)

    record = {
        "tool": "calibrate_fusion_protocol",
        "protocol_version": PROTOCOL_VERSION,
        "artifact_kind": "frozen confirmatory fusion protocol",
        "status": "FROZEN -- awaiting explicit authorisation for a one-shot primary evaluation",
        "primary_data_accessed": False,
        "primary_evaluation_performed": False,

        "supersedes": {
            "artifact": "temporal_risk/frozen_fusion_protocol.json",
            "reason": (
                "that protocol calibrated on a carve drawn from the R3D training "
                "remainder. Source-disjointness does not imply training-disjointness; "
                "the measured memorization gap was +0.244 recall. The candidate set, "
                "the spatial feature, the fixed weight and the calibration method are "
                "carried over UNCHANGED -- only the calibration dataset is substituted, "
                "because the previous one was proved invalid."
            ),
            "invalid_carve": "temporal_risk/fresh_validation_carve.json (retained as evidence, never to be used for validation)",
        },

        "development_data": {
            "identifier": "existing 240-clip carve (R3D holdout)",
            "temporal_source": "temporal_risk/carve_window_scores.csv",
            "spatial_source": "outputs/fusion/carve_spatial_characterization.csv",
            "clips_used": len(rows),
            "clips_in_full_carve": 240,
            "clips_unavailable": 5,
            "clips_unavailable_reason": (
                "5 carve clips are recorded under sanitised Kaggle filenames that cannot "
                "be mapped to local mojibake filenames; the mapping is UNKNOWN and was "
                "not reconstructed"
            ),
            "labels": {"Fight": sum(1 for r in rows if r["truth"] == 1),
                       "NonFight": sum(1 for r in rows if r["truth"] == 0)},
            "spatially_defined": len(defined),
            "spatially_undefined": len(undefined),
            "spatially_undefined_by_class": {
                "Fight": sum(1 for r in undefined if r["truth"] == 1),
                "NonFight": sum(1 for r in undefined if r["truth"] == 0)},
            "missingness_is_class_dependent": True,
            "missingness_bias_direction": (
                "undefined spatial scores abstain (treated as spatial-negative). Most "
                "undefined clips are NonFight, for which abstention gives the CORRECT "
                "answer, so this mildly favours specificity and must be stated when "
                "reporting"),
            "sources": len({r["source"] for r in rows}),
            "status": (
                "DEVELOPMENT resource only. Genuine R3D holdout, but already spent for "
                "checkpoint selection, two aggregation selections and spatial "
                "characterization. Not an unbiased test set; no number here is a "
                "performance claim."),
        },

        "spatial_feature": {
            "name": "speed_normalised_by_group_diagonal",
            "numerator": (
                "mean over all (track, frame) pairs of BehaviorAnalyzer._movement_pixels: "
                "Euclidean centroid displacement in raw image pixels between consecutive "
                "frames of the same track"),
            "denominator": (
                "mean over frames containing at least one person of the diagonal of the "
                "bounding box enclosing ALL person boxes in that frame"),
            "form": "RATIO OF MEANS, not mean of ratios",
            "form_caveat": (
                "mean(speed)/mean(diagonal) is not equal to mean(speed/diagonal). This is "
                "a pre-specified approximation, documented rather than silently changed."),
            "single_person_clips": (
                "defined: the group diagonal collapses to that person's own box diagonal, "
                "so the feature becomes speed relative to apparent body size"),
            "zero_person_frames": "contribute to neither numerator nor denominator",
            "undefined_when": "no consecutive-frame track pair exists, or the mean diagonal is zero",
            "zero_denominator": "guarded; yields None rather than a division error",
            "camera_motion_compensated": False,
            "known_limitations": [
                "raw pixel motion is not compensated for camera motion; RWF-2000 Fight "
                "clips may be filmed differently from NonFight clips, and this protocol "
                "cannot separate subject motion from camera motion",
                "tracker ID switches inflate apparent displacement; the high-quality "
                "subset probe argues against this dominating, but does not exclude it",
                "the feature is a WHOLE-CLIP aggregate and is therefore not a causal "
                "mid-clip alarm signal (see causality below)",
            ],
            "chosen_over": "raw speed_pixels",
            "choice_rationale": (
                "raw speed scored HIGHER on development data (AUC 0.855 vs 0.806). The "
                "normalised form is the conservative choice, taken to reduce the "
                "apparent-size confound, not the flattering one."),
        },

        "normalisation": {
            "method": "empirical percentile (rank) transform",
            "reference_distribution": "the development carve, stored explicitly below",
            "guarantee": (
                "primary scores are ranked against these stored carve values. The "
                "reference travels inside this artifact, so the confirmatory run cannot "
                "normalise primary against its own distribution."),
            "temporal_reference_values": temporal_reference,
            "spatial_reference_values": spatial_reference,
            "temporal_reference_n": len(temporal_reference),
            "spatial_reference_n": len(spatial_reference),
        },

        "candidates": {
            "B0_temporal_only": {
                "rule": "max(window_probabilities) >= 0.14",
                "parameters_frozen": {"threshold": TEMPORAL_THRESHOLD},
                "parameters_calibrated": [],
                "development_matrix": b0_matrix,
                "role": "frozen incumbent comparator",
                "monotone_in_prefix": True,
                "complexity": 0,
            },
            "B1_spatial_only": {
                "rule": "spatial >= theta_s",
                "parameters_calibrated": {"theta_s": cal_b1["threshold"]},
                "calibration": cal_b1,
                "role": "baseline",
                "monotone_in_prefix": False,
                "complexity": 1,
            },
            "F1_or_gate": {
                "rule": "(temporal_max >= 0.14) OR (spatial >= theta_or)",
                "parameters_frozen": {"temporal_threshold": TEMPORAL_THRESHOLD},
                "parameters_calibrated": {"theta_or": cal_f1["threshold"]},
                "calibration": cal_f1,
                "role": "fusion candidate",
                "monotone_in_prefix": False,
                "complexity": 2,
            },
            "F2_rank_sum": {
                "rule": "0.5*pct(temporal_max) + 0.5*pct(spatial) >= theta_f",
                "parameters_frozen": {"weight": F2_WEIGHT,
                                      "weight_status": "FIXED design choice, NOT fitted, never searched"},
                "parameters_calibrated": {"theta_f": cal_f2["threshold"]},
                "calibration": cal_f2,
                "role": "fusion candidate",
                "monotone_in_prefix": False,
                "complexity": 3,
            },
        },

        "operating_point": {
            "method": "false-positive budget matched to the incumbent on development data",
            "budget_false_positives": budget,
            "budget_rationale": (
                "turns each threshold into a calibration constant rather than a fitted "
                "parameter and reduces the comparison to recall at a matched alarm rate"),
            "grid": "every distinct score value present in the development set",
            "determinism": "total order on (|fp - budget|, -tp, -threshold); no RNG",
            "if_budget_unreachable": (
                "the closest achievable false-positive count is taken and "
                "exact_budget_match is recorded false; the protocol is not relaxed"),
        },

        "confirmatory_analysis": {
            "when": "only after explicit authorisation, in a separate task",
            "dataset": "primary 394 clips, opened exactly once",
            "comparison": "B0_temporal_only versus the frozen fusion candidates, paired",
            "primary_endpoint": {
                "question": "does the frozen fusion detector improve paired clip-level "
                            "classification over the frozen temporal-only incumbent?",
                "statistic": "exact McNemar on discordant pairs, two-sided",
                "significance_level": SIGNIFICANCE_LEVEL,
                "direction": "improvement = fusion correct where B0 is wrong, more often "
                             "than the reverse",
                "ties": "concordant pairs are uninformative for McNemar and are reported "
                        "but not tested; zero discordant pairs is reported as p = 1",
            },
            "secondary_endpoints": [
                "confusion matrices, accuracy, precision, recall, specificity, F1",
                "ROC-AUC and PR-AUC for score-valued candidates (B0, B1); not defined for "
                "the boolean OR gate",
                "Wilson intervals on accuracy and recall",
                "paired percentile bootstrap of the accuracy difference, clip-level AND "
                "source-video-clustered",
                "false-alarm non-inferiority against a margin of "
                f"{NON_INFERIORITY_MARGIN_CLIPS} clips",
            ],
            "secondary_significance": "NOT required in every secondary metric; secondaries "
                                      "are descriptive support",
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                          "pairing": "clips resampled as pairs, both arms together"},
            "source_grouping": "metric_intervals.source_video_id, the repository convention",
            "interpretation": {
                "A_improved": "McNemar p < 0.05 with fusion favoured: report fusion as "
                              "improving the detector, with effect size and CI",
                "B_directional": "point estimate favours fusion but p >= 0.05: report "
                                 "directional evidence, insufficient for superiority",
                "C_no_improvement": "no meaningful difference: report that spatial evidence "
                                    "did not translate into improved fused classification "
                                    "under the frozen protocol",
                "D_degraded": "McNemar p < 0.05 against fusion: reject fusion for the final "
                              "detector",
            },
            "prohibited_after_seeing_primary": [
                "changing any threshold", "changing the weight",
                "changing or adding a feature", "adding or removing a candidate",
                "re-running with alternative parameters",
                "inventing a success criterion not stated above",
            ],
        },

        "causality": {
            "temporal_branch": "causal: completed windows only, frozen aggregation, "
                               "first-alarm at (last_frame + 1)/30, scanning stops at the "
                               "first fire",
            "spatial_branch_feature": "NOT causal as frozen: it is a whole-clip aggregate "
                                      "over all 150 frames",
            "consequence": "the frozen fusion decision is a CLIP-LEVEL OFFLINE decision. It "
                           "is a fair comparison with B0, which also consumes all 17 "
                           "windows, but it is not a mid-clip alarm and must not be "
                           "described as one",
            "streaming_variant": "a prefix-mean version is computable but is NOT part of "
                                 "this protocol and has not been evaluated",
            "runtime_class": "causal/offline",
            "measured_throughput": "42.5-84 s per 5 s clip on CPU (YOLO-bound), i.e. "
                                   "8.5x-16.8x slower than real time",
            "real_time_claim": "NOT SUPPORTED on this hardware",
        },

        "reproducibility": {
            "seed_calibration": "none required; calibration is deterministic with no RNG",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "significance_level": SIGNIFICANCE_LEVEL,
            "temporal_checkpoint_sha256": file_sha256(CHECKPOINT),
            "yolo_weights_sha256": file_sha256(YOLO_WEIGHTS),
            "yolo_confidence": 0.25,
            "tracker": {"class": "SimpleTracker", "iou_threshold": 0.3,
                        "max_age": 30, "min_hits": 1,
                        "active_tracks_filter": "last_seen == tracker.frame_idx"},
            "temporal_regime": {"clip_length": 16, "stride": 8, "windows_per_clip": 17},
            "temporal_aggregation": "max, frozen at 0.14",
            "carve_windows_sha256": file_sha256(CARVE_WINDOWS),
            "carve_spatial_sha256": file_sha256(CARVE_SPATIAL),
            "tool_sha256": file_sha256(Path(__file__).resolve()),
            "python": sys.version.split()[0],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    payload = json.dumps(record, indent=2, sort_keys=True)
    record["protocol_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("FROZEN CONFIRMATORY FUSION PROTOCOL")
    print("=" * 74)
    dev = record["development_data"]
    print(f"  development set : {dev['clips_used']} carve clips "
          f"({dev['labels']['Fight']}F/{dev['labels']['NonFight']}N), "
          f"{dev['sources']} sources, {dev['spatially_undefined']} spatially undefined")
    print(f"  alarm budget    : {budget} false positives (incumbent on development)")
    print(f"\n  {'candidate':<20}{'threshold':>14}{'tp':>5}{'fp':>5}{'tn':>5}{'fn':>5}")
    for name, block in record["candidates"].items():
        matrix = block.get("development_matrix") or block["calibration"]["development_matrix"]
        threshold = (block.get("parameters_calibrated") or {}).get("theta_s") \
            or (block.get("parameters_calibrated") or {}).get("theta_or") \
            or (block.get("parameters_calibrated") or {}).get("theta_f") \
            or TEMPORAL_THRESHOLD
        print(f"  {name:<20}{threshold:>14.6f}{matrix['tp']:>5}{matrix['fp']:>5}"
              f"{matrix['tn']:>5}{matrix['fn']:>5}")
    print(f"\n  percentile references stored: temporal n={len(temporal_reference)}, "
          f"spatial n={len(spatial_reference)}")
    print(f"  protocol sha256 : {record['protocol_sha256']}")
    print(f"\nWrote {args.output}")
    print("\nNO PRIMARY DATA WAS READ. Primary evaluation requires explicit authorisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
