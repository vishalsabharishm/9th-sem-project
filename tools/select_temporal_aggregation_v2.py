#!/usr/bin/env python3
"""
tools/select_temporal_aggregation_v2.py

Select a candidate temporal aggregation rule on the CARVE SPLIT ONLY.

THE QUESTION
------------
The frozen sliding-window rule is ``max(window_scores) >= 0.14``. The paired
analysis established that this regime loses Fight recall against whole-clip
inference. Both regimes share the same weights, so the loss is a property of
the AGGREGATION RULE, not of the model. This script asks whether a different
aggregator recovers some of it.

WHY THE DESIGN LOOKS LIKE THIS
------------------------------
The 240 carve clips have already been spent twice: once selecting the
checkpoint's best epoch (monitor roc_auc), and once on a 215-candidate
aggregation sweep that produced ``max >= 0.14``. This is their third use. Two
consequences are designed around rather than ignored:

  1. THRESHOLDS ARE DERIVED, NOT SWEPT. A per-family threshold sweep would add
     hundreds of configurations to a set of 240 clips holding only ~122
     independent source videos. Instead every candidate's threshold is
     calibrated to the incumbent's carve false-positive rate (21/121), so the
     comparison reduces to one well-posed question: at a matched alarm budget,
     which aggregator recovers the most Fight clips? The threshold becomes a
     calibration constant, not a fitted parameter.

  2. THE CANDIDATE SET IS CLOSED. Exactly six configurations, fixed before any
     result was seen, and asserted by a test. max, mean and k-of-n were already
     swept exhaustively by ``select_temporal_aggregation.py``; re-fitting them
     would be re-running a finished experiment on the same data and is not
     done here.

CARVE ONLY -- STRUCTURALLY, NOT BY DISCIPLINE
---------------------------------------------
This script never names, opens or imports the primary evaluation artifacts. A
regression test scans this source for any reference to them. The output JSON is
a VALIDATION CHECKPOINT: it must never carry a primary-test number.

WHAT THIS DOES NOT DO
---------------------
It does not touch the deployed rule. ``max >= 0.14`` remains the frozen
comparator for the eventual primary comparison; the FPR-matched max computed
here is a validation-only control that exists so the candidate families are
compared under the same objective, and the two are labelled distinctly
throughout. Nothing is trained, no threshold in any existing artifact is
altered, and no winner is forced: if no candidate clears the promotion gate the
result records that.

Usage:
    python tools/select_temporal_aggregation_v2.py
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from checkpoint_identity import file_sha256  # noqa: E402
from metric_intervals import source_video_id  # noqa: E402

CARVE_CSV = REPO_ROOT / "temporal_risk" / "carve_window_scores.csv"
DEFAULT_OUTPUT = REPO_ROOT / "temporal_risk" / "frozen_candidate_aggregation.json"

PROTOCOL_VERSION = "v2-carve-only-2026-09"
EXPECTED_CARVE_CLIPS = 240
EXPECTED_WINDOWS_PER_CLIP = 17

# The deployed rule. Frozen: this script never re-tunes it and never replaces
# it. It is the comparator the eventual primary comparison uses.
DEPLOYED_RULE = "max"
DEPLOYED_THRESHOLD = 0.14

# The incumbent's carve confusion, recomputed and asserted at run time rather
# than trusted: tp=85 fp=21 tn=100 fn=34 over 119 Fight / 121 NonFight clips.
INCUMBENT_TP = 85
INCUMBENT_FP = 21
INCUMBENT_TN = 100
INCUMBENT_FN = 34
CARVE_FIGHT_CLIPS = 119
CARVE_NONFIGHT_CLIPS = 121

# Every candidate is calibrated to this alarm budget. Frozen before any
# candidate was evaluated.
TARGET_FALSE_POSITIVES = INCUMBENT_FP
TARGET_FPR = INCUMBENT_FP / CARVE_NONFIGHT_CLIPS  # 21/121 = 0.17355371900826447

# A challenger must recover at least this many Fight clips beyond the
# incumbent's 85 before it is eligible at all. Smaller gaps sit inside carve
# noise given ~122 independent source videos.
PROMOTION_MIN_EXTRA_TP = 5
PROMOTION_MIN_TP = INCUMBENT_TP + PROMOTION_MIN_EXTRA_TP  # 90

CV_FOLDS = 5


class SelectionError(RuntimeError):
    """Raised when the carve data cannot support the frozen protocol."""


# --------------------------------------------------------------------------
# Aggregators -- each maps a PREFIX of window scores to a scalar clip score
# --------------------------------------------------------------------------
#
# Every family is expressed as a scalar score so that "decide" is uniformly
# ``score(prefix) >= threshold``. That is what makes FPR-matching well defined
# for all six configurations, and it is what makes prefix causality checkable
# by one shared argument rather than per family.


def score_max(scores: Sequence[float]) -> float:
    """Top-1. Identical to the deployed rule's statistic."""
    return max(scores)


def make_score_topk(k: int) -> Callable[[Sequence[float]], float]:
    """Mean of the highest ``min(k, m)`` scores in the prefix.

    A one-parameter family that contains both incumbents as endpoints: k=1 is
    exactly max, k=17 is exactly mean. That is the reason to prefer it -- it
    tests directly whether discarding 16 of 17 observations is what costs
    recall, without introducing a family that cannot be compared with either.

    NOT ALARM-MONOTONE, and this matters downstream. The score can FALL as the
    prefix grows: top-2 of [0.9] is 0.9, but top-2 of [0.9, 0.1] is 0.5. So a
    prefix that crosses the threshold can stop crossing it once more windows
    arrive, meaning a streaming alarm could be retracted. That is not a
    causality violation -- the decision at prefix j still reads only windows
    0..j -- but it does mean the first-alarm decision and the final 17-window
    decision can disagree for this family, where for max and consecutive
    persistence they cannot. Measured on carve, not assumed: see
    alarm_retraction_clips in the emitted report.
    """

    def score(scores: Sequence[float]) -> float:
        take = min(k, len(scores))
        return sum(sorted(scores, reverse=True)[:take]) / take

    return score


def make_score_consecutive(n: int) -> Callable[[Sequence[float]], float]:
    """Highest score t for which some run of ``n`` consecutive windows is >= t.

    ``exists a run of n consecutive windows all >= t`` is equivalent to
    ``max over runs of (min within run) >= t``, which turns the persistence
    rule into a scalar score and lets it share the threshold machinery.

    OVERLAPPING WINDOWS -- READ THIS BEFORE INTERPRETING ANY RESULT. Windows
    are 16 frames at stride 8, so two CONSECUTIVE windows share 8 of their 16
    frames. A run of n consecutive windows above threshold is therefore NOT n
    independent pieces of evidence; it covers 8n + 8 frames, not 16n, and half
    of every adjacent pair is literally the same pixels. Persistence here means
    "the signal persisted across overlapping views", which is a weaker claim
    than "n independent detectors agreed".
    """

    def score(scores: Sequence[float]) -> float:
        if len(scores) < n:
            return float("-inf")
        return max(min(scores[i:i + n]) for i in range(len(scores) - n + 1))

    return score


# The closed candidate set. Six configurations, fixed in advance. A test
# asserts this exact set; nothing may be appended because early results
# disappoint.
CANDIDATE_DEFINITIONS: List[dict] = [
    {
        "name": "max_fpr_matched",
        "alarm_monotone": True,
        "family": "max",
        "params": {},
        "complexity": 0,
        "scorer": score_max,
        "role": "validation-only control",
        "description": (
            "max over the prefix, threshold calibrated to the incumbent carve "
            "FPR. NOT the deployed rule: it exists so the candidate families "
            "are compared under the same FPR-matching objective the deployed "
            "rule was never selected under"
        ),
    },
    {
        "name": "topk_2",
        "alarm_monotone": False,
        "family": "topk_mean",
        "params": {"k": 2},
        "complexity": 1,
        "scorer": make_score_topk(2),
        "role": "candidate",
        "description": "mean of the highest min(2, m) scores in the prefix",
    },
    {
        "name": "topk_3",
        "alarm_monotone": False,
        "family": "topk_mean",
        "params": {"k": 3},
        "complexity": 2,
        "scorer": make_score_topk(3),
        "role": "candidate",
        "description": "mean of the highest min(3, m) scores in the prefix",
    },
    {
        "name": "topk_5",
        "alarm_monotone": False,
        "family": "topk_mean",
        "params": {"k": 5},
        "complexity": 3,
        "scorer": make_score_topk(5),
        "role": "candidate",
        "description": "mean of the highest min(5, m) scores in the prefix",
    },
    {
        "name": "consecutive_2",
        "alarm_monotone": True,
        "family": "consecutive_persistence",
        "params": {"n": 2},
        "complexity": 4,
        "scorer": make_score_consecutive(2),
        "role": "candidate",
        "description": (
            "some run of 2 consecutive completed windows at or above threshold; "
            "those 2 windows overlap by 8 of 16 frames"
        ),
    },
    {
        "name": "consecutive_3",
        "alarm_monotone": True,
        "family": "consecutive_persistence",
        "params": {"n": 3},
        "complexity": 5,
        "scorer": make_score_consecutive(3),
        "role": "candidate",
        "description": (
            "some run of 3 consecutive completed windows at or above threshold; "
            "adjacent windows overlap by 8 of 16 frames"
        ),
    },
]

CANDIDATE_NAMES = tuple(entry["name"] for entry in CANDIDATE_DEFINITIONS)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_carve(path: Path = CARVE_CSV) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
    """Return per-clip window scores in window order, and per-clip labels."""
    if not path.is_file():
        raise SelectionError(f"{path} not found; the carve window scores are required.")

    rows: Dict[str, List[Tuple[int, float]]] = {}
    labels: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "carve":
                raise SelectionError(
                    f"{path} contains a row from split {row['split']!r}; this "
                    "script reads the carve split only."
                )
            clip = row["clip"]
            rows.setdefault(clip, []).append(
                (int(row["window_index"]), float(row["fight_probability"]))
            )
            existing = labels.setdefault(clip, row["true_label"])
            if existing != row["true_label"]:
                raise SelectionError(f"{clip} carries conflicting labels in {path}.")

    scores = {
        clip: [value for _, value in sorted(entries)] for clip, entries in rows.items()
    }

    if len(scores) != EXPECTED_CARVE_CLIPS:
        raise SelectionError(
            f"expected {EXPECTED_CARVE_CLIPS} carve clips, found {len(scores)}."
        )
    for clip, values in scores.items():
        if len(values) != EXPECTED_WINDOWS_PER_CLIP:
            raise SelectionError(
                f"{clip} has {len(values)} windows, expected {EXPECTED_WINDOWS_PER_CLIP}."
            )
    fights = sum(1 for label in labels.values() if label == "Fight")
    if fights != CARVE_FIGHT_CLIPS or len(labels) - fights != CARVE_NONFIGHT_CLIPS:
        raise SelectionError(
            f"expected {CARVE_FIGHT_CLIPS} Fight / {CARVE_NONFIGHT_CLIPS} NonFight "
            f"carve clips, found {fights} / {len(labels) - fights}."
        )
    return scores, labels


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def confusion(
    clips: Sequence[str],
    labels: Dict[str, str],
    decisions: Dict[str, bool],
) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for clip in clips:
        positive = labels[clip] == "Fight"
        fired = decisions[clip]
        if positive and fired:
            tp += 1
        elif positive:
            fn += 1
        elif fired:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metrics_from(matrix: Dict[str, int]) -> Dict[str, Optional[float]]:
    tp, fp, tn, fn = matrix["tp"], matrix["fp"], matrix["tn"], matrix["fn"]
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall)
        else None
    )
    return {
        **matrix,
        "accuracy": (tp + tn) / total if total else None,
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "f1": f1,
    }


# --------------------------------------------------------------------------
# Threshold derivation -- calibration, not tuning
# --------------------------------------------------------------------------


def derive_threshold(
    clips: Sequence[str],
    labels: Dict[str, str],
    clip_scores: Dict[str, float],
    target_false_positives: int,
) -> dict:
    """Calibrate a threshold to a fixed false-positive budget on carve.

    The grid is every distinct clip score present, because the confusion matrix
    only changes at those points -- so this is an exhaustive search of the
    genuinely distinct thresholds, not a sampled sweep, and it cannot miss the
    optimum by grid resolution.

    Selection among grid points, in order:
      1. false-positive count closest to the budget (the calibration);
      2. then the most true positives (this is "recall at matched FPR", the
         one pre-specified criterion);
      3. then the highest threshold, purely so the choice is deterministic
         when several thresholds give the identical confusion matrix.
    """
    finite = [clip_scores[clip] for clip in clips if clip_scores[clip] != float("-inf")]
    if not finite:
        raise SelectionError("no finite clip scores; cannot calibrate a threshold.")

    best = None
    for threshold in sorted(set(finite)):
        decisions = {clip: clip_scores[clip] >= threshold for clip in clips}
        matrix = confusion(clips, labels, decisions)
        key = (
            abs(matrix["fp"] - target_false_positives),
            -matrix["tp"],
            -threshold,
        )
        if best is None or key < best[0]:
            best = (key, threshold, matrix)

    _, threshold, matrix = best
    return {
        "threshold": threshold,
        "matrix": matrix,
        "target_false_positives": target_false_positives,
        "target_fpr": target_false_positives / (matrix["fp"] + matrix["tn"]),
        "achieved_false_positives": matrix["fp"],
        "achieved_fpr": matrix["fp"] / (matrix["fp"] + matrix["tn"]),
        "exact_fpr_match": matrix["fp"] == target_false_positives,
        "grid": "every distinct clip score (exhaustive over distinct confusions)",
    }


def evaluate_candidate(
    definition: dict,
    clips: Sequence[str],
    labels: Dict[str, str],
    scores: Dict[str, List[float]],
    target_false_positives: int = TARGET_FALSE_POSITIVES,
) -> dict:
    """Calibrate one candidate's threshold and score it on the given clips."""
    clip_scores = {clip: definition["scorer"](scores[clip]) for clip in clips}
    derived = derive_threshold(clips, labels, clip_scores, target_false_positives)
    return {
        "name": definition["name"],
        "family": definition["family"],
        "params": dict(definition["params"]),
        "role": definition["role"],
        "alarm_monotone": definition["alarm_monotone"],
        "description": definition["description"],
        "threshold": derived["threshold"],
        "target_false_positives": derived["target_false_positives"],
        "target_fpr": TARGET_FPR,
        "achieved_false_positives": derived["achieved_false_positives"],
        "achieved_fpr": derived["achieved_fpr"],
        "exact_fpr_match": derived["exact_fpr_match"],
        "metrics": metrics_from(derived["matrix"]),
    }


def alarm_retraction(
    definition: dict,
    clips: Sequence[str],
    scores: Dict[str, List[float]],
    threshold: float,
) -> dict:
    """Count clips where a streaming alarm would later be withdrawn.

    Measured, not asserted. A candidate whose score can fall as windows arrive
    may cross the threshold on an early prefix and drop back below it by the
    seventeenth window, so its first-alarm decision and its final clip-level
    decision disagree. This is not a causality failure -- each prefix decision
    still reads only its own prefix -- but it is a property the eventual
    early-detection protocol has to state, because "time to first alarm" and
    "did the clip alarm" stop being the same question.
    """
    retracting = []
    for clip in clips:
        series = scores[clip]
        fired = False
        for index in range(len(series)):
            now = definition["scorer"](series[:index + 1]) >= threshold
            if fired and not now:
                retracting.append(clip)
                break
            fired = fired or now
    return {
        "declared_alarm_monotone": definition["alarm_monotone"],
        "alarm_retraction_clips": len(retracting),
        "observed_alarm_monotone": not retracting,
        "note": (
            "clips where the rule crosses the threshold on some prefix and is "
            "back below it by the final window; measured on carve at this "
            "candidate's calibrated threshold"
        ),
    }


# --------------------------------------------------------------------------
# Grouped 5-fold cross-validation -- descriptive only
# --------------------------------------------------------------------------


def build_source_folds(clips: Sequence[str], folds: int = CV_FOLDS) -> Dict[str, int]:
    """Assign whole SOURCE VIDEOS to folds, deterministically and without RNG.

    Clips cut from one source video are near-duplicates of each other, and 18
    carve source videos contribute clips under both labels, so a clip-level
    split would put near-identical clips on both sides of a fold boundary and
    report an optimistic number. Sources move as units.

    Largest source first onto the currently smallest fold -- a deterministic
    greedy balance, since carve source sizes run from 1 to 16 clips and
    round-robin over sorted names would leave folds badly uneven.
    """
    members: Dict[str, List[str]] = {}
    for clip in clips:
        members.setdefault(source_video_id(clip), []).append(clip)

    sizes = [(-len(group), source) for source, group in members.items()]
    sizes.sort()

    counts = [0] * folds
    assignment: Dict[str, int] = {}
    for negative_size, source in sizes:
        fold = min(range(folds), key=lambda index: (counts[index], index))
        assignment[source] = fold
        counts[fold] += -negative_size
    return assignment


def grouped_cross_validation(
    definition: dict,
    clips: Sequence[str],
    labels: Dict[str, str],
    scores: Dict[str, List[float]],
    folds: int = CV_FOLDS,
) -> dict:
    """Re-derive the threshold on each training fold; score the held-out fold.

    Deriving the threshold once on all 240 clips and then reporting per-fold
    numbers would report the training fit, not a generalisation estimate. The
    calibration is redone inside each fold so the number means something.

    DESCRIPTIVE ONLY. Nothing here feeds back into the candidate set, the
    threshold, or the promotion decision -- the main selection is the full-carve
    result. This exists to show whether a candidate's advantage survives being
    calibrated on different clips.
    """
    assignment = build_source_folds(clips, folds)
    fold_of_clip = {clip: assignment[source_video_id(clip)] for clip in clips}

    results = []
    for fold in range(folds):
        held_out = [clip for clip in clips if fold_of_clip[clip] == fold]
        training = [clip for clip in clips if fold_of_clip[clip] != fold]
        if not held_out or not training:
            raise SelectionError(f"fold {fold} is empty; cannot cross-validate.")

        train_nonfight = sum(1 for clip in training if labels[clip] == "NonFight")
        budget = round(TARGET_FPR * train_nonfight)
        train_scores = {clip: definition["scorer"](scores[clip]) for clip in training}
        derived = derive_threshold(training, labels, train_scores, budget)

        threshold = derived["threshold"]
        decisions = {
            clip: definition["scorer"](scores[clip]) >= threshold for clip in held_out
        }
        matrix = confusion(held_out, labels, decisions)
        results.append(
            {
                "fold": fold,
                "clips": len(held_out),
                "sources": len({source_video_id(clip) for clip in held_out}),
                "threshold_from_training_folds": threshold,
                "training_false_positive_budget": budget,
                **metrics_from(matrix),
            }
        )

    def summarise(field: str) -> Dict[str, Optional[float]]:
        values = [row[field] for row in results if row[field] is not None]
        if not values:
            return {"mean": None, "std": None, "folds_defined": 0}
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {"mean": mean, "std": variance ** 0.5, "folds_defined": len(values)}

    return {
        "note": (
            "descriptive only; the threshold is re-derived on each training fold "
            "and never influences the main carve selection or the promotion gate"
        ),
        "folds": results,
        "recall": summarise("recall"),
        "fpr": summarise("fpr"),
        "precision": summarise("precision"),
        "f1": summarise("f1"),
    }


def fold_integrity(clips: Sequence[str], labels: Dict[str, str], folds: int = CV_FOLDS) -> dict:
    """Prove the folds partition the clips and never split a source video."""
    assignment = build_source_folds(clips, folds)
    per_fold: Dict[int, List[str]] = {index: [] for index in range(folds)}
    for clip in clips:
        per_fold[assignment[source_video_id(clip)]].append(clip)

    seen = [clip for group in per_fold.values() for clip in group]
    sources_per_fold = {
        fold: {source_video_id(clip) for clip in group} for fold, group in per_fold.items()
    }
    shared = 0
    for left in range(folds):
        for right in range(left + 1, folds):
            shared += len(sources_per_fold[left] & sources_per_fold[right])

    return {
        "folds": folds,
        "grouping_unit": "source video",
        "grouping": "inferred from the '<source_id>_<index>.avi' filename convention",
        "grouping_helper": "src/metric_intervals.py:source_video_id",
        "sources": len({source_video_id(clip) for clip in clips}),
        "sources_shared_between_folds": shared,
        "clips_assigned": len(seen),
        "clips_assigned_exactly_once": len(seen) == len(set(seen)) == len(clips),
        "fold_sizes": {fold: len(group) for fold, group in per_fold.items()},
        "fold_fight_clips": {
            fold: sum(1 for clip in group if labels[clip] == "Fight")
            for fold, group in per_fold.items()
        },
    }


# --------------------------------------------------------------------------
# Promotion
# --------------------------------------------------------------------------


def promote(evaluations: List[dict]) -> dict:
    """Apply the frozen promotion gate. Never forces a winner."""
    complexity = {entry["name"]: entry["complexity"] for entry in CANDIDATE_DEFINITIONS}

    eligible = [
        row for row in evaluations
        if row["role"] == "candidate" and row["metrics"]["tp"] >= PROMOTION_MIN_TP
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -row["metrics"]["tp"],
            row["metrics"]["fp"],
            complexity[row["name"]],
            row["name"],
        ),
    )

    blocked = [
        {
            "name": row["name"],
            "tp": row["metrics"]["tp"],
            "extra_tp_over_incumbent": row["metrics"]["tp"] - INCUMBENT_TP,
            "required_extra_tp": PROMOTION_MIN_EXTRA_TP,
        }
        for row in evaluations
        if row["role"] == "candidate" and row["metrics"]["tp"] < PROMOTION_MIN_TP
    ]

    gate = {
        "criterion": "carve recall at matched false-positive budget",
        "incumbent_tp": INCUMBENT_TP,
        "incumbent_recall": INCUMBENT_TP / CARVE_FIGHT_CLIPS,
        "required_extra_true_positives": PROMOTION_MIN_EXTRA_TP,
        "required_tp": PROMOTION_MIN_TP,
        "tie_breakers": [
            "higher recall (more true positives)",
            "fewer false positives",
            "lower parameter complexity",
            "retain the incumbent",
        ],
        "candidates_evaluated": sum(1 for row in evaluations if row["role"] == "candidate"),
        "candidates_clearing_gate": len(eligible),
        "candidates_blocked_by_gate": blocked,
    }

    if not ranked:
        return {
            "promoted": False,
            "selected": None,
            "gate": gate,
            "decision": "NO CANDIDATE PROMOTED",
            "rationale": (
                "no candidate reached the pre-registered gate of "
                f"{PROMOTION_MIN_TP} carve true positives "
                f"({PROMOTION_MIN_EXTRA_TP} more Fight clips than the incumbent's "
                f"{INCUMBENT_TP}) at the matched false-positive budget. The gate "
                "was fixed before any candidate was evaluated and is not relaxed "
                "to produce a winner. The deployed rule "
                f"{DEPLOYED_RULE} >= {DEPLOYED_THRESHOLD} therefore stands unchanged, "
                "and no primary evaluation is licensed by this run."
            ),
        }

    winner = ranked[0]
    return {
        "promoted": True,
        "selected": {
            "name": winner["name"],
            "family": winner["family"],
            "params": winner["params"],
            "threshold": winner["threshold"],
            "metrics": winner["metrics"],
        },
        "gate": gate,
        "decision": f"PROMOTED {winner['name']}",
        "rationale": (
            f"{winner['name']} reached {winner['metrics']['tp']} carve true positives "
            f"at {winner['achieved_false_positives']} false positives, clearing the "
            f"pre-registered gate of {PROMOTION_MIN_TP}. Selected on carve recall at a "
            "matched false-positive budget, the single pre-specified criterion; early "
            "detection played no part in selection."
        ),
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def build_report(carve_csv: Path = CARVE_CSV) -> dict:
    scores, labels = load_carve(carve_csv)
    clips = sorted(scores)

    incumbent_decisions = {
        clip: max(scores[clip]) >= DEPLOYED_THRESHOLD for clip in clips
    }
    incumbent_matrix = confusion(clips, labels, incumbent_decisions)
    expected = {"tp": INCUMBENT_TP, "fp": INCUMBENT_FP, "tn": INCUMBENT_TN, "fn": INCUMBENT_FN}
    if incumbent_matrix != expected:
        raise SelectionError(
            f"the deployed rule {DEPLOYED_RULE} >= {DEPLOYED_THRESHOLD} reproduces "
            f"{incumbent_matrix} on carve, expected {expected}. Refusing to select "
            "against an incumbent that does not reproduce."
        )

    evaluations = []
    for definition in CANDIDATE_DEFINITIONS:
        row = evaluate_candidate(definition, clips, labels, scores)
        row["streaming_behaviour"] = alarm_retraction(
            definition, clips, scores, row["threshold"]
        )
        evaluations.append(row)
    cross_validation = {
        definition["name"]: grouped_cross_validation(definition, clips, labels, scores)
        for definition in CANDIDATE_DEFINITIONS
    }
    outcome = promote(evaluations)

    return {
        "tool": "select_temporal_aggregation_v2",
        "protocol_version": PROTOCOL_VERSION,
        "artifact_kind": "carve-validation checkpoint",
        "contains_primary_results": False,
        "scope": (
            "carve split only (240 clips). The 394-clip primary evaluation is not "
            "read, referenced or evaluated anywhere in this run."
        ),
        "dataset": {
            "identifier": "temporal_risk/carve_window_scores.csv",
            "sha256": file_sha256(carve_csv),
            "split": "carve",
            "clips": len(clips),
            "fight_clips": CARVE_FIGHT_CLIPS,
            "nonfight_clips": CARVE_NONFIGHT_CLIPS,
            "windows_per_clip": EXPECTED_WINDOWS_PER_CLIP,
            "windows": len(clips) * EXPECTED_WINDOWS_PER_CLIP,
        },
        "source_grouping": fold_integrity(clips, labels),
        "window_geometry": {
            "clip_length": 16,
            "stride": 8,
            "windows_per_clip": EXPECTED_WINDOWS_PER_CLIP,
            "adjacent_window_frame_overlap": 8,
            "independence_caveat": (
                "consecutive windows share 8 of their 16 frames, so a run of n "
                "consecutive windows above threshold covers 8n + 8 frames and is "
                "NOT n independent observations"
            ),
        },
        "deployed_incumbent": {
            "rule": DEPLOYED_RULE,
            "threshold": DEPLOYED_THRESHOLD,
            "status": "FROZEN -- not re-tuned, not replaced by this script",
            "source": "temporal_risk/frozen_aggregation.json",
            "carve_metrics": metrics_from(incumbent_matrix),
            "reproduced_from_raw_scores": True,
        },
        "threshold_calibration": {
            "method": "false-positive budget matched to the deployed incumbent on carve",
            "target_false_positives": TARGET_FALSE_POSITIVES,
            "target_fpr": TARGET_FPR,
            "rationale": (
                "thresholds are derived, not swept: a per-family sweep would add "
                "hundreds of configurations to 240 clips holding ~122 independent "
                "source videos. Calibrating to a fixed alarm budget turns the "
                "threshold into a constant and reduces the comparison to recall at "
                "matched false-alarm rate"
            ),
        },
        "candidate_set": {
            "closed": True,
            "size": len(CANDIDATE_DEFINITIONS),
            "names": list(CANDIDATE_NAMES),
            "fixed_before_results_were_seen": True,
            "excluded_by_protocol": [
                "EMA / temporal smoothing (adds a free alpha on a sequence already "
                "smoothed by 50% window overlap)",
                "mean re-tuning (all 99 thresholds already swept by "
                "select_temporal_aggregation.py)",
                "k-of-n re-tuning (all 17 k values already swept)",
                "any k or n outside {2, 3, 5} and {2, 3}",
            ],
        },
        "candidates": evaluations,
        "grouped_cross_validation": cross_validation,
        "promotion": outcome,
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "tool_sha256": file_sha256(Path(__file__).resolve()),
            "python": sys.version.split()[0],
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--carve", type=Path, default=CARVE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    try:
        report = build_report(args.carve)
    except SelectionError as error:
        print(f"SELECTION REFUSED: {error}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    incumbent = report["deployed_incumbent"]["carve_metrics"]
    print("TEMPORAL AGGREGATION SELECTION v2 -- CARVE SPLIT ONLY")
    print("=" * 78)
    print(f"  carve            : {report['dataset']['clips']} clips, "
          f"{report['source_grouping']['sources']} source videos")
    print(f"  deployed (FROZEN): {DEPLOYED_RULE} >= {DEPLOYED_THRESHOLD}  "
          f"tp={incumbent['tp']} fp={incumbent['fp']} "
          f"tn={incumbent['tn']} fn={incumbent['fn']}  recall={incumbent['recall']:.4f}")
    print(f"  budget           : {TARGET_FALSE_POSITIVES} false positives "
          f"(FPR {TARGET_FPR:.6f})")
    print(f"  promotion gate   : tp >= {PROMOTION_MIN_TP}")
    print()
    print(f"  {'candidate':18s} {'thresh':>9s} {'tp':>4s} {'fp':>4s} {'tn':>4s} {'fn':>4s} "
          f"{'recall':>8s} {'fpr':>8s}  {'CV recall':>16s}  role")
    for row in report["candidates"]:
        metric = row["metrics"]
        summary = report["grouped_cross_validation"][row["name"]]["recall"]
        cv = (f"{summary['mean']:.4f}+-{summary['std']:.4f}"
              if summary["mean"] is not None else "n/a")
        print(f"  {row['name']:18s} {row['threshold']:9.6f} {metric['tp']:4d} "
              f"{metric['fp']:4d} {metric['tn']:4d} {metric['fn']:4d} "
              f"{metric['recall']:8.4f} {metric['fpr']:8.4f}  {cv:>16s}  {row['role']}")
    print()
    print(f"  DECISION: {report['promotion']['decision']}")
    print(f"  {report['promotion']['rationale']}")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
