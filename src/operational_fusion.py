"""Operational evidence-fusion layer for live video analysis.

WHY THIS EXISTS
---------------
The confirmatory research evaluation computed the fusion candidates offline,
inside ``tools/run_confirmatory_primary_evaluation.py``, over clips that were
already on disk. Nothing in the *operational* pipeline could produce a fusion
decision for a video arriving now. This module closes that gap, and only that
gap: it computes the frozen spatial feature from a running analysis and applies
the frozen decision rules to it.

WHAT IS FROZEN AND NOT REDEFINED HERE
-------------------------------------
Every threshold, weight and reference distribution is READ from
``temporal_risk/frozen_confirmatory_fusion_protocol.json`` at call time and
verified against the SHA-256 recorded when the protocol was frozen. Nothing is
hard-coded, so this file cannot drift from the protocol or quietly become a
second place an operating point lives. If the protocol file is edited, every
call raises rather than silently scoring under new parameters.

WHAT THE RESULT DOES AND DOES NOT MEAN
--------------------------------------
The system implements a configured evidence-fusion layer combining temporal and
interpretable spatial evidence. In the frozen confirmatory evaluation, fusion
did not produce a statistically significant improvement over the temporal-only
baseline. That sentence travels with every result this module returns, in the
``interpretation`` field, because a fusion decision presented without it invites
exactly the conclusion the evaluation failed to support.

B0 (temporal-only) remains the incumbent. F2 is reported alongside it, not in
place of it.

CAUSALITY
---------
The spatial feature is a WHOLE-CLIP aggregate: its denominator is a mean over
every frame containing a person, so it is not defined until the video ends.
Any fusion decision is therefore an offline, whole-video judgement and is
labelled ``offline whole-video evidence fusion``. The temporal-only alarm can
still fire mid-video, because the frozen aggregation rule is monotone in the
prefix; fusion cannot, and this module never pretends otherwise.

ONE IMPLEMENTATION DETAIL WORTH KNOWING
---------------------------------------
``SpatialFeatureAccumulator`` reproduces the frozen implementation's call
ordering exactly -- ``get_previous_bbox`` BEFORE ``update_history`` -- because
that ordering is part of what produced the locked artifact. See the note on
``_observe_speeds``; it is not what the protocol prose describes, and it is
deliberately preserved rather than corrected. ``tests/test_operational_fusion.py``
asserts byte-equality against the research tool's own function on real video, so
the two cannot drift.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from behavior_analyzer import BehaviorAnalyzer
    from tracker import build_tracking_snapshot
    from event_rules import (
        DEFAULT_MOVEMENT_THRESHOLD_PIXELS,
        DEFAULT_STATIONARY_FRAME_THRESHOLD,
    )
except ImportError:  # pragma: no cover - package-style execution
    from src.behavior_analyzer import BehaviorAnalyzer  # type: ignore
    from src.tracker import build_tracking_snapshot  # type: ignore
    from src.event_rules import (  # type: ignore
        DEFAULT_MOVEMENT_THRESHOLD_PIXELS,
        DEFAULT_STATIONARY_FRAME_THRESHOLD,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = REPO_ROOT / "temporal_risk" / "frozen_confirmatory_fusion_protocol.json"

# The hash recorded when the protocol was frozen. Identical to the constant in
# tools/run_confirmatory_primary_evaluation.py and to
# FINAL_experiment_lock_record.json -> hashes.protocol_sha256_internal.
EXPECTED_PROTOCOL_SHA = "ff2d78c5ae064e16b1144b74193b40eb66d71f63069c8594faf13238fb6384cd"

SPATIAL_FEATURE_NAME = "speed_normalised_by_group_diagonal"

# COCO person class, as the frozen spatial scorer filters on.
PERSON_CLASS_ID = 0

FUSION_MODE = "offline whole-video evidence fusion"

# Travels with every result. Deliberately phrased so it cannot be quoted as a
# claim of improvement.
FUSION_INTERPRETATION = (
    "The system implements a configured evidence-fusion layer combining "
    "temporal and interpretable spatial evidence. In the frozen confirmatory "
    "evaluation, fusion did not produce a statistically significant "
    "improvement over the temporal-only baseline."
)

FUSION_NOT_CLAIMED = (
    "Not claimed: that fusion is superior to the temporal-only baseline, that "
    "it improves accuracy, or that it is statistically better. B0 "
    "(temporal-only) remains the incumbent decision."
)

FUSION_CAUSALITY_NOTE = (
    "The spatial feature is a whole-video aggregate, so the fusion decision is "
    "available only after the whole video has been processed. It is not a "
    "causal mid-video alarm. The temporal-only signal may fire earlier."
)

# The frozen protocol's stated handling of a clip whose spatial feature is
# undefined: it abstains, which the rules treat as spatial-negative. The
# measured VALUE stays None everywhere -- only the F2 arithmetic substitutes a
# zero percentile, exactly as the confirmatory evaluation did.
SPATIAL_UNDEFINED_NOTE = (
    "The spatial feature is undefined for this video (no usable track pair, or "
    "no person was ever detected). Per the frozen protocol an undefined "
    "spatial score abstains and is treated as spatial-negative; the value "
    "itself is reported as null and is never imputed as zero."
)


class FusionProtocolError(RuntimeError):
    """The frozen protocol is missing, edited, or not the one that was locked."""


# ---------------------------------------------------------------------------
# The frozen spatial feature, measured from a live run
# ---------------------------------------------------------------------------


@dataclass
class SpatialFeatureAccumulator:
    """Accumulates the frozen spatial feature over a video, frame by frame.

    ``mean(per-track centroid displacement) / mean(person-group diagonal)`` --
    a RATIO OF MEANS, not a mean of ratios, as the protocol specifies.

    Feed it one call per decoded frame, with the person tracks for that frame,
    in decode order. It holds its own BehaviorAnalyzer so that driving it
    cannot perturb the rule engine's analyzer, which maintains separate state
    for a different purpose.
    """

    frames: int = 0
    zero_person_frames: int = 0
    _speeds: List[float] = field(default_factory=list)
    _diagonals: List[float] = field(default_factory=list)
    _counts: List[int] = field(default_factory=list)
    _behaviour: BehaviorAnalyzer = field(default_factory=BehaviorAnalyzer)

    def observe(self, frame_idx: int, tracks: Sequence[Any]) -> None:
        """Record one frame.

        ``tracks`` is every ACTIVE track for the frame, of any class -- not
        only the person tracks. Persons are filtered here for the feature
        itself, but history is updated from all of them, because that is what
        the frozen implementation does: a track whose class changes between
        frames would otherwise carry a different history and yield a different
        displacement.
        """
        self.frames += 1
        all_tracks = list(tracks)
        persons = [t for t in all_tracks if int(t.class_id) == PERSON_CLASS_ID]
        self._counts.append(len(persons))
        if not persons:
            self.zero_person_frames += 1

        self._observe_speeds(persons)

        if persons:
            xs = [float(t.bbox[0]) for t in persons] + [float(t.bbox[2]) for t in persons]
            ys = [float(t.bbox[1]) for t in persons] + [float(t.bbox[3]) for t in persons]
            self._diagonals.append(float(np.hypot(max(xs) - min(xs), max(ys) - min(ys))))

        # Called LAST, and over ALL tracks, matching the frozen implementation.
        # See _observe_speeds for why the ordering matters.
        snapshot = build_tracking_snapshot(frame_idx, all_tracks)
        self._behaviour.update_history(
            snapshot, DEFAULT_MOVEMENT_THRESHOLD_PIXELS, DEFAULT_STATIONARY_FRAME_THRESHOLD
        )

    def _observe_speeds(self, persons: Sequence[Any]) -> None:
        """Displacement samples, collected BEFORE this frame enters history.

        This ordering is load-bearing and is preserved deliberately.
        ``get_previous_bbox`` returns ``history[-2]``. Called here -- before
        ``update_history`` has appended the current frame -- ``history[-1]`` is
        the previous frame and ``history[-2]`` is the one before that, so each
        sample spans TWO frames, not one.

        The protocol prose says "between consecutive frames of the same track",
        which does not describe this. The implementation is nevertheless what
        produced the locked artifact: the same computation generated the
        development reference distribution, theta_s, theta_or and theta_f, and
        the primary spatial scores, so every threshold and percentile is
        expressed in these units and the result is internally consistent.
        Changing it here would silently put live videos on a different scale
        from the frozen thresholds. It is reported as a prose/implementation
        discrepancy and left exactly as it is.
        """
        for track in persons:
            previous = self._behaviour.get_previous_bbox(track.id)
            if previous is not None:
                self._speeds.append(self._behaviour._movement_pixels(previous, track.bbox))

    @property
    def speed_mean(self) -> Optional[float]:
        return float(np.mean(self._speeds)) if self._speeds else None

    @property
    def diagonal_mean(self) -> Optional[float]:
        return float(np.mean(self._diagonals)) if self._diagonals else None

    @property
    def person_count_mean(self) -> Optional[float]:
        return float(np.mean(self._counts)) if self._counts else None

    @property
    def score(self) -> Optional[float]:
        """The feature, or None when it is genuinely undefined.

        None is a real answer and is never replaced by zero: a video in which
        nobody was detected has no motion-relative-to-size, which is not the
        same statement as "no motion".
        """
        speed, diagonal = self.speed_mean, self.diagonal_mean
        if speed is None or not diagonal:
            return None
        return speed / diagonal

    def as_dict(self) -> dict:
        return {
            "feature": SPATIAL_FEATURE_NAME,
            "spatial_score": self.score,
            "defined": self.score is not None,
            "speed_mean_pixels": self.speed_mean,
            "group_diagonal_mean": self.diagonal_mean,
            "person_count_mean": self.person_count_mean,
            "frames": self.frames,
            "zero_person_frames": self.zero_person_frames,
            "displacement_samples": len(self._speeds),
            "form": "RATIO OF MEANS, not mean of ratios",
            "provenance": "measured_from_this_video",
            "note": None if self.score is not None else SPATIAL_UNDEFINED_NOTE,
        }


# ---------------------------------------------------------------------------
# The frozen protocol
# ---------------------------------------------------------------------------


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    """Read the frozen protocol, refusing anything that is not the locked one.

    Both checks the confirmatory tool applies are applied here: the recorded
    field, and a recomputation of the body hash so an edit anywhere in the
    artifact is caught rather than only a mismatched field.
    """
    if not Path(path).is_file():
        raise FusionProtocolError(
            f"the frozen fusion protocol is missing: {path}. Fusion cannot be "
            "computed without it, and no default parameters exist."
        )
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = protocol.get("protocol_sha256")
    if recorded != EXPECTED_PROTOCOL_SHA:
        raise FusionProtocolError(
            f"protocol_sha256 is {recorded}, expected {EXPECTED_PROTOCOL_SHA}. "
            "Refusing to fuse under an unlocked protocol."
        )
    copy = dict(protocol)
    copy.pop("protocol_sha256")
    recomputed = hashlib.sha256(
        json.dumps(copy, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if recomputed != EXPECTED_PROTOCOL_SHA:
        raise FusionProtocolError(
            f"the protocol body hashes to {recomputed}, not {EXPECTED_PROTOCOL_SHA}. "
            "It has been edited since it was frozen."
        )
    return protocol


def percentile_of(value: Optional[float], reference: Sequence[float]) -> Optional[float]:
    """Empirical percentile against the stored development distribution.

    Identical to the confirmatory tool's function, including its tie handling:
    the proportion of reference values strictly below ``value``.
    """
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


@dataclass
class FrozenFusionProtocol:
    """The frozen decision rules, loaded rather than restated."""

    temporal_threshold: float
    theta_s: float
    theta_or: float
    theta_f: float
    weight: float
    temporal_reference: List[float]
    spatial_reference: List[float]
    protocol_sha256: str
    protocol_version: str

    @classmethod
    def load(cls, path: Path = PROTOCOL_PATH) -> "FrozenFusionProtocol":
        protocol = load_protocol(path)
        candidates = protocol["candidates"]
        normalisation = protocol["normalisation"]
        return cls(
            temporal_threshold=candidates["B0_temporal_only"]["parameters_frozen"]["threshold"],
            theta_s=candidates["B1_spatial_only"]["parameters_calibrated"]["theta_s"],
            theta_or=candidates["F1_or_gate"]["parameters_calibrated"]["theta_or"],
            theta_f=candidates["F2_rank_sum"]["parameters_calibrated"]["theta_f"],
            weight=candidates["F2_rank_sum"]["parameters_frozen"]["weight"],
            temporal_reference=list(normalisation["temporal_reference_values"]),
            spatial_reference=list(normalisation["spatial_reference_values"]),
            protocol_sha256=protocol["protocol_sha256"],
            protocol_version=protocol["protocol_version"],
        )

    def evaluate(
        self, temporal_max: Optional[float], spatial: Optional[float]
    ) -> Dict[str, Any]:
        """Apply every frozen candidate to one video's evidence.

        ``temporal_max`` is the maximum completed-window probability. With no
        completed window there is no temporal evidence and no candidate can be
        evaluated, which is reported rather than defaulted.
        """
        if temporal_max is None:
            return {
                "available": False,
                "reason": "no_temporal_evidence",
                "detail": (
                    "No temporal window completed for this video, so no "
                    "candidate can be evaluated. A window needs 16 frames."
                ),
                "mode": FUSION_MODE,
                "interpretation": FUSION_INTERPRETATION,
                "not_claimed": FUSION_NOT_CLAIMED,
            }

        temporal_pct = percentile_of(temporal_max, self.temporal_reference)
        spatial_pct = percentile_of(spatial, self.spatial_reference)

        # The frozen abstention rule: an undefined spatial score contributes a
        # zero percentile to F2 and cannot satisfy B1 or the F1 or-arm. The
        # measured value stays None; only this arithmetic substitutes.
        f2_score = self.weight * temporal_pct + (1 - self.weight) * (
            spatial_pct if spatial_pct is not None else 0.0
        )

        b0 = temporal_max >= self.temporal_threshold
        b1 = spatial is not None and spatial >= self.theta_s
        f1 = b0 or (spatial is not None and spatial >= self.theta_or)
        f2 = f2_score >= self.theta_f

        return {
            "available": True,
            "mode": FUSION_MODE,
            "protocol_version": self.protocol_version,
            "protocol_sha256": self.protocol_sha256,
            "inputs": {
                "temporal_max": temporal_max,
                "spatial_score": spatial,
                "spatial_defined": spatial is not None,
                "temporal_percentile": temporal_pct,
                "spatial_percentile": spatial_pct,
                "spatial_abstained": spatial_pct is None,
            },
            "parameters": {
                "temporal_threshold": self.temporal_threshold,
                "theta_s": self.theta_s,
                "theta_or": self.theta_or,
                "theta_f": self.theta_f,
                "weight": self.weight,
                "status": "frozen; read from the locked protocol, not tuned here",
            },
            "candidates": {
                "B0_temporal_only": {
                    "rule": "max(window_probabilities) >= 0.14",
                    "decision": "Fight" if b0 else "NonFight",
                    "fired": b0,
                    "role": "frozen incumbent -- this is the system's decision",
                },
                "B1_spatial_only": {
                    "rule": "spatial >= theta_s",
                    "decision": "Fight" if b1 else "NonFight",
                    "fired": b1,
                    "role": "baseline, reported for comparison only",
                },
                "F1_or_gate": {
                    "rule": "(temporal_max >= 0.14) OR (spatial >= theta_or)",
                    "decision": "Fight" if f1 else "NonFight",
                    "fired": f1,
                    "role": "configured fusion candidate",
                },
                "F2_rank_sum": {
                    "rule": "0.5*pct(temporal_max) + 0.5*pct(spatial) >= theta_f",
                    "score": f2_score,
                    "decision": "Fight" if f2 else "NonFight",
                    "fired": f2,
                    "role": "configured fusion candidate; the evaluation's primary endpoint",
                },
            },
            "system_decision": {
                "from": "B0_temporal_only",
                "decision": "Fight" if b0 else "NonFight",
                "why": (
                    "B0 is the frozen incumbent. The confirmatory evaluation "
                    "found no statistically significant improvement from "
                    "fusion, so fusion does not displace it."
                ),
            },
            "causality": FUSION_CAUSALITY_NOTE,
            "interpretation": FUSION_INTERPRETATION,
            "not_claimed": FUSION_NOT_CLAIMED,
        }


def describe_fusion_layer() -> dict:
    """Static description for the UI and for preflight."""
    return {
        "feature": SPATIAL_FEATURE_NAME,
        "mode": FUSION_MODE,
        "interpretation": FUSION_INTERPRETATION,
        "not_claimed": FUSION_NOT_CLAIMED,
        "causality": FUSION_CAUSALITY_NOTE,
    }
