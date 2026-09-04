"""Bridges temporal (R3D-18) violence-probability window scores into the
Phase 4/5 event-detection and risk-assessment pipeline.

Provenance discipline (see docs/temporal_risk_integration.md):

- Per-window and per-clip probabilities produced by the temporal model are
  tagged ``measured_model_probability`` -- they come from the network's
  output on real video data (either live inference, or a precomputed score
  transcribed from a Kaggle run), never invented.
- The 0.9 / 0.95 confidence values already hard-coded in
  ``abnormal_event_detector.AbnormalEventDetector`` for the Phase 4
  rule-engine events (Stationary Person / Restricted Area Entry) are
  ``declared_rule_constant`` -- they describe how much the RULE trusts
  itself, not a measurement. This module does not touch those. It only adds
  a new event type whose confidence is always a real measured probability.

Two ways to obtain a clip's window scores:

- Live: ``TemporalInferenceEngine`` as frames stream in (the production path,
  once a fine-tuned checkpoint is deployed on this machine). Feed its
  per-window ``fight_probability`` values into
  :class:`TemporalEventAdapter.evaluate_clip` the same way.
- Precomputed: :class:`PrecomputedWindowScoreSource` reads a
  ``temporal_risk/*_window_scores.csv`` file for a named clip. This is what
  the current demo uses, since the clean-baseline checkpoint
  (``best.pt``, 132.74 MB) has not been downloaded to this machine. Every
  score returned this way is a real number transcribed from Kaggle notebook
  ``notebookbce85b4c2b`` (see ``temporal_risk/window_scoring_manifest.json``)
  -- nothing here is fabricated or estimated.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from abnormal_event_detector import EventDetection
except ImportError:  # pragma: no cover - supports package execution
    from src.abnormal_event_detector import EventDetection


TEMPORAL_EVENT_TYPE = "Temporal Violence Signal"
PROVENANCE_MEASURED = "measured_model_probability"
PROVENANCE_DECLARED = "declared_rule_constant"
# A score this process computed by running the model, as distinct from
# PROVENANCE_MEASURED, which is a real model output transcribed from a Kaggle
# run and replayed from CSV. Both are genuine measurements; only the live one
# proves inference happened here, so a report must be able to tell them apart.
PROVENANCE_LIVE_INFERENCE = "live_model_inference"

# Index of the positive class in the model's output, per
# rwf2000_config.LABEL_TO_INDEX = {"NonFight": 0, "Fight": 1}. Named here so
# the conversion below cannot silently read the wrong column.
FIGHT_CLASS_INDEX = 1

# Event types whose EventDetection.confidence is a fixed number the rule
# engine assigns to itself (see event_rules.py / abnormal_event_detector.py),
# not a value read off a model. Used by risk_assessment.py to tag provenance
# correctly rather than guessing from the number's magnitude.
DECLARED_CONSTANT_EVENT_TYPES = frozenset({
    "Stationary Person",
    "Restricted Area Entry",
})


class TemporalAdapterError(RuntimeError):
    """Raised when the frozen aggregation rule or window scores are unusable."""


@dataclass(frozen=True)
class FrozenAggregationRule:
    """The rule selected by ``tools/select_temporal_aggregation.py``, loaded
    verbatim from ``temporal_risk/frozen_aggregation.json``. This class does
    not re-derive or tune anything -- it only replays the frozen decision."""

    rule: str
    params: dict
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "FrozenAggregationRule":
        path = Path(path)
        if not path.exists():
            raise TemporalAdapterError(
                f"{path} not found. Run tools/select_temporal_aggregation.py first "
                "(on the carve split only) to produce it."
            )
        data = json.loads(path.read_text())
        selected = data["selected"]
        return cls(rule=selected["rule"], params=selected["params"], source_path=path)

    def decide(self, window_scores: List[float]) -> bool:
        """Apply the frozen rule to one clip's window scores.

        Mirrors the four rule families evaluated by
        ``tools/select_temporal_aggregation.py`` / applied by
        ``tools/apply_frozen_aggregation.py`` exactly -- this must never grow
        a fifth option or a different threshold without re-running selection
        on the carve split.
        """
        if not window_scores:
            return False
        if self.rule == "any":
            threshold = self.params["window_threshold"]
            return any(score >= threshold for score in window_scores)
        if self.rule == "k_of_n":
            threshold = self.params["window_threshold"]
            k = self.params["k"]
            return sum(1 for score in window_scores if score >= threshold) >= k
        if self.rule == "max":
            return max(window_scores) >= self.params["threshold"]
        if self.rule == "mean":
            return (sum(window_scores) / len(window_scores)) >= self.params["threshold"]
        raise TemporalAdapterError(f"Unknown frozen aggregation rule: {self.rule!r}")


@dataclass(frozen=True)
class WindowScore:
    """One sliding-window prediction: frames [first_frame, last_frame]."""

    window_index: int
    first_frame: int
    last_frame: int
    fight_probability: float


def window_score_from(
    result: Any,
    window_index: int,
    require_task_specific: bool = True,
) -> WindowScore:
    """Convert one live inference result into a :class:`WindowScore`.

    This is the seam between ``TemporalInferenceEngine`` and the aggregation
    rule. Until it existed, ``WindowScore`` could only be built from a CSV row,
    so a live model had no way to reach the decision path at all.

    ``result`` is duck-typed rather than annotated as
    ``TemporalInferenceResult``: importing that type here would pull torch into
    a module the CSV replay path depends on, for no benefit. Anything carrying
    ``frame_numbers`` and ``prediction.probabilities`` works.

    The four fields map directly, with nothing invented:

    ==================  ==========================================
    ``window_index``    the caller's own counter
    ``first_frame``     ``result.frame_numbers[0]``
    ``last_frame``      ``result.frame_numbers[-1]``
    ``fight_probability`` ``result.prediction.probabilities[0, 1]``
    ==================  ==========================================

    ``require_task_specific`` defaults to True and is the reason this function
    exists rather than the conversion being inlined. ``TemporalPrediction``
    already knows whether its weights were trained for this task; a
    random-init or Kinetics-400 forward pass produces a number in [0, 1] that
    looks exactly like a violence probability and is not one. Refusing it here
    means an untrained model cannot reach ``FrozenAggregationRule`` and raise a
    "High" risk alert. Set it False only in tests that deliberately exercise
    the plumbing with untrained weights.
    """
    frame_numbers = list(getattr(result, "frame_numbers", []) or [])
    if not frame_numbers:
        raise TemporalAdapterError(
            "Inference result carries no frame numbers, so the window it covers "
            "is unknown and no WindowScore can be built from it."
        )
    prediction = getattr(result, "prediction", None)
    if prediction is None:
        raise TemporalAdapterError("Inference result carries no prediction.")

    if require_task_specific and not getattr(prediction, "is_task_specific", False):
        raise TemporalAdapterError(
            f"Refusing to build a WindowScore from a non-task-specific "
            f"prediction (provenance={getattr(prediction, 'provenance', 'unknown')!r}). "
            "Its output is not a violence probability, and feeding it to the "
            "frozen aggregation rule would raise a fabricated alert."
        )

    probabilities = getattr(prediction, "probabilities", None)
    if probabilities is None:
        raise TemporalAdapterError("Prediction carries no probabilities.")
    try:
        fight_probability = float(probabilities[0][FIGHT_CLASS_INDEX])
    except (IndexError, TypeError) as error:
        raise TemporalAdapterError(
            f"Prediction probabilities are not shaped (1, 2); cannot read the "
            f"Fight column at index {FIGHT_CLASS_INDEX}: {error}"
        ) from error

    return WindowScore(
        window_index=int(window_index),
        first_frame=int(frame_numbers[0]),
        last_frame=int(frame_numbers[-1]),
        fight_probability=fight_probability,
    )


class PrecomputedWindowScoreSource:
    """Reads real, already-computed per-window scores for named clips from a
    ``temporal_risk/*_window_scores.csv`` file.

    Used in place of a locally-hosted checkpoint for the demo. Never invents
    a score for a clip it cannot find -- :meth:`get` returns ``None`` and
    callers must handle that explicitly rather than substituting a guess.
    """

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise TemporalAdapterError(f"{self.csv_path} not found.")
        self._by_clip: Dict[str, List[WindowScore]] = {}
        self._true_label: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        with open(self.csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                clip = row["clip"]
                self._by_clip.setdefault(clip, []).append(
                    WindowScore(
                        window_index=int(row["window_index"]),
                        first_frame=int(row["first_frame"]),
                        last_frame=int(row["last_frame"]),
                        fight_probability=float(row["fight_probability"]),
                    )
                )
                self._true_label[clip] = row["true_label"]
        for scores in self._by_clip.values():
            scores.sort(key=lambda w: w.window_index)

    def get(self, clip: str) -> Optional[List[WindowScore]]:
        """Return the ordered window scores for ``clip``, or ``None``."""
        return self._by_clip.get(clip)

    def true_label(self, clip: str) -> Optional[str]:
        """Return the dataset ground-truth label for ``clip``, if known.

        This is available because the demo clips come from a labeled
        evaluation split -- it is never used to influence the aggregation
        decision, only to check/report the pipeline's own output afterward.
        """
        return self._true_label.get(clip)

    def clips(self) -> List[str]:
        return sorted(self._by_clip.keys())


class TemporalEventAdapter:
    """Turns one clip's window scores into an :class:`EventDetection`.

    Produces at most one event per clip: a positive "Temporal Violence
    Signal" only when the frozen aggregation rule fires. A clip the rule does
    not flag produces no event -- this module never emits a "cleared"
    event, matching how the rest of Phase 4 only reports positive findings.
    """

    def __init__(self, frozen_rule: FrozenAggregationRule) -> None:
        self.frozen_rule = frozen_rule

    def evaluate_clip(
        self,
        clip: str,
        window_scores: List[WindowScore],
        frame_offset: int = 0,
        timestamp_seconds: Optional[float] = None,
    ) -> Optional[EventDetection]:
        if not window_scores:
            return None

        probabilities = [w.fight_probability for w in window_scores]
        fired = self.frozen_rule.decide(probabilities)
        if not fired:
            return None

        peak = max(window_scores, key=lambda w: w.fight_probability)
        evidence = [
            f"clip={clip}",
            f"aggregation_rule={self.frozen_rule.rule}",
            f"aggregation_params={self.frozen_rule.params}",
            f"n_windows={len(window_scores)}",
            f"peak_window_index={peak.window_index}",
            f"peak_window_frames=[{peak.first_frame},{peak.last_frame}]",
            f"peak_fight_probability={peak.fight_probability:.6f} ({PROVENANCE_MEASURED})",
            "checkpoint_source=notebook7bb9a86555 v3 clean_experiment/"
            "training/checkpoints/best.pt (epoch 12, frozen threshold 0.16)",
        ]
        return EventDetection(
            event_type=TEMPORAL_EVENT_TYPE,
            description=(
                "Sliding-window temporal model flagged this clip as violent "
                f"under the frozen '{self.frozen_rule.rule}' aggregation rule "
                f"(params={self.frozen_rule.params})."
            ),
            confidence=float(peak.fight_probability),
            evidence=evidence,
            object_id=None,
            frame_number=frame_offset + peak.first_frame,
            timestamp_seconds=timestamp_seconds,
            object_ids=[],
        )
