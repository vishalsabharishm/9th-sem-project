"""Temporal inference interface: frames in, model predictions out.

This module composes the three independent layers into the pipeline

    frames -> TemporalClipBuffer -> ClipPreprocessor -> R3D18TemporalModel

and is the seam the rest of the project will eventually depend on. Only
the model configuration changes when the Kinetics-pretrained backbone is
replaced by a fine-tuned task-specific checkpoint; the buffering,
preprocessing, and calling conventions stay identical.

This module deliberately does NOT emit ``EventDetection`` records or talk
to ``RiskAssessor``. Wiring the temporal signal into the abnormal-event
pipeline is a separate, later step, and must not happen while the only
available weights are non-task-specific (see ``temporal_model``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    from temporal_clip_buffer import Clip, TemporalClipBuffer
    from temporal_model import R3D18TemporalModel, TemporalModelConfig, TemporalPrediction
    from temporal_preprocessing import ClipPreprocessor, PreprocessingConfig
except ImportError:  # pragma: no cover - supports package execution
    from src.temporal_clip_buffer import Clip, TemporalClipBuffer
    from src.temporal_model import (
        R3D18TemporalModel,
        TemporalModelConfig,
        TemporalPrediction,
    )
    from src.temporal_preprocessing import ClipPreprocessor, PreprocessingConfig


DEFAULT_CLIP_LENGTH = 16
DEFAULT_STRIDE = 8


class TemporalInferenceError(RuntimeError):
    """Raised when the temporal inference pipeline is misconfigured."""


@dataclass(frozen=True)
class TemporalInferenceConfig:
    """Configuration for the composed temporal inference pipeline.

    ``score_threshold`` is intentionally ``None``. A decision threshold is
    only meaningful once a task-specific model has been fine-tuned and
    evaluated, and picking one before that would be a fabricated value.
    """

    clip_length: int = DEFAULT_CLIP_LENGTH
    stride: int = DEFAULT_STRIDE
    score_threshold: Optional[float] = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: TemporalModelConfig = field(default_factory=TemporalModelConfig)

    def __post_init__(self) -> None:
        if self.clip_length <= 0:
            raise TemporalInferenceError("clip_length must be a positive integer.")
        if self.stride <= 0:
            raise TemporalInferenceError("stride must be a positive integer.")
        if self.score_threshold is not None and not 0.0 <= self.score_threshold <= 1.0:
            raise TemporalInferenceError(
                "score_threshold must be between 0.0 and 1.0 when set."
            )


@dataclass(frozen=True, eq=False)
class TemporalInferenceResult:
    """One prediction plus the clip context that produced it."""

    prediction: TemporalPrediction
    frame_numbers: List[int]
    inference_seconds: float

    @property
    def is_task_specific(self) -> bool:
        """Whether this result may be treated as a real task prediction."""
        return self.prediction.is_task_specific


class TemporalInferenceEngine:
    """Drive buffering, preprocessing, and inference from a frame stream."""

    def __init__(
        self,
        config: TemporalInferenceConfig = TemporalInferenceConfig(),
        model: Optional[R3D18TemporalModel] = None,
    ) -> None:
        self.config = config
        self.buffer = TemporalClipBuffer(
            clip_length=config.clip_length,
            stride=config.stride,
        )
        self.preprocessor = ClipPreprocessor(config.preprocessing)
        # An explicit model may be injected for testing, or to reuse an
        # already-loaded model across several videos.
        self.model = model if model is not None else R3D18TemporalModel(config.model)

    def add_frame(
        self,
        frame: np.ndarray,
        frame_number: Optional[int] = None,
    ) -> Optional[TemporalInferenceResult]:
        """Buffer one frame, running inference when a clip completes.

        Returns ``None`` for frames that only partially fill the current
        window, so this is safe to call unconditionally in a per-frame
        processing loop.
        """
        self.buffer.add_frame(frame, frame_number)
        if not self.buffer.is_ready():
            return None
        return self.infer_clip(self.buffer.get_clip())

    def infer_clip(self, clip: Clip) -> TemporalInferenceResult:
        """Preprocess and run one already-assembled clip."""
        clip_tensor = self.preprocessor.preprocess(clip)
        started = time.perf_counter()
        prediction = self.model.predict(clip_tensor)
        elapsed = time.perf_counter() - started
        return TemporalInferenceResult(
            prediction=prediction,
            frame_numbers=list(clip.frame_numbers),
            inference_seconds=elapsed,
        )

    def reset(self) -> None:
        """Clear buffered frames without discarding the loaded model."""
        self.buffer.clear()
