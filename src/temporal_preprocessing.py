"""Preprocessing from OpenCV clip frames to R3D-18 input tensors.

This module owns the conversion pipeline and nothing else. It does not
buffer frames (see ``temporal_clip_buffer``) and does not build or run a
model (see ``temporal_model``). The stages are exposed individually so
each can be tested in isolation:

    OpenCV BGR frames (T, H, W, C) uint8
      -> to_rgb            colour conversion
      -> to_video_tensor   layout change to (T, C, H, W)
      -> resize_and_crop   spatial resize then centre crop
      -> normalize         scale to [0, 1] then per-channel standardize
      -> add_batch_dim     final (1, C, T, H, W) R3D-18 input

The default constants are not invented: they are the values torchvision
publishes for its Kinetics-400 video classification preset
(``R3D_18_Weights.KINETICS400_V1.transforms()``), read from the installed
torchvision at development time. ``tests/test_temporal_preprocessing.py``
asserts numerical equivalence against that preset so the two cannot
silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np
import torch
from torchvision.transforms import functional as tv_functional

try:
    from temporal_clip_buffer import Clip
except ImportError:  # pragma: no cover - supports package execution
    from src.temporal_clip_buffer import Clip


# Values published by torchvision for R3D_18_Weights.KINETICS400_V1.
KINETICS400_RESIZE_SIZE: Tuple[int, int] = (128, 171)
KINETICS400_CROP_SIZE: Tuple[int, int] = (112, 112)
KINETICS400_MEAN: Tuple[float, float, float] = (0.43216, 0.394666, 0.37645)
KINETICS400_STD: Tuple[float, float, float] = (0.22803, 0.22145, 0.216989)


class TemporalPreprocessingError(ValueError):
    """Raised when a clip cannot be converted into a model input tensor."""


@dataclass(frozen=True)
class PreprocessingConfig:
    """Explicit, overridable preprocessing parameters.

    Defaults match the torchvision Kinetics-400 video preset. A model
    fine-tuned with different preprocessing must be paired with a config
    describing that preprocessing -- the values are deliberately data
    rather than constants baked into the code.
    """

    resize_size: Tuple[int, int] = KINETICS400_RESIZE_SIZE
    crop_size: Tuple[int, int] = KINETICS400_CROP_SIZE
    mean: Tuple[float, float, float] = KINETICS400_MEAN
    std: Tuple[float, float, float] = KINETICS400_STD
    # OpenCV decodes to BGR; torchvision's pretrained models expect RGB.
    convert_bgr_to_rgb: bool = True

    def __post_init__(self) -> None:
        for name in ("resize_size", "crop_size"):
            value = getattr(self, name)
            if len(value) != 2 or any(int(item) <= 0 for item in value):
                raise TemporalPreprocessingError(
                    f"{name} must be two positive integers, got {value!r}."
                )
        for name in ("mean", "std"):
            if len(getattr(self, name)) != 3:
                raise TemporalPreprocessingError(
                    f"{name} must have three channel values, got {getattr(self, name)!r}."
                )
        if any(float(value) == 0.0 for value in self.std):
            raise TemporalPreprocessingError("std values must be non-zero.")


class ClipPreprocessor:
    """Convert buffered OpenCV frames into an R3D-18 input tensor."""

    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()) -> None:
        self.config = config

    def to_rgb(self, clip_array: np.ndarray) -> np.ndarray:
        """Reverse the channel axis, converting OpenCV BGR to RGB.

        ``.copy()`` is required because the reversed view has a negative
        stride, which ``torch.from_numpy`` rejects.
        """
        if not self.config.convert_bgr_to_rgb:
            return clip_array
        return clip_array[..., ::-1].copy()

    def to_video_tensor(self, clip_array: np.ndarray) -> torch.Tensor:
        """Convert ``(T, H, W, C)`` frames to a ``(T, C, H, W)`` tensor."""
        return torch.from_numpy(clip_array).permute(0, 3, 1, 2).contiguous()

    def resize_and_crop(self, video: torch.Tensor) -> torch.Tensor:
        """Resize to ``resize_size`` then centre crop to ``crop_size``.

        ``antialias=False`` is deliberate and must not be "fixed". The
        torchvision video preset hard-codes it to preserve the exact
        preprocessing the Kinetics-400 weights were trained with, from
        before torchvision changed the argument's default. Enabling
        antialiasing here silently shifts the input distribution away from
        what the pretrained backbone expects.
        """
        resized = tv_functional.resize(
            video,
            list(self.config.resize_size),
            interpolation=tv_functional.InterpolationMode.BILINEAR,
            antialias=False,
        )
        return tv_functional.center_crop(resized, list(self.config.crop_size))

    def normalize(self, video: torch.Tensor) -> torch.Tensor:
        """Scale uint8 values into [0, 1] then standardize per channel."""
        scaled = tv_functional.convert_image_dtype(video, torch.float32)
        return tv_functional.normalize(
            scaled,
            mean=list(self.config.mean),
            std=list(self.config.std),
        )

    def preprocess(self, clip: Union[Clip, np.ndarray]) -> torch.Tensor:
        """Run the full pipeline and return a ``(1, C, T, H, W)`` tensor.

        Accepts either a ``Clip`` from ``TemporalClipBuffer`` or a raw
        ``(T, H, W, C)`` uint8 BGR array, so the preprocessing layer is
        usable without the buffer (e.g. for offline dataset work).
        """
        clip_array = clip.as_array() if isinstance(clip, Clip) else clip
        self._validate_clip_array(clip_array)

        rgb = self.to_rgb(clip_array)
        video = self.to_video_tensor(rgb)
        video = self.resize_and_crop(video)
        video = self.normalize(video)
        # (T, C, H, W) -> (C, T, H, W), then add the batch dimension.
        return video.permute(1, 0, 2, 3).unsqueeze(0)

    def _validate_clip_array(self, clip_array: np.ndarray) -> None:
        if not isinstance(clip_array, np.ndarray):
            raise TemporalPreprocessingError(
                f"Clip must be a numpy array or Clip, got {type(clip_array)!r}."
            )
        if clip_array.ndim != 4:
            raise TemporalPreprocessingError(
                "Clip must have 4 dimensions (T, H, W, C), got shape "
                f"{clip_array.shape}."
            )
        if clip_array.shape[0] == 0:
            raise TemporalPreprocessingError("Clip contains no frames.")
        if clip_array.shape[-1] != 3:
            raise TemporalPreprocessingError(
                "Clip frames must have 3 colour channels (BGR), got "
                f"{clip_array.shape[-1]}."
            )
        if clip_array.dtype != np.uint8:
            # OpenCV decodes to uint8; anything else means an unexpected
            # upstream conversion that would silently break normalization.
            raise TemporalPreprocessingError(
                f"Clip must be uint8 as produced by OpenCV, got {clip_array.dtype}."
            )


def describe_preprocessing(config: PreprocessingConfig) -> Sequence[str]:
    """Return human-readable preprocessing provenance lines for reports."""
    return (
        f"colour_conversion={'BGR->RGB' if config.convert_bgr_to_rgb else 'none'}",
        f"resize_size={tuple(config.resize_size)}",
        f"crop_size={tuple(config.crop_size)}",
        f"mean={tuple(config.mean)}",
        f"std={tuple(config.std)}",
    )
