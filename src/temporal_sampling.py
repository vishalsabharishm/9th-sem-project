"""Temporal frame sampling for clip-based video models.

WHY NOT CONSECUTIVE FRAMES
--------------------------
Every validated RWF-2000 clip is 150 frames at 30 fps, i.e. exactly 5
seconds (confirmed by inspection over all 2000 clips, not assumed). Taking
16 *consecutive* frames would cover 16/30 = 0.53 s -- about a tenth of the
clip -- and can easily miss the event entirely. Frames are therefore
sampled across the whole clip.

SEGMENT SAMPLING
----------------
The clip is divided into ``clip_length`` equal segments and one frame is
taken from each, preserving temporal order and covering the full duration.
For ``total_frames`` T and ``clip_length`` n, segment ``i`` spans the
half-open interval::

    start_i = floor(i * T / n)
    stop_i  = floor((i + 1) * T / n)      # exclusive

For T=150, n=16 each segment is 9.375 frames wide, so segments are 9 or 10
frames long and tile the clip exactly with no gaps or overlaps.

``uniform_sample_indices`` takes the midpoint of each segment. It depends
only on ``(T, n)``, so evaluation is bit-for-bit repeatable::

    T=150, n=16 -> (4, 13, 23, 32, 41, 51, 60, 70, 79, 88, 98, 107, 116,
                    126, 135, 145)

``jittered_sample_indices`` takes a uniformly random frame from within
each segment. The frame still comes from its own segment, so ordering and
coverage are preserved and only the exact phase varies -- this is
augmentation, not a different sampling scheme. It is for training only.

SHORT CLIPS
-----------
When T < n some segments would be empty. Indices are clamped into
``[0, T-1]``, which repeats frames rather than failing. Repetition is
reported by :func:`describe_sampling` so a short clip is never silently
treated as a full one.

Nothing here is hardcoded to 150 frames or to 16: ``clip_length`` comes
from :class:`rwf2000_config.SamplingConfig` and ``total_frames`` is read
from each clip, so different clip lengths and source durations can be
explored without touching this module.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from rwf2000_config import SamplingConfig
except ImportError:  # pragma: no cover - supports package execution
    from src.rwf2000_config import SamplingConfig


UNIFORM_ACROSS_CLIP = "uniform_across_clip"


class TemporalSamplingError(ValueError):
    """Raised when a clip cannot be sampled with the requested settings."""


def _validate(total_frames: int, clip_length: int) -> None:
    if int(clip_length) <= 0:
        raise TemporalSamplingError("clip_length must be a positive integer.")
    if int(total_frames) <= 0:
        raise TemporalSamplingError(
            f"total_frames must be positive, got {total_frames}."
        )


def segment_bounds(total_frames: int, clip_length: int) -> Tuple[Tuple[int, int], ...]:
    """Return the ``(start, stop)`` frame range of each temporal segment.

    ``stop`` is exclusive. Segments tile ``[0, total_frames)`` exactly.
    Exposed so the index arithmetic can be inspected and tested directly
    rather than inferred from the sampled output.
    """
    _validate(total_frames, clip_length)
    total, count = int(total_frames), int(clip_length)
    bounds: List[Tuple[int, int]] = []
    for index in range(count):
        start = (index * total) // count
        stop = ((index + 1) * total) // count
        bounds.append((start, max(stop, start + 1)))
    return tuple(bounds)


def uniform_sample_indices(total_frames: int, clip_length: int) -> Tuple[int, ...]:
    """Return the deterministic midpoint frame index of each segment.

    Used for all evaluation: the same clip always yields the same frames,
    so a reported metric can be reproduced exactly.
    """
    _validate(total_frames, clip_length)
    last = int(total_frames) - 1
    return tuple(
        min(start + (stop - start) // 2, last)
        for start, stop in segment_bounds(total_frames, clip_length)
    )


def jittered_sample_indices(
    total_frames: int,
    clip_length: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, ...]:
    """Return one random frame index from within each segment.

    Training-time augmentation. Order and coverage are preserved because
    each index is drawn from its own segment; only the phase within the
    segment varies.
    """
    _validate(total_frames, clip_length)
    generator = np.random.default_rng() if rng is None else rng
    last = int(total_frames) - 1
    return tuple(
        min(int(generator.integers(start, stop)), last)
        for start, stop in segment_bounds(total_frames, clip_length)
    )


def sample_indices(
    total_frames: int,
    config: SamplingConfig = SamplingConfig(),
    training: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, ...]:
    """Return frame indices for one clip, honouring the sampling config.

    Jitter is applied only when ``training`` is set *and* the config
    enables it. Evaluation is deterministic whenever
    ``config.eval_deterministic`` is set, regardless of any generator
    passed in, so an evaluation run cannot accidentally become random.
    """
    if config.strategy != UNIFORM_ACROSS_CLIP:
        raise TemporalSamplingError(
            f"Unsupported sampling strategy {config.strategy!r}. "
            f"Only {UNIFORM_ACROSS_CLIP!r} is implemented."
        )

    if training and config.train_temporal_jitter:
        return jittered_sample_indices(total_frames, config.clip_length, rng)
    if not training and not config.eval_deterministic:
        return jittered_sample_indices(total_frames, config.clip_length, rng)
    return uniform_sample_indices(total_frames, config.clip_length)


def describe_sampling(
    total_frames: int,
    config: SamplingConfig = SamplingConfig(),
) -> Sequence[str]:
    """Return human-readable sampling provenance lines for reports."""
    indices = uniform_sample_indices(total_frames, config.clip_length)
    repeated = len(indices) != len(set(indices))
    return (
        f"strategy={config.strategy}",
        f"clip_length={config.clip_length}",
        f"source_frames={total_frames}",
        f"approximate_stride={total_frames / config.clip_length:.4f}",
        f"train_temporal_jitter={config.train_temporal_jitter}",
        f"eval_deterministic={config.eval_deterministic}",
        f"eval_indices={list(indices)}",
        f"frames_repeated={repeated}",
    )
