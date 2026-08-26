"""Frame buffering for temporal (clip-based) abnormal-event models.

This module accumulates raw video frames into fixed-length, optionally
strided clips that a temporal model (e.g. a 3D CNN) can consume. It is
deliberately independent of any specific temporal model or training
dataset: it only manages frame ordering, windowing, and shape/type
consistency. Model-specific concerns (resizing, normalization, tensor
layout, channel ordering) belong in a separate model-adapter module that
consumes the ``Clip`` objects produced here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np


class TemporalBufferError(ValueError):
    """Raised when a frame or buffer configuration is invalid."""


@dataclass(frozen=True, eq=False)
class _BufferedFrame:
    """One buffered frame paired with its source frame number.

    ``eq=False`` is deliberate: the default dataclass equality would compare
    the ``frame`` arrays with ``==``, which returns an elementwise array
    rather than a bool and raises ``ValueError`` when Python tries to
    interpret it as one. Frozen dataclasses with ``eq=True`` (the default)
    are also made hashable, which would fail the same way. This type is
    never compared or hashed by this module; disabling both avoids a
    misleading trap for future callers.
    """

    frame_number: int
    frame: np.ndarray


@dataclass(frozen=True, eq=False)
class Clip:
    """An ordered, complete sequence of frames ready for a temporal model.

    ``frames`` holds direct references to the buffered arrays, not copies.
    Callers that need to mutate a frame should copy it explicitly first.

    ``eq=False`` for the same reason as ``_BufferedFrame`` above: the
    ``frames`` list contains numpy arrays, so default dataclass equality/
    hashing would raise rather than behave usefully.
    """

    frames: List[np.ndarray] = field(default_factory=list)
    frame_numbers: List[int] = field(default_factory=list)

    def as_array(self) -> np.ndarray:
        """Stack frames into a single ``(T, H, W, C)`` array.

        This performs one copy (via ``np.stack``) and applies no
        model-specific preprocessing -- no resizing, normalization, or
        channel reordering. Those steps belong to whichever model adapter
        consumes the clip.
        """
        return np.stack(self.frames, axis=0)

    def __len__(self) -> int:
        return len(self.frames)


class TemporalClipBuffer:
    """Accumulate frames into fixed-length, optionally strided clips.

    The buffer is a bounded ring: it never holds more than ``clip_length``
    frames. Once full, it reports a clip as available every ``stride``
    frames added, supporting overlapping windows (``stride < clip_length``),
    disjoint windows (``stride == clip_length``), or gapped windows
    (``stride > clip_length``).

    When ``stride > clip_length``, frames added between emitted windows are
    NOT subsampled into later clips -- they are discarded entirely by the
    ring buffer's eviction before the next window ever becomes ready. For
    example, with ``clip_length=4`` and ``stride=6``, frames arrive
    0..5, but frames 0 and 1 are evicted before the first clip is emitted,
    so the first clip is ``[2, 3, 4, 5]`` and frames 0-1 never appear in
    any clip this buffer ever returns. Choose ``stride > clip_length``
    only when that gap is an intentional sampling decision.

    Typical usage inside a per-frame processing loop::

        buffer = TemporalClipBuffer(clip_length=16, stride=8)
        for frame in frames:
            buffer.add_frame(frame)
            if buffer.is_ready():
                clip = buffer.get_clip()
                # hand `clip` to a temporal model adapter

    ``frame_number`` may be supplied explicitly (e.g. the caller's own
    one-based frame counter, matching the convention already used by
    ``EventDetection.frame_number`` elsewhere in this project) or omitted,
    in which case the buffer assigns its own zero-based sequence.
    """

    def __init__(
        self,
        clip_length: int,
        stride: int = 1,
        expected_shape: Optional[Tuple[int, int, int]] = None,
        expected_dtype: Optional["np.dtype"] = None,
    ) -> None:
        if clip_length <= 0:
            raise TemporalBufferError("clip_length must be a positive integer.")
        if stride <= 0:
            raise TemporalBufferError("stride must be a positive integer.")
        if expected_shape is not None and len(expected_shape) != 3:
            raise TemporalBufferError(
                f"expected_shape must be a 3-tuple (H, W, C), got {expected_shape!r}."
            )

        self.clip_length = clip_length
        self.stride = stride
        # The constructor-supplied constraints (possibly None) are kept
        # separately so `reset()` can restore them rather than permanently
        # switching an explicitly-typed buffer into auto-detect mode.
        self._init_expected_shape = expected_shape
        self._init_expected_dtype = expected_dtype
        self._expected_shape = expected_shape
        self._expected_dtype = expected_dtype
        self._frames: Deque[_BufferedFrame] = deque(maxlen=clip_length)
        self._frames_since_last_clip = 0
        self._auto_frame_number = 0

    def add_frame(self, frame: np.ndarray, frame_number: Optional[int] = None) -> None:
        """Validate and append one frame to the rolling window.

        ``frame_number`` is optional. If omitted, the buffer assigns its
        own zero-based, strictly increasing sequence number. If supplied,
        it is stored as-is (e.g. to preserve the caller's real frame index
        for later traceability) and does not affect the auto-numbering
        sequence used on calls that omit it.

        The oldest frame is evicted automatically once the buffer holds
        ``clip_length`` frames, so this call is O(1) and never grows memory
        beyond the configured window.

        ``frame_number`` is validated before the frame itself so that a
        rejected call never has partial side effects: in particular, the
        first-frame auto-detection of the expected shape/dtype (see
        ``_validate_frame``) must not happen unless the whole call is going
        to succeed. Otherwise a single bad ``frame_number`` on the first
        call would permanently -- and incorrectly -- lock the buffer to
        that rejected frame's shape.
        """
        if frame_number is not None and not isinstance(frame_number, int):
            raise TemporalBufferError(
                f"frame_number must be an int, got {type(frame_number)!r}."
            )

        self._validate_frame(frame)
        if frame_number is None:
            frame_number = self._auto_frame_number
            self._auto_frame_number += 1

        self._frames.append(_BufferedFrame(frame_number=frame_number, frame=frame))
        self._frames_since_last_clip += 1

    def _validate_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TemporalBufferError(
                f"Frame must be a numpy array, got {type(frame)!r}."
            )
        if frame.size == 0:
            raise TemporalBufferError("Cannot buffer an empty frame.")
        if frame.ndim != 3:
            raise TemporalBufferError(
                f"Frame must have 3 dimensions (H, W, C), got shape {frame.shape}."
            )

        if self._expected_shape is None:
            self._expected_shape = frame.shape
        elif frame.shape != self._expected_shape:
            raise TemporalBufferError(
                f"Frame shape {frame.shape} does not match this buffer's "
                f"established shape {self._expected_shape}."
            )

        if self._expected_dtype is None:
            self._expected_dtype = frame.dtype
        elif frame.dtype != self._expected_dtype:
            raise TemporalBufferError(
                f"Frame dtype {frame.dtype} does not match this buffer's "
                f"established dtype {self._expected_dtype}."
            )

    def is_ready(self) -> bool:
        """Return whether a full clip is currently available.

        True once the window holds ``clip_length`` frames and at least
        ``stride`` new frames have arrived since the last clip was
        retrieved (or since construction, for the first clip).
        """
        return (
            len(self._frames) == self.clip_length
            and self._frames_since_last_clip >= self.stride
        )

    def get_clip(self) -> Clip:
        """Return the current window as an ordered, complete ``Clip``.

        Retrieving a clip resets the stride counter so ``is_ready()`` will
        next return ``True`` only after ``stride`` further frames have been
        added. Raises ``TemporalBufferError`` if no clip is available yet.
        """
        if not self.is_ready():
            raise TemporalBufferError(
                f"Clip is not ready: buffered {len(self._frames)}/{self.clip_length} "
                f"frames, {self._frames_since_last_clip}/{self.stride} since last clip."
            )

        ordered = list(self._frames)
        self._frames_since_last_clip = 0
        return Clip(
            frames=[item.frame for item in ordered],
            frame_numbers=[item.frame_number for item in ordered],
        )

    def clear(self) -> None:
        """Clear all buffered frames and window-progress state.

        Any shape/dtype constraint auto-detected from earlier frames is
        also cleared, so a cleared buffer accepts a new frame shape or
        dtype (e.g. after switching to a different video source). A
        constraint explicitly passed to the constructor is restored, not
        cleared. The auto-assigned frame-number sequence restarts from 0.
        """
        self._frames.clear()
        self._frames_since_last_clip = 0
        self._auto_frame_number = 0
        self._expected_shape = self._init_expected_shape
        self._expected_dtype = self._init_expected_dtype

    def __len__(self) -> int:
        return len(self._frames)
