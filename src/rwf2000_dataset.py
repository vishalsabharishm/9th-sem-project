"""RWF-2000 clip datasets for the temporal violence classifier.

Composes existing project pieces rather than reimplementing them:

    rwf2000_splits    which clips belong to which evaluation set
    temporal_sampling which frames are taken from each clip
    temporal_preprocessing  frames -> normalized R3D-18 tensor
    video_loader.long_path  filesystem access that survives MAX_PATH

Three datasets are offered, matching the locked evaluation policy:

``training``
    The official 1600 training clips, unchanged.
``primary_evaluation``
    394 held-out clips: the official 400 minus the 6 confirmed leaks.
    Headline metrics are reported on this set.
``official_validation``
    All 400 held-out clips, for secondary/reference reporting only.

Nothing here writes to the dataset directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from dataset_inspection import normalize_label
    from rwf2000_config import LABEL_TO_INDEX, SamplingConfig
    from rwf2000_splits import build_evaluation_splits, clip_path, split_clip_paths
    from temporal_preprocessing import ClipPreprocessor, PreprocessingConfig
    from temporal_sampling import sample_indices
except ImportError:  # pragma: no cover - supports package execution
    from src.dataset_inspection import normalize_label
    from src.rwf2000_config import LABEL_TO_INDEX, SamplingConfig
    from src.rwf2000_splits import (
        build_evaluation_splits,
        clip_path,
        split_clip_paths,
    )
    from src.temporal_preprocessing import ClipPreprocessor, PreprocessingConfig
    from src.temporal_sampling import sample_indices


TRAIN_SPLIT_DIRECTORY = "train"


class ClipDecodeError(RuntimeError):
    """Raised when a clip's frames cannot be read."""


def label_index_for(relative_path: str) -> int:
    """Return the binary class index for a ``split/class_dir/file`` path.

    The label comes from the directory name with its split prefix removed
    (``val/Val_Fight/x.avi`` -> ``Fight`` -> 1), so the four on-disk class
    directories collapse to the two real classes.
    """
    parts = relative_path.split("/")
    if len(parts) < 3:
        raise ClipDecodeError(
            f"Expected a 'split/class_dir/file' path, got {relative_path!r}."
        )
    label = normalize_label(parts[1], parts[0])
    if label not in LABEL_TO_INDEX:
        raise ClipDecodeError(
            f"Unknown label {label!r} from {relative_path!r}; "
            f"expected one of {sorted(LABEL_TO_INDEX)}."
        )
    return LABEL_TO_INDEX[label]


def decode_frames(path: str, indices: Sequence[int]) -> np.ndarray:
    """Read the requested frame indices from one video, in order.

    Frames are read sequentially rather than by seeking. Seeking with
    ``CAP_PROP_POS_FRAMES`` is unreliable across codecs and can silently
    land on a neighbouring frame, which would make "deterministic"
    evaluation sampling untrue. A 150-frame clip is small enough that a
    linear read is cheap and exact.

    A requested index beyond the end of the stream reuses the last frame
    decoded, so a clip whose container over-reports its length still
    yields a full-length tensor instead of failing mid-epoch.
    """
    if not indices:
        raise ClipDecodeError(f"No frame indices requested for {path!r}.")

    wanted = sorted(set(int(index) for index in indices))
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ClipDecodeError(f"OpenCV could not open clip: {path}")

    decoded: Dict[int, np.ndarray] = {}
    last_frame: Optional[np.ndarray] = None
    try:
        position = 0
        exhausted = False
        for target in wanted:
            while not exhausted and position <= target:
                success, frame = capture.read()
                if not success:
                    exhausted = True
                    break
                last_frame = frame
                position += 1
            if last_frame is None:
                raise ClipDecodeError(f"Clip contains no readable frames: {path}")
            decoded[target] = last_frame
    finally:
        capture.release()

    return np.stack([decoded[int(index)] for index in indices], axis=0)


@dataclass(frozen=True)
class DatasetConfig:
    """How clips are turned into tensors. All values are overridable."""

    sampling: SamplingConfig = SamplingConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    # Training applies temporal jitter; evaluation never does.
    training: bool = False
    # Base seed for jitter. Combined with the epoch and the clip index so
    # augmentation varies across epochs but is reproducible from the seed.
    seed: int = 42


class RWF2000ClipDataset(Dataset):
    """A list of RWF-2000 clips served as ``(C, T, H, W)`` tensors.

    Item ``i`` is ``(clip_tensor, label_index)``. The tensor carries no
    batch dimension; the ``DataLoader`` adds it.
    """

    def __init__(
        self,
        root: Path,
        relative_paths: Sequence[str],
        config: DatasetConfig = DatasetConfig(),
    ) -> None:
        self.root = Path(root)
        self.relative_paths: Tuple[str, ...] = tuple(relative_paths)
        self.config = config
        self.labels: Tuple[int, ...] = tuple(
            label_index_for(path) for path in self.relative_paths
        )
        self.preprocessor = ClipPreprocessor(config.preprocessing)
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.relative_paths)

    def set_epoch(self, epoch: int) -> None:
        """Vary training jitter per epoch while staying seed-reproducible."""
        self._epoch = int(epoch)

    def label_counts(self) -> Dict[str, int]:
        """Return the observed class balance of this dataset."""
        names = {index: name for name, index in LABEL_TO_INDEX.items()}
        counts: Dict[str, int] = {name: 0 for name in LABEL_TO_INDEX}
        for label in self.labels:
            counts[names[label]] += 1
        return counts

    def frame_indices_for(self, item: int, total_frames: int) -> Tuple[int, ...]:
        """Return the frame indices this dataset would sample for ``item``.

        Exposed so evaluation determinism can be asserted directly, and so
        a future Grad-CAM overlay can recover which source frames a
        prediction was actually computed from.
        """
        generator = None
        if self.config.training and self.config.sampling.train_temporal_jitter:
            generator = np.random.default_rng(
                (self.config.seed, self._epoch, int(item))
            )
        return sample_indices(
            total_frames,
            self.config.sampling,
            training=self.config.training,
            rng=generator,
        )

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        relative = self.relative_paths[item]
        path = clip_path(self.root, relative)

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ClipDecodeError(f"OpenCV could not open clip: {relative}")
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if total_frames <= 0:
            raise ClipDecodeError(f"Clip reports no frames: {relative}")

        indices = self.frame_indices_for(item, total_frames)
        frames = decode_frames(path, indices)
        # ClipPreprocessor returns (1, C, T, H, W); the loader collates the
        # batch dimension itself, so drop the singleton here.
        tensor = self.preprocessor.preprocess(frames).squeeze(0)
        return tensor, self.labels[item]


def build_training_dataset(
    root: Path,
    config: Optional[DatasetConfig] = None,
) -> RWF2000ClipDataset:
    """Return the official 1600-clip training set, unchanged.

    The leakage exclusion deliberately does not apply here: the duplicated
    clips are legitimate training data, and removing them would alter the
    official training split.
    """
    settings = config or DatasetConfig(training=True)
    return RWF2000ClipDataset(
        root, split_clip_paths(root, TRAIN_SPLIT_DIRECTORY), settings
    )


def build_primary_evaluation_dataset(
    root: Path,
    config: Optional[DatasetConfig] = None,
) -> RWF2000ClipDataset:
    """Return the 394 clean held-out clips headline metrics are reported on."""
    settings = config or DatasetConfig(training=False)
    splits = build_evaluation_splits(root)
    return RWF2000ClipDataset(root, splits.primary_evaluation, settings)


def build_official_validation_dataset(
    root: Path,
    config: Optional[DatasetConfig] = None,
) -> RWF2000ClipDataset:
    """Return all 400 official held-out clips, for secondary reporting.

    Includes the 6 confirmed leaks, so any metric from this set must be
    labelled as reference-only and never quoted as the headline result.
    """
    settings = config or DatasetConfig(training=False)
    splits = build_evaluation_splits(root)
    return RWF2000ClipDataset(root, splits.official_validation, settings)
