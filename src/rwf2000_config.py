"""Dataset provenance and initial training configuration for RWF-2000.

This module is a declarative record, not executable training logic. It
exists so that a later experiment can be reproduced and so the thesis can
cite exactly what was used.

SCOPE
-----
RWF-2000 trains ONLY the learned temporal violence/action branch of this
project. It is not the dataset for the project as a whole, and it is not
responsible for crowd or crowd-interaction anomaly detection -- that
remains the existing geometric ``CrowdInteractionAnalyzer``.

STATUS OF THE HYPERPARAMETERS
-----------------------------
Every value in :class:`InitialTrainingConfig` is an INITIAL EXPERIMENTAL
SETTING chosen from common practice for small binary video datasets. None
of them has been validated on this dataset, and none should be reported
as a tuned or optimal value. They are a documented starting point.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple


# --------------------------------------------------------------------------
# Dataset provenance (verified against the Zenodo record and the official
# project repository; see the accompanying research report).
# --------------------------------------------------------------------------

ZENODO_RECORD_URL = "https://zenodo.org/records/15687512"
ZENODO_VERSION_DOI = "10.5281/zenodo.15687512"
ZENODO_CONCEPT_DOI = "10.5281/zenodo.15687511"
ZENODO_DEPOSITOR = "Cheng, Ming (Duke Kunshan University, ORCID 0000-0002-4733-3596)"
OFFICIAL_REPOSITORY = (
    "https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection"
)

# The Zenodo deposit is licensed CC-BY-4.0, while the official repository
# additionally restricts redistribution and commercial use. We deliberately
# comply with the stricter union of the two.
ZENODO_LICENSE = "CC-BY-4.0"
REPOSITORY_TERMS = (
    "No modification/redistribution without SMIIP Lab approval.",
    "No commercial use without SMIIP Lab approval.",
    "Must not be used in ways damaging to mental health or personal privacy.",
    "Cite the ICPR 2021 paper when publishing.",
)
INTENDED_USE = "Academic, non-commercial research. Videos are not redistributed."

CITATION_BIBTEX = """@INPROCEEDINGS{9412502,
  author={Cheng, Ming and Cai, Kunjing and Li, Ming},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  title={RWF-2000: An Open Large Scale Video Database for Violence Detection},
  year={2021}, pages={4183-4190},
  doi={10.1109/ICPR48806.2021.9412502}}"""

CITATION_DATASET = (
    "Cheng, M. (2021). RWF2000 - A Large Scale Video Database for Violence "
    "Detection [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15687512"
)

ARCHIVE_TOTAL_BYTES = 9_399_888_359
# Published MD5 digests, used to verify archive integrity after download.
ARCHIVE_MD5: Dict[str, str] = {
    "RWF-2000.7z.001": "3e6004b6e165dac20597d26b7f8f7602",
    "RWF-2000.7z.002": "36375feaa57480718aa9ee5ca5b56282",
    "RWF-2000.7z.003": "9ce41f1a4bfb2fc43cc9e527c10d4b8d",
    "RWF-2000.7z.004": "f8eee28525001f7b48819985ff960ecf",
    "RWF-2000.7z.005": "aafaf9a4a7e40e6d50100ebb59095ea8",
    "RWF-2000.7z.006": "4963fcc70074f5a40f8c10eea64c3e00",
    "RWF-2000.7z.007": "8f6c405c5d90f1aeba090e55317f16df",
    "RWF-2000.7z.008": "1ed0b8093c45b36497284cfb5fdb5a58",
    "RWF-2000.7z.009": "01408eb3f6782b54aa00f401a3b533fc",
    "RWF-2000.7z.010": "0437efd6c0028a04032dcb92d10e3734",
    "RWF-2000.7z.011": "9fc24b9a02ab4181636b3a99366966e3",
    "RWF-2000.7z.012": "3c9b99f2acc42c98ba8a34e90e0c3436",
    "RWF-2000.7z.013": "f8b1145e3d261057243c9da49a88c5b0",
}

# Published dataset characteristics, to be confirmed by our own inspection
# before training. ``dataset_inspection`` produces the observed values.
EXPECTED_CLIP_SECONDS = 5.0
EXPECTED_FPS = 30.0
EXPECTED_FRAMES_PER_CLIP = 150
EXPECTED_TOTAL_CLIPS = 2000
# The published split is 1600 training / 400 held-out clips. The archive
# names the held-out directory ``val`` rather than ``test``; it is still
# the official, untouched evaluation split. The key below is the real
# directory name on disk so that observed and expected counts compare
# directly instead of silently missing each other.
EXPECTED_SPLIT_COUNTS = {"train": 1600, "val": 400}
EXPECTED_LABEL_COUNTS = {"Fight": 1000, "NonFight": 1000}

# --------------------------------------------------------------------------
# Observed on-disk layout (confirmed against the extracted archive, not
# assumed from the paper). RWF-2000 prefixes every class directory with the
# split it sits in, so the four directories below carry only two labels.
# --------------------------------------------------------------------------

OBSERVED_SPLIT_DIRECTORIES = ("train", "val")
OBSERVED_CLASS_DIRECTORIES = {
    "train": ("Train_Fight", "Train_NonFight"),
    "val": ("Val_Fight", "Val_NonFight"),
}

# The two real classes, after the split prefix is stripped.
CANONICAL_LABELS = ("NonFight", "Fight")
# Positive class is ``Fight``: index 1 is the violence class, so a sigmoid
# or softmax column 1 reads as "probability of violence". Not yet consumed
# by any training code -- recorded here so the mapping is defined once.
LABEL_TO_INDEX = {"NonFight": 0, "Fight": 1}


# --------------------------------------------------------------------------
# Selected operating point
# --------------------------------------------------------------------------
#
# Chosen by tools/threshold_sweep.py from the per-clip scores of the
# Kaggle baseline run (notebook5d98537e15 Version #2), NOT hand-picked.
#
# Rule: maximise Fight recall subject to precision >= 0.80, false-positive
# rate <= 0.20, and accuracy no worse than the 0.5 baseline. In words --
# buy as much recall as possible without paying for it in overall
# correctness. Accuracy alone is the wrong objective here: on a balanced
# set it trades one missed assault for one dismissed false alarm, and
# those costs are not equal in a surveillance system.
#
# At 0.14 vs the 0.5 default, on the 394 leak-free held-out clips:
#     recall    0.8200 -> 0.8900   (missed fights 36 -> 22)
#     precision 0.8962 -> 0.8436   (false alarms  19 -> 33)
#     accuracy  0.8604 -> 0.8604   (unchanged)
#     F1        0.8564 -> 0.8662
#
# IMPORTANT CAVEAT. This threshold was selected on the same 394 clips the
# headline metrics are reported on, so it is optimistically biased. The
# project's own SplitPolicy provides for a validation subset carved from
# train (validation_fraction 0.15) precisely so a threshold can be chosen
# without touching the held-out set. Re-derive this value there before
# quoting it as a final result.
SELECTED_DECISION_THRESHOLD: float = 0.14
SELECTED_THRESHOLD_PROVENANCE: Dict[str, str] = {
    "source_run": "kaggle notebook5d98537e15 version 2",
    "predictions": "baseline_results/evaluation/predictions.csv",
    "evaluation_set": "primary (394 leak-free held-out clips)",
    "selection_rule": (
        "max recall s.t. precision >= 0.80, FPR <= 0.20, "
        "accuracy >= baseline accuracy (0.8604)"
    ),
    "selected_on": "the held-out set itself -- re-derive on a train-carved "
                   "validation subset before final reporting",
}

# The archive directory that holds the official held-out evaluation clips.
HELD_OUT_SPLIT_DIRECTORY = "val"


# --------------------------------------------------------------------------
# Confirmed cross-split leakage.
#
# Six validation clips are byte-identical copies of training clips that
# were released under different filenames. Scoring them would be scoring
# the model on data it trained on, so they are excluded from the PRIMARY
# evaluation set.
#
# This leakage is present in the official RWF-2000 release; it is not an
# artefact of how this project downloaded or extracted the data. The
# affected clips are all NonFight.
#
# Nothing here deletes or edits a dataset file. The official ``val``
# directory keeps all 400 clips exactly as released; exclusion happens at
# evaluation time, by relative path.
#
# The list was produced by ``find_confirmed_cross_split_duplicates``: a
# size + first-megabyte scan proposed candidates and a full-file MD5
# confirmed each one. Re-running that function reproduces this table,
# and ``verify_recorded_leakage`` checks it still matches on disk.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakedValidationClip:
    """One validation clip that duplicates a training clip byte for byte."""

    validation_clip: str
    training_clip: str
    md5: str
    label: str = "NonFight"

    def as_dict(self) -> dict:
        """Return a JSON-native representation."""
        return asdict(self)


LEAKED_VALIDATION_CLIPS: Tuple[LeakedValidationClip, ...] = (
    LeakedValidationClip(
        validation_clip="val/Val_NonFight/cw8fPfUL_0.avi",
        training_clip="train/Train_NonFight/EBwt3Thb_0.avi",
        md5="12bf348af2bcfbb46cfc7457c23b5878",
    ),
    LeakedValidationClip(
        validation_clip="val/Val_NonFight/xV69l7Vj_0.avi",
        training_clip="train/Train_NonFight/XQR0CvyY_1.avi",
        md5="1d5801106ee78c9eaa81bd80672c41e0",
    ),
    LeakedValidationClip(
        validation_clip="val/Val_NonFight/aZxUVVqk_0.avi",
        training_clip="train/Train_NonFight/aGfwZ3E4_0.avi",
        md5="441d74e1da6179d8b6d2c7ec7f66fd25",
    ),
    LeakedValidationClip(
        validation_clip="val/Val_NonFight/E9SvEiNt_0.avi",
        training_clip="train/Train_NonFight/ndNHBJbS_0.avi",
        md5="9ede70a8e313fd9b3d0141c5c08cdc5d",
    ),
    LeakedValidationClip(
        validation_clip="val/Val_NonFight/7emxm3za_0.avi",
        training_clip="train/Train_NonFight/uz4dXoN5_0.avi",
        md5="3ea28aa9dceca84c40d61fa8e98572e5",
    ),
    LeakedValidationClip(
        validation_clip="val/Val_NonFight/ZCa5f4yq_0.avi",
        training_clip="train/Train_NonFight/wKMXI2vp_0.avi",
        md5="6c38f855fd154df38712748ec4203d0d",
    ),
)

# Evaluation-set sizes. The official split is untouched; the primary set
# is simply the official one minus the confirmed leaks.
OFFICIAL_VALIDATION_CLIP_COUNT = 400
LEAKAGE_EXCLUDED_CLIP_COUNT = len(LEAKED_VALIDATION_CLIPS)
PRIMARY_EVALUATION_CLIP_COUNT = (
    OFFICIAL_VALIDATION_CLIP_COUNT - LEAKAGE_EXCLUDED_CLIP_COUNT
)


def leakage_excluded_paths() -> FrozenSet[str]:
    """Return the relative paths of validation clips excluded as leakage."""
    return frozenset(clip.validation_clip for clip in LEAKED_VALIDATION_CLIPS)


ZENODO_FILE_URL = (
    "https://zenodo.org/api/records/15687512/files/{name}/content"
)


@dataclass(frozen=True)
class ChecksumResult:
    """Outcome of verifying one archive part against its published digest."""

    name: str
    present: bool
    expected_md5: str
    actual_md5: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.present and self.actual_md5 == self.expected_md5


def file_md5(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the MD5 of a file, streamed so large archives fit in memory."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive_checksums(directory: Path) -> List[ChecksumResult]:
    """Check every published archive part in ``directory``.

    Missing parts are reported rather than raised so a partial download
    can be inspected and resumed. Only a part whose digest matches the
    published value is treated as usable.
    """
    results: List[ChecksumResult] = []
    for name, expected in sorted(ARCHIVE_MD5.items()):
        path = Path(directory) / name
        if not path.is_file():
            results.append(ChecksumResult(name=name, present=False, expected_md5=expected))
            continue
        results.append(
            ChecksumResult(
                name=name,
                present=True,
                expected_md5=expected,
                actual_md5=file_md5(path),
            )
        )
    return results


@dataclass(frozen=True)
class SplitPolicy:
    """How the official split is used. We do not re-split randomly.

    The dataset authors allocated clips so that clips sharing a source
    video are not spread across splits, and the predefined split has been
    checked for leakage. A random re-split would reintroduce exactly that
    leakage, so the official split is preserved and the official held-out
    split is treated as the untouched test set.

    On disk that held-out split is the directory named ``val`` (see
    :data:`HELD_OUT_SPLIT_DIRECTORY`). Despite the name it is the official
    *test* set, not a tuning set: the validation set used for early
    stopping and threshold selection is carved out of ``train`` instead,
    grouped by source video.
    """

    use_official_split: bool = True
    official_test_is_untouched: bool = True
    held_out_split_directory: str = HELD_OUT_SPLIT_DIRECTORY
    validation_carved_from: str = "train"
    validation_fraction: float = 0.15
    validation_grouped_by_source_video: bool = True
    random_resplit_allowed: bool = False


@dataclass(frozen=True)
class SamplingConfig:
    """Temporal sampling from a 150-frame source clip to a model clip.

    Sixteen *consecutive* frames would cover only ~0.53s of a 5s clip and
    can miss the event entirely, so frames are sampled uniformly across
    the whole clip instead.
    """

    clip_length: int = 16
    source_frames_per_clip: int = EXPECTED_FRAMES_PER_CLIP
    strategy: str = "uniform_across_clip"
    train_temporal_jitter: bool = True
    eval_deterministic: bool = True

    @property
    def approximate_stride(self) -> float:
        """Frames skipped between sampled frames."""
        return self.source_frames_per_clip / self.clip_length

    @property
    def effective_fps(self) -> float:
        """Effective sampling rate in frames per second of source video."""
        return EXPECTED_FPS / self.approximate_stride


@dataclass(frozen=True)
class InitialTrainingConfig:
    """INITIAL EXPERIMENTAL SETTINGS -- not tuned, not validated.

    Reported results must state that these were the starting values and
    describe any subsequent tuning. No value here is an empirical result.
    """

    seed: int = 42
    num_classes: int = 2
    pretrained_backbone: bool = True
    frozen_modules: Tuple[str, ...] = ("stem", "layer1", "layer2")
    trainable_modules: Tuple[str, ...] = ("layer3", "layer4", "fc")
    optimizer: str = "AdamW"
    learning_rate_head: float = 1e-3
    learning_rate_backbone: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 30
    scheduler: str = "cosine_annealing"
    early_stopping_metric: str = "val_roc_auc"
    early_stopping_patience: int = 5
    # RWF-2000 is balanced 1000/1000, so no re-weighting is expected.
    # Confirm against observed counts before relying on this.
    class_weighting: Optional[str] = None
    augmentations: Tuple[str, ...] = (
        "random_spatial_crop",
        "horizontal_flip",
        "temporal_jitter",
        "brightness_contrast_jitter",
    )
    # Still unset here on purpose. This dataclass describes the settings a
    # run STARTS from, and the threshold is an output of evaluation, not an
    # input to training. The selected value lives in
    # SELECTED_DECISION_THRESHOLD below, with its provenance attached.
    decision_threshold: Optional[float] = None
    threshold_selection_criterion: str = "max_recall_at_fixed_accuracy"
    evaluation_metrics: Tuple[str, ...] = (
        "roc_auc",
        "confusion_matrix",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "accuracy",
    )


@dataclass(frozen=True)
class ReproducibilityRecord:
    """Everything needed to reproduce or cite one experiment."""

    dataset_source: str = ZENODO_RECORD_URL
    dataset_version_doi: str = ZENODO_VERSION_DOI
    dataset_license: str = ZENODO_LICENSE
    intended_use: str = INTENDED_USE
    model_architecture: str = "torchvision r3d_18 (VideoResNet), 33,371,472 params"
    pretrained_weights: str = "R3D_18_Weights.KINETICS400_V1"
    preprocessing: str = (
        "BGR->RGB, resize (128,171) bilinear antialias=False, crop 112x112, "
        "scale to [0,1], normalize mean=(0.43216,0.394666,0.37645) "
        "std=(0.22803,0.22145,0.216989)"
    )
    split: SplitPolicy = field(default_factory=SplitPolicy)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: InitialTrainingConfig = field(default_factory=InitialTrainingConfig)

    def as_dict(self) -> dict:
        """Return a JSON-native representation for saving alongside results."""
        return asdict(self)
