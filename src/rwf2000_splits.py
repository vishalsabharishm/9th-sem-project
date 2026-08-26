"""Evaluation splits for RWF-2000, with confirmed leakage held out.

The official release ships 400 held-out clips in ``val/``. Six of them are
byte-identical copies of training clips released under different names, so
scoring them measures memorisation rather than generalisation. This module
turns that fact into three explicitly named sets:

``official_validation``
    All 400 clips exactly as released. Nothing is removed from disk.
``leakage_excluded``
    The 6 confirmed duplicates, listed so the exclusion is auditable.
``primary_evaluation``
    The remaining 394 clips -- the set headline metrics are reported on.

Nothing here deletes, moves, or edits a dataset file, and nothing
re-splits the dataset: the exclusion is applied when clips are selected
for evaluation, not on the filesystem. Training data is untouched.

Filesystem access goes through ``video_loader.long_path`` (re-exported by
``dataset_inspection``) so clips whose names exceed the Windows MAX_PATH
limit stay reachable, and so path handling lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from dataset_inspection import (
        VideoRecord,
        find_confirmed_cross_split_duplicates,
        inspect_dataset,
        iter_dataset_files,
        long_path,
    )
    from rwf2000_config import (
        HELD_OUT_SPLIT_DIRECTORY,
        LEAKED_VALIDATION_CLIPS,
        OFFICIAL_VALIDATION_CLIP_COUNT,
        PRIMARY_EVALUATION_CLIP_COUNT,
        leakage_excluded_paths,
    )
    from video_loader import SUPPORTED_VIDEO_EXTENSIONS
except ImportError:  # pragma: no cover - supports package execution
    from src.dataset_inspection import (
        VideoRecord,
        find_confirmed_cross_split_duplicates,
        inspect_dataset,
        iter_dataset_files,
        long_path,
    )
    from src.rwf2000_config import (
        HELD_OUT_SPLIT_DIRECTORY,
        LEAKED_VALIDATION_CLIPS,
        OFFICIAL_VALIDATION_CLIP_COUNT,
        PRIMARY_EVALUATION_CLIP_COUNT,
        leakage_excluded_paths,
    )
    from src.video_loader import SUPPORTED_VIDEO_EXTENSIONS


class LeakageExclusionError(RuntimeError):
    """Raised when the recorded leakage list does not match the dataset."""


@dataclass(frozen=True)
class EvaluationSplits:
    """The three evaluation sets, as relative paths under the dataset root."""

    root: str
    official_validation: Tuple[str, ...] = ()
    leakage_excluded: Tuple[str, ...] = ()
    primary_evaluation: Tuple[str, ...] = ()

    @property
    def official_count(self) -> int:
        """Size of the untouched official held-out split."""
        return len(self.official_validation)

    @property
    def excluded_count(self) -> int:
        """Number of clips withheld as confirmed leakage."""
        return len(self.leakage_excluded)

    @property
    def primary_count(self) -> int:
        """Size of the clean set headline metrics are reported on."""
        return len(self.primary_evaluation)

    def as_dict(self) -> dict:
        """Return a JSON-native representation for the metadata document."""
        return {
            "root": self.root,
            "counts": {
                "official_validation": self.official_count,
                "leakage_excluded": self.excluded_count,
                "primary_evaluation": self.primary_count,
            },
            "official_validation": list(self.official_validation),
            "leakage_excluded": list(self.leakage_excluded),
            "primary_evaluation": list(self.primary_evaluation),
            "leaked_pairs": [clip.as_dict() for clip in LEAKED_VALIDATION_CLIPS],
        }


def clip_path(root: Path, relative_path: str) -> str:
    """Return a filesystem path for one clip that is safe to open.

    Evaluation and future data-loading code should read clips through
    this rather than joining paths themselves: it applies the shared
    ``long_path`` rule, without which the longest-named clips cannot be
    opened on Windows.
    """
    return long_path(Path(root) / relative_path)


def split_clip_paths(root: Path, split: str) -> Tuple[str, ...]:
    """Return every video in one split directory, as sorted relative paths.

    Read straight from disk rather than from a recorded list, so a split
    is whatever the release actually contains.
    """
    root_path = Path(root)
    found: List[str] = []
    for path, split_name, _class_directory in iter_dataset_files(root_path):
        if split_name != split:
            continue
        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        found.append(str(path.relative_to(root_path)).replace("\\", "/"))
    return tuple(sorted(found))


def held_out_clip_paths(
    root: Path,
    split: str = HELD_OUT_SPLIT_DIRECTORY,
) -> Tuple[str, ...]:
    """Return every video in the official held-out split."""
    return split_clip_paths(root, split)


def build_evaluation_splits(
    root: Path,
    strict: bool = True,
) -> EvaluationSplits:
    """Partition the official held-out split into excluded and primary sets.

    The official split is enumerated from disk and then partitioned; no
    clip is moved, copied, or deleted, and no new split is invented. With
    ``strict`` set, a recorded leaked clip that is not present on disk is
    an error rather than a silent no-op -- otherwise a typo or a renamed
    release would quietly stop excluding anything.
    """
    root_path = Path(root)
    official = held_out_clip_paths(root_path)
    recorded = leakage_excluded_paths()

    official_set = set(official)
    missing = sorted(recorded - official_set)
    if missing and strict:
        raise LeakageExclusionError(
            "Recorded leaked clips are absent from the held-out split: "
            f"{missing}. The recorded list and the dataset disagree; "
            "re-confirm with find_confirmed_cross_split_duplicates."
        )

    excluded = tuple(sorted(recorded & official_set))
    primary = tuple(path for path in official if path not in recorded)
    return EvaluationSplits(
        root=str(root_path),
        official_validation=official,
        leakage_excluded=excluded,
        primary_evaluation=primary,
    )


def expected_split_sizes() -> Dict[str, int]:
    """Return the documented sizes of the three evaluation sets."""
    return {
        "official_validation": OFFICIAL_VALIDATION_CLIP_COUNT,
        "leakage_excluded": len(LEAKED_VALIDATION_CLIPS),
        "primary_evaluation": PRIMARY_EVALUATION_CLIP_COUNT,
    }


def verify_recorded_leakage(
    root: Path,
    records: Optional[Sequence[VideoRecord]] = None,
    confirmed: Optional[Sequence[Tuple[str, List[str]]]] = None,
) -> List[str]:
    """Re-derive leakage from the dataset and diff it against the record.

    This is the reproducibility check behind
    :data:`rwf2000_config.LEAKED_VALIDATION_CLIPS`: it re-runs the
    fingerprint-then-full-hash scan and reports any disagreement. An empty
    list means the recorded exclusions still exactly describe the data.

    Passing ``records`` avoids re-probing every clip, and passing an
    already-computed ``confirmed`` list avoids repeating the duplicate
    scan itself.
    """
    root_path = Path(root)
    if confirmed is None:
        if records is None:
            _summary, records = inspect_dataset(root_path)
        confirmed = find_confirmed_cross_split_duplicates(root_path, records)
    observed_pairs = {
        tuple(sorted(members)): digest for digest, members in confirmed
    }
    recorded_pairs = {
        tuple(sorted((clip.training_clip, clip.validation_clip))): clip.md5
        for clip in LEAKED_VALIDATION_CLIPS
    }

    problems: List[str] = []
    for members in sorted(set(observed_pairs) - set(recorded_pairs)):
        problems.append(f"unrecorded duplicate group on disk: {list(members)}")
    for members in sorted(set(recorded_pairs) - set(observed_pairs)):
        problems.append(f"recorded duplicate no longer found on disk: {list(members)}")
    for members in sorted(set(observed_pairs) & set(recorded_pairs)):
        if observed_pairs[members] != recorded_pairs[members]:
            problems.append(
                f"digest changed for {list(members)}: recorded "
                f"{recorded_pairs[members]}, observed {observed_pairs[members]}"
            )
    return problems
