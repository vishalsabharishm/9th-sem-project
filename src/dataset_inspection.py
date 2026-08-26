"""Read-only inspection and metadata extraction for video-clip datasets.

This utility characterises a split/class organised video dataset (such as
RWF-2000) without decoding whole videos: it probes container metadata via
the existing :mod:`video_loader` helpers, so memory use stays flat
regardless of dataset size.

The module is dataset-agnostic. Split and class directory names are
*discovered* from the filesystem rather than assumed, so it does not bake
in any particular archive's layout. Nothing here copies, moves, or
modifies dataset files.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:
    from video_loader import (
        SUPPORTED_VIDEO_EXTENSIONS,
        VideoLoadError,
        long_path,
        open_video,
    )
except ImportError:  # pragma: no cover - supports package execution
    from src.video_loader import (
        SUPPORTED_VIDEO_EXTENSIONS,
        VideoLoadError,
        long_path,
        open_video,
    )

# ``long_path`` is re-exported so dataset tooling has one obvious import
# site for path handling; the implementation lives in ``video_loader``.
__all__ = [
    "DatasetInspectionError",
    "DatasetSummary",
    "VideoRecord",
    "confirm_duplicate_groups",
    "discover_layout",
    "find_confirmed_cross_split_duplicates",
    "find_cross_split_duplicates",
    "find_cross_split_name_collisions",
    "fingerprint",
    "inspect_dataset",
    "iter_dataset_files",
    "long_path",
    "normalize_label",
    "probe_video",
    "write_metadata",
]


# Only the leading bytes are hashed for duplicate detection. Reading whole
# files would mean streaming the entire dataset from disk for a check that
# is meant to be cheap; the size + head digest pair is enough to surface
# candidate duplicates for manual confirmation.
FINGERPRINT_BYTES = 1024 * 1024


class DatasetInspectionError(RuntimeError):
    """Raised when a dataset directory cannot be inspected."""


def normalize_label(class_directory: str, split: str) -> str:
    """Return the class label with a redundant split prefix removed.

    RWF-2000 names its class directories after the split containing them:
    ``train/Train_Fight`` and ``val/Val_Fight`` are the same ``Fight``
    class. Counting the raw directory names would report four classes
    instead of two and would never match the published per-label figures.

    The prefix is stripped only when it actually matches the containing
    split, so a class legitimately called ``Train_Something`` under a
    differently named split is left intact.
    """
    prefix = f"{split}_"
    if class_directory.lower().startswith(prefix.lower()):
        remainder = class_directory[len(prefix):]
        if remainder:
            return remainder
    return class_directory


@dataclass(frozen=True)
class VideoRecord:
    """Probed metadata for one dataset video."""

    relative_path: str
    split: str
    label: str
    size_bytes: int
    readable: bool
    # The raw directory name the clip came from, kept alongside the
    # normalized ``label`` so provenance stays auditable.
    class_directory: str = ""
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        """Return a JSON-native representation."""
        return asdict(self)


@dataclass(frozen=True)
class DatasetSummary:
    """Aggregate statistics for one inspected dataset root."""

    root: str
    total_files: int
    readable_videos: int
    split_counts: Dict[str, int] = field(default_factory=dict)
    label_counts: Dict[str, int] = field(default_factory=dict)
    split_label_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Raw ``<split>/<class dir>`` tallies, before split prefixes are
    # stripped. Kept so the normalization can be audited against disk.
    class_directory_counts: Dict[str, int] = field(default_factory=dict)
    resolution_counts: Dict[str, int] = field(default_factory=dict)
    fps_counts: Dict[str, int] = field(default_factory=dict)
    frame_count_counts: Dict[str, int] = field(default_factory=dict)
    duration_seconds_min: Optional[float] = None
    duration_seconds_max: Optional[float] = None
    duration_seconds_mean: Optional[float] = None
    unreadable: List[str] = field(default_factory=list)
    unexpected_formats: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Return a JSON-native representation."""
        return asdict(self)


def discover_layout(root: Path) -> Dict[str, List[str]]:
    """Return the ``{split: [class, ...]}`` layout found on disk.

    Directory names are read from the filesystem rather than assumed, so
    an archive using ``val`` rather than ``test`` (or different class
    spellings) is reported accurately instead of silently mismatched.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise DatasetInspectionError(f"Dataset root is not a directory: {root_path}")

    layout: Dict[str, List[str]] = {}
    for split_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        layout[split_dir.name] = classes
    return layout


def iter_dataset_files(root: Path) -> Iterator[Tuple[Path, str, str]]:
    """Yield ``(path, split, class_directory)`` for every file two levels down.

    ``os.path.isfile`` is applied to the extended-length form of the path
    so that clips whose names push them past MAX_PATH are still seen; the
    plain ``Path`` is what gets yielded, so callers keep readable paths.
    """
    root_path = Path(root)
    for split, class_dirs in discover_layout(root_path).items():
        for class_dir in class_dirs:
            for path in sorted((root_path / split / class_dir).iterdir()):
                if os.path.isfile(long_path(path)):
                    yield path, split, class_dir


def probe_video(path: Path, split: str, class_directory: str, root: Path) -> VideoRecord:
    """Probe one video's container metadata without decoding its frames."""
    file_path = Path(path)
    relative = str(file_path.relative_to(Path(root))).replace("\\", "/")
    label = normalize_label(class_directory, split)
    try:
        size_bytes = os.stat(long_path(file_path)).st_size
    except OSError:
        size_bytes = 0

    capture = None
    try:
        capture, metadata = open_video(file_path)
        return VideoRecord(
            relative_path=relative,
            split=split,
            label=label,
            class_directory=class_directory,
            size_bytes=size_bytes,
            readable=True,
            fps=round(float(metadata.fps), 4),
            frame_count=int(metadata.frame_count),
            width=int(metadata.width),
            height=int(metadata.height),
            duration_seconds=round(float(metadata.duration_seconds), 4),
        )
    except (VideoLoadError, Exception) as error:  # noqa: B014 - report, never abort
        return VideoRecord(
            relative_path=relative,
            split=split,
            label=label,
            class_directory=class_directory,
            size_bytes=size_bytes,
            readable=False,
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        if capture is not None:
            capture.release()


def inspect_dataset(root: Path) -> Tuple[DatasetSummary, List[VideoRecord]]:
    """Probe every file under ``root`` and return summary plus per-video records."""
    root_path = Path(root)
    records: List[VideoRecord] = []
    unexpected: List[str] = []

    for path, split, class_dir in iter_dataset_files(root_path):
        relative = str(path.relative_to(root_path)).replace("\\", "/")
        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            unexpected.append(relative)
            continue
        records.append(probe_video(path, split, class_dir, root_path))

    return _summarize(root_path, records, unexpected), records


def _summarize(
    root: Path,
    records: Sequence[VideoRecord],
    unexpected: Sequence[str],
) -> DatasetSummary:
    """Aggregate probed records into a reportable summary."""
    split_counts: Counter = Counter()
    label_counts: Counter = Counter()
    class_dir_counts: Counter = Counter()
    split_label: Dict[str, Counter] = defaultdict(Counter)
    resolutions: Counter = Counter()
    fps_values: Counter = Counter()
    frame_counts: Counter = Counter()
    durations: List[float] = []
    unreadable: List[str] = []

    for record in records:
        split_counts[record.split] += 1
        label_counts[record.label] += 1
        split_label[record.split][record.label] += 1
        if record.class_directory:
            class_dir_counts[f"{record.split}/{record.class_directory}"] += 1
        if not record.readable:
            unreadable.append(record.relative_path)
            continue
        resolutions[f"{record.width}x{record.height}"] += 1
        fps_values[f"{record.fps:.2f}"] += 1
        frame_counts[str(record.frame_count)] += 1
        if record.duration_seconds is not None:
            durations.append(record.duration_seconds)

    return DatasetSummary(
        root=str(root),
        total_files=len(records) + len(unexpected),
        readable_videos=sum(1 for record in records if record.readable),
        split_counts=dict(sorted(split_counts.items())),
        label_counts=dict(sorted(label_counts.items())),
        split_label_counts={
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_label.items())
        },
        class_directory_counts=dict(sorted(class_dir_counts.items())),
        resolution_counts=dict(resolutions.most_common()),
        fps_counts=dict(fps_values.most_common()),
        frame_count_counts=dict(frame_counts.most_common()),
        duration_seconds_min=min(durations) if durations else None,
        duration_seconds_max=max(durations) if durations else None,
        duration_seconds_mean=(
            round(sum(durations) / len(durations), 4) if durations else None
        ),
        unreadable=sorted(unreadable),
        unexpected_formats=sorted(unexpected),
    )


def fingerprint(path: Path, head_bytes: int = FINGERPRINT_BYTES) -> str:
    """Return a cheap ``size:md5-of-head`` content fingerprint."""
    target = long_path(Path(path))
    return f"{os.stat(target).st_size}:{_md5_of(path, limit=head_bytes)}"


def _md5_of(path: Path, limit: Optional[int] = None, chunk: int = 1024 * 1024) -> str:
    """Return the MD5 of ``path``, of its first ``limit`` bytes if given.

    One streaming implementation backs both the cheap head fingerprint and
    the full-content digest, so the two can never drift apart.
    """
    digest = hashlib.md5()
    remaining = limit
    with open(long_path(Path(path)), "rb") as handle:
        while remaining is None or remaining > 0:
            size = chunk if remaining is None else min(chunk, remaining)
            block = handle.read(size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def content_digest(path: Path) -> str:
    """Return the MD5 of a file's entire contents.

    Used to promote a candidate duplicate -- found cheaply by
    :func:`fingerprint` -- into a confirmed byte-identical match.
    """
    return _md5_of(path, limit=None)


def find_cross_split_duplicates(
    root: Path,
    records: Sequence[VideoRecord],
) -> List[List[str]]:
    """Return groups of files sharing a fingerprint across different splits.

    Clips derived from one source video appearing in both training and
    test is the leakage mode that most inflates reported accuracy, so any
    group returned here must be investigated before training.
    """
    root_path = Path(root)
    by_fingerprint: Dict[str, List[VideoRecord]] = defaultdict(list)
    for record in records:
        if not record.readable:
            continue
        try:
            key = fingerprint(root_path / record.relative_path)
        except OSError:
            continue
        by_fingerprint[key].append(record)

    groups: List[List[str]] = []
    for matches in by_fingerprint.values():
        if len({record.split for record in matches}) > 1:
            groups.append(sorted(record.relative_path for record in matches))
    return sorted(groups)


def confirm_duplicate_groups(
    root: Path,
    groups: Sequence[Sequence[str]],
) -> List[Tuple[str, List[str]]]:
    """Keep only groups whose members are byte-identical, with their digest.

    :func:`find_cross_split_duplicates` compares sizes and the first
    megabyte, which is enough to *suspect* a duplicate but not to assert
    one -- two different clips sharing an encoder preamble would match.
    Each candidate is therefore re-hashed in full here, and a group is
    reported only when its files agree over every byte.

    Returns ``(md5, [relative paths])`` pairs sorted by digest.
    """
    root_path = Path(root)
    confirmed: List[Tuple[str, List[str]]] = []
    for group in groups:
        by_digest: Dict[str, List[str]] = defaultdict(list)
        for relative in group:
            try:
                by_digest[content_digest(root_path / relative)].append(relative)
            except OSError:
                continue
        for digest, members in by_digest.items():
            if len(members) > 1:
                confirmed.append((digest, sorted(members)))
    return sorted(confirmed)


def find_confirmed_cross_split_duplicates(
    root: Path,
    records: Sequence[VideoRecord],
) -> List[Tuple[str, List[str]]]:
    """Return cross-split duplicates confirmed by a full-content digest.

    This is the reproducible mechanism behind the recorded leakage list:
    a cheap fingerprint scan proposes candidates, and a full-file hash
    confirms them. Re-running it on a fresh copy of the dataset must
    reproduce the same set.
    """
    root_path = Path(root)
    candidates = find_cross_split_duplicates(root_path, records)
    splits = {record.relative_path: record.split for record in records}
    confirmed = confirm_duplicate_groups(root_path, candidates)
    # Confirmation can split a candidate group; re-check that what remains
    # still spans more than one split before calling it leakage.
    return [
        (digest, members)
        for digest, members in confirmed
        if len({splits.get(member) for member in members}) > 1
    ]


def find_cross_split_name_collisions(records: Sequence[VideoRecord]) -> List[str]:
    """Return file stems that occur in more than one split."""
    stems: Dict[str, set] = defaultdict(set)
    for record in records:
        stems[Path(record.relative_path).stem].add(record.split)
    return sorted(stem for stem, splits in stems.items() if len(splits) > 1)


def write_metadata(
    summary: DatasetSummary,
    records: Sequence[VideoRecord],
    output_path: Path,
    leakage: Optional[dict] = None,
) -> Path:
    """Write summary plus per-video records as one JSON document.

    ``leakage`` is an optional block describing confirmed cross-split
    duplicates and the evaluation sets derived from them; it is recorded
    alongside the statistics so a published result can point at exactly
    which clips were excluded and why.

    Only metadata is written; dataset videos are never copied or moved.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "summary": summary.as_dict(),
        "videos": [record.as_dict() for record in records],
    }
    if leakage is not None:
        document["leakage"] = leakage
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path
