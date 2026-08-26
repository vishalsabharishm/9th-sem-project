"""Readiness validation for an extracted RWF-2000 tree.

Answers one question: is this directory safe to build the temporal
violence branch on? It composes the existing inspection utilities rather
than re-probing videos itself:

    dataset_inspection.inspect_dataset          per-clip metadata probe
    dataset_inspection.find_confirmed_...       content-verified leakage
    rwf2000_splits.build_evaluation_splits      400 / 6 / 394 partition

Every check reports rather than raises, so one failure never hides the
rest. ``ValidationReport.ok`` is true only when every check passed.

The class checks are layout-agnostic: labels are resolved with
``normalize_label``, so both a plain ``Fight`` directory and RWF-2000's
split-prefixed ``Train_Fight`` satisfy them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from dataset_inspection import (
        DatasetInspectionError,
        DatasetSummary,
        VideoRecord,
        discover_layout,
        find_confirmed_cross_split_duplicates,
        find_cross_split_name_collisions,
        inspect_dataset,
        normalize_label,
    )
    from rwf2000_config import (
        CANONICAL_LABELS,
        EXPECTED_FPS,
        EXPECTED_FRAMES_PER_CLIP,
        EXPECTED_LABEL_COUNTS,
        EXPECTED_SPLIT_COUNTS,
        EXPECTED_TOTAL_CLIPS,
        LEAKED_VALIDATION_CLIPS,
    )
    from rwf2000_splits import build_evaluation_splits, verify_recorded_leakage
except ImportError:  # pragma: no cover - supports package execution
    from src.dataset_inspection import (
        DatasetInspectionError,
        DatasetSummary,
        VideoRecord,
        discover_layout,
        find_confirmed_cross_split_duplicates,
        find_cross_split_name_collisions,
        inspect_dataset,
        normalize_label,
    )
    from src.rwf2000_config import (
        CANONICAL_LABELS,
        EXPECTED_FPS,
        EXPECTED_FRAMES_PER_CLIP,
        EXPECTED_LABEL_COUNTS,
        EXPECTED_SPLIT_COUNTS,
        EXPECTED_TOTAL_CLIPS,
        LEAKED_VALIDATION_CLIPS,
    )
    from src.rwf2000_splits import build_evaluation_splits, verify_recorded_leakage


@dataclass(frozen=True)
class Check:
    """One named pass/fail observation."""

    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict:
        """Return a JSON-native representation."""
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    """The outcome of validating one dataset root."""

    root: str
    checks: List[Check] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True only when every check passed."""
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> List[Check]:
        """The checks that did not pass."""
        return [check for check in self.checks if not check.passed]

    def as_dict(self) -> dict:
        """Return a JSON-native representation."""
        return {
            "root": self.root,
            "ok": self.ok,
            "checks": [check.as_dict() for check in self.checks],
            "summary": self.summary,
        }

    def to_text(self) -> str:
        """Return a readable report suitable for a notebook cell."""
        lines = [f"Dataset validation: {self.root}", "-" * 60]
        for check in self.checks:
            lines.append(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
        lines.append("-" * 60)
        lines.append(f"RESULT: {'READY' if self.ok else 'NOT READY'}")
        return "\n".join(lines)


def _label_counts(records: Sequence[VideoRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1
    return counts


def validate_dataset(
    root: Path,
    check_leakage: bool = True,
    expect_rwf2000_splits: bool = True,
    summary: Optional[DatasetSummary] = None,
    records: Optional[Sequence[VideoRecord]] = None,
) -> ValidationReport:
    """Validate an extracted dataset tree and return a structured report.

    Pass an existing ``summary``/``records`` pair to reuse a probe that has
    already been done; otherwise the videos are probed here.
    """
    root_path = Path(root)
    checks: List[Check] = []

    if not root_path.is_dir():
        return ValidationReport(
            root=str(root_path),
            checks=[Check("dataset_root_exists", False, f"not a directory: {root_path}")],
        )
    checks.append(Check("dataset_root_exists", True, str(root_path)))

    try:
        layout = discover_layout(root_path)
    except DatasetInspectionError as error:
        return ValidationReport(
            root=str(root_path),
            checks=checks + [Check("layout_discoverable", False, str(error))],
        )
    checks.append(
        Check("layout_discoverable", bool(layout), f"splits={sorted(layout)}")
    )

    if summary is None or records is None:
        summary, records = inspect_dataset(root_path)

    # ---- classes present and non-empty -----------------------------------
    counts = _label_counts(records)
    for label in CANONICAL_LABELS:
        present = counts.get(label, 0) > 0
        checks.append(
            Check(
                f"class_{label}_present",
                present,
                f"{counts.get(label, 0)} clips" if present else "class missing or empty",
            )
        )

    empty_dirs = [
        f"{split}/{class_dir}"
        for split, class_dirs in layout.items()
        for class_dir in class_dirs
        if summary.class_directory_counts.get(f"{split}/{class_dir}", 0) == 0
    ]
    checks.append(
        Check(
            "no_empty_class_directories",
            not empty_dirs,
            "none" if not empty_dirs else f"empty: {empty_dirs}",
        )
    )

    unexpected_labels = sorted(set(counts) - set(CANONICAL_LABELS))
    checks.append(
        Check(
            "labels_map_to_binary_classes",
            not unexpected_labels,
            f"observed={sorted(counts)}"
            + (f" UNEXPECTED={unexpected_labels}" if unexpected_labels else ""),
        )
    )

    # ---- videos actually readable ----------------------------------------
    total = len(records)
    checks.append(
        Check(
            "all_videos_open_in_opencv",
            not summary.unreadable,
            f"{summary.readable_videos}/{total} readable"
            + (f"; unreadable={summary.unreadable[:5]}" if summary.unreadable else ""),
        )
    )
    checks.append(
        Check(
            "no_unexpected_file_formats",
            not summary.unexpected_formats,
            "none"
            if not summary.unexpected_formats
            else f"{len(summary.unexpected_formats)}: {summary.unexpected_formats[:5]}",
        )
    )

    readable = [record for record in records if record.readable]
    bad_frames = [r.relative_path for r in readable if not r.frame_count]
    bad_fps = [r.relative_path for r in readable if not r.fps or r.fps <= 0]
    bad_size = [
        r.relative_path
        for r in readable
        if not r.width or not r.height or r.width <= 0 or r.height <= 0
    ]
    checks.append(
        Check(
            "frame_counts_readable",
            not bad_frames,
            f"ok for {len(readable) - len(bad_frames)}/{len(readable)}"
            + (f"; bad={bad_frames[:5]}" if bad_frames else ""),
        )
    )
    checks.append(
        Check(
            "fps_readable",
            not bad_fps,
            f"ok for {len(readable) - len(bad_fps)}/{len(readable)}"
            + (f"; bad={bad_fps[:5]}" if bad_fps else ""),
        )
    )
    checks.append(
        Check(
            "resolution_readable",
            not bad_size,
            f"ok for {len(readable) - len(bad_size)}/{len(readable)}"
            + (f"; bad={bad_size[:5]}" if bad_size else ""),
        )
    )

    # ---- split integrity / leakage ---------------------------------------
    collisions = find_cross_split_name_collisions(records)
    checks.append(
        Check(
            "no_cross_split_filename_collisions",
            not collisions,
            "none" if not collisions else f"{len(collisions)}: {collisions[:5]}",
        )
    )

    if check_leakage:
        confirmed = find_confirmed_cross_split_duplicates(root_path, records)
        drift = verify_recorded_leakage(root_path, records, confirmed=confirmed)
        checks.append(
            Check(
                "leakage_matches_recorded_list",
                not drift,
                f"{len(confirmed)} confirmed duplicate pair(s), "
                f"{len(LEAKED_VALIDATION_CLIPS)} recorded"
                + (f"; drift={drift[:3]}" if drift else ""),
            )
        )

    if expect_rwf2000_splits:
        # Exact counts, so a tree that is merely "plausible" cannot pass.
        # A duplicated clip or a partial extraction changes these numbers
        # even when every other check still succeeds.
        checks.append(
            Check(
                "total_clip_count_matches_published",
                total == EXPECTED_TOTAL_CLIPS,
                f"observed {total}, expected {EXPECTED_TOTAL_CLIPS}",
            )
        )
        checks.append(
            Check(
                "label_counts_match_published",
                counts == EXPECTED_LABEL_COUNTS,
                f"observed {counts}, expected {EXPECTED_LABEL_COUNTS}",
            )
        )
        checks.append(
            Check(
                "split_counts_match_published",
                summary.split_counts == EXPECTED_SPLIT_COUNTS,
                f"observed {summary.split_counts}, expected {EXPECTED_SPLIT_COUNTS}",
            )
        )
        try:
            splits = build_evaluation_splits(root_path)
            checks.append(
                Check(
                    "evaluation_splits_partition_cleanly",
                    (
                        splits.primary_count
                        == splits.official_count - splits.excluded_count
                        and not set(splits.primary_evaluation)
                        & set(splits.leakage_excluded)
                    ),
                    f"official={splits.official_count} excluded={splits.excluded_count} "
                    f"primary={splits.primary_count}",
                )
            )
        except Exception as error:  # noqa: BLE001 - report, never abort
            checks.append(
                Check("evaluation_splits_partition_cleanly", False, str(error))
            )

    return ValidationReport(
        root=str(root_path),
        checks=checks,
        summary={
            "total_files": summary.total_files,
            "readable_videos": summary.readable_videos,
            "split_counts": summary.split_counts,
            "label_counts": summary.label_counts,
            "split_label_counts": summary.split_label_counts,
            "class_directory_counts": summary.class_directory_counts,
            "resolutions_top5": list(summary.resolution_counts.items())[:5],
            "fps_top5": list(summary.fps_counts.items())[:5],
            "frame_counts_top5": list(summary.frame_count_counts.items())[:5],
            "expected_fps": EXPECTED_FPS,
            "expected_frames_per_clip": EXPECTED_FRAMES_PER_CLIP,
        },
    )
