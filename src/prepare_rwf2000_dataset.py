"""One-command RWF-2000 acquisition, verification, and inspection.

Runs the full dataset-preparation sequence:

    download -> verify MD5 -> extract -> discover layout -> inspect
             -> leakage checks -> write metadata

It is designed for the cloud-GPU environment (Kaggle/Colab) but runs
unchanged locally. Every stage is idempotent: parts whose published MD5
already matches are not re-downloaded, and extraction is skipped when the
target already holds videos. Interrupting and re-running is safe.

This script never trains, never downloads model weights, and never
re-splits the dataset. It reports what it observes; expected values from
the literature live in ``rwf2000_config`` and are only ever compared
against observations, never substituted for them.

Usage::

    python src/prepare_rwf2000_dataset.py --root data/rwf2000
    python src/prepare_rwf2000_dataset.py --root /kaggle/temp/rwf2000 \
        --archives /kaggle/temp/rwf2000_archives
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

try:
    from dataset_inspection import (
        discover_layout,
        find_confirmed_cross_split_duplicates,
        find_cross_split_name_collisions,
        inspect_dataset,
        long_path,
        normalize_label,
        write_metadata,
    )
    from rwf2000_config import (
        ARCHIVE_MD5,
        EXPECTED_LABEL_COUNTS,
        EXPECTED_SPLIT_COUNTS,
        EXPECTED_TOTAL_CLIPS,
        LEAKED_VALIDATION_CLIPS,
        ZENODO_FILE_URL,
        ZENODO_RECORD_URL,
        ZENODO_VERSION_DOI,
        ReproducibilityRecord,
        verify_archive_checksums,
    )
    from rwf2000_splits import build_evaluation_splits, verify_recorded_leakage
except ImportError:  # pragma: no cover - supports package execution
    from src.dataset_inspection import (
        discover_layout,
        find_confirmed_cross_split_duplicates,
        find_cross_split_name_collisions,
        inspect_dataset,
        long_path,
        normalize_label,
        write_metadata,
    )
    from src.rwf2000_config import (
        ARCHIVE_MD5,
        EXPECTED_LABEL_COUNTS,
        EXPECTED_SPLIT_COUNTS,
        EXPECTED_TOTAL_CLIPS,
        LEAKED_VALIDATION_CLIPS,
        ZENODO_FILE_URL,
        ZENODO_RECORD_URL,
        ZENODO_VERSION_DOI,
        ReproducibilityRecord,
        verify_archive_checksums,
    )
    from src.rwf2000_splits import build_evaluation_splits, verify_recorded_leakage


FIRST_PART = "RWF-2000.7z.001"


def _log(message: str) -> None:
    print(message, flush=True)


def find_7z() -> Optional[str]:
    """Locate a 7-Zip executable on Windows or Linux, or return None."""
    for candidate in ("7z", "7za", "7zr"):
        found = shutil.which(candidate)
        if found:
            return found
    windows_paths = (
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    )
    for path in windows_paths:
        if Path(path).is_file():
            return path
    return None


def download_archives(archive_dir: Path) -> None:
    """Download any part that is missing or fails its published checksum."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    for result in verify_archive_checksums(archive_dir):
        if result.ok:
            _log(f"  {result.name}: already verified, skipping")
            continue
        destination = archive_dir / result.name
        if destination.exists():
            _log(f"  {result.name}: present but unverified, re-downloading")
            destination.unlink()
        url = ZENODO_FILE_URL.format(name=result.name)
        _log(f"  {result.name}: downloading...")
        urllib.request.urlretrieve(url, destination)


def verify_archives(archive_dir: Path) -> bool:
    """Verify all 13 parts, printing a per-part result. Returns overall pass."""
    all_ok = True
    for result in verify_archive_checksums(archive_dir):
        if result.ok:
            status = "OK"
        elif not result.present:
            status = "MISSING"
        else:
            status = f"MISMATCH (got {result.actual_md5})"
        if not result.ok:
            all_ok = False
        _log(f"  {result.name}: {status}")
    return all_ok


def extract_archives(archive_dir: Path, target: Path) -> None:
    """Extract the multi-part archive, starting from part .001."""
    target.mkdir(parents=True, exist_ok=True)
    if any(target.rglob("*.avi")) or any(target.rglob("*.mp4")):
        _log("  target already contains videos, skipping extraction")
        return

    executable = find_7z()
    if executable is None:
        raise RuntimeError(
            "No 7-Zip executable found. Install it first "
            "(Linux: apt-get install -y p7zip-full; Windows: 7-Zip)."
        )
    first = archive_dir / FIRST_PART
    if not first.is_file():
        raise RuntimeError(f"Missing first archive part: {first}")

    _log(f"  extracting with {executable} ...")
    completed = subprocess.run(
        [executable, "x", str(first), f"-o{target}", "-y"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Extraction failed: {completed.stderr[-2000:]}")


def _looks_like_dataset_root(path: Path) -> bool:
    """True when ``path`` holds split/class directories containing files.

    That is, ``path/<split>/<class>/<file>`` exists for some split and
    class -- exactly the two-level layout ``dataset_inspection`` expects.
    """
    for split in (p for p in path.iterdir() if p.is_dir()):
        for label in (p for p in split.iterdir() if p.is_dir()):
            if any(os.path.isfile(long_path(child)) for child in label.iterdir()):
                return True
    return False


def resolve_dataset_root(target: Path) -> Path:
    """Return the directory whose children are the split folders.

    The archive may unpack into a nested folder (e.g. ``RWF-2000/``), so
    this descends through single-child wrappers until the split/class
    layout is found. The layout is discovered, never assumed.
    """
    candidate = Path(target)
    for _ in range(4):
        if _looks_like_dataset_root(candidate):
            return candidate
        subdirs = [p for p in candidate.iterdir() if p.is_dir()]
        if len(subdirs) != 1:
            break
        candidate = subdirs[0]
    return candidate


def compare_to_expected(summary) -> List[str]:
    """Report observed-vs-expected differences WITHOUT asserting either.

    Published figures are treated strictly as a cross-check. A mismatch is
    reported for human review; it never overrides what was observed.
    """
    notes: List[str] = []
    observed_total = summary.readable_videos
    if observed_total != EXPECTED_TOTAL_CLIPS:
        notes.append(
            f"observed {observed_total} readable videos vs published "
            f"{EXPECTED_TOTAL_CLIPS} -- investigate before training"
        )
    for label, expected in EXPECTED_LABEL_COUNTS.items():
        observed = summary.label_counts.get(label)
        if observed is None:
            notes.append(f"published label '{label}' not found; observed labels: "
                         f"{sorted(summary.label_counts)}")
        elif observed != expected:
            notes.append(f"label '{label}': observed {observed} vs published {expected}")
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        observed = summary.split_counts.get(split)
        if observed is None:
            notes.append(f"expected split '{split}' not found; observed splits: "
                         f"{sorted(summary.split_counts)}")
        elif observed != expected:
            notes.append(f"split '{split}': observed {observed} vs published {expected}")
    return notes


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Extraction target directory.")
    parser.add_argument("--archives", default=None, help="Archive download directory.")
    parser.add_argument("--metadata", default=None, help="Metadata output directory.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--skip-duplicate-check",
        action="store_true",
        help="Skip the cross-split fingerprint scan (it reads file heads).",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    archive_dir = Path(args.archives) if args.archives else root.parent / "rwf2000_archives"
    metadata_dir = Path(args.metadata) if args.metadata else root.parent / "rwf2000_metadata"

    _log("=" * 68)
    _log("RWF-2000 DATASET PREPARATION")
    _log(f"source : {ZENODO_RECORD_URL}")
    _log(f"DOI    : {ZENODO_VERSION_DOI}")
    _log("=" * 68)

    if not args.skip_download:
        _log(f"\n[1/6] Downloading {len(ARCHIVE_MD5)} archive parts -> {archive_dir}")
        download_archives(archive_dir)
    else:
        _log("\n[1/6] Download skipped by flag")

    _log(f"\n[2/6] Verifying published MD5 checksums")
    if not verify_archives(archive_dir):
        _log("\nFAILED: not all parts verified. Re-run to resume; nothing was extracted.")
        return 1
    _log("  all 13 parts verified")

    _log(f"\n[3/6] Extracting -> {root}")
    extract_archives(archive_dir, root)

    dataset_root = resolve_dataset_root(root)
    _log(f"\n[4/6] Discovering layout under {dataset_root}")
    layout = discover_layout(dataset_root)
    for split, class_dirs in layout.items():
        mapped = {name: normalize_label(name, split) for name in class_dirs}
        _log(f"  observed split '{split}': class dirs {class_dirs}")
        _log(f"    mapped to labels: {mapped}")

    _log("\n[5/6] Inspecting videos (metadata probe only)")
    summary, records = inspect_dataset(dataset_root)
    _log(f"  readable videos    : {summary.readable_videos}")
    _log(f"  split counts       : {summary.split_counts}")
    _log(f"  label counts       : {summary.label_counts}")
    _log(f"  per-split/class    : {summary.split_label_counts}")
    _log(f"  raw class dirs     : {summary.class_directory_counts}")
    _log(f"  resolutions (top 5): {list(summary.resolution_counts.items())[:5]}")
    _log(f"  fps (top 5)        : {list(summary.fps_counts.items())[:5]}")
    _log(f"  frame counts (top5): {list(summary.frame_count_counts.items())[:5]}")
    _log(
        f"  duration min/mean/max: {summary.duration_seconds_min} / "
        f"{summary.duration_seconds_mean} / {summary.duration_seconds_max}"
    )
    if summary.unreadable:
        _log(f"  UNREADABLE ({len(summary.unreadable)}): {summary.unreadable[:5]}")
    if summary.unexpected_formats:
        _log(f"  UNEXPECTED FORMATS: {summary.unexpected_formats[:5]}")

    for note in compare_to_expected(summary):
        _log(f"  NOTE: {note}")

    _log("\n[6/6] Leakage checks and metadata")
    collisions = find_cross_split_name_collisions(records)
    _log(f"  cross-split filename collisions: {len(collisions)}")
    if collisions:
        _log(f"    examples: {collisions[:5]}")

    leakage: Optional[dict] = None
    if args.skip_duplicate_check:
        _log("  cross-split duplicate scan: skipped by flag")
    else:
        confirmed = find_confirmed_cross_split_duplicates(dataset_root, records)
        _log(f"  confirmed cross-split duplicate pairs: {len(confirmed)}")
        for digest, members in confirmed:
            _log(f"    {digest}  {members}")

        drift = verify_recorded_leakage(dataset_root, records, confirmed=confirmed)
        if drift:
            for problem in drift:
                _log(f"  LEAKAGE RECORD MISMATCH: {problem}")
        else:
            _log(f"  recorded leakage list matches the dataset "
                 f"({len(LEAKED_VALIDATION_CLIPS)} pairs)")

        splits = build_evaluation_splits(dataset_root)
        _log(f"  official validation clips : {splits.official_count}")
        _log(f"  leakage-excluded clips    : {splits.excluded_count}")
        _log(f"  primary evaluation clips  : {splits.primary_count}")
        for path in splits.leakage_excluded:
            _log(f"    excluded: {path}")

        leakage = {
            "confirmed_duplicate_pairs": [
                {"md5": digest, "clips": members} for digest, members in confirmed
            ],
            "recorded_list_matches_dataset": not drift,
            "record_mismatches": drift,
            "evaluation_splits": splits.as_dict(),
        }

    metadata_path = write_metadata(
        summary, records, metadata_dir / "rwf2000_metadata.json", leakage=leakage
    )
    record_path = metadata_dir / "reproducibility.json"
    record_path.write_text(
        json.dumps(ReproducibilityRecord().as_dict(), indent=2), encoding="utf-8"
    )
    _log(f"  metadata        -> {metadata_path}")
    _log(f"  reproducibility -> {record_path}")

    _log("\nDone. Official split preserved; nothing was moved or re-split.")
    _log("Review the observed statistics above before any training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
