r"""Kaggle-side acquisition of RWF-2000 from the verified multi-part archive.

WHY THIS EXISTS
---------------
A plain ``7z x`` of this archive cannot succeed on Linux. Twenty-four
``train/Train_Fight`` clips carry CJK filenames of 256-508 UTF-8 bytes,
and every Linux filesystem caps a single path component at 255 *bytes*
(``NAME_MAX``). Those names cannot be written, so 7-Zip reports::

    ERROR: Can not open output file : File name too long

This is not the Windows ``MAX_PATH`` problem that ``video_loader.long_path``
solves. ``MAX_PATH`` limits the *total path* to 260 characters and is
escapable with a ``\\?\`` prefix; ``NAME_MAX`` limits one *component* to
255 bytes and has no escape on any Linux filesystem. The only correct fix
is to write those clips under different names.

STRATEGY
--------
1. Discover and verify the archive parts (all 13 must sit in one
   directory -- a multi-volume 7z aborts mid-stream otherwise, which is
   the usual cause of a partial extraction).
2. Read the archive's own file table, and predict up front exactly which
   entries cannot be written, from their UTF-8 byte length.
3. Bulk-extract with ``7z x``. The over-long entries fail; that is
   expected and tolerated rather than treated as a fatal error.
4. Re-extract only the over-long entries under deterministic sanitised
   names, in a single additional pass.
5. Reconcile disk against the archive table and report anything missing.

Renaming is safe for this project: labels come from the *directory* name,
never the filename (see ``rwf2000_dataset.label_index_for``), and the
recorded leakage exclusions are all short ASCII names in ``val/``, so no
recorded path is ever renamed. A manifest records every rename, and is
written outside the dataset tree so it cannot be mistaken for a clip.

Nothing here trains, and nothing re-downloads the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dataset_inspection import long_path
    from prepare_rwf2000_dataset import find_7z, resolve_dataset_root
    from rwf2000_config import ARCHIVE_MD5, ARCHIVE_TOTAL_BYTES, file_md5
except ImportError:  # pragma: no cover - supports package execution
    from src.dataset_inspection import long_path
    from src.prepare_rwf2000_dataset import find_7z, resolve_dataset_root
    from src.rwf2000_config import ARCHIVE_MD5, ARCHIVE_TOTAL_BYTES, file_md5


# One path component may not exceed this many bytes on Linux filesystems.
LINUX_NAME_MAX_BYTES = 255
FIRST_PART_NAME = "RWF-2000.7z.001"
DEFAULT_SEARCH_ROOTS = ("/kaggle/input", "data")
MANIFEST_NAME = "renamed_clips.json"
# Upper bound on how much recovered clip data is buffered at once.
MAX_STREAM_BYTES = 512 * 1024 * 1024


class KagglePreparationError(RuntimeError):
    """Raised when the archive cannot be located, verified, or extracted."""


@dataclass(frozen=True)
class ArchiveEntry:
    """One file recorded in the archive's table of contents."""

    path: str          # archive-relative, always forward-slashed
    size: int
    crc: str = ""

    @property
    def basename(self) -> str:
        """Final path component."""
        return self.path.rsplit("/", 1)[-1]

    @property
    def name_bytes(self) -> int:
        """UTF-8 byte length of the basename, which is what NAME_MAX caps."""
        return len(self.basename.encode("utf-8"))

    def exceeds_name_limit(self, limit: int = LINUX_NAME_MAX_BYTES) -> bool:
        """Whether this entry's name cannot be written on this filesystem."""
        return self.name_bytes > limit

    def as_dict(self) -> dict:
        """Return a JSON-native representation."""
        return asdict(self)


def _log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------
# 1. Locating and verifying the archive parts
# --------------------------------------------------------------------------


def find_archive_parts(
    search_roots: Sequence[str] = DEFAULT_SEARCH_ROOTS,
) -> List[Path]:
    """Return every ``RWF-2000.7z.0NN`` part found, sorted by part number.

    Kaggle mounts a dataset at an unpredictable depth under
    ``/kaggle/input``, so the parts are searched for rather than assumed
    to be at a fixed path.
    """
    found: Dict[str, Path] = {}
    for root in search_roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for candidate in base.rglob("RWF-2000.7z.0*"):
            if candidate.is_file():
                found.setdefault(candidate.name, candidate)
    return [found[name] for name in sorted(found)]


def verify_parts(
    parts: Sequence[Path],
    check_md5: bool = True,
) -> Tuple[bool, List[dict]]:
    """Check part completeness and, optionally, published MD5 digests.

    An incomplete set is the most common cause of a silently truncated
    extraction: 7-Zip streams volumes in order and stops when the next one
    is absent, having already written thousands of files.
    """
    by_name = {part.name: part for part in parts}
    rows: List[dict] = []
    all_ok = True

    for name, expected in sorted(ARCHIVE_MD5.items()):
        part = by_name.get(name)
        if part is None:
            rows.append({"part": name, "status": "MISSING", "path": None})
            all_ok = False
            continue
        row = {"part": name, "path": str(part), "size": part.stat().st_size}
        if check_md5:
            actual = file_md5(part)
            row["md5"] = actual
            row["status"] = "OK" if actual == expected else "MD5_MISMATCH"
        else:
            row["status"] = "PRESENT"
        if row["status"] not in ("OK", "PRESENT"):
            all_ok = False
        rows.append(row)

    extra = sorted(set(by_name) - set(ARCHIVE_MD5))
    for name in extra:
        rows.append({"part": name, "status": "UNEXPECTED", "path": str(by_name[name])})
    return all_ok, rows


# --------------------------------------------------------------------------
# 2. Reading the archive's table of contents
# --------------------------------------------------------------------------


def list_archive_entries(
    first_part: Path,
    seven_zip: Optional[str] = None,
) -> List[ArchiveEntry]:
    """Return the archive's file table without extracting anything.

    ``-sccUTF-8`` forces UTF-8 console output; without it 7-Zip emits the
    console codepage and non-ASCII names come back mojibake, which would
    make every later comparison wrong.
    """
    executable = seven_zip or find_7z()
    if executable is None:
        raise KagglePreparationError(
            "No 7-Zip executable found. Install it first "
            "(Linux: apt-get install -y p7zip-full)."
        )
    completed = subprocess.run(
        [executable, "l", str(first_part), "-ba", "-slt", "-sccUTF-8"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise KagglePreparationError(
            f"Could not list archive: {completed.stderr[-2000:]}"
        )
    return parse_archive_listing(completed.stdout)


def parse_archive_listing(text: str) -> List[ArchiveEntry]:
    """Parse ``7z l -slt`` output into file entries, skipping directories."""
    entries: List[ArchiveEntry] = []
    current: Dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("Path = "):
            if current:
                _append_entry(entries, current)
            current = {"Path": line[len("Path = "):]}
        elif " = " in line and current:
            key, value = line.split(" = ", 1)
            current[key.strip()] = value.strip()
    if current:
        _append_entry(entries, current)
    return entries


def _append_entry(entries: List[ArchiveEntry], record: Dict[str, str]) -> None:
    if "D" in record.get("Attributes", ""):
        return
    path = record.get("Path", "").replace("\\", "/")
    if not path:
        return
    try:
        size = int(record.get("Size", "0"))
    except ValueError:
        size = 0
    entries.append(ArchiveEntry(path=path, size=size, crc=record.get("CRC", "")))


def oversized_entries(
    entries: Sequence[ArchiveEntry],
    limit: int = LINUX_NAME_MAX_BYTES,
) -> List[ArchiveEntry]:
    """Return entries whose basename cannot be written on this filesystem."""
    return [entry for entry in entries if entry.exceeds_name_limit(limit)]


def entries_absent_from_disk(
    entries: Sequence[ArchiveEntry],
    target: Path,
    manifest_rows: Sequence[dict] = (),
) -> List[ArchiveEntry]:
    """Return entries that the bulk pass did not land correctly.

    Recovery is driven by what is *actually* absent rather than by the
    name-length prediction alone. On a filesystem that happens to accept
    the long names, everything lands in the bulk pass and this returns
    nothing -- so no clip is ever written twice, under both its original
    and a sanitised name, which would inflate the class counts.

    A present-but-wrong-size file counts as absent, so a half-written
    clip from an interrupted run is re-fetched rather than trusted.
    """
    root = Path(target)
    already = {row["original_archive_path"] for row in manifest_rows}
    absent: List[ArchiveEntry] = []
    for entry in entries:
        candidate = root / entry.path
        try:
            size = os.stat(long_path(candidate)).st_size
        except OSError:
            size = -1
        if size == entry.size and entry.size >= 0:
            continue
        if entry.path in already:
            renamed = next(
                row for row in manifest_rows
                if row["original_archive_path"] == entry.path
            )
            alias = root / renamed["written_relative_path"]
            try:
                if os.stat(long_path(alias)).st_size == entry.size:
                    continue
            except OSError:
                pass
        absent.append(entry)
    return absent


# --------------------------------------------------------------------------
# 3. Deterministic safe names
# --------------------------------------------------------------------------


def safe_relative_path(original: str, keep_ascii: int = 60) -> str:
    """Return a short, portable, collision-free path for one entry.

    The original path is hashed so the mapping is deterministic and
    reversible through the manifest, and any ASCII fragment of the stem is
    kept as a readability aid. Names that are entirely non-ASCII collapse
    to ``clip__<hash>``; the directory -- which carries the class label --
    is always preserved untouched.
    """
    directory, _, base = original.rpartition("/")
    stem, dot, extension = base.rpartition(".")
    if not dot:
        stem, extension = base, ""

    digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:12]
    readable = "".join(
        character
        for character in stem
        if character.isascii() and (character.isalnum() or character in "-_")
    )[:keep_ascii]
    name = f"{readable or 'clip'}__{digest}"
    if extension:
        name = f"{name}.{extension}"
    return f"{directory}/{name}" if directory else name


# --------------------------------------------------------------------------
# 4. Extraction
# --------------------------------------------------------------------------


def extract_bulk(
    first_part: Path,
    target: Path,
    seven_zip: Optional[str] = None,
) -> Tuple[int, str]:
    """Run the single bulk ``7z x`` pass.

    A non-zero exit is returned rather than raised: the over-long entries
    are *expected* to fail here and are recovered by
    :func:`extract_oversized`. Genuine failures are distinguished later by
    reconciling against the archive table.
    """
    executable = seven_zip or find_7z()
    if executable is None:
        raise KagglePreparationError("No 7-Zip executable found.")
    target.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [executable, "x", str(first_part), f"-o{target}", "-y"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def extract_oversized(
    first_part: Path,
    entries: Sequence[ArchiveEntry],
    target: Path,
    seven_zip: Optional[str] = None,
    log=_log,
) -> List[dict]:
    """Write the over-long entries under sanitised names, using only 7-Zip.

    No third-party Python package is involved, because a Kaggle session
    frequently has no route to PyPI.

    The fast path asks 7-Zip for all of the entries in ONE ``x -so`` call.
    7-Zip emits them concatenated in archive order, and the table of
    contents gives the exact byte length of each, so the stream can be
    split back into individual files. Every piece is then checked against
    the CRC32 recorded in the archive, which makes the split
    self-verifying: if the ordering assumption were ever wrong, the CRCs
    would not match and the slower per-entry path takes over.

    This matters because the archive is solid: a per-entry extraction has
    to re-decode the whole block each time, while one batched call decodes
    it once.
    """
    if not entries:
        return []
    try:
        rows: List[dict] = []
        for batch in _memory_bounded_batches(entries):
            rows.extend(
                _extract_oversized_batched(first_part, batch, target, seven_zip, log)
            )
        if len(rows) == len(entries):
            return rows
        log(f"  batched pass recovered {len(rows)}/{len(entries)}; retrying per entry")
    except Exception as error:  # noqa: BLE001 - any failure falls back
        log(f"  batched pass failed ({type(error).__name__}: {error});"
            " falling back to per-entry extraction")
    return _extract_oversized_seven_zip(first_part, entries, target, seven_zip, log)


def _memory_bounded_batches(
    entries: Sequence[ArchiveEntry],
    budget_bytes: int = MAX_STREAM_BYTES,
) -> List[List[ArchiveEntry]]:
    """Split entries into groups whose combined size fits in memory.

    The batched recovery buffers a whole group before writing it, so an
    unbounded group could exhaust a Kaggle session's RAM. The expected
    case -- 24 clips, ~247 MB -- stays a single group; a pathological
    case (a bulk pass that dropped hundreds of files) degrades into a few
    groups instead of an out-of-memory crash.
    """
    batches: List[List[ArchiveEntry]] = []
    current: List[ArchiveEntry] = []
    running = 0
    for entry in entries:
        if current and running + entry.size > budget_bytes:
            batches.append(current)
            current, running = [], 0
        current.append(entry)
        running += entry.size
    if current:
        batches.append(current)
    return batches


def _crc32(payload: bytes) -> str:
    """Return the uppercase CRC32 hex digest 7-Zip records for an entry."""
    return format(zlib.crc32(payload) & 0xFFFFFFFF, "08X")


def _extract_oversized_batched(
    first_part: Path,
    entries: Sequence[ArchiveEntry],
    target: Path,
    seven_zip: Optional[str],
    log,
) -> List[dict]:
    """Recover every over-long entry from a single concatenated 7-Zip stream."""
    executable = seven_zip or find_7z()
    if executable is None:
        raise KagglePreparationError("No 7-Zip executable found.")

    ordered = list(entries)
    expected_total = sum(entry.size for entry in ordered)
    log(
        f"  streaming {len(ordered)} over-long entries in one pass "
        f"({expected_total:,} bytes expected)"
    )

    # A list file avoids putting long non-ASCII names on the command line.
    handle, list_path = tempfile.mkstemp(suffix=".txt", text=False)
    os.close(handle)
    Path(list_path).write_text(
        "\n".join(entry.path for entry in ordered) + "\n", encoding="utf-8"
    )
    try:
        completed = subprocess.run(
            [
                executable, "x", "-so", str(first_part),
                f"-i@{list_path}", "-scsUTF-8", "-y",
            ],
            capture_output=True,
        )
    finally:
        os.unlink(list_path)

    stream = completed.stdout or b""
    if len(stream) != expected_total:
        raise KagglePreparationError(
            f"stream was {len(stream):,} bytes, expected {expected_total:,} "
            f"(7z exit={completed.returncode})"
        )

    rows: List[dict] = []
    offset = 0
    for entry in ordered:
        payload = stream[offset : offset + entry.size]
        offset += entry.size
        if entry.crc and _crc32(payload) != entry.crc.upper():
            raise KagglePreparationError(
                f"CRC mismatch while splitting the stream at {entry.basename[:40]}"
            )
        rows.append(_write_member(target, entry, payload))
    return rows


def _write_member(target: Path, entry: ArchiveEntry, payload: bytes) -> dict:
    """Write one recovered entry, renaming only when the name is illegal.

    An entry that is merely missing -- not too long -- keeps its original
    name, so the sanitised-name scheme is applied strictly where the
    filesystem forces it.
    """
    safe = (
        safe_relative_path(entry.path)
        if entry.exceeds_name_limit()
        else entry.path
    )
    destination = Path(target) / safe
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "original_archive_path": entry.path,
        "written_relative_path": safe,
        "original_name_bytes": entry.name_bytes,
        "expected_size": entry.size,
        "written_size": destination.stat().st_size,
        "archive_crc": entry.crc,
    }


def _extract_oversized_seven_zip(
    first_part: Path,
    entries: Sequence[ArchiveEntry],
    target: Path,
    seven_zip: Optional[str],
    log,
) -> List[dict]:
    """Recover the over-long entries one at a time.

    Correct but slow on a solid archive, since each call re-decodes the
    block. Used only when the batched stream could not be verified.
    """
    executable = seven_zip or find_7z()
    if executable is None:
        raise KagglePreparationError("No 7-Zip executable found.")
    rows: List[dict] = []
    for index, entry in enumerate(entries, 1):
        log(f"  [{index}/{len(entries)}] streaming {entry.basename[:36]}...")
        handle, list_path = tempfile.mkstemp(suffix=".txt")
        os.close(handle)
        Path(list_path).write_text(entry.path + "\n", encoding="utf-8")
        try:
            completed = subprocess.run(
                [
                    executable, "x", "-so", str(first_part),
                    f"-i@{list_path}", "-scsUTF-8", "-y",
                ],
                capture_output=True,
            )
        finally:
            os.unlink(list_path)

        payload = completed.stdout or b""
        if completed.returncode != 0 or len(payload) != entry.size:
            log(f"      FAILED (exit={completed.returncode}, "
                f"{len(payload)}/{entry.size} bytes)")
            continue
        if entry.crc and _crc32(payload) != entry.crc.upper():
            log("      FAILED (CRC mismatch)")
            continue
        rows.append(_write_member(target, entry, payload))
    return rows


# --------------------------------------------------------------------------
# 5. Reconciliation
# --------------------------------------------------------------------------


def reconcile(
    entries: Sequence[ArchiveEntry],
    dataset_root: Path,
    manifest_rows: Sequence[dict] = (),
) -> dict:
    """Compare what the archive holds against what reached the disk.

    Renamed clips are matched through the manifest, so a successful rename
    counts as present rather than as a missing original plus an unexpected
    extra.
    """
    root = Path(dataset_root)
    prefix = root.name
    on_disk = set()
    # ``os.walk`` plus a long-path-aware existence test, because a plain
    # ``Path.is_file()`` silently reports False for clips past the Windows
    # MAX_PATH limit -- the same trap this project already hit once.
    for directory, _subdirs, filenames in os.walk(root):
        for filename in filenames:
            candidate = Path(directory) / filename
            if os.path.isfile(long_path(candidate)):
                relative = str(candidate.relative_to(root)).replace("\\", "/")
                on_disk.add(f"{prefix}/{relative}")

    # ``written_relative_path`` is already archive-relative (it starts with
    # the same top-level folder as ``entry.path``), so it must NOT be
    # prefixed again -- doing so produced "RWF-2000/RWF-2000/..." and made
    # every recovered clip look missing.
    renamed = {
        row["original_archive_path"]: row["written_relative_path"]
        for row in manifest_rows
    }

    def size_of(archive_relative: str) -> int:
        local = root / archive_relative.split("/", 1)[1]
        try:
            return os.stat(long_path(local)).st_size
        except OSError:
            return -1

    missing, recovered, truncated = [], [], []
    for entry in entries:
        present = entry.path if entry.path in on_disk else renamed.get(entry.path)
        if not present or present not in on_disk:
            missing.append(entry.path)
            continue
        # An interrupted run can leave a half-written file behind, which
        # would otherwise pass a presence-only check.
        if entry.size and size_of(present) != entry.size:
            truncated.append(entry.path)
        elif present != entry.path:
            recovered.append(entry.path)

    return {
        "archive_files": len(entries),
        "files_on_disk": len(on_disk),
        "recovered_under_safe_names": len(recovered),
        "missing": sorted(missing),
        "missing_count": len(missing),
        "truncated": sorted(truncated),
        "truncated_count": len(truncated),
        "complete": not missing and not truncated,
        "unexpected_on_disk": sorted(
            set(on_disk) - {e.path for e in entries} - set(renamed.values())
        )[:20],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_manifest(metadata_dir: Path) -> List[dict]:
    """Return rename rows from a previous run, or an empty list."""
    path = Path(metadata_dir) / MANIFEST_NAME
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("renamed", [])
    except (ValueError, OSError):
        return []


def _already_complete(
    entries: Sequence[ArchiveEntry],
    target: Path,
    manifest_rows: Sequence[dict],
    log,
) -> bool:
    """Whether a previous run already produced a complete, correct tree.

    Makes re-running cheap and safe after an interrupted attempt: a
    complete tree is left alone, and an incomplete or truncated one is
    rebuilt rather than patched over.
    """
    if not Path(target).is_dir():
        return False
    try:
        root = resolve_dataset_root(Path(target))
        report = reconcile(entries, root, manifest_rows)
    except Exception:  # noqa: BLE001 - any problem means "not complete"
        return False
    if report["complete"]:
        return True
    log(
        f"  previous output is incomplete "
        f"(missing={report['missing_count']}, truncated={report['truncated_count']})"
        " -- re-extracting"
    )
    return False


def prepare(
    target: Path,
    search_roots: Sequence[str] = DEFAULT_SEARCH_ROOTS,
    metadata_dir: Optional[Path] = None,
    check_md5: bool = True,
    skip_extraction: bool = False,
    fresh: bool = False,
    log=_log,
) -> dict:
    """Run the full Kaggle preparation and return a machine-readable report."""
    target = Path(target)
    metadata = Path(metadata_dir) if metadata_dir else target.parent / "rwf2000_metadata"

    log("=" * 68)
    log("RWF-2000 KAGGLE PREPARATION")
    log("=" * 68)

    log("\n[1/5] Locating archive parts")
    parts = find_archive_parts(search_roots)
    if not parts:
        raise KagglePreparationError(
            f"No RWF-2000.7z.0NN parts found under {list(search_roots)}. "
            "Attach the verified-archives dataset to the notebook."
        )
    log(f"  found {len(parts)} part(s) in {parts[0].parent}")

    ok, rows = verify_parts(parts, check_md5=check_md5)
    for row in rows:
        if row["status"] not in ("OK", "PRESENT"):
            log(f"  {row['part']}: {row['status']}")
    log(f"  parts verified: {ok} ({len(ARCHIVE_MD5)} expected)")
    if not ok:
        raise KagglePreparationError(
            "Archive parts are incomplete or corrupted. A multi-volume 7z "
            "stops mid-stream when a volume is missing, which produces "
            "exactly the partial extraction seen here. Re-upload all 13 parts."
        )

    first_part = next(p for p in parts if p.name == FIRST_PART_NAME)

    log("\n[2/5] Reading archive table of contents")
    entries = list_archive_entries(first_part)
    oversized = oversized_entries(entries)
    log(f"  archive contains {len(entries)} files ({ARCHIVE_TOTAL_BYTES:,} bytes)")
    log(
        f"  {len(oversized)} name(s) exceed the {LINUX_NAME_MAX_BYTES}-byte "
        "limit and will be written under safe names"
    )

    manifest_rows: List[dict] = _load_manifest(metadata)
    if skip_extraction:
        log("\n[3/5] Extraction skipped by flag")
    elif _already_complete(entries, target, manifest_rows, log):
        log("\n[3/5] Existing extraction is already complete -- nothing to redo")
    else:
        if fresh and target.exists():
            log(f"\n[3/5] --fresh: removing previous output at {target}")
            shutil.rmtree(target, ignore_errors=True)

        log(f"\n[3/5] Bulk extraction -> {target}")
        code, output = extract_bulk(first_part, target)
        name_errors = output.count("File name too long")
        log(f"  7z exit={code}, 'File name too long' errors={name_errors}")
        if code != 0 and name_errors == 0:
            raise KagglePreparationError(
                f"Extraction failed for a reason other than long names:\n"
                f"{output[-2000:]}"
            )

        log("\n[4/5] Recovering entries the bulk pass could not write")
        to_recover = entries_absent_from_disk(entries, target)
        log(
            f"  predicted over-limit: {len(oversized)}; "
            f"actually absent after bulk pass: {len(to_recover)}"
        )
        if to_recover:
            manifest_rows = extract_oversized(
                first_part, to_recover, target, log=log
            )
            log(f"  recovered {len(manifest_rows)}/{len(to_recover)}")
        else:
            log("  nothing to recover -- this filesystem accepted every name")
            manifest_rows = []

    dataset_root = resolve_dataset_root(target)
    log(f"\n[5/5] Reconciling against the archive table")
    log(f"  resolved dataset root: {dataset_root}")
    report = reconcile(entries, dataset_root, manifest_rows)
    log(f"  archive files : {report['archive_files']}")
    log(f"  files on disk : {report['files_on_disk']}")
    log(f"  recovered     : {report['recovered_under_safe_names']}")
    log(f"  missing       : {report['missing_count']}")
    log(f"  truncated     : {report['truncated_count']}")
    for path in report["missing"][:10]:
        log(f"    MISSING: {path}")
    for path in report["truncated"][:10]:
        log(f"    TRUNCATED: {path}")

    metadata.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "name_limit_bytes": LINUX_NAME_MAX_BYTES,
                "renamed": manifest_rows,
                "reconciliation": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"  manifest      -> {manifest_path}")

    return {
        "dataset_root": str(dataset_root),
        "parts": rows,
        "renamed": manifest_rows,
        "reconciliation": report,
        "manifest_path": str(manifest_path),
        "complete": report["complete"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="/kaggle/working/rwf2000")
    parser.add_argument(
        "--search-root",
        action="append",
        dest="search_roots",
        default=None,
        help="Where to look for the archive parts (repeatable).",
    )
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--skip-md5", action="store_true", help="Skip part checksums.")
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Only locate, verify, and reconcile what is already extracted.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete any previous output before extracting.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the readiness validation after preparing, and print the verdict.",
    )
    return parser


def bootstrap(
    target: Path,
    search_roots: Sequence[str] = DEFAULT_SEARCH_ROOTS,
    metadata_dir: Optional[Path] = None,
    check_md5: bool = True,
    skip_extraction: bool = False,
    fresh: bool = False,
    validate: bool = True,
    log=_log,
) -> Tuple[bool, dict]:
    """Prepare the dataset and, optionally, run the readiness validation.

    Returns ``(ready, report)``. ``ready`` is true only when extraction
    reconciled completely *and* every validation check passed, so a caller
    never has to interpret partial success.
    """
    report = prepare(
        target=target,
        search_roots=search_roots,
        metadata_dir=metadata_dir,
        check_md5=check_md5,
        skip_extraction=skip_extraction,
        fresh=fresh,
        log=log,
    )
    if not report["complete"]:
        log("\nEXTRACTION INCOMPLETE -- validation not attempted.")
        return False, report

    if not validate:
        return True, report

    try:
        from rwf2000_validation import validate_dataset
    except ImportError:  # pragma: no cover - supports package execution
        from src.rwf2000_validation import validate_dataset

    log("\n" + "=" * 68)
    log("DATASET VALIDATION")
    log("=" * 68)
    validation = validate_dataset(Path(report["dataset_root"]))
    log(validation.to_text())
    for key in ("split_counts", "label_counts", "fps_top5", "frame_counts_top5"):
        log(f"{key}: {validation.summary[key]}")
    report["validation"] = validation.as_dict()
    return validation.ok, report


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    try:
        ready, report = bootstrap(
            target=Path(args.target),
            search_roots=args.search_roots or list(DEFAULT_SEARCH_ROOTS),
            metadata_dir=Path(args.metadata_dir) if args.metadata_dir else None,
            check_md5=not args.skip_md5,
            skip_extraction=args.skip_extraction,
            fresh=args.fresh,
            validate=args.validate,
        )
    except KagglePreparationError as error:
        print(f"\nPREPARATION FAILED: {error}", file=sys.stderr)
        return 1
    if args.validate:
        print("\nRESULT: READY" if ready else "\nRESULT: NOT READY")
    else:
        print("\nComplete." if ready else "\nINCOMPLETE - see missing list.")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
