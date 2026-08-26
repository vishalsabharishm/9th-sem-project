"""Tests for Kaggle-side preparation and dataset readiness validation.

The filesystem-limit logic is tested against the archive's real
characteristics without needing the archive: entry names are synthesised
at the same byte lengths. Nothing here extracts, downloads, or trains.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_inspection import long_path
from rwf2000_kaggle import (
    entries_absent_from_disk,
    _write_member,
    LINUX_NAME_MAX_BYTES,
    ArchiveEntry,
    _crc32,
    oversized_entries,
    parse_archive_listing,
    reconcile,
    safe_relative_path,
    verify_parts,
)
from rwf2000_validation import validate_dataset

LISTING = """Path = RWF-2000
Size = 0
Attributes = D_ drwxrwxr-x

Path = RWF-2000/train/Train_Fight/short.avi
Size = 1234
CRC = AABBCCDD
Attributes = _ -rw-rw-r--

Path = RWF-2000/val/Val_NonFight/other.avi
Size = 4321
CRC = 11223344
Attributes = _ -rw-rw-r--
"""


def write_clip(path: Path, frames: int = 6, size: int = 32, seed: int = 0):
    """Write a small real video so OpenCV probing is genuinely exercised."""
    Path(long_path(path.parent)).mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        long_path(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (size, size)
    )
    if not writer.isOpened():
        raise unittest.SkipTest("No MJPG writer available in this OpenCV build.")
    rng = np.random.default_rng(seed)
    for _ in range(frames):
        writer.write(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
    writer.release()


class ArchiveListingTests(unittest.TestCase):
    """Parsing 7-Zip's -slt table of contents."""

    def test_parses_files_and_skips_directories(self):
        entries = parse_archive_listing(LISTING)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].path, "RWF-2000/train/Train_Fight/short.avi")
        self.assertEqual(entries[0].size, 1234)
        self.assertEqual(entries[0].crc, "AABBCCDD")

    def test_backslash_paths_are_normalised(self):
        entries = parse_archive_listing(
            "Path = RWF-2000\\train\\Train_Fight\\a.avi\nSize = 1\nAttributes = _ -rw-\n"
        )

        self.assertEqual(entries[0].path, "RWF-2000/train/Train_Fight/a.avi")


class NameLimitTests(unittest.TestCase):
    """The Linux NAME_MAX rule is about bytes, not characters."""

    def test_limit_is_measured_in_utf8_bytes(self):
        # 100 CJK characters = 300 UTF-8 bytes: legal on Windows by
        # character count, impossible on Linux by byte count.
        entry = ArchiveEntry(path="RWF-2000/train/Train_Fight/" + "中" * 100 + ".avi", size=1)

        self.assertEqual(len(entry.basename), 104)
        self.assertEqual(entry.name_bytes, 300 + 4)
        self.assertTrue(entry.exceeds_name_limit())

    def test_ascii_names_are_unaffected(self):
        entry = ArchiveEntry(path="RWF-2000/train/Train_NonFight/EBwt3Thb_0.avi", size=1)

        self.assertFalse(entry.exceeds_name_limit())

    def test_boundary_is_exactly_255_bytes(self):
        at_limit = ArchiveEntry(path="d/" + "a" * LINUX_NAME_MAX_BYTES, size=1)
        over_limit = ArchiveEntry(path="d/" + "a" * (LINUX_NAME_MAX_BYTES + 1), size=1)

        self.assertFalse(at_limit.exceeds_name_limit())
        self.assertTrue(over_limit.exceeds_name_limit())

    def test_oversized_entries_selects_only_the_impossible_ones(self):
        entries = [
            ArchiveEntry(path="RWF-2000/train/Train_Fight/ok.avi", size=1),
            ArchiveEntry(path="RWF-2000/train/Train_Fight/" + "中" * 90 + ".avi", size=1),
        ]

        selected = oversized_entries(entries)

        self.assertEqual(len(selected), 1)
        self.assertIn("中", selected[0].basename)


class SafeNameTests(unittest.TestCase):
    """Sanitised names must be short, portable, stable, and unique."""

    def setUp(self):
        self.original = "RWF-2000/train/Train_Fight/" + "中" * 200 + "_urlgot_222.avi"

    def test_result_fits_the_filesystem_limit(self):
        safe = safe_relative_path(self.original)

        self.assertLessEqual(
            len(safe.rsplit("/", 1)[-1].encode("utf-8")), LINUX_NAME_MAX_BYTES
        )

    def test_directory_and_extension_are_preserved(self):
        safe = safe_relative_path(self.original)

        # The directory carries the class label, so it must never change.
        self.assertTrue(safe.startswith("RWF-2000/train/Train_Fight/"))
        self.assertTrue(safe.endswith(".avi"))

    def test_name_is_pure_ascii(self):
        self.assertTrue(safe_relative_path(self.original).isascii())

    def test_mapping_is_deterministic(self):
        self.assertEqual(
            safe_relative_path(self.original), safe_relative_path(self.original)
        )

    def test_distinct_originals_do_not_collide(self):
        names = {
            safe_relative_path("RWF-2000/train/Train_Fight/" + "中" * 200 + f"_{i}.avi")
            for i in range(50)
        }

        self.assertEqual(len(names), 50)

    def test_ascii_prefix_is_kept_for_traceability(self):
        safe = safe_relative_path("RWF-2000/train/Train_Fight/1060314" + "中" * 200 + ".avi")

        self.assertIn("1060314", safe)


class PartVerificationTests(unittest.TestCase):
    """A missing volume is the usual cause of a truncated extraction."""

    def test_missing_parts_are_reported_and_fail(self):
        ok, rows = verify_parts([], check_md5=False)

        self.assertFalse(ok)
        self.assertEqual(len(rows), 13)
        self.assertTrue(all(row["status"] == "MISSING" for row in rows))


class ReconciliationTests(unittest.TestCase):
    """Renamed clips must count as present, not as missing plus extra."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "RWF-2000"
        (self.root / "train" / "Train_Fight").mkdir(parents=True)
        (self.root / "train" / "Train_Fight" / "ok.avi").write_bytes(b"x")
        (self.root / "train" / "Train_Fight" / "renamed__abc123.avi").write_bytes(b"y")

    def tearDown(self):
        self._temp.cleanup()

    def test_renamed_entry_counts_as_present(self):
        entries = [
            ArchiveEntry(path="RWF-2000/train/Train_Fight/ok.avi", size=1),
            ArchiveEntry(path="RWF-2000/train/Train_Fight/" + "中" * 100 + ".avi", size=1),
        ]
        manifest = [
            {
                "original_archive_path": entries[1].path,
                "written_relative_path": "RWF-2000/train/Train_Fight/renamed__abc123.avi",
            }
        ]

        report = reconcile(entries, self.root, manifest)

        self.assertTrue(report["complete"])
        self.assertEqual(report["missing_count"], 0)
        self.assertEqual(report["recovered_under_safe_names"], 1)

    def test_genuinely_missing_entry_is_reported(self):
        entries = [
            ArchiveEntry(path="RWF-2000/train/Train_Fight/ok.avi", size=1),
            ArchiveEntry(path="RWF-2000/val/Val_Fight/absent.avi", size=1),
        ]

        report = reconcile(entries, self.root, [])

        self.assertFalse(report["complete"])
        self.assertEqual(report["missing"], ["RWF-2000/val/Val_Fight/absent.avi"])


class TruncationDetectionTests(unittest.TestCase):
    """An interrupted run leaves half-written files that must not pass."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "RWF-2000"
        (self.root / "train" / "Train_Fight").mkdir(parents=True)
        (self.root / "train" / "Train_Fight" / "a.avi").write_bytes(b"12345")

    def tearDown(self):
        self._temp.cleanup()

    def test_correct_size_is_complete(self):
        entries = [ArchiveEntry(path="RWF-2000/train/Train_Fight/a.avi", size=5)]

        report = reconcile(entries, self.root, [])

        self.assertTrue(report["complete"])
        self.assertEqual(report["truncated_count"], 0)

    def test_short_file_is_flagged_not_accepted(self):
        entries = [ArchiveEntry(path="RWF-2000/train/Train_Fight/a.avi", size=999)]

        report = reconcile(entries, self.root, [])

        self.assertFalse(report["complete"])
        self.assertEqual(report["truncated"], ["RWF-2000/train/Train_Fight/a.avi"])
        self.assertEqual(report["missing_count"], 0)


class RecoveryTargetingTests(unittest.TestCase):
    """Recovery is driven by what is absent, never by prediction alone."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.target = Path(self._temp.name) / "out"
        (self.target / "RWF-2000" / "train" / "Train_Fight").mkdir(parents=True)

    def tearDown(self):
        self._temp.cleanup()

    def _place(self, relative, payload):
        path = self.target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def test_entry_already_on_disk_is_not_recovered_again(self):
        # Otherwise the clip would exist twice -- once under its original
        # name and once sanitised -- inflating the class counts.
        entry = ArchiveEntry(path="RWF-2000/train/Train_Fight/a.avi", size=4)
        self._place(entry.path, b"abcd")

        self.assertEqual(entries_absent_from_disk([entry], self.target), [])

    def test_absent_entry_is_selected(self):
        entry = ArchiveEntry(path="RWF-2000/train/Train_Fight/gone.avi", size=4)

        self.assertEqual(entries_absent_from_disk([entry], self.target), [entry])

    def test_wrong_size_counts_as_absent(self):
        entry = ArchiveEntry(path="RWF-2000/train/Train_Fight/a.avi", size=99)
        self._place(entry.path, b"abcd")

        self.assertEqual(entries_absent_from_disk([entry], self.target), [entry])

    def test_entry_already_recovered_under_a_safe_name_is_skipped(self):
        original = "RWF-2000/train/Train_Fight/" + "中" * 100 + ".avi"
        entry = ArchiveEntry(path=original, size=4)
        safe = safe_relative_path(original)
        self._place(safe, b"abcd")
        manifest = [
            {"original_archive_path": original, "written_relative_path": safe}
        ]

        self.assertEqual(
            entries_absent_from_disk([entry], self.target, manifest), []
        )


class ManifestConventionTests(unittest.TestCase):
    """The rename manifest and reconciliation must share one coordinate system.

    Regression test: reconciliation used to re-prefix the written path,
    producing ``RWF-2000/RWF-2000/...`` so every recovered clip looked
    missing and a fully recovered dataset reported NOT READY.
    """

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.target = Path(self._temp.name) / "out"
        self.root = self.target / "RWF-2000"
        (self.root / "train" / "Train_Fight").mkdir(parents=True)

    def tearDown(self):
        self._temp.cleanup()

    def test_written_path_is_archive_relative_and_reconciles(self):
        original = "RWF-2000/train/Train_Fight/" + "中" * 100 + ".avi"
        entry = ArchiveEntry(path=original, size=4, crc=_crc32(b"abcd"))

        row = _write_member(self.target, entry, b"abcd")

        # It must start with the archive's top-level folder, exactly like
        # entry.path, so reconcile can compare the two directly.
        self.assertTrue(row["written_relative_path"].startswith("RWF-2000/"))
        self.assertTrue((self.target / row["written_relative_path"]).is_file())

        report = reconcile([entry], self.root, [row])

        self.assertTrue(report["complete"], report)
        self.assertEqual(report["missing_count"], 0)
        self.assertEqual(report["recovered_under_safe_names"], 1)

    def test_entry_with_a_legal_name_keeps_it(self):
        entry = ArchiveEntry(path="RWF-2000/train/Train_Fight/fine.avi", size=4)

        row = _write_member(self.target, entry, b"abcd")

        self.assertEqual(row["written_relative_path"], entry.path)
        self.assertTrue(reconcile([entry], self.root, [row])["complete"])


class StreamSplittingTests(unittest.TestCase):
    """The concatenated 7z stream is split by size and checked by CRC."""

    def test_crc_matches_zlib_for_known_payload(self):
        # 7-Zip records CRC32 as uppercase hex; this is the value the
        # splitter compares each slice against.
        self.assertEqual(_crc32(b"123456789"), "CBF43926")

    def test_slicing_by_recorded_sizes_reproduces_each_payload(self):
        payloads = [b"a" * 10, b"b" * 5, b"c" * 7]
        entries = [
            ArchiveEntry(path=f"RWF-2000/train/Train_Fight/f{i}.avi",
                         size=len(p), crc=_crc32(p))
            for i, p in enumerate(payloads)
        ]
        stream = b"".join(payloads)

        offset, recovered = 0, []
        for entry in entries:
            chunk = stream[offset : offset + entry.size]
            offset += entry.size
            self.assertEqual(_crc32(chunk), entry.crc)
            recovered.append(chunk)

        self.assertEqual(recovered, payloads)
        self.assertEqual(offset, len(stream))


class ValidationTests(unittest.TestCase):
    """Readiness checks over a small stand-in dataset."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "RWF-2000"
        seed = 0
        for split, class_dir in (
            ("train", "Train_Fight"),
            ("train", "Train_NonFight"),
            ("val", "Val_Fight"),
            ("val", "Val_NonFight"),
        ):
            for index in range(2):
                seed += 1
                write_clip(self.root / split / class_dir / f"{class_dir}_{index}.avi", seed=seed)

    def tearDown(self):
        self._temp.cleanup()

    def _validate(self, **kwargs):
        return validate_dataset(
            self.root, check_leakage=False, expect_rwf2000_splits=False, **kwargs
        )

    def test_clean_dataset_is_ready(self):
        report = self._validate()

        self.assertTrue(report.ok, report.to_text())

    def test_both_classes_are_detected_through_split_prefixes(self):
        report = self._validate()
        names = {check.name for check in report.checks if check.passed}

        self.assertIn("class_Fight_present", names)
        self.assertIn("class_NonFight_present", names)

    def test_missing_class_fails(self):
        import shutil

        shutil.rmtree(self.root / "train" / "Train_Fight")
        shutil.rmtree(self.root / "val" / "Val_Fight")

        report = self._validate()

        self.assertFalse(report.ok)
        self.assertIn(
            "class_Fight_present", {check.name for check in report.failures}
        )

    def test_empty_class_directory_fails(self):
        (self.root / "train" / "EmptyClass").mkdir()

        report = self._validate()

        self.assertFalse(report.ok)
        self.assertIn(
            "no_empty_class_directories", {check.name for check in report.failures}
        )

    def test_unreadable_video_is_reported(self):
        (self.root / "val" / "Val_Fight" / "broken.avi").write_bytes(b"not a video")

        report = self._validate()

        self.assertFalse(report.ok)
        self.assertIn(
            "all_videos_open_in_opencv", {check.name for check in report.failures}
        )

    def test_unexpected_file_format_is_reported(self):
        (self.root / "train" / "Train_Fight" / "notes.txt").write_text("x", encoding="utf-8")

        report = self._validate()

        self.assertFalse(report.ok)
        self.assertIn(
            "no_unexpected_file_formats", {check.name for check in report.failures}
        )

    def test_summary_records_observed_characteristics(self):
        report = self._validate()

        self.assertEqual(report.summary["readable_videos"], 8)
        self.assertEqual(report.summary["label_counts"], {"Fight": 4, "NonFight": 4})

    def test_missing_root_fails_cleanly(self):
        report = validate_dataset(self.root / "nope", check_leakage=False)

        self.assertFalse(report.ok)
        self.assertIn("dataset_root_exists", {c.name for c in report.failures})

    def test_report_renders_as_text(self):
        text = self._validate().to_text()

        self.assertIn("RESULT:", text)
        self.assertIn("class_Fight_present", text)


if __name__ == "__main__":
    unittest.main()
