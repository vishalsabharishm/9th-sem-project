"""End-to-end regression test for the RWF-2000 Kaggle extraction pipeline.

The unit tests around ``rwf2000_kaggle`` exercise its pieces in isolation.
This one drives :func:`rwf2000_kaggle.prepare` from end to end against a
synthetic multi-volume archive that reproduces, together, every failure
mode this project has actually hit:

* a top-level ``RWF-2000/`` folder inside the archive, which nests into
  ``RWF-2000/RWF-2000`` unless the root is resolved;
* CJK basenames past the 255-byte ``NAME_MAX``, which the bulk pass cannot
  write and the recovery pass must land under safe names;
* a solid stream that has to be split back apart by recorded sizes and
  verified against each entry's CRC32.

It also pins the two regressions that previously corrupted the tree: a
re-run must not write any clip a second time (which inflates the class
counts), and a truncated clip must be repaired rather than trusted.

7-Zip is not required. A stand-in binary implementing only the three
invocations the pipeline issues is generated into a temporary directory,
so the test runs anywhere. The long-name failures are produced by the real
filesystem, not simulated -- on a filesystem that accepts the long names
the recovery pass correctly does nothing and the test still passes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import rwf2000_kaggle  # noqa: E402
from rwf2000_config import ARCHIVE_MD5  # noqa: E402


SPLITS = {
    "train": ("Train_Fight", "Train_NonFight"),
    "val": ("Fight", "NonFight"),
}
CLIPS_PER_CLASS_DIR = 6
OVER_LONG_PER_TRAIN_DIR = 2

# 3 UTF-8 bytes per character, so this comfortably exceeds NAME_MAX.
_LONG_STEM_TEMPLATE = "打架斗殴现场视频记录{index}" + "监控画面片段" * 16


SEVEN_ZIP_STAND_IN = '''\
import json, os, sys, zlib
from pathlib import Path

MANIFEST = json.loads(Path(os.environ["FAKE7Z_MANIFEST"]).read_text(encoding="utf-8"))
ORDER = [(path, Path(blob)) for path, blob in MANIFEST["order"]]

def do_list():
    for relative, blob in ORDER:
        payload = blob.read_bytes()
        crc = format(zlib.crc32(payload) & 0xFFFFFFFF, "08X")
        sys.stdout.write(
            "Path = %s\\nSize = %d\\nAttributes = A\\nCRC = %s\\n\\n"
            % (relative, len(payload), crc)
        )
    return 0

def do_extract(target):
    failures = 0
    for relative, blob in ORDER:
        destination = Path(target) / relative
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(blob.read_bytes())
        except OSError as error:
            if error.errno == 36:
                sys.stderr.write(
                    "ERROR: Can not open output file : File name too long\\n"
                )
                failures += 1
            else:
                raise
    return 2 if failures else 0

def do_stream(list_file):
    wanted = set(
        line for line in Path(list_file).read_text(encoding="utf-8").splitlines() if line
    )
    for relative, blob in ORDER:          # archive order, not requested order
        if relative in wanted:
            sys.stdout.buffer.write(blob.read_bytes())
    sys.stdout.buffer.flush()
    return 0

argv = sys.argv[1:]
if argv and argv[0] == "l":
    sys.exit(do_list())
if "x" in argv:
    if "-so" in argv:
        sys.exit(do_stream(next(a[3:] for a in argv if a.startswith("-i@"))))
    sys.exit(do_extract(next(a[2:] for a in argv if a.startswith("-o"))))
sys.exit(1)
'''


class ExtractionPipelineEndToEndTests(unittest.TestCase):
    """Drive ``prepare`` over a synthetic archive and assert the whole outcome."""

    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.work = Path(cls._temporary.name)
        cls.blobs_dir = cls.work / "blobs"
        cls.parts = cls.work / "parts"
        cls.target = cls.work / "target"
        cls.metadata = cls.work / "metadata"
        cls.blobs_dir.mkdir()
        cls.parts.mkdir()

        cls.payloads, cls.over_long = cls._build_archive()
        cls.expected_total = len(SPLITS) * 2 * CLIPS_PER_CLASS_DIR

        for name in ARCHIVE_MD5:  # a complete set of volumes must be present
            (cls.parts / name).write_bytes(b"stub volume")

        cls._install_stand_in()
        cls.report = cls._prepare()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("FAKE7Z_MANIFEST", None)
        cls._temporary.cleanup()

    # -- fixture construction ---------------------------------------------

    @classmethod
    def _build_archive(cls):
        """Write the payload blobs and the archive table that names them.

        The archive paths are the authority; blobs carry the bytes, because
        an over-long archive path cannot itself exist on this filesystem.
        """
        order = []
        payloads = {}
        over_long = []
        for split, class_dirs in SPLITS.items():
            for class_dir in class_dirs:
                for index in range(CLIPS_PER_CLASS_DIR):
                    if split == "train" and index < OVER_LONG_PER_TRAIN_DIR:
                        stem = _LONG_STEM_TEMPLATE.format(index=index)
                    else:
                        stem = f"{class_dir}_{index:04d}"
                    archive_path = f"RWF-2000/{split}/{class_dir}/{stem}.avi"
                    if len(f"{stem}.avi".encode("utf-8")) > 255:
                        over_long.append(archive_path)
                    payload = f"clip:{split}/{class_dir}/{index}".encode("utf-8") * (
                        index + 3
                    )
                    blob = cls.blobs_dir / f"blob_{len(order):04d}.bin"
                    blob.write_bytes(payload)
                    payloads[archive_path] = payload
                    order.append([archive_path, str(blob)])

        manifest = cls.work / "archive_table.json"
        manifest.write_text(json.dumps({"order": order}), encoding="utf-8")
        os.environ["FAKE7Z_MANIFEST"] = str(manifest)
        return payloads, over_long

    @classmethod
    def _install_stand_in(cls):
        """Put a 7-Zip stand-in named ``7z`` at the front of ``PATH``."""
        binary_dir = cls.work / "bin"
        binary_dir.mkdir()
        script = binary_dir / "fake7z.py"
        script.write_text(SEVEN_ZIP_STAND_IN, encoding="utf-8")
        launcher = binary_dir / "7z"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
        )
        launcher.chmod(0o755)
        if os.name == "nt":  # pragma: no cover - the shim is POSIX-only
            raise unittest.SkipTest("The 7-Zip stand-in requires a POSIX shell.")
        os.environ["PATH"] = f"{binary_dir}{os.pathsep}{os.environ['PATH']}"

    @classmethod
    def _prepare(cls):
        return rwf2000_kaggle.prepare(
            target=cls.target,
            search_roots=(str(cls.parts),),
            metadata_dir=cls.metadata,
            check_md5=False,          # the stub volumes carry no real bytes
            log=lambda message: None,
        )

    @property
    def dataset_root(self) -> Path:
        return Path(self.report["dataset_root"])

    def _clip_count(self) -> int:
        return sum(1 for _ in self.dataset_root.rglob("*.avi"))

    # -- the assertions ----------------------------------------------------

    def test_nested_top_level_folder_is_resolved(self):
        self.assertEqual(self.dataset_root, self.target / "RWF-2000")

    def test_every_archive_entry_reaches_disk(self):
        reconciliation = self.report["reconciliation"]
        self.assertEqual(reconciliation["archive_files"], self.expected_total)
        self.assertEqual(reconciliation["missing"], [])
        self.assertEqual(reconciliation["truncated"], [])
        self.assertTrue(reconciliation["complete"])

    def test_over_long_names_are_recovered_under_safe_names(self):
        recovered = self.report["reconciliation"]["recovered_under_safe_names"]
        def landed(archive_path: str) -> bool:
            # ``Path.exists()`` raises ENAMETOOLONG rather than returning
            # False for exactly these names, which is the condition here.
            try:
                return (self.target / archive_path).exists()
            except OSError:
                return False

        landed_directly = [path for path in self.over_long if landed(path)]
        # On a filesystem that accepts the long names nothing needs
        # recovering; otherwise every one of them must have been recovered.
        self.assertEqual(recovered, len(self.over_long) - len(landed_directly))

    def test_recovered_clips_are_byte_identical_to_the_archive(self):
        for row in self.report["renamed"]:
            written = self.target / row["written_relative_path"]
            self.assertTrue(written.is_file(), row["written_relative_path"])
            self.assertEqual(
                written.read_bytes(),
                self.payloads[row["original_archive_path"]],
                f"content differs for {row['original_archive_path'][:40]}",
            )

    def test_safe_names_fit_the_filesystem_limit(self):
        for row in self.report["renamed"]:
            name = Path(row["written_relative_path"]).name
            self.assertLessEqual(len(name.encode("utf-8")), 255)

    def test_no_clip_is_written_twice(self):
        """The duplicate-inflation regression: counts must be exact."""
        self.assertEqual(self._clip_count(), self.expected_total)
        for split, class_dirs in SPLITS.items():
            for class_dir in class_dirs:
                directory = self.dataset_root / split / class_dir
                self.assertEqual(
                    sum(1 for _ in directory.glob("*.avi")),
                    CLIPS_PER_CLASS_DIR,
                    f"{split}/{class_dir} does not hold exactly "
                    f"{CLIPS_PER_CLASS_DIR} clips",
                )

    def test_labels_come_from_the_directory_not_the_filename(self):
        """Renaming must never move a clip out of its class directory."""
        for row in self.report["renamed"]:
            original_directory = row["original_archive_path"].rsplit("/", 1)[0]
            written_directory = row["written_relative_path"].rsplit("/", 1)[0]
            self.assertEqual(original_directory, written_directory)

    def test_rerun_leaves_a_complete_tree_untouched(self):
        before = {
            path: path.stat().st_mtime_ns for path in self.dataset_root.rglob("*.avi")
        }
        second = self._prepare()
        after = {
            path: path.stat().st_mtime_ns for path in self.dataset_root.rglob("*.avi")
        }
        self.assertEqual(before, after, "a re-run rewrote already-correct clips")
        self.assertTrue(second["reconciliation"]["complete"])
        self.assertEqual(self._clip_count(), self.expected_total)

    def test_truncated_clip_is_repaired_rather_than_trusted(self):
        victim = sorted(self.dataset_root.rglob("*.avi"))[0]
        original = victim.read_bytes()
        victim.write_bytes(b"truncated")
        try:
            repaired = self._prepare()
            self.assertTrue(
                repaired["reconciliation"]["complete"],
                "a truncated clip survived a re-run",
            )
            self.assertEqual(victim.read_bytes(), original)
            self.assertEqual(self._clip_count(), self.expected_total)
        finally:
            if victim.read_bytes() != original:  # pragma: no cover - safety net
                victim.write_bytes(original)


class IncompleteVolumeSetTests(unittest.TestCase):
    """A missing volume must stop the run before anything is extracted."""

    def test_missing_volume_refuses_to_extract(self):
        with tempfile.TemporaryDirectory() as name:
            work = Path(name)
            parts = work / "parts"
            parts.mkdir()
            for part in list(ARCHIVE_MD5)[:-1]:  # one volume short
                (parts / part).write_bytes(b"stub volume")
            target = work / "target"
            with self.assertRaises(rwf2000_kaggle.KagglePreparationError):
                rwf2000_kaggle.prepare(
                    target=target,
                    search_roots=(str(parts),),
                    metadata_dir=work / "metadata",
                    check_md5=False,
                    log=lambda message: None,
                )
            self.assertFalse(
                target.exists(), "extraction started despite a missing volume"
            )


if __name__ == "__main__":
    unittest.main()
