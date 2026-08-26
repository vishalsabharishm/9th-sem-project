"""Tests for leakage-excluded evaluation splits.

Six RWF-2000 validation clips duplicate training clips byte for byte.
These tests pin down that the exclusion is applied to evaluation only:
the official 400-clip split stays whole, the primary set is exactly the
394 remaining clips, and nothing touches the filesystem.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dataset_inspection
import video_loader
from dataset_inspection import content_digest, long_path
from rwf2000_config import (
    HELD_OUT_SPLIT_DIRECTORY,
    LEAKAGE_EXCLUDED_CLIP_COUNT,
    LEAKED_VALIDATION_CLIPS,
    OFFICIAL_VALIDATION_CLIP_COUNT,
    PRIMARY_EVALUATION_CLIP_COUNT,
    leakage_excluded_paths,
)
from rwf2000_splits import (
    LeakageExclusionError,
    build_evaluation_splits,
    clip_path,
    expected_split_sizes,
    held_out_clip_paths,
)

REAL_DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "rwf2000" / "RWF-2000"


def write_video(path: Path, frames: int = 3, size: int = 32, content_seed: int = 0):
    """Write a tiny real video, creating parent directories as needed."""
    Path(long_path(path.parent)).mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        long_path(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (size, size)
    )
    if not writer.isOpened():
        raise unittest.SkipTest("No MJPG writer available in this OpenCV build.")
    rng = np.random.default_rng(seed=content_seed)
    for _ in range(frames):
        writer.write(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
    writer.release()


class RecordedLeakageTests(unittest.TestCase):
    """The recorded exclusion list itself must be coherent."""

    def test_exactly_six_pairs_are_recorded(self):
        self.assertEqual(len(LEAKED_VALIDATION_CLIPS), 6)
        self.assertEqual(LEAKAGE_EXCLUDED_CLIP_COUNT, 6)

    def test_every_excluded_clip_is_in_the_held_out_split(self):
        for clip in LEAKED_VALIDATION_CLIPS:
            self.assertTrue(
                clip.validation_clip.startswith(f"{HELD_OUT_SPLIT_DIRECTORY}/"),
                f"{clip.validation_clip} is not in the held-out split",
            )

    def test_every_duplicate_source_is_a_training_clip(self):
        # The point of the exclusion is that these clips were trained on.
        for clip in LEAKED_VALIDATION_CLIPS:
            self.assertTrue(clip.training_clip.startswith("train/"))

    def test_recorded_clips_and_digests_are_unique(self):
        validation = [clip.validation_clip for clip in LEAKED_VALIDATION_CLIPS]
        digests = [clip.md5 for clip in LEAKED_VALIDATION_CLIPS]

        self.assertEqual(len(set(validation)), 6)
        self.assertEqual(len(set(digests)), 6)
        self.assertEqual(len(leakage_excluded_paths()), 6)

    def test_documented_counts_are_self_consistent(self):
        self.assertEqual(OFFICIAL_VALIDATION_CLIP_COUNT, 400)
        self.assertEqual(PRIMARY_EVALUATION_CLIP_COUNT, 394)
        self.assertEqual(
            OFFICIAL_VALIDATION_CLIP_COUNT - LEAKAGE_EXCLUDED_CLIP_COUNT,
            PRIMARY_EVALUATION_CLIP_COUNT,
        )
        self.assertEqual(
            expected_split_sizes(),
            {
                "official_validation": 400,
                "leakage_excluded": 6,
                "primary_evaluation": 394,
            },
        )


class SyntheticExclusionTests(unittest.TestCase):
    """Partitioning behaviour, exercised on a small stand-in dataset.

    The fixture reuses the real recorded filenames so the test proves the
    actual recorded list drives the exclusion, not a placeholder.
    """

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "RWF-2000"
        seed = 0
        write_video(self.root / "train" / "Train_NonFight" / "keep.avi", content_seed=1)
        for clip in LEAKED_VALIDATION_CLIPS:
            seed += 1
            write_video(self.root / clip.validation_clip, content_seed=seed)
        # Four clean held-out clips, so official = 10 and primary = 4.
        for index in range(4):
            seed += 1
            write_video(
                self.root / "val" / "Val_Fight" / f"clean_{index}.avi",
                content_seed=seed,
            )

    def tearDown(self):
        self._temp.cleanup()

    def test_official_split_is_enumerated_whole(self):
        splits = build_evaluation_splits(self.root)

        self.assertEqual(splits.official_count, 10)
        self.assertEqual(len(held_out_clip_paths(self.root)), 10)

    def test_exactly_the_recorded_clips_are_excluded(self):
        splits = build_evaluation_splits(self.root)

        self.assertEqual(set(splits.leakage_excluded), leakage_excluded_paths())
        self.assertEqual(splits.excluded_count, 6)

    def test_primary_is_official_minus_excluded(self):
        splits = build_evaluation_splits(self.root)

        self.assertEqual(splits.primary_count, 4)
        self.assertEqual(
            set(splits.primary_evaluation),
            set(splits.official_validation) - set(splits.leakage_excluded),
        )
        # Excluded and primary must not overlap, and primary must not
        # smuggle in anything outside the official split.
        self.assertEqual(
            set(splits.primary_evaluation) & set(splits.leakage_excluded), set()
        )
        self.assertTrue(
            set(splits.primary_evaluation).issubset(set(splits.official_validation))
        )

    def test_training_clips_are_never_part_of_any_evaluation_set(self):
        splits = build_evaluation_splits(self.root)

        for group in (splits.official_validation, splits.primary_evaluation):
            self.assertFalse(any(path.startswith("train/") for path in group))

    def test_nothing_is_removed_from_disk(self):
        before = sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if os.path.isfile(long_path(p))
        )

        build_evaluation_splits(self.root)

        after = sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if os.path.isfile(long_path(p))
        )
        self.assertEqual(before, after)
        self.assertEqual(len(after), 11)

    def test_missing_recorded_clip_is_an_error_not_a_silent_skip(self):
        os.remove(long_path(self.root / LEAKED_VALIDATION_CLIPS[0].validation_clip))

        with self.assertRaises(LeakageExclusionError):
            build_evaluation_splits(self.root)

        relaxed = build_evaluation_splits(self.root, strict=False)
        self.assertEqual(relaxed.excluded_count, 5)

    def test_splits_serialize_for_the_metadata_document(self):
        document = build_evaluation_splits(self.root).as_dict()

        self.assertEqual(document["counts"]["leakage_excluded"], 6)
        self.assertEqual(len(document["leaked_pairs"]), 6)
        self.assertIn("md5", document["leaked_pairs"][0])


class LongPathDatasetAccessTests(unittest.TestCase):
    """Dataset access must route through the one shared long-path helper."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "RWF-2000"

    def tearDown(self):
        for directory, _, filenames in os.walk(self.root):
            for name in filenames:
                target = os.path.join(directory, name)
                if len(target) >= 260:
                    os.remove(long_path(Path(target)))
        self._temp.cleanup()

    def test_path_handling_is_not_duplicated(self):
        # dataset_inspection re-exports the helper rather than owning a
        # second copy of the rule.
        self.assertIs(dataset_inspection.long_path, video_loader.long_path)

    def test_clip_path_applies_the_shared_helper(self):
        relative = "val/Val_Fight/clip.avi"

        self.assertEqual(clip_path(self.root, relative), long_path(self.root / relative))

    def test_over_long_clip_is_reachable_through_clip_path(self):
        if os.name != "nt":
            self.skipTest("MAX_PATH only constrains Windows.")
        class_dir = self.root / "val" / "Val_Fight"
        padding = 265 - len(str(class_dir)) - len(".avi") - 1
        if padding < 1:
            self.skipTest("Temporary directory is already too deep to test.")
        name = "v" * padding + ".avi"
        write_video(class_dir / name, content_seed=7)
        self.assertGreaterEqual(len(str(class_dir / name)), 260)

        relative = f"val/Val_Fight/{name}"
        capture = cv2.VideoCapture(clip_path(self.root, relative))
        try:
            self.assertTrue(capture.isOpened())
        finally:
            capture.release()

        # The over-long clip must also survive enumeration and hashing.
        self.assertIn(relative, held_out_clip_paths(self.root))
        self.assertEqual(len(content_digest(self.root / relative)), 32)


@unittest.skipUnless(
    REAL_DATASET_ROOT.is_dir(),
    "Extracted RWF-2000 dataset not present.",
)
class RealDatasetSplitTests(unittest.TestCase):
    """The documented 400 / 6 / 394 figures, checked against the real data."""

    @classmethod
    def setUpClass(cls):
        cls.splits = build_evaluation_splits(REAL_DATASET_ROOT)

    def test_official_validation_set_keeps_all_400_clips(self):
        self.assertEqual(self.splits.official_count, OFFICIAL_VALIDATION_CLIP_COUNT)

    def test_six_clips_are_excluded_as_leakage(self):
        self.assertEqual(self.splits.excluded_count, 6)
        self.assertEqual(set(self.splits.leakage_excluded), leakage_excluded_paths())

    def test_primary_evaluation_set_has_394_clips(self):
        self.assertEqual(self.splits.primary_count, PRIMARY_EVALUATION_CLIP_COUNT)
        self.assertEqual(self.splits.primary_count, 394)

    def test_recorded_pairs_are_still_byte_identical(self):
        # The reason for excluding them, re-checked directly: each pair
        # must hash the same over its full contents, and match the digest
        # recorded in the configuration.
        for clip in LEAKED_VALIDATION_CLIPS:
            validation = content_digest(REAL_DATASET_ROOT / clip.validation_clip)
            training = content_digest(REAL_DATASET_ROOT / clip.training_clip)

            self.assertEqual(validation, training, clip.validation_clip)
            self.assertEqual(validation, clip.md5, clip.validation_clip)

    def test_training_split_is_unchanged_at_1600_clips(self):
        train = held_out_clip_paths(REAL_DATASET_ROOT, split="train")

        self.assertEqual(len(train), 1600)


if __name__ == "__main__":
    unittest.main()
