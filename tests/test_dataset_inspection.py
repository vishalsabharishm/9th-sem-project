import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_inspection import (
    DatasetInspectionError,
    discover_layout,
    find_cross_split_duplicates,
    find_cross_split_name_collisions,
    inspect_dataset,
    long_path,
    normalize_label,
    write_metadata,
)


def write_video(
    path: Path,
    frames: int = 5,
    size: int = 32,
    fps: float = 30.0,
    content_seed: int = 0,
) -> None:
    """Write a tiny real video so probing exercises actual OpenCV decoding.

    ``content_seed`` varies the pixel data between files. Without it every
    generated clip would be byte-identical, and the cross-split duplicate
    check would correctly flag the whole fixture as duplicates -- which
    would not represent a real dataset.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (size, size)
    )
    if not writer.isOpened():
        raise unittest.SkipTest("No MJPG writer available in this OpenCV build.")
    rng = np.random.default_rng(seed=content_seed)
    for _ in range(frames):
        writer.write(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
    writer.release()


class DatasetFixture(unittest.TestCase):
    """Builds a small split/class dataset mirroring RWF-2000's shape."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "dataset"
        content_seed = 0
        for split, label, count in (
            ("train", "Fight", 2),
            ("train", "NonFight", 2),
            ("test", "Fight", 1),
            ("test", "NonFight", 1),
        ):
            for index in range(count):
                content_seed += 1
                write_video(
                    self.root / split / label / f"{split}_{label}_{index}.avi",
                    content_seed=content_seed,
                )

    def tearDown(self):
        self._temp.cleanup()


class LayoutDiscoveryTests(DatasetFixture):
    def test_discovers_splits_and_classes_without_assuming_names(self):
        layout = discover_layout(self.root)

        self.assertEqual(sorted(layout), ["test", "train"])
        self.assertEqual(layout["train"], ["Fight", "NonFight"])

    def test_missing_root_raises(self):
        with self.assertRaises(DatasetInspectionError):
            discover_layout(self.root / "does_not_exist")


class InspectionTests(DatasetFixture):
    def test_counts_videos_by_split_and_class(self):
        summary, records = inspect_dataset(self.root)

        self.assertEqual(summary.readable_videos, 6)
        self.assertEqual(summary.split_counts, {"test": 2, "train": 4})
        self.assertEqual(summary.label_counts, {"Fight": 3, "NonFight": 3})
        self.assertEqual(
            summary.split_label_counts["train"], {"Fight": 2, "NonFight": 2}
        )
        self.assertEqual(len(records), 6)

    def test_probes_real_video_characteristics(self):
        _, records = inspect_dataset(self.root)
        record = records[0]

        self.assertTrue(record.readable)
        self.assertEqual((record.width, record.height), (32, 32))
        self.assertEqual(record.frame_count, 5)
        self.assertAlmostEqual(record.fps, 30.0, places=1)
        self.assertIsNotNone(record.duration_seconds)

    def test_unreadable_video_is_reported_not_raised(self):
        broken = self.root / "train" / "Fight" / "corrupt.avi"
        broken.write_bytes(b"this is not a video")

        summary, _ = inspect_dataset(self.root)

        self.assertIn("train/Fight/corrupt.avi", summary.unreadable)
        # The readable videos are still counted; one bad file must not
        # abort inspection of the rest of the dataset.
        self.assertEqual(summary.readable_videos, 6)

    def test_unexpected_formats_are_listed_separately(self):
        (self.root / "train" / "Fight" / "notes.txt").write_text("x", encoding="utf-8")

        summary, records = inspect_dataset(self.root)

        self.assertEqual(summary.unexpected_formats, ["train/Fight/notes.txt"])
        self.assertEqual(len(records), 6)


class LeakageCheckTests(DatasetFixture):
    def test_detects_identical_video_across_splits(self):
        source = self.root / "train" / "Fight" / "train_Fight_0.avi"
        shutil.copy(source, self.root / "test" / "Fight" / "leaked_copy.avi")

        _, records = inspect_dataset(self.root)
        duplicates = find_cross_split_duplicates(self.root, records)

        self.assertEqual(len(duplicates), 1)
        self.assertIn("test/Fight/leaked_copy.avi", duplicates[0])

    def test_clean_dataset_reports_no_duplicates(self):
        _, records = inspect_dataset(self.root)

        self.assertEqual(find_cross_split_duplicates(self.root, records), [])
        self.assertEqual(find_cross_split_name_collisions(records), [])

    def test_detects_shared_filename_stem_across_splits(self):
        write_video(self.root / "test" / "Fight" / "train_Fight_0.avi", frames=3)

        _, records = inspect_dataset(self.root)

        self.assertIn("train_Fight_0", find_cross_split_name_collisions(records))


class LabelNormalizationTests(unittest.TestCase):
    """RWF-2000 prefixes class directories with their split name."""

    def test_strips_prefix_matching_the_containing_split(self):
        self.assertEqual(normalize_label("Train_Fight", "train"), "Fight")
        self.assertEqual(normalize_label("Train_NonFight", "train"), "NonFight")
        self.assertEqual(normalize_label("Val_Fight", "val"), "Fight")
        self.assertEqual(normalize_label("Val_NonFight", "val"), "NonFight")

    def test_leaves_unprefixed_class_names_untouched(self):
        self.assertEqual(normalize_label("Fight", "train"), "Fight")
        self.assertEqual(normalize_label("NonFight", "test"), "NonFight")

    def test_does_not_strip_a_prefix_from_a_different_split(self):
        # A class genuinely called ``Train_Fight`` inside a ``val`` split
        # is not the ``val`` prefix pattern, so it must survive intact.
        self.assertEqual(normalize_label("Train_Fight", "val"), "Train_Fight")

    def test_never_produces_an_empty_label(self):
        self.assertEqual(normalize_label("train_", "train"), "train_")


class SplitPrefixedLayoutTests(unittest.TestCase):
    """End-to-end counting over the real RWF-2000 directory shape."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "dataset"
        content_seed = 0
        for split, class_dir, count in (
            ("train", "Train_Fight", 3),
            ("train", "Train_NonFight", 2),
            ("val", "Val_Fight", 1),
            ("val", "Val_NonFight", 1),
        ):
            for index in range(count):
                content_seed += 1
                write_video(
                    self.root / split / class_dir / f"{class_dir}_{index}.avi",
                    content_seed=content_seed,
                )

    def tearDown(self):
        self._temp.cleanup()

    def test_layout_reports_raw_directory_names(self):
        layout = discover_layout(self.root)

        self.assertEqual(sorted(layout), ["train", "val"])
        self.assertEqual(layout["train"], ["Train_Fight", "Train_NonFight"])
        self.assertEqual(layout["val"], ["Val_Fight", "Val_NonFight"])

    def test_counts_collapse_to_two_binary_labels(self):
        summary, _ = inspect_dataset(self.root)

        # Four directories, but only two real classes.
        self.assertEqual(summary.label_counts, {"Fight": 4, "NonFight": 3})
        self.assertEqual(
            summary.split_label_counts,
            {
                "train": {"Fight": 3, "NonFight": 2},
                "val": {"Fight": 1, "NonFight": 1},
            },
        )

    def test_raw_class_directories_remain_auditable(self):
        summary, records = inspect_dataset(self.root)

        self.assertEqual(
            summary.class_directory_counts,
            {
                "train/Train_Fight": 3,
                "train/Train_NonFight": 2,
                "val/Val_Fight": 1,
                "val/Val_NonFight": 1,
            },
        )
        fight = next(r for r in records if r.split == "train" and r.label == "Fight")
        self.assertEqual(fight.class_directory, "Train_Fight")


class LongPathTests(unittest.TestCase):
    """Clips whose names push them past the Windows MAX_PATH limit."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "d"
        write_video(self.root / "train" / "Train_Fight" / "short.avi", content_seed=1)

    def tearDown(self):
        # ``shutil.rmtree`` walks with plain paths, so a clip created past
        # MAX_PATH would block cleanup of the temporary directory. Remove
        # those through the prefixed form first.
        for directory, _, filenames in os.walk(self.root):
            for name in filenames:
                target = os.path.join(directory, name)
                if len(target) >= 260:
                    os.remove(long_path(Path(target)))
        self._temp.cleanup()

    def test_prefix_is_only_applied_on_windows(self):
        result = long_path(Path("relative") / "clip.avi")

        if os.name == "nt":
            self.assertTrue(result.startswith("\\\\?\\"))
        else:
            self.assertFalse(result.startswith("\\\\?\\"))
        self.assertTrue(result.endswith(os.path.join("relative", "clip.avi")))

    def test_already_prefixed_path_is_not_double_prefixed(self):
        if os.name != "nt":
            self.skipTest("Extended-length prefixes are Windows-only.")
        once = long_path(Path("C:/some/clip.avi"))

        self.assertEqual(long_path(Path(once)), once)

    def test_over_long_clip_is_still_inspected(self):
        if os.name != "nt":
            self.skipTest("MAX_PATH only constrains Windows.")
        class_dir = self.root / "train" / "Train_Fight"
        # Pad the stem so the absolute path exceeds MAX_PATH on name alone.
        padding = 265 - len(str(class_dir)) - len(".avi") - 1
        if padding < 1:
            self.skipTest("Temporary directory is already too deep to test.")
        target = class_dir / ("f" * padding + ".avi")
        self.assertGreaterEqual(len(str(target)), 260)

        write_video(Path(long_path(target)), content_seed=99)
        summary, records = inspect_dataset(self.root)

        self.assertEqual(summary.readable_videos, 2)
        self.assertIn(target.name, [Path(r.relative_path).name for r in records])


class MetadataWritingTests(DatasetFixture):
    def test_metadata_json_contains_summary_and_per_video_records(self):
        summary, records = inspect_dataset(self.root)
        output = Path(self._temp.name) / "metadata" / "dataset.json"

        written = write_metadata(summary, records, output)

        self.assertTrue(written.is_file())
        document = json.loads(written.read_text(encoding="utf-8"))
        self.assertEqual(document["summary"]["readable_videos"], 6)
        self.assertEqual(len(document["videos"]), 6)
        self.assertIn("relative_path", document["videos"][0])


if __name__ == "__main__":
    unittest.main()
