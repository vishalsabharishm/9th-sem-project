"""Tests for the carved-validation (leakage-safe) evaluation protocol.

The protocol's central claim is that no clip used to make a choice is ever a
clip a reported number is measured on. These tests check that claim directly
against the real dataset, rather than trusting the documentation.

Tests that need the extracted 12 GB dataset skip cleanly without it; the
name-resolution and label tests do not need it.
"""

import csv
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carved_validation import (  # noqa: E402
    ARCHIVE_PREFIX,
    DEFAULT_CARVE_CSV,
    PROTOCOL_CARVED,
    PROTOCOL_EXPERIMENT1,
    RESOLUTION_ENCODING,
    RESOLUTION_EXACT,
    RESOLUTION_SAFE_NAME,
    SUPPORTED_PROTOCOLS,
    CarvedValidationError,
    clip_label,
    load_carved_split,
    recorded_carve_clips,
    resolve_recorded_clip,
    verify_disjoint,
)
from rwf2000_config import LEAKED_VALIDATION_CLIPS  # noqa: E402
from rwf2000_kaggle import safe_relative_path  # noqa: E402

REAL_DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "rwf2000" / "RWF-2000"
EXPECTED_CARVE = 240
EXPECTED_TRAIN_REMAINDER = 1360
EXPECTED_PRIMARY = 394


class RecordedMembershipTests(unittest.TestCase):
    """The carve split is recovered from a committed artifact, not a seed."""

    def test_carve_csv_records_exactly_240_clips(self):
        recorded = recorded_carve_clips(DEFAULT_CARVE_CSV)
        self.assertEqual(len(recorded), EXPECTED_CARVE)

    def test_recorded_labels_are_the_two_canonical_classes(self):
        recorded = recorded_carve_clips(DEFAULT_CARVE_CSV)
        self.assertEqual(set(recorded.values()), {"Fight", "NonFight"})

    def test_every_recorded_carve_clip_is_from_the_train_split(self):
        for clip in recorded_carve_clips(DEFAULT_CARVE_CSV):
            self.assertTrue(
                clip.startswith("train/"),
                f"{clip!r} is not a train clip; the carve subset must come from train.",
            )

    def test_missing_csv_is_an_error_not_an_empty_split(self):
        with self.assertRaises(CarvedValidationError):
            recorded_carve_clips(Path("does/not/exist.csv"))


class ClipLabelTests(unittest.TestCase):
    def test_label_comes_from_the_class_directory(self):
        self.assertEqual(clip_label("train/Train_Fight/a.avi"), "Fight")
        self.assertEqual(clip_label("train/Train_NonFight/a.avi"), "NonFight")
        self.assertEqual(clip_label("val/Val_Fight/a.avi"), "Fight")
        self.assertEqual(clip_label("val/Val_NonFight/a.avi"), "NonFight")


class NameResolutionTests(unittest.TestCase):
    """The three exact resolution mechanisms, and the refusal to guess."""

    def test_exact_name_resolves_first(self):
        local, method = resolve_recorded_clip(
            "train/Train_Fight/a.avi", {"train/Train_Fight/a.avi"}, {}, {}
        )
        self.assertEqual(local, "train/Train_Fight/a.avi")
        self.assertEqual(method, RESOLUTION_EXACT)

    def test_kaggle_safe_name_resolves_by_sha1(self):
        original = "train/Train_Fight/中文_urlgot_1.avi"
        safe = safe_relative_path(ARCHIVE_PREFIX + original).rsplit("/", 1)[-1]
        local, method = resolve_recorded_clip(
            f"train/Train_Fight/{safe}", set(), {safe: original}, {}
        )
        self.assertEqual(local, original)
        self.assertEqual(method, RESOLUTION_SAFE_NAME)

    def test_encoding_roundtrip_resolves(self):
        original = "train/Train_Fight/╤çx.avi"
        recorded = original.encode("utf-8").decode("cp1252")
        local, method = resolve_recorded_clip(recorded, set(), {}, {recorded: original})
        self.assertEqual(local, original)
        self.assertEqual(method, RESOLUTION_ENCODING)

    def test_unresolvable_name_returns_none_rather_than_a_guess(self):
        local, method = resolve_recorded_clip(
            "train/Train_Fight/nope.avi", {"train/Train_Fight/similar.avi"}, {}, {}
        )
        self.assertIsNone(local)
        self.assertIsNone(method)


class ProtocolNameTests(unittest.TestCase):
    def test_both_protocols_are_named_and_distinct(self):
        self.assertIn(PROTOCOL_CARVED, SUPPORTED_PROTOCOLS)
        self.assertIn(PROTOCOL_EXPERIMENT1, SUPPORTED_PROTOCOLS)
        self.assertNotEqual(PROTOCOL_CARVED, PROTOCOL_EXPERIMENT1)


@unittest.skipUnless(REAL_DATASET_ROOT.is_dir(), "Extracted RWF-2000 dataset not present.")
class RealSplitTests(unittest.TestCase):
    """The claims that matter, checked against the actual 2000 clips."""

    @classmethod
    def setUpClass(cls):
        cls.split = load_carved_split(REAL_DATASET_ROOT)

    def test_every_recorded_carve_clip_resolves_on_disk(self):
        self.assertEqual(len(self.split.carve_validation), EXPECTED_CARVE)

    def test_split_sizes_partition_the_official_train_split(self):
        self.assertEqual(len(self.split.carve_validation), EXPECTED_CARVE)
        self.assertEqual(len(self.split.train_remainder), EXPECTED_TRAIN_REMAINDER)
        self.assertEqual(
            len(self.split.carve_validation) + len(self.split.train_remainder), 1600
        )

    def test_primary_evaluation_is_the_leak_free_394(self):
        self.assertEqual(len(self.split.primary_evaluation), EXPECTED_PRIMARY)

    def test_the_three_sets_are_disjoint_and_leak_free(self):
        self.assertEqual(verify_disjoint(self.split), [])

    def test_no_confirmed_leaked_clip_appears_anywhere(self):
        leaked = {clip.validation_clip for clip in LEAKED_VALIDATION_CLIPS}
        for name in ("carve_validation", "train_remainder", "primary_evaluation"):
            self.assertEqual(
                leaked & set(getattr(self.split, name)), set(),
                f"a confirmed-leaked clip reached {name}",
            )

    def test_carve_never_overlaps_the_reporting_set(self):
        """The single most important property of the protocol."""
        self.assertEqual(
            set(self.split.carve_validation) & set(self.split.primary_evaluation), set()
        )

    def test_recorded_labels_match_the_on_disk_class_directory(self):
        for clip, label in self.split.labels.items():
            self.assertEqual(clip_label(clip), label, f"label disagreement for {clip!r}")

    def test_carve_label_balance_matches_the_recorded_run(self):
        counts = self.split.label_counts(self.split.carve_validation)
        self.assertEqual(counts, {"Fight": 119, "NonFight": 121})

    def test_split_is_stable_across_calls(self):
        again = load_carved_split(REAL_DATASET_ROOT)
        self.assertEqual(again.carve_validation, self.split.carve_validation)
        self.assertEqual(again.train_remainder, self.split.train_remainder)

    def test_as_dict_records_that_membership_was_recovered(self):
        document = self.split.as_dict()
        self.assertEqual(document["protocol"], PROTOCOL_CARVED)
        self.assertIn("recovered", document["membership_source"])
        self.assertEqual(document["counts"]["carve_validation"], EXPECTED_CARVE)


@unittest.skipUnless(REAL_DATASET_ROOT.is_dir(), "Extracted RWF-2000 dataset not present.")
class CarvedDatasetBuilderTests(unittest.TestCase):
    """The dataset builders expose the split with the right sizes and labels."""

    def test_builders_return_the_expected_sizes(self):
        from rwf2000_dataset import (
            build_carved_train_dataset,
            build_carved_validation_dataset,
        )

        train = build_carved_train_dataset(REAL_DATASET_ROOT)
        carve = build_carved_validation_dataset(REAL_DATASET_ROOT)
        self.assertEqual(len(train), EXPECTED_TRAIN_REMAINDER)
        self.assertEqual(len(carve), EXPECTED_CARVE)
        self.assertEqual(carve.label_counts(), {"NonFight": 121, "Fight": 119})

    def test_carve_validation_dataset_samples_deterministically(self):
        from rwf2000_dataset import build_carved_validation_dataset

        carve = build_carved_validation_dataset(REAL_DATASET_ROOT)
        self.assertFalse(carve.config.training)
        self.assertEqual(
            carve.frame_indices_for(0, 150), carve.frame_indices_for(0, 150)
        )


class TrainerProtocolWiringTests(unittest.TestCase):
    """The trainer must offer both protocols and default to the correct one."""

    def test_default_protocol_is_carved_validation(self):
        from temporal_training import TrainingRunConfig

        self.assertEqual(TrainingRunConfig(dataset_root=Path("root")).protocol, PROTOCOL_CARVED)

    def test_unknown_protocol_is_rejected(self):
        from temporal_training import TrainingError, TrainingRunConfig

        with self.assertRaises(TrainingError):
            TrainingRunConfig(dataset_root=Path("root"), protocol="whatever")

    def test_experiment1_protocol_is_still_available(self):
        from temporal_training import TrainingRunConfig

        config = TrainingRunConfig(dataset_root=Path("root"), protocol=PROTOCOL_EXPERIMENT1)
        self.assertEqual(config.protocol, PROTOCOL_EXPERIMENT1)

    def test_protocol_is_recorded_in_the_run_document(self):
        from temporal_training import TrainingRunConfig

        document = TrainingRunConfig(dataset_root=Path("root")).as_dict()
        self.assertEqual(document["protocol"], PROTOCOL_CARVED)

    def test_cli_exposes_both_protocols_and_defaults_to_carved(self):
        from temporal_training import build_arg_parser

        args = build_arg_parser().parse_args(["--train"])
        self.assertEqual(args.protocol, PROTOCOL_CARVED)
        args = build_arg_parser().parse_args(["--train", "--protocol", PROTOCOL_EXPERIMENT1])
        self.assertEqual(args.protocol, PROTOCOL_EXPERIMENT1)


if __name__ == "__main__":
    unittest.main()
