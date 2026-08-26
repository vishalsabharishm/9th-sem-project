"""Tests for the isolated R3D-18 violence-classification training pipeline.

Covers model construction, the 2-class head, dataset/split integration,
temporal sampling, preprocessing, checkpointing, and evaluation
determinism. No test trains a model, and none downloads pretrained
weights: every model here is randomly initialised on purpose.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_inspection import long_path
from rwf2000_config import LABEL_TO_INDEX, SamplingConfig
from rwf2000_dataset import (
    DatasetConfig,
    RWF2000ClipDataset,
    build_primary_evaluation_dataset,
    build_training_dataset,
    decode_frames,
    label_index_for,
)
from rwf2000_splits import clip_path, split_clip_paths
from temporal_metrics import binary_metrics, pr_auc, roc_auc
from temporal_model import R3D18TemporalModel, TemporalModelConfig
from temporal_preprocessing import ClipPreprocessor, PreprocessingConfig
from temporal_sampling import (
    TemporalSamplingError,
    describe_sampling,
    jittered_sample_indices,
    sample_indices,
    segment_bounds,
    uniform_sample_indices,
)
from temporal_training import (
    NUM_CLASSES,
    Checkpointer,
    TrainingError,
    TrainingHistory,
    TrainingRunConfig,
    build_model,
    build_optimizer,
    build_scheduler,
    freeze_modules,
    parameter_groups,
    set_seed,
)

REAL_DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "rwf2000" / "RWF-2000"
SOURCE_FRAMES = 150


def write_clip(path: Path, frames: int = SOURCE_FRAMES, size: int = 64, seed: int = 0):
    """Write a small multi-frame AVI so decoding is genuinely exercised."""
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


class TemporalSamplingTests(unittest.TestCase):
    """Frame-index arithmetic, on the real 150-frame clip length."""

    def test_segments_tile_the_clip_exactly(self):
        bounds = segment_bounds(150, 16)

        self.assertEqual(len(bounds), 16)
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], 150)
        for (_, stop), (next_start, _) in zip(bounds, bounds[1:]):
            self.assertEqual(stop, next_start)  # no gaps, no overlaps
        self.assertEqual(sum(stop - start for start, stop in bounds), 150)

    def test_uniform_indices_are_documented_values(self):
        self.assertEqual(
            uniform_sample_indices(150, 16),
            (4, 13, 23, 32, 41, 51, 60, 70, 79, 88, 98, 107, 116, 126, 135, 145),
        )

    def test_uniform_indices_are_ordered_and_in_range(self):
        indices = uniform_sample_indices(150, 16)

        self.assertEqual(list(indices), sorted(indices))
        self.assertTrue(all(0 <= index < 150 for index in indices))

    def test_uniform_sampling_is_repeatable(self):
        self.assertEqual(
            uniform_sample_indices(150, 16), uniform_sample_indices(150, 16)
        )

    def test_clip_length_is_configurable(self):
        for length in (8, 16, 32, 64):
            self.assertEqual(len(uniform_sample_indices(150, length)), length)

    def test_jitter_stays_inside_its_own_segment(self):
        rng = np.random.default_rng(0)
        bounds = segment_bounds(150, 16)

        for _ in range(50):
            indices = jittered_sample_indices(150, 16, rng)
            self.assertEqual(len(indices), 16)
            for index, (start, stop) in zip(indices, bounds):
                self.assertTrue(start <= index < stop)
            self.assertEqual(list(indices), sorted(indices))

    def test_jitter_is_reproducible_from_a_seed(self):
        first = jittered_sample_indices(150, 16, np.random.default_rng(7))
        second = jittered_sample_indices(150, 16, np.random.default_rng(7))

        self.assertEqual(first, second)

    def test_jitter_actually_varies(self):
        rng = np.random.default_rng(3)
        draws = {jittered_sample_indices(150, 16, rng) for _ in range(20)}

        self.assertGreater(len(draws), 1)

    def test_evaluation_ignores_jitter_even_when_given_a_generator(self):
        config = SamplingConfig()
        first = sample_indices(150, config, training=False, rng=np.random.default_rng(1))
        second = sample_indices(150, config, training=False, rng=np.random.default_rng(2))

        self.assertEqual(first, second)
        self.assertEqual(first, uniform_sample_indices(150, config.clip_length))

    def test_training_applies_jitter_when_enabled(self):
        config = SamplingConfig()
        sampled = sample_indices(150, config, training=True, rng=np.random.default_rng(5))

        self.assertEqual(len(sampled), config.clip_length)

    def test_jitter_can_be_disabled_by_configuration(self):
        config = SamplingConfig(train_temporal_jitter=False)
        sampled = sample_indices(150, config, training=True, rng=np.random.default_rng(5))

        self.assertEqual(sampled, uniform_sample_indices(150, config.clip_length))

    def test_short_clip_repeats_frames_rather_than_failing(self):
        indices = uniform_sample_indices(5, 16)

        self.assertEqual(len(indices), 16)
        self.assertTrue(all(0 <= index < 5 for index in indices))

    def test_invalid_inputs_are_rejected(self):
        for total, length in ((0, 16), (-1, 16), (150, 0), (150, -4)):
            with self.assertRaises(TemporalSamplingError):
                uniform_sample_indices(total, length)

    def test_unsupported_strategy_is_rejected(self):
        with self.assertRaises(TemporalSamplingError):
            sample_indices(150, SamplingConfig(strategy="consecutive"))

    def test_description_records_the_indices_used(self):
        lines = describe_sampling(150, SamplingConfig())

        self.assertIn("clip_length=16", lines)
        self.assertIn("frames_repeated=False", lines)


class PreprocessingShapeTests(unittest.TestCase):
    """The existing preprocessor, exercised at the sampled clip length."""

    def test_produces_the_r3d18_input_shape(self):
        clip = np.random.default_rng(0).integers(
            0, 256, (16, 64, 64, 3), dtype=np.uint8
        )

        tensor = ClipPreprocessor().preprocess(clip)

        self.assertEqual(tuple(tensor.shape), (1, 3, 16, 112, 112))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_normalized_values_are_standardized_not_raw(self):
        # A saturated frame proves the per-channel std division happened:
        # simple [0, 1] scaling would cap at 1.0, whereas dividing by a
        # std of ~0.22 pushes the value well above it.
        clip = np.full((16, 64, 64, 3), 255, dtype=np.uint8)

        tensor = ClipPreprocessor().preprocess(clip)

        self.assertGreater(float(tensor.max()), 2.0)
        self.assertTrue(bool(torch.isfinite(tensor).all()))

    def test_crop_size_is_configurable(self):
        clip = np.random.default_rng(1).integers(0, 256, (8, 64, 64, 3), dtype=np.uint8)
        config = PreprocessingConfig(resize_size=(64, 80), crop_size=(56, 56))

        tensor = ClipPreprocessor(config).preprocess(clip)

        self.assertEqual(tuple(tensor.shape), (1, 3, 8, 56, 56))


class ModelConstructionTests(unittest.TestCase):
    """Model building and the replaced 2-class head. No weights downloaded."""

    def test_builds_a_two_class_head(self):
        model = R3D18TemporalModel(
            TemporalModelConfig(num_classes=2, pretrained_backbone=False)
        )

        self.assertEqual(model.model.fc.out_features, 2)

    def test_forward_pass_returns_two_logits_per_clip(self):
        model = R3D18TemporalModel(
            TemporalModelConfig(num_classes=2, pretrained_backbone=False)
        )
        clip = torch.randn(2, 3, 16, 112, 112)

        with torch.no_grad():
            logits = model.model(clip)

        self.assertEqual(tuple(logits.shape), (2, 2))
        probabilities = torch.softmax(logits, dim=1)
        self.assertTrue(
            torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-5)
        )

    def test_untrained_model_is_not_task_specific(self):
        model = R3D18TemporalModel(
            TemporalModelConfig(num_classes=2, pretrained_backbone=False)
        )

        # Guards the project's integrity rule: only a real checkpoint may
        # be read as a violence prediction.
        self.assertFalse(model.is_task_specific)

    def test_feature_layer_is_exposed_for_future_gradcam(self):
        model = R3D18TemporalModel(
            TemporalModelConfig(num_classes=2, pretrained_backbone=False)
        )
        captured = {}

        def hook(_module, _inputs, output):
            captured["shape"] = tuple(output.shape)

        handle = model.feature_layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model.model(torch.randn(1, 3, 16, 112, 112))
        finally:
            handle.remove()

        # (N, 512, T', H', W') spatio-temporal maps, still un-pooled.
        self.assertEqual(len(captured["shape"]), 5)
        self.assertEqual(captured["shape"][1], 512)

    def test_build_model_uses_two_classes(self):
        config = TrainingRunConfig(
            dataset_root=Path("unused"), pretrained_backbone=False
        )

        model = build_model(config)

        self.assertEqual(model.model.fc.out_features, NUM_CLASSES)
        self.assertEqual(model.config.class_labels, ("NonFight", "Fight"))


class OptimizerAndFreezingTests(unittest.TestCase):
    """Parameter freezing and learning-rate groups."""

    def setUp(self):
        self.config = TrainingRunConfig(
            dataset_root=Path("unused"), pretrained_backbone=False
        )
        self.module = build_model(self.config).model

    def test_freezing_disables_gradients_for_named_stages(self):
        frozen = freeze_modules(self.module, ("stem", "layer1"))

        self.assertEqual(frozen, ["stem", "layer1"])
        self.assertTrue(
            all(not p.requires_grad for p in self.module.stem.parameters())
        )
        self.assertTrue(all(p.requires_grad for p in self.module.fc.parameters()))

    def test_freezing_an_unknown_module_is_an_error(self):
        with self.assertRaises(TrainingError):
            freeze_modules(self.module, ("does_not_exist",))

    def test_head_and_backbone_get_separate_learning_rates(self):
        freeze_modules(self.module, self.config.frozen_modules)
        groups = parameter_groups(self.module, self.config)

        rates = sorted(group["lr"] for group in groups)
        self.assertEqual(
            rates,
            sorted([self.config.learning_rate_backbone, self.config.learning_rate_head]),
        )

    def test_optimizer_and_scheduler_build_from_config(self):
        freeze_modules(self.module, self.config.frozen_modules)
        optimizer = build_optimizer(self.module, self.config)
        scheduler = build_scheduler(optimizer, self.config)

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertIsNotNone(scheduler)


class TrainingConfigTests(unittest.TestCase):
    """The run configuration is explicit and validated."""

    def test_defaults_come_from_the_documented_initial_config(self):
        config = TrainingRunConfig.from_initial_config(Path("root"))

        self.assertEqual(config.seed, 42)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.optimizer, "AdamW")
        self.assertEqual(config.frozen_modules, ("stem", "layer1", "layer2"))
        self.assertEqual(config.sampling.clip_length, 16)

    def test_overrides_are_applied(self):
        config = TrainingRunConfig.from_initial_config(
            Path("root"), epochs=3, batch_size=4, device="cpu"
        )

        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.batch_size, 4)

    def test_invalid_settings_are_rejected(self):
        for override in ({"epochs": 0}, {"batch_size": 0}, {"optimizer": "Nope"}):
            with self.assertRaises(TrainingError):
                TrainingRunConfig(dataset_root=Path("root"), **override)

    def test_config_serializes_for_the_run_record(self):
        document = TrainingRunConfig(dataset_root=Path("root")).as_dict()

        self.assertEqual(document["dataset_root"], "root")
        self.assertIsInstance(json.dumps(document, default=str), str)

    def test_seeding_is_reproducible(self):
        set_seed(123)
        first = torch.randn(4)
        set_seed(123)
        second = torch.randn(4)

        self.assertTrue(torch.equal(first, second))


class CheckpointConfigurationTests(unittest.TestCase):
    """Best-model selection and checkpoint contents."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.directory = Path(self._temp.name) / "ckpt"
        self.config = TrainingRunConfig(
            dataset_root=Path("unused"), pretrained_backbone=False
        )
        self.module = build_model(self.config).model

    def tearDown(self):
        self._temp.cleanup()

    def _metrics(self, score):
        # Two samples whose scores yield the requested ROC-AUC exactly.
        return binary_metrics([0, 1], [0.0, 1.0] if score > 0.5 else [1.0, 0.0])

    def test_first_result_always_improves(self):
        checkpointer = Checkpointer(self.directory, monitor="roc_auc")

        self.assertTrue(checkpointer.is_improvement(0.1))

    def test_higher_is_better_for_max_mode(self):
        checkpointer = Checkpointer(self.directory, monitor="roc_auc")
        checkpointer.best_value = 0.8

        self.assertTrue(checkpointer.is_improvement(0.9))
        self.assertFalse(checkpointer.is_improvement(0.7))

    def test_lower_is_better_for_min_mode(self):
        checkpointer = Checkpointer(self.directory, monitor="roc_auc", mode="min")
        checkpointer.best_value = 0.5

        self.assertTrue(checkpointer.is_improvement(0.4))
        self.assertFalse(checkpointer.is_improvement(0.6))

    def test_uncomputable_metric_never_wins(self):
        checkpointer = Checkpointer(self.directory, monitor="roc_auc")

        self.assertFalse(checkpointer.is_improvement(None))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(TrainingError):
            Checkpointer(self.directory, mode="sideways")

    def test_saving_writes_last_and_best(self):
        checkpointer = Checkpointer(self.directory, monitor="roc_auc")

        written = checkpointer.save(self.module, 1, self._metrics(1.0), self.config)

        self.assertTrue(checkpointer.last_path.is_file())
        self.assertTrue(checkpointer.best_path.is_file())
        self.assertIsNotNone(written["best"])
        self.assertEqual(checkpointer.best_epoch, 1)

    def test_worse_epoch_updates_last_but_not_best(self):
        checkpointer = Checkpointer(self.directory, monitor="roc_auc")
        checkpointer.save(self.module, 1, self._metrics(1.0), self.config)
        best_mtime = checkpointer.best_path.stat().st_mtime_ns

        written = checkpointer.save(self.module, 2, self._metrics(0.0), self.config)

        self.assertIsNone(written["best"])
        self.assertEqual(checkpointer.best_epoch, 1)
        self.assertEqual(checkpointer.best_path.stat().st_mtime_ns, best_mtime)

    def test_checkpoint_records_classes_and_config(self):
        checkpointer = Checkpointer(self.directory, monitor="roc_auc")
        checkpointer.save(self.module, 1, self._metrics(1.0), self.config)

        payload = torch.load(checkpointer.best_path, map_location="cpu", weights_only=False)

        self.assertEqual(payload["num_classes"], 2)
        self.assertEqual(payload["class_labels"], ["NonFight", "Fight"])
        self.assertEqual(payload["epoch"], 1)
        self.assertIn("state_dict", payload)
        self.assertIn("config", payload)

    def test_history_is_written_with_its_configuration(self):
        history = TrainingHistory()
        history.append({"epoch": 1, "train_loss": 0.5})

        path = history.write(self.directory / "history.json", self.config)
        document = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(document["epochs"]), 1)
        self.assertIn("config", document)
        self.assertIn("reproducibility", document)


class MetricsTests(unittest.TestCase):
    """Metric correctness, including the undefined cases."""

    def test_perfect_separation_scores_one(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.roc_auc, 1.0)
        self.assertEqual(metrics.f1, 1.0)
        self.assertEqual(metrics.confusion_matrix.true_positive, 2)
        self.assertEqual(metrics.confusion_matrix.false_positive, 0)

    def test_inverted_ranking_scores_zero_auc(self):
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)

    def test_tied_scores_give_chance_auc(self):
        self.assertEqual(roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_auc_is_undefined_for_a_single_class(self):
        self.assertIsNone(roc_auc([1, 1, 1], [0.2, 0.7, 0.9]))
        self.assertIsNone(pr_auc([0, 0, 0], [0.2, 0.7, 0.9]))

    def test_precision_undefined_when_nothing_predicted_positive(self):
        metrics = binary_metrics([0, 1], [0.1, 0.2], threshold=0.9)

        self.assertIsNone(metrics.precision)
        self.assertIsNone(metrics.f1)
        self.assertEqual(metrics.recall, 0.0)

    def test_confusion_matrix_counts_every_sample(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8])

        self.assertEqual(metrics.confusion_matrix.total, 4)
        self.assertEqual(metrics.support, {"NonFight": 2, "Fight": 2})

    def test_metrics_serialize(self):
        document = binary_metrics([0, 1], [0.2, 0.8]).as_dict()

        self.assertIn("confusion_matrix", document)
        self.assertIsInstance(json.dumps(document), str)


class SyntheticDatasetTests(unittest.TestCase):
    """Dataset construction, labelling, and evaluation determinism."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "RWF-2000"
        for index, (split, class_dir) in enumerate(
            (
                ("train", "Train_Fight"),
                ("train", "Train_NonFight"),
                ("val", "Val_Fight"),
                ("val", "Val_NonFight"),
            )
        ):
            write_clip(self.root / split / class_dir / f"clip_{index}.avi", seed=index)

    def tearDown(self):
        self._temp.cleanup()

    def test_labels_map_split_prefixed_directories_to_binary_classes(self):
        self.assertEqual(label_index_for("train/Train_Fight/a.avi"), LABEL_TO_INDEX["Fight"])
        self.assertEqual(
            label_index_for("val/Val_NonFight/a.avi"), LABEL_TO_INDEX["NonFight"]
        )

    def test_training_dataset_uses_the_official_train_split(self):
        dataset = build_training_dataset(self.root)

        self.assertEqual(len(dataset), 2)
        self.assertTrue(all(p.startswith("train/") for p in dataset.relative_paths))
        self.assertEqual(dataset.label_counts(), {"NonFight": 1, "Fight": 1})

    def test_dataset_item_has_the_model_input_shape(self):
        dataset = build_training_dataset(self.root)

        clip, label = dataset[0]

        self.assertEqual(tuple(clip.shape), (3, 16, 112, 112))
        self.assertIn(label, (0, 1))

    def test_evaluation_sampling_is_deterministic_across_calls(self):
        dataset = RWF2000ClipDataset(
            self.root,
            split_clip_paths(self.root, "val"),
            DatasetConfig(training=False),
        )

        first = dataset.frame_indices_for(0, SOURCE_FRAMES)
        second = dataset.frame_indices_for(0, SOURCE_FRAMES)

        self.assertEqual(first, second)
        self.assertEqual(first, uniform_sample_indices(SOURCE_FRAMES, 16))

    def test_evaluation_sampling_ignores_the_epoch(self):
        dataset = RWF2000ClipDataset(
            self.root,
            split_clip_paths(self.root, "val"),
            DatasetConfig(training=False),
        )

        dataset.set_epoch(0)
        first = dataset.frame_indices_for(0, SOURCE_FRAMES)
        dataset.set_epoch(9)
        second = dataset.frame_indices_for(0, SOURCE_FRAMES)

        self.assertEqual(first, second)

    def test_evaluation_returns_identical_tensors_on_repeat(self):
        dataset = RWF2000ClipDataset(
            self.root,
            split_clip_paths(self.root, "val"),
            DatasetConfig(training=False),
        )

        first, _ = dataset[0]
        second, _ = dataset[0]

        self.assertTrue(torch.equal(first, second))

    def test_training_sampling_varies_by_epoch_but_is_seed_reproducible(self):
        def indices(epoch, seed=42):
            dataset = RWF2000ClipDataset(
                self.root,
                split_clip_paths(self.root, "train"),
                DatasetConfig(training=True, seed=seed),
            )
            dataset.set_epoch(epoch)
            return dataset.frame_indices_for(0, SOURCE_FRAMES)

        self.assertEqual(indices(1), indices(1))          # reproducible
        self.assertNotEqual(indices(1), indices(2))       # varies per epoch

    def test_decoded_frame_count_matches_requested_indices(self):
        relative = split_clip_paths(self.root, "val")[0]

        frames = decode_frames(clip_path(self.root, relative), [0, 10, 20, 149])

        self.assertEqual(frames.shape[0], 4)
        self.assertEqual(frames.dtype, np.uint8)
        self.assertEqual(frames.shape[-1], 3)


@unittest.skipUnless(
    REAL_DATASET_ROOT.is_dir(), "Extracted RWF-2000 dataset not present."
)
class RealDatasetIntegrationTests(unittest.TestCase):
    """Wiring against the validated dataset. No clips are decoded here."""

    def test_training_dataset_is_the_official_1600_clips(self):
        dataset = build_training_dataset(REAL_DATASET_ROOT)

        self.assertEqual(len(dataset), 1600)
        self.assertEqual(dataset.label_counts(), {"NonFight": 800, "Fight": 800})

    def test_primary_evaluation_dataset_is_394_clean_clips(self):
        dataset = build_primary_evaluation_dataset(REAL_DATASET_ROOT)

        self.assertEqual(len(dataset), 394)
        self.assertEqual(dataset.label_counts(), {"NonFight": 194, "Fight": 200})

    def test_primary_evaluation_excludes_every_leaked_clip(self):
        from rwf2000_config import leakage_excluded_paths

        dataset = build_primary_evaluation_dataset(REAL_DATASET_ROOT)

        self.assertEqual(
            set(dataset.relative_paths) & leakage_excluded_paths(), set()
        )

    def test_evaluation_dataset_never_contains_training_clips(self):
        dataset = build_primary_evaluation_dataset(REAL_DATASET_ROOT)

        self.assertFalse(any(p.startswith("train/") for p in dataset.relative_paths))


if __name__ == "__main__":
    unittest.main()
