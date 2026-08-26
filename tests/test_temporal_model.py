import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temporal_inference import (
    TemporalInferenceConfig,
    TemporalInferenceEngine,
    TemporalInferenceError,
)
from temporal_model import (
    KINETICS400_NUM_CLASSES,
    RANDOM_INIT,
    R3D18TemporalModel,
    TemporalModelConfig,
    TemporalModelError,
    kinetics400_labels,
    resolve_device,
)

# Small clips keep CPU forward passes to a fraction of a second; a full
# 16-frame clip takes roughly a second on the CPU-only dev machine.
TEST_CLIP_FRAMES = 4


def synthetic_input(frames: int = TEST_CLIP_FRAMES) -> torch.Tensor:
    """Return a preprocessed-shaped (N, C, T, H, W) tensor."""
    return torch.randn(1, 3, frames, 112, 112)


def bgr_frame(height: int = 64, width: int = 80) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


class ModelConstructionTests(unittest.TestCase):
    def test_defaults_to_kinetics_head_size_without_downloading_weights(self):
        model = R3D18TemporalModel()

        self.assertEqual(model.model.fc.out_features, KINETICS400_NUM_CLASSES)
        self.assertEqual(model.provenance, RANDOM_INIT)
        self.assertFalse(model.is_task_specific)

    def test_custom_num_classes_replaces_the_classification_head(self):
        model = R3D18TemporalModel(TemporalModelConfig(num_classes=2))

        self.assertEqual(model.model.fc.out_features, 2)
        self.assertEqual(model.model.fc.in_features, 512)

    def test_untrained_model_is_never_task_specific(self):
        model = R3D18TemporalModel(TemporalModelConfig(num_classes=2))
        prediction = model.predict(synthetic_input())

        self.assertFalse(prediction.is_task_specific)
        self.assertIn("NOT a violence", prediction.describe())

    def test_rejects_invalid_num_classes(self):
        with self.assertRaises(TemporalModelError):
            TemporalModelConfig(num_classes=0)

    def test_rejects_mismatched_class_labels(self):
        with self.assertRaises(TemporalModelError):
            TemporalModelConfig(num_classes=2, class_labels=("only_one",))


class DeviceSelectionTests(unittest.TestCase):
    def test_cpu_resolves(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")

    @unittest.skipIf(torch.cuda.is_available(), "Requires a machine without CUDA.")
    def test_cuda_request_fails_loudly_when_unavailable(self):
        with self.assertRaises(TemporalModelError):
            resolve_device("cuda")


class ForwardPassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = R3D18TemporalModel(TemporalModelConfig(num_classes=2))

    def test_forward_pass_returns_one_logit_per_class(self):
        prediction = self.model.predict(synthetic_input())

        self.assertEqual(tuple(prediction.logits.shape), (1, 2))
        self.assertEqual(tuple(prediction.probabilities.shape), (1, 2))
        self.assertEqual(prediction.num_classes, 2)

    def test_probabilities_form_a_distribution(self):
        prediction = self.model.predict(synthetic_input())

        self.assertAlmostEqual(float(prediction.probabilities.sum()), 1.0, places=5)
        self.assertIn(prediction.predicted_index, (0, 1))

    def test_rejects_non_tensor_input(self):
        with self.assertRaises(TemporalModelError):
            self.model.predict(np.zeros((1, 3, 4, 112, 112), dtype=np.float32))

    def test_rejects_wrong_dimensionality(self):
        with self.assertRaises(TemporalModelError):
            self.model.predict(torch.randn(3, TEST_CLIP_FRAMES, 112, 112))

    def test_rejects_wrong_channel_count(self):
        with self.assertRaises(TemporalModelError):
            self.model.predict(torch.randn(1, 1, TEST_CLIP_FRAMES, 112, 112))


class CheckpointLoadingTests(unittest.TestCase):
    def test_loading_a_checkpoint_marks_the_model_task_specific(self):
        config = TemporalModelConfig(num_classes=2)
        source = R3D18TemporalModel(config)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fine_tuned.pt"
            torch.save(source.model.state_dict(), checkpoint)

            loaded = R3D18TemporalModel(config)
            self.assertFalse(loaded.is_task_specific)

            loaded.load_checkpoint(checkpoint)

            self.assertTrue(loaded.is_task_specific)
            self.assertIn("fine_tuned.pt", loaded.provenance)

    def test_missing_checkpoint_raises(self):
        model = R3D18TemporalModel(TemporalModelConfig(num_classes=2))

        with self.assertRaises(TemporalModelError):
            model.load_checkpoint(Path("does_not_exist_checkpoint.pt"))

    def test_incompatible_checkpoint_raises(self):
        two_class = R3D18TemporalModel(TemporalModelConfig(num_classes=2))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "two_class.pt"
            torch.save(two_class.model.state_dict(), checkpoint)

            three_class = R3D18TemporalModel(TemporalModelConfig(num_classes=3))
            with self.assertRaises(TemporalModelError):
                three_class.load_checkpoint(checkpoint)


class KineticsLabelTests(unittest.TestCase):
    def test_kinetics_labels_are_action_classes_not_violence_classes(self):
        labels = kinetics400_labels()

        self.assertEqual(len(labels), KINETICS400_NUM_CLASSES)
        # Guards the integrity claim that Kinetics-400 provides no
        # violence category to borrow a score from.
        self.assertNotIn("violence", labels)
        self.assertNotIn("fighting", labels)


class InferenceEngineTests(unittest.TestCase):
    def _engine(self, clip_length=TEST_CLIP_FRAMES, stride=TEST_CLIP_FRAMES):
        config = TemporalInferenceConfig(
            clip_length=clip_length,
            stride=stride,
            model=TemporalModelConfig(num_classes=2),
        )
        return TemporalInferenceEngine(config)

    def test_returns_none_until_a_clip_is_complete(self):
        engine = self._engine()

        for index in range(TEST_CLIP_FRAMES - 1):
            self.assertIsNone(engine.add_frame(bgr_frame(), index))

        result = engine.add_frame(bgr_frame(), TEST_CLIP_FRAMES - 1)
        self.assertIsNotNone(result)

    def test_result_carries_clip_context_and_is_not_task_specific(self):
        engine = self._engine()

        result = None
        for index in range(TEST_CLIP_FRAMES):
            result = engine.add_frame(bgr_frame(), index) or result

        self.assertEqual(result.frame_numbers, list(range(TEST_CLIP_FRAMES)))
        self.assertEqual(tuple(result.prediction.logits.shape), (1, 2))
        self.assertGreater(result.inference_seconds, 0.0)
        self.assertFalse(result.is_task_specific)

    def test_threshold_is_unset_by_default(self):
        # A decision threshold is only meaningful after fine-tuning and
        # evaluation; defaulting to a number would be a fabricated value.
        self.assertIsNone(TemporalInferenceConfig().score_threshold)

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(TemporalInferenceError):
            TemporalInferenceConfig(clip_length=0)
        with self.assertRaises(TemporalInferenceError):
            TemporalInferenceConfig(stride=0)
        with self.assertRaises(TemporalInferenceError):
            TemporalInferenceConfig(score_threshold=1.5)


if __name__ == "__main__":
    unittest.main()
