import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temporal_clip_buffer import TemporalClipBuffer
from temporal_preprocessing import (
    ClipPreprocessor,
    PreprocessingConfig,
    TemporalPreprocessingError,
)


def bgr_clip(frames: int = 4, height: int = 64, width: int = 80, value: int = 128) -> np.ndarray:
    """Return a constant-valued (T, H, W, C) uint8 clip in OpenCV BGR order."""
    return np.full((frames, height, width, 3), value, dtype=np.uint8)


class ColourConversionTests(unittest.TestCase):
    def test_bgr_is_converted_to_rgb(self):
        preprocessor = ClipPreprocessor()
        clip = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        clip[..., 0] = 10  # B
        clip[..., 1] = 20  # G
        clip[..., 2] = 30  # R

        rgb = preprocessor.to_rgb(clip)

        self.assertEqual(rgb[0, 0, 0, 0], 30)
        self.assertEqual(rgb[0, 0, 0, 1], 20)
        self.assertEqual(rgb[0, 0, 0, 2], 10)

    def test_conversion_can_be_disabled(self):
        preprocessor = ClipPreprocessor(PreprocessingConfig(convert_bgr_to_rgb=False))
        clip = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        clip[..., 0] = 10

        unchanged = preprocessor.to_rgb(clip)

        self.assertEqual(unchanged[0, 0, 0, 0], 10)


class TensorLayoutTests(unittest.TestCase):
    def test_preprocess_returns_batched_channels_first_clip(self):
        preprocessor = ClipPreprocessor()

        tensor = preprocessor.preprocess(bgr_clip(frames=4))

        # R3D-18 expects (N, C, T, H, W).
        self.assertEqual(tuple(tensor.shape), (1, 3, 4, 112, 112))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_preprocess_accepts_a_clip_from_the_buffer(self):
        buffer = TemporalClipBuffer(clip_length=3)
        for index in range(3):
            buffer.add_frame(bgr_clip(frames=1)[0], index)

        tensor = ClipPreprocessor().preprocess(buffer.get_clip())

        self.assertEqual(tuple(tensor.shape), (1, 3, 3, 112, 112))

    def test_crop_size_is_configurable(self):
        config = PreprocessingConfig(resize_size=(64, 64), crop_size=(32, 32))

        tensor = ClipPreprocessor(config).preprocess(bgr_clip(frames=2))

        self.assertEqual(tuple(tensor.shape), (1, 3, 2, 32, 32))


class NormalizationTests(unittest.TestCase):
    def test_constant_frame_normalizes_to_expected_per_channel_values(self):
        config = PreprocessingConfig()
        tensor = ClipPreprocessor(config).preprocess(bgr_clip(value=128))

        scaled = 128 / 255.0
        for channel in range(3):
            expected = (scaled - config.mean[channel]) / config.std[channel]
            actual = tensor[0, channel].mean().item()
            self.assertAlmostEqual(actual, expected, places=4)

    def test_matches_torchvision_reference_preset_exactly(self):
        """Guards against silent drift from the pretrained preprocessing.

        The Kinetics-400 weights were trained with torchvision's video
        preset, including its deliberate ``antialias=False``. If our
        pipeline diverges, a pretrained backbone silently receives an
        input distribution it was never trained on.
        """
        from torchvision.models.video import R3D_18_Weights

        rng = np.random.default_rng(seed=0)
        clip = rng.integers(0, 256, size=(4, 120, 160, 3), dtype=np.uint8)

        ours = ClipPreprocessor().preprocess(clip)

        rgb = clip[..., ::-1].copy()
        reference_input = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()
        reference = R3D_18_Weights.KINETICS400_V1.transforms()(reference_input)
        reference = reference.unsqueeze(0)

        self.assertTrue(torch.allclose(ours, reference, atol=1e-6))


class InvalidClipInputTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ClipPreprocessor()

    def test_rejects_non_array_input(self):
        with self.assertRaises(TemporalPreprocessingError):
            self.preprocessor.preprocess("not a clip")

    def test_rejects_wrong_dimensionality(self):
        with self.assertRaises(TemporalPreprocessingError):
            self.preprocessor.preprocess(np.zeros((4, 64, 3), dtype=np.uint8))

    def test_rejects_empty_clip(self):
        with self.assertRaises(TemporalPreprocessingError):
            self.preprocessor.preprocess(np.zeros((0, 64, 64, 3), dtype=np.uint8))

    def test_rejects_non_three_channel_frames(self):
        with self.assertRaises(TemporalPreprocessingError):
            self.preprocessor.preprocess(np.zeros((2, 64, 64, 4), dtype=np.uint8))

    def test_rejects_non_uint8_input(self):
        with self.assertRaises(TemporalPreprocessingError):
            self.preprocessor.preprocess(np.zeros((2, 64, 64, 3), dtype=np.float32))


class PreprocessingConfigValidationTests(unittest.TestCase):
    def test_rejects_malformed_crop_size(self):
        with self.assertRaises(TemporalPreprocessingError):
            PreprocessingConfig(crop_size=(112,))

    def test_rejects_zero_std(self):
        with self.assertRaises(TemporalPreprocessingError):
            PreprocessingConfig(std=(0.0, 0.2, 0.2))


if __name__ == "__main__":
    unittest.main()
