"""Tests for Grad-CAM over the R3D-18 temporal violence decision.

The point of this module is that the system can say something about WHY it
called a clip violent. The pre-existing YOLO Grad-CAM cannot do that -- it
explains a person detection, and the violence decision is made entirely by the
temporal branch. So these tests pin the properties that make the temporal map
worth showing to anyone:

  * it is computed from the ACTUAL task logit, not a proxy;
  * every heatmap slice is attributable to a specific SOURCE FRAME;
  * it refuses to explain a model whose head was never trained for the task;
  * it refuses the whole-clip regime, whose frames are not consecutive and
    therefore cannot be aligned slice-to-frame;
  * it never claims faithfulness, which has not been tested.

No explainability metric is asserted, because none is computed.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from temporal_gradcam import (  # noqa: E402
    TemporalGradCAMError,
    explain_window,
    to_source_frame,
)
from temporal_model import R3D18TemporalModel, TemporalModelConfig  # noqa: E402


def build_model():
    return R3D18TemporalModel(TemporalModelConfig(num_classes=2, pretrained_backbone=False))


def clip(frames=16):
    torch.manual_seed(0)
    return torch.randn(1, 3, frames, 112, 112)


class ProvenanceGuardTests(unittest.TestCase):
    def test_untrained_head_is_refused(self):
        """A random head still yields a confident-looking map. Refuse it."""
        with self.assertRaises(TemporalGradCAMError) as caught:
            explain_window(build_model(), clip(), list(range(16)))
        self.assertIn("not task-specific", str(caught.exception))

    def test_model_without_provenance_is_refused(self):
        class Bare:
            model = None
            feature_layer = None
        with self.assertRaises(TemporalGradCAMError) as caught:
            explain_window(Bare(), clip(), list(range(16)))
        self.assertIn("no provenance", str(caught.exception))

    def test_guard_accepts_property_or_callable_flag(self):
        """R3D18TemporalModel exposes a bool property; predictions use a field."""
        model = build_model()
        self.assertIsInstance(model.is_task_specific, bool)
        self.assertFalse(model.is_task_specific)


class AlignmentTests(unittest.TestCase):
    def test_each_slice_maps_to_its_source_frame(self):
        result = explain_window(build_model(), clip(), list(range(128, 144)),
                                require_task_specific=False)
        self.assertEqual(result.heatmaps.shape, (16, 112, 112))
        self.assertEqual(result.frame_numbers[0], 128)
        self.assertEqual(result.frame_numbers[-1], 143)
        self.assertEqual(len(result.frame_numbers), result.heatmaps.shape[0])
        self.assertIn(result.peak_frame(), range(128, 144))

    def test_non_consecutive_frames_are_refused(self):
        """The whole-clip regime samples across 150 frames and cannot be aligned."""
        sampled = list(range(0, 160, 10))   # 16 frames spread across the clip
        self.assertEqual(len(sampled), 16, "fixture must hit the consecutive guard, "
                         "not the frame-count guard")
        with self.assertRaises(TemporalGradCAMError) as caught:
            explain_window(build_model(), clip(), sampled, require_task_specific=False)
        self.assertIn("not consecutive", str(caught.exception))

    def test_frame_count_must_match_the_tensor(self):
        with self.assertRaises(TemporalGradCAMError) as caught:
            explain_window(build_model(), clip(), list(range(8)),
                           require_task_specific=False)
        self.assertIn("could not be aligned", str(caught.exception))

    def test_temporal_downsampling_is_recorded_not_hidden(self):
        """Upsampling to 16 slices must not disguise 2 real temporal positions."""
        result = explain_window(build_model(), clip(), list(range(16)),
                                require_task_specific=False)
        self.assertEqual(result.raw_temporal_positions, 2)
        self.assertEqual(result.frames_per_raw_position, 8)

    def test_rejects_a_batch(self):
        with self.assertRaises(TemporalGradCAMError):
            explain_window(build_model(), torch.randn(2, 3, 16, 112, 112),
                           list(range(16)), require_task_specific=False)


class SaliencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = explain_window(build_model(), clip(), list(range(16)),
                                    require_task_specific=False)

    def test_heatmaps_are_normalised_and_non_negative(self):
        self.assertGreaterEqual(float(self.result.heatmaps.min()), 0.0)
        self.assertAlmostEqual(float(self.result.heatmaps.max()), 1.0, places=5)

    def test_explanation_targets_the_fight_logit(self):
        self.assertEqual(self.result.class_index, 1)
        self.assertEqual(self.result.class_label, "Fight")
        self.assertTrue(0.0 <= self.result.probability <= 1.0)

    def test_a_different_class_gives_a_different_map(self):
        """Evidence the map really is gradient-derived, not an activation blob."""
        other = explain_window(build_model(), clip(), list(range(16)), class_index=0,
                               class_label="NonFight", require_task_specific=False)
        self.assertFalse(np.allclose(self.result.heatmaps, other.heatmaps),
                         "Fight and NonFight produced identical maps")

    def test_is_deterministic(self):
        again = explain_window(build_model(), clip(), list(range(16)),
                               require_task_specific=False)
        np.testing.assert_allclose(self.result.heatmaps, again.heatmaps, rtol=0, atol=0)

    def test_model_is_left_in_eval_mode_with_clean_gradients(self):
        model = build_model()
        model.model.eval()
        explain_window(model, clip(), list(range(16)), require_task_specific=False)
        self.assertFalse(model.model.training)
        for parameter in model.model.parameters():
            self.assertTrue(parameter.grad is None or torch.all(parameter.grad == 0))


class HonestyTests(unittest.TestCase):
    def test_faithfulness_is_never_claimed(self):
        result = explain_window(build_model(), clip(), list(range(16)),
                                require_task_specific=False)
        self.assertFalse(result.faithfulness_tested)
        payload = result.as_dict()
        self.assertFalse(payload["faithfulness_tested"])
        self.assertIn("not evidence that the", payload["interpretation_limit"])
        self.assertIn("diagnostic", payload["interpretation_limit"])

    def test_provenance_travels_with_the_explanation(self):
        result = explain_window(build_model(), clip(), list(range(16)),
                                require_task_specific=False)
        self.assertEqual(result.provenance, "random_init")
        self.assertEqual(result.as_dict()["provenance"], "random_init")


class GeometryTests(unittest.TestCase):
    def test_crop_offset_and_scale_are_inverted(self):
        mapping = to_source_frame(np.zeros((112, 112)), 240, 320)
        # resize (128, 171), crop 112 -> offsets (8, 29)
        self.assertEqual(mapping["crop_origin_in_resized"], (8, 29))
        top, left, bottom, right = mapping["visible_region_in_source"]
        self.assertLess(top, bottom)
        self.assertLess(left, right)
        self.assertLessEqual(bottom, 240)
        self.assertLessEqual(right, 320)

    def test_discarded_border_is_reported(self):
        mapping = to_source_frame(np.zeros((112, 112)), 240, 320)
        border = mapping["discarded_border_pixels_in_resized"]
        self.assertEqual(border["top"] + border["bottom"], 128 - 112)
        self.assertEqual(border["left"] + border["right"], 171 - 112)
        self.assertIn("never saw", mapping["caveat"])

    def test_wrong_heatmap_shape_is_refused(self):
        with self.assertRaises(TemporalGradCAMError):
            to_source_frame(np.zeros((64, 64)), 240, 320)


class YoloGradCamIsNotAViolenceExplanationTests(unittest.TestCase):
    """The distinction that motivated this module, pinned so it is not lost."""

    def test_yolo_gradcam_does_not_import_the_temporal_model(self):
        source = (REPO_ROOT / "src" / "yolo_gradcam.py").read_text(encoding="utf-8")
        self.assertNotIn("temporal_model", source)
        self.assertNotIn("R3D", source)

    def test_temporal_gradcam_hooks_the_temporal_feature_layer(self):
        source = (REPO_ROOT / "src" / "temporal_gradcam.py").read_text(encoding="utf-8")
        self.assertIn("feature_layer", source)
        self.assertIn("register_forward_hook", source)


if __name__ == "__main__":
    unittest.main()
