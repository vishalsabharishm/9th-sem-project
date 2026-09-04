"""Tests for the R3D runtime seam and the shared engine factory.

Covers integration Phases 1 and 2:

- Phase 1: ``temporal_event_adapter.window_score_from`` -- the converter that
  lets a live inference result reach the frozen aggregation rule at all.
- Phase 2: ``temporal_runtime`` -- the single place that builds and validates
  the violence model, so the demo and the offline scorer cannot drift apart.

Nothing here needs the checkpoint. Checkpoint-shaped payloads are synthesised
with torch.save; a real R3D-18 checkpoint is 132 MB and would make the suite
unusable, and what these tests exercise is the guard and the configuration,
both of which read metadata rather than weights.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from checkpoint_identity import CheckpointIntegrityError  # noqa: E402
from rwf2000_config import CANONICAL_LABELS  # noqa: E402
from temporal_event_adapter import (  # noqa: E402
    FIGHT_CLASS_INDEX,
    PROVENANCE_LIVE_INFERENCE,
    PROVENANCE_MEASURED,
    TemporalAdapterError,
    WindowScore,
    window_score_from,
)
from temporal_runtime import (  # noqa: E402
    RUNTIME_CLIP_LENGTH,
    RUNTIME_CLASS_LABELS,
    RUNTIME_NUM_CLASSES,
    RUNTIME_POSITIVE_CLASS_INDEX,
    RUNTIME_STRIDE,
    TemporalRuntimeError,
    build_violence_engine,
    build_violence_model,
    describe_runtime,
    verified_checkpoint,
    violence_model_config,
)


class FakePrediction:
    """Minimal stand-in for TemporalPrediction: the fields the seam reads."""

    def __init__(self, probabilities, is_task_specific=True, provenance="checkpoint:best.pt"):
        self.probabilities = probabilities
        self.is_task_specific = is_task_specific
        self.provenance = provenance


class FakeResult:
    """Minimal stand-in for TemporalInferenceResult."""

    def __init__(self, frame_numbers, prediction):
        self.frame_numbers = frame_numbers
        self.prediction = prediction


def result_for(first, last, fight_probability, **kwargs):
    frames = list(range(first, last + 1))
    probabilities = torch.tensor([[1.0 - fight_probability, fight_probability]])
    return FakeResult(frames, FakePrediction(probabilities, **kwargs))


# --------------------------------------------------------------------------
# Phase 1
# --------------------------------------------------------------------------


class WindowScoreConversionTests(unittest.TestCase):
    """The four-field mapping, exactly."""

    def test_all_four_fields_map_as_specified(self):
        score = window_score_from(result_for(8, 23, 0.75), window_index=1)
        self.assertIsInstance(score, WindowScore)
        self.assertEqual(score.window_index, 1)
        self.assertEqual(score.first_frame, 8)
        self.assertEqual(score.last_frame, 23)
        self.assertAlmostEqual(score.fight_probability, 0.75, places=6)

    def test_fight_is_the_positive_class_at_index_one(self):
        """Reading column 0 would silently invert every decision."""
        self.assertEqual(FIGHT_CLASS_INDEX, 1)
        probabilities = torch.tensor([[0.9, 0.1]])  # NonFight 0.9, Fight 0.1
        score = window_score_from(FakeResult([0, 15], FakePrediction(probabilities)), 0)
        self.assertAlmostEqual(score.fight_probability, 0.1, places=6)

    def test_frame_numbers_are_preserved_from_the_engine(self):
        score = window_score_from(result_for(128, 143, 0.5), window_index=16)
        self.assertEqual((score.first_frame, score.last_frame), (128, 143))

    def test_window_index_is_the_callers_counter_not_derived(self):
        score = window_score_from(result_for(0, 15, 0.5), window_index=99)
        self.assertEqual(score.window_index, 99)

    def test_probability_is_a_plain_float_not_a_tensor(self):
        score = window_score_from(result_for(0, 15, 0.5), 0)
        self.assertIsInstance(score.fight_probability, float)

    def test_the_committed_window_geometry_round_trips(self):
        """A full clip's worth of windows converts to the recorded bounds."""
        bounds = [(i * 8, i * 8 + 15) for i in range(17)]
        scores = [
            window_score_from(result_for(first, last, 0.3), index)
            for index, (first, last) in enumerate(bounds)
        ]
        self.assertEqual([(s.first_frame, s.last_frame) for s in scores], bounds)
        self.assertEqual([s.window_index for s in scores], list(range(17)))


class ConversionGuardTests(unittest.TestCase):
    """A non-task-specific prediction must not reach the aggregation rule."""

    def test_untrained_prediction_is_rejected_by_default(self):
        result = result_for(0, 15, 0.9, is_task_specific=False, provenance="random_init")
        with self.assertRaises(TemporalAdapterError) as caught:
            window_score_from(result, 0)
        self.assertIn("non-task-specific", str(caught.exception))

    def test_the_rejection_names_the_provenance(self):
        result = result_for(0, 15, 0.9, is_task_specific=False, provenance="random_init")
        with self.assertRaises(TemporalAdapterError) as caught:
            window_score_from(result, 0)
        self.assertIn("random_init", str(caught.exception))

    def test_the_guard_can_be_waived_explicitly_for_plumbing_tests(self):
        result = result_for(0, 15, 0.9, is_task_specific=False, provenance="random_init")
        score = window_score_from(result, 0, require_task_specific=False)
        self.assertAlmostEqual(score.fight_probability, 0.9, places=6)

    def test_missing_frame_numbers_are_rejected(self):
        with self.assertRaises(TemporalAdapterError):
            window_score_from(FakeResult([], FakePrediction(torch.tensor([[0.5, 0.5]]))), 0)

    def test_missing_prediction_is_rejected(self):
        with self.assertRaises(TemporalAdapterError):
            window_score_from(FakeResult([0, 15], None), 0)

    def test_wrongly_shaped_probabilities_are_rejected(self):
        result = FakeResult([0, 15], FakePrediction(torch.tensor([[0.5]])))
        with self.assertRaises(TemporalAdapterError):
            window_score_from(result, 0)

    def test_live_and_replay_provenance_are_distinguishable(self):
        self.assertNotEqual(PROVENANCE_LIVE_INFERENCE, PROVENANCE_MEASURED)
        self.assertIn("live", PROVENANCE_LIVE_INFERENCE)


class RealEngineConversionTests(unittest.TestCase):
    """The seam against the actual engine, with untrained weights.

    Uses require_task_specific=False deliberately: this checks the plumbing and
    the window geometry, and asserts nothing about the values produced.
    """

    def test_engine_results_convert_to_the_committed_geometry(self):
        import numpy as np
        from temporal_inference import TemporalInferenceConfig, TemporalInferenceEngine
        from temporal_model import TemporalModelConfig

        engine = TemporalInferenceEngine(
            TemporalInferenceConfig(
                clip_length=RUNTIME_CLIP_LENGTH,
                stride=RUNTIME_STRIDE,
                model=TemporalModelConfig(num_classes=2, device="cpu", pretrained_backbone=False),
            )
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        scores = []
        for index in range(48):
            result = engine.add_frame(frame, frame_number=index)
            if result is not None:
                scores.append(
                    window_score_from(result, len(scores), require_task_specific=False)
                )
        self.assertEqual([(s.first_frame, s.last_frame) for s in scores],
                         [(0, 15), (8, 23), (16, 31), (24, 39), (32, 47)])
        for score in scores:
            self.assertTrue(0.0 <= score.fight_probability <= 1.0)


# --------------------------------------------------------------------------
# Phase 2
# --------------------------------------------------------------------------


def trained_payload(**overrides):
    """A payload the Step-3 guard accepts."""
    payload = {
        "state_dict": {"fc.weight": torch.zeros(2, 4)},
        "epoch": 12,
        "monitor": "roc_auc",
        "monitored_value": 0.9353,
        "metrics": {"accuracy": 0.875},
        "config": {"protocol": "carved_validation", "pretrained_backbone": True,
                   "limit_train_clips": None, "limit_eval_clips": None},
        "num_classes": 2,
        "class_labels": list(CANONICAL_LABELS),
        "training_provenance": "kinetics400_pretrained+untrained_head",
        "protocol": "carved_validation",
    }
    payload.update(overrides)
    return payload


class RuntimeConstantTests(unittest.TestCase):
    """The regime constants must match the committed manifest."""

    def test_clip_length_and_stride_match_the_manifest(self):
        import json

        manifest = json.loads(
            (REPO_ROOT / "temporal_risk" / "window_scoring_manifest.json").read_text()
        )
        self.assertEqual(RUNTIME_CLIP_LENGTH, manifest["clip_length"])
        self.assertEqual(RUNTIME_STRIDE, manifest["stride"])

    def test_class_labels_match_the_manifest_and_are_canonical(self):
        import json

        manifest = json.loads(
            (REPO_ROOT / "temporal_risk" / "window_scoring_manifest.json").read_text()
        )
        self.assertEqual(list(RUNTIME_CLASS_LABELS), manifest["class_labels"])
        self.assertEqual(RUNTIME_CLASS_LABELS, tuple(CANONICAL_LABELS))

    def test_two_classes_never_kinetics_four_hundred(self):
        self.assertEqual(RUNTIME_NUM_CLASSES, 2)

    def test_positive_class_index_is_fight(self):
        self.assertEqual(RUNTIME_POSITIVE_CLASS_INDEX, 1)
        self.assertEqual(RUNTIME_CLASS_LABELS[RUNTIME_POSITIVE_CLASS_INDEX], "Fight")

    def test_describe_runtime_reports_the_regime(self):
        described = describe_runtime()
        self.assertEqual(described["clip_length"], 16)
        self.assertEqual(described["stride"], 8)
        self.assertEqual(described["num_classes"], 2)
        self.assertEqual(described["class_labels"], ["NonFight", "Fight"])


class ModelConfigTests(unittest.TestCase):
    def test_config_pins_the_violence_head_not_the_kinetics_default(self):
        """TemporalModelConfig() alone defaults to 400; the factory must not."""
        from temporal_model import TemporalModelConfig

        self.assertEqual(TemporalModelConfig().num_classes, 400)
        config = violence_model_config(Path("nowhere.pt"), "cpu")
        self.assertEqual(config.num_classes, 2)
        self.assertEqual(config.class_labels, RUNTIME_CLASS_LABELS)

    def test_config_does_not_request_a_kinetics_download(self):
        self.assertFalse(violence_model_config(Path("x.pt"), "cpu").pretrained_backbone)

    def test_config_carries_the_checkpoint_path_and_device(self):
        config = violence_model_config(Path("some/best.pt"), "cpu")
        self.assertEqual(config.checkpoint_path, Path("some/best.pt"))
        self.assertEqual(config.device, "cpu")


class GuardEnforcementTests(unittest.TestCase):
    """Every builder must run the Step-3 checkpoint guard first."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, payload, name="best.pt"):
        path = self.tmp / name
        torch.save(payload, path)
        return path

    def test_missing_checkpoint_is_rejected(self):
        for builder in (verified_checkpoint, build_violence_model, build_violence_engine):
            with self.subTest(builder=builder.__name__):
                with self.assertRaises(CheckpointIntegrityError):
                    builder(self.tmp / "does_not_exist.pt")

    def test_random_init_checkpoint_is_rejected(self):
        path = self._write(trained_payload(training_provenance="random_init"))
        for builder in (build_violence_model, build_violence_engine):
            with self.subTest(builder=builder.__name__):
                with self.assertRaises(CheckpointIntegrityError) as caught:
                    builder(path)
                self.assertIn("randomly initialised", str(caught.exception))

    def test_smoke_test_subset_checkpoint_is_rejected(self):
        payload = trained_payload()
        payload["config"]["limit_train_clips"] = 4
        path = self._write(payload)
        with self.assertRaises(CheckpointIntegrityError):
            build_violence_engine(path)

    def test_checkpoint_with_no_monitored_value_is_rejected(self):
        path = self._write(trained_payload(monitored_value=None))
        with self.assertRaises(CheckpointIntegrityError):
            build_violence_model(path)

    def test_a_bare_state_dict_is_rejected(self):
        path = self.tmp / "bare.pt"
        torch.save({"fc.weight": torch.zeros(2, 4)}, path)
        with self.assertRaises(CheckpointIntegrityError):
            build_violence_engine(path)

    def test_an_acceptable_checkpoint_passes_the_guard(self):
        """The guard accepts; loading then fails on the synthetic state dict.

        Separating the two is the point: verification reads metadata, and a
        payload that passes it is not required to contain real R3D-18 weights.
        """
        path = self._write(trained_payload())
        verdict = verified_checkpoint(path)
        self.assertTrue(verdict.accepted)
        with self.assertRaises(TemporalRuntimeError):
            build_violence_model(path)


class NoDuplicateRegimeTests(unittest.TestCase):
    """The offline scorer must not keep its own model construction."""

    def test_score_temporal_windows_uses_the_shared_factory(self):
        source = (REPO_ROOT / "tools" / "score_temporal_windows.py").read_text(encoding="utf-8")
        self.assertIn("build_violence_model", source)

    def test_score_temporal_windows_no_longer_builds_a_model_itself(self):
        source = (REPO_ROOT / "tools" / "score_temporal_windows.py").read_text(encoding="utf-8")
        self.assertNotIn("TemporalModelConfig(", source)
        self.assertNotIn("num_classes=2", source)


if __name__ == "__main__":
    unittest.main()
