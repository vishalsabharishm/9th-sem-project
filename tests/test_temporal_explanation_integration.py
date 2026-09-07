"""Integration tests for surfacing temporal Grad-CAM in the demo/dashboard.

These cover the seam between the explainer and whatever renders it. The unit
behaviour of Grad-CAM itself is pinned in tests/test_temporal_gradcam.py; what
matters here is that the payload reaching a UI cannot be rendered dishonestly:

  * a valid explanation reaches the data layer with frames still attached;
  * a non-contiguous or unusable regime is REFUSED with a stated reason rather
    than shown as a blank or fabricated panel;
  * the four kinds of number stay distinguishable, so a hardcoded 0.9 can never
    be drawn as model confidence;
  * the risk level always carries its unvalidated status;
  * faithfulness is still not claimed now that the map is visible;
  * the replay/live distinction survives -- replay has no model and therefore
    cannot produce saliency at all.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import temporal_gradcam  # noqa: E402
from temporal_explanation import (  # noqa: E402
    PROVENANCE_CONFIGURED,
    PROVENANCE_DECLARED_CONSTANT,
    PROVENANCE_GRADIENT_SALIENCY,
    PROVENANCE_MEASURED_PROBABILITY,
    RISK_STATUS_TEXT,
    SALIENCY_EXPLANATION,
    SALIENCY_HEADING,
    describe_provenance_legend,
    explain_frame_window,
    risk_presentation,
    unavailable,
)
from temporal_inference import TemporalInferenceConfig, TemporalInferenceEngine  # noqa: E402
from temporal_model import R3D18TemporalModel, TemporalModelConfig  # noqa: E402

THRESHOLD = 0.14


def build_engine():
    model = R3D18TemporalModel(TemporalModelConfig(num_classes=2, pretrained_backbone=False))
    return TemporalInferenceEngine(TemporalInferenceConfig(clip_length=16, stride=8), model=model)


def frames(count=16, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(count)]


class _AllowUntrained:
    """Bypass the provenance refusal so payload SHAPE can be tested.

    The refusal itself is asserted separately; without this the shape tests
    would need a real trained checkpoint, which is not in the repository.
    """

    def __enter__(self):
        self._original = temporal_gradcam._require_task_specific
        temporal_gradcam._require_task_specific = lambda model: None
        return self

    def __exit__(self, *exc):
        temporal_gradcam._require_task_specific = self._original
        return False


class ValidExplanationReachesTheDataLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with _AllowUntrained():
            cls.payload = explain_frame_window(
                build_engine(), frames(), list(range(24, 40)), THRESHOLD,
                source_height=240, source_width=320,
            )

    def test_payload_is_available_and_carries_a_heatmap(self):
        self.assertTrue(self.payload["available"])
        self.assertEqual(self.payload["heatmaps"].shape, (16, 112, 112))

    def test_temporal_evidence_has_everything_the_ui_must_show(self):
        evidence = self.payload["temporal_evidence"]
        for field in ("fight_probability", "decision", "decision_threshold",
                      "window_first_frame", "window_last_frame", "model_provenance"):
            self.assertIn(field, evidence)
        self.assertEqual(evidence["decision_threshold"], THRESHOLD)
        self.assertIn(evidence["decision"], ("Fight", "NonFight"))

    def test_decision_follows_the_threshold_passed_in(self):
        evidence = self.payload["temporal_evidence"]
        expected = "Fight" if evidence["fight_probability"] >= THRESHOLD else "NonFight"
        self.assertEqual(evidence["decision"], expected)


class TemporalAlignmentTests(unittest.TestCase):
    def test_source_frame_range_survives_into_the_payload(self):
        with _AllowUntrained():
            payload = explain_frame_window(
                build_engine(), frames(), list(range(24, 40)), THRESHOLD)
        evidence, saliency = payload["temporal_evidence"], payload["saliency"]
        self.assertEqual(evidence["window_first_frame"], 24)
        self.assertEqual(evidence["window_last_frame"], 39)
        self.assertEqual(saliency["frame_numbers"], list(range(24, 40)))
        self.assertIn(saliency["peak_frame"], range(24, 40))

    def test_raw_temporal_resolution_is_disclosed_not_implied(self):
        """16 displayed slices must not imply 16 resolved temporal positions."""
        with _AllowUntrained():
            payload = explain_frame_window(
                build_engine(), frames(), list(range(16)), THRESHOLD)
        saliency = payload["saliency"]
        self.assertEqual(saliency["displayed_temporal_slices"], 16)
        self.assertEqual(saliency["raw_temporal_positions"], 2)
        self.assertIn("interpolated", saliency["temporal_resolution_caveat"])
        self.assertIn("do not indicate frame-level temporal detail",
                      saliency["temporal_resolution_caveat"])

    def test_non_contiguous_regime_is_refused_with_a_reason(self):
        """Whole-clip sampling must not be given fabricated frame alignment."""
        sampled = list(range(0, 160, 10))
        self.assertEqual(len(sampled), 16)
        with _AllowUntrained():
            payload = explain_frame_window(build_engine(), frames(), sampled, THRESHOLD)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["reason"], "saliency_unavailable")
        self.assertIn("not consecutive", payload["detail"])
        self.assertNotIn("heatmaps", payload)

    def test_frame_count_mismatch_is_refused(self):
        payload = explain_frame_window(build_engine(), frames(8), list(range(16)), THRESHOLD)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["reason"], "frame_mismatch")


class GeometryReuseTests(unittest.TestCase):
    def test_source_geometry_comes_from_the_single_implementation(self):
        with _AllowUntrained():
            payload = explain_frame_window(
                build_engine(), frames(), list(range(16)), THRESHOLD,
                source_height=240, source_width=320)
        geometry = payload["saliency"]["source_geometry"]
        self.assertEqual(geometry["crop_origin_in_resized"], (8, 29))
        self.assertIn("never saw", geometry["caveat"])

    def test_no_second_geometry_implementation_exists(self):
        source = (REPO_ROOT / "src" / "temporal_explanation.py").read_text(encoding="utf-8")
        self.assertIn("to_source_frame", source)
        self.assertNotIn("center_crop", source)
        self.assertNotIn("resize(", source)


class ProvenanceSeparationTests(unittest.TestCase):
    def test_the_four_kinds_of_number_are_named_apart(self):
        legend = describe_provenance_legend()
        self.assertEqual(
            set(legend),
            {PROVENANCE_MEASURED_PROBABILITY, PROVENANCE_GRADIENT_SALIENCY,
             PROVENANCE_DECLARED_CONSTANT, PROVENANCE_CONFIGURED},
        )
        self.assertIn("NOT model confidence", legend[PROVENANCE_DECLARED_CONSTANT])

    def test_model_probability_is_tagged_as_measured(self):
        with _AllowUntrained():
            payload = explain_frame_window(
                build_engine(), frames(), list(range(16)), THRESHOLD)
        self.assertEqual(payload["temporal_evidence"]["fight_probability_provenance"],
                         PROVENANCE_MEASURED_PROBABILITY)

    def test_saliency_is_tagged_separately_from_probability(self):
        with _AllowUntrained():
            payload = explain_frame_window(
                build_engine(), frames(), list(range(16)), THRESHOLD)
        self.assertEqual(payload["saliency"]["provenance"], PROVENANCE_GRADIENT_SALIENCY)
        self.assertNotEqual(payload["saliency"]["provenance"],
                            payload["temporal_evidence"]["fight_probability_provenance"])

    def test_hardcoded_rule_confidence_is_never_a_model_probability(self):
        legend = describe_provenance_legend()
        declared = legend[PROVENANCE_DECLARED_CONSTANT]
        self.assertIn("0.9", declared)
        self.assertNotIn("model probability", declared.replace("NOT model confidence", ""))


class RiskPresentationTests(unittest.TestCase):
    def test_risk_level_always_carries_unvalidated_status(self):
        for level in ("Low", "Medium", "High"):
            presentation = risk_presentation(level)
            self.assertEqual(presentation["level"], level)
            self.assertFalse(presentation["risk_level_is_validated"])
            self.assertEqual(presentation["status_text"], RISK_STATUS_TEXT)
            self.assertEqual(presentation["provenance"], PROVENANCE_CONFIGURED)

    def test_probability_language_is_explicitly_forbidden(self):
        forbidden = risk_presentation("High")["forbidden_wording"]
        for phrase in ("risk probability", "risk confidence", "risk score"):
            self.assertIn(phrase, forbidden)

    def test_status_text_says_not_validated(self):
        self.assertIn("not validated as a risk model", RISK_STATUS_TEXT.lower())


class HonestyTests(unittest.TestCase):
    def test_faithfulness_stays_false_now_that_the_map_is_visible(self):
        with _AllowUntrained():
            payload = explain_frame_window(
                build_engine(), frames(), list(range(16)), THRESHOLD)
        self.assertFalse(payload["saliency"]["faithfulness_tested"])
        self.assertFalse(unavailable("x")["faithfulness_tested"])

    def test_ui_wording_makes_no_causal_or_attention_claim(self):
        text = f"{SALIENCY_HEADING} {SALIENCY_EXPLANATION}".lower()
        self.assertIn("saliency", text)
        self.assertIn("not a causal", text)
        for forbidden in ("attention", "proves", "ground truth localization"):
            self.assertNotIn(forbidden, text)

    def test_untrained_model_is_refused_at_the_integration_layer_too(self):
        """The provenance guard must not be bypassable by going through the payload."""
        payload = explain_frame_window(
            build_engine(), frames(), list(range(16)), THRESHOLD)
        self.assertFalse(payload["available"])
        self.assertIn("not task-specific", payload["detail"])


class ReplayCannotBeExplainedTests(unittest.TestCase):
    def test_replay_source_holds_no_model_to_explain(self):
        """CSV replay has a recorded probability but no weights and no tensor."""
        import run_demo
        source = run_demo.PrecomputedReplaySource
        self.assertFalse(hasattr(source, "model"))
        text = (REPO_ROOT / "tools" / "run_demo.py").read_text(encoding="utf-8")
        self.assertIn("class PrecomputedReplaySource", text)

    def test_unavailable_payload_states_a_reason_rather_than_being_blank(self):
        payload = unavailable("replay_has_no_model", "CSV replay carries no weights.")
        self.assertFalse(payload["available"])
        self.assertTrue(payload["reason"])
        self.assertTrue(payload["detail"])
        self.assertEqual(payload["heading"], SALIENCY_HEADING)


class GradCamStaysOffThePerFrameFathTests(unittest.TestCase):
    def test_every_explanation_entry_point_takes_a_window_not_a_frame(self):
        """The unit of explanation must be a window, so cost cannot go per-frame.

        Checking the property rather than a name list: explain_window is a
        legitimate re-export from temporal_gradcam and is itself window-level,
        so its presence is fine -- what would not be fine is any entry point
        that accepts a single frame.
        """
        import inspect
        import temporal_explanation

        entry_points = sorted(n for n in dir(temporal_explanation) if n.startswith("explain"))
        self.assertEqual(
            entry_points,
            ["explain_completed_window", "explain_frame_window", "explain_window"],
        )
        for name in entry_points:
            parameters = list(inspect.signature(
                getattr(temporal_explanation, name)).parameters)
            self.assertFalse(
                any(p in ("frame", "image", "img") for p in parameters),
                f"{name} accepts a single frame: {parameters}",
            )
            self.assertTrue(
                any(p in ("clip", "frames", "clip_tensor", "frame_numbers")
                    for p in parameters),
                f"{name} takes no window-shaped argument: {parameters}",
            )

    def test_run_demo_does_not_call_the_explainer_in_its_frame_loop(self):
        text = (REPO_ROOT / "tools" / "run_demo.py").read_text(encoding="utf-8")
        self.assertNotIn("explain_frame_window", text)
        self.assertNotIn("explain_completed_window", text)


if __name__ == "__main__":
    unittest.main()
