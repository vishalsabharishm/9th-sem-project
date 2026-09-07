"""Tests that the risk layer cannot be mistaken for a validated risk model.

The layer maps an event type to Low / Medium / High through a fixed dictionary.
That is a useful routing convenience and a perfectly reasonable engineering
choice, but it is not calibrated, not learned, and not validated against any
outcome -- RWF-2000 carries no risk ground truth, so it could not be. The danger
is presentational: a "High" badge in a dashboard reads like an estimated
probability of harm.

These tests pin the separation of the four concerns and ensure the caveat
travels inside every emitted record, rather than living only in documentation
that a UI author may never read.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from abnormal_event_detector import EventDetection  # noqa: E402
from risk_assessment import (  # noqa: E402
    DEFAULT_RISK_MAPPING,
    RISK_LAYER_DECLARATION,
    RiskAssessor,
    describe_risk_layer,
)
from temporal_event_adapter import (  # noqa: E402
    PROVENANCE_DECLARED,
    PROVENANCE_MEASURED,
    TEMPORAL_EVENT_TYPE,
)


class LayerSeparationTests(unittest.TestCase):
    def test_the_four_concerns_are_named_separately(self):
        layers = RISK_LAYER_DECLARATION["layers"]
        self.assertEqual(
            set(layers),
            {"1_event_detection", "2_evidence", "3_confidence", "4_risk_interpretation"},
        )

    def test_risk_interpretation_is_declared_unevaluated(self):
        """The dictionary mapping is the one layer with no empirical support."""
        interpretation = RISK_LAYER_DECLARATION["layers"]["4_risk_interpretation"]
        self.assertFalse(interpretation["empirically_evaluated"])
        self.assertIn("NOT a risk model", interpretation["note"])

    def test_detection_layer_records_that_spatial_rules_were_never_evaluated(self):
        note = RISK_LAYER_DECLARATION["layers"]["1_event_detection"]["note"]
        self.assertIn("NO violence ground truth", note)

    def test_confidence_layer_records_the_hardcoded_constants(self):
        note = RISK_LAYER_DECLARATION["layers"]["3_confidence"]["note"]
        self.assertIn("HARDCODED", note)
        self.assertIn("0.9", note)

    def test_declaration_states_it_is_not_a_validated_model(self):
        self.assertFalse(RISK_LAYER_DECLARATION["is_validated_risk_model"])
        self.assertFalse(RISK_LAYER_DECLARATION["risk_ground_truth_available"])
        forbidden = RISK_LAYER_DECLARATION["what_may_not_be_claimed"]
        for word in ("calibrated", "probabilistic", "validated"):
            self.assertIn(word, forbidden)

    def test_limitation_is_structural_not_pending(self):
        self.assertIn("structural", RISK_LAYER_DECLARATION["to_validate_this_layer"])

    def test_describe_returns_a_copy_callers_cannot_corrupt(self):
        first = describe_risk_layer()
        first["is_validated_risk_model"] = True
        self.assertFalse(describe_risk_layer()["is_validated_risk_model"])


class EmittedRecordTests(unittest.TestCase):
    """The caveat must be in the data, not only in the docs."""

    def _assess(self, event_type, confidence=None):
        event = EventDetection(event_type=event_type, description="x", confidence=confidence)
        return RiskAssessor().assess(event).as_dict()

    def test_every_record_declares_the_risk_level_unvalidated(self):
        for event_type in DEFAULT_RISK_MAPPING:
            record = self._assess(event_type, 0.9)
            self.assertFalse(record["risk_level_is_validated"], event_type)
            self.assertIn("not calibrated", record["risk_level_basis"], event_type)

    def test_declared_constants_are_not_reported_as_model_confidence(self):
        record = self._assess("Stationary Person", 0.9)
        self.assertEqual(record["confidence"], 0.9)
        self.assertEqual(record["confidence_provenance"], PROVENANCE_DECLARED)

    def test_temporal_confidence_is_marked_measured(self):
        record = self._assess(TEMPORAL_EVENT_TYPE, 0.83)
        self.assertEqual(record["confidence_provenance"], PROVENANCE_MEASURED)

    def test_rules_without_confidence_do_not_gain_one(self):
        for event_type in ("Crowding", "Proximity/Interaction"):
            record = self._assess(event_type, None)
            self.assertIsNone(record["confidence"], event_type)
            self.assertIsNone(record["confidence_provenance"], event_type)

    def test_unknown_event_type_is_not_silently_assigned_a_risk(self):
        record = self._assess("Something Unmapped", 0.5)
        self.assertEqual(record["risk_level"], "Unknown")
        self.assertIn("No risk mapping", record["reason"])

    def test_assessor_invents_no_confidence(self):
        """A risk level must never become a number the model did not produce."""
        record = self._assess("Crowding", None)
        self.assertIsNone(record["confidence"])
        self.assertIn("risk_level", record)
        self.assertNotIn("risk_score", record)
        self.assertNotIn("probability", record)


class NoProbabilityLanguageTests(unittest.TestCase):
    def test_risk_levels_are_ordinal_labels_not_numbers(self):
        for level in DEFAULT_RISK_MAPPING.values():
            self.assertIn(level, {"Low", "Medium", "High"})
            self.assertNotIsInstance(level, (int, float))

    def test_module_exposes_no_risk_probability_helper(self):
        import risk_assessment
        for name in dir(risk_assessment):
            self.assertNotIn("probab", name.lower(),
                             f"{name} suggests a probability the layer cannot produce")


if __name__ == "__main__":
    unittest.main()
