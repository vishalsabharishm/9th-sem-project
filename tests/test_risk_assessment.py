import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abnormal_event_detector import EventDetection
from risk_assessment import RiskAssessor


class RiskAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.assessor = RiskAssessor()

    def test_stationary_person_maps_to_low_risk(self):
        assessment = self.assessor.assess(
            EventDetection("Stationary Person", "Person remained stationary.", confidence=0.9),
            frame_number=12,
        )
        self.assertEqual(assessment.risk_level, "Low")
        self.assertEqual(assessment.confidence, 0.9)
        self.assertEqual(assessment.frame_number, 12)

    def test_restricted_area_entry_maps_to_medium_risk(self):
        assessment = self.assessor.assess(
            EventDetection("Restricted Area Entry", "Person entered a restricted region.", confidence=0.95)
        )
        self.assertEqual(assessment.risk_level, "Medium")
        self.assertEqual(assessment.confidence, 0.95)

    def test_unknown_event_has_safe_default_handling(self):
        assessment = self.assessor.assess(EventDetection("Unmapped Event", "No configured mapping."))
        self.assertEqual(assessment.risk_level, "Unknown")
        self.assertIsNone(assessment.confidence)
        self.assertIn("No risk mapping", assessment.reason)


if __name__ == "__main__":
    unittest.main()
