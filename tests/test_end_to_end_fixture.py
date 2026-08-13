"""Integration verification for an optional, user-supplied person fixture."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from end_to_end_fixture import PERSON_FIXTURE_PATH, run_person_fixture


@unittest.skipUnless(
    PERSON_FIXTURE_PATH.is_file(),
    "Requires a real-person fixture at data/person_fixture.mp4; see README.",
)
class EndToEndPersonFixtureTests(unittest.TestCase):
    def test_detection_tracking_and_event_rules(self):
        result = run_person_fixture()

        self.assertGreater(result.frames_processed, 0)
        self.assertGreater(result.detection_count, 0)
        self.assertTrue(result.person_detected)
        self.assertGreater(result.tracked_object_count, 0)
        self.assertTrue(result.persistent_ids_produced)
        self.assertTrue(result.results_json_path.is_file())
        self.assertTrue(result.risk_assessment_path.is_file())

        event_types = sorted({event.event_type for event in result.events})
        print(
            "Fixture result: "
            f"detections={result.detection_count}, "
            f"person_detected={result.person_detected}, "
            f"tracked_objects={result.tracked_object_count}, "
            f"persistent_ids={result.persistent_ids_produced}, "
            f"events={event_types}"
        )


if __name__ == "__main__":
    unittest.main()
