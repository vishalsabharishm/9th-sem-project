import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abnormal_event_detector import AbnormalEventDetector
from behavior_analyzer import BehaviorAnalyzer
from event_rules import EventRule, EventRuleEngine
from tracker import Track, TrackingSnapshot


class AbnormalEventDetectionTests(unittest.TestCase):
    def test_stationary_and_roi_rules(self):
        detector = AbnormalEventDetector()
        rules = [
            EventRule(name="stationary", description="Person remains stationary", parameters={"stationary_frame_threshold": 5, "movement_threshold_pixels": 3}),
            EventRule(name="restricted_area", description="Person enters restricted region", parameters={"restricted_region": (0, 0, 100, 100)}),
        ]
        engine = EventRuleEngine(rules)
        detector.rule_engine = engine

        track = Track(id=1, bbox=[0, 0, 20, 20], class_id=0, conf=0.95, last_seen=0, hits=1, history=[(10, 10)])
        snapshot = TrackingSnapshot(frame_idx=1, tracks=[track])

        for _ in range(5):
            detector.process_snapshot(snapshot)

        events = detector.process_snapshot(snapshot)
        self.assertTrue(any(event.event_type == "Stationary Person" for event in events))

        roi_track = Track(id=2, bbox=[10, 10, 30, 30], class_id=0, conf=0.95, last_seen=0, hits=1, history=[(10, 10)])
        roi_snapshot = TrackingSnapshot(frame_idx=2, tracks=[roi_track])
        events = detector.process_snapshot(roi_snapshot)
        self.assertTrue(any(event.event_type == "Restricted Area Entry" for event in events))


if __name__ == "__main__":
    unittest.main()
