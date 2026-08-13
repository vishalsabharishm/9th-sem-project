import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abnormal_event_detector import AbnormalEventDetector
from event_rules import EventRule, EventRuleEngine
from tracker import Track, TrackingSnapshot


def person(track_id, bbox):
    return Track(id=track_id, bbox=bbox, class_id=0, conf=0.9, last_seen=0)


class CrowdInteractionTests(unittest.TestCase):
    def _detector(self, rules):
        return AbnormalEventDetector(EventRuleEngine(rules))

    def test_crowd_below_threshold_has_no_event(self):
        detector = self._detector([EventRule("crowding", "", parameters={"minimum_person_count": 3, "persistence_frames": 1})])
        events = detector.process_snapshot(TrackingSnapshot(1, [person(1, [0, 0, 10, 10]), person(2, [20, 0, 30, 10])]))
        self.assertFalse(any(event.event_type == "Crowding" for event in events))

    def test_crowd_threshold_and_configuration_trigger_once(self):
        detector = self._detector([EventRule("crowding", "", parameters={"minimum_person_count": 2, "persistence_frames": 2})])
        snapshot = TrackingSnapshot(1, [person(1, [0, 0, 10, 10]), person(2, [20, 0, 30, 10])])
        self.assertFalse(detector.process_snapshot(snapshot))
        events = detector.process_snapshot(TrackingSnapshot(2, snapshot.tracks))
        crowding = [event for event in events if event.event_type == "Crowding"]
        self.assertEqual(len(crowding), 1)
        self.assertEqual(crowding[0].object_ids, [1, 2])
        self.assertEqual(crowding[0].frame_number, 2)
        self.assertIsNone(crowding[0].confidence)
        self.assertFalse(any(event.event_type == "Crowding" for event in detector.process_snapshot(TrackingSnapshot(3, snapshot.tracks))))

    def test_proximity_below_threshold_has_no_event(self):
        detector = self._detector([EventRule("proximity_interaction", "", parameters={"normalized_distance_threshold": 0.1, "persistence_frames": 1})])
        events = detector.process_snapshot(TrackingSnapshot(1, [person(1, [0, 0, 10, 10]), person(2, [100, 0, 110, 10])]))
        self.assertFalse(any(event.event_type == "Proximity/Interaction" for event in events))

    def test_proximity_sustained_triggers_once(self):
        detector = self._detector([EventRule("proximity_interaction", "", parameters={"normalized_distance_threshold": 0.5, "persistence_frames": 2})])
        tracks = [person(4, [0, 0, 10, 10]), person(9, [12, 0, 22, 10])]
        self.assertFalse(detector.process_snapshot(TrackingSnapshot(1, tracks)))
        events = detector.process_snapshot(TrackingSnapshot(2, tracks))
        proximity = [event for event in events if event.event_type == "Proximity/Interaction"]
        self.assertEqual(len(proximity), 1)
        self.assertEqual(proximity[0].object_ids, [4, 9])
        self.assertIn("normalized_centroid_distance", proximity[0].evidence[0])


if __name__ == "__main__":
    unittest.main()
