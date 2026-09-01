"""Regression tests for stale-track handling in the live demo path.

``SimpleTracker`` deliberately retains a track for ``max_age`` frames after its
last detection, keeping its final bounding box. That retention is useful to the
tracker itself, but a retained track has zero centroid motion, so anything that
reasons about *what is present now* must filter it out first:

- ``BehaviorAnalyzer`` scores the frozen box as stationary, so the rule engine
  reports a "Stationary Person" for an object that has already left the scene.
- ``draw_tracks`` renders the frozen box, so the annotated video shows a ghost
  detection sitting on empty ground.

``tracker.active_tracks`` is the filter, and ``tools/run_demo.py`` applies it to
both the rule snapshot and the drawing call. These tests pin that behaviour so
the demo cannot silently regress to feeding retained tracks to the rules again.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abnormal_event_detector import AbnormalEventDetector  # noqa: E402
from tracker import SimpleTracker, active_tracks, build_tracking_snapshot  # noqa: E402


PERSON_CLASS_ID = 0
DETECTED_FRAMES = 4
TOTAL_FRAMES = 14


def _moving_box(step: int) -> np.ndarray:
    """A person box that moves 20 px per frame -- never genuinely stationary."""
    x = 100.0 + 20.0 * step
    return np.array([[x, 100.0, x + 50.0, 220.0]])


def _run(total_frames: int, detected_frames: int, use_filter: bool):
    """Drive tracker + rules, optionally applying the active-track filter.

    The object is detected and moving for ``detected_frames`` frames, then the
    detector loses it completely. Because it was always moving while visible,
    any "Stationary Person" event is unambiguously an artifact of the retained
    frozen box, not genuine stationarity.
    """
    tracker = SimpleTracker()
    detector = AbnormalEventDetector()
    event_types = []
    drawn_per_frame = []

    for frame in range(1, total_frames + 1):
        if frame <= detected_frames:
            boxes = _moving_box(frame - 1)
            class_ids = np.array([PERSON_CLASS_ID])
            confidences = np.array([0.9])
        else:
            boxes = np.empty((0, 4))
            class_ids = np.array([])
            confidences = np.array([])

        returned = tracker.update(boxes, class_ids, confidences)
        tracks = active_tracks(returned, tracker.frame_idx) if use_filter else returned
        # run_demo.py hands the same list to the rule engine and to draw_tracks.
        drawn_per_frame.append(len(tracks))
        events = detector.process_snapshot(build_tracking_snapshot(frame, tracks))
        event_types.extend(event.event_type for event in events)

    return event_types, drawn_per_frame


class ActiveTrackFilterTests(unittest.TestCase):
    """Unit behaviour of the filter itself."""

    def test_keeps_only_tracks_seen_this_frame(self):
        tracker = SimpleTracker()
        tracker.update(_moving_box(0), np.array([PERSON_CLASS_ID]), np.array([0.9]))
        returned = tracker.update(np.empty((0, 4)), np.array([]), np.array([]))

        # The tracker still retains the lost track -- that is unchanged.
        self.assertEqual(len(returned), 1)
        self.assertLess(returned[0].last_seen, tracker.frame_idx)
        # The filter drops it.
        self.assertEqual(active_tracks(returned, tracker.frame_idx), [])

    def test_keeps_a_track_that_is_still_detected(self):
        tracker = SimpleTracker()
        tracker.update(_moving_box(0), np.array([PERSON_CLASS_ID]), np.array([0.9]))
        returned = tracker.update(_moving_box(1), np.array([PERSON_CLASS_ID]), np.array([0.9]))

        kept = active_tracks(returned, tracker.frame_idx)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].last_seen, tracker.frame_idx)

    def test_does_not_mutate_or_copy_track_objects(self):
        tracker = SimpleTracker()
        returned = tracker.update(_moving_box(0), np.array([PERSON_CLASS_ID]), np.array([0.9]))
        kept = active_tracks(returned, tracker.frame_idx)
        self.assertIs(kept[0], returned[0])

    def test_empty_input_is_empty_output(self):
        self.assertEqual(active_tracks([], 7), [])


class StaleTrackRegressionTests(unittest.TestCase):
    """The bug this filter exists to prevent, end to end through the rules."""

    def test_unfiltered_tracks_fabricate_stationary_events(self):
        """Documents the defect: without the filter, a departed object fires."""
        events, _ = _run(TOTAL_FRAMES, DETECTED_FRAMES, use_filter=False)
        self.assertIn(
            "Stationary Person",
            events,
            "Expected the unfiltered path to fabricate stationary events; if this "
            "fails, SimpleTracker's retention behaviour changed and the filter's "
            "rationale needs revisiting.",
        )

    def test_filtered_tracks_raise_no_stationary_event_for_a_departed_object(self):
        """The fix: an object that left the scene raises nothing."""
        events, _ = _run(TOTAL_FRAMES, DETECTED_FRAMES, use_filter=True)
        self.assertNotIn("Stationary Person", events)

    def test_filtering_strictly_reduces_fabricated_events(self):
        unfiltered, _ = _run(TOTAL_FRAMES, DETECTED_FRAMES, use_filter=False)
        filtered, _ = _run(TOTAL_FRAMES, DETECTED_FRAMES, use_filter=True)
        self.assertGreater(
            unfiltered.count("Stationary Person"), filtered.count("Stationary Person")
        )

    def test_no_boxes_are_drawn_after_the_object_leaves(self):
        """No ghost boxes: nothing is rendered once detections stop."""
        _, drawn = _run(TOTAL_FRAMES, DETECTED_FRAMES, use_filter=True)
        self.assertEqual(drawn[:DETECTED_FRAMES], [1] * DETECTED_FRAMES)
        self.assertEqual(
            drawn[DETECTED_FRAMES:],
            [0] * (TOTAL_FRAMES - DETECTED_FRAMES),
            "A retained track was still being drawn after its last detection.",
        )

    def test_unfiltered_path_draws_ghost_boxes(self):
        """Documents the defect on the drawing side."""
        _, drawn = _run(TOTAL_FRAMES, DETECTED_FRAMES, use_filter=False)
        self.assertTrue(
            any(count > 0 for count in drawn[DETECTED_FRAMES:]),
            "Expected retained tracks to be drawn on the unfiltered path.",
        )

    def test_a_genuinely_stationary_detected_person_still_fires(self):
        """The filter must not suppress real events: a person who is present
        every frame and not moving must still raise Stationary Person."""
        tracker = SimpleTracker()
        detector = AbnormalEventDetector()
        still = np.array([[300.0, 300.0, 350.0, 420.0]])
        events = []
        for frame in range(1, TOTAL_FRAMES + 1):
            returned = tracker.update(still, np.array([PERSON_CLASS_ID]), np.array([0.9]))
            tracks = active_tracks(returned, tracker.frame_idx)
            self.assertEqual(len(tracks), 1, "A detected person must never be filtered out.")
            events.extend(
                e.event_type
                for e in detector.process_snapshot(build_tracking_snapshot(frame, tracks))
            )
        self.assertIn("Stationary Person", events)


class RunDemoUsesTheFilterTests(unittest.TestCase):
    """The demo must apply the filter, not just have it available."""

    def test_run_demo_calls_active_tracks(self):
        source = (
            Path(__file__).resolve().parents[1] / "tools" / "run_demo.py"
        ).read_text(encoding="utf-8")
        # Asserted as booleans so a failure reports the diagnosis rather than
        # dumping the whole module into the test output.
        self.assertTrue(
            "active_tracks(" in source,
            "tools/run_demo.py no longer calls active_tracks(). Retained tracks "
            "would reach the rule engine and be drawn as ghost boxes again.",
        )
        self.assertFalse(
            "tracks = tracker.update(" in source,
            "tools/run_demo.py assigns tracker.update() output directly to "
            "`tracks` again; it must wrap it in active_tracks() before the rule "
            "snapshot and draw_tracks.",
        )


if __name__ == "__main__":
    unittest.main()
