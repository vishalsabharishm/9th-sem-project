"""Tests for window-scoring geometry and time-to-alarm.

Both tools read committed artifacts only, so these tests need neither the
extracted dataset nor a checkpoint. Two of them are the substantive
reproducibility checks this step exists to add:

- the committed CSVs' window bounds really are what this project's
  ``TemporalClipBuffer`` produces (previously assumed, never verified);
- time-to-alarm, derived independently, reproduces the detection counts the
  frozen aggregation result already records.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from score_temporal_windows import (  # noqa: E402
    read_recorded_bounds,
    verify_geometry,
    window_bounds,
)
from time_to_alarm import SOURCE_FPS, alarm_frame, compute  # noqa: E402
from temporal_event_adapter import FrozenAggregationRule, WindowScore  # noqa: E402

PRIMARY_CSV = REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"
CARVE_CSV = REPO_ROOT / "temporal_risk" / "carve_window_scores.csv"
FROZEN_JSON = REPO_ROOT / "temporal_risk" / "frozen_aggregation.json"
SOURCE_FRAMES = 150


class WindowGeometryTests(unittest.TestCase):
    """The buffer's windows must match the recorded ones exactly."""

    def test_150_frames_at_16_stride_8_gives_17_windows(self):
        bounds = window_bounds(SOURCE_FRAMES, 16, 8)
        self.assertEqual(len(bounds), 17)
        self.assertEqual(bounds[0], (0, 15))
        self.assertEqual(bounds[-1], (128, 143))

    def test_windows_are_contiguous_at_the_configured_stride(self):
        bounds = window_bounds(SOURCE_FRAMES, 16, 8)
        for (first, last) in bounds:
            self.assertEqual(last - first + 1, 16)
        for previous, current in zip(bounds, bounds[1:]):
            self.assertEqual(current[0] - previous[0], 8)

    def test_a_clip_shorter_than_the_window_yields_nothing(self):
        self.assertEqual(window_bounds(10, 16, 8), ())

    def test_no_partial_window_is_emitted_at_the_tail(self):
        """150 frames do not divide evenly; the final 6 must be dropped."""
        bounds = window_bounds(SOURCE_FRAMES, 16, 8)
        self.assertLess(bounds[-1][1], SOURCE_FRAMES - 1)
        self.assertEqual(bounds[-1][1], 143)

    def test_committed_primary_csv_matches_the_buffer(self):
        report = verify_geometry(PRIMARY_CSV, SOURCE_FRAMES, 16, 8)
        self.assertTrue(report["match"], report["first_mismatches"])
        self.assertEqual(report["clips_checked"], 394)
        self.assertEqual(report["clips_mismatched"], 0)

    def test_committed_carve_csv_matches_the_buffer(self):
        report = verify_geometry(CARVE_CSV, SOURCE_FRAMES, 16, 8)
        self.assertTrue(report["match"], report["first_mismatches"])
        self.assertEqual(report["clips_checked"], 240)

    def test_every_clip_in_both_csvs_uses_one_identical_window_layout(self):
        for csv_path in (PRIMARY_CSV, CARVE_CSV):
            distinct = {bounds for bounds in read_recorded_bounds(csv_path).values()}
            self.assertEqual(len(distinct), 1, f"{csv_path.name} has mixed window layouts")


class AlarmFrameTests(unittest.TestCase):
    """The alarm definition, on synthetic windows with a known answer."""

    @staticmethod
    def _windows(probabilities):
        return [
            WindowScore(window_index=i, first_frame=i * 8, last_frame=i * 8 + 15,
                        fight_probability=p)
            for i, p in enumerate(probabilities)
        ]

    def setUp(self):
        self.rule = FrozenAggregationRule.load(FROZEN_JSON)

    def test_alarm_is_the_last_frame_of_the_first_qualifying_window(self):
        frame = alarm_frame(self._windows([0.0, 0.9, 0.0]), self.rule)
        self.assertEqual(frame, 23)  # window 1 covers [8, 23]

    def test_no_alarm_when_no_window_qualifies(self):
        self.assertIsNone(alarm_frame(self._windows([0.0, 0.01, 0.02]), self.rule))

    def test_alarm_on_the_very_first_window(self):
        self.assertEqual(alarm_frame(self._windows([0.99, 0.0]), self.rule), 15)

    def test_a_later_high_window_does_not_move_an_earlier_alarm(self):
        """Causality: the alarm is when it FIRST fires, not when it peaks."""
        frame = alarm_frame(self._windows([0.5, 0.0, 0.99]), self.rule)
        self.assertEqual(frame, 15)

    def test_empty_window_list_never_alarms(self):
        self.assertIsNone(alarm_frame([], self.rule))


class TimeToAlarmTests(unittest.TestCase):
    """The measurement must agree with results the project already recorded."""

    @classmethod
    def setUpClass(cls):
        cls.report = compute("primary")
        cls.recorded = json.loads(
            (REPO_ROOT / "temporal_risk" / "primary_aggregation_result.json").read_text()
        )["primary_split"]["metrics"]

    def test_detected_fight_clips_equal_the_recorded_true_positives(self):
        self.assertEqual(self.report["fight_clips"]["detected"], self.recorded["tp"])

    def test_censored_fight_clips_equal_the_recorded_false_negatives(self):
        self.assertEqual(
            self.report["fight_clips"]["censored_never_alarmed"], self.recorded["fn"]
        )

    def test_false_alarms_equal_the_recorded_false_positives(self):
        self.assertEqual(self.report["nonfight_clips"]["false_alarms"], self.recorded["fp"])

    def test_recall_matches_the_recorded_recall(self):
        self.assertAlmostEqual(
            self.report["fight_clips"]["recall"], self.recorded["recall"], places=6
        )

    def test_split_totals_are_the_394_primary_clips(self):
        self.assertEqual(self.report["fight_clips"]["total"], 200)
        self.assertEqual(self.report["nonfight_clips"]["total"], 194)

    def test_alarm_times_are_bounded_by_the_first_and_last_window(self):
        """No alarm can precede window 0 completing, or follow window 16.

        Reported values are rounded to 4 dp, so the lower bound is compared
        with a tolerance of half a unit in the last place rather than exactly.
        """
        summary = self.report["fight_clips"]["time_to_alarm_over_detected_only"]
        earliest_possible = 16 / SOURCE_FPS  # window 0 ends at frame 15
        self.assertGreaterEqual(summary["min_seconds"], earliest_possible - 5e-5)
        self.assertLessEqual(summary["max_seconds"], self.report["max_possible_alarm_seconds"])

    def test_censored_clips_carry_no_alarm_time(self):
        for row in self.report["per_clip"]:
            if not row["alarmed"]:
                self.assertIsNone(row["time_to_alarm_seconds"])
                self.assertIsNone(row["alarm_frame"])

    def test_detected_clips_all_carry_an_alarm_time(self):
        for row in self.report["per_clip"]:
            if row["alarmed"]:
                self.assertIsNotNone(row["time_to_alarm_seconds"])

    def test_every_primary_clip_appears_exactly_once(self):
        clips = [row["clip"] for row in self.report["per_clip"]]
        self.assertEqual(len(clips), 394)
        self.assertEqual(len(set(clips)), 394)

    def test_carve_split_reproduces_the_frozen_selection_counts(self):
        report = compute("carve")
        selected = json.loads(FROZEN_JSON.read_text())["selected"]["metrics"]
        self.assertEqual(report["fight_clips"]["detected"], selected["tp"])
        self.assertEqual(report["fight_clips"]["censored_never_alarmed"], selected["fn"])
        self.assertEqual(report["nonfight_clips"]["false_alarms"], selected["fp"])


class DemoAgreementTests(unittest.TestCase):
    """Time-to-alarm must agree with what the live demo actually reported."""

    def test_matches_the_demo_fire_frames_for_the_built_in_clips(self):
        # Values produced by tools/run_demo.py and recorded in the Step 0
        # golden reference: temporal_signal_first_frame per built-in clip.
        expected = {
            "val/Val_Fight/trtrhrt_1049.avi": 15,
            "val/Val_NonFight/ZCUy99AN_0.avi": None,
            "val/Val_NonFight/39BFeYnbu-I_0.avi": 23,
        }
        rows = {row["clip"]: row for row in compute("primary")["per_clip"]}
        for clip, frame in expected.items():
            self.assertEqual(rows[clip]["alarm_frame"], frame, f"disagreement on {clip}")


if __name__ == "__main__":
    unittest.main()
