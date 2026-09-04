"""Tests for the demo's window-score source abstraction (integration Phase 3).

The demo used to compute the clip-level temporal decision BEFORE opening the
video, from the whole CSV, and the frame loop only replayed when it would have
fired. These tests pin the inverted structure: scores arrive during the loop,
and the decision is taken afterwards from what was actually observed.

They also pin the two properties that make the refactor safe:

- the CSV source stays the default, so the reviewed demo is unchanged;
- an unavailable source raises rather than silently falling back, so an output
  can never be ambiguous about which mode produced it.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from run_demo import (  # noqa: E402
    SOURCE_CSV,
    SOURCE_LIVE,
    SUPPORTED_TEMPORAL_SOURCES,
    LiveInferenceSource,
    PrecomputedReplaySource,
    WindowScoreSource,
    build_window_source,
    current_window_probability,
    run_demo,
)
from temporal_event_adapter import (  # noqa: E402
    FrozenAggregationRule,
    PrecomputedWindowScoreSource,
    TemporalEventAdapter,
    WindowScore,
)

CLIP_KEY = "val/Val_NonFight/39BFeYnbu-I_0.avi"
PRIMARY_CSV = REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"
FROZEN_JSON = REPO_ROOT / "temporal_risk" / "frozen_aggregation.json"
FRAME = np.zeros((8, 8, 3), dtype=np.uint8)


def committed_windows(clip_key=CLIP_KEY):
    return PrecomputedWindowScoreSource(PRIMARY_CSV).get(clip_key)


class SourceSelectionTests(unittest.TestCase):
    def test_csv_is_the_default_for_run_demo(self):
        import inspect

        signature = inspect.signature(run_demo)
        parameter = signature.parameters["temporal_source"]
        self.assertEqual(parameter.default, SOURCE_CSV)
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_csv_selection_returns_the_replay_source(self):
        source = build_window_source(SOURCE_CSV, CLIP_KEY, committed_windows())
        self.assertIsInstance(source, PrecomputedReplaySource)
        self.assertEqual(source.name, SOURCE_CSV)

    def test_both_sources_are_named_and_distinct(self):
        self.assertEqual(set(SUPPORTED_TEMPORAL_SOURCES), {SOURCE_CSV, SOURCE_LIVE})

    def test_an_unknown_source_is_refused(self):
        with self.assertRaises(SystemExit):
            build_window_source("guess", CLIP_KEY, committed_windows())

    def test_live_without_a_checkpoint_raises_rather_than_falling_back_to_csv(self):
        """A silent fallback would make an output ambiguous about its origin.

        Updated in Phase 4: live inference is now implemented, so this no
        longer raises NotImplementedError. The enduring property it protects is
        unchanged -- selecting live without a usable checkpoint must fail, even
        when valid CSV windows are sitting right there.
        """
        with self.assertRaises(SystemExit) as caught:
            build_window_source(SOURCE_LIVE, CLIP_KEY, committed_windows(), checkpoint=None)
        message = str(caught.exception)
        self.assertIn("--checkpoint", message)
        self.assertIn("never silently become a CSV replay", message)

    def test_live_source_cannot_be_constructed_without_a_valid_checkpoint(self):
        """Constructing the source directly is guarded too, not just the factory."""
        from checkpoint_identity import CheckpointIntegrityError

        with self.assertRaises(CheckpointIntegrityError):
            LiveInferenceSource(REPO_ROOT / "does" / "not" / "exist.pt")

    def test_the_abstract_base_defines_the_contract(self):
        base = WindowScoreSource()
        with self.assertRaises(NotImplementedError):
            base.windows_completed_at(0, FRAME)
        with self.assertRaises(NotImplementedError):
            base.hud_probability(0, [])


class CausalAccumulationTests(unittest.TestCase):
    """Scores must arrive during the loop, one window at a time."""

    def setUp(self):
        self.windows = committed_windows()
        self.source = PrecomputedReplaySource(CLIP_KEY, self.windows)

    def test_no_window_is_available_before_the_first_completes(self):
        for frame_idx in range(15):
            self.assertEqual(self.source.windows_completed_at(frame_idx, FRAME), [])

    def test_each_window_arrives_exactly_at_its_last_frame(self):
        for window in self.windows:
            completed = self.source.windows_completed_at(window.last_frame, FRAME)
            self.assertIn(window, completed)

    def test_a_window_never_arrives_early(self):
        for window in self.windows:
            for frame_idx in range(window.last_frame):
                self.assertNotIn(window, self.source.windows_completed_at(frame_idx, FRAME))

    def test_replaying_a_whole_clip_yields_every_window_once_in_order(self):
        observed = []
        for frame_idx in range(150):
            observed.extend(self.source.windows_completed_at(frame_idx, FRAME))
        self.assertEqual(len(observed), 17)
        self.assertEqual([w.window_index for w in observed], list(range(17)))
        self.assertEqual(observed, list(self.windows))

    def test_accumulated_windows_reproduce_the_committed_geometry(self):
        observed = []
        for frame_idx in range(150):
            observed.extend(self.source.windows_completed_at(frame_idx, FRAME))
        self.assertEqual(
            [(w.first_frame, w.last_frame) for w in observed],
            [(i * 8, i * 8 + 15) for i in range(17)],
        )

    def test_the_decision_emerges_during_the_replay_not_before(self):
        """The alarm frame must be derivable from accumulation alone."""
        rule = FrozenAggregationRule.load(FROZEN_JSON)
        running, fired_at = [], None
        for frame_idx in range(150):
            for window in self.source.windows_completed_at(frame_idx, FRAME):
                running.append(window.fight_probability)
            if fired_at is None and running and rule.decide(running):
                fired_at = frame_idx
        self.assertEqual(fired_at, 23)  # recorded first-fire frame for this clip


class HudBehaviourTests(unittest.TestCase):
    """The replay HUD keeps its look-ahead; that is what preserves byte-identity."""

    def setUp(self):
        self.windows = committed_windows()
        self.source = PrecomputedReplaySource(CLIP_KEY, self.windows)

    def test_hud_matches_the_pre_inversion_function_exactly(self):
        for frame_idx in range(150):
            self.assertEqual(
                self.source.hud_probability(frame_idx, []),
                current_window_probability(self.windows, frame_idx),
                f"HUD changed at frame {frame_idx}",
            )

    def test_hud_does_not_depend_on_what_has_been_observed(self):
        """Replay can look ahead; a live source will not be able to."""
        self.assertEqual(
            self.source.hud_probability(4, []),
            self.source.hud_probability(4, list(self.windows)),
        )


class AfterLoopEvaluationTests(unittest.TestCase):
    """evaluate_clip must consume the accumulated windows, after the loop."""

    def test_run_demo_evaluates_after_the_frame_loop(self):
        source = (REPO_ROOT / "tools" / "run_demo.py").read_text(encoding="utf-8")
        release = source.index("cap.release()")
        evaluate = source.index("adapter.evaluate_clip(")
        self.assertGreater(
            evaluate, release,
            "evaluate_clip runs before the loop ends; the decision is not causal",
        )

    def test_run_demo_evaluates_the_observed_windows(self):
        source = (REPO_ROOT / "tools" / "run_demo.py").read_text(encoding="utf-8")
        self.assertIn("adapter.evaluate_clip(clip_key, observed_windows)", source)

    def test_run_demo_no_longer_reads_the_full_window_list_in_the_loop(self):
        source = (REPO_ROOT / "tools" / "run_demo.py").read_text(encoding="utf-8")
        self.assertNotIn("[w for w in windows if w.last_frame == frame_idx]", source)

    def test_evaluating_accumulated_windows_matches_evaluating_all_of_them(self):
        """For a 150-frame clip every window completes, so the two agree."""
        windows = committed_windows()
        source = PrecomputedReplaySource(CLIP_KEY, windows)
        observed = []
        for frame_idx in range(150):
            observed.extend(source.windows_completed_at(frame_idx, FRAME))
        adapter = TemporalEventAdapter(FrozenAggregationRule.load(FROZEN_JSON))
        from_all = adapter.evaluate_clip(CLIP_KEY, list(windows))
        from_observed = adapter.evaluate_clip(CLIP_KEY, observed)
        self.assertIsNotNone(from_observed)
        self.assertEqual(from_observed.evidence, from_all.evidence)
        self.assertEqual(from_observed.confidence, from_all.confidence)


class ProvenanceTests(unittest.TestCase):
    def test_replay_source_describes_itself_honestly(self):
        described = PrecomputedReplaySource(CLIP_KEY, committed_windows()).describe()
        self.assertEqual(described["temporal_source"], SOURCE_CSV)
        self.assertIn("no model ran in this process", described["note"])
        self.assertEqual(described["clip_key"], CLIP_KEY)


class DemoOutputStabilityTests(unittest.TestCase):
    """The committed Step-1 reference artifacts must remain reproducible."""

    REFERENCE = REPO_ROOT / "outputs" / "step1_after_stale_track_fix"

    @unittest.skipUnless(
        (REPO_ROOT / "outputs" / "step1_after_stale_track_fix").is_dir(),
        "Step-1 reference outputs not present",
    )
    def test_reference_artifacts_are_still_present_for_comparison(self):
        for stem in ("trtrhrt_1049", "ZCUy99AN_0", "39BFeYnbu-I_0"):
            for suffix in (".mp4", "_risk_assessment.json", "_explanation.md"):
                path = self.REFERENCE / f"demo_{stem}{suffix}"
                self.assertTrue(path.is_file(), f"missing reference artifact {path.name}")

    def test_run_demo_signature_is_backward_compatible(self):
        """web/server.py calls run_demo positionally with four arguments."""
        import inspect

        positional = [
            name for name, parameter in inspect.signature(run_demo).parameters.items()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        self.assertEqual(positional, ["video_path", "clip_key", "model_path", "output_dir"])


if __name__ == "__main__":
    unittest.main()
