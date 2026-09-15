"""Tests for the operational evidence-fusion layer.

The property this file exists to protect: the live spatial feature must be the
SAME computation that produced the locked research artifact. The research
implementation lives inside tools/run_confirmatory_primary_evaluation.py, which
is the provenance of a protected artifact and must not be edited, so the
operational version is necessarily a second implementation. A second
implementation that is never compared to the first is a drift waiting to
happen, and the equivalence test below is what stops that.

Everything else here guards honesty properties: undefined stays undefined,
thresholds come from the locked protocol rather than from this codebase, and no
result can be read as a claim that fusion improved anything.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from operational_fusion import (  # noqa: E402
    EXPECTED_PROTOCOL_SHA,
    FUSION_INTERPRETATION,
    FUSION_MODE,
    PROTOCOL_PATH,
    FrozenFusionProtocol,
    FusionProtocolError,
    SpatialFeatureAccumulator,
    describe_fusion_layer,
    load_protocol,
    percentile_of,
)
from tracker import build_tracking_snapshot  # noqa: E402

YOLO_WEIGHTS = REPO_ROOT / "models" / "yolov8s.pt"
FIXTURE_VIDEO = REPO_ROOT / "data" / "person_fixture.mp4"


class _Track:
    """Minimal stand-in with the attributes the accumulator reads."""

    def __init__(self, track_id, bbox, class_id=0):
        self.id = track_id
        self.bbox = np.asarray(bbox, dtype=float)
        self.class_id = class_id


class ProtocolLoadingTests(unittest.TestCase):
    def test_protocol_matches_the_locked_hash(self):
        protocol = load_protocol()
        self.assertEqual(protocol["protocol_sha256"], EXPECTED_PROTOCOL_SHA)

    def test_expected_sha_agrees_with_the_lock_record(self):
        """This module must not carry its own idea of which protocol is locked."""
        lock = json.loads(
            (REPO_ROOT / "outputs" / "fusion" / "FINAL_experiment_lock_record.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(lock["hashes"]["protocol_sha256_internal"], EXPECTED_PROTOCOL_SHA)

    def test_an_edited_protocol_is_refused(self):
        import tempfile

        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["candidates"]["B0_temporal_only"]["parameters_frozen"]["threshold"] = 0.5
        with tempfile.TemporaryDirectory() as tmp:
            edited = Path(tmp) / "edited.json"
            edited.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
            with self.assertRaises(FusionProtocolError):
                load_protocol(edited)

    def test_a_missing_protocol_is_refused_with_no_defaults(self):
        with self.assertRaises(FusionProtocolError):
            load_protocol(REPO_ROOT / "temporal_risk" / "does_not_exist.json")

    def test_parameters_come_from_the_protocol_not_from_source(self):
        """Every threshold must be readable out of the locked file."""
        protocol = load_protocol()
        frozen = FrozenFusionProtocol.load()
        candidates = protocol["candidates"]
        self.assertEqual(frozen.temporal_threshold,
                         candidates["B0_temporal_only"]["parameters_frozen"]["threshold"])
        self.assertEqual(frozen.theta_s,
                         candidates["B1_spatial_only"]["parameters_calibrated"]["theta_s"])
        self.assertEqual(frozen.theta_or,
                         candidates["F1_or_gate"]["parameters_calibrated"]["theta_or"])
        self.assertEqual(frozen.theta_f,
                         candidates["F2_rank_sum"]["parameters_calibrated"]["theta_f"])
        self.assertEqual(frozen.weight,
                         candidates["F2_rank_sum"]["parameters_frozen"]["weight"])

    def test_source_hard_codes_no_threshold_value(self):
        """A literal threshold in this module would be a second source of truth."""
        source = (REPO_ROOT / "src" / "operational_fusion.py").read_text(encoding="utf-8")
        for value in ("0.023439943309201124", "0.44120893993145877", "0.49133532299990373"):
            self.assertNotIn(value, source,
                             f"{value} is hard-coded; it must be read from the protocol")


class PercentileTests(unittest.TestCase):
    def test_matches_the_confirmatory_tools_percentile(self):
        from run_confirmatory_primary_evaluation import percentile_of as reference

        values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 3.7]
        distribution = sorted([0.1, 0.2, 0.25, 0.4, 0.8, 0.9])
        for value in values:
            self.assertEqual(percentile_of(value, distribution),
                             reference(value, distribution), f"diverged at {value}")

    def test_undefined_value_has_no_percentile(self):
        self.assertIsNone(percentile_of(None, [0.1, 0.2]))

    def test_empty_reference_has_no_percentile(self):
        self.assertIsNone(percentile_of(0.5, []))


class SpatialFeatureTests(unittest.TestCase):
    def test_no_persons_leaves_the_feature_undefined_not_zero(self):
        accumulator = SpatialFeatureAccumulator()
        for frame in range(10):
            accumulator.observe(frame, [])
        self.assertIsNone(accumulator.score)
        self.assertIsNone(accumulator.speed_mean)
        payload = accumulator.as_dict()
        self.assertFalse(payload["defined"])
        self.assertIsNone(payload["spatial_score"])
        self.assertEqual(payload["zero_person_frames"], 10)

    def test_a_single_frame_yields_no_displacement_sample(self):
        accumulator = SpatialFeatureAccumulator()
        accumulator.observe(0, [_Track(1, [0, 0, 20, 40])])
        self.assertIsNone(accumulator.score)


class SpatialHistoryOffsetTests(unittest.TestCase):
    """Pins the EXACT history offset the frozen implementation uses.

    These tests verify behaviour; they must never be used to change it. The
    frozen thresholds, the development percentile reference and every locked
    primary result were produced by this offset, so if any assertion here
    fails, the live feature has silently moved onto a different scale from the
    thresholds it is compared against.

    See docs/SPATIAL_FEATURE_SEMANTICS.md.
    """

    @staticmethod
    def _box(x):
        return [x, 0.0, x + 20.0, 40.0]

    def test_sample_spans_two_appearances_not_one_transition(self):
        """10 px of motion per frame must yield 20 px samples, not 10."""
        accumulator = SpatialFeatureAccumulator()
        for frame in range(6):
            accumulator.observe(frame, [_Track(1, self._box(frame * 10.0))])
        self.assertEqual(accumulator._speeds, [20.0, 20.0, 20.0, 20.0])
        self.assertAlmostEqual(accumulator.speed_mean, 20.0)

    def test_the_accessor_really_returns_the_second_most_recent_box(self):
        """The root cause, asserted directly on BehaviorAnalyzer."""
        from behavior_analyzer import BehaviorAnalyzer

        analyzer = BehaviorAnalyzer()
        for frame in range(4):
            analyzer.update_history(
                build_tracking_snapshot(frame, [_Track(1, self._box(frame * 10.0))]), 3.0, 5)
        # history is [0, 10, 20, 30]; the accessor must hand back 20, not 30.
        self.assertAlmostEqual(float(analyzer.get_previous_bbox(1)[0]), 20.0)

    def test_warm_up_costs_the_first_two_appearances(self):
        """A track seen k times contributes max(k - 2, 0) samples."""
        for appearances, expected in [(1, 0), (2, 0), (3, 1), (4, 2), (6, 4)]:
            with self.subTest(appearances=appearances):
                accumulator = SpatialFeatureAccumulator()
                for frame in range(appearances):
                    accumulator.observe(frame, [_Track(1, self._box(frame * 10.0))])
                self.assertEqual(len(accumulator._speeds), expected)

    def test_across_a_dropout_the_gap_is_appearances_not_wall_clock_frames(self):
        """The span is NOT a fixed two frames when the tracker drops a track."""
        accumulator = SpatialFeatureAccumulator()
        seen = {0: 0.0, 1: 10.0, 5: 50.0, 7: 70.0}
        for frame in range(8):
            tracks = [_Track(1, self._box(seen[frame]))] if frame in seen else []
            accumulator.observe(frame, tracks)
        # 3rd appearance (frame 5) measures against the 1st (frame 0): 50 px
        #    across five wall-clock frames.
        # 4th appearance (frame 7) measures against the 2nd (frame 1): 60 px
        #    across six wall-clock frames.
        self.assertEqual(accumulator._speeds, [50.0, 60.0])

    def test_a_non_person_track_does_not_perturb_a_person_history(self):
        """History is per track id, so other classes cannot shift the offset."""
        accumulator = SpatialFeatureAccumulator()
        for frame in range(6):
            accumulator.observe(frame, [
                _Track(1, self._box(frame * 10.0)),
                _Track(2, self._box(500.0), class_id=2),
            ])
        self.assertEqual(accumulator._speeds, [20.0, 20.0, 20.0, 20.0])

    def test_the_ratio_uses_exactly_these_samples(self):
        """Numerator is the mean of this sample set; denominator is separate."""
        accumulator = SpatialFeatureAccumulator()
        for frame in range(6):
            accumulator.observe(frame, [_Track(1, self._box(frame * 10.0))])
        self.assertAlmostEqual(accumulator.speed_mean,
                               sum(accumulator._speeds) / len(accumulator._speeds))
        # Denominator counts person-bearing frames, including the warm-up ones
        # that contributed no displacement sample.
        self.assertEqual(len(accumulator._diagonals), 6)
        self.assertEqual(accumulator.score,
                         accumulator.speed_mean / accumulator.diagonal_mean)

    def test_the_payload_describes_the_numerator_accurately(self):
        """A consumer must not be told this is a consecutive-frame speed."""
        payload = SpatialFeatureAccumulator().as_dict()
        self.assertIn("two-appearance history offset", payload["numerator"])
        self.assertIn("NOT a consecutive-frame speed", payload["numerator"])
        self.assertEqual(payload["semantics_reference"],
                         "docs/SPATIAL_FEATURE_SEMANTICS.md")

    def test_the_module_does_not_call_it_a_consecutive_frame_speed(self):
        """Guards against the loose wording creeping back into the source.

        Checked per line, requiring a negation on the same line. The docstrings
        deliberately QUOTE the protocol's "between consecutive frames of the
        same track", and list the phrases not to use, in order to say they do
        not describe the code -- so a flat substring ban would flag the very
        text that exists to prevent the confusion. What must not appear is one
        of these phrases ASSERTED, i.e. on a line carrying no negation.
        """
        import re

        source_path = REPO_ROOT / "src" / "operational_fusion.py"
        source = source_path.read_text(encoding="utf-8")
        loose = ("one-frame speed", "frame-to-frame velocity",
                 "consecutive-frame speed", "consecutive-frame displacement",
                 "between consecutive frames")
        negations = ("not", "never", "none of those")
        # Sentence granularity: the prose wraps across lines, so a negation for
        # a banned phrase routinely sits on the line above it.
        flat = re.sub(r"\s+", " ", source.lower())
        for sentence in flat.split("."):
            for phrase in loose:
                if phrase in sentence and not any(n in sentence for n in negations):
                    self.fail(f"operational_fusion.py asserts {phrase!r} without "
                              f"negation, in: {sentence.strip()!r}")
        self.assertIn("two-appearance", flat)


class SpatialFeatureGeometryTests(unittest.TestCase):
    """The denominator and the person filter, independent of the history offset."""

    def test_zero_person_frames_do_not_enter_the_denominator(self):
        accumulator = SpatialFeatureAccumulator()
        accumulator.observe(0, [_Track(1, [0, 0, 20, 40])])
        accumulator.observe(1, [])
        accumulator.observe(2, [_Track(1, [0, 0, 20, 40])])
        self.assertEqual(len(accumulator._diagonals), 2)

    def test_single_person_diagonal_is_that_persons_own_box(self):
        accumulator = SpatialFeatureAccumulator()
        accumulator.observe(0, [_Track(1, [0, 0, 30, 40])])
        self.assertAlmostEqual(accumulator.diagonal_mean, 50.0)  # 3-4-5

    def test_non_person_tracks_are_excluded_from_the_feature(self):
        accumulator = SpatialFeatureAccumulator()
        accumulator.observe(0, [_Track(1, [0, 0, 30, 40], class_id=2)])
        self.assertIsNone(accumulator.diagonal_mean)
        self.assertEqual(accumulator.zero_person_frames, 1)

    def test_ratio_of_means_not_mean_of_ratios(self):
        """The protocol specifies a ratio of means; they differ numerically."""
        accumulator = SpatialFeatureAccumulator()
        boxes = [[0, 0, 10, 10], [40, 0, 50, 10], [80, 0, 90, 10], [81, 0, 200, 10]]
        for frame, box in enumerate(boxes):
            accumulator.observe(frame, [_Track(1, box)])
        expected = accumulator.speed_mean / accumulator.diagonal_mean
        self.assertAlmostEqual(accumulator.score, expected)


class SpatialFeatureEquivalenceTests(unittest.TestCase):
    """The anti-drift guarantee: same video, same number, as the research tool."""

    @unittest.skipUnless(YOLO_WEIGHTS.is_file(), "YOLO weights not present")
    @unittest.skipUnless(FIXTURE_VIDEO.is_file(), "fixture video not present")
    def test_accumulator_reproduces_the_research_implementation(self):
        import cv2
        from detection import load_yolo_model
        from run_confirmatory_primary_evaluation import (
            RUNTIME_CONFIDENCE, spatial_score_for,
        )
        from tracker import SimpleTracker, active_tracks

        model = load_yolo_model(YOLO_WEIGHTS)

        # The research tool, run on this video exactly as it ran on primary clips.
        reference = spatial_score_for(FIXTURE_VIDEO.name, model, FIXTURE_VIDEO.parent)

        # The operational accumulator, driven the way run_demo drives it.
        accumulator = SpatialFeatureAccumulator()
        tracker = SimpleTracker()
        capture = cv2.VideoCapture(str(FIXTURE_VIDEO))
        frame_idx = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detections = model(frame, conf=RUNTIME_CONFIDENCE, verbose=False)[0]
                if detections.boxes is not None and len(detections.boxes):
                    boxes = detections.boxes.xyxy.cpu().numpy()
                    classes = detections.boxes.cls.cpu().numpy().astype(int)
                    confidences = detections.boxes.conf.cpu().numpy()
                else:
                    boxes = np.empty((0, 4))
                    classes = np.array([], dtype=int)
                    confidences = np.array([])
                tracks = active_tracks(tracker.update(boxes, classes, confidences),
                                       tracker.frame_idx)
                accumulator.observe(frame_idx, tracks)
                frame_idx += 1
        finally:
            capture.release()

        self.assertEqual(accumulator.frames, reference["frames"])
        self.assertEqual(accumulator.zero_person_frames, reference["zero_person_frames"])
        self.assertAlmostEqual(accumulator.speed_mean, reference["speed_mean_pixels"], places=9)
        self.assertAlmostEqual(accumulator.diagonal_mean, reference["group_diagonal_mean"],
                               places=9)
        self.assertAlmostEqual(accumulator.score, reference["spatial_score"], places=12,
                               msg="the operational feature has drifted from the frozen one")


class FusionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = FrozenFusionProtocol.load()

    def test_no_temporal_evidence_is_reported_not_defaulted(self):
        result = self.protocol.evaluate(None, 0.05)
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "no_temporal_evidence")

    def test_undefined_spatial_abstains_and_cannot_fire_a_spatial_rule(self):
        result = self.protocol.evaluate(0.9, None)
        self.assertTrue(result["available"])
        self.assertIsNone(result["inputs"]["spatial_score"])
        self.assertFalse(result["inputs"]["spatial_defined"])
        self.assertTrue(result["inputs"]["spatial_abstained"])
        self.assertFalse(result["candidates"]["B1_spatial_only"]["fired"])

    def test_undefined_spatial_is_never_reported_as_zero(self):
        result = self.protocol.evaluate(0.9, None)
        self.assertIsNone(result["inputs"]["spatial_score"])
        self.assertIsNone(result["inputs"]["spatial_percentile"])

    def test_f2_treats_an_abstaining_spatial_as_zero_percentile(self):
        """The frozen arithmetic, reproduced exactly."""
        result = self.protocol.evaluate(1.0, None)
        expected = self.protocol.weight * percentile_of(1.0, self.protocol.temporal_reference)
        self.assertAlmostEqual(result["candidates"]["F2_rank_sum"]["score"], expected)

    def test_b0_matches_the_frozen_threshold(self):
        self.assertTrue(self.protocol.evaluate(0.14, None)["candidates"]["B0_temporal_only"]["fired"])
        self.assertFalse(self.protocol.evaluate(0.139, None)["candidates"]["B0_temporal_only"]["fired"])

    def test_system_decision_is_b0_not_fusion(self):
        result = self.protocol.evaluate(0.9, 0.9)
        self.assertEqual(result["system_decision"]["from"], "B0_temporal_only")
        self.assertEqual(result["system_decision"]["decision"],
                         result["candidates"]["B0_temporal_only"]["decision"])

    def test_f1_is_an_or_gate_over_the_frozen_arms(self):
        low_temporal = 0.01
        result = self.protocol.evaluate(low_temporal, self.protocol.theta_or)
        self.assertFalse(result["candidates"]["B0_temporal_only"]["fired"])
        self.assertTrue(result["candidates"]["F1_or_gate"]["fired"])

    def test_every_result_carries_the_no_improvement_statement(self):
        for temporal, spatial in [(0.9, 0.9), (0.01, None), (None, 0.5)]:
            result = self.protocol.evaluate(temporal, spatial)
            self.assertEqual(result["interpretation"], FUSION_INTERPRETATION)
            self.assertIn("not_claimed", result)

    def test_the_statement_does_not_claim_improvement(self):
        """Checks the ASSERTION, not the disclaimer.

        The disclaimer necessarily quotes the claims it denies ("Not claimed:
        that fusion is superior..."), so scanning it for those phrases would
        flag the very text that exists to prevent them.
        """
        claim = FUSION_INTERPRETATION.lower()
        self.assertIn("did not produce a statistically significant improvement", claim)
        for forbidden in ("fusion is superior", "fusion improves", "outperforms",
                          "better than", "significant improvement over the temporal-only "
                                          "baseline was found"):
            self.assertNotIn(forbidden, claim)

    def test_the_disclaimer_explicitly_denies_the_forbidden_claims(self):
        denial = describe_fusion_layer()["not_claimed"].lower()
        self.assertTrue(denial.startswith("not claimed:"))
        for denied in ("superior", "improves accuracy", "statistically better"):
            self.assertIn(denied, denial)

    def test_fusion_is_labelled_offline_whole_video(self):
        result = self.protocol.evaluate(0.9, 0.5)
        self.assertEqual(result["mode"], FUSION_MODE)
        self.assertIn("whole-video", FUSION_MODE)
        self.assertIn("not a causal mid-video alarm", result["causality"])

    def test_result_records_which_protocol_produced_it(self):
        result = self.protocol.evaluate(0.9, 0.5)
        self.assertEqual(result["protocol_sha256"], EXPECTED_PROTOCOL_SHA)


if __name__ == "__main__":
    unittest.main()
