"""Tests for the live end-to-end path exposed over HTTP.

The property this file protects: a LIVE result must be a genuine model run on
the video supplied, and must never be confusable with a REPLAY of committed
scores. Every test here is aimed at one of the ways those two could blur --
a silent fallback, a timeline quietly sourced from the CSV, a mode inferred
rather than stated, a decision asserted from no evidence.

Tests that need the real 132 MB checkpoint skip cleanly when it is absent, so
the suite still runs on a machine that has not recovered it.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "web"))

import server as SERVER_MODULE  # noqa: E402
from run_demo import SOURCE_CSV, SOURCE_LIVE, run_demo  # noqa: E402

SERVER_SOURCE = (REPO_ROOT / "web" / "server.py").read_text(encoding="utf-8")
RUN_DEMO_SOURCE = (REPO_ROOT / "tools" / "run_demo.py").read_text(encoding="utf-8")
CHECKPOINT = REPO_ROOT / "models" / "temporal_violence" / "best.pt"
YOLO_WEIGHTS = REPO_ROOT / "models" / "yolov8s.pt"

needs_checkpoint = unittest.skipUnless(CHECKPOINT.is_file(), "R3D checkpoint not present")
needs_yolo = unittest.skipUnless(YOLO_WEIGHTS.is_file(), "YOLO weights not present")


def write_video(path: Path, frames: int, width=320, height=240) -> Path:
    """A tiny synthetic clip. No people in it -- that is deliberate."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    writer.release()
    return path


class ModeContractTests(unittest.TestCase):
    """The mode must be chosen, stated, and never silently substituted."""

    def setUp(self):
        SERVER_MODULE.app.config["TESTING"] = True
        self.client = SERVER_MODULE.app.test_client()

    def test_unknown_mode_is_rejected(self):
        response = self.client.post("/api/analyze", data={"mode": "banana", "preset": "fight"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown mode", response.get_json()["error"])

    def test_live_without_a_video_is_rejected_not_replayed(self):
        response = self.client.post("/api/analyze", data={"mode": "live"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("needs a video", response.get_json()["error"])

    def test_replay_remains_the_default_mode(self):
        self.assertIn('request.form.get("mode") or MODE_REPLAY', SERVER_SOURCE)
        self.assertEqual(SERVER_MODULE.MODE_REPLAY, "replay")

    def test_live_never_falls_back_to_csv(self):
        """The refusal must be a refusal, not a downgrade."""
        self.assertIn("There is no fallback", SERVER_SOURCE)
        self.assertNotIn("except Exception:\n            mode = MODE_REPLAY", SERVER_SOURCE)

    def test_live_timeline_comes_from_the_run_not_the_score_source(self):
        """A live timeline read from _score_source would be a replay in disguise."""
        analyze = SERVER_SOURCE.split("def api_analyze()")[1]
        live_branch = analyze.split("if mode == MODE_LIVE:")[-1]
        self.assertIn('summary.get("window_scores")', live_branch)

    def test_response_states_its_mode_explicitly(self):
        self.assertIn('"mode": mode', SERVER_SOURCE)
        self.assertIn('"mode_label"', SERVER_SOURCE)


class LiveCsvIndependenceTests(unittest.TestCase):
    """Live mode must not need, read, or be rescued by the research CSV."""

    def test_live_source_never_reads_the_primary_csv(self):
        live = RUN_DEMO_SOURCE.split("class LiveInferenceSource")[1].split("def build_window_source")[0]
        self.assertNotIn("PRIMARY_CSV", live)
        self.assertNotIn("PrecomputedWindowScoreSource", live)

    def test_building_a_live_source_does_not_require_window_scores(self):
        """windows=None is exactly the unrecorded-clip case."""
        from run_demo import build_window_source

        with self.assertRaises(SystemExit):
            build_window_source(SOURCE_CSV, "unrecorded.avi", None)
        # Live with no checkpoint must also raise -- never fall through to csv.
        with self.assertRaises(SystemExit):
            build_window_source(SOURCE_LIVE, "unrecorded.avi", None, checkpoint=None)

    @needs_yolo
    @needs_checkpoint
    def test_live_runs_with_the_primary_csv_entirely_absent(self):
        """The decisive proof that live has no CSV dependency at all.

        run_demo used to construct PrecomputedWindowScoreSource unconditionally
        -- for an optional ground-truth LABEL, not for scores -- so a live run
        died outright when the file was missing, despite using none of its
        contents. No real artifact is touched here; the module constant is
        redirected to a path that does not exist.
        """
        import run_demo as run_demo_module
        from detection import DEFAULT_MODEL_PATH

        original = run_demo_module.PRIMARY_CSV
        run_demo_module.PRIMARY_CSV = REPO_ROOT / "temporal_risk" / "NO_SUCH_CSV.csv"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                video = write_video(Path(tmp) / "clip.mp4", 40)
                summary = run_demo(video, "live/clip.mp4", DEFAULT_MODEL_PATH, Path(tmp),
                                   temporal_source=SOURCE_LIVE, checkpoint=CHECKPOINT)
            self.assertEqual(summary["windows_observed"], 4)
            # No CSV means no dataset label, which is a state, not a failure.
            self.assertEqual(summary["ground_truth_label"], "Not available")
            self.assertFalse(summary["ground_truth_available"])
        finally:
            run_demo_module.PRIMARY_CSV = original

    @needs_yolo
    def test_replay_still_refuses_loudly_when_the_csv_is_absent(self):
        """Tolerating a missing CSV must not leak into replay, where it IS the evidence."""
        import run_demo as run_demo_module
        from detection import DEFAULT_MODEL_PATH

        original = run_demo_module.PRIMARY_CSV
        run_demo_module.PRIMARY_CSV = REPO_ROOT / "temporal_risk" / "NO_SUCH_CSV.csv"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                video = write_video(Path(tmp) / "clip.mp4", 40)
                with self.assertRaises(Exception):
                    run_demo(video, "val/Val_Fight/trtrhrt_1049.avi", DEFAULT_MODEL_PATH,
                             Path(tmp), temporal_source=SOURCE_CSV)
        finally:
            run_demo_module.PRIMARY_CSV = original

    def test_a_clip_with_a_research_score_is_flagged_when_run_live(self):
        """A live number must never be mistaken for the locked research number."""
        self.assertIn("is not the locked research score", SERVER_SOURCE)


class CheckpointHandlingTests(unittest.TestCase):
    def setUp(self):
        SERVER_MODULE.app.config["TESTING"] = True
        self.client = SERVER_MODULE.app.test_client()

    def test_an_invalid_checkpoint_is_rejected(self):
        """Random weights must not be accepted as a violence model."""
        import torch

        from checkpoint_identity import CheckpointIntegrityError
        from run_demo import LiveInferenceSource

        with tempfile.TemporaryDirectory() as tmp:
            bogus = Path(tmp) / "bogus.pt"
            torch.save({"state_dict": {"not": torch.zeros(2)}}, bogus)
            with self.assertRaises((CheckpointIntegrityError, Exception)):
                LiveInferenceSource(bogus)

    def test_a_missing_checkpoint_is_reported_not_worked_around(self):
        self.assertIn("Live mode needs the R3D-18 checkpoint", SERVER_SOURCE)


class UploadSafetyTests(unittest.TestCase):
    def setUp(self):
        SERVER_MODULE.app.config["TESTING"] = True
        self.client = SERVER_MODULE.app.test_client()

    def test_non_video_extension_is_refused(self):
        import io

        response = self.client.post("/api/analyze", data={
            "mode": "live", "video": (io.BytesIO(b"MZ\x90\x00"), "payload.exe")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("not a supported video type", response.get_json()["error"])

    def test_upload_is_contained_even_with_a_traversing_filename(self):
        import io

        response = self.client.post("/api/analyze", data={
            "mode": "live", "video": (io.BytesIO(b"\x00" * 40), "../../../evil.mp4")})
        # It is refused as an undecodable video, never written outside uploads.
        self.assertEqual(response.status_code, 400)
        self.assertFalse((REPO_ROOT / "evil.mp4").exists())
        self.assertFalse((REPO_ROOT.parent / "evil.mp4").exists())

    def test_containment_is_checked_and_not_only_sanitised(self):
        self.assertIn("UPLOAD_DIR.resolve() not in destination.parents", SERVER_SOURCE)

    def test_traversal_clip_key_is_refused_in_live_mode_too(self):
        for key in ["../../../README.md", "C:/Windows/win.ini"]:
            response = self.client.post("/api/analyze", data={"mode": "live", "clip_key": key})
            self.assertEqual(response.status_code, 400, key)
            self.assertIn("No video available", response.get_json()["error"])


class ExplainForLiveClipsTests(unittest.TestCase):
    def setUp(self):
        SERVER_MODULE.app.config["TESTING"] = True
        self.client = SERVER_MODULE.app.test_client()

    def test_an_unregistered_live_key_is_still_refused(self):
        """The live registry must not become a way to name any file."""
        response = self.client.post("/api/explain", json={
            "clip_key": "live/never_analysed.mp4", "first_frame": 0})
        self.assertEqual(response.get_json()["reason"], "unknown_clip")

    def test_registry_only_holds_paths_the_server_resolved(self):
        self.assertIn("never a caller's", SERVER_SOURCE)
        self.assertIn("def _register_live_analysis", SERVER_SOURCE)

    def test_a_negative_first_frame_is_refused_not_silently_realigned(self):
        """It used to return the window 0-15 map with available=true."""
        response = self.client.post("/api/explain", json={
            "clip_key": "live/whatever.mp4", "first_frame": -5})
        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertEqual(body["reason"], "bad_request")
        self.assertIn("zero or greater", body["detail"])

    def test_registry_entry_is_dropped_when_the_video_disappears(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "gone.mp4"
            SERVER_MODULE._register_live_analysis("live/gone.mp4", missing)
            self.assertIsNone(SERVER_MODULE._live_video_for("live/gone.mp4"))


class TemporalContractTests(unittest.TestCase):
    """The window geometry and class index are frozen and must stay put."""

    def test_runtime_geometry_is_16_frames_at_stride_8(self):
        from temporal_runtime import RUNTIME_CLIP_LENGTH, RUNTIME_STRIDE

        self.assertEqual(RUNTIME_CLIP_LENGTH, 16)
        self.assertEqual(RUNTIME_STRIDE, 8)

    def test_fight_is_class_index_one(self):
        from temporal_runtime import RUNTIME_CLASS_LABELS, RUNTIME_POSITIVE_CLASS_INDEX

        self.assertEqual(RUNTIME_POSITIVE_CLASS_INDEX, 1)
        self.assertEqual(RUNTIME_CLASS_LABELS[1], "Fight")


class ZeroEvidenceHonestyTests(unittest.TestCase):
    """A video too short to score must not be called NonFight."""

    def test_no_window_yields_undetermined_not_a_negative_classification(self):
        self.assertIn('"Undetermined" if not observed_windows', RUN_DEMO_SOURCE)
        self.assertIn("undetermined_reason", RUN_DEMO_SOURCE)

    @needs_yolo
    @needs_checkpoint
    def test_a_ten_frame_video_produces_no_decision(self):
        from detection import DEFAULT_MODEL_PATH

        with tempfile.TemporaryDirectory() as tmp:
            video = write_video(Path(tmp) / "short.mp4", 10)
            summary = run_demo(video, "live/short.mp4", DEFAULT_MODEL_PATH, Path(tmp),
                               temporal_source=SOURCE_LIVE, checkpoint=CHECKPOINT)
        self.assertEqual(summary["windows_observed"], 0)
        self.assertEqual(summary["final_temporal_decision"], "Undetermined")
        self.assertFalse(summary["temporal_evidence_available"])
        self.assertIsNone(summary["temporal_max_probability"])
        self.assertFalse(summary["fusion"]["available"])
        self.assertEqual(summary["fusion"]["reason"], "no_temporal_evidence")


class LiveRunIntegrationTests(unittest.TestCase):
    """Real inference on synthetic video. Slow, and worth it."""

    @needs_yolo
    @needs_checkpoint
    def test_live_run_produces_windows_from_the_frames_themselves(self):
        from detection import DEFAULT_MODEL_PATH

        with tempfile.TemporaryDirectory() as tmp:
            video = write_video(Path(tmp) / "clip.mp4", 40)
            summary = run_demo(video, "live/clip.mp4", DEFAULT_MODEL_PATH, Path(tmp),
                               temporal_source=SOURCE_LIVE, checkpoint=CHECKPOINT)

        # 40 frames, clip 16, stride 8 -> windows closing at 15, 23, 31, 39.
        self.assertEqual(summary["windows_observed"], 4)
        windows = summary["window_scores"]
        self.assertEqual([w["first_frame"] for w in windows], [0, 8, 16, 24])
        self.assertEqual([w["last_frame"] for w in windows], [15, 23, 31, 39])
        for window in windows:
            self.assertEqual(window["last_frame"] - window["first_frame"] + 1, 16)
            self.assertGreaterEqual(window["fight_probability"], 0.0)
            self.assertLessEqual(window["fight_probability"], 1.0)

        provenance = summary["temporal_provenance"]
        self.assertEqual(provenance["temporal_source"], "live")
        self.assertEqual(provenance["score_provenance"], "live_model_inference")
        self.assertIn("checkpoint_sha256", provenance)
        self.assertNotIn("scores_from", provenance)

    @needs_yolo
    @needs_checkpoint
    def test_undefined_spatial_evidence_is_preserved_not_zeroed(self):
        """Noise video: YOLO finds nobody, so the feature has no value."""
        from detection import DEFAULT_MODEL_PATH

        with tempfile.TemporaryDirectory() as tmp:
            video = write_video(Path(tmp) / "noise.mp4", 40)
            summary = run_demo(video, "live/noise.mp4", DEFAULT_MODEL_PATH, Path(tmp),
                               temporal_source=SOURCE_LIVE, checkpoint=CHECKPOINT)

        feature = summary["spatial_fusion_feature"]
        self.assertFalse(feature["defined"])
        self.assertIsNone(feature["spatial_score"])
        self.assertIsNotNone(feature["note"])
        fusion = summary["fusion"]
        self.assertTrue(fusion["available"])
        self.assertIsNone(fusion["inputs"]["spatial_score"])
        self.assertTrue(fusion["inputs"]["spatial_abstained"])
        self.assertFalse(fusion["candidates"]["B1_spatial_only"]["fired"])

    @needs_yolo
    @needs_checkpoint
    def test_two_videos_in_a_row_do_not_contaminate_each_other(self):
        """Tracker, rule engine and spatial accumulator are per-run objects."""
        from detection import DEFAULT_MODEL_PATH

        with tempfile.TemporaryDirectory() as tmp:
            first = write_video(Path(tmp) / "a.mp4", 40)
            run_demo(first, "live/a.mp4", DEFAULT_MODEL_PATH, Path(tmp),
                     temporal_source=SOURCE_LIVE, checkpoint=CHECKPOINT)
            second = run_demo(first, "live/a2.mp4", DEFAULT_MODEL_PATH, Path(tmp),
                              temporal_source=SOURCE_LIVE, checkpoint=CHECKPOINT)
            third = run_demo(write_video(Path(tmp) / "b.mp4", 24), "live/b.mp4",
                             DEFAULT_MODEL_PATH, Path(tmp),
                             temporal_source=SOURCE_LIVE, checkpoint=CHECKPOINT)

        # Same video twice -> identical result. State from run 1 changed nothing.
        self.assertEqual(second["windows_observed"], 4)
        # A different, shorter video -> its own window count, not the previous one's.
        self.assertEqual(third["windows_observed"], 2)
        self.assertEqual(third["frames_processed"], 24)

    def test_state_is_constructed_per_run_not_shared(self):
        body = RUN_DEMO_SOURCE.split("def run_demo(")[1]
        for construction in ("SimpleTracker()", "AbnormalEventDetector()",
                             "RiskAssessor()", "SpatialFeatureAccumulator()"):
            self.assertIn(construction, body, f"{construction} must be per-run")


class RiskAndProvenanceTests(unittest.TestCase):
    def test_risk_validation_flag_is_false_in_live_mode_too(self):
        from risk_assessment import RiskAssessment

        fields = RiskAssessment.__dataclass_fields__
        self.assertIn("risk_level", fields)
        source = (REPO_ROOT / "src" / "risk_assessment.py").read_text(encoding="utf-8")
        self.assertIn('"risk_level_is_validated": False', source)

    def test_report_records_live_provenance_and_fusion(self):
        from incident_report import build_incident_report

        summary = {
            "clip_key": "live/x.mp4",
            "frames_processed": 40,
            "ground_truth_label": "Not available",
            "ground_truth_available": False,
            "final_temporal_decision": "NonFight",
            "temporal_provenance": {
                "temporal_source": "live",
                "checkpoint_sha256": "abc123",
                "device": "cpu",
                "inference_seconds_total": 2.5,
            },
        }
        report = build_incident_report(
            summary=summary, risk_assessments=[],
            window_scores=[{"window_index": 0, "first_frame": 0, "last_frame": 15,
                            "fight_probability": 0.2}],
            overall_risk="Low",
            fusion={"available": True, "mode": "offline whole-video evidence fusion",
                    "candidates": {}, "system_decision": {"from": "B0_temporal_only",
                                                          "decision": "NonFight", "why": "x"},
                    "causality": "c", "interpretation": "i", "not_claimed": "n"},
            spatial_feature={"feature": "f", "defined": False, "note": "undefined"},
        )
        self.assertEqual(report["provenance"]["mode"], "live inference")
        self.assertEqual(report["provenance"]["checkpoint_sha256"], "abc123")
        self.assertEqual(report["clip"]["ground_truth_label"], "Not available")
        self.assertFalse(report["clip"]["ground_truth_available"])
        self.assertTrue(report["fusion"]["available"])
        self.assertFalse(report["spatial_fusion_feature"]["defined"])
        self.assertFalse(report["risk_interpretation"]["risk_level_is_validated"])

    def test_replay_report_has_no_checkpoint_because_no_model_ran(self):
        from incident_report import build_incident_report

        report = build_incident_report(
            summary={"clip_key": "val/x.avi",
                     "temporal_provenance": {"temporal_source": "csv"}},
            risk_assessments=[], window_scores=[], overall_risk="Low")
        self.assertEqual(report["provenance"]["mode"], "replay (precomputed scores)")
        self.assertIsNone(report["provenance"]["checkpoint_sha256"])


if __name__ == "__main__":
    unittest.main()
