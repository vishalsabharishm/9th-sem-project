"""Tests for Phase 4: genuine live R3D-18 inference in the demo runtime.

Most tests here need no checkpoint -- they cover source selection, the refusal
paths, and the provenance contract. The tests that require the real 132 MB
``best.pt`` skip cleanly when it is absent, so the suite still runs on a
machine that has not recovered it.

The property these tests exist to protect: live mode must never silently become
something else. Not a CSV replay, not an untrained model, not a partially
verified checkpoint. Every failure path raises.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from checkpoint_identity import CheckpointIntegrityError, file_sha256  # noqa: E402
from run_demo import (  # noqa: E402
    SOURCE_CSV,
    SOURCE_LIVE,
    LiveInferenceSource,
    PrecomputedReplaySource,
    build_window_source,
)
from temporal_event_adapter import (  # noqa: E402
    PROVENANCE_LIVE_INFERENCE,
    PrecomputedWindowScoreSource,
)

CHECKPOINT = REPO_ROOT / "models" / "temporal_violence" / "best.pt"
EXPECTED_SHA256 = "a32271bfb5a9c273f84672b16ecd5fa351fdd617d39a2c5c73f5c667255a559a"
EXPECTED_BYTES = 132_744_907
PRIMARY_CSV = REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"

FP_CLIP = REPO_ROOT / "data" / "rwf2000" / "RWF-2000" / "val" / "Val_NonFight" / "39BFeYnbu-I_0.avi"
FP_KEY = "val/Val_NonFight/39BFeYnbu-I_0.avi"
# One of the six leakage-excluded clips: present on disk, absent from the CSV.
UNRECORDED_CLIP = REPO_ROOT / "data" / "rwf2000" / "RWF-2000" / "val" / "Val_NonFight" / "cw8fPfUL_0.avi"

HAS_CHECKPOINT = CHECKPOINT.is_file()
HAS_CLIPS = FP_CLIP.is_file()
FRAME = np.zeros((64, 64, 3), dtype=np.uint8)


def committed_windows(clip_key=FP_KEY):
    return PrecomputedWindowScoreSource(PRIMARY_CSV).get(clip_key)


class CheckpointIdentityTests(unittest.TestCase):
    """The recovered artifact must stay exactly what was verified."""

    @unittest.skipUnless(HAS_CHECKPOINT, "best.pt not present")
    def test_checkpoint_hash_is_the_recovered_one(self):
        self.assertEqual(file_sha256(CHECKPOINT), EXPECTED_SHA256)

    @unittest.skipUnless(HAS_CHECKPOINT, "best.pt not present")
    def test_checkpoint_size_is_exact(self):
        self.assertEqual(CHECKPOINT.stat().st_size, EXPECTED_BYTES)

    @unittest.skipUnless(HAS_CHECKPOINT, "best.pt not present")
    def test_checkpoint_passes_the_repository_guard(self):
        from checkpoint_identity import verify_trained_checkpoint

        verdict = verify_trained_checkpoint(CHECKPOINT)
        self.assertTrue(verdict.accepted, verdict.rejections)

    @unittest.skipUnless(HAS_CHECKPOINT, "best.pt not present")
    def test_checkpoint_is_the_clean_baseline_not_experiment_one(self):
        """Guards against accidentally using notebook5d98537e15 artifacts."""
        from checkpoint_identity import inspect_checkpoint

        metadata = inspect_checkpoint(CHECKPOINT)
        self.assertEqual(metadata["epoch"], 12)
        self.assertAlmostEqual(metadata["monitored_value"], 0.935342732134176, places=12)
        self.assertEqual(metadata["num_classes"], 2)
        self.assertEqual(list(metadata["class_labels"]), ["NonFight", "Fight"])
        config = metadata["config"]
        self.assertTrue(config.get("holdout_validation"))
        self.assertEqual(config.get("validation_fraction"), 0.15)
        # Carve-validation support, not the 394-clip reporting split.
        self.assertEqual(metadata["metrics"]["support"], {"NonFight": 121, "Fight": 119})

    def test_no_weight_file_is_tracked_by_git(self):
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
        self.assertEqual([p for p in tracked if p.endswith((".pt", ".pth", ".ckpt"))], [])


class SourceSelectionTests(unittest.TestCase):
    """CSV stays the default; live never degrades."""

    def test_csv_remains_the_default(self):
        import inspect
        from run_demo import run_demo

        self.assertEqual(inspect.signature(run_demo).parameters["temporal_source"].default,
                         SOURCE_CSV)

    def test_live_without_a_checkpoint_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            build_window_source(SOURCE_LIVE, FP_KEY, committed_windows(), checkpoint=None)
        self.assertIn("--checkpoint", str(caught.exception))

    def test_live_never_falls_back_to_csv_when_the_checkpoint_is_bad(self):
        """Even with valid CSV windows available, a bad checkpoint must raise."""
        with tempfile.TemporaryDirectory() as directory:
            bogus = Path(directory) / "best.pt"
            torch.save({"fc.weight": torch.zeros(2, 4)}, bogus)
            with self.assertRaises(CheckpointIntegrityError):
                build_window_source(
                    SOURCE_LIVE, FP_KEY, committed_windows(), checkpoint=bogus
                )

    def test_live_never_falls_back_to_random_initialisation(self):
        with tempfile.TemporaryDirectory() as directory:
            smoke = Path(directory) / "best.pt"
            torch.save(
                {
                    "state_dict": {"fc.weight": torch.zeros(2, 4)},
                    "epoch": 1, "monitor": "roc_auc", "monitored_value": 1.0,
                    "metrics": {"accuracy": 1.0},
                    "config": {"pretrained_backbone": False,
                               "limit_train_clips": 4, "limit_eval_clips": 4},
                    "num_classes": 2, "class_labels": ["NonFight", "Fight"],
                    "training_provenance": "random_init",
                },
                smoke,
            )
            with self.assertRaises(CheckpointIntegrityError) as caught:
                build_window_source(SOURCE_LIVE, FP_KEY, None, checkpoint=smoke)
            self.assertIn("randomly initialised", str(caught.exception))

    def test_csv_mode_refuses_a_clip_with_no_committed_row(self):
        with self.assertRaises(SystemExit) as caught:
            build_window_source(SOURCE_CSV, "val/Val_NonFight/cw8fPfUL_0.avi", None)
        self.assertIn("--temporal-source live", str(caught.exception))

    def test_unknown_source_is_refused(self):
        with self.assertRaises(SystemExit):
            build_window_source("magic", FP_KEY, committed_windows())

    def test_cli_exposes_both_sources_with_csv_default(self):
        from run_demo import main  # noqa: F401  (ensures the module parses)
        import argparse
        import run_demo as module

        source = (REPO_ROOT / "tools" / "run_demo.py").read_text(encoding="utf-8")
        self.assertIn('"--temporal-source"', source)
        self.assertIn('"--checkpoint"', source)
        self.assertIn("default=SOURCE_CSV", source)


@unittest.skipUnless(HAS_CHECKPOINT, "best.pt not present")
class LiveSourceTests(unittest.TestCase):
    """The live source against the real checkpoint."""

    @classmethod
    def setUpClass(cls):
        cls.source = LiveInferenceSource(CHECKPOINT, device="cpu")

    def test_uses_the_shared_runtime_factory_geometry(self):
        from temporal_runtime import RUNTIME_CLIP_LENGTH, RUNTIME_STRIDE

        self.assertEqual(self.source.clip_length, RUNTIME_CLIP_LENGTH)
        self.assertEqual(self.source.stride, RUNTIME_STRIDE)

    def test_model_is_task_specific(self):
        self.assertTrue(self.source.engine.model.is_task_specific)
        self.assertTrue(self.source.model_provenance.startswith("checkpoint"))

    def test_model_has_the_two_class_violence_head(self):
        self.assertEqual(self.source.engine.model.config.num_classes, 2)
        self.assertEqual(self.source.engine.model.config.class_labels, ("NonFight", "Fight"))

    def test_no_window_before_sixteen_frames(self):
        source = LiveInferenceSource(CHECKPOINT, device="cpu")
        for index in range(15):
            self.assertEqual(source.windows_completed_at(index, FRAME), [])

    def test_first_window_completes_at_frame_fifteen(self):
        source = LiveInferenceSource(CHECKPOINT, device="cpu")
        for index in range(15):
            source.windows_completed_at(index, FRAME)
        completed = source.windows_completed_at(15, FRAME)
        self.assertEqual(len(completed), 1)
        self.assertEqual((completed[0].first_frame, completed[0].last_frame), (0, 15))
        self.assertEqual(completed[0].window_index, 0)

    def test_windows_arrive_at_stride_eight(self):
        source = LiveInferenceSource(CHECKPOINT, device="cpu")
        bounds = []
        for index in range(40):
            for score in source.windows_completed_at(index, FRAME):
                bounds.append((score.first_frame, score.last_frame))
        self.assertEqual(bounds, [(0, 15), (8, 23), (16, 31), (24, 39)])

    def test_hud_is_strictly_causal(self):
        """Unlike replay, live cannot report a window that has not completed."""
        source = LiveInferenceSource(CHECKPOINT, device="cpu")
        self.assertIsNone(source.hud_probability(0, []))
        observed = []
        for index in range(16):
            observed.extend(source.windows_completed_at(index, FRAME))
        self.assertEqual(source.hud_probability(15, observed), observed[-1].fight_probability)

    def test_describe_records_full_live_provenance(self):
        source = LiveInferenceSource(CHECKPOINT, device="cpu")
        for index in range(16):
            source.windows_completed_at(index, FRAME)
        described = source.describe()
        self.assertEqual(described["temporal_source"], SOURCE_LIVE)
        self.assertEqual(described["score_provenance"], PROVENANCE_LIVE_INFERENCE)
        self.assertEqual(described["checkpoint_sha256"], EXPECTED_SHA256)
        self.assertEqual(described["checkpoint_bytes"], EXPECTED_BYTES)
        self.assertEqual(described["window_geometry"]["clip_length"], 16)
        self.assertEqual(described["window_geometry"]["stride"], 8)
        self.assertEqual(described["window_geometry"]["windows_completed"], 1)
        self.assertGreater(described["inference_seconds_total"], 0.0)
        self.assertIn("no CSV was read", described["note"])

    def test_replay_provenance_is_never_claimed_by_the_live_source(self):
        described = LiveInferenceSource(CHECKPOINT, device="cpu").describe()
        self.assertNotIn("scores_from", described)
        self.assertNotEqual(described["temporal_source"], SOURCE_CSV)


@unittest.skipUnless(HAS_CHECKPOINT and HAS_CLIPS, "checkpoint or dataset not present")
class LiveVersusCommittedScoresTests(unittest.TestCase):
    """Live inference reproduces the committed Kaggle scores within tolerance.

    Close agreement is SUPPORTING EVIDENCE that the checkpoint, preprocessing,
    window geometry and model loading are consistent with the recorded run. It
    is not a component-level proof: the test compares end-to-end outputs, so it
    cannot attribute correctness to any individual stage, and a compensating
    pair of errors would not be detected here.
    """

    # The tolerance is NOT tuned to force a pass: the observed maximum absolute
    # difference was 1.8e-5, well inside it.
    #
    # A difference of this magnitude is CONSISTENT WITH running on a different
    # device (CPU here, a T4 when the CSV was recorded) and with the CSV storing
    # only 6 decimals, but neither contribution was isolated and no causal claim
    # is made. Establishing that would need the same weights run on both devices
    # with full-precision output retained, which this repository cannot do.
    TOLERANCE = 1e-4

    @classmethod
    def setUpClass(cls):
        import cv2

        source = LiveInferenceSource(CHECKPOINT, device="cpu")
        capture = cv2.VideoCapture(str(FP_CLIP))
        cls.live, index = [], 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cls.live.extend(source.windows_completed_at(index, frame))
            index += 1
        capture.release()
        cls.committed = committed_windows()

    def test_same_number_of_windows(self):
        self.assertEqual(len(self.live), len(self.committed))
        self.assertEqual(len(self.live), 17)

    def test_identical_window_geometry(self):
        self.assertEqual(
            [(w.first_frame, w.last_frame) for w in self.live],
            [(w.first_frame, w.last_frame) for w in self.committed],
        )

    def test_probabilities_agree_within_tolerance(self):
        for live, committed in zip(self.live, self.committed):
            with self.subTest(window=live.window_index):
                self.assertAlmostEqual(
                    live.fight_probability, committed.fight_probability,
                    delta=self.TOLERANCE,
                )

    def test_the_clip_level_decision_is_identical(self):
        rule_threshold = 0.14
        self.assertEqual(
            max(w.fight_probability for w in self.live) >= rule_threshold,
            max(w.fight_probability for w in self.committed) >= rule_threshold,
        )


@unittest.skipUnless(HAS_CHECKPOINT and UNRECORDED_CLIP.is_file(),
                     "checkpoint or the unrecorded clip not present")
class UnrecordedClipTests(unittest.TestCase):
    """Evidence replay cannot fake: scoring a clip with no CSV row.

    A clip absent from the committed file has nothing to replay, so a score for
    it can only have come from a forward pass in this process.
    """

    def test_the_clip_really_has_no_committed_row(self):
        self.assertIsNone(
            PrecomputedWindowScoreSource(PRIMARY_CSV).get("val/Val_NonFight/cw8fPfUL_0.avi")
        )

    def test_live_scores_it_anyway(self):
        import cv2

        source = LiveInferenceSource(CHECKPOINT, device="cpu")
        capture = cv2.VideoCapture(str(UNRECORDED_CLIP))
        observed, index = [], 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            observed.extend(source.windows_completed_at(index, frame))
            index += 1
        capture.release()
        self.assertEqual(len(observed), 17)
        for score in observed:
            self.assertTrue(0.0 <= score.fight_probability <= 1.0)


class GroundTruthLabelTests(unittest.TestCase):
    """A missing dataset label must never surface as Python ``None``."""

    def test_the_placeholder_is_a_readable_string(self):
        from run_demo import GROUND_TRUTH_UNAVAILABLE

        self.assertIsInstance(GROUND_TRUTH_UNAVAILABLE, str)
        self.assertNotEqual(GROUND_TRUTH_UNAVAILABLE, "None")
        self.assertTrue(GROUND_TRUTH_UNAVAILABLE.strip())

    def test_absence_is_preserved_separately_from_the_display_string(self):
        """The placeholder is cosmetic; the fact stays machine-readable."""
        source = (REPO_ROOT / "tools" / "run_demo.py").read_text(encoding="utf-8")
        self.assertIn('"ground_truth_available": true_label is not None', source)
        self.assertIn("GROUND_TRUTH_UNAVAILABLE", source)

    def test_a_known_clip_keeps_its_real_label(self):
        """CSV mode must be unaffected -- the placeholder only fills a gap."""
        labels = PrecomputedWindowScoreSource(PRIMARY_CSV)
        self.assertEqual(labels.true_label(FP_KEY), "NonFight")

    def test_an_unrecorded_clip_has_no_label_to_report(self):
        labels = PrecomputedWindowScoreSource(PRIMARY_CSV)
        self.assertIsNone(labels.true_label("val/Val_NonFight/cw8fPfUL_0.avi"))


class WebServerModeSelectionTests(unittest.TestCase):
    """The browser UI now offers live mode, and must choose it EXPLICITLY.

    This class previously asserted the opposite -- that web/server.py never
    mentioned ``temporal_source`` at all, because live inference was
    deliberately kept on the CLI while the UI stayed replay-only. That scope
    boundary has been lifted on purpose; the UI now exposes both modes.

    What those tests were really protecting is not the boundary but the
    property behind it: replay and live must never be confusable. That property
    is stronger here than it was there, so it is asserted directly instead of
    via the absence of a substring.
    """

    def setUp(self):
        self.source = (REPO_ROOT / "web" / "server.py").read_text(encoding="utf-8")

    def test_web_server_selects_the_temporal_source_explicitly(self):
        self.assertIn("temporal_source=", self.source)
        self.assertIn('"live" if mode == MODE_LIVE else "csv"', self.source)

    def test_mode_is_read_from_the_request_and_defaults_to_replay(self):
        self.assertIn('request.form.get("mode") or MODE_REPLAY', self.source)

    def test_an_unknown_mode_is_rejected_rather_than_guessed(self):
        self.assertIn("if mode not in SUPPORTED_MODES:", self.source)

    def test_live_mode_requires_a_checkpoint_and_never_falls_back(self):
        """A live request that cannot run live must fail, not replay."""
        self.assertIn("checkpoint=(checkpoint if mode == MODE_LIVE else None)", self.source)
        self.assertIn("There is no fallback", self.source)

    def test_live_window_scores_come_from_the_run_not_the_csv(self):
        """The one place a live result could silently become a replayed one."""
        self.assertIn('summary.get("window_scores")', self.source)

    def test_the_response_states_which_mode_produced_it(self):
        self.assertIn('"mode": mode', self.source)
        self.assertIn("mode_label", self.source)


if __name__ == "__main__":
    unittest.main()
