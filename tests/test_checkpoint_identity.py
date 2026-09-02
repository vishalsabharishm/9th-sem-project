"""Tests for checkpoint identity and the provenance guard.

The hole being closed: ``R3D18TemporalModel.load_checkpoint`` sets provenance
to ``checkpoint:<name>`` for any loadable file, so ``is_task_specific`` is True
even for the randomly initialised checkpoint ``--validate-pipeline`` writes.
Verified empirically during Step 3 on a real smoke-test checkpoint, which the
old guard accepted and the new guard rejects.

Payloads here are synthetic and tiny -- a real R3D-18 checkpoint is 132 MB and
would make the suite unusable. What matters is the recorded metadata, which is
exactly what the guard reads, so a one-tensor state dict exercises the same
code path as the real thing.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from checkpoint_identity import (  # noqa: E402
    EXPECTED_NUM_CLASSES,
    REQUIRED_KEYS,
    CheckpointIntegrityError,
    CheckpointVerdict,
    checkpoint_record,
    file_sha256,
    inspect_checkpoint,
    render_checkpoint_md,
    verify_trained_checkpoint,
)
from rwf2000_config import CANONICAL_LABELS  # noqa: E402


def trained_payload(**overrides):
    """A payload shaped like a genuine completed training run."""
    payload = {
        "state_dict": {"fc.weight": torch.zeros(2, 4)},
        "epoch": 12,
        "monitor": "roc_auc",
        "monitored_value": 0.935342732134176,
        "metrics": {"accuracy": 0.875, "roc_auc": 0.9353},
        "config": {
            "protocol": "carved_validation",
            "seed": 42,
            "pretrained_backbone": True,
            "limit_train_clips": None,
            "limit_eval_clips": None,
            "optimizer": "AdamW",
        },
        "num_classes": EXPECTED_NUM_CLASSES,
        "class_labels": list(CANONICAL_LABELS),
        "training_provenance": "kinetics400_pretrained+untrained_head",
        "protocol": "carved_validation",
        "saved_at": "2026-08-25T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def write(payload, directory: Path, name: str = "best.pt") -> Path:
    path = Path(directory) / name
    torch.save(payload, path)
    return path


class TempCheckpointCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class AcceptanceTests(TempCheckpointCase):
    """A valid trained checkpoint must be accepted."""

    def test_a_genuine_trained_checkpoint_is_accepted(self):
        verdict = verify_trained_checkpoint(write(trained_payload(), self.tmp))
        self.assertTrue(verdict.accepted, verdict.rejections)
        self.assertEqual(verdict.rejections, ())

    def test_acceptance_does_not_depend_on_the_filename(self):
        """`best.pt` is a path convention, not evidence."""
        good = write(trained_payload(), self.tmp, name="some_other_name.pt")
        self.assertTrue(verify_trained_checkpoint(good).accepted)
        bad = write(trained_payload(training_provenance="random_init"),
                    self.tmp, name="best.pt")
        self.assertFalse(verify_trained_checkpoint(bad).accepted)

    def test_experiment1_checkpoint_is_accepted_but_warned_about(self):
        payload = trained_payload(protocol="experiment1_primary_monitor")
        payload["config"]["protocol"] = "experiment1_primary_monitor"
        verdict = verify_trained_checkpoint(write(payload, self.tmp))
        self.assertTrue(verdict.accepted, verdict.rejections)
        self.assertTrue(
            any("optimistically biased" in w for w in verdict.warnings),
            verdict.warnings,
        )


class RejectionTests(TempCheckpointCase):
    """Every documented rejection rule."""

    def test_random_init_provenance_is_rejected(self):
        verdict = verify_trained_checkpoint(
            write(trained_payload(training_provenance="random_init"), self.tmp)
        )
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("randomly initialised" in r for r in verdict.rejections))

    def test_missing_monitored_value_is_rejected(self):
        verdict = verify_trained_checkpoint(
            write(trained_payload(monitored_value=None), self.tmp)
        )
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("monitored_value is None" in r for r in verdict.rejections))

    def test_missing_metrics_block_is_rejected(self):
        verdict = verify_trained_checkpoint(write(trained_payload(metrics=None), self.tmp))
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("metrics block is None" in r for r in verdict.rejections))

    def test_each_required_key_is_required(self):
        for key in REQUIRED_KEYS:
            payload = trained_payload()
            del payload[key]
            with self.subTest(missing=key):
                verdict = verify_trained_checkpoint(write(payload, self.tmp, f"{key}.pt"))
                self.assertFalse(verdict.accepted, f"{key} should be required")

    def test_subsetted_smoke_test_run_is_rejected(self):
        """The exact configuration --validate-pipeline produces."""
        payload = trained_payload()
        payload["config"]["limit_train_clips"] = 4
        payload["config"]["limit_eval_clips"] = 4
        verdict = verify_trained_checkpoint(write(payload, self.tmp))
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("subset of the data" in r for r in verdict.rejections))

    def test_wrong_number_of_classes_is_rejected(self):
        verdict = verify_trained_checkpoint(write(trained_payload(num_classes=400), self.tmp))
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("num_classes" in r for r in verdict.rejections))

    def test_wrong_class_labels_are_rejected(self):
        verdict = verify_trained_checkpoint(
            write(trained_payload(class_labels=["cat", "dog"]), self.tmp)
        )
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("class_labels" in r for r in verdict.rejections))

    def test_zero_epoch_is_rejected(self):
        verdict = verify_trained_checkpoint(write(trained_payload(epoch=0), self.tmp))
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("epoch" in r for r in verdict.rejections))

    def test_a_bare_state_dict_is_rejected(self):
        """torch.load succeeding is not evidence of training."""
        path = self.tmp / "bare.pt"
        torch.save({"fc.weight": torch.zeros(2, 4)}, path)
        verdict = verify_trained_checkpoint(path)
        self.assertFalse(verdict.accepted)

    def test_a_missing_file_is_rejected_not_crashed(self):
        verdict = verify_trained_checkpoint(self.tmp / "nope.pt")
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("not found" in r for r in verdict.rejections))

    def test_a_non_torch_file_is_rejected_not_crashed(self):
        path = self.tmp / "notacheckpoint.pt"
        path.write_bytes(b"this is not a torch archive")
        verdict = verify_trained_checkpoint(path)
        self.assertFalse(verdict.accepted)

    def test_raise_if_rejected_names_every_reason(self):
        payload = trained_payload(training_provenance="random_init", monitored_value=None)
        verdict = verify_trained_checkpoint(write(payload, self.tmp))
        with self.assertRaises(CheckpointIntegrityError) as caught:
            verdict.raise_if_rejected()
        message = str(caught.exception)
        self.assertIn("randomly initialised", message)
        self.assertIn("monitored_value is None", message)

    def test_raise_if_rejected_is_a_no_op_when_accepted(self):
        verdict = verify_trained_checkpoint(write(trained_payload(), self.tmp))
        self.assertIs(verdict.raise_if_rejected(), verdict)


class LegacyCheckpointTests(TempCheckpointCase):
    """The authentic best.pt predates the training_provenance field."""

    def test_legacy_pretrained_checkpoint_is_accepted_with_an_inference_warning(self):
        payload = trained_payload()
        del payload["training_provenance"]
        del payload["protocol"]
        verdict = verify_trained_checkpoint(write(payload, self.tmp))
        self.assertTrue(verdict.accepted, verdict.rejections)
        self.assertTrue(verdict.inferred_initialisation)
        self.assertTrue(any("inferred" in w for w in verdict.warnings))

    def test_legacy_smoke_test_checkpoint_is_still_rejected(self):
        """Without the explicit field, pretrained_backbone=False must catch it."""
        payload = trained_payload()
        del payload["training_provenance"]
        payload["config"]["pretrained_backbone"] = False
        verdict = verify_trained_checkpoint(write(payload, self.tmp))
        self.assertFalse(verdict.accepted)
        self.assertTrue(any("pretrained_backbone is False" in r for r in verdict.rejections))

    def test_legacy_checkpoint_with_no_initialisation_evidence_is_rejected(self):
        payload = trained_payload()
        del payload["training_provenance"]
        del payload["config"]["pretrained_backbone"]
        verdict = verify_trained_checkpoint(write(payload, self.tmp))
        self.assertFalse(verdict.accepted)


class HashingTests(TempCheckpointCase):
    """Hash recording must be deterministic and content-addressed."""

    def test_sha256_is_deterministic_across_calls(self):
        path = write(trained_payload(), self.tmp)
        self.assertEqual(file_sha256(path), file_sha256(path))

    def test_sha256_matches_a_known_value_for_known_bytes(self):
        path = self.tmp / "known.bin"
        path.write_bytes(b"abc")
        self.assertEqual(
            file_sha256(path),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_identical_content_hashes_identically_under_different_names(self):
        payload = trained_payload()
        first = write(payload, self.tmp, "a.pt")
        second = self.tmp / "b.pt"
        second.write_bytes(first.read_bytes())
        self.assertEqual(file_sha256(first), file_sha256(second))

    def test_different_content_hashes_differently(self):
        a = write(trained_payload(), self.tmp, "a.pt")
        b = write(trained_payload(epoch=13), self.tmp, "b.pt")
        self.assertNotEqual(file_sha256(a), file_sha256(b))

    def test_chunk_size_does_not_change_the_hash(self):
        path = write(trained_payload(), self.tmp)
        self.assertEqual(file_sha256(path, chunk_bytes=64), file_sha256(path))


class RecordTests(TempCheckpointCase):
    """The identity record must preserve provenance metadata."""

    def test_record_preserves_training_metadata(self):
        path = write(trained_payload(), self.tmp)
        record = checkpoint_record(path, retrieved_from="unit test")
        self.assertEqual(record["training"]["training_provenance"],
                         "kinetics400_pretrained+untrained_head")
        self.assertEqual(record["training"]["protocol"], "carved_validation")
        self.assertEqual(record["training"]["monitor"], "roc_auc")
        self.assertEqual(record["training"]["epoch"], 12)
        self.assertAlmostEqual(record["training"]["monitored_value"], 0.935342732134176)
        self.assertEqual(record["identity"]["retrieved_from"], "unit test")
        self.assertEqual(record["identity"]["sha256"], file_sha256(path))
        self.assertEqual(record["identity"]["bytes"], path.stat().st_size)

    def test_record_preserves_the_training_configuration(self):
        record = checkpoint_record(write(trained_payload(), self.tmp))
        self.assertEqual(record["configuration"]["seed"], 42)
        self.assertEqual(record["configuration"]["optimizer"], "AdamW")
        self.assertTrue(record["configuration"]["pretrained_backbone"])

    def test_record_is_deterministic_apart_from_the_timestamp(self):
        path = write(trained_payload(), self.tmp)
        first = checkpoint_record(path)
        second = checkpoint_record(path)
        first.pop("recorded_at")
        second.pop("recorded_at")
        self.assertEqual(first, second)

    def test_record_is_json_serialisable(self):
        record = checkpoint_record(write(trained_payload(), self.tmp))
        self.assertIsInstance(json.dumps(record, default=str), str)

    def test_record_of_a_rejected_checkpoint_carries_its_reasons(self):
        path = write(trained_payload(training_provenance="random_init"), self.tmp)
        record = checkpoint_record(path)
        self.assertFalse(record["verification"]["accepted"])
        self.assertTrue(record["verification"]["rejections"])


class InspectTests(TempCheckpointCase):
    def test_inspect_reads_metadata_without_building_a_model(self):
        metadata = inspect_checkpoint(write(trained_payload(), self.tmp))
        self.assertEqual(metadata["epoch"], 12)
        self.assertEqual(metadata["num_classes"], EXPECTED_NUM_CLASSES)
        self.assertEqual(metadata["missing_required_keys"], [])
        self.assertEqual(metadata["tensor_count"], 1)

    def test_inspect_reports_missing_keys_rather_than_raising(self):
        payload = trained_payload()
        del payload["metrics"]
        metadata = inspect_checkpoint(write(payload, self.tmp))
        self.assertIn("metrics", metadata["missing_required_keys"])


class MarkdownTests(unittest.TestCase):
    def test_absent_checkpoint_is_documented(self):
        rendered = render_checkpoint_md(
            [], missing=[{"filename": "best.pt", "status": "ABSENT from this working tree"}]
        )
        self.assertIn("Known but absent", rendered)
        self.assertIn("best.pt", rendered)
        self.assertIn("ABSENT", rendered)

    def test_states_that_weights_are_not_committed(self):
        rendered = render_checkpoint_md([])
        self.assertIn("not committed", rendered)

    def test_rejected_record_is_rendered_as_rejected(self):
        record = {
            "identity": {"filename": "x.pt", "sha256": "0" * 64, "bytes": 1,
                         "modified_utc": "2026-01-01T00:00:00+00:00", "retrieved_from": None},
            "verification": {"accepted": False, "rejections": ["random init"],
                             "warnings": [], "inferred_initialisation": False},
            "training": {"training_provenance": "random_init", "protocol": None,
                         "monitor": "roc_auc", "monitored_value": 1.0, "epoch": 1,
                         "num_classes": 2, "class_labels": ["NonFight", "Fight"]},
            "configuration": {},
        }
        rendered = render_checkpoint_md([record])
        self.assertIn("**NO**", rendered)
        self.assertIn("random init", rendered)


class CommittedCheckpointDocTests(unittest.TestCase):
    """The committed CHECKPOINT.md must state the real situation."""

    def test_checkpoint_md_exists_and_records_the_absent_checkpoint(self):
        path = REPO_ROOT / "models" / "temporal_violence" / "CHECKPOINT.md"
        self.assertTrue(path.is_file(), "models/temporal_violence/CHECKPOINT.md is missing")
        text = path.read_text(encoding="utf-8")
        self.assertIn("best.pt", text)
        self.assertIn("ABSENT", text)
        self.assertIn("NOT RECORDED", text)

    def test_no_checkpoint_weights_are_tracked_by_git(self):
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
        offenders = [p for p in tracked if p.endswith((".pt", ".pth", ".ckpt"))]
        self.assertEqual(offenders, [], f"model weights must not be committed: {offenders}")


if __name__ == "__main__":
    unittest.main()
