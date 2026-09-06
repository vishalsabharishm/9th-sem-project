"""Tests for the FROZEN confirmatory fusion protocol.

The protocol exists so that a later authorised task can evaluate fusion on the
protected primary split in one shot, with no free parameters left. These tests
guard the properties that make that defensible:

  * PRIMARY IS UNTOUCHED. Nothing in the calibration path can read primary, and
    the normalisation reference travels inside the frozen artifact so the
    confirmatory run cannot rank primary against its own distribution.
  * CALIBRATION IS DETERMINISTIC and confined to development data.
  * NOTHING IS LEFT UNFROZEN. The candidate set is closed, the 0.5/0.5 weight is
    a fixed design choice, thresholds are numeric in the artifact, and the
    success criteria are written down before primary is seen.
  * THE INVALID CARVE STAYS INVALID. The fresh carve overlaps R3D training data
    and must never be reclassified as a validation set.

Provenance is checked through recorded metadata and artifact filenames rather
than brittle substring matching wherever possible.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

PROTOCOL = REPO_ROOT / "temporal_risk" / "frozen_confirmatory_fusion_protocol.json"
SUPERSEDED = REPO_ROOT / "temporal_risk" / "frozen_fusion_protocol.json"
INVALID_CARVE = REPO_ROOT / "temporal_risk" / "fresh_validation_carve.json"
BLOCKED = REPO_ROOT / "outputs" / "fusion" / "fusion_blocked_contamination.json"
CARVE_WINDOWS = REPO_ROOT / "temporal_risk" / "carve_window_scores.csv"
CARVE_SPATIAL = REPO_ROOT / "outputs" / "fusion" / "carve_spatial_characterization.csv"

# Primary artifacts by FILENAME. A tool that opens any of these has left the
# development sandbox.
PRIMARY_ARTIFACTS = (
    "primary_window_scores.csv",
    "primary_aggregation_result.json",
    "paired_comparison.json",
    "clean_baseline_alignment.json",
    "predictions.csv",
)
CALIBRATION_TOOL = REPO_ROOT / "tools" / "calibrate_fusion_protocol.py"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@unittest.skipUnless(PROTOCOL.is_file(), "confirmatory protocol has not been frozen")
class FrozenConfirmatoryProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load(PROTOCOL)

    def test_declares_primary_untouched(self):
        self.assertFalse(self.protocol["primary_data_accessed"])
        self.assertFalse(self.protocol["primary_evaluation_performed"])
        self.assertIn("FROZEN", self.protocol["status"])

    def test_calibration_used_only_development_artifacts(self):
        development = self.protocol["development_data"]
        self.assertEqual(development["temporal_source"], "temporal_risk/carve_window_scores.csv")
        self.assertEqual(development["spatial_source"],
                         "outputs/fusion/carve_spatial_characterization.csv")
        for source in (development["temporal_source"], development["spatial_source"]):
            for artifact in PRIMARY_ARTIFACTS:
                self.assertNotIn(artifact, source)

    def test_development_set_is_declared_not_an_unbiased_test_set(self):
        status = self.protocol["development_data"]["status"]
        self.assertIn("DEVELOPMENT", status)
        self.assertIn("not an unbiased test set", status.lower())

    def test_candidate_set_is_closed_and_thresholds_are_numeric(self):
        candidates = self.protocol["candidates"]
        self.assertEqual(set(candidates),
                         {"B0_temporal_only", "B1_spatial_only", "F1_or_gate", "F2_rank_sum"})
        for name in ("B1_spatial_only", "F1_or_gate", "F2_rank_sum"):
            calibrated = candidates[name]["parameters_calibrated"]
            self.assertEqual(len(calibrated), 1, f"{name} should calibrate exactly one number")
            value = next(iter(calibrated.values()))
            self.assertIsInstance(value, float, name)

    def test_temporal_incumbent_is_frozen_at_0_14(self):
        b0 = self.protocol["candidates"]["B0_temporal_only"]
        self.assertEqual(b0["parameters_frozen"]["threshold"], 0.14)
        self.assertEqual(b0["parameters_calibrated"], [])

    def test_f2_weight_is_a_fixed_design_choice_not_a_fitted_one(self):
        frozen = self.protocol["candidates"]["F2_rank_sum"]["parameters_frozen"]
        self.assertEqual(frozen["weight"], 0.5)
        self.assertIn("NOT fitted", frozen["weight_status"])
        self.assertIn("never searched", frozen["weight_status"])
        import calibrate_fusion_protocol as calibrator
        self.assertEqual(calibrator.F2_WEIGHT, 0.5)

    def test_normalisation_reference_travels_inside_the_artifact(self):
        """This is what stops the confirmatory run normalising against primary."""
        normalisation = self.protocol["normalisation"]
        self.assertGreater(len(normalisation["temporal_reference_values"]), 100)
        self.assertGreater(len(normalisation["spatial_reference_values"]), 100)
        self.assertEqual(normalisation["temporal_reference_n"],
                         len(normalisation["temporal_reference_values"]))
        self.assertEqual(normalisation["spatial_reference_n"],
                         len(normalisation["spatial_reference_values"]))
        self.assertEqual(normalisation["temporal_reference_values"],
                         sorted(normalisation["temporal_reference_values"]))
        self.assertEqual(normalisation["spatial_reference_values"],
                         sorted(normalisation["spatial_reference_values"]))

    def test_normalisation_reference_size_matches_the_development_set(self):
        """A reference larger than the carve would mean foreign data leaked in."""
        development = self.protocol["development_data"]
        normalisation = self.protocol["normalisation"]
        self.assertEqual(normalisation["temporal_reference_n"], development["clips_used"])
        self.assertEqual(normalisation["spatial_reference_n"], development["spatially_defined"])
        self.assertLessEqual(development["clips_used"], 240)

    def test_operating_point_is_deterministic_with_no_rng(self):
        operating = self.protocol["operating_point"]
        self.assertIn("no RNG", operating["determinism"])
        self.assertIn("not relaxed", operating["if_budget_unreachable"])

    def test_missingness_bias_direction_is_recorded(self):
        development = self.protocol["development_data"]
        self.assertTrue(development["missingness_is_class_dependent"])
        self.assertIn("favours specificity", development["missingness_bias_direction"])

    def test_unresolved_confounder_is_recorded(self):
        feature = self.protocol["spatial_feature"]
        self.assertFalse(feature["camera_motion_compensated"])
        self.assertTrue(any("camera motion" in limit for limit in feature["known_limitations"]))
        self.assertEqual(feature["form"], "RATIO OF MEANS, not mean of ratios")

    def test_spatial_feature_choice_was_conservative_not_flattering(self):
        feature = self.protocol["spatial_feature"]
        self.assertEqual(feature["chosen_over"], "raw speed_pixels")
        self.assertIn("conservative", feature["choice_rationale"])

    def test_success_criteria_are_written_before_primary_is_seen(self):
        analysis = self.protocol["confirmatory_analysis"]
        self.assertEqual(analysis["primary_endpoint"]["significance_level"], 0.05)
        self.assertIn("McNemar", analysis["primary_endpoint"]["statistic"])
        self.assertEqual(set(analysis["interpretation"]),
                         {"A_improved", "B_directional", "C_no_improvement", "D_degraded"})
        prohibited = " ".join(analysis["prohibited_after_seeing_primary"]).lower()
        for forbidden in ("threshold", "weight", "feature", "candidate"):
            self.assertIn(forbidden, prohibited)

    def test_reproducibility_block_pins_every_moving_part(self):
        repro = self.protocol["reproducibility"]
        for field in ("bootstrap_seed", "bootstrap_resamples", "significance_level",
                      "temporal_checkpoint_sha256", "yolo_weights_sha256", "yolo_confidence",
                      "tracker", "temporal_regime", "temporal_aggregation",
                      "carve_windows_sha256", "carve_spatial_sha256", "tool_sha256"):
            self.assertIn(field, repro)
        self.assertEqual(len(repro["temporal_checkpoint_sha256"]), 64)
        self.assertEqual(repro["tracker"]["iou_threshold"], 0.3)
        self.assertEqual(repro["temporal_regime"]["windows_per_clip"], 17)

    def test_calibration_hashes_match_the_artifacts_on_disk(self):
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from checkpoint_identity import file_sha256
        repro = self.protocol["reproducibility"]
        self.assertEqual(repro["carve_windows_sha256"], file_sha256(CARVE_WINDOWS))
        self.assertEqual(repro["carve_spatial_sha256"], file_sha256(CARVE_SPATIAL))

    def test_causality_is_stated_honestly(self):
        causality = self.protocol["causality"]
        self.assertIn("NOT causal", causality["spatial_branch_feature"])
        self.assertEqual(causality["runtime_class"], "causal/offline")
        self.assertIn("NOT SUPPORTED", causality["real_time_claim"])

    def test_protocol_is_hashed(self):
        self.assertEqual(len(self.protocol["protocol_sha256"]), 64)


class CalibrationIsReproducibleTests(unittest.TestCase):
    @unittest.skipUnless(PROTOCOL.is_file() and CARVE_SPATIAL.is_file(), "artifacts absent")
    def test_calibration_reruns_to_identical_thresholds(self):
        """Determinism, checked by re-running rather than asserted."""
        import calibrate_fusion_protocol as calibrator
        rows = calibrator.load_development()
        b0 = {r["clip"]: r["temporal_max"] >= 0.14 for r in rows}
        budget = calibrator.confusion(rows, b0)["fp"]
        first = calibrator.calibrate(rows, lambda r: r["spatial"], budget)
        second = calibrator.calibrate(rows, lambda r: r["spatial"], budget)
        self.assertEqual(first["threshold"], second["threshold"])
        recorded = load(PROTOCOL)["candidates"]["B1_spatial_only"]["parameters_calibrated"]
        self.assertAlmostEqual(first["threshold"], recorded["theta_s"], places=12)

    @unittest.skipUnless(CARVE_SPATIAL.is_file(), "artifacts absent")
    def test_calibration_reads_only_carve_data(self):
        import calibrate_fusion_protocol as calibrator
        rows = calibrator.load_development()
        for row in rows:
            self.assertTrue(row["clip"].startswith("train/"),
                            f"{row['clip']} is not a carve clip")

    @unittest.skipUnless(CARVE_SPATIAL.is_file(), "artifacts absent")
    def test_calibration_refuses_a_non_carve_split(self):
        import calibrate_fusion_protocol as calibrator
        self.assertTrue(hasattr(calibrator, "CalibrationError"))
        source = CALIBRATION_TOOL.read_text(encoding="utf-8")
        self.assertIn('row["split"] != "carve"', source)


class PrimaryProtectionTests(unittest.TestCase):
    """No fusion tool may reach a primary artifact."""

    FUSION_TOOLS = ("calibrate_fusion_protocol.py", "build_fresh_validation_carve.py",
                    "score_fresh_carve.py", "evaluate_fusion.py")

    def test_no_fusion_tool_opens_a_primary_artifact(self):
        """AST-level: no reader call whose literal argument names a primary artifact."""
        import ast
        readers = {"open", "read_text", "read_bytes", "read_csv", "DictReader"}
        for name in self.FUSION_TOOLS:
            path = REPO_ROOT / "tools" / name
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if function not in readers:
                    continue
                for literal in ast.walk(node):
                    if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                        for artifact in PRIMARY_ARTIFACTS:
                            self.assertNotIn(artifact, literal.value,
                                             f"{name}:{node.lineno} reads {artifact}")

    def test_calibration_tool_declares_no_primary_path_constant(self):
        import calibrate_fusion_protocol as calibrator
        for attribute in dir(calibrator):
            value = getattr(calibrator, attribute)
            if isinstance(value, Path):
                text = str(value).replace("\\", "/")
                for artifact in PRIMARY_ARTIFACTS:
                    self.assertNotIn(artifact, text, f"{attribute} points at {artifact}")

    def test_no_primary_derived_artifact_was_generated(self):
        fusion_dir = REPO_ROOT / "outputs" / "fusion"
        if not fusion_dir.is_dir():
            return
        for path in fusion_dir.iterdir():
            self.assertNotIn("primary", path.name.lower(), f"{path.name} looks primary-derived")


class InvalidCarveStaysInvalidTests(unittest.TestCase):
    @unittest.skipUnless(PROTOCOL.is_file(), "protocol absent")
    def test_protocol_records_what_it_supersedes_and_why(self):
        supersedes = load(PROTOCOL)["supersedes"]
        self.assertEqual(supersedes["artifact"], "temporal_risk/frozen_fusion_protocol.json")
        self.assertIn("training", supersedes["reason"].lower())
        self.assertIn("0.244", supersedes["reason"])
        self.assertIn("never to be used for validation", supersedes["invalid_carve"])

    def test_superseded_evaluator_refuses_to_run(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "evaluate_fusion.py")],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
        )
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        message = result.stderr
        self.assertIn("REFUSED", message)
        self.assertIn("TRAINING data", message)

    @unittest.skipUnless(INVALID_CARVE.is_file(), "invalid carve manifest absent")
    def test_invalid_carve_evidence_is_retained(self):
        """The failed carve must stay documented, not deleted."""
        self.assertTrue(INVALID_CARVE.is_file())
        self.assertTrue(BLOCKED.is_file())
        record = load(BLOCKED)
        self.assertFalse(record["action_taken"]["fusion_result_claimed"])

    @unittest.skipUnless(PROTOCOL.is_file(), "protocol absent")
    def test_confirmatory_protocol_does_not_use_the_invalid_carve(self):
        protocol = load(PROTOCOL)
        self.assertNotIn("fresh_validation_carve",
                         protocol["development_data"]["temporal_source"])
        self.assertNotIn("fresh_carve", protocol["development_data"]["spatial_source"])


if __name__ == "__main__":
    unittest.main()
