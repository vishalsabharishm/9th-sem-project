"""Tests for the fresh validation carve and the pre-registered fusion protocol.

What these pin, in order of how badly a regression would hurt:

  * PROVENANCE. The fresh carve must be source-disjoint from the already-spent
    240-clip carve and from the held-out primary split, and must exclude the
    clips whose Kaggle-to-local filename mapping is UNKNOWN.
  * NO LEAKAGE. Fitting happens on the fit split only. The eval split, the old
    carve and the 394-clip primary split are never fitted on.
  * THE FROZEN PROTOCOL. The candidate set is closed, the F2 weight is fixed,
    the temporal rule stays at max >= 0.14, and the operating point is a
    matched false-positive budget rather than a tuned threshold.

Carve-construction tests read only the manifest and the dataset directory, so
they run anywhere the dataset is present. Evaluation tests skip cleanly until
the scoring step has produced its outputs.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from metric_intervals import source_video_id  # noqa: E402

MANIFEST = REPO_ROOT / "temporal_risk" / "fresh_validation_carve.json"
PROTOCOL = REPO_ROOT / "temporal_risk" / "frozen_fusion_protocol.json"
CARVE_CSV = REPO_ROOT / "temporal_risk" / "carve_window_scores.csv"
EVALUATION = REPO_ROOT / "outputs" / "fusion" / "fusion_evaluation.json"

# Artifact FILENAMES, not bare words: "paired_comparisons" is a legitimate
# JSON key in evaluate_fusion.py and must not trip a leakage guard.
FORBIDDEN = ("primary_window_scores.csv", "primary_aggregation_result.json",
             "paired_comparison.json", "primary_window_scores")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@unittest.skipUnless(MANIFEST.is_file(), "fresh carve manifest has not been built")
class FreshCarveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load(MANIFEST)
        cls.fit = cls.manifest["splits"]["fit"]["members"]
        cls.eval = cls.manifest["splits"]["eval"]["members"]

    def test_fit_and_eval_share_no_source_video(self):
        fit_sources = {m["source"] for m in self.fit}
        eval_sources = {m["source"] for m in self.eval}
        self.assertEqual(fit_sources & eval_sources, set())

    def test_no_clip_appears_twice(self):
        clips = [m["clip"] for m in self.fit + self.eval]
        self.assertEqual(len(clips), len(set(clips)))

    def test_source_disjoint_from_the_already_spent_carve(self):
        import csv
        with open(CARVE_CSV, newline="", encoding="utf-8") as handle:
            old = {row["clip"] for row in csv.DictReader(handle)}
        old_sources = {source_video_id(c) for c in old}
        fresh_sources = {m["source"] for m in self.fit + self.eval}
        self.assertEqual(fresh_sources & old_sources, set())
        self.assertEqual({m["clip"] for m in self.fit + self.eval} & old, set())

    def test_source_disjoint_from_the_held_out_primary_split(self):
        """Nothing from val/ may appear: it is reserved for confirmatory work."""
        for member in self.fit + self.eval:
            self.assertTrue(member["clip"].startswith("train/"), member["clip"])

    def test_unmappable_clips_are_excluded_by_whole_source_group(self):
        exclusion = self.manifest["pool"]["exclusion"]
        self.assertEqual(exclusion["non_ascii_clips"], 27)
        self.assertGreater(exclusion["excluded_source_groups"], 0)
        self.assertIn("UNKNOWN", exclusion["reason"])
        for member in self.fit + self.eval:
            name = member["clip"].rsplit("/", 1)[-1]
            self.assertTrue(all(ord(ch) < 128 for ch in name), member["clip"])

    def test_selection_consulted_no_model_output(self):
        self.assertFalse(self.manifest["fitting_occurred"])
        self.assertFalse(self.manifest["primary_data_accessed"])
        self.assertFalse(self.manifest["model_scores_consulted"])
        self.assertIn("filenames", self.manifest["selection_basis"])

    def test_manifest_is_hashed_and_seeded(self):
        self.assertEqual(len(self.manifest["manifest_sha256"]), 64)
        self.assertEqual(self.manifest["seed"], 20260906)

    def test_splits_are_reasonably_balanced(self):
        for split in ("fit", "eval"):
            labels = self.manifest["splits"][split]["labels"]
            fight, other = labels["Fight"], labels["NonFight"]
            self.assertLess(abs(fight - other) / (fight + other), 0.15, split)

    def test_construction_is_deterministic(self):
        import build_fresh_validation_carve as builder
        import csv
        with open(CARVE_CSV, newline="", encoding="utf-8") as handle:
            old = {row["clip"] for row in csv.DictReader(handle)}
        info = builder.build_pool(builder.DATASET_ROOT, old)
        first, _ = builder.assign_splits(info["pool"])
        second, _ = builder.assign_splits(info["pool"])
        self.assertEqual(first, second)
        self.assertEqual(
            sorted(c for c, _ in first["eval"]),
            sorted(m["clip"] for m in self.eval),
        )


@unittest.skipUnless(PROTOCOL.is_file(), "fusion protocol has not been frozen")
class FrozenProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load(PROTOCOL)

    def test_candidate_set_is_closed_and_exact(self):
        names = [c["name"] for c in self.protocol["candidates"]]
        self.assertEqual(names, ["B0_temporal_only", "B1_spatial_only",
                                 "F1_or_gate", "F2_rank_sum"])

    def test_temporal_incumbent_is_unchanged(self):
        b0 = next(c for c in self.protocol["candidates"] if c["name"] == "B0_temporal_only")
        self.assertIn("0.14", b0["rule"])
        self.assertEqual(b0["parameters_fitted"], "none")

    def test_f2_weight_is_fixed_not_fitted(self):
        f2 = next(c for c in self.protocol["candidates"] if c["name"] == "F2_rank_sum")
        self.assertIn("NOT fitted", f2["parameters_fitted"])
        import evaluate_fusion
        self.assertEqual(evaluate_fusion.F2_WEIGHT, 0.5)

    def test_spatial_score_choice_is_the_conservative_one(self):
        spatial = self.protocol["spatial_score"]
        self.assertEqual(spatial["chosen_over"], "raw speed_pixels")
        self.assertIn("conservative", spatial["rationale"])
        self.assertIn("camera motion", spatial["unresolved_confounder"])

    def test_missing_spatial_data_abstains_rather_than_imputing(self):
        rule = self.protocol["missing_data_rule"]
        self.assertIn("spatial-negative", rule["statement"])
        self.assertIn("imputing", rule["rationale"])

    def test_protocol_forbids_fitting_on_protected_splits(self):
        prohibited = " ".join(self.protocol["prohibited"])
        self.assertIn("eval split", prohibited)
        self.assertIn("primary split", prohibited)
        self.assertIn("240-clip carve", prohibited)

    def test_protocol_states_its_power_limit_before_seeing_data(self):
        self.assertIn("inconclusive", self.protocol["statistical_notes"]["power"])


class NoLeakageTests(unittest.TestCase):
    def test_new_tools_never_name_a_primary_artifact(self):
        for name in ("build_fresh_validation_carve.py", "score_fresh_carve.py",
                     "evaluate_fusion.py"):
            source = (REPO_ROOT / "tools" / name).read_text(encoding="utf-8")
            for artifact in FORBIDDEN:
                self.assertNotIn(artifact, source, f"{name} names {artifact}")

    def test_evaluation_tool_fits_only_on_the_fit_split(self):
        source = (REPO_ROOT / "tools" / "evaluate_fusion.py").read_text(encoding="utf-8")
        self.assertIn("calibrate(fit_rows", source)
        self.assertNotIn("calibrate(eval_rows", source)


BLOCKED_JSON = REPO_ROOT / "outputs" / "fusion" / "fusion_blocked_contamination.json"
FIT_SCORES = REPO_ROOT / "outputs" / "fusion" / "fresh_carve_fit_scores.json"
TRAINING_SUMMARY = REPO_ROOT / "temporal_risk" / "clean_baseline_training_summary.json"


class FreshCarveIsTrainingDataTests(unittest.TestCase):
    """The blocking finding, pinned so it cannot be quietly forgotten.

    The fresh carve is source-disjoint from the spent carve and from primary,
    and it is untouched by SELECTION -- all verified above. It is not untouched
    by TRAINING: the 240-clip carve is the R3D holdout, so the 1360-clip
    remainder the fresh carve was drawn from is the R3D training set. Every
    temporal score on it is a prediction on data the model was fitted to, which
    is why no fusion result was produced.
    """

    def test_training_summary_shows_the_carve_was_the_holdout(self):
        summary = load(TRAINING_SUMMARY)
        self.assertTrue(summary["summary"]["holdout_validation"])
        self.assertEqual(summary["summary"]["monitor_set"], "train-carved validation")
        self.assertAlmostEqual(summary["config"]["validation_fraction"], 0.15, places=6)

    @unittest.skipUnless(MANIFEST.is_file(), "manifest absent")
    def test_fresh_carve_is_drawn_from_the_training_remainder(self):
        """Every fresh-carve clip lives in train/ and outside the holdout carve."""
        import csv
        manifest = load(MANIFEST)
        with open(CARVE_CSV, newline="", encoding="utf-8") as handle:
            holdout = {row["clip"] for row in csv.DictReader(handle)}
        members = manifest["splits"]["fit"]["members"] + manifest["splits"]["eval"]["members"]
        for member in members:
            self.assertTrue(member["clip"].startswith("train/"), member["clip"])
            self.assertNotIn(member["clip"], holdout, member["clip"])

    @unittest.skipUnless(BLOCKED_JSON.is_file(), "blocking record absent")
    def test_blocking_record_states_no_fusion_result_was_produced(self):
        record = load(BLOCKED_JSON)
        self.assertFalse(record["action_taken"]["eval_split_read"])
        self.assertFalse(record["action_taken"]["fusion_evaluation_run"])
        self.assertFalse(record["action_taken"]["fusion_result_claimed"])
        self.assertFalse(record["primary_data_accessed"])
        self.assertIn("PARKED", record["decision"])

    @unittest.skipUnless(FIT_SCORES.is_file() and BLOCKED_JSON.is_file(), "fit scores absent")
    def test_memorization_gap_reproduces_from_the_stored_fit_scores(self):
        """Recompute the gap rather than trusting the number in the record."""
        rows = [r for r in load(FIT_SCORES)["clips"] if "error" not in r]
        tp = sum(1 for r in rows if r["label"] == "Fight" and r["temporal_max"] >= 0.14)
        fn = sum(1 for r in rows if r["label"] == "Fight" and r["temporal_max"] < 0.14)
        train_recall = tp / (tp + fn)
        holdout_recall = 85 / 119          # the frozen carve confusion, unchanged
        self.assertGreater(train_recall, 0.90, "training-split recall should be inflated")
        self.assertGreater(train_recall - holdout_recall, 0.15,
                           "the memorization gap should be large")
        record = load(BLOCKED_JSON)
        self.assertAlmostEqual(
            record["evidence"]["memorization_gap_recall"],
            train_recall - holdout_recall, places=9,
        )

    @unittest.skipUnless(FIT_SCORES.is_file(), "fit scores absent")
    def test_headroom_was_too_small_for_the_frozen_endpoint(self):
        """Why the experiment was degenerate, not merely disappointing."""
        rows = [r for r in load(FIT_SCORES)["clips"] if "error" not in r]
        missed = sum(1 for r in rows if r["label"] == "Fight" and r["temporal_max"] < 0.14)
        budget = sum(1 for r in rows if r["label"] == "NonFight" and r["temporal_max"] >= 0.14)
        self.assertLessEqual(missed, 5, "recoverable pool should be tiny on training data")
        self.assertLessEqual(budget, 5, "alarm budget should be tiny on training data")

    def test_no_fusion_evaluation_artifact_was_produced(self):
        self.assertFalse(
            EVALUATION.is_file(),
            "a fusion evaluation exists; it must not, the eval split is contaminated",
        )


@unittest.skipUnless(EVALUATION.is_file(), "fusion evaluation was deliberately not run")
class FusionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load(EVALUATION)

    def test_evaluation_declares_no_protected_data_was_used(self):
        self.assertFalse(self.report["primary_data_accessed"])
        self.assertFalse(self.report["existing_240_carve_used"])

    def test_all_four_candidates_were_evaluated(self):
        self.assertEqual(set(self.report["results"]),
                         {"B0_temporal_only", "B1_spatial_only", "F1_or_gate", "F2_rank_sum"})

    def test_every_candidate_decided_the_same_clips(self):
        total = self.report["eval_composition"]["clips"]
        for name, entry in self.report["results"].items():
            self.assertEqual(entry["tp"] + entry["fp"] + entry["tn"] + entry["fn"], total, name)

    def test_comparisons_are_paired_not_independent(self):
        for name, block in self.report["paired_comparisons"].items():
            for key in ("overall", "fight_detection", "nonfight_false_alarms"):
                self.assertIn("McNemar", block[key]["method"], f"{name}.{key}")

    def test_bootstraps_are_seeded_and_reported_at_both_levels(self):
        for name, block in self.report["paired_comparisons"].items():
            for key in ("accuracy_difference_clip_level", "accuracy_difference_source_level"):
                self.assertEqual(block[key]["seed"], 42, f"{name}.{key}")
        self.assertIn("source videos",
                      list(self.report["paired_comparisons"].values())[0]
                      ["accuracy_difference_source_level"]["method"])

    def test_missingness_is_reported_by_class(self):
        composition = self.report["eval_composition"]
        self.assertIn("spatially_undefined_by_class", composition)
        self.assertEqual(
            composition["spatially_undefined"],
            sum(composition["spatially_undefined_by_class"].values()),
        )


if __name__ == "__main__":
    unittest.main()
