"""Reproducibility tests for the paired clean-baseline vs sliding-window audit.

These pin the two things that would otherwise be unverifiable after the fact:

  * the exact statistics reported in the paired comparison, recomputed here
    from the raw committed artifacts rather than read back from the report; and
  * the alignment evidence, which is what licenses the paired test at all.

The McNemar p-values are checked against the definition -- a two-sided exact
binomial on the discordant pairs -- not against remembered values, following the
same rule as tests/test_metric_intervals.py.

Everything reads committed artifacts plus the two analysis outputs. No dataset,
no checkpoint, no scikit-learn and no GPU is required. The tests skip cleanly
when the analysis has not been generated, because the alignment step needs the
recovered checkpoint that is deliberately not in the repository.
"""

import csv
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "outputs" / "temporal_risk" / "paired_clean_vs_sliding_window"
PAIRED_JSON = ANALYSIS_DIR / "paired_comparison.json"
ALIGNMENT_JSON = ANALYSIS_DIR / "clean_baseline_alignment.json"
EVIDENCE_JSON = ANALYSIS_DIR / "alignment_evidence.json"
WINDOW_SCORES = REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"

CLIPS = 394
WINDOWS_PER_CLIP = 17
CLEAN_THRESHOLD = 0.16
SLIDING_THRESHOLD = 0.14


def exact_mcnemar(b, c):
    """Two-sided exact binomial p-value on discordant counts, from the definition."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@unittest.skipUnless(PAIRED_JSON.is_file(), "paired comparison has not been generated")
class PairedComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load(PAIRED_JSON)
        cls.records = cls.report["per_clip"]
        cls.windows = {}
        with open(WINDOW_SCORES, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cls.windows.setdefault(row["clip"], []).append(row)

    def test_every_primary_clip_is_represented_exactly_once(self):
        self.assertEqual(len(self.records), CLIPS)
        self.assertEqual(len({r["clip"] for r in self.records}), CLIPS)
        self.assertEqual(set(self.windows), {r["clip"] for r in self.records})

    def test_each_clip_has_the_full_window_regime(self):
        for clip, rows in self.windows.items():
            self.assertEqual(len(rows), WINDOWS_PER_CLIP, clip)

    def test_labels_agree_for_every_clip(self):
        alignment = {r["clip"]: r for r in load(ALIGNMENT_JSON)["per_clip"]}
        for record in self.records:
            row = alignment[record["clip"]]
            self.assertEqual(row["dataset_label"], row["recovered_label"], record["clip"])
            self.assertEqual(record["true_label"], row["recovered_label"], record["clip"])

    def test_clean_decisions_follow_the_documented_threshold(self):
        for record in self.records:
            expected = "Fight" if record["clean_probability"] >= CLEAN_THRESHOLD else "NonFight"
            self.assertEqual(expected, record["clean_predicted"], record["clip"])

    def test_sliding_decisions_follow_the_documented_frozen_rule(self):
        """max over the 17 window probabilities >= 0.14, recomputed from the CSV."""
        for record in self.records:
            peak = max(
                float(row["fight_probability"]) for row in self.windows[record["clip"]]
            )
            self.assertAlmostEqual(peak, record["sliding_max_probability"], places=12)
            expected = "Fight" if peak >= SLIDING_THRESHOLD else "NonFight"
            self.assertEqual(expected, record["sliding_predicted"], record["clip"])

    def _table(self, keep):
        cells = {"both": 0, "b": 0, "c": 0, "neither": 0}
        for record in self.records:
            if not keep(record):
                continue
            clean = record["clean_predicted"] == record["true_label"]
            sliding = record["sliding_predicted"] == record["true_label"]
            if clean and sliding:
                cells["both"] += 1
            elif clean:
                cells["b"] += 1
            elif sliding:
                cells["c"] += 1
            else:
                cells["neither"] += 1
        return cells

    def test_overall_discordant_counts_and_p_value(self):
        cells = self._table(lambda r: True)
        self.assertEqual((cells["b"], cells["c"]), (46, 29))
        recorded = self.report["paired_overall"]["mcnemar"]
        self.assertEqual((recorded["b"], recorded["c"]), (46, 29))
        self.assertAlmostEqual(
            recorded["p_value"], exact_mcnemar(cells["b"], cells["c"]), places=12
        )

    def test_fight_only_discordant_counts_and_p_value(self):
        cells = self._table(lambda r: r["true_label"] == "Fight")
        self.assertEqual((cells["b"], cells["c"]), (28, 6))
        recorded = self.report["paired_fight_only"]["mcnemar"]
        self.assertEqual((recorded["b"], recorded["c"]), (28, 6))
        self.assertAlmostEqual(
            recorded["p_value"], exact_mcnemar(cells["b"], cells["c"]), places=12
        )

    def test_nonfight_only_discordant_counts_and_p_value(self):
        cells = self._table(lambda r: r["true_label"] == "NonFight")
        self.assertEqual((cells["b"], cells["c"]), (18, 23))
        recorded = self.report["paired_nonfight_only"]["mcnemar"]
        self.assertEqual((recorded["b"], recorded["c"]), (18, 23))
        self.assertAlmostEqual(
            recorded["p_value"], exact_mcnemar(cells["b"], cells["c"]), places=12
        )

    def test_the_test_used_is_paired_not_independent_samples(self):
        for block in ("paired_overall", "paired_fight_only", "paired_nonfight_only"):
            method = self.report[block]["mcnemar"]["method"]
            self.assertIn("discordant", method)
            self.assertIn("McNemar", method)

    def test_bootstrap_is_seeded_and_paired_at_both_levels(self):
        difference = self.report["paired_overall"][
            "accuracy_difference_sliding_minus_clean"
        ]
        self.assertEqual(set(difference), {"clip_level", "source_video_level"})
        for level, block in difference.items():
            self.assertEqual(block["seed"], 42, level)
            self.assertEqual(block["resamples"], 10000, level)
            self.assertIn("paired", block["method"], level)
            self.assertLess(block["ci_low"], block["ci_high"], level)
        self.assertIn("cluster", difference["source_video_level"]["method"])

    def test_censoring_is_not_imputed_as_a_detection(self):
        early = self.report["early_detection"]
        censored = [
            r for r in self.records
            if r["truth"] == 1 and r["first_alarm_frame"] is None
        ]
        self.assertEqual(len(censored), early["not_detected_right_censored"])
        for record in censored:
            self.assertIsNone(record["first_alarm_seconds"])
        window = early["censoring_aware_over_all_fight_clips"]
        self.assertTrue(window["quantiles_strictly_below_censoring_time"])

    def test_regimes_are_documented_as_different_random_variables(self):
        regimes = self.report["regimes"]
        self.assertEqual(regimes["clean_baseline"]["threshold"], CLEAN_THRESHOLD)
        self.assertIn("0.14", regimes["sliding_window"]["rule"])
        self.assertIn("max", regimes["sliding_window"]["rule"])
        explanation = regimes["why_not_merely_a_threshold_comparison"]
        self.assertIn("different random variables", explanation)
        self.assertIn("maximum", explanation)

    def test_analysis_used_the_clean_baseline_checkpoint_not_experiment_one(self):
        provenance = self.report["checkpoint_provenance"]
        self.assertTrue(provenance["is_clean_baseline_notebook"])
        self.assertFalse(provenance["is_experiment1_notebook"])
        self.assertIn("notebook7bb9a86555", provenance["window_scoring_checkpoint"])
        self.assertNotIn("notebook5d98537e15", provenance["window_scoring_checkpoint"])

    def test_disagreement_table_matches_the_report(self):
        with open(ANALYSIS_DIR / "disagreement_table.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), self.report["disagreements"]["total"])
        expected = [
            r for r in self.records if r["clean_correct"] != r["sliding_correct"]
        ]
        self.assertEqual(len(rows), len(expected))


@unittest.skipUnless(EVIDENCE_JSON.is_file(), "alignment evidence has not been generated")
class AlignmentEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = load(EVIDENCE_JSON)

    def test_alignment_is_supported_by_the_local_rerun(self):
        alignment = load(ALIGNMENT_JSON)
        self.assertTrue(alignment["alignment_supported"])
        self.assertEqual(alignment["label_mismatches"], 0)
        self.assertEqual(alignment["rows_over_tolerance"], 0)

    def test_permutation_test_is_seeded_and_beats_every_alternative(self):
        permutation = self.evidence["permutation_test"]
        self.assertEqual(permutation["seed"], 42)
        self.assertEqual(permutation["permutations"], 20000)
        self.assertEqual(permutation["permutations_at_least_as_good_as_observed"], 0)
        self.assertLess(permutation["observed_mean_abs_difference"], 1e-5)
        self.assertGreater(permutation["permuted_min"], 1e-2)
        self.assertGreater(permutation["times_better_than_a_typical_wrong_mapping"], 1000)

    def test_residual_ties_cannot_move_a_single_cell(self):
        ties = self.evidence["tie_analysis"]
        self.assertEqual(ties["adjacent_near_tied_pairs"], 99)
        self.assertEqual(ties["adjacent_cross_label_inadmissible"], 8)
        self.assertEqual(ties["adjacent_same_label_admissible"], 91)
        self.assertEqual(ties["adjacent_same_label_and_same_clean_decision"], 91)
        self.assertGreater(ties["admissible_swaps_within_clusters"], 91)
        self.assertEqual(ties["admissible_swaps_that_would_change_a_cell"], 0)
        self.assertTrue(ties["tables_invariant_to_residual_ambiguity"])

    def test_evidence_regenerates_byte_identically(self):
        """The whole point of the tool: the p-value must not drift between runs."""
        before = EVIDENCE_JSON.read_bytes()
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "alignment_evidence.py")],
            capture_output=True,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertEqual(before, EVIDENCE_JSON.read_bytes())


if __name__ == "__main__":
    unittest.main()
