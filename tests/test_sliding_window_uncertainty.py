"""Tests for uncertainty quantification of the sliding-window system result.

This is the result the deployed system actually produces (accuracy 0.7893), as
opposed to Experiment 1's superseded whole-clip 0.8604 that Step 4 covered.

Everything reads committed artifacts. No checkpoint, no dataset, no
scikit-learn.

Two properties are pinned hard because a silent change to either would
invalidate every number in the paper:

- the confusion matrix TP=160 FP=43 TN=151 FN=40 must be recomputable exactly
  from the committed window scores plus the frozen rule;
- the 40 Fight clips that never alarm must never be imputed as 4.8-second
  alarms.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from metric_intervals import cluster_ids, wilson_interval  # noqa: E402
from sliding_window_uncertainty import (  # noqa: E402
    UncertaintyError,
    alarm_quantiles,
    assert_reproduces_recorded,
    bootstrap_alarm_median,
    clean_baseline_intervals,
    compute,
    confusion_from,
    f1_at_rule,
    load_sliding_window,
    paired_comparison_status,
)

# Recorded in temporal_risk/primary_aggregation_result.json.
TP, FP, TN, FN = 160, 43, 151, 40
N = 394
PREDICTIONS_SHA256 = "76b7b078f533f927a03e6e5b136cc06aaf03b45d7381bc1c7b810c08d32cf799"

# A small number of resamples keeps the suite fast; reproducibility is what is
# being tested, not the exact width.
FAST_RESAMPLES = 400


class ReproductionTests(unittest.TestCase):
    """The point estimate must be recomputable before any interval is claimed."""

    @classmethod
    def setUpClass(cls):
        cls.clips, cls.truth, cls.score, cls.fired = load_sliding_window()

    def test_394_clips_with_the_recorded_class_balance(self):
        self.assertEqual(self.truth.size, N)
        self.assertEqual(int((self.truth == 1).sum()), 200)
        self.assertEqual(int((self.truth == 0).sum()), 194)

    def test_confusion_matrix_is_exactly_the_recorded_one(self):
        confusion = confusion_from(self.truth, self.fired)
        self.assertEqual(confusion["true_positive"], TP)
        self.assertEqual(confusion["false_positive"], FP)
        self.assertEqual(confusion["true_negative"], TN)
        self.assertEqual(confusion["false_negative"], FN)

    def test_accuracy_point_estimate_is_preserved(self):
        confusion = confusion_from(self.truth, self.fired)
        accuracy = (confusion["true_positive"] + confusion["true_negative"]) / N
        self.assertAlmostEqual(accuracy, 0.7893401015228426, places=13)
        self.assertEqual(round(accuracy, 4), 0.7893)

    def test_reproduction_check_passes_against_the_committed_record(self):
        report = assert_reproduces_recorded(confusion_from(self.truth, self.fired))
        self.assertTrue(report["exact_match"])
        self.assertEqual(report["accuracy_difference"], 0.0)
        self.assertFalse(report["checkpoint_required"])

    def test_reproduction_check_refuses_a_mismatched_confusion(self):
        """Intervals must never be reported around an unreproducible estimate."""
        with self.assertRaises(UncertaintyError):
            assert_reproduces_recorded(
                {"true_positive": 1, "false_positive": 1,
                 "true_negative": 1, "false_negative": 1}
            )

    def test_clip_score_is_the_max_over_windows(self):
        from temporal_event_adapter import PrecomputedWindowScoreSource

        source = PrecomputedWindowScoreSource(
            REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"
        )
        for position, clip in enumerate(self.clips[:25]):
            expected = max(w.fight_probability for w in source.get(clip))
            self.assertAlmostEqual(self.score[position], expected, places=12)

    def test_the_fired_decision_comes_from_the_frozen_rule(self):
        """max >= 0.14 for the currently frozen rule."""
        self.assertTrue(np.array_equal(self.fired, self.score >= 0.14))


class SlidingWindowIntervalTests(unittest.TestCase):
    """The reported intervals, pinned."""

    @classmethod
    def setUpClass(cls):
        cls.report = compute(resamples=FAST_RESAMPLES)

    def test_wilson_accuracy_interval(self):
        row = self.report["wilson_intervals"]["accuracy"]
        self.assertEqual((row["successes"], row["total"]), (TP + TN, N))
        self.assertAlmostEqual(row["ci_low"], 0.7464, places=4)
        self.assertAlmostEqual(row["ci_high"], 0.8267, places=4)

    def test_wilson_recall_interval(self):
        row = self.report["wilson_intervals"]["recall_sensitivity"]
        self.assertEqual((row["successes"], row["total"]), (160, 200))
        self.assertAlmostEqual(row["ci_low"], 0.7391, places=4)
        self.assertAlmostEqual(row["ci_high"], 0.8495, places=4)

    def test_wilson_specificity_interval(self):
        row = self.report["wilson_intervals"]["specificity"]
        self.assertEqual((row["successes"], row["total"]), (151, 194))
        self.assertAlmostEqual(row["ci_low"], 0.7148, places=4)
        self.assertAlmostEqual(row["ci_high"], 0.8311, places=4)

    def test_precision_and_npv_are_flagged_conditional(self):
        for name in ("precision_ppv", "npv"):
            self.assertTrue(self.report["wilson_intervals"][name]["denominator_is_random"])

    def test_precision_denominator_is_the_number_of_alarms(self):
        row = self.report["wilson_intervals"]["precision_ppv"]
        self.assertEqual((row["successes"], row["total"]), (160, TP + FP))

    def test_intervals_agree_with_the_shared_wilson_implementation(self):
        """No bespoke arithmetic: Step 4's machinery is reused unchanged."""
        low, high = wilson_interval(TP + TN, N)
        row = self.report["wilson_intervals"]["accuracy"]
        self.assertAlmostEqual(row["ci_low"], low, places=15)
        self.assertAlmostEqual(row["ci_high"], high, places=15)

    def test_f1_gets_a_bootstrap_not_a_wilson_interval(self):
        self.assertNotIn("f1", self.report["wilson_intervals"])
        self.assertIn("f1", self.report["bootstrap_intervals_stratified"])

    def test_rank_metrics_are_reported_with_a_non_comparability_note(self):
        for name in ("roc_auc", "pr_auc"):
            self.assertIsNotNone(
                self.report["bootstrap_intervals_stratified"][name]["ci_low"]
            )
        self.assertIn("NOT comparable", self.report["rank_metric_note"])

    def test_every_point_estimate_lies_inside_its_interval(self):
        for group in ("bootstrap_intervals_stratified", "bootstrap_intervals_clustered"):
            for name, row in self.report[group].items():
                if row["ci_low"] is None:
                    continue
                with self.subTest(group=group, metric=name):
                    self.assertLessEqual(row["ci_low"], row["point"])
                    self.assertLessEqual(row["point"], row["ci_high"])


class BootstrapReproducibilityTests(unittest.TestCase):
    def test_same_seed_gives_identical_intervals(self):
        first = compute(resamples=FAST_RESAMPLES, seed=99)
        second = compute(resamples=FAST_RESAMPLES, seed=99)
        for name in ("accuracy", "f1", "roc_auc", "pr_auc"):
            with self.subTest(metric=name):
                self.assertEqual(
                    first["bootstrap_intervals_stratified"][name]["ci_low"],
                    second["bootstrap_intervals_stratified"][name]["ci_low"],
                )
                self.assertEqual(
                    first["bootstrap_intervals_stratified"][name]["ci_high"],
                    second["bootstrap_intervals_stratified"][name]["ci_high"],
                )

    def test_different_seeds_give_different_intervals(self):
        first = compute(resamples=FAST_RESAMPLES, seed=1)
        second = compute(resamples=FAST_RESAMPLES, seed=2)
        self.assertNotEqual(
            first["bootstrap_intervals_stratified"]["roc_auc"]["ci_low"],
            second["bootstrap_intervals_stratified"]["roc_auc"]["ci_low"],
        )

    def test_seed_and_resample_count_are_recorded(self):
        report = compute(resamples=FAST_RESAMPLES, seed=7)
        row = report["bootstrap_intervals_stratified"]["f1"]
        self.assertEqual(row["seed"], 7)
        self.assertEqual(row["resamples"], FAST_RESAMPLES)


class SourceGroupingTests(unittest.TestCase):
    """Grouping is applied directly here -- the clips are named."""

    @classmethod
    def setUpClass(cls):
        cls.report = compute(resamples=FAST_RESAMPLES)

    def test_grouping_is_available_and_reduces_the_effective_sample(self):
        cluster = self.report["cluster_structure"]
        self.assertEqual(cluster["clips"], N)
        self.assertEqual(cluster["clusters"], 174)
        self.assertLess(cluster["clusters"], cluster["clips"])

    def test_grouping_is_labelled_inferred_not_proven(self):
        cluster = self.report["cluster_structure"]
        self.assertIn("inferred", cluster["grouping"])
        self.assertIn("not confirmed", cluster["caveat"])

    def test_clustered_intervals_are_wider_than_clip_level(self):
        """The whole point: clip-level intervals are anti-conservative."""
        for name in ("accuracy", "f1", "roc_auc", "pr_auc"):
            strat = self.report["bootstrap_intervals_stratified"][name]
            clust = self.report["bootstrap_intervals_clustered"][name]
            with self.subTest(metric=name):
                self.assertGreater(
                    clust["ci_high"] - clust["ci_low"],
                    strat["ci_high"] - strat["ci_low"],
                )

    def test_clustering_uses_clip_names_directly(self):
        """No positional-alignment inference is needed for this regime."""
        clips, _, _, _ = load_sliding_window()
        ids = cluster_ids(clips)
        self.assertEqual(len(ids), N)
        self.assertEqual(len(set(ids.tolist())), 174)


class CensoringTests(unittest.TestCase):
    """The 40 non-alarming Fight clips must never become 4.8-second alarms."""

    @classmethod
    def setUpClass(cls):
        cls.report = compute(resamples=FAST_RESAMPLES)
        cls.latency = cls.report["time_to_alarm"]

    def test_detected_and_censored_counts(self):
        self.assertEqual(self.latency["n_fight_clips"], 200)
        self.assertEqual(self.latency["n_detected"], 160)
        self.assertEqual(self.latency["n_right_censored"], 40)
        self.assertAlmostEqual(self.latency["detected_fraction"], 0.8, places=12)

    def test_conditional_and_unconditional_medians_are_different_quantities(self):
        conditional = self.latency["conditional_on_detection"]["median_seconds"]
        unconditional = self.latency["unconditional_over_all_fight_clips"]["median_seconds"]
        self.assertAlmostEqual(conditional, 0.5333, places=4)
        self.assertAlmostEqual(unconditional, 0.8, places=4)
        self.assertNotEqual(conditional, unconditional)
        self.assertLess(conditional, unconditional)

    def test_unconditional_p75_is_much_larger_than_the_conditional_one(self):
        conditional = self.latency["conditional_on_detection"]["p75_seconds"]
        unconditional = self.latency["unconditional_over_all_fight_clips"]["q75_seconds"]
        self.assertAlmostEqual(conditional, 1.6, places=4)
        self.assertAlmostEqual(unconditional, 3.7333, places=4)
        self.assertGreater(unconditional, 2 * conditional)

    def test_quantiles_above_the_detected_fraction_are_not_reached(self):
        uncond = self.latency["unconditional_over_all_fight_clips"]
        self.assertIsNotNone(uncond["q80_seconds"])
        quantiles = alarm_quantiles([0.5] * 3, 10)  # only 30% ever alarm
        self.assertIsNone(quantiles["unconditional_over_all_fight_clips"]["median_seconds"])

    def test_censored_clips_are_not_imputed_at_the_censoring_time(self):
        """A synthetic case where imputation would visibly change the answer."""
        detected = [1.0, 1.0, 1.0]
        imputed_median = float(np.median(detected + [4.8, 4.8, 4.8, 4.8]))
        quantiles = alarm_quantiles(detected, n_fight=7)
        honest = quantiles["unconditional_over_all_fight_clips"]["median_seconds"]
        self.assertIsNone(honest, "3 of 7 detected cannot reach a median")
        self.assertAlmostEqual(imputed_median, 4.8, places=6)

    def test_censoring_is_described_as_administrative(self):
        self.assertIn("administrative", self.latency["censoring"])
        self.assertIn("4.8", self.latency["censoring"])

    def test_a_conflation_warning_is_carried_in_the_output(self):
        self.assertIn("different quantities", self.latency["do_not_conflate"])

    def test_median_bootstrap_is_reproducible(self):
        first = bootstrap_alarm_median([0.5, 0.8, 1.0, 1.2], 6, 2, 0.95, 200, 42)
        second = bootstrap_alarm_median([0.5, 0.8, 1.0, 1.2], 6, 2, 0.95, 200, 42)
        self.assertEqual(first["ci_low"], second["ci_low"])
        self.assertEqual(first["ci_high"], second["ci_high"])

    def test_median_bootstrap_discards_rather_than_imputes_degenerate_resamples(self):
        result = bootstrap_alarm_median([0.5], 10, 9, 0.95, 100, 42)
        self.assertEqual(result["usable_resamples"] + result["discarded_resamples"], 100)
        self.assertGreater(result["discarded_resamples"], 0)


class CleanBaselineTests(unittest.TestCase):
    """Wilson from counts is legitimate; a bootstrap without per-clip data is not."""

    @classmethod
    def setUpClass(cls):
        cls.clean = clean_baseline_intervals(0.95)

    def test_point_estimate_is_preserved(self):
        accuracy = self.clean["point_estimates"]["accuracy"]
        self.assertAlmostEqual(accuracy, 0.8324873096446701, places=13)
        self.assertEqual(round(accuracy, 4), 0.8325)

    def test_confusion_matrix_is_the_recorded_one(self):
        self.assertEqual(
            self.clean["confusion_matrix"],
            {"true_positive": 182, "false_positive": 48,
             "true_negative": 146, "false_negative": 18},
        )

    def test_wilson_intervals_are_available_from_counts_alone(self):
        row = self.clean["wilson_intervals"]["accuracy"]
        self.assertEqual((row["successes"], row["total"]), (328, 394))
        self.assertAlmostEqual(row["ci_low"], 0.7924, places=4)
        self.assertAlmostEqual(row["ci_high"], 0.8661, places=4)

    def test_bootstrap_and_clustering_are_declared_unavailable(self):
        unavailable = self.clean["not_available"]
        for key in ("per_clip_predictions", "bootstrap_intervals", "f1_interval",
                    "roc_auc_and_pr_auc_intervals", "source_video_clustering"):
            self.assertIn(key, unavailable)
            self.assertTrue(unavailable[key])

    def test_no_f1_or_auc_interval_is_fabricated(self):
        self.assertNotIn("bootstrap_intervals", self.clean)
        self.assertNotIn("f1", self.clean["wilson_intervals"])


class PairedComparisonTests(unittest.TestCase):
    """The comparison must be refused, with the reason recorded."""

    @classmethod
    def setUpClass(cls):
        cls.status = paired_comparison_status()

    def test_formal_test_is_refused(self):
        self.assertFalse(self.status["formal_test_possible"])

    def test_the_blocking_reason_names_the_missing_artifact(self):
        self.assertIn("per-clip", self.status["blocking_reason"])
        self.assertIn("McNemar", self.status["blocking_reason"])

    def test_unpaired_tests_are_explicitly_refused_too(self):
        reason = self.status["why_unpaired_tests_are_also_refused"]
        self.assertIn("not independent", reason)

    def test_the_confounded_alternative_is_documented_as_rejected(self):
        rejected = self.status["alternative_pairing_considered_and_rejected"]
        self.assertIn("confound", rejected)
        self.assertIn("notebook5d98537e15", rejected)

    def test_what_would_unblock_it_is_stated(self):
        self.assertIn("predictions.csv", self.status["what_would_unblock_it"])

    def test_the_two_results_share_a_checkpoint_and_clips(self):
        self.assertTrue(self.status["same_checkpoint"])
        self.assertTrue(self.status["same_clips"])


class ArtifactIntegrityTests(unittest.TestCase):
    def test_experiment1_predictions_csv_is_unmodified(self):
        import hashlib

        path = REPO_ROOT / "outputs" / "temporal_violence" / "evaluation" / "predictions.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, PREDICTIONS_SHA256)

    def test_the_frozen_rule_is_unchanged(self):
        frozen = json.loads(
            (REPO_ROOT / "temporal_risk" / "frozen_aggregation.json").read_text()
        )["selected"]
        self.assertEqual(frozen["rule"], "max")
        self.assertEqual(frozen["params"], {"threshold": 0.14})

    def test_the_recorded_aggregation_result_is_unchanged(self):
        metrics = json.loads(
            (REPO_ROOT / "temporal_risk" / "primary_aggregation_result.json").read_text()
        )["primary_split"]["metrics"]
        self.assertEqual(
            (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]), (TP, FP, TN, FN)
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.7893401015228426, places=13)


if __name__ == "__main__":
    unittest.main()
