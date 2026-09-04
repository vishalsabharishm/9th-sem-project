"""Tests for confidence intervals on the temporal evaluation metrics.

The Wilson tests do not compare against remembered table values -- during
development two such values turned out to be continuity-corrected figures for a
*different* estimator. They check the interval against its own definition
instead: the Wilson bounds are exactly the two roots of

    (p_hat - p) / sqrt(p(1-p)/n)  =  +/- z

which is self-contained and cannot be got wrong by misremembering a table.

Everything here reads committed artifacts only. No dataset, no checkpoint, no
scikit-learn required.
"""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from metric_intervals import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SEED,
    METHOD_BOOTSTRAP_CLUSTER,
    METHOD_BOOTSTRAP_STRATIFIED,
    METHOD_WILSON,
    NO_INTERVAL_CLAIMED,
    IntervalError,
    bootstrap_metric_interval,
    cluster_ids,
    cluster_summary,
    confusion_intervals,
    proportion_interval,
    source_video_id,
    wilson_interval,
    z_for,
)

SKLEARN_PRESENT = importlib.util.find_spec("sklearn") is not None

# The historical Experiment-1 confusion matrix, from
# docs/temporal_baseline_results.json. Not recomputed here; used as fixed input.
TP, FP, TN, FN = 164, 19, 175, 36
N = TP + FP + TN + FN  # 394


class CriticalValueTests(unittest.TestCase):
    def test_two_sided_95_percent_z(self):
        self.assertAlmostEqual(z_for(0.95), 1.959963984540054, places=12)

    def test_common_levels(self):
        self.assertAlmostEqual(z_for(0.90), 1.6448536269514722, places=12)
        self.assertAlmostEqual(z_for(0.99), 2.5758293035489004, places=12)

    def test_invalid_confidence_is_rejected(self):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with self.subTest(confidence=bad), self.assertRaises(IntervalError):
                z_for(bad)


class WilsonDefinitionTests(unittest.TestCase):
    """The bounds must satisfy the score equation they are defined by."""

    CASES = [(1, 20), (15, 20), (5, 10), (3, 7), (339, 394), (164, 200),
             (175, 194), (164, 183), (200, 394), (1, 1000)]

    def test_each_bound_solves_the_score_equation(self):
        z = z_for(0.95)
        for successes, total in self.CASES:
            low, high = wilson_interval(successes, total)
            p_hat = successes / total
            with self.subTest(successes=successes, total=total):
                for bound, expected in ((low, +z), (high, -z)):
                    if 0.0 < bound < 1.0:
                        statistic = (p_hat - bound) / math.sqrt(
                            bound * (1 - bound) / total
                        )
                        self.assertAlmostEqual(statistic, expected, places=9)

    def test_point_estimate_lies_inside_the_interval(self):
        for successes, total in self.CASES:
            low, high = wilson_interval(successes, total)
            with self.subTest(successes=successes, total=total):
                self.assertLessEqual(low, successes / total)
                self.assertLessEqual(successes / total, high)

    def test_interval_stays_within_zero_and_one(self):
        for successes, total in self.CASES + [(0, 5), (5, 5)]:
            low, high = wilson_interval(successes, total)
            with self.subTest(successes=successes, total=total):
                self.assertGreaterEqual(low, 0.0)
                self.assertLessEqual(high, 1.0)


class WilsonEdgeCaseTests(unittest.TestCase):
    def test_zero_successes_gives_a_lower_bound_of_exactly_zero(self):
        low, high = wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertLess(high, 1.0)

    def test_all_successes_gives_an_upper_bound_of_exactly_one(self):
        low, high = wilson_interval(10, 10)
        self.assertEqual(high, 1.0)
        self.assertGreater(low, 0.0)
        self.assertLess(low, 1.0)

    def test_extremes_do_not_collapse_to_zero_width(self):
        """The reason Wilson is used instead of Wald: Wald gives [0,0] here."""
        for successes, total in ((0, 30), (30, 30)):
            low, high = wilson_interval(successes, total)
            with self.subTest(successes=successes):
                self.assertGreater(high - low, 0.05)

    def test_single_observation_is_allowed_and_very_wide(self):
        low, high = wilson_interval(1, 1)
        self.assertEqual(high, 1.0)
        self.assertLess(low, 0.3)

    def test_zero_total_is_rejected(self):
        with self.assertRaises(IntervalError):
            wilson_interval(0, 0)

    def test_successes_outside_the_denominator_are_rejected(self):
        with self.assertRaises(IntervalError):
            wilson_interval(11, 10)
        with self.assertRaises(IntervalError):
            wilson_interval(-1, 10)

    def test_half_is_symmetric_about_one_half(self):
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual((low + high) / 2, 0.5, places=12)

    def test_complement_symmetry(self):
        low_a, high_a = wilson_interval(30, 100)
        low_b, high_b = wilson_interval(70, 100)
        self.assertAlmostEqual(low_a, 1 - high_b, places=15)
        self.assertAlmostEqual(high_a, 1 - low_b, places=15)

    def test_width_shrinks_as_n_grows(self):
        narrow = wilson_interval(500, 1000)
        wide = wilson_interval(50, 100)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_width_grows_with_confidence(self):
        low90, high90 = wilson_interval(50, 100, 0.90)
        low99, high99 = wilson_interval(50, 100, 0.99)
        self.assertLess(high90 - low90, high99 - low99)


class HistoricalValueTests(unittest.TestCase):
    """The intervals for the actual 394-sample Experiment-1 result."""

    @classmethod
    def setUpClass(cls):
        cls.intervals = confusion_intervals(TP, FP, TN, FN)

    def test_sample_size_is_394(self):
        self.assertEqual(N, 394)
        self.assertEqual(self.intervals["accuracy"].total, 394)

    def test_accuracy_interval(self):
        interval = self.intervals["accuracy"]
        self.assertEqual(interval.successes, 339)
        self.assertAlmostEqual(interval.point, 0.8604060913705583, places=12)
        self.assertAlmostEqual(interval.low, 0.8226939769, places=9)
        self.assertAlmostEqual(interval.high, 0.8911582215, places=9)

    def test_recall_interval_uses_the_fight_clips_as_denominator(self):
        interval = self.intervals["recall_sensitivity"]
        self.assertEqual((interval.successes, interval.total), (164, 200))
        self.assertAlmostEqual(interval.point, 0.82, places=12)
        self.assertAlmostEqual(interval.low, 0.7608852497, places=9)
        self.assertAlmostEqual(interval.high, 0.8670537414, places=9)
        self.assertFalse(interval.denominator_is_random)

    def test_specificity_interval_uses_the_nonfight_clips(self):
        interval = self.intervals["specificity"]
        self.assertEqual((interval.successes, interval.total), (175, 194))
        self.assertAlmostEqual(interval.low, 0.8521082842, places=9)
        self.assertAlmostEqual(interval.high, 0.9364018740, places=9)
        self.assertFalse(interval.denominator_is_random)

    def test_precision_is_marked_as_having_a_random_denominator(self):
        interval = self.intervals["precision_ppv"]
        self.assertEqual((interval.successes, interval.total), (164, 183))
        self.assertAlmostEqual(interval.point, 0.8961748633879781, places=12)
        self.assertAlmostEqual(interval.low, 0.8435395015, places=9)
        self.assertAlmostEqual(interval.high, 0.9325195216, places=9)
        self.assertTrue(interval.denominator_is_random)
        self.assertIn("not fixed by the evaluation design", interval.note)

    def test_npv_is_also_marked_random(self):
        self.assertTrue(self.intervals["npv"].denominator_is_random)

    def test_prevalence_reflects_the_split_balance(self):
        interval = self.intervals["prevalence_fight"]
        self.assertEqual((interval.successes, interval.total), (200, 394))

    def test_no_wilson_interval_is_produced_for_f1_or_the_aucs(self):
        """The central scope rule of Step 4A."""
        for name in ("f1", "roc_auc", "pr_auc", "auc"):
            self.assertNotIn(name, self.intervals)

    def test_every_produced_interval_uses_the_wilson_method(self):
        for interval in self.intervals.values():
            self.assertEqual(interval.method, METHOD_WILSON)

    def test_reasons_are_recorded_for_the_metrics_without_wilson_intervals(self):
        """Each exclusion must carry a self-contained statistical reason."""
        self.assertEqual(set(NO_INTERVAL_CLAIMED), {"f1", "roc_auc", "pr_auc"})
        for name, reason in NO_INTERVAL_CLAIMED.items():
            with self.subTest(metric=name):
                lowered = reason.lower()
                self.assertGreater(len(reason), 80, "reason is too terse to be useful")
                self.assertIn("not a binomial proportion", lowered)
                self.assertTrue(
                    "rank statistic" in lowered or "harmonic mean" in lowered,
                    f"{name} does not say what it is instead",
                )

    def test_empty_confusion_matrix_is_rejected(self):
        with self.assertRaises(IntervalError):
            confusion_intervals(0, 0, 0, 0)


class ClusterStructureTests(unittest.TestCase):
    """The independence assumption is violated, and that must stay visible."""

    def test_source_id_strips_the_clip_index(self):
        self.assertEqual(source_video_id("val/Val_Fight/0Ow4cotKOuw_3.avi"), "0Ow4cotKOuw")
        self.assertEqual(source_video_id("val/Val_NonFight/39BFeYnbu-I_0.avi"), "39BFeYnbu-I")

    def test_source_id_ignores_the_class_directory(self):
        """25 source videos contribute clips to both classes; they are one cluster."""
        self.assertEqual(
            source_video_id("val/Val_Fight/39BFeYnbu-I_0.avi"),
            source_video_id("val/Val_NonFight/39BFeYnbu-I_2.avi"),
        )

    def test_a_stem_without_an_index_is_its_own_cluster(self):
        self.assertEqual(source_video_id("val/Val_Fight/oddname.avi"), "oddname")

    def test_cluster_ids_group_shared_sources_together(self):
        clips = [
            "val/Val_Fight/a_0.avi", "val/Val_Fight/a_1.avi",
            "val/Val_NonFight/a_0.avi", "val/Val_Fight/b_0.avi",
        ]
        ids = cluster_ids(clips)
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[0], ids[2])
        self.assertNotEqual(ids[0], ids[3])

    def test_summary_reports_the_dependence(self):
        clips = ["v/a_0.avi", "v/a_1.avi", "v/a_2.avi", "v/b_0.avi"]
        summary = cluster_summary(clips)
        self.assertEqual(summary["clips"], 4)
        self.assertEqual(summary["clusters"], 2)
        self.assertEqual(summary["max_cluster_size"], 3)
        self.assertEqual(summary["singleton_clusters"], 1)
        self.assertIn("inferred", summary["grouping"])
        self.assertIn("not confirmed", summary["caveat"])

    def test_real_primary_split_is_not_independent(self):
        """Recorded so a future change cannot quietly restore the assumption."""
        import csv

        path = REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"
        with open(path, newline="", encoding="utf-8", errors="surrogateescape") as handle:
            clips = sorted({row["clip"] for row in csv.DictReader(handle)})
        summary = cluster_summary(clips)
        self.assertEqual(summary["clips"], 394)
        self.assertLess(summary["clusters"], 394,
                        "if this passes with 394 clusters the clips would be independent")
        self.assertEqual(summary["clusters"], 174)
        self.assertGreater(summary["mean_cluster_size"], 2.0)


class BootstrapTests(unittest.TestCase):
    @staticmethod
    def _mean_score(truth, score):
        return float(np.mean(score))

    def setUp(self):
        rng = np.random.default_rng(7)
        self.truth = np.array([0] * 50 + [1] * 50)
        self.score = np.concatenate([rng.uniform(0, 0.6, 50), rng.uniform(0.4, 1.0, 50)])

    def test_bootstrap_is_reproducible_from_the_seed(self):
        first = bootstrap_metric_interval(
            "m", self.truth, self.score, self._mean_score, resamples=200, seed=11
        )
        second = bootstrap_metric_interval(
            "m", self.truth, self.score, self._mean_score, resamples=200, seed=11
        )
        self.assertEqual(first.low, second.low)
        self.assertEqual(first.high, second.high)

    def test_a_different_seed_gives_a_different_interval(self):
        first = bootstrap_metric_interval(
            "m", self.truth, self.score, self._mean_score, resamples=200, seed=11
        )
        second = bootstrap_metric_interval(
            "m", self.truth, self.score, self._mean_score, resamples=200, seed=12
        )
        self.assertNotEqual((first.low, first.high), (second.low, second.high))

    def test_point_estimate_lies_inside_the_interval(self):
        interval = bootstrap_metric_interval(
            "m", self.truth, self.score, self._mean_score, resamples=500, seed=3
        )
        self.assertLessEqual(interval.low, interval.point)
        self.assertLessEqual(interval.point, interval.high)

    def test_stratified_scheme_is_labelled(self):
        interval = bootstrap_metric_interval(
            "m", self.truth, self.score, self._mean_score, resamples=50, seed=1
        )
        self.assertEqual(interval.method, METHOD_BOOTSTRAP_STRATIFIED)

    def test_cluster_scheme_is_labelled(self):
        clusters = np.repeat(np.arange(50), 2)
        interval = bootstrap_metric_interval(
            "m", self.truth, self.score, self._mean_score,
            resamples=50, seed=1, clusters=clusters,
        )
        self.assertEqual(interval.method, METHOD_BOOTSTRAP_CLUSTER)

    def test_degenerate_resamples_are_discarded_not_scored_as_zero(self):
        def only_defined_with_both_classes(truth, score):
            return None if len(set(truth.tolist())) < 2 else 0.5

        clusters = np.array([0] * 50 + [1] * 50)  # each cluster is one whole class
        interval = bootstrap_metric_interval(
            "m", self.truth, self.score, only_defined_with_both_classes,
            resamples=100, seed=5, clusters=clusters,
        )
        self.assertGreater(interval.discarded_resamples, 0)
        self.assertEqual(interval.usable_resamples + interval.discarded_resamples, 100)

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(IntervalError):
            bootstrap_metric_interval("m", [0, 1], [0.1], self._mean_score)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(IntervalError):
            bootstrap_metric_interval("m", [], [], self._mean_score)


class RoundingTieTests(unittest.TestCase):
    """Step 3's finding must not be silently 'fixed' by Step 4."""

    def test_predictions_csv_still_stores_only_six_decimals(self):
        import csv

        path = REPO_ROOT / "outputs" / "temporal_violence" / "evaluation" / "predictions.csv"
        with open(path, newline="", encoding="utf-8") as handle:
            raw = [row["fight_probability"] for row in csv.DictReader(handle)]
        places = {len(value.split(".")[1]) for value in raw}
        self.assertEqual(places, {6}, "stored precision changed")
        self.assertLess(len(set(raw)), len(raw), "ties should still be present")

    def test_the_ties_are_still_present_and_unrepaired(self):
        import collections
        import csv

        path = REPO_ROOT / "outputs" / "temporal_violence" / "evaluation" / "predictions.csv"
        with open(path, newline="", encoding="utf-8") as handle:
            raw = [row["fight_probability"] for row in csv.DictReader(handle)]
        counts = collections.Counter(raw)
        tied_rows = sum(n for n in counts.values() if n > 1)
        self.assertEqual(len(raw), 394)
        self.assertEqual(tied_rows, 105, "the recorded tie count changed")
        self.assertEqual(counts["1.000000"], 48)

    def test_no_code_path_dejitters_or_perturbs_the_stored_scores(self):
        """The interval tools must read the file as-is."""
        source = (REPO_ROOT / "tools" / "metric_confidence_intervals.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("jitter", "5e-7", "dejitter", "+ epsilon"):
            self.assertNotIn(forbidden, source)


@unittest.skipUnless(SKLEARN_PRESENT, "scikit-learn is not installed (optional cross-check)")
class SklearnCrossCheckTests(unittest.TestCase):
    """Runs only where scikit-learn is available; never required."""

    def test_crosscheck_reports_agreement(self):
        from crosscheck_metrics_sklearn import compute

        report = compute()
        self.assertEqual(report["status"], "COMPARED")
        self.assertTrue(report["all_metrics_agree"], report["comparisons"])
        self.assertTrue(report["confusion_matrix_agrees"])

    def test_rounding_limitation_is_still_reported(self):
        from crosscheck_metrics_sklearn import compute

        report = compute()
        rounding = report["rounding_limitation_still_stands"]
        self.assertNotEqual(rounding["roc_auc"]["difference_from_recorded"], 0.0)


class SklearnAbsenceTests(unittest.TestCase):
    """The cross-check must degrade cleanly, which is the normal case here."""

    def test_tool_reports_skipped_rather_than_failing_when_absent(self):
        import crosscheck_metrics_sklearn as module

        report = module.compute()
        if SKLEARN_PRESENT:
            self.assertEqual(report["status"], "COMPARED")
        else:
            self.assertEqual(report["status"], "SKIPPED")
            self.assertFalse(report["sklearn_available"])
            self.assertIn("optional", report["reason"])
            self.assertIsNone(report["sklearn_version"])

    def test_temporal_metrics_values_are_reported_either_way(self):
        import crosscheck_metrics_sklearn as module

        report = module.compute()
        self.assertIn("temporal_metrics_values", report)
        self.assertEqual(report["sample_count"], 394)

    def test_scikit_learn_is_not_a_project_dependency(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertNotIn("scikit-learn", requirements)
        self.assertNotIn("sklearn", requirements)


if __name__ == "__main__":
    unittest.main()
