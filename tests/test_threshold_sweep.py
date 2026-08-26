"""Tests for the decision-threshold sweep.

The sweep decides the operating point of the whole violence branch, so
the arithmetic is checked against hand-computed counts rather than
against itself, and the selection rule is checked for the property that
actually matters: it must never buy recall by flooding the operator with
false alarms.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from threshold_sweep import (  # noqa: E402
    OperatingPoint,
    SweepError,
    evaluate_at,
    read_predictions,
    recommend,
    render_table,
    sweep,
)


def write_predictions(path: Path, rows) -> None:
    """Write a predictions CSV in the evaluation script's format."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "true_label", "fight_probability", "predicted_label"])
        for index, (label, score) in enumerate(rows):
            writer.writerow([index, label, f"{score:.6f}", "unused"])


class EvaluateAtTests(unittest.TestCase):
    """The confusion counts at one threshold, checked by hand."""

    # Four fights scored .9 .8 .4 .1 and four non-fights scored .7 .3 .2 .05.
    TRUTHS = [1, 1, 1, 1, 0, 0, 0, 0]
    SCORES = [0.9, 0.8, 0.4, 0.1, 0.7, 0.3, 0.2, 0.05]

    def test_counts_at_a_middling_threshold(self):
        # At 0.5: fights .9 .8 caught, .4 .1 missed; non-fight .7 alarms.
        point = evaluate_at(self.TRUTHS, self.SCORES, 0.5)
        self.assertEqual(point.true_positive, 2)
        self.assertEqual(point.false_negative, 2)
        self.assertEqual(point.false_positive, 1)
        self.assertEqual(point.true_negative, 3)
        self.assertAlmostEqual(point.recall, 0.5)
        self.assertAlmostEqual(point.specificity, 0.75)
        self.assertAlmostEqual(point.precision, 2 / 3)
        self.assertAlmostEqual(point.accuracy, 5 / 8)

    def test_lowering_the_threshold_never_lowers_recall(self):
        previous = -1.0
        for threshold in (0.9, 0.7, 0.5, 0.3, 0.1):
            recall = evaluate_at(self.TRUTHS, self.SCORES, threshold).recall
            self.assertGreaterEqual(recall, previous)
            previous = recall

    def test_threshold_is_inclusive(self):
        """``score >= threshold`` -- a clip exactly at the threshold alarms."""
        point = evaluate_at([1], [0.4], 0.4)
        self.assertEqual(point.true_positive, 1)

    def test_f1_is_the_harmonic_mean(self):
        point = evaluate_at(self.TRUTHS, self.SCORES, 0.5)
        expected = 2 * point.precision * point.recall / (point.precision + point.recall)
        self.assertAlmostEqual(point.f1, expected)

    def test_balanced_accuracy_and_youden_agree(self):
        point = evaluate_at(self.TRUTHS, self.SCORES, 0.5)
        self.assertAlmostEqual(
            point.balanced_accuracy, (point.recall + point.specificity) / 2
        )
        self.assertAlmostEqual(
            point.youden_j, point.recall + point.specificity - 1
        )

    def test_precision_is_none_when_nothing_is_predicted_positive(self):
        point = evaluate_at(self.TRUTHS, self.SCORES, 1.01)
        self.assertEqual(point.true_positive + point.false_positive, 0)
        self.assertIsNone(point.precision)
        self.assertIsNone(point.f1)


class SweepGridTests(unittest.TestCase):
    """The grid itself."""

    def test_grid_covers_the_requested_range_inclusively(self):
        points = sweep([1, 0], [0.9, 0.1], start=0.1, stop=0.5, step=0.1)
        self.assertEqual([p.threshold for p in points], [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_zero_step_is_rejected(self):
        with self.assertRaises(SweepError):
            sweep([1, 0], [0.9, 0.1], step=0.0)


class RecommendationTests(unittest.TestCase):
    """The selection rule, which is the opinionated part."""

    def _point(self, threshold, recall, specificity, precision, false_positive):
        return OperatingPoint(
            threshold=threshold,
            true_positive=0,
            false_positive=false_positive,
            true_negative=0,
            false_negative=0,
            accuracy=0.0,
            precision=precision,
            recall=recall,
            f1=0.5,
            specificity=specificity,
            balanced_accuracy=(recall + specificity) / 2,
            youden_j=recall + specificity - 1,
        )

    def test_prefers_recall_over_accuracy_within_the_budget(self):
        low_recall = self._point(0.5, recall=0.80, specificity=0.95, precision=0.94, false_positive=10)
        high_recall = self._point(0.3, recall=0.90, specificity=0.88, precision=0.86, false_positive=23)
        best, _ = recommend(
            [low_recall, high_recall],
            min_precision=0.75,
            max_false_positive_rate=0.15,
        )
        self.assertEqual(best.threshold, 0.3)

    def test_refuses_to_buy_recall_with_unacceptable_false_alarms(self):
        acceptable = self._point(0.4, recall=0.85, specificity=0.90, precision=0.89, false_positive=19)
        reckless = self._point(0.01, recall=1.00, specificity=0.10, precision=0.53, false_positive=175)
        best, _ = recommend([acceptable, reckless], min_precision=0.75, max_false_positive_rate=0.15)
        self.assertEqual(best.threshold, 0.4)

    def test_falls_back_and_says_so_when_nothing_qualifies(self):
        poor = self._point(0.5, recall=0.4, specificity=0.5, precision=0.4, false_positive=90)
        best, rationale = recommend([poor], min_precision=0.9, max_false_positive_rate=0.01)
        self.assertEqual(best.threshold, 0.5)
        self.assertIn("fell back", rationale)

    def test_ties_on_recall_break_toward_fewer_false_positives(self):
        noisy = self._point(0.2, recall=0.9, specificity=0.88, precision=0.86, false_positive=23)
        quiet = self._point(0.4, recall=0.9, specificity=0.92, precision=0.90, false_positive=15)
        best, _ = recommend([noisy, quiet], min_precision=0.75, max_false_positive_rate=0.15)
        self.assertEqual(best.threshold, 0.4)


class ReadPredictionsTests(unittest.TestCase):
    """Reading the evaluation CSV."""

    def test_round_trips_labels_and_scores(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "predictions.csv"
            write_predictions(path, [("Fight", 0.9), ("NonFight", 0.2)])
            truths, scores = read_predictions(path)
            self.assertEqual(truths, [1, 0])
            self.assertEqual(scores, [0.9, 0.2])

    def test_stored_predicted_label_is_ignored(self):
        """The old threshold must not leak into the new sweep."""
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "predictions.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["index", "true_label", "fight_probability", "predicted_label"]
                )
                # predicted_label deliberately contradicts the probability.
                writer.writerow([0, "Fight", "0.900000", "NonFight"])
                writer.writerow([1, "NonFight", "0.100000", "Fight"])
            truths, scores = read_predictions(path)
            point = evaluate_at(truths, scores, 0.5)
            self.assertEqual(point.true_positive, 1)
            self.assertEqual(point.true_negative, 1)
            self.assertEqual(point.false_positive, 0)

    def test_missing_column_is_reported(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "bad.csv"
            path.write_text("index,true_label\n0,Fight\n", encoding="utf-8")
            with self.assertRaises(SweepError):
                read_predictions(path)

    def test_single_class_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "predictions.csv"
            write_predictions(path, [("Fight", 0.9), ("Fight", 0.8)])
            with self.assertRaises(SweepError):
                read_predictions(path)

    def test_missing_file_is_reported(self):
        with self.assertRaises(SweepError):
            read_predictions(Path("/nonexistent/predictions.csv"))


class CommandLineTests(unittest.TestCase):
    """The CLI writes the artifacts the phase needs."""

    def test_writes_json_csv_and_table(self):
        with tempfile.TemporaryDirectory() as name:
            work = Path(name)
            predictions = work / "predictions.csv"
            rows = [("Fight", 0.9 - 0.01 * i) for i in range(50)]
            rows += [("NonFight", 0.4 - 0.005 * i) for i in range(50)]
            write_predictions(predictions, rows)

            output = work / "sweep"
            code = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "tools" / "threshold_sweep.py"),
                    "--predictions", str(predictions),
                    "--output-dir", str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(code.returncode, 0, code.stderr)
            self.assertTrue((output / "threshold_sweep.json").is_file())
            self.assertTrue((output / "threshold_sweep.csv").is_file())
            self.assertTrue((output / "threshold_sweep.txt").is_file())

            record = json.loads((output / "threshold_sweep.json").read_text())
            self.assertEqual(record["clips"], 100)
            self.assertEqual(record["positives"], 50)
            self.assertIn("selected", record)
            self.assertIn("baseline", record)

    def test_missing_predictions_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as name:
            code = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "tools" / "threshold_sweep.py"),
                    "--predictions", str(Path(name) / "absent.csv"),
                    "--output-dir", str(Path(name) / "out"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(code.returncode, 0)


class RenderTests(unittest.TestCase):
    """The human-readable table."""

    def test_every_threshold_appears_once(self):
        points = sweep([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.1, 0.9, 0.2)
        text = render_table(points)
        self.assertEqual(len(text.splitlines()), len(points) + 2)


if __name__ == "__main__":
    unittest.main()


class SelectedThresholdTests(unittest.TestCase):
    """The threshold recorded in config must match the real predictions.

    Pins the documented operating point to the evidence it came from, so a
    later edit to either the constant or the scores cannot silently drift
    apart from the numbers quoted in docs/temporal_baseline_results.md.
    """

    PREDICTIONS = (
        REPOSITORY_ROOT / "outputs" / "temporal_violence" / "evaluation" /
        "predictions.csv"
    )

    def setUp(self):
        if not self.PREDICTIONS.is_file():
            self.skipTest("baseline predictions.csv is not present")
        sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

    def test_recorded_threshold_reproduces_documented_metrics(self):
        from rwf2000_config import SELECTED_DECISION_THRESHOLD

        truths, scores = read_predictions(self.PREDICTIONS)
        point = evaluate_at(truths, scores, SELECTED_DECISION_THRESHOLD)

        self.assertEqual(SELECTED_DECISION_THRESHOLD, 0.14)
        self.assertEqual(point.true_positive, 178)
        self.assertEqual(point.false_negative, 22)
        self.assertEqual(point.false_positive, 33)
        self.assertEqual(point.true_negative, 161)
        self.assertAlmostEqual(point.accuracy, 0.8604, places=4)
        self.assertAlmostEqual(point.precision, 0.8436, places=4)
        self.assertAlmostEqual(point.recall, 0.8900, places=4)
        self.assertAlmostEqual(point.f1, 0.8662, places=4)
        self.assertAlmostEqual(point.specificity, 0.8299, places=4)

    def test_baseline_half_reproduces_the_kaggle_published_matrix(self):
        """The 0.5 row must still match what the Kaggle run reported."""
        truths, scores = read_predictions(self.PREDICTIONS)
        point = evaluate_at(truths, scores, 0.5)
        self.assertEqual(
            (point.true_positive, point.false_positive,
             point.false_negative, point.true_negative),
            (164, 19, 36, 175),
        )
        self.assertAlmostEqual(point.accuracy, 0.8604, places=4)

    def test_selected_threshold_beats_the_default_on_recall(self):
        from rwf2000_config import SELECTED_DECISION_THRESHOLD

        truths, scores = read_predictions(self.PREDICTIONS)
        selected = evaluate_at(truths, scores, SELECTED_DECISION_THRESHOLD)
        default = evaluate_at(truths, scores, 0.5)
        self.assertGreater(selected.recall, default.recall)
        self.assertLess(selected.false_negative, default.false_negative)
        # And must not have paid for it with overall accuracy.
        self.assertGreaterEqual(selected.accuracy, default.accuracy)
