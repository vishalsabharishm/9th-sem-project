import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risk_assessment import RiskAssessor
from temporal_event_adapter import (
    PROVENANCE_MEASURED,
    TEMPORAL_EVENT_TYPE,
    FrozenAggregationRule,
    PrecomputedWindowScoreSource,
    TemporalEventAdapter,
    WindowScore,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_JSON = REPO_ROOT / "temporal_risk" / "frozen_aggregation.json"
CARVE_CSV = REPO_ROOT / "temporal_risk" / "carve_window_scores.csv"
PRIMARY_CSV = REPO_ROOT / "temporal_risk" / "primary_window_scores.csv"


class FrozenAggregationRuleTests(unittest.TestCase):
    def test_max_rule_matches_threshold_semantics(self):
        rule = FrozenAggregationRule(rule="max", params={"threshold": 0.5}, source_path=Path("x"))
        self.assertTrue(rule.decide([0.1, 0.6, 0.2]))
        self.assertFalse(rule.decide([0.1, 0.4, 0.2]))

    def test_k_of_n_rule(self):
        rule = FrozenAggregationRule(
            rule="k_of_n", params={"k": 2, "window_threshold": 0.5}, source_path=Path("x")
        )
        self.assertTrue(rule.decide([0.6, 0.6, 0.1]))
        self.assertFalse(rule.decide([0.6, 0.1, 0.1]))

    def test_empty_scores_never_fire(self):
        rule = FrozenAggregationRule(rule="max", params={"threshold": 0.1}, source_path=Path("x"))
        self.assertFalse(rule.decide([]))


class TemporalEventAdapterTests(unittest.TestCase):
    def test_fired_clip_produces_measured_event(self):
        rule = FrozenAggregationRule(rule="max", params={"threshold": 0.5}, source_path=Path("x"))
        adapter = TemporalEventAdapter(rule)
        scores = [
            WindowScore(window_index=0, first_frame=0, last_frame=15, fight_probability=0.2),
            WindowScore(window_index=1, first_frame=8, last_frame=23, fight_probability=0.91),
        ]
        event = adapter.evaluate_clip("demo_clip.avi", scores)
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, TEMPORAL_EVENT_TYPE)
        self.assertAlmostEqual(event.confidence, 0.91)
        self.assertTrue(any(PROVENANCE_MEASURED in e for e in event.evidence))

    def test_clip_below_threshold_produces_no_event(self):
        rule = FrozenAggregationRule(rule="max", params={"threshold": 0.5}, source_path=Path("x"))
        adapter = TemporalEventAdapter(rule)
        scores = [WindowScore(window_index=0, first_frame=0, last_frame=15, fight_probability=0.1)]
        self.assertIsNone(adapter.evaluate_clip("demo_clip.avi", scores))

    def test_risk_assessor_maps_temporal_event_to_high_with_measured_provenance(self):
        rule = FrozenAggregationRule(rule="max", params={"threshold": 0.1}, source_path=Path("x"))
        adapter = TemporalEventAdapter(rule)
        scores = [WindowScore(window_index=0, first_frame=0, last_frame=15, fight_probability=0.8)]
        event = adapter.evaluate_clip("demo_clip.avi", scores)
        assessment = RiskAssessor().assess(event)
        self.assertEqual(assessment.risk_level, "High")
        self.assertEqual(assessment.confidence_provenance, PROVENANCE_MEASURED)


class PrecomputedWindowScoreSourceAndFrozenRuleOnRealDataTests(unittest.TestCase):
    """Sanity checks against the actual recovered artifacts, not synthetic data."""

    @classmethod
    def setUpClass(cls):
        if not (FROZEN_JSON.exists() and CARVE_CSV.exists() and PRIMARY_CSV.exists()):
            raise unittest.SkipTest("recovered temporal_risk artifacts not present")
        cls.frozen_rule = FrozenAggregationRule.load(FROZEN_JSON)
        cls.carve_source = PrecomputedWindowScoreSource(CARVE_CSV)
        cls.primary_source = PrecomputedWindowScoreSource(PRIMARY_CSV)

    def test_carve_metrics_reproduce_selection_output(self):
        """Replaying the frozen rule clip-by-clip over the carve CSV via the
        adapter must reproduce the exact confusion matrix that
        tools/select_temporal_aggregation.py recorded as 'selected'."""
        import json

        selected = json.loads(FROZEN_JSON.read_text())["selected"]
        adapter = TemporalEventAdapter(self.frozen_rule)

        tp = fp = tn = fn = 0
        for clip in self.carve_source.clips():
            scores = self.carve_source.get(clip)
            fired = adapter.evaluate_clip(clip, scores) is not None
            actual_fight = self.carve_source.true_label(clip) == "Fight"
            if fired and actual_fight:
                tp += 1
            elif fired and not actual_fight:
                fp += 1
            elif not fired and actual_fight:
                fn += 1
            else:
                tn += 1

        self.assertEqual(tp, selected["metrics"]["tp"])
        self.assertEqual(fp, selected["metrics"]["fp"])
        self.assertEqual(tn, selected["metrics"]["tn"])
        self.assertEqual(fn, selected["metrics"]["fn"])

    def test_known_demo_clips_fire_as_expected(self):
        adapter = TemporalEventAdapter(self.frozen_rule)

        fight_scores = self.primary_source.get("val/Val_Fight/trtrhrt_1049.avi")
        self.assertIsNotNone(fight_scores)
        self.assertIsNotNone(adapter.evaluate_clip("val/Val_Fight/trtrhrt_1049.avi", fight_scores))

        nonfight_scores = self.primary_source.get("val/Val_NonFight/ZCUy99AN_0.avi")
        self.assertIsNotNone(nonfight_scores)
        self.assertIsNone(adapter.evaluate_clip("val/Val_NonFight/ZCUy99AN_0.avi", nonfight_scores))


if __name__ == "__main__":
    unittest.main()
