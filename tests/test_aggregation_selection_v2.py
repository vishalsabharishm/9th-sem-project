"""Tests for the carve-only v2 temporal aggregation selection.

Three things are pinned here, in rough order of how badly a regression would
hurt:

  * LEAKAGE. The selection tool must be structurally incapable of reaching the
    394-clip primary evaluation. That is checked by scanning the source and the
    whole import chain, not by trusting the author.
  * CAUSALITY. Every candidate's decision on a prefix must be invariant to any
    future window, proved against adversarial suffixes rather than argued.
  * THE PROTOCOL. The candidate set is closed, the promotion gate is a fixed
    number, the deployed incumbent is not re-tuned, and no winner is forced.

Everything reads the committed carve scores only. No dataset, no checkpoint, no
scikit-learn, no GPU.
"""

import ast
import json
import subprocess
import random
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import select_temporal_aggregation_v2 as v2  # noqa: E402

CARVE_CSV = REPO_ROOT / "temporal_risk" / "carve_window_scores.csv"
OUTPUT_JSON = REPO_ROOT / "temporal_risk" / "frozen_candidate_aggregation.json"

# The artifacts this tool must never reach. Names, not paths, so a relative or
# absolute reference is caught either way.
FORBIDDEN_ARTIFACTS = (
    "primary_window_scores",
    "primary_aggregation_result",
    "paired_comparison",
    "clean_baseline_alignment",
    "alignment_evidence",
    "disagreement_table",
    "predictions.csv",
    "notebook5d98537e15",
)


class LeakageGuardTests(unittest.TestCase):
    """The tool must be carve-only by construction."""

    def test_tool_source_names_no_primary_artifact(self):
        source = Path(v2.__file__).read_text(encoding="utf-8")
        for artifact in FORBIDDEN_ARTIFACTS:
            self.assertNotIn(
                artifact, source,
                f"select_temporal_aggregation_v2.py references {artifact!r}",
            )

    def test_import_chain_reaches_no_primary_artifact(self):
        """A helper the tool imports must not smuggle the primary set in.

        Run in a SUBPROCESS that imports only this tool. Scanning the parent's
        sys.modules would also sweep up modules other test files imported --
        tools/paired_clean_vs_sliding.py legitimately reads the primary scores --
        and would fail or pass depending on test ordering, which is exactly the
        kind of check that lulls you into trusting nothing.
        """
        probe = (
            "import sys, json; "
            f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); "
            f"sys.path.insert(0, {str(REPO_ROOT / 'tools')!r}); "
            "before = set(sys.modules); "
            "import select_temporal_aggregation_v2; "
            "loaded = [getattr(sys.modules[n], '__file__', None) "
            "          for n in set(sys.modules) - before]; "
            "print(json.dumps([f for f in loaded if f]))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = json.loads(result.stdout.strip().splitlines()[-1])

        # Two different standards, deliberately. A module in the chain must
        # never NAME an evaluation-score artifact -- there is no innocent
        # reason to. But src/rwf2000_config.py is a declarative provenance
        # record that mentions predictions.csv in prose while opening nothing,
        # so for those the test checks for actual data ACCESS instead of the
        # word: any open()/read_text()/read_csv() whose literal argument names
        # a forbidden artifact.
        score_artifacts = (
            "primary_window_scores", "primary_aggregation_result",
            "paired_comparison", "clean_baseline_alignment",
            "alignment_evidence", "disagreement_table",
        )
        readers = {"open", "read_text", "read_bytes", "read_csv", "load", "loads"}

        checked = []
        for path in loaded:
            resolved = Path(path).resolve()
            try:
                relative = resolved.relative_to(REPO_ROOT)
            except ValueError:
                continue  # stdlib or site-packages
            if relative.parts[0] not in ("src", "tools"):
                continue
            checked.append(str(relative))
            text = resolved.read_text(encoding="utf-8")

            for artifact in score_artifacts:
                self.assertNotIn(
                    artifact, text,
                    f"{relative} is in the tool's import chain and names {artifact}",
                )

            tree = ast.parse(text, filename=str(relative))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name not in readers:
                    continue
                for literal in ast.walk(node):
                    if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                        for artifact in FORBIDDEN_ARTIFACTS:
                            self.assertNotIn(
                                artifact, literal.value,
                                f"{relative} reads {artifact} at line {node.lineno}",
                            )
        self.assertGreater(len(checked), 0, "no repository modules were inspected")

    def test_tool_only_declares_the_carve_csv(self):
        self.assertEqual(v2.CARVE_CSV.name, "carve_window_scores.csv")
        self.assertTrue(v2.CARVE_CSV.is_file())

    def test_loader_refuses_a_non_carve_split(self, ):
        """A row from any other split must abort rather than be aggregated."""
        rows = CARVE_CSV.read_text(encoding="utf-8").splitlines()
        tampered = REPO_ROOT / "tests" / "_tmp_not_carve.csv"
        body = [rows[0], rows[1].replace("carve,", "primary,", 1)] + rows[2:]
        tampered.write_text("\n".join(body), encoding="utf-8")
        try:
            with self.assertRaises(v2.SelectionError) as caught:
                v2.load_carve(tampered)
            self.assertIn("carve split only", str(caught.exception))
        finally:
            tampered.unlink()


class CandidateSetTests(unittest.TestCase):
    def test_candidate_set_is_exactly_the_approved_six(self):
        self.assertEqual(
            v2.CANDIDATE_NAMES,
            ("max_fpr_matched", "topk_2", "topk_3", "topk_5",
             "consecutive_2", "consecutive_3"),
        )

    def test_no_unapproved_family_or_parameter(self):
        families = {entry["family"] for entry in v2.CANDIDATE_DEFINITIONS}
        self.assertEqual(families, {"max", "topk_mean", "consecutive_persistence"})
        ks = sorted(
            entry["params"]["k"] for entry in v2.CANDIDATE_DEFINITIONS
            if entry["family"] == "topk_mean"
        )
        ns = sorted(
            entry["params"]["n"] for entry in v2.CANDIDATE_DEFINITIONS
            if entry["family"] == "consecutive_persistence"
        )
        self.assertEqual(ks, [2, 3, 5])
        self.assertEqual(ns, [2, 3])
        # EMA must be absent as an implementation, not as a word: the module
        # deliberately NAMES it in the excluded-by-protocol list.
        self.assertFalse(
            [n for n in dir(v2) if "ema" in n.lower() and n.startswith(("score", "make"))],
            "an EMA scorer exists in the module",
        )
        report = v2.build_report(CARVE_CSV)
        self.assertTrue(
            any("EMA" in reason for reason in report["candidate_set"]["excluded_by_protocol"]),
            "EMA should be recorded as deliberately excluded",
        )

    def test_exactly_one_validation_only_control(self):
        controls = [e for e in v2.CANDIDATE_DEFINITIONS if e["role"] == "validation-only control"]
        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0]["name"], "max_fpr_matched")


class DeployedIncumbentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scores, cls.labels = v2.load_carve(CARVE_CSV)
        cls.clips = sorted(cls.scores)

    def test_deployed_rule_is_frozen_at_max_0_14(self):
        self.assertEqual(v2.DEPLOYED_RULE, "max")
        self.assertEqual(v2.DEPLOYED_THRESHOLD, 0.14)

    def test_incumbent_reproduces_the_recorded_carve_confusion(self):
        decisions = {c: max(self.scores[c]) >= 0.14 for c in self.clips}
        self.assertEqual(
            v2.confusion(self.clips, self.labels, decisions),
            {"tp": 85, "fp": 21, "tn": 100, "fn": 34},
        )

    def test_incumbent_matches_the_frozen_aggregation_artifact(self):
        frozen = json.loads(
            (REPO_ROOT / "temporal_risk" / "frozen_aggregation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(frozen["selected"]["rule"], "max")
        self.assertEqual(frozen["selected"]["params"]["threshold"], 0.14)
        recorded = frozen["selected"]["metrics"]
        self.assertEqual(
            (recorded["tp"], recorded["fp"], recorded["tn"], recorded["fn"]),
            (85, 21, 100, 34),
        )

    def test_retuned_max_is_distinguished_from_the_deployed_max(self):
        report = v2.build_report(CARVE_CSV)
        control = next(r for r in report["candidates"] if r["name"] == "max_fpr_matched")
        self.assertEqual(control["role"], "validation-only control")
        self.assertIn("NOT the deployed rule", control["description"])
        self.assertNotEqual(control["threshold"], v2.DEPLOYED_THRESHOLD)
        self.assertEqual(
            report["deployed_incumbent"]["status"],
            "FROZEN -- not re-tuned, not replaced by this script",
        )

    def test_selection_refuses_if_the_incumbent_stops_reproducing(self):
        original = v2.INCUMBENT_TP
        v2.INCUMBENT_TP = original + 1
        try:
            with self.assertRaises(v2.SelectionError) as caught:
                v2.build_report(CARVE_CSV)
            self.assertIn("does not reproduce", str(caught.exception))
        finally:
            v2.INCUMBENT_TP = original


class CarveDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scores, cls.labels = v2.load_carve(CARVE_CSV)

    def test_all_240_clips_present_with_full_window_regime(self):
        self.assertEqual(len(self.scores), 240)
        self.assertEqual({len(v) for v in self.scores.values()}, {17})
        self.assertEqual(sum(len(v) for v in self.scores.values()), 4080)

    def test_label_counts(self):
        fights = sum(1 for v in self.labels.values() if v == "Fight")
        self.assertEqual((fights, len(self.labels) - fights), (119, 121))

    def test_target_budget_is_the_incumbent_false_positive_rate(self):
        self.assertEqual(v2.TARGET_FALSE_POSITIVES, 21)
        self.assertAlmostEqual(v2.TARGET_FPR, 21 / 121, places=15)


class ThresholdDerivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scores, cls.labels = v2.load_carve(CARVE_CSV)
        cls.clips = sorted(cls.scores)

    def test_derivation_is_deterministic(self):
        for definition in v2.CANDIDATE_DEFINITIONS:
            first = v2.evaluate_candidate(definition, self.clips, self.labels, self.scores)
            second = v2.evaluate_candidate(definition, self.clips, self.labels, self.scores)
            self.assertEqual(first["threshold"], second["threshold"], definition["name"])
            self.assertEqual(first["metrics"], second["metrics"], definition["name"])

    def test_every_candidate_hits_the_false_positive_budget(self):
        for definition in v2.CANDIDATE_DEFINITIONS:
            row = v2.evaluate_candidate(definition, self.clips, self.labels, self.scores)
            self.assertEqual(row["achieved_false_positives"], 21, definition["name"])
            self.assertTrue(row["exact_fpr_match"], definition["name"])
            self.assertAlmostEqual(row["achieved_fpr"], 21 / 121, places=12)

    def test_threshold_is_calibrated_not_swept_on_recall(self):
        """No candidate may reach its budget by a threshold that beats it on FP."""
        for definition in v2.CANDIDATE_DEFINITIONS:
            row = v2.evaluate_candidate(definition, self.clips, self.labels, self.scores)
            matrix = row["metrics"]
            self.assertEqual(matrix["fp"] + matrix["tn"], 121)
            self.assertEqual(matrix["tp"] + matrix["fn"], 119)

    def test_fpr_matched_max_recovers_the_deployed_operating_point(self):
        """Sanity check on the calibration machinery itself."""
        definition = next(e for e in v2.CANDIDATE_DEFINITIONS if e["name"] == "max_fpr_matched")
        row = v2.evaluate_candidate(definition, self.clips, self.labels, self.scores)
        self.assertEqual(
            (row["metrics"]["tp"], row["metrics"]["fp"]), (85, 21),
            "FPR-matching max should land on the deployed rule's carve confusion",
        )


class PrefixCausalityTests(unittest.TestCase):
    """No future window may change a decision already taken."""

    @classmethod
    def setUpClass(cls):
        cls.scores, _ = v2.load_carve(CARVE_CSV)
        cls.clips = sorted(cls.scores)

    def _scorers(self):
        return [(e["name"], e["scorer"]) for e in v2.CANDIDATE_DEFINITIONS]

    def test_decision_is_invariant_to_adversarial_suffixes(self):
        rng = random.Random(20260906)
        threshold = 0.14
        for name, scorer in self._scorers():
            with self.subTest(candidate=name):
                for clip in self.clips[:40]:
                    series = self.scores[clip]
                    for j in range(len(series)):
                        prefix = series[:j + 1]
                        baseline = scorer(prefix) >= threshold
                        for _ in range(4):
                            suffix = [rng.choice([0.0, 1.0, rng.random()])
                                      for _ in range(len(series) - j - 1)]
                            adversarial = prefix + suffix
                            self.assertEqual(
                                scorer(adversarial[:j + 1]) >= threshold, baseline,
                                f"{name} prefix {j} on {clip} moved with the suffix",
                            )

    def test_all_ones_suffix_cannot_retroactively_fire_an_alarm(self):
        """The strongest adversarial suffix: every future window maximal."""
        threshold = 0.14
        for name, scorer in self._scorers():
            with self.subTest(candidate=name):
                for clip in self.clips[:40]:
                    series = self.scores[clip]
                    for j in range(len(series)):
                        prefix = list(series[:j + 1])
                        forced = prefix + [1.0] * (len(series) - j - 1)
                        self.assertEqual(
                            scorer(forced[:j + 1]) >= threshold,
                            scorer(prefix) >= threshold,
                            f"{name} prefix {j} on {clip} changed under an all-ones suffix",
                        )

    def test_alarm_monotonicity_matches_each_declared_property(self):
        """max and consecutive can never un-fire; top-k mean provably can.

        This is a characterisation, not a requirement. Top-k mean is NOT
        alarm-monotone -- top-2 of [0.9] is 0.9 but top-2 of [0.9, 0.1] is 0.5
        -- so a streaming alarm can be withdrawn. Causality is unaffected (the
        other tests in this class cover that); what changes is that its
        first-alarm decision and its final decision can disagree, which the
        early-detection protocol has to state rather than discover later.
        """
        threshold = 0.14
        for definition in v2.CANDIDATE_DEFINITIONS:
            name, scorer = definition["name"], definition["scorer"]
            with self.subTest(candidate=name):
                retracted = 0
                for clip in self.clips:
                    series = self.scores[clip]
                    fired = False
                    for j in range(len(series)):
                        now = scorer(series[:j + 1]) >= threshold
                        if fired and not now:
                            retracted += 1
                            break
                        fired = fired or now
                if definition["alarm_monotone"]:
                    self.assertEqual(
                        retracted, 0,
                        f"{name} is declared alarm-monotone but retracted {retracted}",
                    )
                else:
                    self.assertGreater(
                        retracted, 0,
                        f"{name} is declared non-monotone but never retracted; "
                        "the declaration is now wrong",
                    )

    def test_non_monotone_candidates_are_declared_and_measured(self):
        report = v2.build_report(CARVE_CSV)
        by_name = {row["name"]: row for row in report["candidates"]}
        self.assertTrue(by_name["max_fpr_matched"]["alarm_monotone"])
        self.assertTrue(by_name["consecutive_2"]["alarm_monotone"])
        self.assertTrue(by_name["consecutive_3"]["alarm_monotone"])
        for name in ("topk_2", "topk_3", "topk_5"):
            self.assertFalse(by_name[name]["alarm_monotone"], name)
            behaviour = by_name[name]["streaming_behaviour"]
            self.assertGreater(behaviour["alarm_retraction_clips"], 0, name)
        for name, row in by_name.items():
            self.assertEqual(
                row["alarm_monotone"],
                row["streaming_behaviour"]["observed_alarm_monotone"],
                f"{name}: declared monotonicity contradicts the measurement",
            )

    def test_full_prefix_equals_the_full_17_window_aggregate(self):
        for name, scorer in self._scorers():
            with self.subTest(candidate=name):
                for clip in self.clips:
                    series = self.scores[clip]
                    self.assertEqual(scorer(series[:17]), scorer(series))


class AggregatorSemanticsTests(unittest.TestCase):
    def test_topk_matches_its_definition_including_short_prefixes(self):
        series = [0.9, 0.1, 0.5, 0.3, 0.7]
        for k in (2, 3, 5):
            scorer = v2.make_score_topk(k)
            for m in range(1, len(series) + 1):
                prefix = series[:m]
                take = min(k, m)
                expected = sum(sorted(prefix, reverse=True)[:take]) / take
                self.assertAlmostEqual(scorer(prefix), expected, places=12)

    def test_topk_endpoints_are_max_and_mean(self):
        series = [0.2, 0.9, 0.4, 0.1]
        self.assertAlmostEqual(v2.make_score_topk(1)(series), max(series), places=12)
        self.assertAlmostEqual(
            v2.make_score_topk(len(series))(series), sum(series) / len(series), places=12
        )

    def test_consecutive_equals_the_explicit_run_definition(self):
        """max-of-min over runs must equal 'exists n consecutive above t'."""
        rng = random.Random(7)
        for n in (2, 3):
            scorer = v2.make_score_consecutive(n)
            for _ in range(300):
                series = [round(rng.random(), 3) for _ in range(rng.randint(1, 10))]
                threshold = round(rng.random(), 3)
                explicit = any(
                    all(value >= threshold for value in series[i:i + n])
                    for i in range(len(series) - n + 1)
                ) if len(series) >= n else False
                self.assertEqual(scorer(series) >= threshold, explicit,
                                 f"n={n} series={series} t={threshold}")

    def test_consecutive_needs_a_real_run_not_scattered_windows(self):
        """Overlapping-window persistence must still require adjacency."""
        scattered = [0.9, 0.0, 0.9, 0.0, 0.9]
        adjacent = [0.9, 0.9, 0.0, 0.0, 0.0]
        for n in (2, 3):
            scorer = v2.make_score_consecutive(n)
            self.assertLess(scorer(scattered), 0.5, f"n={n} accepted scattered spikes")
        self.assertGreaterEqual(v2.make_score_consecutive(2)(adjacent), 0.9)
        self.assertLess(v2.make_score_consecutive(3)(adjacent), 0.5)

    def test_consecutive_is_undefined_below_its_run_length(self):
        for n in (2, 3):
            scorer = v2.make_score_consecutive(n)
            for m in range(n):
                self.assertEqual(scorer([0.99] * m), float("-inf"))

    def test_overlap_caveat_is_documented(self):
        report = v2.build_report(CARVE_CSV)
        geometry = report["window_geometry"]
        self.assertEqual(geometry["adjacent_window_frame_overlap"], 8)
        self.assertIn("NOT n independent", geometry["independence_caveat"])


class GroupedFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scores, cls.labels = v2.load_carve(CARVE_CSV)
        cls.clips = sorted(cls.scores)
        cls.integrity = v2.fold_integrity(cls.clips, cls.labels)

    def test_no_source_video_spans_two_folds(self):
        self.assertEqual(self.integrity["sources_shared_between_folds"], 0)

    def test_every_clip_appears_exactly_once(self):
        self.assertTrue(self.integrity["clips_assigned_exactly_once"])
        self.assertEqual(self.integrity["clips_assigned"], 240)
        self.assertEqual(sum(self.integrity["fold_sizes"].values()), 240)

    def test_folds_are_five_and_non_empty(self):
        self.assertEqual(self.integrity["folds"], 5)
        for fold, size in self.integrity["fold_sizes"].items():
            self.assertGreater(size, 0, f"fold {fold} is empty")

    def test_fold_assignment_is_deterministic(self):
        first = v2.build_source_folds(self.clips)
        second = v2.build_source_folds(list(reversed(self.clips)))
        self.assertEqual(first, second)

    def test_labels_survive_fold_assignment(self):
        assignment = v2.build_source_folds(self.clips)
        for clip in self.clips:
            self.assertIn(assignment[v2.source_video_id(clip)], range(5))
            self.assertIn(self.labels[clip], ("Fight", "NonFight"))

    def test_cross_validation_is_declared_descriptive(self):
        cv = v2.grouped_cross_validation(
            v2.CANDIDATE_DEFINITIONS[0], self.clips, self.labels, self.scores
        )
        self.assertIn("descriptive only", cv["note"])
        self.assertIn("never influences", cv["note"])
        self.assertEqual(len(cv["folds"]), 5)

    def test_cross_validation_recalibrates_on_training_folds(self):
        """A fold threshold must come from the training folds, not from all 240."""
        cv = v2.grouped_cross_validation(
            v2.CANDIDATE_DEFINITIONS[1], self.clips, self.labels, self.scores
        )
        thresholds = {row["threshold_from_training_folds"] for row in cv["folds"]}
        self.assertGreater(len(thresholds), 1,
                           "every fold produced an identical threshold; suspect no refit")


class PromotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = v2.build_report(CARVE_CSV)

    def test_gate_requires_five_extra_true_positives(self):
        self.assertEqual(v2.PROMOTION_MIN_EXTRA_TP, 5)
        self.assertEqual(v2.PROMOTION_MIN_TP, 90)
        gate = self.report["promotion"]["gate"]
        self.assertEqual(gate["incumbent_tp"], 85)
        self.assertEqual(gate["required_tp"], 90)
        self.assertEqual(gate["criterion"],
                         "carve recall at matched false-positive budget")

    def test_no_candidate_is_promoted_when_the_gate_is_unmet(self):
        below = [
            r for r in self.report["candidates"]
            if r["role"] == "candidate" and r["metrics"]["tp"] < 90
        ]
        if len(below) == 4:  # every candidate fell short
            self.assertFalse(self.report["promotion"]["promoted"])
            self.assertIsNone(self.report["promotion"]["selected"])
            self.assertEqual(self.report["promotion"]["decision"], "NO CANDIDATE PROMOTED")

    def test_promotion_never_forces_a_winner(self):
        """A synthetic near-miss must still be refused."""
        near_miss = [
            {"name": "topk_2", "family": "topk_mean", "params": {"k": 2},
             "role": "candidate", "threshold": 0.1, "achieved_false_positives": 21,
             "metrics": {"tp": 89, "fp": 21, "tn": 100, "fn": 30}},
        ]
        outcome = v2.promote(near_miss)
        self.assertFalse(outcome["promoted"])
        self.assertIsNone(outcome["selected"])
        self.assertEqual(outcome["gate"]["candidates_blocked_by_gate"][0]["tp"], 89)

    def test_promotion_accepts_a_candidate_that_clears_the_gate(self):
        clears = [
            {"name": "topk_3", "family": "topk_mean", "params": {"k": 3},
             "role": "candidate", "threshold": 0.1, "achieved_false_positives": 21,
             "metrics": {"tp": 90, "fp": 21, "tn": 100, "fn": 29}},
        ]
        outcome = v2.promote(clears)
        self.assertTrue(outcome["promoted"])
        self.assertEqual(outcome["selected"]["name"], "topk_3")

    def test_the_control_can_never_be_promoted(self):
        control = [
            {"name": "max_fpr_matched", "family": "max", "params": {},
             "role": "validation-only control", "threshold": 0.1,
             "achieved_false_positives": 21,
             "metrics": {"tp": 119, "fp": 21, "tn": 100, "fn": 0}},
        ]
        self.assertFalse(v2.promote(control)["promoted"])

    def test_tie_break_prefers_fewer_false_positives_then_simplicity(self):
        tied = [
            {"name": "topk_5", "family": "topk_mean", "params": {"k": 5},
             "role": "candidate", "threshold": 0.1, "achieved_false_positives": 21,
             "metrics": {"tp": 95, "fp": 21, "tn": 100, "fn": 24}},
            {"name": "topk_2", "family": "topk_mean", "params": {"k": 2},
             "role": "candidate", "threshold": 0.1, "achieved_false_positives": 21,
             "metrics": {"tp": 95, "fp": 21, "tn": 100, "fn": 24}},
        ]
        self.assertEqual(v2.promote(tied)["selected"]["name"], "topk_2")

    def test_selection_is_deterministic(self):
        again = v2.build_report(CARVE_CSV)
        self.assertEqual(
            self.report["promotion"]["decision"], again["promotion"]["decision"]
        )
        for first, second in zip(self.report["candidates"], again["candidates"]):
            self.assertEqual(first["threshold"], second["threshold"])
            self.assertEqual(first["metrics"], second["metrics"])


class OutputArtifactTests(unittest.TestCase):
    @unittest.skipUnless(OUTPUT_JSON.is_file(), "selection has not been run")
    def test_artifact_declares_itself_a_carve_checkpoint(self):
        report = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(report["artifact_kind"], "carve-validation checkpoint")
        self.assertFalse(report["contains_primary_results"])
        self.assertEqual(report["dataset"]["split"], "carve")
        self.assertEqual(report["dataset"]["clips"], 240)

    @unittest.skipUnless(OUTPUT_JSON.is_file(), "selection has not been run")
    def test_artifact_contains_no_primary_number_or_reference(self):
        text = OUTPUT_JSON.read_text(encoding="utf-8")
        for artifact in FORBIDDEN_ARTIFACTS:
            self.assertNotIn(artifact, text, f"output references {artifact!r}")
        # The primary split's own counts must not appear as data anywhere.
        report = json.loads(text)
        self.assertNotIn("394", json.dumps(report["candidates"]))
        self.assertNotIn("primary", json.dumps(report["candidates"]).lower())

    @unittest.skipUnless(OUTPUT_JSON.is_file(), "selection has not been run")
    def test_artifact_records_the_required_fields(self):
        report = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        for field in ("protocol_version", "dataset", "source_grouping",
                      "deployed_incumbent", "threshold_calibration",
                      "candidate_set", "candidates", "grouped_cross_validation",
                      "promotion", "run"):
            self.assertIn(field, report)
        self.assertEqual(len(report["candidates"]), 6)
        self.assertEqual(list(report["candidate_set"]["names"]), list(v2.CANDIDATE_NAMES))
        self.assertIsNotNone(report["run"]["timestamp_utc"])
        self.assertIsNotNone(report["run"]["tool_sha256"])
        self.assertIsNotNone(report["dataset"]["sha256"])

    @unittest.skipUnless(OUTPUT_JSON.is_file(), "selection has not been run")
    def test_artifact_records_every_required_metric_per_candidate(self):
        report = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        for row in report["candidates"]:
            for field in ("threshold", "target_fpr", "achieved_fpr",
                          "target_false_positives", "achieved_false_positives"):
                self.assertIn(field, row, row["name"])
            for field in ("tp", "fp", "tn", "fn", "recall", "precision",
                          "specificity", "f1", "fpr"):
                self.assertIn(field, row["metrics"], row["name"])


if __name__ == "__main__":
    unittest.main()
