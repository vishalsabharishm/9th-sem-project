"""Tests for the demo polish phase: preflight, incident report, timeline, cache.

Each group guards a property that would otherwise be easy to lose:

  * PREFLIGHT must keep the three tiers apart. The standard demo replays
    committed scores and does NOT need the R3D checkpoint; reporting a missing
    checkpoint as a blocker for replay would send someone hunting for a file
    they do not need.
  * THE REPORT must not invent values. Crowding and Proximity genuinely carry no
    confidence, and a report that filled in a plausible number would be
    fabricating evidence.
  * THE TIMELINE must offer only windows that were actually scored, so a window
    offered for saliency is always one whose source frames can be rebuilt.
  * THE CACHE must key on checkpoint identity, so a replaced file cannot be
    served from a stale entry.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from demo_preflight import (  # noqa: E402
    TIER_LIVE,
    TIER_REPLAY,
    TIER_SALIENCY,
    run_preflight,
)
from incident_report import (  # noqa: E402
    LIMITATIONS,
    build_incident_report,
    render_incident_report_text,
)

TEMPLATE = REPO_ROOT / "web" / "templates" / "index.html"
APP_JS = REPO_ROOT / "web" / "static" / "app.js"
SERVER = REPO_ROOT / "web" / "server.py"


def sample_summary(source="csv"):
    return {
        "clip_key": "val/Val_Fight/example_0.avi",
        "frames_processed": 150,
        "ground_truth_label": "Fight",
        "ground_truth_available": True,
        "final_temporal_decision": "Fight",
        "temporal_violence_signal_fired": True,
        "temporal_signal_first_frame": 15,
        "windows_observed": 17,
        "temporal_provenance": {
            "temporal_source": source,
            "note": "replayed from the committed CSV; no model ran in this process",
            "scores_from": "temporal_risk/primary_window_scores.csv",
        },
    }


SAMPLE_WINDOWS = [
    {"window_index": 0, "first_frame": 0, "last_frame": 15, "fight_probability": 0.11},
    {"window_index": 1, "first_frame": 8, "last_frame": 23, "fight_probability": 0.97},
    {"window_index": 2, "first_frame": 16, "last_frame": 31, "fight_probability": 0.44},
]

SAMPLE_RISKS = [
    {"event_type": "Stationary Person", "risk_level": "Low", "confidence": 0.9,
     "confidence_provenance": "declared_rule_constant", "risk_level_is_validated": False},
    {"event_type": "Stationary Person", "risk_level": "Low", "confidence": 0.9,
     "confidence_provenance": "declared_rule_constant", "risk_level_is_validated": False},
    {"event_type": "Crowding", "risk_level": "Medium", "confidence": None,
     "confidence_provenance": None, "risk_level_is_validated": False},
]


class PreflightTierTests(unittest.TestCase):
    def test_all_three_tiers_are_reported(self):
        result = run_preflight()
        tiers = {check.tier for check in result.checks}
        self.assertEqual(tiers, {TIER_REPLAY, TIER_LIVE, TIER_SALIENCY})

    def test_missing_checkpoint_does_not_block_replay(self):
        """The documented replay design needs no R3D checkpoint."""
        result = run_preflight(r3d_checkpoint=Path("models/definitely_absent.pt"))
        self.assertTrue(result.replay_ready)
        self.assertFalse(result.live_ready)
        self.assertFalse(result.saliency_ready)

    def test_missing_checkpoint_remedy_says_replay_still_works(self):
        result = run_preflight(r3d_checkpoint=Path("models/definitely_absent.pt"))
        remedies = " ".join(c.remedy for c in result.failures())
        self.assertIn("standard replay demo does NOT", remedies)

    def test_missing_temporal_csv_does_block_replay(self):
        result = run_preflight(temporal_csv=Path("temporal_risk/absent.csv"))
        self.assertFalse(result.replay_ready)
        self.assertTrue(any("temporal score CSV" == c.name for c in result.failures(TIER_REPLAY)))

    def test_temporal_csv_schema_and_size_are_checked(self):
        result = run_preflight()
        names = {c.name: c for c in result.checks}
        self.assertTrue(names["temporal CSV schema"].ok)
        self.assertTrue(names["temporal CSV contents"].ok)
        self.assertIn("394 clips, 6698 rows", names["temporal CSV contents"].detail)

    def test_failures_carry_an_actionable_remedy(self):
        result = run_preflight(r3d_checkpoint=Path("models/absent.pt"))
        for check in result.failures():
            self.assertTrue(check.remedy, f"{check.name} has no remedy text")

    def test_render_states_what_is_blocked(self):
        text = run_preflight(r3d_checkpoint=Path("models/absent.pt")).render()
        self.assertIn("Standard demo is ready", text)
        self.assertIn("Saliency is unavailable", text)

    def test_result_note_explains_the_replay_distinction(self):
        self.assertIn("REPLAY", run_preflight().as_dict()["note"])


class IncidentReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_incident_report(
            sample_summary(), SAMPLE_RISKS, SAMPLE_WINDOWS, "High")

    def test_required_sections_are_present(self):
        for field in ("report_version", "generated_utc", "clip", "temporal_evidence",
                      "spatial_events", "risk_interpretation", "saliency",
                      "provenance", "limitations", "not_claimed"):
            self.assertIn(field, self.report)

    def test_peak_probability_comes_from_the_windows_given(self):
        temporal = self.report["temporal_evidence"]
        self.assertAlmostEqual(temporal["peak_probability"], 0.97)
        self.assertEqual(temporal["peak_window_index"], 1)
        self.assertEqual(temporal["peak_window_frames"], [8, 23])

    def test_risk_carries_its_validation_disclaimer(self):
        risk = self.report["risk_interpretation"]
        self.assertEqual(risk["level"], "High")
        self.assertFalse(risk["risk_level_is_validated"])
        self.assertIn("not validated as a risk model", risk["disclaimer"].lower())

    def test_rules_without_confidence_do_not_gain_one(self):
        crowding = next(e for e in self.report["spatial_events"]
                        if e["event_type"] == "Crowding")
        self.assertIsNone(crowding["confidence"])
        self.assertIsNone(crowding["confidence_provenance"])

    def test_declared_constants_stay_labelled_as_declared(self):
        stationary = next(e for e in self.report["spatial_events"]
                          if e["event_type"] == "Stationary Person")
        self.assertEqual(stationary["confidence"], 0.9)
        self.assertEqual(stationary["confidence_provenance"], "declared_rule_constant")
        self.assertEqual(stationary["occurrences"], 2)

    def test_replay_provenance_is_explicit(self):
        provenance = self.report["provenance"]
        self.assertEqual(provenance["temporal_source"], "csv")
        self.assertIn("replay", provenance["mode"])

    def test_live_provenance_is_distinguished_from_replay(self):
        report = build_incident_report(
            sample_summary(source="live"), SAMPLE_RISKS, SAMPLE_WINDOWS, "High")
        self.assertEqual(report["provenance"]["mode"], "live inference")

    def test_nothing_is_claimed_that_was_not_established(self):
        claims = self.report["not_claimed"]
        for key in ("calibrated_risk", "faithful_explanation", "real_time", "causality"):
            self.assertFalse(claims[key])

    def test_limitations_travel_inside_the_report(self):
        self.assertEqual(len(self.report["limitations"]), len(LIMITATIONS))
        joined = " ".join(self.report["limitations"]).lower()
        self.assertIn("not calibrated", joined)
        self.assertIn("faithfulness", joined)
        self.assertIn("no real-time", joined)

    def test_absent_saliency_is_distinguished_from_failed_saliency(self):
        not_requested = build_incident_report(
            sample_summary(), SAMPLE_RISKS, SAMPLE_WINDOWS, "High")["saliency"]
        self.assertFalse(not_requested["requested"])

        failed = build_incident_report(
            sample_summary(), SAMPLE_RISKS, SAMPLE_WINDOWS, "High",
            saliency={"available": False, "reason": "checkpoint_unavailable",
                      "detail": "no weights"})["saliency"]
        self.assertTrue(failed["requested"])
        self.assertFalse(failed["available"])
        self.assertEqual(failed["status"], "checkpoint_unavailable")

    def test_present_saliency_keeps_its_caveats(self):
        payload = {
            "available": True,
            "temporal_evidence": {"window_first_frame": 8, "window_last_frame": 23},
            "saliency": {"peak_frame": 17, "displayed_temporal_slices": 16,
                         "raw_temporal_positions": 2,
                         "temporal_resolution_caveat": "interpolated from 2 positions",
                         "faithfulness_tested": False},
        }
        block = build_incident_report(sample_summary(), SAMPLE_RISKS, SAMPLE_WINDOWS,
                                      "High", saliency=payload)["saliency"]
        self.assertTrue(block["available"])
        self.assertFalse(block["faithfulness_tested"])
        self.assertEqual(block["raw_temporal_positions"], 2)
        self.assertIn("not a causal", block["interpretation"].lower())

    def test_empty_windows_yield_none_not_a_fabricated_peak(self):
        report = build_incident_report(sample_summary(), SAMPLE_RISKS, [], "Unknown")
        self.assertIsNone(report["temporal_evidence"]["peak_probability"])
        self.assertIsNone(report["temporal_evidence"]["peak_window_index"])

    def test_structure_is_deterministic_apart_from_the_timestamp(self):
        first = build_incident_report(sample_summary(), SAMPLE_RISKS, SAMPLE_WINDOWS,
                                      "High", generated_at="fixed")
        second = build_incident_report(sample_summary(), SAMPLE_RISKS, SAMPLE_WINDOWS,
                                       "High", generated_at="fixed")
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(second, sort_keys=True))

    def test_text_rendering_contains_the_disclaimer(self):
        text = render_incident_report_text(self.report)
        self.assertIn("INCIDENT REPORT", text)
        self.assertIn("not validated as a risk model", text.lower())
        self.assertIn("LIMITATIONS", text)
        self.assertIn("none recorded", text)   # Crowding has no confidence


class PathTraversalTests(unittest.TestCase):
    """clip_key is user-controlled and was joined to the dataset root unchecked.

    Two distinct escapes were reachable through /api/explain before this guard:
    ``..`` segments walked out of the dataset, and pathlib REPLACES the base
    when the right-hand operand is absolute, so 'C:/Windows/win.ini' resolved to
    exactly that. Any file OpenCV would open could be reached, and the differing
    response for a present versus absent file worked as a file-existence oracle
    for arbitrary paths.
    """

    def setUp(self):
        sys.path.insert(0, str(REPO_ROOT / "web"))
        import server
        self.server = server

    def test_legitimate_clip_still_resolves(self):
        resolved = self.server._resolve_video_path("val/Val_Fight/trtrhrt_1049.avi")
        self.assertIsNotNone(resolved)
        self.assertIn("RWF-2000", str(resolved))

    def test_dot_dot_traversal_is_refused(self):
        for key in ("../../../README.md",
                    "../../../models/yolov8s.pt",
                    "../../../../../../Windows/win.ini",
                    "val/../../../../README.md"):
            self.assertIsNone(self.server._resolve_video_path(key),
                              f"traversal not blocked: {key}")

    def test_absolute_path_is_refused(self):
        """pathlib drops the base for an absolute operand -- the subtler escape."""
        for key in ("C:/Windows/System32/drivers/etc/hosts",
                    r"C:\Windows\win.ini",
                    "/etc/passwd"):
            self.assertIsNone(self.server._resolve_video_path(key),
                              f"absolute path not blocked: {key}")

    def test_dataset_root_itself_is_not_a_clip(self):
        self.assertIsNone(self.server._resolve_video_path(""))
        self.assertIsNone(self.server._resolve_video_path("."))

    def test_explain_checks_membership_before_touching_the_filesystem(self):
        """An unknown key must reveal nothing about what exists on disk."""
        source = SERVER.read_text(encoding="utf-8")
        explain = source[source.index("def api_explain"):source.index("def api_preflight")]
        membership = explain.index('_score_source.get(clip_key) is None')
        resolve = explain.index("_resolve_video_path(clip_key)")
        self.assertLess(membership, resolve,
                        "membership must be checked before the path is resolved")

    def test_errors_do_not_echo_absolute_filesystem_paths(self):
        source = SERVER.read_text(encoding="utf-8")
        explain = source[source.index("def api_explain"):source.index("def api_preflight")]
        self.assertNotIn("{video_path}", explain,
                         "error text must not leak the resolved absolute path")


class MalformedRequestTests(unittest.TestCase):
    """A wrong TYPE is a client error, not a server error.

    ``(payload.get("clip_key") or "").strip()`` raises AttributeError on a list
    or a number, which surfaced as HTTP 500 -- a crash where the caller merely
    sent the wrong type.
    """

    def setUp(self):
        sys.path.insert(0, str(REPO_ROOT / "web"))
        import server
        self.extract = server._clip_key_from

    def test_non_string_clip_keys_are_rejected_not_crashed(self):
        for bad in ([], ["a"], {"a": 1}, 5, 3.2, True, None):
            self.assertIsNone(self.extract({"clip_key": bad}), repr(bad))

    def test_blank_and_missing_are_rejected(self):
        self.assertIsNone(self.extract({}))
        self.assertIsNone(self.extract({"clip_key": ""}))
        self.assertIsNone(self.extract({"clip_key": "   "}))

    def test_valid_key_is_returned_stripped(self):
        self.assertEqual(self.extract({"clip_key": "  val/a.avi  "}), "val/a.avi")

    def test_both_endpoints_use_the_shared_extractor(self):
        """Checks ASSIGNMENTS, not any occurrence of the old pattern.

        The retired expression is quoted inside _clip_key_from's docstring to
        explain the bug it fixes, so a plain substring search flags the very
        documentation that describes the fix.
        """
        source = SERVER.read_text(encoding="utf-8")
        self.assertEqual(source.count("_clip_key_from(payload)"), 2,
                         "explain and report must both use the guarded extractor")
        # Only JSON payloads need the guard. /api/analyze reads request.form,
        # where Flask's .get() always yields str or None, so .strip() there is
        # type-safe and correctly left alone.
        assignments = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith("clip_key =")
        ]
        self.assertTrue(assignments, "no clip_key assignment found")
        for line in assignments:
            guarded = "_clip_key_from(payload)" in line
            form_sourced = "request.form.get" in line
            self.assertTrue(
                guarded or form_sourced,
                f"clip_key assigned from an unguarded JSON payload: {line}",
            )
        self.assertTrue(
            any("request.form.get" in line for line in assignments),
            "expected the analyze endpoint to read clip_key from request.form",
        )


class StaleStateTests(unittest.TestCase):
    """Two bugs found in the hardening audit, both about state outliving its run."""

    def test_failed_saliency_reaches_the_report_as_a_failure(self):
        """It was recorded only on success, so a failure read as 'not requested'.

        The UI would say "Saliency unavailable (checkpoint_unavailable)" while
        the exported report said saliency was never asked for -- the exact
        inconsistency the requested/available split exists to prevent.
        """
        script = APP_JS.read_text(encoding="utf-8")
        record_at = script.index("__recordSaliency(data)")
        failure_at = script.index("if (!data.available) {", record_at - 2000)
        self.assertLess(record_at, failure_at,
                        "saliency must be recorded BEFORE the failure branch returns")

    def test_report_distinguishes_failure_from_absence(self):
        failed = build_incident_report(
            sample_summary(), SAMPLE_RISKS, SAMPLE_WINDOWS, "High",
            saliency={"available": False, "reason": "checkpoint_unavailable",
                      "detail": "no weights on this machine"})["saliency"]
        self.assertTrue(failed["requested"])
        self.assertFalse(failed["available"])
        self.assertEqual(failed["status"], "checkpoint_unavailable")
        self.assertIn("no weights", failed["detail"])

    def test_a_new_analyze_clears_the_previous_runs_state(self):
        """A failed analyze left the OLD clip armed for Explain and Export."""
        script = APP_JS.read_text(encoding="utf-8")
        submit = script[script.index("els.analyzeBtn.disabled = true;"):
                        script.index('const res = await fetch("/api/analyze"')]
        for cleared in ("__clearAnalysis", "__lastClipKey = null",
                        "selectedWindow = null"):
            self.assertIn(cleared, submit,
                          f"{cleared} must be reset before a new analyze starts")

    def test_clear_hook_drops_both_analysis_and_saliency(self):
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn("window.__clearAnalysis = () => { lastAnalysis = null; "
                      "lastSaliency = null; };", script)

    def test_explain_button_is_disabled_when_state_is_cleared(self):
        script = APP_JS.read_text(encoding="utf-8")
        submit = script[script.index("els.analyzeBtn.disabled = true;"):
                        script.index('const res = await fetch("/api/analyze"')]
        self.assertIn("staleExplain.disabled = true", submit)


class SelectedWindowInReportTests(unittest.TestCase):
    """The report showed the peak window but not the one the operator chose."""

    def test_selected_window_is_recorded_when_supplied(self):
        report = build_incident_report(
            sample_summary(), SAMPLE_RISKS, SAMPLE_WINDOWS, "High",
            selected_window=SAMPLE_WINDOWS[2])
        selected = report["selected_window"]
        self.assertEqual(selected["window_index"], 2)
        self.assertEqual(selected["first_frame"], 16)
        self.assertEqual(selected["last_frame"], 31)

    def test_selected_window_is_not_conflated_with_the_peak(self):
        report = build_incident_report(
            sample_summary(), SAMPLE_RISKS, SAMPLE_WINDOWS, "High",
            selected_window=SAMPLE_WINDOWS[2])
        self.assertNotEqual(report["selected_window"]["window_index"],
                            report["temporal_evidence"]["peak_window_index"])
        self.assertIn("not necessarily the peak", report["selected_window"]["note"])

    def test_selected_window_is_none_when_not_supplied(self):
        report = build_incident_report(
            sample_summary(), SAMPLE_RISKS, SAMPLE_WINDOWS, "High")
        self.assertIsNone(report["selected_window"])

    def test_text_rendering_keeps_selected_and_peak_apart(self):
        report = build_incident_report(
            sample_summary(), SAMPLE_RISKS, SAMPLE_WINDOWS, "High",
            selected_window=SAMPLE_WINDOWS[2])
        text = render_incident_report_text(report)
        self.assertIn("SELECTED WINDOW", text)
        self.assertIn("not necessarily the peak", text)

    def test_client_sends_the_selected_window(self):
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn("selected_window: selectedWindow", script)

    def test_server_forwards_the_selected_window(self):
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn('selected_window = payload.get("selected_window")', source)
        self.assertIn("selected_window=selected_window", source)


class ReportEndpointTests(unittest.TestCase):
    def test_report_route_is_registered(self):
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn('@app.route("/api/report", methods=["POST"])', source)

    def test_report_does_not_rerun_the_pipeline(self):
        source = SERVER.read_text(encoding="utf-8")
        report_fn = source[source.index("def api_report"):source.index("@app.route(\"/api/analyze\"")]
        self.assertNotIn("run_demo(", report_fn)

    def test_preflight_route_is_registered(self):
        self.assertIn('@app.route("/api/preflight")', SERVER.read_text(encoding="utf-8"))


class TimelineTests(unittest.TestCase):
    def test_free_form_frame_input_is_gone(self):
        """A numeric box let an examiner request an unscored start frame."""
        markup = TEMPLATE.read_text(encoding="utf-8")
        self.assertNotIn('id="saliencyFrame"', markup)
        self.assertIn('id="selectedWindow"', markup)
        self.assertIn('id="windowTable"', markup)

    def test_explain_button_starts_disabled(self):
        markup = TEMPLATE.read_text(encoding="utf-8")
        button = markup[markup.index('id="explainBtn"') - 60:markup.index('id="explainBtn"') + 60]
        self.assertIn("disabled", button)

    def test_client_uses_the_selected_window_not_a_typed_number(self):
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn("selectedWindow.first_frame", script)
        self.assertNotIn('getElementById("saliencyFrame")', script)

    def test_windows_come_only_from_the_analysis_response(self):
        """No window may be synthesised in the client."""
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn("renderWindowChart(data.window_scores", script)
        self.assertNotIn("for (let i = 0; i < 17", script)

    def test_empty_window_list_shows_an_explicit_unavailable_state(self):
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn("No temporal windows available", script)
        self.assertIn("saliency unavailable", script)

    def test_timeline_states_that_clicking_does_not_run_the_model(self):
        markup = " ".join(TEMPLATE.read_text(encoding="utf-8").split())
        self.assertIn("replayed from the committed scores", markup)
        self.assertIn("does not run the model", markup)

    def test_per_window_decision_uses_the_frozen_threshold(self):
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn("probability >= 0.14", script)


class EngineCacheTests(unittest.TestCase):
    def test_cache_key_includes_checkpoint_digest(self):
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn("digest = file_sha256(checkpoint)", source)
        self.assertIn("key = (str(checkpoint.resolve()), digest, device)", source)

    def test_cache_is_guarded_by_a_lock(self):
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn("_engine_cache_lock", source)
        self.assertIn("threading.Lock()", source)

    def test_cache_holds_only_a_model_never_scores(self):
        source = SERVER.read_text(encoding="utf-8")
        cache_fn = source[source.index("def _cached_engine"):source.index("@app.route(\"/\")")]
        for forbidden in ("fight_probability", "window_scores", "PRIMARY_CSV"):
            self.assertNotIn(forbidden, cache_fn,
                             "the engine cache must never become a score source")

    def test_replay_still_reads_the_precomputed_source(self):
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn("_score_source = PrecomputedWindowScoreSource(PRIMARY_CSV)", source)


if __name__ == "__main__":
    unittest.main()
