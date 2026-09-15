"""Demo-readiness regression tests for the dashboard presentation path.

These exist because two presentation bugs were found in the final audit, both of
the same kind: the honest value existed in the data but never reached the
screen.

  * The severity badge rendered "High" with no qualification. The underlying
    record carried risk_level_is_validated, buried inside the risk JSON, but the
    analyze response shipped no status text and the UI showed none. A viewer --
    or a viva examiner -- saw an unqualified severity.
  * The saliency endpoint returned a placeholder risk level of "--", a
    meaningless value on screen.

Endpoint tests that would run the full YOLO pipeline are skipped by default:
they take ~30 s per call. The wiring they protect is asserted statically.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from temporal_explanation import (  # noqa: E402
    RISK_STATUS_TEXT,
    describe_provenance_legend,
    risk_presentation,
)

TEMPLATE = REPO_ROOT / "web" / "templates" / "index.html"
APP_JS = REPO_ROOT / "web" / "static" / "app.js"
SERVER = REPO_ROOT / "web" / "server.py"


class RiskStatusReachesTheScreenTests(unittest.TestCase):
    """The bug: a validated-status flag that existed but was never rendered."""

    def test_risk_presentation_needs_no_placeholder_level(self):
        payload = risk_presentation()
        self.assertNotIn("level", payload,
                         "a caller that does not know the level must not invent one")
        self.assertFalse(payload["risk_level_is_validated"])
        self.assertEqual(payload["status_text"], RISK_STATUS_TEXT)

    def test_risk_presentation_keeps_a_level_when_one_is_known(self):
        self.assertEqual(risk_presentation("High")["level"], "High")

    def test_analyze_response_ships_the_risk_status(self):
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn('"risk_status": risk_status', source)
        self.assertIn("risk_status = risk_presentation()", source)

    def test_saliency_endpoint_does_not_fabricate_a_level(self):
        source = SERVER.read_text(encoding="utf-8")
        self.assertNotIn('risk_presentation("--")', source)

    def test_template_has_a_slot_beside_the_badge(self):
        markup = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn('id="riskBadge"', markup)
        self.assertIn('id="riskStatus"', markup)

    def test_client_fills_that_slot_from_the_response(self):
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn('getElementById("riskStatus")', script)
        self.assertIn("data.risk_status", script)
        self.assertIn("risk_level_is_validated", script)


class NoMisleadingWordingTests(unittest.TestCase):
    def test_ui_never_implies_a_calibrated_risk_probability(self):
        for path in (TEMPLATE, APP_JS):
            text = path.read_text(encoding="utf-8").lower()
            for forbidden in ("risk probability", "risk confidence", "risk score"):
                self.assertNotIn(forbidden, text, f"{path.name} contains {forbidden!r}")

    def test_ui_makes_no_real_time_claim(self):
        for path in (TEMPLATE, APP_JS):
            text = path.read_text(encoding="utf-8")
            lowered = text.lower()
            for phrase in ("real-time", "real time", "realtime"):
                if phrase in lowered:
                    # Permitted only where it DENIES the claim.
                    for line in text.splitlines():
                        if phrase in line.lower():
                            self.assertTrue(
                                "not_real_time" in line or "not real" in line.lower(),
                                f"{path.name} asserts real-time: {line.strip()[:80]}",
                            )

    def test_saliency_caveat_is_present_in_the_markup(self):
        """Whitespace-normalised: the copy is line-wrapped in the template."""
        markup = " ".join(TEMPLATE.read_text(encoding="utf-8").split())
        self.assertIn("not a causal or ground-truth explanation", markup)
        self.assertIn("Not proof of causality", markup)
        self.assertIn("not attention", markup)
        self.assertIn("not ground-truth localization", markup)

    def test_replay_and_live_are_still_distinguished(self):
        """The UI now offers both modes, so the distinction must be louder.

        This used to assert the exact sentence "cannot get a live temporal
        score", written when the UI was replay-only and an upload could never
        be scored. The UI now runs live inference, so that sentence is gone;
        what it protected -- that a viewer can always tell which mode produced
        what they are looking at -- is asserted directly instead.
        """
        markup = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("precomputed temporal score", markup)

        # Both modes are named, described, and selectable.
        self.assertIn('data-mode="replay"', markup)
        self.assertIn('data-mode="live"', markup)
        self.assertIn("REPLAY &mdash; precomputed research evidence", markup)
        self.assertIn("LIVE MODEL &mdash; actual video inference", markup)
        self.assertIn("No temporal model runs.", markup)

        # The result carries a banner saying which mode produced it.
        self.assertIn('id="modeBanner"', markup)

        # Live inference is not sold as real time.
        self.assertIn("Offline, not real time", markup)

    def test_the_timeline_caption_does_not_claim_replay_unconditionally(self):
        """A live timeline described as "replayed" would misstate provenance."""
        markup = TEMPLATE.read_text(encoding="utf-8")
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn('id="timelineProvenance"', markup)
        # The client rewrites it per run, from the response's own mode.
        self.assertIn("timelineProvenance", script)
        self.assertIn("R3D-18 forward ", script)

    def test_fusion_is_never_presented_as_an_improvement(self):
        markup = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("did\n        <strong>not</strong> produce a statistically significant improvement",
                      markup)
        self.assertIn("not</em> because it performs better", markup)


class TwoResolutionLimitationIsVisibleTests(unittest.TestCase):
    def test_client_renders_the_raw_temporal_position_count(self):
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn("raw_temporal_positions", script)
        self.assertIn("resolved temporal positions", script)

    def test_client_renders_the_interpolation_caveat(self):
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn("temporal_resolution_caveat", script)

    def test_client_shows_the_faithfulness_flag(self):
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn("faithfulness_tested", script)


class ProvenanceLegendTests(unittest.TestCase):
    def test_analyze_response_ships_the_legend(self):
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn('"provenance_legend": describe_provenance_legend()', source)

    def test_legend_separates_declared_constants_from_model_confidence(self):
        legend = describe_provenance_legend()
        self.assertIn("NOT model confidence", legend["declared_rule_constant"])


class AssetIntegrityTests(unittest.TestCase):
    def test_every_referenced_static_asset_exists(self):
        markup = TEMPLATE.read_text(encoding="utf-8")
        import re
        for name in re.findall(r"filename='([^']+)'", markup):
            self.assertTrue((REPO_ROOT / "web" / "static" / name).is_file(),
                            f"missing static asset {name}")

    def test_demo_preset_videos_exist(self):
        import run_demo
        for key, entry in run_demo.DEMO_CLIPS.items():
            self.assertTrue(Path(entry["video"]).exists(), f"{key}: {entry['video']}")

    def test_explain_endpoint_is_registered(self):
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn('@app.route("/api/explain", methods=["POST"])', source)


class SaliencyStaysOffTheAnalyzePathTests(unittest.TestCase):
    def test_analyze_does_not_compute_saliency(self):
        """Grad-CAM costs ~2.6 s/window; it must stay behind an explicit action."""
        source = SERVER.read_text(encoding="utf-8")
        analyze = source[source.index('def api_analyze'):]
        self.assertNotIn("explain_frame_window", analyze)

    def test_client_requests_saliency_only_on_button_click(self):
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn('getElementById("explainBtn")', script)
        self.assertIn('addEventListener("click"', script)


if __name__ == "__main__":
    unittest.main()
