"""Tests for the guided multi-step workflow.

The dashboard is now six client-side views over ONE Flask template. These tests
protect the properties that a re-layout could quietly break:

  * every view the router can reach actually exists in the markup;
  * the workflow cannot be skipped -- later steps start locked;
  * analysis starts ONLY on an explicit button press;
  * the processing screen never fabricates progress;
  * the scientific disclosures survived being moved between pages;
  * no backend contract moved with the UI.

They assert structure and wording, not appearance.
"""
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "web"))

TEMPLATE = REPO_ROOT / "web" / "templates" / "index.html"
APP_JS = REPO_ROOT / "web" / "static" / "app.js"
STYLE = REPO_ROOT / "web" / "static" / "style.css"
SERVER = REPO_ROOT / "web" / "server.py"

MARKUP = TEMPLATE.read_text(encoding="utf-8")
SCRIPT = APP_JS.read_text(encoding="utf-8")
FLAT = " ".join(MARKUP.split())

VIEWS = ("home", "setup", "processing", "results", "explain", "report")


class ViewStructureTests(unittest.TestCase):
    def test_every_routed_view_exists_in_the_markup(self):
        for view in VIEWS:
            self.assertIn(f'data-view="{view}"', MARKUP, f"view '{view}' is missing")

    def test_the_router_knows_exactly_these_views(self):
        match = re.search(r"const VIEW_ORDER = \[(.*?)\];", SCRIPT, re.S)
        self.assertIsNotNone(match, "VIEW_ORDER not found")
        declared = re.findall(r'"([a-z]+)"', match.group(1))
        self.assertEqual(tuple(declared), VIEWS)

    def test_only_home_is_visible_before_anything_happens(self):
        for view in VIEWS:
            block = re.search(r'<section class="([^"]*)" data-view="%s"' % view, MARKUP)
            self.assertIsNotNone(block, view)
            classes = block.group(1)
            if view == "home":
                self.assertNotIn("hidden", classes, "home must be the landing view")
            else:
                self.assertIn("hidden", classes, f"{view} must start hidden")

    def test_stepper_covers_every_step_after_home(self):
        for view in VIEWS[1:]:
            self.assertIn(f'data-goto="{view}"', MARKUP, f"no stepper entry for {view}")

    def test_later_steps_start_locked(self):
        """The workflow cannot be skipped before an analysis exists."""
        for step in re.findall(r"<button class=\"step\"[^>]*>", MARKUP):
            self.assertIn("disabled", step, f"step not initially locked: {step}")

    def test_single_template_and_no_new_dependency(self):
        templates = list((REPO_ROOT / "web" / "templates").glob("*.html"))
        self.assertEqual([p.name for p in templates], ["index.html"])
        self.assertFalse((REPO_ROOT / "package.json").exists(),
                         "a frontend build system must not have been introduced")

    def test_no_external_resources(self):
        """The demo must work on a laptop with no internet."""
        for path in (TEMPLATE, STYLE, APP_JS):
            text = path.read_text(encoding="utf-8")
            for hit in re.findall(r'(?:src|href)="(https?://[^"]+)"', text):
                self.fail(f"{path.name} loads an external resource: {hit}")


class ExplicitStartTests(unittest.TestCase):
    def test_analysis_starts_only_on_an_explicit_click(self):
        self.assertIn('els.analyzeBtn.addEventListener("click", runAnalyze);', SCRIPT)
        # The Replay-demo shortcut arms the mode; it must not run anything.
        shortcut = SCRIPT.split("startReplayDemo")[1].split("});")[0]
        self.assertNotIn("runAnalyze", shortcut,
                         "the home shortcut must not start an analysis by itself")

    def test_start_button_is_disabled_until_a_video_is_chosen(self):
        tag = re.search(r"<button id=\"analyzeBtn\"[^>]*>", MARKUP)
        self.assertIsNotNone(tag)
        self.assertIn("disabled", tag.group(0))

    def test_results_unlock_only_after_a_successful_response(self):
        body = SCRIPT[SCRIPT.index("async function runAnalyze"):]
        ok_branch = body.index('setStatus("Analysis complete."')
        self.assertLess(body.index('unlockStep("results")') - ok_branch, 400,
                        "results must unlock on the success path only")
        self.assertIn("lockStepsAfter(\"processing\")", body,
                      "a new run must re-lock the steps a previous run unlocked")


class ProcessingHonestyTests(unittest.TestCase):
    def test_pipeline_lists_the_real_stages(self):
        for stage in ("video_input", "yolo_detection", "object_tracking",
                      "spatial_analysis", "temporal_analysis", "evidence_fusion",
                      "risk_interpretation", "explanation_ready"):
            self.assertIn(f'data-stage="{stage}"', MARKUP, f"missing stage {stage}")

    def test_stage_ids_match_the_names_the_server_publishes(self):
        """The client keys off the server's own stage vocabulary."""
        server = SERVER.read_text(encoding="utf-8")
        for stage in ("yolo_detection", "object_tracking", "spatial_analysis",
                      "temporal_analysis"):
            self.assertIn(f'"{stage}"', server, f"{stage} not declared server-side")
        for stage in ("evidence_fusion", "risk_interpretation"):
            self.assertIn(f'"{stage}"', server)

    def test_interleaved_stages_are_declared_not_animated_as_a_sequence(self):
        """Detection/tracking/spatial/temporal run together, once per frame.

        Showing them completing one after another would depict a pipeline this
        project does not have, so the UI states the concurrency instead.
        """
        self.assertIn('id="interleavedNote"', MARKUP)
        self.assertIn("together, once per decoded frame", MARKUP)
        self.assertIn("INTERLEAVED_STAGES", SERVER.read_text(encoding="utf-8"))

    def test_processing_is_driven_by_server_phases_not_a_timer(self):
        self.assertIn("function applyJobPhase", SCRIPT)
        self.assertIn("async function pollJob", SCRIPT)
        poll = SCRIPT[SCRIPT.index("async function pollJob"):]
        poll = poll[:poll.index("\n}")]
        self.assertIn("/api/analysis/", poll)
        self.assertNotIn("Math.random", poll)

    def test_explanation_ready_does_not_mean_saliency_was_generated(self):
        block = MARKUP[MARKUP.index('data-stage="explanation_ready"'):]
        block = block[:block.index("</li>")]
        self.assertIn("no saliency is generated automatically", block)

    def test_processing_screen_promises_no_percentage(self):
        block = MARKUP[MARKUP.index('data-view="processing"'):MARKUP.index('data-view="results"')]
        self.assertNotIn("%", block, "the processing view must not show a progress percentage")
        self.assertIn("indeterminate", block)

    def test_stage_states_come_from_response_fields(self):
        fn = SCRIPT[SCRIPT.index("function setStagesFromResult"):]
        fn = fn[:fn.index("\n}")]
        for field in ("frames_processed", "spatial_fusion_feature", "window_scores", "fusion"):
            self.assertIn(field, fn, f"{field} must drive its stage state")
        self.assertNotIn("Math.random", fn)

    def test_a_failed_run_shows_an_actionable_error(self):
        self.assertIn("function showProcessingError", SCRIPT)
        self.assertIn("Analysis did not complete.", SCRIPT)
        self.assertIn("Back to setup", SCRIPT)
        # The server's own message is shown, escaped, rather than a generic one.
        fn = SCRIPT[SCRIPT.index("function showProcessingError"):]
        fn = fn[:fn.index("\n}")]
        self.assertIn("escapeHtml(message)", fn)


class DisclosuresSurviveTheReorganisationTests(unittest.TestCase):
    """Every scientific caveat must still be on a page the examiner sees."""

    def test_mode_distinction_is_intact(self):
        self.assertIn("REPLAY &mdash; precomputed research evidence", MARKUP)
        self.assertIn("LIVE MODEL &mdash; actual video inference", MARKUP)
        self.assertIn("No temporal model runs.", MARKUP)
        self.assertIn("Offline, not real time", MARKUP)
        self.assertIn('id="modeBanner"', MARKUP)

    def test_risk_disclaimer_is_intact(self):
        self.assertIn('id="riskStatus"', MARKUP)
        self.assertIn("data.risk_status", SCRIPT)
        self.assertIn("risk_level_is_validated", SCRIPT)

    def test_saliency_disclaimers_are_intact(self):
        self.assertIn("not a causal or ground-truth explanation", FLAT)
        self.assertIn("Not proof of causality", FLAT)
        self.assertIn("not attention", FLAT)
        self.assertIn("not ground-truth localization", FLAT)

    def test_fusion_is_not_sold_as_an_improvement(self):
        self.assertIn(
            "fusion did <strong>not</strong> produce a statistically "
            "significant improvement over the temporal-only baseline", FLAT)
        for forbidden in ("fusion improves", "fusion is superior", "outperforms"):
            self.assertNotIn(forbidden, FLAT.lower())

    def test_no_risk_probability_language_anywhere(self):
        for path in (TEMPLATE, APP_JS):
            text = path.read_text(encoding="utf-8").lower()
            for forbidden in ("risk probability", "risk confidence", "risk score"):
                self.assertNotIn(forbidden, text, f"{path.name} contains {forbidden!r}")


class VideoPlaybackTests(unittest.TestCase):
    def test_both_players_exist_with_controls(self):
        for player in ("resultVideo", "previewVideo"):
            tag = re.search(r"<video[^>]*\bid=\"%s\"[^>]*>" % player, MARKUP)
            self.assertIsNotNone(tag, f"{player} is missing")
            self.assertIn("controls", tag.group(0), f"{player} has no controls")

    def test_an_undecodable_annotated_video_is_reported_not_hidden(self):
        """The backend writes mp4v, which browsers do not decode."""
        self.assertIn('id="videoFallback"', MARKUP)
        self.assertIn("els.resultVideo.onerror", SCRIPT)
        self.assertIn("cannot play the annotated video", SCRIPT)
        # It must offer the file rather than merely apologising.
        self.assertIn("Open annotated video", SCRIPT)

    def test_a_browser_playable_copy_is_requested_not_assumed(self):
        """The annotated artifact stays mp4v; a preview copy is made beside it."""
        self.assertIn("/api/preview_video", SCRIPT)
        server = SERVER.read_text(encoding="utf-8")
        self.assertIn("BROWSER_PREVIEW_SUFFIX", server)
        # Only codecs browsers actually decode may be produced.
        codecs = server[server.index("BROWSER_CODECS = ("):]
        codecs = codecs[:codecs.index(")\n")]
        self.assertNotIn("mp4v", codecs, "mp4v is what the browser cannot play")
        for playable in ("avc1", "VP90", "VP80"):
            self.assertIn(playable, codecs)

    def test_the_original_annotated_artifact_is_never_overwritten(self):
        server = SERVER.read_text(encoding="utf-8")
        fn = server[server.index("def _browser_preview_for("):]
        fn = fn[:fn.index("\n@app.route")]
        # The preview always gets its own name derived from the source stem.
        self.assertIn("source.stem + BROWSER_PREVIEW_SUFFIX", fn)
        # And a failed encode removes its own output rather than leaving junk.
        self.assertIn("target.unlink(missing_ok=True)", fn)

    def test_a_failed_conversion_degrades_instead_of_breaking(self):
        self.assertIn("conversion_unavailable", SERVER.read_text(encoding="utf-8"))
        self.assertIn("Open annotated video", SCRIPT)

    def test_new_analysis_resets_the_session(self):
        self.assertIn('data-reset="session"', MARKUP)
        self.assertIn("function resetSession", SCRIPT)
        fn = SCRIPT[SCRIPT.index("function resetSession"):]
        fn = fn[:fn.index("\n}")]
        for cleared in ("state.session = null", "selectedWindow = null",
                        "__clearAnalysis", "__lastClipKey = null"):
            self.assertIn(cleared, fn, f"{cleared} must be reset")

    def test_timeline_timestamps_are_measured_not_assumed(self):
        """fps is derived from the preview duration, never hard-coded."""
        fn = SCRIPT[SCRIPT.index("function measureFps"):]
        fn = fn[:fn.index("\n}\n")]
        self.assertIn("frames / duration", fn)
        for guess in ("30.0", "= 30;", "25.0"):
            self.assertNotIn(guess, fn, "frame rate must not be assumed")

    def test_leaving_a_view_pauses_its_video(self):
        self.assertIn("if (owner && owner.dataset.view !== name && !v.paused) v.pause();", SCRIPT)


class BackendContractUnchangedTests(unittest.TestCase):
    """The productization must not have moved a single API contract."""

    def test_endpoints_are_unchanged(self):
        source = SERVER.read_text(encoding="utf-8")
        for route in ("/api/analyze", "/api/explain", "/api/report",
                      "/api/preflight", "/api/presets", "/api/clips"):
            self.assertIn(f'@app.route("{route}"', source)

    def test_client_still_posts_the_same_fields(self):
        self.assertIn('formData.append("mode", state.mode)', SCRIPT)
        self.assertIn('formData.append("preset"', SCRIPT)
        self.assertIn('formData.append("clip_key"', SCRIPT)
        self.assertIn('formData.append("video"', SCRIPT)

    def test_windows_still_come_only_from_the_response(self):
        self.assertIn("renderWindowChart(data.window_scores", SCRIPT)
        self.assertNotIn("for (let i = 0; i < 17", SCRIPT)

    def test_the_explain_picker_reuses_the_same_windows_and_selection(self):
        fn = SCRIPT[SCRIPT.index("function renderExplainPicker"):]
        fn = fn[:fn.index("\nfunction selectWindow")]
        self.assertIn("selectWindow(w)", fn, "the picker must share one selection path")
        self.assertNotIn("fetch(", fn, "selecting a window must not call the backend")


class BrowserPreviewEndpointTests(unittest.TestCase):
    """The presentation copy, exercised through the real Flask app."""

    def setUp(self):
        import server as SERVER_MODULE

        SERVER_MODULE.app.config["TESTING"] = True
        self.module = SERVER_MODULE
        self.client = SERVER_MODULE.app.test_client()

    def test_a_non_string_name_is_refused(self):
        response = self.client.post("/api/preview_video", json={"video_name": 42})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["reason"], "bad_request")

    def test_a_traversing_name_cannot_escape_the_output_directory(self):
        for name in ("../../../README.md", "..\\..\\server.py", "/etc/passwd"):
            response = self.client.post("/api/preview_video", json={"video_name": name})
            self.assertIn(response.status_code, (400, 404), name)
            self.assertFalse(response.get_json().get("available"), name)

    def test_an_unknown_video_is_reported_not_guessed(self):
        response = self.client.post("/api/preview_video",
                                    json={"video_name": "demo_nothing_here.mp4"})
        self.assertEqual(response.status_code, 404)
        body = response.get_json()
        self.assertFalse(body["available"])
        self.assertEqual(body["reason"], "missing")

    def test_only_browser_playable_codecs_are_candidates(self):
        """mp4v is exactly what the browser cannot decode."""
        fourccs = [c[0] for c in self.module.BROWSER_CODECS]
        self.assertNotIn("mp4v", fourccs)
        self.assertIn("avc1", fourccs)

    def test_the_preview_name_never_collides_with_the_original(self):
        suffix = self.module.BROWSER_PREVIEW_SUFFIX
        self.assertTrue(suffix)
        source = Path("demo_example.mp4")
        for _fourcc, ext, _label in self.module.BROWSER_CODECS:
            preview = source.with_name(source.stem + suffix + ext)
            self.assertNotEqual(preview.name, source.name)


class StaleVideoTests(unittest.TestCase):
    def test_the_previous_runs_footage_is_cleared_before_the_new_preview(self):
        """Converting takes seconds; the old clip must not fill the gap.

        Leaving it there showed one clip's annotated frames beside another
        clip's evidence, which is the stale-state failure this project has
        repeatedly guarded against.
        """
        fn = SCRIPT[SCRIPT.index("async function loadBrowserPreview"):]
        fn = fn[:fn.index("\n}")]
        clear_at = fn.index('els.resultVideo.removeAttribute("src")')
        fetch_at = fn.index('fetch("/api/preview_video"')
        self.assertLess(clear_at, fetch_at,
                        "the old source must be cleared before the new one is fetched")
        self.assertLess(fn.index("state.fps = null"), fetch_at,
                        "the measured frame rate belongs to the old video")


class JobLifecycleTests(unittest.TestCase):
    def setUp(self):
        import server as SERVER_MODULE

        SERVER_MODULE.app.config["TESTING"] = True
        self.client = SERVER_MODULE.app.test_client()

    def test_an_unknown_job_is_a_clean_404(self):
        response = self.client.get("/api/analysis/not-a-real-job")
        self.assertEqual(response.status_code, 404)
        self.assertIn("Unknown or expired", response.get_json()["error"])

    def test_an_invalid_mode_never_starts_a_job(self):
        response = self.client.post("/api/analysis", data={"mode": "banana", "preset": "fight"})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("job_id", response.get_json())

    def test_live_without_a_video_never_starts_a_job(self):
        response = self.client.post("/api/analysis", data={"mode": "live"})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("job_id", response.get_json())

    def test_the_job_declares_which_stages_are_concurrent(self):
        server = SERVER.read_text(encoding="utf-8")
        self.assertIn("INTERLEAVED_STAGES", server)
        self.assertIn("POST_LOOP_STAGES", server)
        # And the temporal wording must differ by mode, so replay can never
        # read as model execution.
        self.assertIn("running R3D-18 inference", server)
        self.assertIn("replaying committed window probabilities", server)

    def test_no_phase_carries_a_percentage(self):
        server = SERVER.read_text(encoding="utf-8")
        phases = server[server.index("ANALYSIS_PHASES = ("):]
        phases = phases[:phases.index(")")]
        self.assertNotIn("percent", phases.lower())
        self.assertIn("analysing", phases)


class ProcessingChoreographyTests(unittest.TestCase):
    """The animation must never claim work that has not happened."""

    def test_nothing_is_processing_before_the_server_reports_a_phase(self):
        """The request has not even been accepted when the view opens."""
        body = SCRIPT[SCRIPT.index("async function runAnalyze"):]
        body = body[:body.index("async function pollJob")] if "async function pollJob" in body else body
        pre_submit = body[:body.index("/api/analysis")]
        self.assertIn('setStages("waiting")', pre_submit)
        self.assertNotIn('setStages("processing")', pre_submit,
                         "nothing may read as running before the job is accepted")

    def test_the_interleaved_block_gets_one_scan_not_per_row_flow(self):
        """A row-to-row flow would depict a handoff that does not happen."""
        css = STYLE.read_text(encoding="utf-8")
        self.assertIn("group-active", css)
        self.assertIn("groupScan", css)
        self.assertIn("strip.classList.toggle(\"group-active\"", SCRIPT)
        # The connector between rows must stay static.
        self.assertNotIn("connectorFlow", css)

    def test_post_loop_stages_are_revealed_as_complete_not_as_active(self):
        """Fusion and risk finished inside run_demo before finalising began.

        Staggering them is a reveal of finished work. Marking them
        "processing" during finalising would assert they were executing then,
        which is false.
        """
        fn = SCRIPT[SCRIPT.index("function applyJobPhase"):]
        fn = fn[:fn.index("\nasync function pollJob")]
        finalising = fn[fn.index('phase === "finalising"'):fn.index('phase === "complete"')]
        self.assertIn('li.dataset.state = "complete"', finalising)
        self.assertIn("li.dataset.reveal", finalising)
        # The only thing genuinely in progress at finalising is artifact reading.
        self.assertIn('li.dataset.state = "processing"', finalising)

    def test_completion_state_is_announced_without_overclaiming(self):
        self.assertIn('id="completionBanner"', MARKUP)
        banner = MARKUP[MARKUP.index('id="completionBanner"'):]
        banner = banner[:banner.index("</div>", banner.index("</span>"))]
        self.assertIn("Analysis complete", banner)
        self.assertIn("Evidence available for inspection", banner)
        # It must not claim an explanation was produced.
        self.assertNotIn("saliency generated", banner.lower())

    def test_auto_navigation_leaves_a_manual_fallback(self):
        self.assertIn('id="viewResultsBtn"', MARKUP)
        self.assertIn("showView(\"results\")", SCRIPT)

    def test_a_rerun_clears_the_previous_completion_state(self):
        body = SCRIPT[SCRIPT.index("async function runAnalyze"):]
        pre_submit = body[:body.index("/api/analysis")]
        self.assertIn('classList.remove("settled", "group-active")', pre_submit)
        self.assertIn("delete li.dataset.reveal", pre_submit)


class PreviewPreparationTests(unittest.TestCase):
    def test_the_player_is_hidden_while_a_new_preview_converts(self):
        """currentSrc lingers after load(), so clearing must be visible."""
        fn = SCRIPT[SCRIPT.index("async function loadBrowserPreview"):]
        fn = fn[:fn.index("\n}")]
        hide_at = fn.index('els.resultVideo.classList.add("hidden")')
        fetch_at = fn.index('fetch("/api/preview_video"')
        self.assertLess(hide_at, fetch_at, "hide the player before converting")
        self.assertIn("preview-spinner", fn)
        self.assertIn("Preparing a browser-playable preview", fn)

    def test_the_player_is_revealed_only_with_a_new_source(self):
        fn = SCRIPT[SCRIPT.index("async function loadBrowserPreview"):]
        fn = fn[:fn.index("\n}")]
        reveal_at = fn.index('els.resultVideo.classList.remove("hidden");\n      els.resultVideo.src')
        self.assertGreater(reveal_at, fn.index('fetch("/api/preview_video"'))


class FaviconTests(unittest.TestCase):
    def test_the_page_declares_its_own_icon(self):
        """Without this the browser requests /favicon.ico and logs a 404."""
        self.assertIn('rel="icon"', MARKUP)
        self.assertIn("data:image/svg+xml", MARKUP)
        # Inline, so it is still one request-free page on an offline machine.
        icon = MARKUP[MARKUP.index('rel="icon"'):]
        icon = icon[:icon.index("/>")]
        self.assertNotIn("http", icon.replace("http://www.w3.org/2000/svg", ""))


class SaliencyBelongsToItsWindowTests(unittest.TestCase):
    """A heatmap describes exactly one window and must not outlive the selection.

    Found in browser testing: after generating saliency for window 11 and then
    clicking window 3, the panel still showed frames 88-103 under a label
    reading "Window 3" -- and the exported report paired that saliency block
    with selected_window frames 24-39, putting a factual inconsistency into a
    delivered artifact.
    """

    @staticmethod
    def _fn(name):
        """One top-level function body, ending at the next top-level line."""
        start = SCRIPT.index("function " + name)
        out = []
        for i, line in enumerate(SCRIPT[start:].splitlines(keepends=True)):
            if i and line.startswith("}"):
                out.append(line)
                break
            out.append(line)
        return "".join(out)

    def test_changing_the_selection_clears_the_previous_saliency(self):
        fn = self._fn("selectWindow(w)")
        self.assertIn("clearSaliencyForNewWindow()", fn)
        # Only on an actual change -- re-selecting the same window keeps it.
        self.assertIn("selectedWindow.window_index !== w.window_index", fn)

    def test_clearing_drops_the_payload_the_report_reads(self):
        """Clearing only the panel would still export a mismatched report."""
        fn = self._fn("clearSaliencyForNewWindow")
        self.assertIn("__clearSaliency", fn)
        self.assertIn("saliencyResult", fn)
        self.assertIn("window.__clearSaliency = () =>", SCRIPT)

    def test_the_cleared_state_explains_itself(self):
        self.assertIn("Saliency cleared", SCRIPT)
        self.assertIn("described the previously", SCRIPT)
        self.assertIn("Generate it again for this window", SCRIPT)

    def test_a_stale_badge_was_not_used_instead(self):
        """A heatmap labelled stale is still one an examiner can misread, and
        the exported report has no way to render a badge at all."""
        fn = self._fn("clearSaliencyForNewWindow")
        self.assertNotIn("stale", fn.lower())


class NewRunLeavesNothingOfTheOldOneTests(unittest.TestCase):
    """Starting a second analysis must not show the first one's evidence.

    Found in review rehearsal. Three separate holes, all on the path an
    examiner actually walks (run Fight, explain, report, then run the false
    positive):

    * the home "New Analysis" button declared data-reset="session" but the
      delegated reset handler only fires for elements that also carry
      data-goto, which that button does not -- so entering a new analysis from
      the home page reset nothing and left results, explain and report
      unlocked on the PREVIOUS clip;
    * resetSession() dropped the exported state but not the pixels, so the old
      video, heatmap and window selection stayed in the hidden views;
    * the processing screen kept the previous run's "Completed in 29.84s"
      label while a new run was starting, where it reads as this run's timing.
    """

    @staticmethod
    def _fn(name):
        start = SCRIPT.index("function " + name)
        out = []
        for i, line in enumerate(SCRIPT[start:].splitlines(keepends=True)):
            if i and line.startswith("}"):
                out.append(line)
                break
            out.append(line)
        return "".join(out)

    def test_reset_clears_the_previous_runs_visible_evidence(self):
        fn = self._fn("resetSession")
        self.assertIn("clearRunEvidence()", fn)

    def test_clearing_covers_every_surface_that_showed_the_old_run(self):
        fn = self._fn("clearRunEvidence")
        for element in ("resultVideo", "saliencyResult", "selectedWindow",
                        "explainBtn", "processingElapsed", "completionBanner",
                        "modeBanner"):
            with self.subTest(element=element):
                self.assertIn(element, fn)

    def test_the_old_video_is_detached_not_merely_hidden(self):
        """A hidden <video> that keeps its src still holds the old clip."""
        fn = self._fn("clearRunEvidence")
        self.assertIn('removeAttribute("src")', fn)
        self.assertIn("load()", fn)

    def test_the_replay_live_distinction_does_not_survive_a_reset(self):
        """A stale "LIVE MODEL" over a replay run would misdescribe the run."""
        fn = self._fn("clearRunEvidence")
        self.assertIn("modeBanner", fn)
        self.assertIn('classList.remove("live", "replay")', fn)

    def test_home_entry_point_resets_like_the_in_flow_buttons(self):
        """The button carries data-reset="session"; that must be true of it."""
        marker = 'document.getElementById("startNewAnalysis")'
        handler = SCRIPT[SCRIPT.index(marker):]
        handler = handler[:handler.index("});") + 3]
        self.assertIn("resetSession()", handler)
        self.assertIn('lockStepsAfter("setup")', handler)

    def test_the_home_button_still_declares_the_reset_it_performs(self):
        button = MARKUP[MARKUP.index('id="startNewAnalysis"'):]
        button = button[:button.index(">") + 1]
        self.assertIn('data-reset="session"', button)

    def test_starting_a_run_clears_the_previous_runs_timing(self):
        """applyJobPhase only overwrites this once the server answers; until
        then the old duration would be on screen for the new run."""
        start = SCRIPT.index('const priorBanner = document.getElementById("completionBanner")')
        window = SCRIPT[start:start + 600]
        self.assertIn("priorElapsed", window)
        self.assertIn('priorElapsed.textContent = ""', window)


class HierarchyKeepsEveryFigureTests(unittest.TestCase):
    """Promoting the verdict must not quietly drop the numbers under it.

    The results headline was four equal tiles; it is now one dominant decision
    with its peak probability, over a supporting row. That is a presentation
    change, so the test is that nothing was lost in the move: the same four
    figures, each still carrying the provenance note that says where it came
    from.
    """

    @staticmethod
    def _fn(name):
        start = SCRIPT.index("function " + name)
        out = []
        for i, line in enumerate(SCRIPT[start:].splitlines(keepends=True)):
            if i and line.startswith("}"):
                out.append(line)
                break
            out.append(line)
        return "".join(out)

    def test_all_four_figures_are_still_rendered(self):
        """Each figure must still carry a label. Casing is presentation: the
        decision's label moved into the eyebrow and its note when the verdict
        became the page's hero, so the check is on the words, not their case."""
        fn = self._fn("renderTemporalHeadline").lower()
        for label in ("system decision", "peak fight probability",
                      "first alarm", "windows scored"):
            with self.subTest(label=label):
                self.assertIn(label, fn)

    def test_every_figure_keeps_its_provenance_note(self):
        """A number without its provenance is exactly what this UI must not show."""
        fn = self._fn("renderTemporalHeadline")
        for note in ("frozen rule: max", "measured model probability",
                     "earliest window over threshold", "16 frames, stride 8"):
            with self.subTest(note=note):
                self.assertIn(note, fn)

    def test_the_decision_tone_is_derived_from_the_decision(self):
        """Alert styling must track the actual decision, not be decorative."""
        fn = self._fn("renderTemporalHeadline")
        self.assertIn('decision === "Fight"', fn)
        self.assertIn('decision === "NonFight"', fn)

    def test_the_headline_reads_the_same_fields_as_before(self):
        fn = self._fn("renderTemporalHeadline")
        self.assertIn("summary.final_temporal_decision", fn)
        self.assertIn("summary.temporal_signal_first_frame", fn)
        self.assertIn("peak.fight_probability", fn)
        self.assertIn("data.window_scores", fn)

    def test_no_value_is_invented_when_a_figure_is_missing(self):
        fn = self._fn("renderTemporalHeadline")
        self.assertIn('"Unavailable"', fn)
        self.assertIn('"Did not fire"', fn)

    def test_the_timestamp_still_requires_a_measured_frame_rate(self):
        """Seconds may only appear when fps was measured from the preview."""
        fn = self._fn("renderTemporalHeadline")
        self.assertIn("state.fps", fn)

    def test_view_entry_animation_is_reapplied_rather_than_assumed(self):
        """The section is reused, so the animation must be restarted by hand."""
        fn = self._fn("showView")
        self.assertIn("view-enter", fn)
        self.assertIn("offsetWidth", fn)

    def test_reduced_motion_is_honoured_globally(self):
        css = STYLE.read_text(encoding="utf-8")
        block = css[css.index("@media (prefers-reduced-motion: reduce)"):]
        block = block[:block.index("}\n\n")]
        self.assertIn("animation-duration", block)
        self.assertIn("transition-duration", block)
        self.assertIn("!important", block)

    def test_the_button_system_is_not_overridden_by_container_rules(self):
        """Element-scoped rules outrank .btn-primary, so they must exempt it."""
        css = STYLE.read_text(encoding="utf-8")
        self.assertIn(".report-actions button:not(.btn)", css)
        self.assertIn(".saliency-controls button:not(.btn)", css)


class DesignSystemTests(unittest.TestCase):
    """What the redesign must keep true, independent of taste.

    These do not pin colours or spacing -- those are meant to change. They pin
    the properties that stop the interface from regressing into something that
    needs a network, breaks on a small screen, or animates for people who have
    asked it not to.
    """

    @classmethod
    def setUpClass(cls):
        cls.css = STYLE.read_text(encoding="utf-8")

    def test_no_external_font_or_asset_is_requested(self):
        """The demo machine may have no internet at review time."""
        for forbidden in ("@import", "url(http", "fonts.googleapis", "cdn."):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.css)

    def test_every_colour_comes_from_the_token_set(self):
        """One coherent system, not dozens of unrelated literals."""
        import re
        root = self.css[self.css.index(":root {"):self.css.index("* { box-sizing")]
        declared = set(re.findall(r"^\s*(--[\w-]+):", root, re.M))
        used = set(re.findall(r"var\((--[\w-]+)", self.css))
        self.assertTrue(declared, "no tokens declared")
        # Nothing may reference a token that does not exist, and nothing may be
        # declared and then never used.
        self.assertEqual(used - declared, set(), "token referenced but not declared")
        self.assertEqual(declared - used, set(), "token declared but never used")

    def test_reduced_motion_disables_movement_but_not_state(self):
        block = self.css[self.css.index("@media (prefers-reduced-motion: reduce)"):]
        block = block[:block.index("/* ====")]
        self.assertIn("animation-duration", block)
        self.assertIn("transition-duration", block)
        self.assertIn("!important", block)
        # State must still be visible, so opacity is forced ON rather than off.
        self.assertIn("opacity: 1 !important", block)

    def test_the_light_theme_is_the_default(self):
        """The review-day interface is light; nothing gates that on a class."""
        root = self.css[self.css.index(":root {"):self.css.index("* { box-sizing")]
        self.assertIn("--bg:", root)
        self.assertIn("--surface:", root)
        # The multi-line body rule, not the "html, body { height: 100%; }" one.
        marker = chr(10) + "body {" + chr(10)
        body = self.css[self.css.index(marker):]
        body = body[:body.index("}")]
        self.assertIn("var(--bg)", body)
        self.assertIn("var(--text)", body)

    def test_print_keeps_the_report_complete(self):
        block = self.css[self.css.index("@media print"):]
        self.assertIn("display: block !important", block,
                      "collapsed disclosures must open on paper")
        self.assertIn(".table-scroll", block, "scrolling regions must expand")

    def test_responsive_breakpoints_exist_for_the_verified_widths(self):
        for width in ("1080px", "900px", "700px"):
            with self.subTest(width=width):
                self.assertIn("max-width: " + width, self.css)

    def test_focus_is_always_visible(self):
        """Keyboard users must never lose the caret position."""
        self.assertIn(":focus-visible", self.css)
        self.assertGreaterEqual(self.css.count("outline: 2px solid var(--accent)"), 8)

    def test_the_concurrent_block_is_one_surface(self):
        """Four separate pills would read as four separate jobs."""
        self.assertIn(".pipeline.group-active li.grouped", self.css)
        self.assertIn("groupScan", self.css)


if __name__ == "__main__":
    unittest.main()
