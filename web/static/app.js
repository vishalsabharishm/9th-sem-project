// web/static/app.js
//
// Client-side only: builds the form data for /api/analyze and renders
// whatever the server returns. It never computes a risk level, confidence,
// or evidence value itself -- everything shown here comes straight from the
// JSON the Flask server produced from the real run_demo() pipeline output.

const state = {
  activeTab: "presets",
  selectedPreset: null,
  mode: "replay", // "replay" | "live" -- always sent explicitly, never inferred
  clips: [], // [{clip_key, true_label}]
  presets: [],
  view: "home",
  reached: ["home"],
};

const els = {
  tabBtns: document.querySelectorAll(".tab-btn"),
  tabPanels: document.querySelectorAll(".tab-panel"),
  presetList: document.getElementById("presetList"),
  clipSearch: document.getElementById("clipSearch"),
  clipSearchInfo: document.getElementById("clipSearchInfo"),
  clipDataList: document.getElementById("clipDataList"),
  videoUpload: document.getElementById("videoUpload"),
  uploadClipSearch: document.getElementById("uploadClipSearch"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  status: document.getElementById("status"),
  resultsPanel: document.getElementById("resultsPanel"),
  resultVideo: document.getElementById("resultVideo"),
  videoCaption: document.getElementById("videoCaption"),
  riskBadge: document.getElementById("riskBadge"),
  truthBadge: document.getElementById("truthBadge"),
  fpNotice: document.getElementById("fpNotice"),
  temporalTable: document.getElementById("temporalTable"),
  eventsTable: document.getElementById("eventsTable").querySelector("tbody"),
  windowChart: document.getElementById("windowChart"),
  explanation: document.getElementById("explanation"),
  artifactLinks: document.getElementById("artifactLinks"),
  modeSelect: document.getElementById("modeSelect"),
  modeBanner: document.getElementById("modeBanner"),
  liveNote: document.getElementById("liveNote"),
  fusionPanel: document.getElementById("fusionPanel"),
  uploadHintReplay: document.getElementById("uploadHintReplay"),
  uploadHintLive: document.getElementById("uploadHintLive"),
  timelineProvenance: document.getElementById("timelineProvenance"),
  preflightChips: document.getElementById("preflightChips"),
  preflightDetail: document.getElementById("preflightDetail"),
  stageStrip: document.getElementById("stageStrip"),
  dropZone: document.getElementById("dropZone"),
  uploadMeta: document.getElementById("uploadMeta"),
  emptyHero: document.getElementById("emptyHero"),
  hudState: document.getElementById("hudState"),
  riskCard: document.getElementById("riskCard"),
  spatialPanel: document.getElementById("spatialPanel"),
  reportPreview: document.getElementById("reportPreview"),
  toast: document.getElementById("toast"),
  // Guided-workflow shell
  stepper: document.getElementById("stepper"),
  previewVideo: document.getElementById("previewVideo"),
  previewEmpty: document.getElementById("previewEmpty"),
  summaryList: document.getElementById("summaryList"),
  summaryPath: document.getElementById("summaryPath"),
  processingSub: document.getElementById("processingSub"),
  processingError: document.getElementById("processingError"),
  viewResultsBtn: document.getElementById("viewResultsBtn"),
  resultsSub: document.getElementById("resultsSub"),
  explainWindowList: document.getElementById("explainWindowList"),
  videoFallback: document.getElementById("videoFallback"),
};

// ---------------------------------------------------------------------------
// View router.
//
// Client-side only: one Flask template, six views. The backend routes, request
// shapes and response fields are untouched -- this changes how the examiner
// moves through the application, not what the application computes.
// ---------------------------------------------------------------------------
const VIEW_ORDER = ["home", "setup", "processing", "results", "explain", "report"];

function showView(name) {
  if (!VIEW_ORDER.includes(name)) return;
  state.view = name;
  document.querySelectorAll(".view").forEach(function (section) {
    section.classList.toggle("hidden", section.dataset.view !== name);
  });
  document.querySelectorAll("#stepper .step").forEach(function (btn) {
    const target = btn.dataset.goto;
    btn.classList.toggle("active", target === name);
    btn.classList.toggle("done", state.reached.indexOf(target) !== -1 && target !== name);
  });
  // Pause any playing video when leaving a view so audio never follows the
  // examiner around the application.
  document.querySelectorAll("video").forEach(function (v) {
    const owner = v.closest(".view");
    if (owner && owner.dataset.view !== name && !v.paused) v.pause();
  });
  window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });
}

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function unlockStep(name) {
  if (state.reached.indexOf(name) === -1) state.reached.push(name);
  document.querySelectorAll("#stepper .step").forEach(function (btn) {
    if (state.reached.indexOf(btn.dataset.goto) !== -1) btn.disabled = false;
  });
}

function lockStepsAfter(name) {
  const cut = VIEW_ORDER.indexOf(name);
  state.reached = state.reached.filter(function (v) { return VIEW_ORDER.indexOf(v) <= cut; });
  document.querySelectorAll("#stepper .step").forEach(function (btn) {
    btn.disabled = state.reached.indexOf(btn.dataset.goto) === -1;
    if (btn.disabled) btn.classList.remove("done", "active");
  });
}

// Any element carrying data-goto navigates, including the stepper itself.
document.addEventListener("click", function (event) {
  const target = event.target.closest("[data-goto]");
  if (!target || target.disabled) return;
  showView(target.dataset.goto);
});

const brandHome = document.getElementById("brandHome");
if (brandHome) brandHome.addEventListener("click", function () { showView("home"); });

const startNew = document.getElementById("startNewAnalysis");
if (startNew) startNew.addEventListener("click", function () {
  unlockStep("setup");
  showView("setup");
});

const startReplay = document.getElementById("startReplayDemo");
if (startReplay) startReplay.addEventListener("click", function () {
  // Pre-arms the deterministic research demo, but never starts it: analysis
  // begins only when the examiner presses Start Analysis.
  setMode("replay");
  const radio = document.querySelector('.mode-option[data-mode="replay"] input');
  if (radio) radio.checked = true;
  switchTab("presets");
  unlockStep("setup");
  showView("setup");
  showToast("Replay demo armed -- pick a built-in clip, then Start Analysis.");
});

if (els.viewResultsBtn) els.viewResultsBtn.addEventListener("click", function () {
  showView("results");
});

// The frozen per-window operating point, used for the chart's threshold line.
// decisionFor() below remains the single place the comparison is written.
const FROZEN_THRESHOLD = 0.14;

// Headroom so a p = 1.0 bar does not touch the top edge of the chart.
const BAR_SCALE = 0.86;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

let toastTimer = null;
function showToast(message, isError) {
  if (!els.toast) return;
  els.toast.textContent = message;
  els.toast.classList.toggle("error", Boolean(isError));
  els.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove("show"), 3600);
}

// The backend returns a final result, not incremental progress. So the strip
// shows "processing" for the whole chain during a run and is marked complete
// only from fields the response genuinely contains -- no fabricated timing.
function setStages(state) {
  if (!els.stageStrip) return;
  els.stageStrip.querySelectorAll("li").forEach((li) => {
    if (state === null) li.removeAttribute("data-state");
    else li.dataset.state = state;
  });
}

function setStagesFromResult(data) {
  if (!els.stageStrip) return;
  const summary = data.summary || {};
  const feature = data.spatial_fusion_feature || null;
  const fusion = data.fusion || null;
  const known = {
    video: summary.frames_processed > 0 ? "complete" : "warning",
    yolo: "complete",
    tracking: "complete",
    // An undefined spatial feature is a warning, not a failure: it means no
    // usable track pair existed, which the panel below states plainly.
    spatial: feature && feature.defined ? "complete" : "warning",
    temporal: (data.window_scores || []).length ? "complete" : "warning",
    fusion: fusion && fusion.available ? "complete" : "warning",
    explanation: "complete",
  };
  els.stageStrip.querySelectorAll("li").forEach((li) => {
    li.dataset.state = known[li.dataset.stage] || "complete";
  });
}

function setStatus(text, isError) {
  els.status.textContent = text || "";
  els.status.classList.toggle("error", Boolean(isError));
  if (isError && text) showToast(text, true);
}

function updateAnalyzeEnabled() {
  const hasFile = els.videoUpload.files && els.videoUpload.files.length > 0;
  if (state.activeTab === "presets") {
    els.analyzeBtn.disabled = !state.selectedPreset;
  } else if (state.activeTab === "browse") {
    // Live mode can score any dataset clip; replay needs one with a CSV row.
    els.analyzeBtn.disabled =
      state.mode === "live"
        ? !els.clipSearch.value.trim()
        : !isKnownClip(els.clipSearch.value.trim());
  } else {
    // The clip_key requirement exists only because replay has to find a
    // precomputed score. Live reads no CSV, so a file is all it needs.
    els.analyzeBtn.disabled =
      state.mode === "live"
        ? !hasFile
        : !(hasFile && isKnownClip(els.uploadClipSearch.value.trim()));
  }
}

// ---------------------------------------------------------------------------
// Setup summary and video preview.
//
// Describes only what has actually been chosen. The "expected processing path"
// is a statement about which code path will run, not a prediction of results.
// ---------------------------------------------------------------------------
function currentSelection() {
  const file = els.videoUpload.files && els.videoUpload.files[0];
  if (state.activeTab === "presets") {
    const preset = state.selectedPreset;
    const known = state.presets.find(function (x) { return x.id === preset; });
    return preset
      ? { label: "Built-in demo", name: preset, detail: known ? known.clip_key : "",
          truth: known ? known.true_label : null, url: null }
      : null;
  }
  if (state.activeTab === "browse") {
    const key = els.clipSearch.value.trim();
    if (!key) return null;
    const known = state.clips.find(function (c) { return c.clip_key === key; });
    return { label: "Dataset clip", name: key.split("/").pop(), detail: key,
             truth: known ? known.true_label : null, url: null };
  }
  if (file) {
    return { label: "Uploaded video", name: file.name,
             detail: formatBytes(file.size) + (file.type ? " · " + file.type : ""),
             truth: null, url: URL.createObjectURL(file) };
  }
  return null;
}

let previewUrl = null;

function renderSummary() {
  const pick = currentSelection();
  const live = state.mode === "live";

  // Preview: a local object URL for an upload, nothing for a server-side clip
  // (the dataset video is not exposed for streaming before analysis).
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
  if (pick && pick.url) {
    previewUrl = pick.url;
    els.previewVideo.src = previewUrl;
    els.previewVideo.classList.remove("hidden");
    els.previewEmpty.classList.add("hidden");
  } else {
    els.previewVideo.removeAttribute("src");
    els.previewVideo.load();
    els.previewVideo.classList.add("hidden");
    els.previewEmpty.classList.remove("hidden");
    els.previewEmpty.querySelector("span").textContent = pick
      ? "Preview available after analysis for dataset clips"
      : "No video selected yet";
  }

  const rows = [
    ["Mode", live ? "LIVE MODEL -- actual video inference"
                  : "REPLAY -- precomputed research evidence"],
    ["Video", pick ? pick.name : "Not selected"],
    ["Source", pick ? pick.label : "—"],
  ];
  if (pick && pick.detail) rows.push(["Detail", pick.detail]);
  if (pick && pick.truth) rows.push(["Ground truth", pick.truth]);

  els.summaryList.innerHTML = rows.map(function (r) {
    return "<dt>" + escapeHtml(r[0]) + "</dt><dd>" + escapeHtml(r[1]) + "</dd>";
  }).join("");

  els.summaryPath.innerHTML = live
    ? "Expected path: decode &rarr; YOLO &rarr; tracking &rarr; spatial feature &rarr; " +
      "<strong>R3D-18 forward pass per 16-frame window</strong> &rarr; frozen aggregation " +
      "&rarr; configured fusion &rarr; risk. Offline, not real time."
    : "Expected path: decode &rarr; YOLO &rarr; tracking &rarr; spatial feature &rarr; " +
      "<strong>replayed window probabilities from the committed CSV</strong> &rarr; frozen " +
      "aggregation &rarr; configured fusion &rarr; risk. No temporal model runs.";
}

function setMode(mode) {
  state.mode = mode;
  els.modeSelect.querySelectorAll(".mode-option").forEach((el) =>
    el.classList.toggle("selected", el.dataset.mode === mode)
  );
  const live = mode === "live";
  els.uploadHintReplay.classList.toggle("hidden", live);
  els.uploadHintLive.classList.toggle("hidden", !live);
  els.uploadClipSearch.classList.toggle("hidden", live);
  updateAnalyzeEnabled();
  renderSummary();
}

els.modeSelect.addEventListener("change", (event) => {
  if (event.target.name === "mode") setMode(event.target.value);
});

function isKnownClip(clipKey) {
  return state.clips.some((c) => c.clip_key === clipKey);
}

function switchTab(tab) {
  state.activeTab = tab;
  els.tabBtns.forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  els.tabPanels.forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== tab));
  updateAnalyzeEnabled();
  renderSummary();
}

els.tabBtns.forEach((btn) => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));

async function loadPresets() {
  const res = await fetch("/api/presets");
  const presets = await res.json();
  state.presets = presets;
  els.presetList.innerHTML = "";
  presets.forEach((p) => {
    const card = document.createElement("div");
    card.className = "preset-card";
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.innerHTML =
      `<span class="pc-name">${escapeHtml(p.id)}</span>` +
      `<span class="pc-meta">${escapeHtml(p.clip_key)} &middot; ground truth: ` +
      `${escapeHtml(p.true_label)}</span>`;
    const choose = () => {
      state.selectedPreset = p.id;
      document.querySelectorAll(".preset-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      updateAnalyzeEnabled();
      renderSummary();
    };
    card.addEventListener("click", choose);
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(); }
    });
    els.presetList.appendChild(card);
  });
}

async function loadClips() {
  const res = await fetch("/api/clips");
  state.clips = await res.json();
  els.clipDataList.innerHTML = "";
  state.clips.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.clip_key;
    opt.label = `${c.clip_key} (${c.true_label})`;
    els.clipDataList.appendChild(opt);
  });
}

// ---------------------------------------------------------------------------
// Preflight. Reported, never interpreted: the three tiers come straight from
// GET /api/preflight, and an unavailable capability is shown as unavailable
// with the server's own remedy text rather than glossed over.
// ---------------------------------------------------------------------------
const PREFLIGHT_TIERS = [
  ["replay_ready", "Replay", "replay"],
  ["live_inference_ready", "Live Model", "live_inference"],
  ["saliency_ready", "Saliency", "saliency"],
];

async function loadPreflight() {
  try {
    const result = await (await fetch("/api/preflight")).json();
    const chips = PREFLIGHT_TIERS.map(([key, label]) => {
      const ok = Boolean(result[key]);
      return `<span class="chip ${ok ? "chip-ok" : "chip-warn"}">` +
             `${label} ${ok ? "ready" : "unavailable"}</span>`;
    }).join("");
    els.preflightChips.innerHTML = chips;

    els.preflightDetail.innerHTML = PREFLIGHT_TIERS.map(([key, label, tier]) => {
      const ok = Boolean(result[key]);
      let why = "";
      if (!ok) {
        const failed = (result.checks || []).filter((c) => c.tier === tier && !c.ok);
        why = failed.map((c) => c.remedy || c.detail).filter(Boolean)[0] || "";
      }
      return `<div class="pf-row ${ok ? "ok" : "bad"}">` +
             `<span class="pf-icon">${ok ? "✓" : "⚠"}</span>` +
             `<span><span class="pf-name">${label}</span> ` +
             `${ok ? "ready" : "not available"}` +
             (why ? `<span class="pf-why">${escapeHtml(why)}</span>` : "") +
             `</span></div>`;
    }).join("");
  } catch (err) {
    els.preflightChips.innerHTML = '<span class="chip chip-warn">Status unavailable</span>';
    els.preflightDetail.innerHTML =
      '<div class="pf-row bad"><span class="pf-icon">⚠</span>' +
      "<span>Could not reach the preflight endpoint.</span></div>";
  }
}

// ---------------------------------------------------------------------------
// Drag-and-drop upload. The hidden file input remains the source of truth, so
// nothing downstream has to know how the file arrived.
// ---------------------------------------------------------------------------
function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes, i = 0;
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1; }
  return `${value.toFixed(value < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

function renderUploadMeta() {
  const file = els.videoUpload.files && els.videoUpload.files[0];
  if (!file) {
    els.uploadMeta.classList.add("hidden");
    els.uploadMeta.innerHTML = "";
    els.dropZone.classList.remove("hidden");
    return;
  }
  els.dropZone.classList.add("hidden");
  els.uploadMeta.classList.remove("hidden");
  els.uploadMeta.innerHTML =
    `<span><span class="um-name">${escapeHtml(file.name)}</span>` +
    `<br><span class="um-size">${formatBytes(file.size)}` +
    (file.type ? ` &middot; ${escapeHtml(file.type)}` : "") + `</span></span>` +
    '<button type="button" id="uploadClear" aria-label="Remove selected video">Remove</button>';
  document.getElementById("uploadClear").addEventListener("click", () => {
    els.videoUpload.value = "";
    renderUploadMeta();
    updateAnalyzeEnabled();
    renderSummary();
  });
}

if (els.dropZone) {
  els.dropZone.addEventListener("click", () => els.videoUpload.click());
  els.dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.videoUpload.click(); }
  });
  ["dragenter", "dragover"].forEach((type) =>
    els.dropZone.addEventListener(type, (e) => {
      e.preventDefault();
      els.dropZone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    els.dropZone.addEventListener(type, (e) => {
      e.preventDefault();
      els.dropZone.classList.remove("dragover");
    })
  );
  els.dropZone.addEventListener("drop", (e) => {
    const dropped = e.dataTransfer && e.dataTransfer.files;
    if (dropped && dropped.length) {
      els.videoUpload.files = dropped;
      els.videoUpload.dispatchEvent(new Event("change"));
    }
  });
}

els.clipSearch.addEventListener("input", () => {
  const match = state.clips.find((c) => c.clip_key === els.clipSearch.value.trim());
  els.clipSearchInfo.textContent = match ? `Ground truth: ${match.true_label}` : "";
  updateAnalyzeEnabled();
  renderSummary();
});

els.uploadClipSearch.addEventListener("input", updateAnalyzeEnabled);

els.videoUpload.addEventListener("change", () => {
  // Convenience only: suggest a clip_key whose filename matches the
  // uploaded file's name. The user can still change it -- this never
  // silently decides anything on its own.
  const file = els.videoUpload.files[0];
  if (file && !els.uploadClipSearch.value) {
    const match = state.clips.find((c) => c.clip_key.endsWith("/" + file.name));
    if (match) els.uploadClipSearch.value = match.clip_key;
  }
  renderUploadMeta();
  updateAnalyzeEnabled();
  renderSummary();
});

els.analyzeBtn.addEventListener("click", runAnalyze);

async function runAnalyze() {
  const formData = new FormData();
  // Always explicit. The server rejects an unknown mode rather than guessing,
  // and never falls back from live to replay, so this value decides which
  // pipeline actually runs.
  formData.append("mode", state.mode);
  if (state.activeTab === "presets") {
    formData.append("preset", state.selectedPreset);
  } else if (state.activeTab === "browse") {
    formData.append("clip_key", els.clipSearch.value.trim());
  } else {
    // Live needs no clip_key: it reads no CSV and so has nothing to look up.
    if (state.mode !== "live") {
      formData.append("clip_key", els.uploadClipSearch.value.trim());
    }
    formData.append("video", els.videoUpload.files[0]);
  }

  // Move to the dedicated processing view. Results become reachable only when
  // the backend actually returns them.
  lockStepsAfter("processing");
  unlockStep("processing");
  showView("processing");
  els.processingError.classList.add("hidden");
  els.processingError.textContent = "";
  els.viewResultsBtn.classList.add("hidden");
  els.processingSub.textContent = state.mode === "live"
    ? "Running R3D-18 inference locally on CPU. This is not real time."
    : "Replaying committed window probabilities. No temporal model runs.";

  els.analyzeBtn.disabled = true;
  els.analyzeBtn.classList.add("busy");
  // Indeterminate: the backend reports a final result, not progress, so the
  // whole chain reads "processing" rather than faking a percentage.
  setStages("processing");
  setStatus(
    state.mode === "live"
      ? "LIVE MODEL: YOLO detection + tracking + spatial feature + an R3D-18 forward pass per 16-frame window. Offline, not real time -- expect roughly 0.5s per window plus detection, so a few minutes for a long video."
      : "REPLAY: YOLO detection + tracking + Phase-4 rules + replayed temporal scores + risk assessment... this can take 30-90s on CPU.",
    false
  );
  els.resultsPanel.classList.add("hidden");

  // Drop the previous run's state before starting a new one. Without this a
  // failed analyze leaves the OLD clip armed: Explain would explain the
  // previous clip and Export would write the previous clip's report, while the
  // screen shows an error for the new one.
  if (window.__clearAnalysis) window.__clearAnalysis();
  window.__lastClipKey = null;
  selectedWindow = null;
  const staleExplain = document.getElementById("explainBtn");
  if (staleExplain) staleExplain.disabled = true;
  const staleSelection = document.getElementById("selectedWindow");
  if (staleSelection) staleSelection.textContent = "Select a window in the timeline above.";
  const staleSaliency = document.getElementById("saliencyResult");
  if (staleSaliency) staleSaliency.innerHTML = "";

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      const message = data.error || `Request failed (HTTP ${res.status}).`;
      setStatus(message, true);
      setStages("warning");
      showProcessingError(message);
      return;
    }
    setStatus("Analysis complete.", false);
    showToast("Analysis complete.");
    renderResult(data);
    unlockStep("results");
    unlockStep("explain");
    unlockStep("report");
    els.viewResultsBtn.classList.remove("hidden");
    els.processingSub.textContent = "Analysis complete.";
    // Hand the examiner straight to the results rather than leaving them on a
    // finished progress screen.
    setTimeout(function () {
      if (state.view === "processing") showView("results");
    }, 700);
  } catch (err) {
    const message = `Request failed: ${err}`;
    setStatus(message, true);
    setStages("warning");
    showProcessingError(message);
  } finally {
    els.analyzeBtn.classList.remove("busy");
    updateAnalyzeEnabled();
  }
}

function showProcessingError(message) {
  els.processingError.innerHTML =
    "<strong>Analysis did not complete.</strong><p>" + escapeHtml(message) + "</p>" +
    '<button class="btn btn-ghost btn-sm" data-goto="setup" type="button">Back to setup</button>';
  els.processingError.classList.remove("hidden");
  els.processingSub.textContent = "The run stopped before producing a result.";
}

function renderResult(data) {
  els.resultsPanel.classList.remove("hidden");

  // Remember which clip was analysed so the on-demand saliency request knows
  // what to explain. Set here rather than at submit time so it always reflects
  // the clip the displayed results actually came from.
  window.__lastClipKey = data.summary.clip_key;
  if (window.__recordAnalysis) window.__recordAnalysis(data);

  if (els.emptyHero) els.emptyHero.classList.add("hidden");
  setStagesFromResult(data);
  renderModeBanner(data);
  renderFusion(data.fusion, data.spatial_fusion_feature);
  renderSpatialEvidence(data);
  renderHud(data);
  renderReportPreview(data);

  const summary = data.summary || {};
  els.resultsSub.textContent =
    (summary.clip_key || "") + " · " + (summary.frames_processed || 0) +
    " frames · " + (data.window_scores || []).length + " scored window(s)";

  // The timeline caption must describe THIS run. Saying "replayed" above a
  // live timeline would misdescribe the one number an examiner cares most
  // about where it came from.
  if (els.timelineProvenance) {
    els.timelineProvenance.innerHTML =
      data.mode === "live"
        ? "Each probability below was produced by an <strong>R3D-18 forward " +
          "pass on the frames of this video</strong> during this run; no " +
          "precomputed score was read."
        : "These probabilities are <strong>replayed from the committed " +
          "scores</strong>; no temporal model ran in this process.";
  }

  // The severity badge is the most prominent number on screen and is a
  // CONFIGURED constant, not a calibrated estimate. Render its status beside it
  // so a viewer never sees "High" unqualified.
  const statusEl = document.getElementById("riskStatus");
  if (statusEl && data.risk_status) {
    statusEl.textContent = data.risk_status.status_text;
    statusEl.title =
      "Provenance: " + data.risk_status.provenance +
      " | risk_level_is_validated: " + data.risk_status.risk_level_is_validated;
  }

  // ---------------------------------------------------------------------
  // Annotated video.
  //
  // run_demo writes it with OpenCV's "mp4v" fourcc, i.e. MPEG-4 Part 2, which
  // Chrome and Firefox do not decode -- and this machine has no H.264 encoder
  // available to OpenCV, so the backend cannot currently produce a
  // browser-native codec. Rather than leave a dead player on screen, the
  // failure is caught and stated plainly with the file offered for download.
  // ---------------------------------------------------------------------
  els.videoFallback.classList.add("hidden");
  els.resultVideo.classList.remove("hidden");
  els.resultVideo.onerror = function () {
    els.resultVideo.classList.add("hidden");
    els.videoFallback.classList.remove("hidden");
    els.videoFallback.innerHTML =
      "<strong>This browser cannot play the annotated video.</strong>" +
      "<p>The file was written successfully and contains every annotated " +
      "frame, but it uses the MPEG-4 Part 2 codec, which browsers do not " +
      "decode. Open or download it to view the detections and track IDs.</p>" +
      '<a class="btn btn-ghost btn-sm" href="' + data.video_url +
      '" target="_blank" rel="noopener">Open annotated video</a>';
  };
  els.resultVideo.src = data.video_url + "?t=" + Date.now();
  els.videoCaption.textContent = `${data.summary.clip_key} -- ${data.summary.frames_processed} frames processed`;

  els.riskBadge.textContent = data.overall_risk;
  els.riskBadge.className = `badge risk-${data.overall_risk}`;
  if (els.riskCard) els.riskCard.classList.toggle("is-high", data.overall_risk === "High");
  els.truthBadge.textContent = `Ground truth (dataset label, not used by the pipeline): ${data.summary.ground_truth_label}`;

  const isFalsePositive =
    data.summary.temporal_violence_signal_fired && data.summary.ground_truth_label === "NonFight";
  els.fpNotice.classList.toggle("hidden", !isFalsePositive);

  renderTemporalTable(data.temporal_signal, data.summary);
  renderEventsTable(data.event_summary);
  renderWindowChart(data.window_scores, data.summary.temporal_signal_first_frame);
  els.explanation.innerHTML = renderMarkdown(data.explanation_text);

  els.artifactLinks.innerHTML = `
    <a href="${data.video_url}" target="_blank" rel="noopener">Annotated video</a>
    <a href="${data.risk_json_url}" target="_blank" rel="noopener">Risk assessment JSON</a>
    <a href="${data.explanation_url}" target="_blank" rel="noopener">Explanation report (.md)</a>
  `;
}

function renderTemporalTable(temporalSignal, summary) {
  if (!temporalSignal) {
    els.temporalTable.innerHTML = `<tr><td>Temporal Violence Signal</td><td>Did not fire for this clip.</td></tr>`;
    return;
  }
  const ev = temporalSignal.evidence_parsed || {};
  const rows = [
    ["Fired at frame", summary.temporal_signal_first_frame],
    ["Confidence", Number(temporalSignal.confidence).toFixed(6)],
    ["Confidence provenance", temporalSignal.confidence_provenance],
    ["Aggregation rule", ev.aggregation_rule],
    ["Aggregation params", ev.aggregation_params],
    ["Peak window index", ev.peak_window_index],
    ["Peak window frames", ev.peak_window_frames],
    ["Peak fight probability", ev.peak_fight_probability],
    ["Number of windows", ev.n_windows],
    ["Checkpoint source", ev.checkpoint_source],
  ];
  els.temporalTable.innerHTML = rows
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
    .join("");
}

function renderEventsTable(eventSummary) {
  els.eventsTable.innerHTML = (eventSummary || [])
    .map(
      (row) =>
        `<tr><td>${row.event_type}</td><td>${row.risk_level}</td><td>${row.occurrences}</td></tr>`
    )
    .join("");
}

// The temporal branch scores overlapping 16-frame windows, not the whole clip.
// Making each scored window selectable is what turns that from a sentence in a
// report into something an examiner can see and poke at.
//
// Windows are NOT invented here: every bar comes from a scored window in the
// analyze response, so a window offered for saliency is always one for which
// source frames can be reconstructed.
let selectedWindow = null;

function decisionFor(probability) {
  // The frozen rule is max-over-windows >= 0.14. Per window this shows whether
  // that window alone would clear the threshold; the clip decision remains the
  // maximum, and is reported separately.
  return probability >= 0.14 ? "Fight" : "NonFight";
}

// The Explain view needs its own list of the SAME scored windows -- ids must
// stay unique, so the timeline lives on Results and this is a second control
// bound to the identical data and the identical selectWindow().
function renderExplainPicker(windowScores) {
  if (!els.explainWindowList) return;
  const windows = windowScores || [];
  if (!windows.length) {
    els.explainWindowList.innerHTML =
      '<p class="muted">No temporal windows are available to explain. ' +
      "No 16-frame window completed for this video.</p>";
    return;
  }
  const peak = windows.reduce(function (a, b) {
    return b.fight_probability > a.fight_probability ? b : a;
  });
  els.explainWindowList.innerHTML = windows.map(function (w) {
    const decision = decisionFor(w.fight_probability);
    return '<button class="wp-item" type="button" data-index="' + w.window_index + '">' +
      '<span class="wp-idx">#' + w.window_index +
      (w.window_index === peak.window_index ? ' <em>peak</em>' : "") + "</span>" +
      '<span class="wp-frames">frames ' + w.first_frame + "–" + w.last_frame + "</span>" +
      '<span class="wp-prob">' + w.fight_probability.toFixed(4) + "</span>" +
      '<span class="wp-dec' + (decision === "Fight" ? " is-fight" : "") + '">' + decision + "</span>" +
      "</button>";
  }).join("");
  els.explainWindowList.querySelectorAll(".wp-item").forEach(function (btn) {
    btn.addEventListener("click", function () {
      const w = windows.find(function (x) {
        return x.window_index === Number(btn.dataset.index);
      });
      if (w) selectWindow(w);
    });
  });
}

function selectWindow(w) {
  selectedWindow = w;
  const label = document.getElementById("selectedWindow");
  if (label) {
    label.textContent =
      `Window ${w.window_index} - frames ${w.first_frame}-${w.last_frame}` +
      ` (p = ${w.fight_probability.toFixed(4)})`;
  }
  const button = document.getElementById("explainBtn");
  if (button) button.disabled = false;

  document.querySelectorAll(".window-bar").forEach((el) => {
    el.classList.toggle("selected", Number(el.dataset.index) === w.window_index);
  });
  document.querySelectorAll("#windowTable tbody tr").forEach((row) => {
    row.classList.toggle("selected", Number(row.dataset.index) === w.window_index);
  });
  document.querySelectorAll("#explainWindowList .wp-item").forEach((item) => {
    item.classList.toggle("selected", Number(item.dataset.index) === w.window_index);
  });
}

function renderWindowChart(windowScores, firedFrame) {
  els.windowChart.innerHTML = "";
  const table = document.querySelector("#windowTable tbody");
  if (table) table.innerHTML = "";
  selectedWindow = null;
  const button = document.getElementById("explainBtn");
  if (button) button.disabled = true;

  const windows = windowScores || [];
  const label = document.getElementById("selectedWindow");
  if (!windows.length) {
    // An explicit unavailable state, never a silently empty strip.
    els.windowChart.innerHTML =
      '<p class="muted">No temporal windows available &mdash; no 16-frame ' +
      "window completed for this video, so the temporal model never ran.</p>";
    if (label) {
      label.textContent =
        "No scored windows, so saliency unavailable. Select a completed " +
        "temporal window to generate an explanation.";
    }
    renderExplainPicker([]);
    return;
  }

  // The frozen operating point, drawn so the per-window decisions are legible
  // as a geometry rather than only as a column of text.
  const threshold = document.createElement("div");
  threshold.className = "chart-threshold";
  threshold.innerHTML = `<span>threshold ${FROZEN_THRESHOLD}</span>`;
  els.windowChart.appendChild(threshold);

  const tip = document.createElement("div");
  tip.className = "chart-tip";
  els.windowChart.appendChild(tip);

  const peak = windows.reduce((a, b) =>
    b.fight_probability > a.fight_probability ? b : a);

  windows.forEach((w) => {
    const bar = document.createElement("div");
    const fired =
      firedFrame !== null && firedFrame !== undefined &&
      w.last_frame >= firedFrame && w.first_frame <= firedFrame;
    const decision = decisionFor(w.fight_probability);
    bar.className = "window-bar" +
      (decision === "Fight" ? " fired" : "") +
      (w.window_index === peak.window_index ? " peak" : "");
    bar.style.height = `${Math.max(6, w.fight_probability * 100 * BAR_SCALE)}%`;
    bar.dataset.index = w.window_index;
    bar.tabIndex = 0;
    bar.setAttribute("role", "button");
    const description =
      `window ${w.window_index}: frames [${w.first_frame},${w.last_frame}] -> ` +
      `${w.fight_probability.toFixed(4)}` +
      (fired ? " (first alarm falls in this window)" : "") +
      " - click to select for saliency";
    bar.setAttribute("aria-label", description);

    const showTip = () => {
      tip.innerHTML =
        `<b>Window ${w.window_index}</b>${w.window_index === peak.window_index ? " (peak)" : ""}<br>` +
        `frames ${w.first_frame}–${w.last_frame}<br>` +
        `p = ${w.fight_probability.toFixed(4)}<br>` +
        `<span class="${decision === "Fight" ? "tip-fight" : ""}">${decision}</span>` +
        (fired ? "<br>first alarm" : "");
      tip.classList.add("show");
      const chartBox = els.windowChart.getBoundingClientRect();
      const barBox = bar.getBoundingClientRect();
      const left = Math.min(
        Math.max(barBox.left - chartBox.left + barBox.width / 2 - tip.offsetWidth / 2, 4),
        chartBox.width - tip.offsetWidth - 4
      );
      tip.style.left = `${left}px`;
      tip.style.top = "6px";
    };
    const hideTip = () => tip.classList.remove("show");

    bar.addEventListener("mouseenter", showTip);
    bar.addEventListener("focus", showTip);
    bar.addEventListener("mouseleave", hideTip);
    bar.addEventListener("blur", hideTip);
    bar.addEventListener("click", () => selectWindow(w));
    bar.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectWindow(w); }
    });
    els.windowChart.appendChild(bar);

    if (table) {
      const row = document.createElement("tr");
      row.dataset.index = w.window_index;
      row.tabIndex = 0;
      row.innerHTML =
        `<td>${w.window_index}</td>` +
        `<td>${w.first_frame}–${w.last_frame}</td>` +
        `<td>${w.fight_probability.toFixed(4)}</td>` +
        `<td class="${decision === "Fight" ? "is-fight" : ""}">${decision}</td>`;
      row.addEventListener("click", () => selectWindow(w));
      row.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectWindow(w); }
      });
      table.appendChild(row);
    }
  });

  // Position the threshold line against the SAME basis the bars use: a bar's
  // percentage height resolves against the flex content box, so measuring it
  // is the only way the line lands exactly where a bar of p = 0.14 would end.
  // Computed after the bars are in the DOM so clientHeight is real.
  const style = getComputedStyle(els.windowChart);
  const padBottom = parseFloat(style.paddingBottom) || 0;
  const padTop = parseFloat(style.paddingTop) || 0;
  const contentHeight = els.windowChart.clientHeight - padTop - padBottom;
  threshold.style.bottom = `${padBottom + FROZEN_THRESHOLD * BAR_SCALE * contentHeight}px`;

  renderExplainPicker(windows);

  // Default to the peak window: the one the frozen max rule actually used.
  selectWindow(peak);
}

// Minimal, dependency-free renderer for the fixed structure
// write_explanation_report() in tools/run_demo.py produces: #/##/### headers,
// **bold**, and pipe tables. Not a general-purpose Markdown engine.
function renderMarkdown(text) {
  const lines = text.split("\n");
  let html = "";
  let inTable = false;
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      html += `<p>${inline(paragraph.join(" "))}</p>`;
      paragraph = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed.startsWith("|")) {
      if (!inTable) {
        flushParagraph();
        html += "<table>";
        inTable = true;
      }
      const cells = trimmed.split("|").slice(1, -1).map((c) => c.trim());
      const isSeparator = cells.every((c) => /^:?-+:?$/.test(c));
      if (!isSeparator) {
        const nextLine = lines[i + 1] ? lines[i + 1].trim() : "";
        const nextCells = nextLine.startsWith("|") ? nextLine.split("|").slice(1, -1).map((c) => c.trim()) : [];
        const nextIsHeaderSeparator = nextCells.length > 0 && nextCells.every((c) => /^:?-+:?$/.test(c));
        const tag = nextIsHeaderSeparator ? "th" : "td";
        html += "<tr>" + cells.map((c) => `<${tag}>${inline(c)}</${tag}>`).join("") + "</tr>";
      }
      continue;
    } else if (inTable) {
      html += "</table>";
      inTable = false;
    }

    if (trimmed.startsWith("### ")) {
      flushParagraph();
      html += `<h3>${inline(trimmed.slice(4))}</h3>`;
    } else if (trimmed.startsWith("## ")) {
      flushParagraph();
      html += `<h2>${inline(trimmed.slice(3))}</h2>`;
    } else if (trimmed.startsWith("# ")) {
      flushParagraph();
      html += `<h1>${inline(trimmed.slice(2))}</h1>`;
    } else if (trimmed.startsWith("- ")) {
      flushParagraph();
      html += `<div>&bull; ${inline(trimmed.slice(2))}</div>`;
    } else if (trimmed === "") {
      flushParagraph();
    } else {
      paragraph.push(trimmed);
    }
  }
  if (inTable) html += "</table>";
  flushParagraph();
  return html;
}

function inline(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
}

loadPresets();
loadClips();
loadPreflight();
showView("home");
renderSummary();

// ---------------------------------------------------------------------------
// Gradient-based temporal saliency, requested explicitly for ONE window.
// Never wired into the analyze path: Grad-CAM needs a backward pass and would
// roughly double demo cost if run for all 17 windows of every clip.
// ---------------------------------------------------------------------------
(function () {
  const button = document.getElementById("explainBtn");
  if (!button) return;

  button.addEventListener("click", async () => {
    const target = document.getElementById("saliencyResult");
    const clipKey = window.__lastClipKey;
    if (!clipKey) {
      target.innerHTML = '<p class="error">Run an analysis first, then explain one of its windows.</p>';
      return;
    }
    if (!selectedWindow) {
      target.innerHTML =
        '<p class="error">Select a window in the timeline first.</p>';
      return;
    }
    const firstFrame = selectedWindow.first_frame;
    button.disabled = true;
    target.innerHTML = "<p>Computing saliency (one forward + one backward pass)...</p>";
    try {
      const response = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clip_key: clipKey, first_frame: firstFrame }),
      });
      const data = await response.json();
      // Record BEFORE the failure branch. A failed attempt must reach the
      // report as a failure with its reason; recording only on success would
      // make the report say "not requested" while the screen says
      // "unavailable", which is exactly the inconsistency the report's
      // requested/available split exists to prevent.
      if (window.__recordSaliency) window.__recordSaliency(data);
      if (!data.available) {
        // An explicit reason, never a blank panel.
        target.innerHTML =
          '<p class="error"><strong>Saliency unavailable</strong> (' +
          (data.reason || "unknown") + "): " + (data.detail || "") + "</p>";
        return;
      }
      const ev = data.temporal_evidence;
      const sal = data.saliency;
      target.innerHTML =
        '<div class="saliency-grid">' +
        "<div><h4>Temporal evidence</h4><ul>" +
        "<li>Fight probability: <strong>" + (ev.fight_probability * 100).toFixed(1) +
        "%</strong> <span class=\"prov\">(measured model probability)</span></li>" +
        "<li>Decision: <strong>" + ev.decision + "</strong> at threshold " +
        ev.decision_threshold + "</li>" +
        "<li>Window: frames <strong>" + ev.window_first_frame + "-" +
        ev.window_last_frame + "</strong></li>" +
        "<li>Model provenance: " + ev.model_provenance + "</li>" +
        "</ul></div>" +
        "<div><h4>Saliency</h4>" +
        (sal.peak_slice_png_base64
          ? '<img class="saliency-img" alt="temporal saliency for the peak frame" src="data:image/png;base64,' +
            sal.peak_slice_png_base64 + '">'
          : "<p>(no image)</p>") +
        "<ul><li>Peak frame: <strong>" + sal.peak_frame + "</strong></li>" +
        "<li>Displayed slices: " + sal.displayed_temporal_slices +
        ", from <strong>" + sal.raw_temporal_positions +
        "</strong> resolved temporal positions</li>" +
        '<li class="caveat">' + sal.temporal_resolution_caveat + "</li>" +
        "<li>Faithfulness tested: <strong>" + sal.faithfulness_tested + "</strong></li>" +
        "</ul></div></div>" +
        '<p class="muted">' + data.not_real_time + "</p>";
    } catch (err) {
      target.innerHTML = '<p class="error">' + err + "</p>";
    } finally {
      button.disabled = false;
    }
  });
})();

// ---------------------------------------------------------------------------
// Incident report export.
//
// Sends back the analysis payload the client already holds rather than asking
// the server to re-run anything, so the report always describes the run on
// screen. Saliency is attached only if the examiner actually computed it.
// ---------------------------------------------------------------------------
(function () {
  let lastAnalysis = null;
  let lastSaliency = null;

  window.__recordAnalysis = (data) => { lastAnalysis = data; lastSaliency = null; };
  window.__clearAnalysis = () => { lastAnalysis = null; lastSaliency = null; };
  window.__recordSaliency = (data) => { lastSaliency = data; };

  async function download(format) {
    const status = document.getElementById("reportStatus");
    if (!lastAnalysis) {
      status.textContent = "Run an analysis first.";
      status.className = "status error";
      return;
    }
    status.textContent = "Building report...";
    status.className = "status";
    try {
      const response = await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clip_key: lastAnalysis.summary.clip_key,
          summary: lastAnalysis.summary,
          window_scores: lastAnalysis.window_scores,
          overall_risk: lastAnalysis.overall_risk,
          selected_window: selectedWindow,
          saliency: lastSaliency,
          fusion: lastAnalysis.fusion,
          spatial_fusion_feature: lastAnalysis.spatial_fusion_feature,
          format: format,
        }),
      });
      if (!response.ok) {
        let message = `HTTP ${response.status}`;
        try { message = (await response.json()).error || message; } catch (e) {}
        status.textContent = message;
        status.className = "status error";
        return;
      }
      const blob = await response.blob();
      const stem = lastAnalysis.summary.clip_key.split("/").pop().replace(/\.[^.]+$/, "");
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `incident_${stem}.${format === "text" ? "txt" : "json"}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(link.href);
      status.textContent =
        "Report downloaded" + (lastSaliency ? " (including saliency)." : " (no saliency requested).");
      status.className = "status";
    } catch (err) {
      status.textContent = String(err);
      status.className = "status error";
    }
  }

  const jsonBtn = document.getElementById("reportJsonBtn");
  const textBtn = document.getElementById("reportTextBtn");
  if (jsonBtn) jsonBtn.addEventListener("click", () => download("json"));
  if (textBtn) textBtn.addEventListener("click", () => download("text"));
})();


// ---------------------------------------------------------------------------
// Mode banner and fusion panel.
//
// Both render from the server's own fields. Nothing here decides what mode a
// result came from or what a fusion rule concluded -- inferring either on the
// client would be a second source of truth for the one distinction this system
// most needs to keep straight.
// ---------------------------------------------------------------------------

function renderModeBanner(data) {
  const live = data.mode === "live";
  els.modeBanner.textContent =
    data.mode_label || (live ? "LIVE MODEL" : "REPLAY");
  els.modeBanner.classList.toggle("live", live);
  els.modeBanner.classList.toggle("replay", !live);

  const provenance = (data.summary && data.summary.temporal_provenance) || {};
  if (live) {
    const windows =
      (provenance.window_geometry && provenance.window_geometry.windows_completed) ?? "?";
    const seconds = provenance.inference_seconds_total;
    els.modeBanner.title =
      `R3D-18 ran in this process on frames decoded from this video. ` +
      `${windows} window(s), ${seconds ?? "?"}s of inference, ` +
      `checkpoint ${provenance.checkpoint_sha256 || "unknown"}.`;
  } else {
    els.modeBanner.title =
      "Per-window probabilities were replayed from the committed CSV. " +
      "No temporal model ran in this process.";
  }

  if (data.live_note) {
    els.liveNote.textContent = data.live_note;
    els.liveNote.classList.remove("hidden");
  } else {
    els.liveNote.textContent = "";
    els.liveNote.classList.add("hidden");
  }
}

function renderFusion(fusion, feature) {
  if (!fusion) {
    els.fusionPanel.innerHTML =
      '<p class="muted">No fusion result was returned for this run.</p>';
    return;
  }
  if (!fusion.available) {
    els.fusionPanel.innerHTML =
      `<p class="muted">Fusion unavailable (${fusion.reason}): ` +
      `${fusion.detail || ""}</p>`;
    return;
  }

  const inputs = fusion.inputs || {};
  // An undefined spatial feature is shown as undefined. It is never rendered
  // as 0, which would read as "measured no motion" rather than "not measurable".
  const spatialText = inputs.spatial_defined
    ? `${Number(inputs.spatial_score).toFixed(6)} (percentile ${Number(
        inputs.spatial_percentile
      ).toFixed(3)})`
    : "undefined &mdash; abstains, treated as spatial-negative";

  const rows = Object.entries(fusion.candidates)
    .map(([name, candidate]) => {
      const score =
        typeof candidate.score === "number" ? candidate.score.toFixed(6) : "&mdash;";
      const incumbent = name === fusion.system_decision.from;
      return (
        `<tr class="${incumbent ? "incumbent" : ""}">` +
        `<td><code>${name}</code>${incumbent ? " <em>(system decision)</em>" : ""}</td>` +
        `<td>${candidate.rule}</td>` +
        `<td>${score}</td>` +
        `<td class="${candidate.fired ? "fired" : ""}">${candidate.decision}</td>` +
        `</tr>`
      );
    })
    .join("");

  const featureRow = feature
    ? `<p class="muted">Spatial feature <code>${feature.feature}</code>: ` +
      (feature.defined
        ? `mean speed ${feature.speed_mean_pixels.toFixed(3)} px / mean group diagonal ` +
          `${feature.group_diagonal_mean.toFixed(3)} px over ${feature.frames} frames ` +
          `(${feature.zero_person_frames} with no person detected).`
        : `undefined for this video &mdash; ${feature.note}`) +
      `</p>`
    : "";

  els.fusionPanel.innerHTML =
    `<p class="muted"><strong>${fusion.mode}</strong>. ${fusion.causality}</p>` +
    featureRow +
    `<p class="muted">Temporal max ${Number(inputs.temporal_max).toFixed(6)} ` +
    `(percentile ${Number(inputs.temporal_percentile).toFixed(3)}); spatial ${spatialText}.</p>` +
    `<table class="fusion-table"><thead><tr><th>candidate</th><th>rule</th>` +
    `<th>score</th><th>decision</th></tr></thead><tbody>${rows}</tbody></table>` +
    `<p class="muted">${fusion.system_decision.why}</p>`;
}


// ---------------------------------------------------------------------------
// HUD state.
//
// Three states only, each reflecting something the response actually says.
// The alert state is entered ONLY when the frozen temporal rule fired; it is
// never used decoratively, so seeing it always means the same thing.
// ---------------------------------------------------------------------------
function renderHud(data) {
  if (!els.hudState) return;
  const summary = data.summary || {};
  const fired = Boolean(summary.temporal_violence_signal_fired);
  const undetermined = summary.final_temporal_decision === "Undetermined";

  els.hudState.classList.toggle("alert", fired);
  els.hudState.classList.toggle("clear", !fired && !undetermined);
  els.hudState.textContent = undetermined
    ? "NO TEMPORAL EVIDENCE"
    : fired
    ? "ABNORMAL EVENT DETECTED"
    : "NO ACTIVE THREAT";
}

// ---------------------------------------------------------------------------
// Spatial evidence.
//
// Three groups kept deliberately apart, because conflating them is the single
// most misleading thing this panel could do:
//
//   OBSERVED EVIDENCE  -- detections and measurements taken from this video
//   MODEL PROBABILITY  -- the R3D output, the only learned quantity present
//   CONFIGURED RISK    -- an engineer-declared mapping, not a probability
//
// An undefined measurement renders as "Unavailable", never as 0.
// ---------------------------------------------------------------------------
function evRow(key, value, unavailable) {
  const cls = unavailable ? "ev-v unavailable" : "ev-v";
  const shown = unavailable ? "Unavailable" : value;
  return `<div class="ev-row"><span class="ev-k">${escapeHtml(key)}</span>` +
         `<span class="${cls}">${escapeHtml(shown)}</span></div>`;
}

function renderSpatialEvidence(data) {
  if (!els.spatialPanel) return;
  const feature = data.spatial_fusion_feature || null;
  const summary = data.summary || {};
  const events = data.event_summary || [];

  if (!feature && !events.length) {
    els.spatialPanel.innerHTML =
      '<p class="muted">No valid spatial evidence available for this video.</p>';
    return;
  }

  const defined = Boolean(feature && feature.defined);
  const observed =
    '<div class="ev-group is-observed"><h4>Observed evidence</h4>' +
    evRow("Frames processed", summary.frames_processed, summary.frames_processed == null) +
    evRow("Mean persons / frame",
          feature && feature.person_count_mean != null
            ? Number(feature.person_count_mean).toFixed(2) : "",
          !feature || feature.person_count_mean == null) +
    evRow("Frames with no person",
          feature ? `${feature.zero_person_frames} / ${feature.frames}` : "", !feature) +
    evRow("Displacement samples",
          feature ? feature.displacement_samples : "", !feature) +
    evRow("Mean group diagonal",
          feature && feature.group_diagonal_mean != null
            ? `${Number(feature.group_diagonal_mean).toFixed(1)} px` : "",
          !feature || feature.group_diagonal_mean == null) +
    evRow("Spatial feature",
          defined ? Number(feature.spatial_score).toFixed(6) : "", !defined) +
    "</div>";

  const rules = events.length
    ? events.map((e) => evRow(`${e.event_type}`, `${e.occurrences} frame-events`, false)).join("")
    : '<div class="ev-row"><span class="ev-k">Rule events</span>' +
      '<span class="ev-v unavailable">None raised</span></div>';

  const model =
    '<div class="ev-group is-model"><h4>Model probability</h4>' +
    evRow("Temporal max (R3D)",
          summary.temporal_max_probability != null
            ? Number(summary.temporal_max_probability).toFixed(6) : "",
          summary.temporal_max_probability == null) +
    evRow("Windows scored", (data.window_scores || []).length, false) +
    evRow("Decision", summary.final_temporal_decision, !summary.final_temporal_decision) +
    '<div class="ev-row"><span class="ev-k">Provenance</span>' +
    '<span class="ev-v">measured_model_probability</span></div>' +
    "</div>";

  const configured =
    '<div class="ev-group is-configured"><h4>Configured risk</h4>' +
    evRow("Severity", data.overall_risk, !data.overall_risk) +
    '<div class="ev-row"><span class="ev-k">Validated</span>' +
    '<span class="ev-v unavailable">No</span></div>' +
    '<div class="ev-row"><span class="ev-k">Provenance</span>' +
    '<span class="ev-v">configured_interpretation</span></div>' +
    rules +
    "</div>";

  els.spatialPanel.innerHTML = observed + model + configured;
  if (feature && !defined && feature.note) {
    els.spatialPanel.insertAdjacentHTML(
      "beforeend", `<p class="caveat">${escapeHtml(feature.note)}</p>`);
  }
}

// ---------------------------------------------------------------------------
// Incident-report preview. Summarises what the export WILL contain, read from
// the same payload the export sends -- it never describes a field the report
// would not carry.
// ---------------------------------------------------------------------------
function renderReportPreview(data) {
  if (!els.reportPreview) return;
  const summary = data.summary || {};
  const provenance = summary.temporal_provenance || {};
  const fusion = data.fusion || {};
  const items = [
    ["Incident", summary.clip_key || "--"],
    ["Mode", data.mode === "live" ? "Live inference" : "Replay (precomputed)"],
    ["Temporal evidence", `${(data.window_scores || []).length} window(s)`],
    ["Spatial evidence",
      data.spatial_fusion_feature && data.spatial_fusion_feature.defined
        ? "Measured" : "Unavailable"],
    ["Fusion", fusion.available ? "Configured candidates" : "Not available"],
    ["Risk", `${data.overall_risk} (configured)`],
    ["Explanation", "On demand"],
    ["Provenance", provenance.checkpoint_sha256
      ? `checkpoint ${String(provenance.checkpoint_sha256).slice(0, 12)}…`
      : "committed CSV"],
  ];
  els.reportPreview.innerHTML = items
    .map(([k, v]) => `<div class="rp-item"><span class="rp-k">${escapeHtml(k)}</span>` +
                     `<span class="rp-v">${escapeHtml(v)}</span></div>`)
    .join("");
}
