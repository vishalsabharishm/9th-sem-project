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
  // One place the analysis session lives. Cleared by New Analysis so a new run
  // can never inherit the previous run's evidence.
  session: null,
  fps: null,          // measured from the preview video, never assumed
};

function resetSession() {
  const note = document.getElementById("presetNote");
  if (note) { note.classList.add("hidden"); note.innerHTML = ""; }
  state.session = null;
  state.fps = null;
  selectedWindow = null;
  if (window.__clearAnalysis) window.__clearAnalysis();
  window.__lastClipKey = null;
  clearRunEvidence();
}

// Drop what the PREVIOUS run put on screen.
//
// __clearAnalysis already drops the exported state, which is what stops a
// stale report being written. This drops the pixels: without it a new
// configuration can still be read against the old run's video, heatmap,
// window selection and timing, because those live in views that are merely
// hidden rather than cleared.
function clearRunEvidence() {
  const video = document.getElementById("resultVideo");
  if (video) {
    video.pause();
    video.removeAttribute("src");
    video.load();
  }
  const saliency = document.getElementById("saliencyResult");
  if (saliency) saliency.innerHTML = "";
  const selection = document.getElementById("selectedWindow");
  if (selection) {
    selection.classList.remove("is-chosen");
    selection.textContent = "Select a window in the timeline above.";
  }
  const explain = document.getElementById("explainBtn");
  if (explain) explain.disabled = true;
  const elapsed = document.getElementById("processingElapsed");
  if (elapsed) elapsed.textContent = "";
  const banner = document.getElementById("completionBanner");
  if (banner) banner.classList.add("hidden");
  // Replay vs live is the one distinction this system most needs to keep
  // straight, so a stale "LIVE MODEL" must never sit above a new run's
  // results. renderModeBanner always rewrites this from the server's own
  // field; this only covers the gap before a result exists.
  const mode = document.getElementById("modeBanner");
  if (mode) {
    mode.textContent = "--";
    mode.classList.remove("live", "replay");
  }
}

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

// ---------------------------------------------------------------------------
// Scenes.
//
// Home and Report are documents: ivory, editorial, the brand. Setup,
// Processing, Results and Explain are the investigation, and they run on a
// charcoal stage -- because the things that matter on those screens are
// luminous objects (a surveillance frame, a probability curve, a saliency
// map), and every one of them reads better against dark than against paper.
//
// The switch is a data attribute on <body>; the stylesheet redefines the
// surface and text tokens beneath it, so every existing component follows
// without being rewritten.
// ---------------------------------------------------------------------------
const DARK_SCENES = ["setup", "processing", "results", "explain"];

function applyScene(name) {
  const scene = DARK_SCENES.indexOf(name) !== -1 ? "dark" : "light";
  if (document.body.dataset.scene !== scene) document.body.dataset.scene = scene;
}

function showView(name) {
  if (!VIEW_ORDER.includes(name)) return;
  state.view = name;
  applyScene(name);
  document.querySelectorAll(".view").forEach(function (section) {
    section.classList.toggle("hidden", section.dataset.view !== name);
  });
  // A short coordinated entrance on the view being revealed. Restarted by
  // hand because the element is reused rather than recreated, so the
  // animation would otherwise only ever play once. Honoured by the global
  // prefers-reduced-motion rule, which collapses its duration.
  const shown = document.querySelector('.view[data-view="' + name + '"]');
  if (shown) {
    shown.classList.remove("view-enter");
    void shown.offsetWidth;
    shown.classList.add("view-enter");
  }
  document.querySelectorAll("#stepper .step").forEach(function (btn) {
    const target = btn.dataset.goto;
    btn.classList.toggle("active", target === name);
    btn.classList.toggle("done", state.reached.indexOf(target) !== -1 && target !== name);
  });
  // After the active class is set, not before: the indicator is measured from
  // the active button, so moving it first would track the PREVIOUS step.
  moveStepperPill();

  // The rail cannot scroll while it is display:none, so the selected window
  // is brought into view when Explain actually becomes visible.
  if (name === "explain") {
    const chosen = document.querySelector("#explainWindowList .wp-item.selected");
    if (chosen && chosen.scrollIntoView) {
      chosen.scrollIntoView({ block: "nearest", inline: "center", behavior: "auto" });
    }
  }
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

// ---------------------------------------------------------------------------
// Chrome details: one indicator that travels, and a bar that separates itself
// from the page only once there is something behind it.
// ---------------------------------------------------------------------------

// Measured from the active button rather than hard-coded, so the indicator
// tracks whatever the layout does at any width, including after a wrap.
function moveStepperPill() {
  const pill = document.getElementById("stepperPill");
  if (!pill) return;
  const active = document.querySelector("#stepper .step.active");
  if (!active) {
    pill.classList.remove("on");
    return;
  }
  pill.style.width = `${active.offsetWidth}px`;
  pill.style.transform =
    `translate3d(${active.offsetLeft}px, ${active.offsetTop}px, 0)`;
  pill.style.height = `${active.offsetHeight}px`;
  pill.classList.add("on");
}

window.addEventListener("resize", moveStepperPill);

(function watchScroll() {
  const bar = document.querySelector(".topbar");
  if (!bar) return;
  const apply = () => bar.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", apply, { passive: true });
  apply();
})();

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
  // "New Analysis" is an explicit reset, not just navigation: the previous
  // run's evidence must not survive into a new configuration.
  if (target.dataset.reset === "session") {
    resetSession();
    lockStepsAfter("setup");
    setStages(null);
  }
  showView(target.dataset.goto);
});

const brandHome = document.getElementById("brandHome");
if (brandHome) brandHome.addEventListener("click", function () { showView("home"); });

const startNew = document.getElementById("startNewAnalysis");
if (startNew) startNew.addEventListener("click", function () {
  // This button carries data-reset="session" like the in-flow "New Analysis"
  // buttons, but the delegated reset handler only fires for elements that
  // ALSO carry data-goto, which this one does not. Resetting here is what
  // makes the declared attribute true, so entering a new analysis from the
  // home page leaves exactly as little behind as entering it from results.
  resetSession();
  lockStepsAfter("setup");
  setStages(null);
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
    video_input: summary.frames_processed > 0 ? "complete" : "warning",
    yolo_detection: "complete",
    object_tracking: "complete",
    // An undefined spatial feature is a warning, not a failure: it means no
    // usable track pair existed, which the panel below states plainly.
    spatial_analysis: feature && feature.defined ? "complete" : "warning",
    temporal_analysis: (data.window_scores || []).length ? "complete" : "warning",
    evidence_fusion: fusion && fusion.available ? "complete" : "warning",
    risk_interpretation: data.overall_risk ? "complete" : "warning",
    explanation_ready: "complete",
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
    const meta = PRESET_META[preset];
    return preset
      ? { label: meta ? "Demo library · " + meta.role : "Built-in demo",
          name: meta ? meta.name : preset,
          detail: known ? known.clip_key : "",
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
    ["Mode", live ? "LIVE MODEL \u2014 actual video inference"
                  : "REPLAY \u2014 precomputed research evidence"],
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


// ---------------------------------------------------------------------------
// Demo library.
//
// Presentation labels only. Every factual claim here is already documented and
// already shown elsewhere in this UI: the ground-truth labels come from the
// /api/presets response, and the false-positive characterisation is the same
// one the results page has always carried in #fpNotice and that
// docs/demo_instructions.md records. Nothing is asserted that the run will not
// itself demonstrate.
// ---------------------------------------------------------------------------
const PRESET_META = {
  fight: {
    role: "TRUE POSITIVE",
    name: "Fight",
    blurb: "Known fight example. The temporal signal fires.",
    tone: "ok",
  },
  nonfight: {
    role: "TRUE NEGATIVE",
    name: "NonFight",
    blurb: "Known normal footage. The temporal signal does not fire.",
    tone: "ok",
  },
  fp: {
    role: "CHALLENGING CASE",
    name: "False Positive",
    blurb: "NonFight footage the system classifies as Fight.",
    tone: "warn",
    decision: "Fight",
  },
};

// Shown only for the false-positive clip. It is kept in the demo deliberately:
// a demonstration that only shows successes is not evidence.
function renderPresetNote(presetId) {
  const note = document.getElementById("presetNote");
  if (!note) return;
  if (presetId !== "fp") {
    note.classList.add("hidden");
    note.innerHTML = "";
    return;
  }
  note.classList.remove("hidden");
  note.innerHTML =
    "<strong>Why show this case?</strong>" +
    "<p>This clip is a known false-positive example. The ground-truth label " +
    "is NonFight, while the system raises a Fight decision. It demonstrates " +
    "that the detector can produce false alarms, and gives a concrete failure " +
    "case to inspect \u2014 this project's own primary-split precision " +
    "(0.788) predicts that some will occur.</p>";
}

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
    const meta = PRESET_META[p.id] || {
      role: "DEMO CLIP", name: p.id, blurb: "", tone: "neutral",
    };
    card.classList.add("tone-" + meta.tone);
    card.innerHTML =
      `<span class="pc-role">${escapeHtml(meta.role)}</span>` +
      `<span class="pc-name">${escapeHtml(meta.name)}</span>` +
      (meta.blurb ? `<span class="pc-blurb">${escapeHtml(meta.blurb)}</span>` : "") +
      `<span class="pc-meta">${escapeHtml(p.clip_key)}</span>` +
      `<span class="pc-truth">Ground truth: <strong>${escapeHtml(p.true_label)}</strong>` +
      (meta.decision
        ? ` &middot; system decision: <strong class="is-alarm">${escapeHtml(meta.decision)}</strong>`
        : "") +
      "</span>";
    const choose = () => {
      state.selectedPreset = p.id;
      document.querySelectorAll(".preset-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      renderPresetNote(p.id);
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
  if (els.stageStrip) {
    els.stageStrip.classList.remove("settled", "group-active");
    els.stageStrip.querySelectorAll("li").forEach(function (li) {
      delete li.dataset.reveal;
      li.dataset.state = "waiting";
    });
  }
  const priorBanner = document.getElementById("completionBanner");
  if (priorBanner) priorBanner.classList.add("hidden");
  // The previous run's duration must not sit on the processing screen while a
  // new run is starting: for the moment before the server reports its first
  // phase it would read as this run's timing.
  const priorElapsed = document.getElementById("processingElapsed");
  if (priorElapsed) priorElapsed.textContent = "";
  els.processingSub.textContent = state.mode === "live"
    ? "Running R3D-18 inference locally on CPU. This is not real time."
    : "Replaying committed window probabilities. No temporal model runs.";

  els.analyzeBtn.disabled = true;
  els.analyzeBtn.classList.add("busy");
  // Nothing claims to be running until the server reports a phase. The request
  // has not even been accepted yet here, so showing fusion or risk as
  // "processing" would be a claim about work that has not started.
  setStages("waiting");
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
  if (staleSelection) {
    staleSelection.classList.remove("is-chosen");
    staleSelection.textContent = "Select a window in the timeline above.";
  }
  const staleSaliency = document.getElementById("saliencyResult");
  if (staleSaliency) staleSaliency.innerHTML = "";

  try {
    // The dashboard drives the background job so the processing screen can
    // report phases the server actually reached. /api/analyze still exists and
    // still behaves identically for any other caller.
    const started = await fetch("/api/analysis", { method: "POST", body: formData });
    const job = await started.json();
    if (!started.ok) {
      const message = job.error || `Request failed (HTTP ${started.status}).`;
      setStatus(message, true);
      setStages("warning");
      showProcessingError(message);
      return;
    }
    applyJobPhase(job);
    const data = await pollJob(job.job_id);
    if (data === null) return;
    setStatus("Analysis complete.", false);
    showToast("Analysis complete.");
    renderResult(data);
    unlockStep("results");
    unlockStep("explain");
    unlockStep("report");

    // Marks the arrival of evidence so the results scene plays its entrance
    // once per analysis, not on every later visit to the view. Deliberately
    // after the unlocks: presentation must never sit between the success
    // signal and the step it unlocks.
    const resultsView = document.getElementById("viewResults");
    if (resultsView) {
      resultsView.classList.remove("reveal");
      void resultsView.offsetWidth;
      resultsView.classList.add("reveal");
    }
    els.viewResultsBtn.classList.remove("hidden");
    els.processingSub.textContent = "Analysis complete. Evidence available for inspection.";
    const note = document.getElementById("interleavedNote");
    if (note) note.classList.add("hidden");
    const elapsedEl = document.getElementById("processingElapsed");
    if (elapsedEl && state.session && state.session.elapsed != null) {
      elapsedEl.textContent = `Completed in ${state.session.elapsed}s of local processing.`;
    }
    const strip = els.stageStrip;
    if (strip) {
      strip.classList.remove("group-active");
      strip.classList.add("settled");
    }
    const banner = document.getElementById("completionBanner");
    if (banner) banner.classList.remove("hidden");
    // Hand the examiner straight to the results rather than leaving them on a
    // finished progress screen.
    setTimeout(function () {
      if (state.view === "processing") showView("results");
    }, 1100);
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
  loadBrowserPreview(data);
  els.videoCaption.textContent =
    `${data.summary.clip_key} \u00b7 ${data.summary.frames_processed} frames processed`;

  els.riskBadge.textContent = data.overall_risk;
  els.riskBadge.className = `badge risk-${data.overall_risk}`;
  if (els.riskCard) els.riskCard.classList.toggle("is-high", data.overall_risk === "High");
  els.truthBadge.textContent = `Ground truth (dataset label, not used by the pipeline): ${data.summary.ground_truth_label}`;

  const isFalsePositive =
    data.summary.temporal_violence_signal_fired && data.summary.ground_truth_label === "NonFight";
  els.fpNotice.classList.toggle("hidden", !isFalsePositive);

  renderTemporalHeadline(data);
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


// ---------------------------------------------------------------------------
// Temporal evidence headline.
//
// The four figures an examiner asks for first, lifted out of the provenance
// table so they are readable from across a room. Every value is read from the
// analysis response; the frame-to-seconds conversion uses the frame rate
// MEASURED from the preview, and is simply omitted when that is unavailable.
// ---------------------------------------------------------------------------
function renderTemporalHeadline(data) {
  const target = document.getElementById("temporalHeadline");
  if (!target) return;
  const summary = data.summary || {};
  const windows = data.window_scores || [];
  const peak = windows.length
    ? windows.reduce(function (a, b) {
        return b.fight_probability > a.fight_probability ? b : a;
      })
    : null;

  const alarmFrame = summary.temporal_signal_first_frame;
  const alarmText = alarmFrame == null
    ? "Did not fire"
    : (state.fps
        ? (alarmFrame / state.fps).toFixed(2) + " s (frame " + alarmFrame + ")"
        : "frame " + alarmFrame);

  const decision = summary.final_temporal_decision || "\u2014";
  const peakText = peak
    ? (peak.fight_probability * 100).toFixed(2) + "%"
    : "Unavailable";

  // The verdict and the number that produced it lead; the rest steps down.
  // Same four figures, same four provenance notes, same source fields as
  // before -- only the order of the eye changes.
  const tone = decision === "Fight" ? " is-alarm"
    : decision === "NonFight" ? " is-clear" : "";
  // The eyebrow states what the decision MEANS, derived from the decision
  // itself -- the same wording the burned-in video badge already uses. It
  // introduces no new claim and no new value.
  const headline = decision === "Fight" ? "Incident detected"
    : decision === "NonFight" ? "No abnormal event detected"
    : "Decision unavailable";

  const hero =
    '<div class="decision-hero' + tone + '">' +
      '<div class="dh-main">' +
        '<span class="dh-k">' + escapeHtml(headline) + "</span>" +
        '<span class="dh-v">' + escapeHtml(decision) + "</span>" +
        '<span class="dh-note">system decision \u00b7 frozen rule: max \u2265 0.14</span>' +
      "</div>" +
      '<div class="dh-peak">' +
        '<span class="dh-pv">' + escapeHtml(peakText) + "</span>" +
        '<span class="dh-pk">Peak fight probability</span>' +
        '<span class="dh-note">measured model probability</span>' +
      "</div>" +
    "</div>";

  const stats = [
    ["First alarm", alarmText, "earliest window over threshold"],
    ["Windows scored", String(windows.length), "16 frames, stride 8"],
  ];
  const row = '<div class="stat-row">' + stats.map(function (item) {
    return '<div class="stat">' +
      '<span class="stat-k">' + escapeHtml(item[0]) + "</span>" +
      '<span class="stat-v">' + escapeHtml(item[1]) + "</span>" +
      '<span class="stat-note">' + escapeHtml(item[2]) + "</span></div>";
  }).join("") + "</div>";

  target.innerHTML = hero + row;
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
function renderExplainPicker(windowScores, keepSelection) {
  if (!els.explainWindowList) return;
  const windows = windowScores || [];
  state.lastWindows = windows;
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
      '<span class="wp-frames">frames ' + w.first_frame + "–" + w.last_frame +
      (windowTimestamp(w) ? '<em class="wp-time">' + windowTimestamp(w) + "</em>" : "") +
      "</span>" +
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
  if (keepSelection && selectedWindow) selectWindow(selectedWindow);
}

function selectWindow(w) {
  // A heatmap belongs to one window. Moving the selection elsewhere must drop
  // it: otherwise the panel showed frames 88-103 under a label reading
  // "Window 3", and -- worse -- the exported report paired that saliency block
  // with a selected_window of frames 24-39, which is a factual inconsistency
  // in a delivered artifact. Re-selecting the SAME window keeps its result.
  const changed = !selectedWindow || selectedWindow.window_index !== w.window_index;
  if (changed) clearSaliencyForNewWindow();

  selectedWindow = w;
  const label = document.getElementById("selectedWindow");
  if (label) {
    // Same three facts as the sentence it replaces, ordered so the window
    // being explained is what the eye lands on first.
    const windows = state.lastWindows || [];
    const isPeak = windows.length && windows.reduce(function (a, b) {
      return b.fight_probability > a.fight_probability ? b : a;
    }).window_index === w.window_index;
    label.classList.add("is-chosen");
    label.innerHTML =
      '<span class="sw-idx">Window ' + escapeHtml(String(w.window_index)) + "</span>" +
      (isPeak ? '<span class="sw-peak">PEAK</span>' : "") +
      '<span class="sw-meta">Frames ' + escapeHtml(String(w.first_frame)) +
        "\u2013" + escapeHtml(String(w.last_frame)) + "</span>" +
      '<span class="sw-p">p = ' + escapeHtml(w.fight_probability.toFixed(4)) + "</span>";
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
    const isSelected = Number(item.dataset.index) === w.window_index;
    item.classList.toggle("selected", isSelected);
    // On the stage the picker is a horizontal rail, so the chosen window can
    // sit off-screen. Bring it back into view without moving the page.
    if (isSelected && item.scrollIntoView) {
      item.scrollIntoView({
        block: "nearest",
        inline: "center",
        behavior: prefersReducedMotion() ? "auto" : "smooth",
      });
    }
  });
}


// ---------------------------------------------------------------------------
// Saliency belongs to the window it was computed for.
//
// Clearing is deliberately total: the panel, the recorded payload the report
// export reads, and nothing else. A "stale" badge was considered and rejected
// -- a heatmap labelled stale is still a heatmap an examiner can misread, and
// the report has no way to show a badge at all.
// ---------------------------------------------------------------------------
function clearSaliencyForNewWindow() {
  const target = document.getElementById("saliencyResult");
  if (target && target.innerHTML) {
    target.innerHTML =
      '<p class="muted">Saliency cleared: it described the previously ' +
      "selected window. Generate it again for this window.</p>";
  }
  if (window.__clearSaliency) window.__clearSaliency();
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
      label.classList.remove("is-chosen");
      label.textContent =
        "No scored windows, so saliency unavailable. Select a completed " +
        "temporal window to generate an explanation.";
    }
    renderExplainPicker([]);
    return;
  }

  const peak = windows.reduce((a, b) =>
    b.fight_probability > a.fight_probability ? b : a);

  // ---------------------------------------------------------------------
  // Geometry.
  //
  // One point per SCORED window, joined by straight segments. Deliberately
  // not a smoothed curve: a spline would draw probability values between
  // windows that the model never produced. The dots are the measurements;
  // the line only helps the eye travel between them.
  //
  // All of it is expressed in a 0-100 viewBox so the drawing survives a
  // resize without measuring anything, and strokes stay hairline-thin
  // through vector-effect rather than scaling with the box.
  // ---------------------------------------------------------------------
  const n = windows.length;
  const xOf = (i) => ((i + 0.5) / n) * 100;
  const yOf = (prob) => 100 - prob * BAR_SCALE * 100;

  const points = windows.map((w, i) => [xOf(i), yOf(w.fight_probability)]);
  const line = points.map(([x, y], i) =>
    `${i ? "L" : "M"}${x.toFixed(3)},${y.toFixed(3)}`).join(" ");
  const area =
    `M${points[0][0].toFixed(3)},100 ` +
    points.map(([x, y]) => `L${x.toFixed(3)},${y.toFixed(3)}`).join(" ") +
    ` L${points[n - 1][0].toFixed(3)},100 Z`;
  const thresholdY = yOf(FROZEN_THRESHOLD).toFixed(3);

  const plot = document.createElement("div");
  plot.className = "tl-plot";
  plot.innerHTML =
    '<svg class="tl-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">' +
      '<defs><linearGradient id="tlFill" x1="0" y1="0" x2="0" y2="1">' +
        '<stop offset="0%" class="tl-stop-a"/><stop offset="100%" class="tl-stop-b"/>' +
      "</linearGradient></defs>" +
      `<path class="tl-area" d="${area}" />` +
      `<path class="tl-line" d="${line}" pathLength="1" vector-effect="non-scaling-stroke" />` +
      `<line class="tl-threshold" x1="0" x2="100" y1="${thresholdY}" y2="${thresholdY}" ` +
        'vector-effect="non-scaling-stroke" />' +
    "</svg>" +
    `<span class="tl-thr-label" style="top:${thresholdY}%">` +
      `frozen threshold ${FROZEN_THRESHOLD}</span>`;
  els.windowChart.appendChild(plot);

  const tip = document.createElement("div");
  tip.className = "chart-tip";
  els.windowChart.appendChild(tip);

  // ---------------------------------------------------------------------
  // One hit column per window.
  //
  // These carry the class and data-index the rest of the app already uses,
  // so selection, keyboard access and the saliency contract are unchanged.
  // The selected band, the peak chip and the hover highlight are drawn from
  // their state in CSS, which is why selectWindow() needs no knowledge of
  // this drawing at all.
  // ---------------------------------------------------------------------
  // Overlapping windows mean the alarm frame falls inside more than one of
  // them, which is true and is what the tooltip says. The hairline marks the
  // moment once, on the earliest window containing it -- the one that made the
  // decision available.
  let alarmMarked = false;

  windows.forEach((w, i) => {
    const fired =
      firedFrame !== null && firedFrame !== undefined &&
      w.last_frame >= firedFrame && w.first_frame <= firedFrame;
    const marksAlarm = fired && !alarmMarked;
    if (marksAlarm) alarmMarked = true;
    const decision = decisionFor(w.fight_probability);

    const col = document.createElement("div");
    col.className = "window-bar" +
      (decision === "Fight" ? " fired" : "") +
      (w.window_index === peak.window_index ? " peak" : "") +
      (marksAlarm ? " alarm" : "");
    col.dataset.index = w.window_index;
    col.tabIndex = 0;
    col.setAttribute("role", "button");
    col.setAttribute("aria-label",
      `window ${w.window_index}: frames [${w.first_frame},${w.last_frame}] -> ` +
      `${w.fight_probability.toFixed(4)}` +
      (fired ? " (first alarm falls in this window)" : "") +
      " - click to select for saliency");
    col.innerHTML =
      '<span class="tl-guide" aria-hidden="true"></span>' +
      `<span class="tl-dot" style="bottom:${(w.fight_probability * BAR_SCALE * 100).toFixed(3)}%"></span>` +
      (marksAlarm ? '<span class="tl-alarm" aria-hidden="true"></span>' : "");

    // The PEAK label is wider than one column once there are many windows, so
    // it is placed on the plot rather than inside the column it marks --
    // otherwise it overflows that column and the layout audit rightly flags
    // it. Clamped away from both edges so it stays inside the plot.
    if (w.window_index === peak.window_index) {
      const chip = document.createElement("span");
      chip.className = "tl-peak";
      chip.textContent = "PEAK";
      chip.style.left = `${Math.min(Math.max(xOf(i), 7), 93)}%`;
      plot.appendChild(chip);
    }

    const showTip = () => {
      tip.innerHTML =
        `<b>Window ${w.window_index}</b>${w.window_index === peak.window_index ? " (peak)" : ""}<br>` +
        `frames ${w.first_frame}\u2013${w.last_frame}<br>` +
        `p = ${w.fight_probability.toFixed(4)}<br>` +
        `<span class="${decision === "Fight" ? "tip-fight" : ""}">${decision}</span>` +
        (fired ? "<br>first alarm" : "");
      tip.classList.add("show");
      const chartBox = plot.getBoundingClientRect();
      const colBox = col.getBoundingClientRect();
      const left = Math.min(
        Math.max(colBox.left - chartBox.left + colBox.width / 2 - tip.offsetWidth / 2, 4),
        chartBox.width - tip.offsetWidth - 4
      );
      tip.style.left = `${left}px`;
    };
    const hideTip = () => tip.classList.remove("show");

    col.addEventListener("mouseenter", showTip);
    col.addEventListener("focus", showTip);
    col.addEventListener("mouseleave", hideTip);
    col.addEventListener("blur", hideTip);
    col.addEventListener("click", () => selectWindow(w));
    col.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectWindow(w); }
    });
    plot.appendChild(col);

    if (table) {
      const row = document.createElement("tr");
      row.dataset.index = w.window_index;
      row.tabIndex = 0;
      row.innerHTML =
        `<td>${w.window_index}` +
        (w.window_index === peak.window_index
          ? ' <span class="peak-badge">PEAK</span>' : "") +
        `</td>` +
        `<td>${w.first_frame}\u2013${w.last_frame}</td>` +
        `<td>${w.fight_probability.toFixed(4)}</td>` +
        `<td class="${decision === "Fight" ? "is-fight" : ""}">${decision}</td>`;
      row.addEventListener("click", () => selectWindow(w));
      row.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectWindow(w); }
      });
      table.appendChild(row);
    }
  });

  // Frame extent, so the horizontal axis means something.
  const axis = document.createElement("div");
  axis.className = "tl-axis";
  axis.innerHTML =
    `<span>frame ${windows[0].first_frame}</span>` +
    `<span>${n} scored windows</span>` +
    `<span>frame ${windows[n - 1].last_frame}</span>`;
  els.windowChart.appendChild(axis);

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
  // Dropped when the selection moves, so an export can never pair one window's
  // saliency with a different window's selected_window.
  window.__clearSaliency = () => { lastSaliency = null; };

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
function renderReportMeta(data) {
  const target = document.getElementById("reportMeta");
  if (!target) return;
  const summary = data.summary || {};
  const provenance = summary.temporal_provenance || {};
  // Every field is read from the run; nothing -- including any timestamp -- is
  // invented. generated_utc comes from the report the server builds, so it is
  // deliberately not shown here.
  const rows = [
    ["Analysis ID", summary.clip_key || "--"],
    ["Mode", data.mode === "live" ? "Live inference" : "Replay (precomputed)"],
    ["Frames processed", summary.frames_processed != null ? String(summary.frames_processed) : "Not available"],
    ["Evidence status", (data.window_scores || []).length
      ? `${(data.window_scores || []).length} scored temporal window(s)`
      : "No temporal window completed"],
    ["Temporal source", provenance.temporal_source === "live"
      ? "R3D-18 inference in this process" : "Committed window probabilities"],
  ];
  target.innerHTML = rows.map(function (r) {
    return "<dt>" + escapeHtml(r[0]) + "</dt><dd>" + escapeHtml(r[1]) + "</dd>";
  }).join("");
}

function renderReportPreview(data) {
  if (!els.reportPreview) return;
  renderReportMeta(data);
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


// ---------------------------------------------------------------------------
// Processing: driven by real server phases, never by a timer.
//
// What the server can honestly observe is limited, and the UI says so rather
// than inventing a sequence. run_demo performs detection, tracking, spatial
// accumulation and temporal windowing INTERLEAVED, once per decoded frame --
// they do not complete one after another -- and aggregation, fusion and risk
// run after that loop inside the same call. So during processing the
// interleaved stages are shown active TOGETHER, labelled as such, and the
// post-loop stages stay queued until the result arrives. No percentage is ever
// shown, because none is knowable.
// ---------------------------------------------------------------------------
const STAGE_PHASE_TEXT = {
  queued: "Queued",
  validating: "Validating the request",
  analysing: "Analysing footage",
  finalising: "Reading analysis artifacts",
  complete: "Analysis complete",
  failed: "Analysis failed",
};

function applyJobPhase(job) {
  const phase = job.phase;
  els.processingSub.textContent = STAGE_PHASE_TEXT[phase] || phase;

  const strip = els.stageStrip;
  if (!strip) return;
  const interleaved = job.interleaved_stages || [];
  const postLoop = job.post_loop_stages || [];

  strip.querySelectorAll("li").forEach(function (li) {
    const stage = li.dataset.stage;
    if (phase === "validating" || phase === "queued") {
      li.dataset.state = stage === "video_input" ? "processing" : "waiting";
    } else if (phase === "analysing") {
      if (stage === "video_input") li.dataset.state = "complete";
      else if (interleaved.indexOf(stage) !== -1) li.dataset.state = "processing";
      else li.dataset.state = "waiting";
    } else if (phase === "finalising") {
      // By the time the server reports finalising, run_demo has returned: the
      // frame loop, fusion and risk have ALL finished inside that one call.
      // So these are revealed as complete, staggered purely so the eye can
      // follow -- the cascade is a reveal of finished work, never a claim that
      // fusion or risk are executing now. Only explanation_ready is genuinely
      // in progress here: the server is reading the artifacts it just wrote.
      if (stage === "video_input" || interleaved.indexOf(stage) !== -1) {
        li.dataset.state = "complete";
      } else if (postLoop.indexOf(stage) !== -1) {
        li.dataset.state = "complete";
        li.dataset.reveal = String(postLoop.indexOf(stage));
      } else {
        li.dataset.state = "processing";
      }
    } else if (phase === "complete") {
      li.dataset.state = "complete";
    }
  });

  // The temporal row says which of the two things is happening, from the
  // server's own wording -- replay must never read as model execution.
  const temporal = strip.querySelector('li[data-stage="temporal_analysis"] .stage-detail');
  if (temporal && job.temporal_note) temporal.textContent = job.temporal_note;

  const note = document.getElementById("interleavedNote");
  if (note) note.classList.toggle("hidden", phase !== "analysing");
  // The scan marks the interleaved BLOCK as working. Deliberately not a
  // row-to-row flow animation: these stages do not hand off to one another.
  strip.classList.toggle("group-active", phase === "analysing");

  const elapsed = document.getElementById("processingElapsed");
  if (elapsed) {
    elapsed.textContent = job.elapsed != null ? `${job.elapsed}s elapsed` : "";
  }
}

async function pollJob(jobId) {
  // 700 ms: responsive enough to feel live, light enough to be invisible next
  // to a multi-second analysis.
  while (true) {
    await new Promise(function (r) { setTimeout(r, 700); });
    let job;
    try {
      const res = await fetch(`/api/analysis/${jobId}`);
      job = await res.json();
      if (!res.ok) {
        const message = job.error || `Lost contact with the analysis (HTTP ${res.status}).`;
        setStatus(message, true);
        setStages("warning");
        showProcessingError(message);
        return null;
      }
    } catch (err) {
      const message = `Lost contact with the analysis: ${err}`;
      setStatus(message, true);
      setStages("warning");
      showProcessingError(message);
      return null;
    }
    applyJobPhase(job);
    if (job.phase === "failed") {
      setStatus(job.error || "Analysis failed.", true);
      setStages("warning");
      showProcessingError(job.error || "Analysis failed.");
      return null;
    }
    if (job.phase === "complete") {
      state.session = { job_id: jobId, elapsed: job.elapsed };
      return job.result;
    }
  }
}

// ---------------------------------------------------------------------------
// Browser-playable annotated video.
//
// The pipeline writes the annotated video with OpenCV's mp4v fourcc, which
// browsers do not decode. The server re-encodes the SAME frames into a
// browser-native codec on request, beside the original, and never in place of
// it. If no encoder is available the original is offered for download instead.
// ---------------------------------------------------------------------------
async function loadBrowserPreview(data) {
  const original = data.video_url;
  const name = original.split("/").pop();
  els.videoFallback.classList.add("hidden");
  els.resultVideo.classList.remove("hidden");

  // Clear the previous run's footage FIRST, and make the clearing VISIBLE.
  // Converting a preview takes a few seconds, and leaving the old video on
  // screen meanwhile showed one clip's annotated frames beside another clip's
  // evidence. The element is hidden rather than merely re-sourced because
  // currentSrc lingers until a new resource selection completes, so "cleared"
  // has to be something the viewer can see, not just technically true.
  els.resultVideo.removeAttribute("src");
  els.resultVideo.load();
  els.resultVideo.classList.add("hidden");
  state.fps = null;
  const provenanceEl = document.getElementById("videoProvenance");
  if (provenanceEl) provenanceEl.textContent = "";
  els.videoFallback.classList.remove("hidden");
  els.videoFallback.innerHTML =
    '<span class="preview-spinner" aria-hidden="true"></span>' +
    "<p>Preparing a browser-playable preview of the annotated video…</p>";

  function offerOriginal(detail) {
    els.resultVideo.classList.add("hidden");
    els.videoFallback.classList.remove("hidden");
    if (provenanceEl) provenanceEl.textContent = "";
    els.videoFallback.innerHTML =
      "<strong>This browser cannot play the annotated video.</strong>" +
      "<p>" + escapeHtml(detail) + " The file itself is complete and contains " +
      "every annotated frame.</p>" +
      '<a class="btn btn-ghost btn-sm" href="' + original +
      '" target="_blank" rel="noopener">Open annotated video</a>';
  }

  els.resultVideo.onerror = function () {
    offerOriginal("The browser could not decode it.");
  };

  try {
    const res = await fetch("/api/preview_video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_name: name }),
    });
    const preview = await res.json();
    if (preview.available) {
      els.videoFallback.classList.add("hidden");
      els.resultVideo.classList.remove("hidden");
      els.resultVideo.src = preview.url + "?t=" + Date.now();
      const caption = document.getElementById("videoProvenance");
      if (caption) {
        caption.textContent =
          "Browser preview (" + preview.codec + ") re-encoded from the same " +
          "annotated frames. The original artifact is unchanged.";
      }
      measureFps(data);
      return;
    }
    offerOriginal(preview.detail || "No browser-playable encoder is available.");
  } catch (err) {
    // Conversion is a convenience; failing it must not break the results page.
    els.videoFallback.classList.add("hidden");
    els.resultVideo.classList.remove("hidden");
    els.resultVideo.src = original + "?t=" + Date.now();
    measureFps(data);
  }
}

// Frames-per-second is MEASURED from the preview's own duration against the
// frame count the analysis reported. It is never assumed, and when it cannot
// be measured the timeline simply shows frames instead of timestamps.
function measureFps(data) {
  const frames = (data.summary || {}).frames_processed;
  els.resultVideo.onloadedmetadata = function () {
    const duration = els.resultVideo.duration;
    if (frames && duration && isFinite(duration) && duration > 0) {
      state.fps = frames / duration;
      renderExplainPicker((data.window_scores || []), true);
      renderTemporalHeadline(data);
    }
  };
}

function windowTimestamp(w) {
  if (!state.fps) return null;
  const start = w.first_frame / state.fps;
  const end = (w.last_frame + 1) / state.fps;
  return start.toFixed(2) + "\u2013" + end.toFixed(2) + "s";
}


// Print uses the existing @media print stylesheet; no PDF dependency.
const printBtn = document.getElementById("reportPrintBtn");
if (printBtn) {
  printBtn.addEventListener("click", function () {
    window.print();
  });
}
