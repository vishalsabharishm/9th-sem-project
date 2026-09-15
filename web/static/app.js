// web/static/app.js
//
// Client-side only: builds the form data for /api/analyze and renders
// whatever the server returns. It never computes a risk level, confidence,
// or evidence value itself -- everything shown here comes straight from the
// JSON the Flask server produced from the real run_demo() pipeline output.

const state = {
  activeTab: "presets",
  selectedPreset: null,
  clips: [], // [{clip_key, true_label}]
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
};

function setStatus(text, isError) {
  els.status.textContent = text || "";
  els.status.classList.toggle("error", Boolean(isError));
}

function updateAnalyzeEnabled() {
  if (state.activeTab === "presets") {
    els.analyzeBtn.disabled = !state.selectedPreset;
  } else if (state.activeTab === "browse") {
    els.analyzeBtn.disabled = !isKnownClip(els.clipSearch.value.trim());
  } else {
    const hasFile = els.videoUpload.files && els.videoUpload.files.length > 0;
    els.analyzeBtn.disabled = !(hasFile && isKnownClip(els.uploadClipSearch.value.trim()));
  }
}

function isKnownClip(clipKey) {
  return state.clips.some((c) => c.clip_key === clipKey);
}

function switchTab(tab) {
  state.activeTab = tab;
  els.tabBtns.forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  els.tabPanels.forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== tab));
  updateAnalyzeEnabled();
}

els.tabBtns.forEach((btn) => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));

async function loadPresets() {
  const res = await fetch("/api/presets");
  const presets = await res.json();
  els.presetList.innerHTML = "";
  presets.forEach((p) => {
    const card = document.createElement("div");
    card.className = "preset-card";
    card.innerHTML = `
      <div>
        <strong>${p.id}</strong>
        <span class="clip-key">${p.clip_key} -- ground truth: ${p.true_label}</span>
      </div>
    `;
    card.addEventListener("click", () => {
      state.selectedPreset = p.id;
      document.querySelectorAll(".preset-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      updateAnalyzeEnabled();
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

els.clipSearch.addEventListener("input", () => {
  const match = state.clips.find((c) => c.clip_key === els.clipSearch.value.trim());
  els.clipSearchInfo.textContent = match ? `Ground truth: ${match.true_label}` : "";
  updateAnalyzeEnabled();
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
  updateAnalyzeEnabled();
});

els.analyzeBtn.addEventListener("click", runAnalyze);

async function runAnalyze() {
  const formData = new FormData();
  if (state.activeTab === "presets") {
    formData.append("preset", state.selectedPreset);
  } else if (state.activeTab === "browse") {
    formData.append("clip_key", els.clipSearch.value.trim());
  } else {
    formData.append("clip_key", els.uploadClipSearch.value.trim());
    formData.append("video", els.videoUpload.files[0]);
  }

  els.analyzeBtn.disabled = true;
  setStatus("Running YOLO detection + tracking + Phase-4 rules + temporal signal + risk assessment... this can take 30-90s on CPU.", false);
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
      setStatus(data.error || `Request failed (HTTP ${res.status}).`, true);
      return;
    }
    setStatus("Done.", false);
    renderResult(data);
  } catch (err) {
    setStatus(`Request failed: ${err}`, true);
  } finally {
    updateAnalyzeEnabled();
  }
}

function renderResult(data) {
  els.resultsPanel.classList.remove("hidden");

  // Remember which clip was analysed so the on-demand saliency request knows
  // what to explain. Set here rather than at submit time so it always reflects
  // the clip the displayed results actually came from.
  window.__lastClipKey = data.summary.clip_key;
  if (window.__recordAnalysis) window.__recordAnalysis(data);

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

  els.resultVideo.src = data.video_url + "?t=" + Date.now();
  els.videoCaption.textContent = `${data.summary.clip_key} -- ${data.summary.frames_processed} frames processed`;

  els.riskBadge.textContent = `Risk: ${data.overall_risk}`;
  els.riskBadge.className = `badge risk-${data.overall_risk}`;
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
      '<p class="error">No temporal windows available for this clip.</p>';
    if (label) label.textContent = "No scored windows - saliency unavailable.";
    return;
  }

  windows.forEach((w) => {
    const bar = document.createElement("div");
    const fired =
      firedFrame !== null && firedFrame !== undefined &&
      w.last_frame >= firedFrame && w.first_frame <= firedFrame;
    bar.className = "window-bar" + (w.fight_probability >= 0.5 ? " fired" : "");
    bar.style.height = `${Math.max(6, w.fight_probability * 100)}%`;
    bar.dataset.index = w.window_index;
    bar.title =
      `window ${w.window_index}: frames [${w.first_frame},${w.last_frame}] -> ` +
      `${w.fight_probability.toFixed(4)}` +
      (fired ? " (first alarm falls in this window)" : "") +
      " - click to select for saliency";
    bar.addEventListener("click", () => selectWindow(w));
    els.windowChart.appendChild(bar);

    if (table) {
      const row = document.createElement("tr");
      row.dataset.index = w.window_index;
      row.innerHTML =
        `<td>${w.window_index}</td>` +
        `<td>${w.first_frame}-${w.last_frame}</td>` +
        `<td>${w.fight_probability.toFixed(4)}</td>` +
        `<td>${decisionFor(w.fight_probability)}</td>`;
      row.addEventListener("click", () => selectWindow(w));
      table.appendChild(row);
    }
  });

  // Default to the peak window: the one the frozen max rule actually used.
  const peak = windows.reduce((a, b) =>
    b.fight_probability > a.fight_probability ? b : a);
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
