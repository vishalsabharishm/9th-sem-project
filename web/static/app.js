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

function renderWindowChart(windowScores, firedFrame) {
  els.windowChart.innerHTML = "";
  (windowScores || []).forEach((w) => {
    const bar = document.createElement("div");
    const fired = firedFrame !== null && firedFrame !== undefined && w.last_frame >= firedFrame && w.first_frame <= firedFrame;
    bar.className = "window-bar" + (w.fight_probability >= 0.5 ? " fired" : "");
    bar.style.height = `${Math.max(6, w.fight_probability * 100)}%`;
    bar.title = `window ${w.window_index}: frames [${w.first_frame},${w.last_frame}] -> ${w.fight_probability.toFixed(4)}`;
    els.windowChart.appendChild(bar);
  });
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
