#!/usr/bin/env python3
"""
web/server.py

Local Flask front end for the existing, already-verified P0-P4 pipeline.

This file adds a browser UI and a thin HTTP layer ONLY. It does not change,
re-implement, retrain, or duplicate any detection / tracking / temporal /
risk-assessment logic. Every "Analyze" request calls tools/run_demo.py's own
``run_demo()`` function directly -- the exact same function
``python tools/run_demo.py --clip fight`` calls from the command line.
Nothing here fabricates a result: a clip with no precomputed temporal score
in temporal_risk/primary_window_scores.csv is reported as unsupported, not
silently guessed (see docs/web_ui.md and docs/demo_instructions.md).

Run (from the project root, with the project's own venv active):

    .venv\\Scripts\\activate
    python web/server.py

Then open http://127.0.0.1:5000 in a browser.
"""
from __future__ import annotations

import json as _json
import sys
import threading
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from flask import Flask, jsonify, render_template, request, send_from_directory  # noqa: E402

from config import OUTPUTS_DIR  # noqa: E402
from detection import DEFAULT_MODEL_PATH  # noqa: E402
from temporal_event_adapter import (  # noqa: E402
    TEMPORAL_EVENT_TYPE,
    PrecomputedWindowScoreSource,
)
from run_demo import DEMO_CLIPS, PRIMARY_CSV, run_demo  # noqa: E402  (unmodified pipeline entry point)

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEMO_OUTPUT_DIR = OUTPUTS_DIR / "demo"

app = Flask(
    __name__,
    static_folder=str(APP_DIR / "static"),
    template_folder=str(APP_DIR / "templates"),
)

# tools/run_demo.py loads the YOLO model fresh and opens its own
# VideoCapture/VideoWriter on every call -- exactly like running it from the
# command line. This lock only keeps two browser tabs from racing on that
# shared local state; it does not change anything run_demo() computes.
_analyze_lock = threading.Lock()

# Read-only: enumerates the 394 primary-split clips that already have a
# real, precomputed temporal score (temporal_risk/primary_window_scores.csv).
# See docs/demo_instructions.md, "Running on a different clip".
_score_source = PrecomputedWindowScoreSource(PRIMARY_CSV)


def _resolve_video_path(clip_key: str) -> Path:
    """clip_key looks like 'val/Val_Fight/trtrhrt_1049.avi' -- the same
    relative path DEMO_CLIPS uses under data/rwf2000/RWF-2000/."""
    return REPO_ROOT / "data" / "rwf2000" / "RWF-2000" / clip_key


def _parse_evidence(evidence: list) -> dict:
    """Evidence is a list of 'key=value' strings produced by
    src/temporal_event_adapter.py. Split it into a dict for display only --
    the values themselves are not touched or re-derived."""
    parsed = {}
    for item in evidence:
        if "=" in item:
            key, value = item.split("=", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def _summarize_events(risk_assessments: list) -> list:
    """Group risk assessments by (event_type, risk_level), the same
    aggregation tools/run_demo.py's write_explanation_report already does
    for its own table -- pure counting of already-computed records."""
    counts: dict = {}
    for ra in risk_assessments:
        key = (ra["event_type"], ra["risk_level"])
        counts[key] = counts.get(key, 0) + 1
    return [
        {"event_type": event_type, "risk_level": risk_level, "occurrences": count}
        for (event_type, risk_level), count in sorted(counts.items(), key=lambda kv: -kv[1])
    ]


def _overall_risk(summary: dict, risk_assessments: list) -> str:
    """Headline classification for the UI: derived purely from the risk
    levels RiskAssessor already assigned, never a new score. High whenever
    the Temporal Violence Signal fired (RiskAssessor maps it to High);
    otherwise the highest Phase-4 rule risk level present, else 'None'."""
    if summary.get("temporal_violence_signal_fired"):
        return "High"
    levels_present = {ra["risk_level"] for ra in risk_assessments}
    for level in ("High", "Medium", "Low"):
        if level in levels_present:
            return level
    return "None"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/presets")
def api_presets():
    """The 3 built-in demo clips from tools/run_demo.py's own DEMO_CLIPS."""
    return jsonify(
        [
            {
                "id": preset_id,
                "clip_key": preset["clip_key"],
                "true_label": _score_source.true_label(preset["clip_key"]),
            }
            for preset_id, preset in DEMO_CLIPS.items()
        ]
    )


@app.route("/api/clips")
def api_clips():
    """Every primary-split clip with a real precomputed temporal score."""
    return jsonify(
        [
            {"clip_key": clip_key, "true_label": _score_source.true_label(clip_key)}
            for clip_key in _score_source.clips()
        ]
    )


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    preset = (request.form.get("preset") or "").strip()
    clip_key = (request.form.get("clip_key") or "").strip()
    uploaded = request.files.get("video")

    if preset:
        if preset not in DEMO_CLIPS:
            return jsonify({"error": f"Unknown preset '{preset}'."}), 400
        resolved_clip_key = DEMO_CLIPS[preset]["clip_key"]
        video_path = DEMO_CLIPS[preset]["video"]
    elif clip_key:
        if _score_source.get(clip_key) is None:
            return (
                jsonify(
                    {
                        "error": (
                            f"No precomputed temporal score exists for clip_key '{clip_key}' in "
                            "temporal_risk/primary_window_scores.csv. Live scoring is not available "
                            "on this machine (the R3D-18 checkpoint, best.pt, was never downloaded "
                            "here) -- pick a clip_key from the browse list, which only lists clips "
                            "that already have a real score."
                        )
                    }
                ),
                400,
            )
        resolved_clip_key = clip_key
        if uploaded and uploaded.filename:
            safe_name = Path(uploaded.filename).name
            video_path = UPLOAD_DIR / safe_name
            uploaded.save(str(video_path))
        else:
            video_path = _resolve_video_path(clip_key)
            if not video_path.exists():
                return jsonify({"error": f"Video file not found locally: {video_path}"}), 400
    else:
        return (
            jsonify({"error": "Provide either 'preset' (fight/nonfight/fp) or a 'clip_key' from /api/clips."}),
            400,
        )

    try:
        with _analyze_lock:
            summary = run_demo(Path(video_path), resolved_clip_key, DEFAULT_MODEL_PATH, DEMO_OUTPUT_DIR)
    except SystemExit as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # surfaced to the UI verbatim, never swallowed
        traceback.print_exc()
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    stem = Path(resolved_clip_key).stem
    risk_json_path = DEMO_OUTPUT_DIR / f"demo_{stem}_risk_assessment.json"
    explanation_path = DEMO_OUTPUT_DIR / f"demo_{stem}_explanation.md"

    risk_assessments = _json.loads(risk_json_path.read_text(encoding="utf-8"))
    explanation_text = explanation_path.read_text(encoding="utf-8")

    temporal_signal = next(
        (ra for ra in risk_assessments if ra["event_type"] == TEMPORAL_EVENT_TYPE), None
    )
    if temporal_signal is not None:
        temporal_signal = dict(temporal_signal)
        temporal_signal["evidence_parsed"] = _parse_evidence(temporal_signal["evidence"])

    windows = _score_source.get(resolved_clip_key) or []
    window_scores = [
        {
            "window_index": w.window_index,
            "first_frame": w.first_frame,
            "last_frame": w.last_frame,
            "fight_probability": w.fight_probability,
        }
        for w in windows
    ]

    return jsonify(
        {
            "summary": summary,
            "overall_risk": _overall_risk(summary, risk_assessments),
            "temporal_signal": temporal_signal,
            "event_summary": _summarize_events(risk_assessments),
            "window_scores": window_scores,
            "explanation_text": explanation_text,
            "video_url": f"/outputs/demo/{Path(summary['annotated_video']).name}",
            "risk_json_url": f"/outputs/demo/{risk_json_path.name}",
            "explanation_url": f"/outputs/demo/{explanation_path.name}",
        }
    )


@app.route("/outputs/demo/<path:filename>")
def outputs_demo(filename):
    return send_from_directory(str(DEMO_OUTPUT_DIR), filename)


if __name__ == "__main__":
    print(f"Repo root:  {REPO_ROOT}")
    print(f"Model:      {DEFAULT_MODEL_PATH}")
    print(f"Scores CSV: {PRIMARY_CSV} ({len(_score_source.clips())} clips)")
    print("Serving on http://127.0.0.1:5000  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
