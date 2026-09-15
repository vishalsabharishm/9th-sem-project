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
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from flask import Flask, jsonify, render_template, request, send_from_directory  # noqa: E402

from config import OUTPUTS_DIR  # noqa: E402
from detection import DEFAULT_MODEL_PATH  # noqa: E402
import base64
import threading
import time

import numpy as np

from temporal_explanation import (  # noqa: E402
    describe_provenance_legend,
    explain_frame_window,
    risk_presentation,
    unavailable as explanation_unavailable,
)
from temporal_runtime import RUNTIME_CLIP_LENGTH, build_violence_engine  # noqa: E402
from checkpoint_identity import file_sha256  # noqa: E402
from demo_preflight import run_preflight  # noqa: E402
from incident_report import (  # noqa: E402
    build_incident_report,
    render_incident_report_text,
)

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

# The frozen decision threshold, passed to the explainer rather than redefined
# there, so this file cannot become a second place an operating point lives.
FROZEN_TEMPORAL_THRESHOLD = 0.14

# ---------------------------------------------------------------------------
# Process-local engine cache for saliency.
#
# /api/explain previously rebuilt the engine on every request, reloading a
# 132 MB checkpoint each time -- about 1.9 s of a 3.4 s response. The cache key
# includes the checkpoint's SHA-256, so replacing the file on disk produces a
# different key and forces a rebuild rather than silently serving a stale model.
#
# This caches ONLY the model object. It is never a source of temporal scores:
# CSV replay continues to read _score_source, and nothing here can turn a
# replayed probability into a live one.
# ---------------------------------------------------------------------------
_engine_cache: dict = {}
_engine_cache_lock = threading.Lock()


def _cached_engine(checkpoint: Path, device: str = "cpu"):
    """Return a verified engine, rebuilding only when the checkpoint changes."""
    digest = file_sha256(checkpoint)
    key = (str(checkpoint.resolve()), digest, device)
    with _engine_cache_lock:
        hit = _engine_cache.get(key)
        if hit is not None:
            return hit, True
        # build_violence_engine runs the Step-3 provenance guard and raises on a
        # random-init, smoke-test or unverifiable checkpoint. Only a verified
        # engine ever reaches the cache.
        engine, _verdict = build_violence_engine(checkpoint, device=device)
        _engine_cache.clear()          # at most one checkpoint is ever cached
        _engine_cache[key] = engine
        return engine, False


DATASET_ROOT = REPO_ROOT / "data" / "rwf2000" / "RWF-2000"


def _clip_key_from(payload: dict) -> Optional[str]:
    """Extract clip_key as a string, or None if the caller sent something else.

    ``(payload.get("clip_key") or "").strip()`` raises AttributeError when the
    JSON carries a list or a number, which surfaced as HTTP 500 -- a crash where
    the caller simply sent the wrong type. A malformed request deserves an
    actionable 400, not a server error.
    """
    value = payload.get("clip_key")
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _resolve_video_path(clip_key: str) -> Optional[Path]:
    """Resolve a clip_key under the dataset root, or None if it escapes.

    clip_key looks like 'val/Val_Fight/trtrhrt_1049.avi' -- the same relative
    path DEMO_CLIPS uses under data/rwf2000/RWF-2000/.

    clip_key is user-controlled, and a bare ``root / clip_key`` is not safe for
    two reasons. ``..`` segments walk out of the dataset, and pathlib REPLACES
    the base entirely when the right-hand operand is absolute, so
    'C:/Windows/win.ini' would resolve to exactly that. Both were reachable
    before this guard: a request could open any file on disk that OpenCV would
    accept, and the differing error for a present versus absent file worked as
    a file-existence oracle for arbitrary paths.

    Returning None rather than raising keeps the caller's error handling
    uniform: an out-of-tree key is simply not a resolvable clip.
    """
    candidate = (DATASET_ROOT / clip_key).resolve()
    root = DATASET_ROOT.resolve()
    if candidate == root or root not in candidate.parents:
        return None
    return candidate


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


@app.route("/api/explain", methods=["POST"])
def api_explain():
    """Gradient-based temporal saliency for ONE completed window, on demand.

    Deliberately a separate endpoint rather than part of /api/analyze. Grad-CAM
    needs a forward AND a backward pass (~1.8 s per window on this CPU), so
    computing it for all 17 windows of every analysis would roughly double the
    demo's cost for output nobody asked to see. It is also impossible in the
    default replay path: CSV replay has a recorded probability but no weights
    and no tensor, so there is nothing to differentiate. Both facts are returned
    as explicit reasons rather than an empty panel.
    """
    payload = request.get_json(silent=True) or {}
    clip_key = _clip_key_from(payload)
    if clip_key is None:
        return jsonify(explanation_unavailable(
            "bad_request", "clip_key is required and must be a string.")), 400
    try:
        first_frame = int(payload.get("first_frame"))
    except (TypeError, ValueError):
        return jsonify(explanation_unavailable(
            "bad_request", "first_frame must be an integer.")), 400
    checkpoint = REPO_ROOT / "models" / "temporal_violence" / "best.pt"
    if not checkpoint.is_file():
        return jsonify(explanation_unavailable(
            "checkpoint_unavailable",
            f"The R3D-18 checkpoint is not present at {checkpoint}. Saliency "
            "requires the model itself; the replayed per-window probabilities "
            "cannot be differentiated.",
        ))

    # Membership first, matching the guard /api/analyze already applies. A key
    # that is not a known scored clip cannot be explained, and checking this
    # before touching the filesystem means an unknown key reveals nothing about
    # what does or does not exist on disk.
    if _score_source.get(clip_key) is None:
        return jsonify(explanation_unavailable(
            "unknown_clip",
            f"'{clip_key}' is not one of the clips with precomputed temporal "
            "scores. Pick a clip from the browse list or a built-in preset.",
        ))

    video_path = _resolve_video_path(clip_key)
    if video_path is None or not video_path.exists():
        return jsonify(explanation_unavailable(
            "video_unavailable",
            f"No video available for clip '{clip_key}' under the dataset root.",
        ))

    import cv2
    capture = cv2.VideoCapture(str(video_path))
    frames, numbers = [], []
    index = 0
    while len(frames) < RUNTIME_CLIP_LENGTH:
        ok, frame = capture.read()
        if not ok:
            break
        if index >= first_frame:
            frames.append(frame)
            numbers.append(index)
        index += 1
    capture.release()
    if len(frames) < RUNTIME_CLIP_LENGTH:
        return jsonify(explanation_unavailable(
            "window_incomplete",
            f"Only {len(frames)} frames available from frame {first_frame}; a "
            f"window needs {RUNTIME_CLIP_LENGTH}.",
        ))

    try:
        with _analyze_lock:
            engine, cache_hit = _cached_engine(checkpoint, device="cpu")
            started = time.perf_counter()
            result = explain_frame_window(
                engine, frames, numbers, FROZEN_TEMPORAL_THRESHOLD,
                source_height=frames[0].shape[0], source_width=frames[0].shape[1],
            )
            elapsed = time.perf_counter() - started
    except Exception as exc:  # surfaced, never swallowed
        traceback.print_exc()
        return jsonify(explanation_unavailable(
            "explanation_failed", f"{type(exc).__name__}: {exc}")), 500

    if not result.get("available"):
        return jsonify(result)

    # The heatmap array is not JSON; send a compact per-slice summary plus a
    # base64 PNG of the peak slice, so the UI can render without the client
    # ever reconstructing a tensor.
    heatmaps = result.pop("heatmaps")
    peak_index = int(np.argmax(heatmaps.sum(axis=(1, 2))))
    peak_map = (heatmaps[peak_index] * 255).astype("uint8")
    coloured = cv2.applyColorMap(peak_map, cv2.COLORMAP_JET)
    ok, buffer = cv2.imencode(".png", coloured)
    result["saliency"]["peak_slice_png_base64"] = (
        base64.b64encode(buffer.tobytes()).decode("ascii") if ok else None
    )
    result["saliency"]["per_slice_mass"] = [
        float(value) for value in heatmaps.sum(axis=(1, 2))
    ]
    result["risk_presentation"] = risk_presentation()
    result["provenance_legend"] = describe_provenance_legend()
    result["compute_seconds"] = elapsed
    result["computed_on_demand"] = True
    result["checkpoint_cache_hit"] = cache_hit
    result["not_real_time"] = (
        f"Saliency took {elapsed:.1f} s for one window on CPU. This is an "
        "on-demand diagnostic, not real-time inference."
    )
    return jsonify(result)


@app.route("/api/preflight")
def api_preflight():
    """What is ready, and what each missing piece actually blocks."""
    from run_demo import DEMO_CLIPS as _clips
    result = run_preflight(
        repo_root=REPO_ROOT,
        temporal_csv=PRIMARY_CSV,
        yolo_weights=Path(DEFAULT_MODEL_PATH),
        demo_videos={key: Path(entry["video"]) for key, entry in _clips.items()},
    )
    return jsonify(result.as_dict())


@app.route("/api/report", methods=["POST"])
def api_report():
    """Downloadable incident report for a clip already analysed.

    Rebuilt from the artifacts the analysis wrote rather than re-running the
    pipeline, so the report describes exactly the run the examiner is looking
    at. Saliency is included only if the caller passes the payload it already
    received; nothing is recomputed here.
    """
    payload = request.get_json(silent=True) or {}
    clip_key = _clip_key_from(payload)
    if clip_key is None:
        return jsonify({"error": "clip_key is required and must be a string."}), 400

    stem = Path(clip_key).stem
    risk_json_path = DEMO_OUTPUT_DIR / f"demo_{stem}_risk_assessment.json"
    if not risk_json_path.is_file():
        return (
            jsonify({"error": (
                f"No analysis found for '{clip_key}'. Run Analyze first: the "
                "report is built from that run's artifacts, not by re-running "
                "the pipeline."
            )}),
            404,
        )

    try:
        risk_assessments = _json.loads(risk_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Could not read the analysis: {type(exc).__name__}: {exc}"}), 500

    summary = payload.get("summary") or {}
    window_scores = payload.get("window_scores") or []
    overall_risk = payload.get("overall_risk") or "Unknown"
    saliency = payload.get("saliency")
    selected_window = payload.get("selected_window")

    report = build_incident_report(
        summary=summary,
        risk_assessments=risk_assessments,
        window_scores=window_scores,
        overall_risk=overall_risk,
        saliency=saliency,
        selected_window=selected_window,
    )
    if (payload.get("format") or "json").lower() == "text":
        return app.response_class(
            render_incident_report_text(report),
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment; filename=incident_{stem}.txt"},
        )
    return jsonify(report)


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
            if video_path is None or not video_path.exists():
                return (
                    jsonify({"error": (
                        f"No video available for clip '{clip_key}' under the "
                        "dataset root."
                    )}),
                    400,
                )
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
    # The severity label is a configured constant, and the dashboard renders it
    # as a prominent badge. Ship its validation status alongside so the UI can
    # never show a level without the caveat that qualifies it.
    risk_status = risk_presentation()
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
            "risk_status": risk_status,
            "provenance_legend": describe_provenance_legend(),
        }
    )


@app.route("/outputs/demo/<path:filename>")
def outputs_demo(filename):
    return send_from_directory(str(DEMO_OUTPUT_DIR), filename)


if __name__ == "__main__":
    from run_demo import DEMO_CLIPS as _startup_clips

    _preflight = run_preflight(
        repo_root=REPO_ROOT,
        temporal_csv=PRIMARY_CSV,
        yolo_weights=Path(DEFAULT_MODEL_PATH),
        demo_videos={k: Path(v["video"]) for k, v in _startup_clips.items()},
    )
    print(_preflight.render())
    print()
    if not _preflight.replay_ready:
        # Refuse rather than start a dashboard whose first click will fail.
        print("Refusing to start: the standard demo cannot run. Fix the REPLAY "
              "failures above.")
        raise SystemExit(2)

    print(f"Repo root:  {REPO_ROOT}")
    print(f"Model:      {DEFAULT_MODEL_PATH}")
    print(f"Scores CSV: {PRIMARY_CSV} ({len(_score_source.clips())} clips)")
    print("Serving on http://127.0.0.1:5000  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
