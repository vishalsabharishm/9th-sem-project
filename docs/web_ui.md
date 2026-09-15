# Web demo UI

A local browser front end for the existing P0-P4 pipeline: video in, "Analyze",
processed video + risk classification + temporal evidence + explanation out.
It is a thin presentation layer, not a new implementation of the pipeline --
see "What this does and does not do" below.

## Why Flask

The project's own `.venv` already has the full Flask stack installed
(`flask`, `werkzeug`, `jinja2`, `click`, `itsdangerous`, `blinker`,
`markupsafe`) even though nothing in the project used it yet. That made
Flask the architecture that needs **zero new dependencies** -- no
`pip install`, no `requirements.txt` change, no JS build tooling. The UI
itself is a static Jinja2 template plus one CSS file and one vanilla-JS file,
with no CDN or npm dependency, so it works with no internet connection during
a live review.

## What this does and does not do

- `web/server.py` imports `run_demo()` and `DEMO_CLIPS` from
  `tools/run_demo.py` **directly and unmodified**. Every "Analyze" click in
  the browser runs the exact same function
  `python tools/run_demo.py --clip fight` calls from the command line --
  same YOLO model, same tracker, same Phase-4 rules, same frozen temporal
  aggregation rule, same risk mapping. This file does not duplicate,
  re-derive, or alter any of that logic.
- It also imports `PrecomputedWindowScoreSource` (already defined in
  `src/temporal_event_adapter.py`) read-only, to list which of the 394
  primary-split clips have a real precomputed temporal score, and to look up
  ground-truth labels for display.
- The R3D-18 checkpoint (`best.pt`) is still not on this machine. Uploading
  an arbitrary new video therefore cannot get a live temporal score --
  exactly the same limitation `tools/run_demo.py --video-path/--clip-key`
  already has (see `docs/demo_instructions.md`). The UI never fabricates a
  score for such a clip; `/api/analyze` returns a 400 with an explanation
  instead.
- No file under `src/`, `tools/`, `temporal_risk/`, `tests/`, `models/`, or
  `data/` was changed to build this.

## Files

```
web/
  server.py            Flask app: routes + calls into the existing pipeline
  templates/index.html UI markup
  static/style.css      styling
  static/app.js         fetch calls + rendering (includes a small, purpose-built
                         Markdown renderer for the fixed structure
                         write_explanation_report() produces -- headers, bold,
                         pipe tables -- not a general Markdown engine)
```

New runtime-only folder (created automatically, not checked in with content):
`web/uploads/` -- holds videos uploaded through the UI's "Upload" tab.

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | The UI page. |
| `/api/presets` | GET | The 3 built-in demo clips (`DEMO_CLIPS` from `run_demo.py`) with ground-truth labels. |
| `/api/clips` | GET | All 394 primary-split clips that already have a real precomputed temporal score. |
| `/api/analyze` | POST | Runs `run_demo()` on the selected video and returns its real output, plus the generated video/JSON/explanation URLs. |
| `/outputs/demo/<file>` | GET | Serves whatever `run_demo()` just wrote to `outputs/demo/` (video, risk JSON, explanation markdown). |

### `/api/analyze` request shape

`multipart/form-data`.

`mode` selects the pipeline and is **never inferred**:

- `mode=replay` (the default) -- per-window probabilities come from the
  committed CSV and no temporal model runs.
- `mode=live` -- R3D-18 runs on the frames of the supplied video. Requires the
  checkpoint; needs no CSV row and reads none. An unknown mode is rejected, and
  a live request that cannot run live returns an error rather than a replay.

In `mode=replay`, supply one of:

- `preset` = `fight` \| `nonfight` \| `fp` -- runs a built-in clip.
- `clip_key` = one of the 394 keys from `/api/clips` -- resolves the video
  straight from `data/rwf2000/RWF-2000/...` on disk, no upload needed.
- `clip_key` **and** `video` (a file) -- runs the uploaded file's frames
  through the live pipeline, paired with `clip_key`'s precomputed temporal
  score. Only meaningful if the uploaded file is (or closely matches) that
  clip -- the UI says this explicitly.

In `mode=live`, supply one of:

- `video` (a file) -- any video, scored from scratch. No `clip_key` needed.
  The filename is sanitised and the destination checked for containment; only
  `.avi .mp4 .mov .mkv .webm .mpg .mpeg` are accepted.
- `preset` or `clip_key` -- runs the model over that clip instead of replaying
  it. If the clip also has a research score, the response carries a
  `live_note` saying the live number is an engineering artifact and is not the
  locked research score.

Live results are keyed `live/<filename>` and registered in-process so
`/api/explain` and `/api/report` can serve them. The registry holds only paths
the server itself resolved, and it is not persisted across a restart.

### `/api/analyze` response shape

```json
{
  "summary": { ...the exact dict run_demo() returns... },
  "overall_risk": "High | Medium | Low | None",
  "temporal_signal": { ...the Temporal Violence Signal risk-assessment record, plus evidence_parsed... } | null,
  "event_summary": [ {"event_type": ..., "risk_level": ..., "occurrences": ...}, ... ],
  "window_scores": [ {"window_index": ..., "first_frame": ..., "last_frame": ..., "fight_probability": ...}, ... ],
  "mode": "replay | live",
  "mode_label": "REPLAY -- precomputed research evidence | LIVE MODEL -- actual video inference",
  "live_note": "...set only when a live run used a clip that also has a research score..." | null,
  "fusion": { ...configured fusion candidates, or available:false with a reason... },
  "spatial_fusion_feature": { ...measured feature, or spatial_score:null when undefined... },
  "explanation_text": "...the full explanation .md content...",
  "video_url": "/outputs/demo/demo_<stem>.mp4",
  "risk_json_url": "/outputs/demo/demo_<stem>_risk_assessment.json",
  "explanation_url": "/outputs/demo/demo_<stem>_explanation.md"
}
```

`overall_risk` is derived purely from the risk levels `RiskAssessor` already
assigned (High whenever the Temporal Violence Signal fired, else the highest
Phase-4 rule risk level present, else "None") -- it is a display-only
summary, not a new score.

## Concurrency note

`run_demo()` loads the YOLO model and opens its own `VideoCapture`/
`VideoWriter` on every call, exactly like the CLI. `web/server.py` wraps that
call in a `threading.Lock()` purely so two browser tabs analyzing at once
don't race on that shared local state -- it does not change what gets
computed, and a single request still takes the same ~30-90s on CPU as
`python tools/run_demo.py` does from the command line.

## Running it

See the "Web demo UI" section of the main `README.md`.
