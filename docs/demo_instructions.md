# Running the final-review demo

`tools/run_demo.py` runs the complete pipeline -- YOLO detection, tracking,
Phase 4 abnormal-event rules, the temporal violence signal, risk assessment,
and an explanation report -- on one video, live, on this machine.

## Quick start

```bash
python tools/run_demo.py --clip fight
python tools/run_demo.py --clip nonfight
python tools/run_demo.py --clip fp
```

Each run takes roughly 30-90 seconds on CPU (YOLOv8s inference per frame is
the bottleneck; GPU speeds this up if available). Output goes to
`outputs/demo/demo_<clip>.mp4`, `..._risk_assessment.json`, and
`..._explanation.md`.

## The three built-in clips, and why each is included

| `--clip` | video | ground truth | what the demo does |
|---|---|---|---|
| `fight` | `data/rwf2000/RWF-2000/val/Val_Fight/trtrhrt_1049.avi` | Fight | Temporal Violence Signal fires (peak probability ~1.0). This is the clean positive case for the review. |
| `nonfight` | `data/rwf2000/RWF-2000/val/Val_NonFight/ZCUy99AN_0.avi` | NonFight | Signal does not fire (all window scores 0.0). Shows the system does not cry wolf on ordinary footage. |
| `fp` | `data/rwf2000/RWF-2000/val/Val_NonFight/39BFeYnbu-I_0.avi` | NonFight | Signal **fires anyway** -- a real false positive from the primary-split evaluation. Included deliberately: a demo that only shows successes is not evidence, and this project's own numbers (precision 0.788 on primary) predict some false positives will occur. Showing one, and being able to explain it via `explanation.md`'s per-window score table, is a stronger answer to a reviewer's "does this actually work" than hiding it. |

All three are real clips from the RWF-2000 validation split, with real,
already-computed per-window scores from the clean-baseline checkpoint (see
`docs/temporal_risk_integration.md`). Nothing about the temporal signal in
this demo is fabricated or hand-picked to look good -- `fp` is direct
evidence of that.

## What to point at during the review

1. **The video overlay** (top-left HUD): a live-updating "windowed fight
   probability" readout and a red "TEMPORAL VIOLENCE SIGNAL" banner that
   turns on causally -- only once enough windows have been "seen" (in frame
   order) for the frozen aggregation rule to actually fire, the same way a
   real deployment would only know partway through a clip, not at frame 0.
2. **The explanation report** (`outputs/demo/demo_<clip>_explanation.md`):
   the full 17-row per-window probability table, the frozen aggregation
   rule and its parameters, and every risk assessment raised, each tagged
   with its confidence provenance (`measured_model_probability` for the
   temporal signal, `declared_rule_constant` for the Phase 4 rules).
3. **The risk-assessment JSON**: the machine-readable version of the same
   information, suitable for a "here's the actual output artifact" slide.

## Running on a different clip

```bash
python tools/run_demo.py --video-path path\to\some_clip.avi --clip-key "val/Val_Fight/some_clip.avi"
```

`--clip-key` must match a `clip` value in
`temporal_risk/primary_window_scores.csv` exactly (394 real primary-split
clips are available -- 6698 rows / 17 windows per clip; `--video-path` must
point at the matching video file under `data/rwf2000/`). In **replay** mode
there is no support for scoring a clip that was not already scored in Phase B:
replay only ever replays a committed probability, and never invents one. To
score a clip that has no committed score, use **live** mode:
`--temporal-source live --checkpoint models/temporal_violence/best.pt`, which
runs a real R3D-18 forward pass per 16-frame window (offline, not real time).

## If something looks wrong live

- **No detections at all**: `models/yolov8s.pt` must be present (it is,
  22.5 MB, checked into `models/`).
- **`KeyError` / "No precomputed window scores found"**: the clip key does
  not match the CSV exactly, including the `val/Val_Fight/...` prefix and
  file extension.
- **Video won't open**: check the path is correct and the `.avi` codec is
  readable by the OpenCV build in `requirements.txt` (already verified
  working for all three built-in clips as of this writing).
