# Explainable AI-Based Abnormal Event Detection and Risk Assessment in Surveillance Videos

## Current implementation

The project currently implements a Phase 1--6 foundation, plus a temporal
violence-detection branch that is now trained, evaluated, and wired into the
event/risk pipeline:

1. OpenCV video validation, loading, metadata inspection, and previewing.
2. YOLOv8s frame-by-frame object detection and annotation.
3. Two retained tracking paths: an inspectable greedy IoU tracker and an Ultralytics ByteTrack path with a BoT-SORT fallback.
4. Basic rule-based abnormal-event detection: stationary person, restricted-area entry, generic crowding, and generic proximity/interaction.
5. Transparent, configurable rule-based risk assessment for those event records.
6. Grad-CAM at two levels: `src/temporal_gradcam.py` explains the R3D-18
   **violence decision** (the branch that actually classifies Fight/NonFight),
   surfaced on demand in the demo; `src/yolo_gradcam.py` explains a YOLOv8s
   person detection, which is a different question and cannot explain the
   violence decision.
7. A temporal violence-detection branch (R3D-18, see below) whose sliding-window
   scores feed a frozen aggregation rule (`src/temporal_event_adapter.py`),
   producing a `Temporal Violence Signal` event that `RiskAssessor` maps to
   `High` risk alongside the Phase 4 rules. See
   `docs/temporal_risk_integration.md`.

Dashboard/UI, SHAP, and stabbing detection are not implemented. Crowding and
proximity remain generic geometric signals, not violence or stabbing
classifiers. Violence detection itself *is* now implemented end-to-end (see
point 7). The clean-baseline checkpoint (`best.pt`, 132.74 MB) is now present
on this machine, so the system runs in either of two explicitly selected
modes: **replay**, which reads the committed per-window probabilities and runs
no temporal model, and **live**, which runs R3D-18 on the frames of a video
supplied now and needs no precomputed score at all. Replay stays the default
for the reproducible research presentation. See `docs/demo_instructions.md`.

A browser UI for this pipeline now exists -- see "Web demo UI" under Runners
below and `docs/web_ui.md`.

## Project structure

```text
data/          Input media. `person_fixture.mp4` is supplied by the user; data/rwf2000 holds RWF-2000.
docs/          Project documentation.
models/        Local weights, including `yolov8s.pt`.
outputs/       Generated videos, frames, tracking JSON, risk JSON, XAI images, and outputs/demo/.
src/           Video loader, detection, tracking, event rules, runners, and temporal_event_adapter.py.
temporal_risk/ Recovered window-score CSVs, manifest, and frozen_aggregation.json (see docs/temporal_risk_integration.md).
tests/         Unit and optional end-to-end integration tests.
tools/         select/apply_frozen_aggregation.py, run_demo.py, and other one-off scripts.
web/           Browser UI (Flask) for the pipeline -- see docs/web_ui.md.
```

## RWF-2000 dataset pipeline

RWF-2000 trains only the learned temporal violence branch. It is not the dataset
for the project overall, and crowd/interaction anomaly detection remains the
existing geometric analyzers.

| Script | Role |
|---|---|
| `src/rwf2000_kaggle.py` | Kaggle-side: find/verify archive parts, extract, recover filenames Linux cannot write, reconcile |
| `tools/build_kaggle_cell.py` | Packages the modules above into the one self-contained `kaggle_bootstrap_cell.py` |
| `src/prepare_rwf2000_dataset.py` | Verify checksums, discover layout, inspect clips, leakage checks, write metadata |
| `src/rwf2000_validation.py` | Readiness gate: classes, readability, fps/frames/resolution, leakage, split partition |
| `src/rwf2000_config.py` | Provenance, checksums, split policy, recorded leakage, initial hyperparameters |
| `src/rwf2000_splits.py` | Official 400 / leakage-excluded 6 / primary 394 evaluation sets |
| `src/rwf2000_dataset.py` | Torch datasets for training and evaluation |

### Reproducing the dataset preparation

Local (archive already verified in `data/rwf2000_archives/`):

```bash
python src/prepare_rwf2000_dataset.py --root data/rwf2000     --archives data/rwf2000_archives --skip-download
```

Kaggle (see `docs/cloud_setup_rwf2000.md` for the full procedure and the
filename-length problem it solves):

Paste the entire contents of `kaggle_bootstrap_cell.py` into one notebook cell
and run it. It is self-contained: no internet, no `pip install`, no py7zr, and
no access to this source tree. Regenerate it after editing any embedded module:

```bash
python tools/build_kaggle_cell.py
```

Then gate on readiness before any training:

```python
from pathlib import Path
from rwf2000_validation import validate_dataset
from prepare_rwf2000_dataset import resolve_dataset_root
print(validate_dataset(resolve_dataset_root(Path("/kaggle/working/rwf2000"))).to_text())
```

### Validated dataset facts

2000 clips, all 150 frames at 30 fps (5.0 s). Official split 1600 train / 400
held-out, balanced 1000 Fight / 1000 NonFight. Six held-out clips are
byte-identical duplicates of training clips **in the official release**; they
stay on disk and in the official 400, and are excluded only from the 394-clip
primary evaluation set. Headline metrics are reported on those 394.

**Never resolve dataset paths by hand.** The archive nests a top-level
`RWF-2000/`, and some clip names exceed platform filename limits. Use
`resolve_dataset_root()` for the root and `rwf2000_splits.clip_path()` for
clips; both are already long-path safe.

## Temporal violence branch (R3D-18)

There are **two** trained runs of this branch. They must not be conflated,
and only the second is used downstream:

1. **Experiment 1** (`notebook5d98537e15` V2) — an early run whose decision
   threshold (0.14) was selected on the same 394 clips used for reporting.
   Its own docs (`docs/temporal_baseline_results.md`) flag this as
   "optimistically biased." Kept for the record, not used for anything
   downstream.
2. **Clean baseline** (`notebook7bb9a86555` V3, `clean_experiment/`) — the
   run this project now builds on. It carves a validation subset out of
   train (`validation_fraction=0.15`) and selects both early-stopping and
   the decision threshold on that carved subset, touching the 394-clip
   primary set only once, for final reporting. Full numbers:
   `docs/clean_baseline_results.md`.

Clean-baseline headline (394 leak-free held-out clips, threshold **0.16**,
checkpoint from **epoch 12**, best carve-validation ROC-AUC
0.935342732134176):

| Accuracy | Precision (Fight) | Recall (Fight) | F1 | Specificity | ROC-AUC | PR-AUC | ms/clip |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8325 | 0.7913 | 0.9100 | 0.8465 | 0.7526 | 0.9251 | 0.9260 | 31.01 |

Confusion matrix: TP=182, FP=48, TN=146, FN=18.

R3D-18 initialised from Kinetics-400, `stem`/`layer1`/`layer2` frozen. This
is still a baseline, not a tuned final result -- the point of the clean run
is trustworthy numbers, not better ones.

### Sliding-window scoring and temporal aggregation

The clean-baseline checkpoint (`best.pt`, epoch 12) was then run in
**sliding-window** mode (`clip_length=16`, `stride=8`, 17 windows/clip) over
two splits, producing per-window `fight_probability` scores:

| split | clips | windows | output |
|---|---:|---:|---|
| carve | 240 | 4080 | `temporal_risk/carve_window_scores.csv` |
| primary | 394 | 6698 | `temporal_risk/primary_window_scores.csv` |

`tools/select_temporal_aggregation.py` selects how those per-window scores
become one clip-level decision -- **on the 240-clip carve split only**
(any / k-of-n / mean / max candidates; objective: maximise recall subject to
precision >= 0.80 and FPR <= 0.20). The frozen result
(`temporal_risk/frozen_aggregation.json`): **`max` aggregation, threshold
0.14** -- precision 0.8019, recall 0.7143, F1 0.7556 on carve.

`tools/apply_frozen_aggregation.py` then applies that frozen rule, unchanged,
to the 394-clip primary split (never used for selection): precision 0.788,
recall 0.80, F1 0.794. This is honestly lower than the whole-clip clean
baseline above -- sliding-window aggregation and single-pass whole-clip
inference are different regimes evaluated on the same clips, not
interchangeable numbers. See `docs/temporal_risk_integration.md` for the
full comparison and why the drop is expected rather than a bug.

### Integration

`src/temporal_event_adapter.py` turns a clip's window scores into a
`Temporal Violence Signal` `EventDetection` using the frozen rule above,
tagging its confidence `measured_model_probability` (as opposed to the
Phase 4 rule engine's fixed 0.9/0.95 confidences, tagged
`declared_rule_constant` -- see `risk_assessment._confidence_provenance`).
`RiskAssessor` maps `Temporal Violence Signal` to `High` risk. `tools/run_demo.py`
runs the full pipeline end-to-end on a real clip; see `docs/demo_instructions.md`.

Live inference (frames -> `TemporalInferenceEngine` -> window scores, no CSV)
is supported by the same adapter interface but not yet exercised end-to-end,
because the clean-baseline checkpoint (132.74 MB) has not been downloaded to
this machine -- the demo instead uses the real, precomputed window scores
above, which is explicitly an acceptable substitute for a working final-review
demo, not a shortcut that fabricates results.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Runners

### Phase 1 video preview

`video_loader.py` provides the Phase 1 preview functions. `main.py` no longer runs this preview; it runs the ByteTrack path below.

### Phase 2 detection demo

```bash
python src/run_phase2.py
```

Runs YOLOv8s annotation on the sample image and sample video.

### ByteTrack export path

```bash
python src/main.py
```

Runs Ultralytics ByteTrack (with a BoT-SORT fallback) and writes a tracked video plus `outputs/tracking_results.json`. This path currently exports tracking records but does not invoke event rules.

### Phase 4 IoU-tracking and event-rule demo

```bash
python src/run_phase3.py
```

Despite its legacy filename, this uses the greedy IoU `SimpleTracker`, invokes stationary, restricted-area, crowding, and proximity rules, and writes `outputs/phase4_abnormal_events.mp4`.

### Full pipeline demo (YOLO + tracking + rules + temporal violence signal + risk)

```bash
python tools/run_demo.py --clip fight      # a real RWF-2000 Fight clip -- correctly flagged
python tools/run_demo.py --clip nonfight   # a real RWF-2000 NonFight clip -- correctly not flagged
python tools/run_demo.py --clip fp         # a real RWF-2000 NonFight clip the model false-positives on
```

Writes an annotated video, a risk-assessment JSON, and a markdown explanation
report to `outputs/demo/`. See `docs/demo_instructions.md` for what each
output means and how to read it during the review.

### Web demo UI (browser)

A local browser front end for the same pipeline above, as a guided six-step
workflow -- Home, Setup, Processing, Results, Explain, Report. It calls
`tools/run_demo.py`'s own `run_demo()` function directly; it does not
reimplement or change any detection/tracking/temporal/risk logic.

Note the wording: the dashboard shows the **measured model probability**, the
observed spatial evidence, and a **configured severity/risk interpretation**
that is explicitly not a validated risk model. It does not show a "risk
confidence" or a risk probability, because no such calibrated quantity exists
in this project.

See `docs/web_ui.md` for the architecture and
**`docs/FINAL_REVIEW_CHECKLIST.md`** for the review-day runbook.

```bash
.venv\Scripts\activate
python web\server.py
```

Then open <http://127.0.0.1:5000> in a browser. Choose an inference mode
(REPLAY or LIVE MODEL), pick a video (built-in preset, any of the 394
primary-split clips, or an upload), and click **Start Analysis**. A replay run
takes the same ~30-90s on CPU as the command-line demo, because it is the same
code; a live run additionally spends roughly 0.65s per 16-frame window on
R3D-18 inference.

The server checks its own environment before serving and refuses to start if
the standard replay demo cannot run. A missing R3D checkpoint blocks live
inference and saliency only -- replay needs no checkpoint and keeps working.

**Startup preflight.** The server checks its own environment before serving and
prints the result in three tiers. The distinction matters: the standard demo is
**replay** — temporal probabilities come from the committed
`temporal_risk/primary_window_scores.csv` and no model is run for them — so a
missing R3D checkpoint blocks *live inference and saliency*, not the standard
demo. Replay failures are fatal and the server refuses to start rather than
serve a dashboard whose first click will fail. The same report is available at
`GET /api/preflight`. Nothing ever falls back silently from live to replay.

**Two modes, chosen explicitly.** The dashboard asks for a mode before it asks
for a video, and the result banner states which one produced what is on screen.
*REPLAY* replays the committed probabilities for the 394 primary-split clips and
runs no temporal model. *LIVE MODEL* accepts any video — including one that has
never been scored — and runs YOLO, tracking, the spatial fusion feature and an
actual R3D-18 forward pass per complete 16-frame window. The mode is never
inferred and never falls back: a live request that cannot run live fails with
an actionable error rather than quietly returning a replay. Live inference is
**offline, not real time** — roughly 0.65 s per window on this CPU.

**Temporal window timeline.** The temporal branch scores overlapping 16-frame
windows at stride 8 and takes the maximum; it does not score the clip as a
whole. The timeline shows every scored window with its frame range, probability
and per-window decision. Click a bar or row to select it for saliency. Only
genuinely scored windows are offered, so a selected window can always be
reconstructed from source frames. **Clicking a window does not run the model.**
In replay the probabilities shown are replayed; in live mode they were produced
by a forward pass during that run.

**Evidence fusion.** The operational pipeline implements a configured
evidence-fusion layer combining the temporal probability with an interpretable
spatial motion feature: mean centroid displacement over the implementation's
two-appearance history offset, normalised by the mean person-group diagonal
(a ratio of means). That numerator is **not** a consecutive-frame speed — the
accessor it uses returns `history[-2]`, so each sample skips one intervening
appearance of the track. The frozen thresholds and every locked result were
generated with exactly this implementation, so it is preserved rather than
corrected; see **`docs/SPATIAL_FEATURE_SEMANTICS.md`**. Thresholds, weight and
the percentile reference distributions are read from the frozen protocol, not
restated in code. Because
the spatial feature is a whole-video aggregate, fusion is **offline whole-video
evidence fusion** and is not a causal mid-video alarm; the temporal-only signal
can still fire earlier. In the frozen confirmatory evaluation, fusion did
**not** produce a statistically significant improvement over the temporal-only
baseline (exact McNemar p = 0.771), so the temporal-only rule remains the
system's decision. Fusion is implemented because the research question concerns
an explainable multi-signal system, not because it performs better.

**Gradient-based temporal saliency.** On demand only, for the selected window
(`POST /api/explain`). It needs a backward pass as well as a forward one, about
2.6 s per window on CPU, so it is never computed for every window of every
analysis. The map is a gradient-based localisation diagnostic: it is **not**
causal, **not** ground-truth localization, **not** attention, and has **not**
undergone a faithfulness test (`faithfulness_tested` is `false` on every
result). The panel also states that its 16 displayed slices are interpolated
from only 2 temporal positions the backbone actually resolves.

**Incident report export.** After an analysis, download a JSON or plain-text
report (`POST /api/report`). It is built from that run's artifacts and
recomputes nothing. Rules that carry no confidence keep none — no value is
invented — the configured severity label is never converted into a probability,
and the validation disclaimer and limitations travel inside the report body
rather than in a footnote.

**Risk wording.** The severity label is a configured mapping from event type,
displayed with "Configured severity/risk interpretation — not validated as a
risk model". It is not calibrated, not learned, and not validated against any
outcome. No real-time deployment claim is made anywhere: measured throughput is
roughly seven times slower than real time on CPU.

## Real-person fixture verification

The bundled `data/sample_video.mp4` is a 60-frame, 320x240 sample that yields zero YOLO detections. It is not a valid end-to-end people-detection fixture.

To perform the Phase 2--4 integration verification, provide a short, lawfully usable surveillance-style MP4 with at least one clearly visible person at:

```text
data/person_fixture.mp4
```

Use a stable camera, normal lighting, sufficient resolution for YOLOv8s, and keep at least one person visible for two or more consecutive frames. Then run:

```bash
python -m unittest tests.test_end_to_end_fixture -v
```

The optional test runs:

```text
video loader -> YOLOv8s detection -> SimpleTracker -> abnormal-event rules
```

It requires at least one detection, a detected `person`, and a persistent tracking ID. It writes `outputs/tracking_results.json` and `outputs/risk_assessment.json`, and prints emitted event types. The test is intentionally skipped until `data/person_fixture.mp4` exists.

## Phase 5 risk assessment

`src/risk_assessment.py` maps the existing event records without any machine-learning prediction:

- `Stationary Person` -> `Low` (confidence provenance: `declared_rule_constant`)
- `Restricted Area Entry` -> `Medium` (confidence provenance: `declared_rule_constant`)
- `Crowding` -> `Medium`
- `Proximity/Interaction` -> `Low`
- `Temporal Violence Signal` -> `High` (confidence provenance: `measured_model_probability`; see `src/temporal_event_adapter.py`)
- Unmapped event types -> `Unknown` (no fabricated risk level or confidence)

The mapping is configurable by constructing `RiskAssessor` with a different event-to-level dictionary. Each JSON record contains the event type, risk level, mapping reason, existing confidence when available, frame/timestamp fields when supplied, involved object IDs, and event evidence. The fixture runner writes results to `outputs/risk_assessment.json`.

## Generic crowd and interaction rules

`Crowding` counts unique active person track IDs in each snapshot. Its technical-validation defaults are `minimum_person_count=3` and `persistence_frames=3`; it emits once after the condition persists and resets after it clears.

`Proximity/Interaction` evaluates every pair of active person tracks. It divides centroid distance by the diagonal enclosing all current person boxes because the existing tracking snapshot does not include image dimensions. The defaults are `normalized_distance_threshold=0.20` and `persistence_frames=3`; each pair emits once after sustained proximity and resets when separated.

Both rules are configured through `EventRule` parameters in `src/event_rules.py`. They are generic geometric abnormal-behaviour signals only, not crowd-safety research findings and not violence/stabbing classifiers. `person_fixture.mp4` is a small technical validation fixture, not a research dataset and not evidence of generalization.

## Test suite

```bash
python -m unittest discover -s tests -v
```

The existing abnormal-event unit test runs without a video fixture. The optional integration test is skipped until the real-person fixture is provided.

## Current phase status

- Phase 1 — Video Loading: completed.
- Phase 2 — YOLO Detection: partially completed; model loading/inference works, but no bundled real-person positive fixture exists.
- Phase 3 — Tracking: partially completed; both paths are implemented, but no bundled real-person end-to-end evidence exists.
- Phase 4 — Abnormal Event Detection: partially completed; two rules and their unit test exist, but real-video validation awaits the fixture.
- Phase 5 — Risk Assessment: completed; transparent mappings and fixture output are covered by the passing test suite.
- Phase 6 — Explainable AI: Grad-CAM covers the **R3D-18 violence decision**
  (`src/temporal_gradcam.py`), aligned to source frames and surfaced in the demo,
  as well as YOLO person detection (`src/yolo_gradcam.py`). Faithfulness is
  deliberately **not** claimed: no deletion, insertion or counterfactual test has
  been run, and `faithfulness_tested` is `false` on every result.
- Confirmatory fusion experiment — **completed and permanently closed**. Decision C:
  no meaningful improvement from spatial/temporal fusion over the temporal-only
  detector (exact McNemar p = 0.771). See **`docs/CONFIRMATORY_FUSION_RESULT.md`**.
  The 394-clip primary split is now permanently spent.
- Temporal violence branch — clean baseline completed: R3D-18 fine-tuned on RWF-2000, accuracy **0.8325 / ROC-AUC 0.9251** on the 394 leak-free clips at the frozen threshold 0.16 (see "Clean baseline" above; supersedes the earlier, threshold-biased Experiment 1 number). Sliding-window aggregation selected on a carve split and applied to primary (`temporal_risk/`); integrated into the event/risk pipeline via `src/temporal_event_adapter.py`, producing a `Temporal Violence Signal` event mapped to `High` risk. Demo: `tools/run_demo.py`, and a browser UI on top of it (`web/`, see `docs/web_ui.md`).
- Phase 7: not started.

The real-person fixture verification has also completed Phases 2--4: YOLO detection, persistent tracking IDs, and the currently implemented event rules are covered by the passing integration test.

## Phase 6 explainable AI

Two Grad-CAM implementations answer two different questions. Only one of them
explains the violence decision.

**`src/temporal_gradcam.py` — the violence decision.** Grad-CAM against the R3D-18
task logit, hooking `layer4`. For a 16-frame window that layer emits
`(1, 512, 2, 7, 7)`: the backbone resolves only **two** temporal positions, and the
16 displayed slices are interpolated from them — recorded in the payload so the
upsampling is never mistaken for frame-level detail. Windows are 16 *consecutive*
frames, so slice *i* maps to source frame `first_frame + i`; the non-contiguous
whole-clip regime is **refused** rather than given fabricated alignment. It also
refuses any checkpoint that is not task-specific, because a random head still
produces a confident-looking map.

**`src/yolo_gradcam.py` — a person detection.** Explains the raw YOLO class score
for a post-NMS detection. It says *where a person is*, not why a clip was called
violent, and must not be presented as an explanation of the violence decision.

Neither is a faithfulness-tested explanation. Grad-CAM localises where a logit's
gradient concentrates; a map can be plausible and wrong.

To generate an explanation from the first frame of the real-person fixture:

```bash
python -m unittest tests.test_yolo_gradcam -v
```

The output is saved as `outputs/test_yolo_gradcam_person.jpg` during the test; the module default is `outputs/yolo_gradcam_person.jpg`. The overlay combines the original frame with a Grad-CAM heatmap derived from gradients through YOLOv8s's final convolutional feature layer.

Limitations: this is a class-score localization explanation, not a causal explanation, event explanation, or risk explanation. The implementation reloads the same local weights for the differentiable pass because Ultralytics prediction mode caches inference tensors that cannot participate in autograd. SHAP is not implemented.
