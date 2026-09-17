# Final review checklist — Sentinel Vision dashboard

For the external examiner review. Every figure here was measured on the review
machine; nothing is aspirational. If a number differs on the day, trust the
screen, not this page.

The CLI walkthrough lives in `docs/demo_instructions.md`; this page covers the
browser dashboard.

---

## Before the review

Run these in order from the project root. Expected results are stated so you
can tell at a glance whether something is wrong.

| # | Step | Command / action | Expect |
|---|---|---|---|
| 1 | Environment | `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` | prompt shows `(.venv)` |
| 2 | Dependencies | `pip install -r requirements.txt` | already satisfied; no build step, no internet needed at demo time |
| 3 | Start server | `python web/server.py` | preflight prints, then `Serving on http://127.0.0.1:5000` |
| 4 | Preflight | read the startup banner | `replay ready : True`, `live ready : True`, `saliency ready : True` |
| 5 | Open dashboard | `http://127.0.0.1:5000` | Home page, three green chips top-right |
| 6 | Report export | run one analysis and export JSON + TXT | two files download |

**The server refuses to start if replay is not ready.** That is deliberate — it
is better than a dashboard whose first click fails.

### What the three readiness tiers mean

- **Replay ready** — the committed per-window probabilities
  (`temporal_risk/primary_window_scores.csv`) and YOLO weights are present.
  This is the standard demo path and needs **no** R3D checkpoint.
- **Live Model ready** — `models/temporal_violence/best.pt` is present, so
  R3D-18 can run on a new video.
- **Saliency ready** — same checkpoint, needed for Grad-CAM.

If the R3D checkpoint is missing, replay still works and the UI says so.
Live and saliency are blocked — the system never silently substitutes replayed
scores for live inference.

### Demo clips (already on disk, nothing to prepare)

| Preset | Clip | Ground truth | Why it is in the demo |
|---|---|---|---|
| Fight | `val/Val_Fight/trtrhrt_1049.avi` | Fight | Clean positive. Signal fires, peak probability ~1.0 |
| Nonfight | `val/Val_NonFight/ZCUy99AN_0.avi` | NonFight | Signal does not fire |
| **Fp** | `val/Val_NonFight/39BFeYnbu-I_0.avi` | NonFight | **A real false positive.** Shown on purpose |
| Live upload | `data/person_fixture.mp4` | Not available | Never had a precomputed score — proves live inference |

---

## Demo flow (about 6–8 minutes)

1. **Frame the problem.** Abnormal-event detection in surveillance video, with
   evidence an examiner can inspect rather than a single opaque score.
2. **Home page.** Point at the six pipeline capabilities and the readiness
   chips. Click **New Analysis**.
3. **Setup.** Choose **REPLAY** and the **Fight** preset. Read the summary
   panel aloud: it states the mode, the clip, its ground truth and the exact
   processing path that is about to run. Click **Start Analysis**.
4. **Processing** (~35–60 s). Say what the screen is showing: detection,
   tracking, spatial analysis and temporal windowing are **highlighted as a
   group because they genuinely run together, once per decoded frame** — this
   is not a sequential pipeline, and the UI does not pretend it is. Note that
   there is no percentage, because the backend reports phases, not progress.
5. **Results** opens automatically.
   - **Video** — annotated with YOLO boxes and track IDs, playing in the
     browser.
   - **Risk** — read the disclaimer out loud: *"Configured severity/risk
     interpretation — not validated as a risk model."*
   - **Ground truth** — point out it is labelled *"dataset label, not used by
     the pipeline"*.
   - **Spatial evidence** — three separate columns: OBSERVED EVIDENCE,
     MODEL PROBABILITY, CONFIGURED RISK. Undefined values read *Unavailable*,
     never zero.
   - **Fusion** — four frozen candidates; B0 (temporal-only) is the system
     decision, and the panel states that fusion showed **no statistically
     significant improvement** (exact McNemar p = 0.771).
6. **Timeline.** Click a window. Nothing is recomputed — selection is pure UI.
   The peak window is marked; the 0.14 threshold line is drawn.
7. **Explain.** Pick the peak window, click **Explain this window**
   (~3–7 s). Then say precisely what it is:
   > Gradient-based temporal saliency. **Not** a causal explanation, **not**
   > ground-truth localization, **not** attention, and **not**
   > faithfulness-tested. The backbone resolves only two temporal positions
   > across the window, so the displayed slices are interpolated.

   All of that is on screen; you are reading it, not adding it.
8. **Report.** Open the Report step. The letterhead carries the analysis ID,
   mode, frames processed and evidence status — all from the run. Export
   **JSON** and **TXT**.
9. **The honest failure.** Click **New Analysis**, choose **REPLAY** and the
   **Fp** preset. The system decides *Fight*; ground truth is *NonFight*; the
   UI shows an explicit false-positive notice. Say why it is in the demo: the
   project's own primary-split precision is 0.788, so false positives are
   predicted, and showing one is stronger evidence than hiding it.
10. **Live inference (optional, ~2 min).** **New Analysis** → **LIVE MODEL** →
    Upload tab → `data/person_fixture.mp4` → Start. The temporal row reads
    *"Running R3D-18 inference"* and the result banner reads *LIVE MODEL*.
    This clip has no precomputed score, so the numbers can only have come from
    the model running now.

---

## Measured timings (this machine, CPU)

| Operation | Time |
|---|---|
| Replay analysis (150-frame clip) | 35–60 s |
| Live analysis (204-frame clip, 24 windows) | 60–115 s |
| R3D inference alone | ~0.65 s per 16-frame window |
| Grad-CAM, one window | 3–7 s |
| Browser preview conversion | ~3 s first view, ~0.01 s cached |

Live inference is **offline** processing. No real-time claim is made anywhere.

---

## If something goes wrong

| Symptom | Cause | Do this |
|---|---|---|
| Server refuses to start | replay preflight failed | read the printed remedy; usually the temporal CSV or YOLO weights are missing |
| "Live Model unavailable" | `best.pt` absent | use REPLAY; it needs no checkpoint |
| Video shows "cannot play" | no browser-native encoder available | click **Open annotated video**; the file is complete |
| Explanation unavailable | window incomplete, or no checkpoint | pick a window with 16 full frames |
| Report says "Run Analyze first" | no analysis for that clip yet | run the analysis before exporting |
| Page looks stale | browser refresh resets to a clean Home state | re-run the analysis; nothing is corrupted |

**Backup plan.** If live inference misbehaves on the day, the entire review can
be given in REPLAY mode on the Fight and Fp presets — that path needs no
checkpoint, is deterministic, and exercises every part of the UI except live
R3D.

---

## What not to claim

Preserve these distinctions under questioning:

- Replay **replays committed probabilities**; it does not run the model.
- Live inference is **offline**, not real time.
- Risk severity is a **configured mapping**, not a calibrated or validated
  risk model, and never a probability.
- Grad-CAM is **saliency**, not causality and not ground-truth localization.
- Ground-truth labels are dataset metadata and are **not used** by the
  pipeline's decision.
- Fusion is a **configured** layer; the confirmatory evaluation found no
  statistically significant improvement over temporal-only.
- The primary split is **permanently spent** — one confirmatory evaluation,
  already run and locked.
