# Temporal evaluation

How the temporal violence branch is evaluated in this repository: the splits,
the windowing, the metrics, and the time-to-alarm measurement added in Step 2.

Companion to `docs/EXPERIMENT_REPRODUCIBILITY.md`, which covers protocol
provenance and what can and cannot be re-run here.

---

## 1. Dataset and splits

RWF-2000: 2000 clips, every one exactly **150 frames at exactly 30.00 fps**
(5.000 s) — verified across all 2000, not assumed
(`data/rwf2000_metadata/rwf2000_metadata.json`: `fps_counts = {"30.00": 2000}`,
`frame_count_counts = {"150": 2000}`).

| Set | Clips | Fight | NonFight | Role |
|---|---:|---:|---:|---|
| `train_remainder` | 1360 | 681 | 679 | gradient updates |
| `carve_validation` | 240 | 119 | 121 | early stopping, threshold selection, aggregation-rule selection |
| `primary_evaluation` | 394 | 200 | 194 | reported metrics, touched once |
| `leakage_excluded` | 6 | 0 | 6 | never used anywhere |
| `official_validation` | 400 | 200 | 200 | reference only; contains the 6 leaks |

Built by `src/carved_validation.py`; exposed as datasets by
`build_carved_train_dataset`, `build_carved_validation_dataset`,
`build_primary_evaluation_dataset` in `src/rwf2000_dataset.py`.

### Leakage handling

Six of the 400 official validation clips are **byte-identical copies of
training clips** released under different filenames — confirmed by a
size-then-full-MD5 scan, recorded with both filenames and the shared digest in
`rwf2000_config.LEAKED_VALIDATION_CLIPS`, and re-derivable with
`rwf2000_splits.verify_recorded_leakage()`.

They are excluded **at selection time, by relative path**. No file is moved,
renamed or deleted; `val/` still holds all 400 exactly as released. All six are
NonFight, which is why the primary split is 200/194 rather than balanced.

They are excluded from evaluation but deliberately **kept in training**: they
are legitimate training data, and removing them would alter the official train
split.

### Disjointness is checked, not asserted

`carved_validation.verify_disjoint()` returns every violation it can find:
pairwise overlap between the three sets, and any confirmed-leaked clip
appearing in any of them. `temporal_training.build_dataloaders` calls it and
refuses to train if it reports anything. `tests/test_carved_validation.py`
runs it against the real dataset.

---

## 2. Preprocessing and sampling

Unchanged by Step 2; documented here because evaluation numbers depend on it.

| Stage | Value |
|---|---|
| Colour | BGR → RGB |
| Resize | (128, 171) bilinear, `antialias=False` |
| Crop | 112 × 112 centre |
| Scale | uint8 → [0, 1] |
| Normalise | mean (0.43216, 0.394666, 0.37645), std (0.22803, 0.22145, 0.216989) |
| Tensor | (1, C, T, H, W) |

These are torchvision's published Kinetics-400 video preset, and
`tests/test_temporal_preprocessing.py` asserts numerical equivalence against
it so the two cannot drift apart. `antialias=False` is deliberate and must not
be "fixed" — it preserves the preprocessing the pretrained backbone expects.

**Whole-clip sampling** (training and clip-level evaluation): 16 frames, one
from each of 16 equal segments across the full 150. Evaluation takes segment
midpoints and is deterministic; training jitters within each segment, seeded on
`(seed, epoch, clip_index)`.

**Sliding-window sampling** (streaming regime): consecutive 16-frame windows at
stride 8.

---

## 3. Window generation

`TemporalClipBuffer(clip_length=16, stride=8)` over 150 frames yields **17
complete windows**:

```
(0,15) (8,23) (16,31) (24,39) (32,47) (40,55) (48,63) (56,71) (64,79)
(72,87) (80,95) (88,103) (96,111) (104,119) (112,127) (120,135) (128,143)
```

`last_frame` is inclusive. The final six frames (144–149) never complete an
eighteenth window and are not padded into one.

**Verified against the committed data.** `tools/score_temporal_windows.py
--verify-geometry` confirms that all 394 primary and all 240 carve clips in the
committed CSVs use exactly this layout — one identical bound sequence across
every clip, zero mismatches. Previously this correspondence was assumed.

---

## 4. Two inference regimes — do not quote them interchangeably

| | Whole-clip | Sliding-window |
|---|---|---|
| Input | 16 frames spanning the full 5 s | 17 × 16 consecutive frames, stride 8 |
| Decision | one forward pass, threshold **0.16** | 17 passes, aggregated by **max ≥ 0.14** |
| Accuracy on primary | 0.8325 | 0.7893 |
| Precision | 0.7913 | 0.7882 |
| Recall | 0.9100 | 0.8000 |
| F1 | 0.8465 | 0.7940 |
| ROC-AUC | 0.9251 | n/a (rule is not a ranking) |
| Confusion (TP/FP/TN/FN) | 182 / 48 / 146 / 18 | 160 / 43 / 151 / 40 |
| Can update mid-clip? | No | **Yes** |

Both are on the same 394 clips. The whole-clip figure is the **model-only upper
bound**; the sliding-window figure is **what the system actually does**, and is
the number that should lead any report of system performance.

The two thresholds are numerically close but are different quantities: 0.16
thresholds a single whole-clip probability; 0.14 thresholds the maximum of 17
window probabilities. `rwf2000_config.SELECTED_DECISION_THRESHOLD` (also 0.14)
is a third, superseded quantity from the biased Experiment 1.

---

## 5. Metrics

`src/temporal_metrics.py`, numpy only, no scikit-learn dependency.

- ROC-AUC via the rank (Mann–Whitney U) identity, ties handled by averaging ranks.
- PR-AUC as step-wise average precision, `Σ (R_n − R_{n−1}) · P_n`, no
  trapezoidal interpolation.
- Degenerate inputs return `None`, never 0. A `None` must be reported as
  "not computable".
- `Fight` is the positive class (index 1).

Not yet cross-checked against a reference implementation — see
"Remaining issues".

---

## 6. Checkpoint selection

Under `carved_validation`:

1. Monitor **ROC-AUC on `carve_validation`** after every epoch.
2. Save `last.pt` every epoch; overwrite `best.pt` only on improvement.
3. Early stop after 5 epochs without improvement (max 30).
4. Select the decision threshold on `carve_validation` — maximise recall
   subject to precision ≥ 0.80, FPR ≤ 0.20, accuracy ≥ 0.8750.
5. Select the aggregation rule on `carve_validation` only
   (`tools/select_temporal_aggregation.py`).
6. Apply everything, frozen, to `primary_evaluation` exactly once
   (`tools/apply_frozen_aggregation.py`).

The recorded run stopped at epoch 17 with best epoch 12, monitor ROC-AUC
0.93534, and selected threshold 0.16.

**Caveat that must be stated in any write-up:** the carve split carries three
separate decisions — early stopping, threshold selection, and aggregation-rule
selection. `primary_evaluation` stays clean throughout, which is the property
that matters, but carve-split metrics are selection-set metrics and must not be
quoted as generalisation evidence.

---

## 7. Time to alarm

New in Step 2. `tools/time_to_alarm.py`. Reads committed CSVs only — no model,
no checkpoint, no video decoding.

### Definition

An **alarm** is the moment the frozen aggregation rule would first fire on a
streaming clip, using exactly the causal replay `tools/run_demo.py` performs:

- A window contributes only once its `last_frame` has been observed. A window
  covering [8,23] is not usable at frame 8; it is usable at frame 23.
- After each window completes, the frozen rule is applied to all windows
  completed so far.
- The **alarm frame** is the `last_frame` of the earliest window at which the
  rule first holds.
- **Time to alarm** = `(alarm_frame + 1) / 30.0` seconds — after observing
  frame index *f*, *f*+1 frames have elapsed.

The rule is applied through `FrozenAggregationRule.decide`, not by
special-casing `max`, so the measurement stays correct if the rule is ever
re-selected as `k_of_n` or `mean`.

Earliest possible alarm: 16/30 = **0.533 s**. Latest: 144/30 = **4.800 s**.

Validated against the live demo: the computed alarm frames for the three
built-in clips (15, none, 23) match `temporal_signal_first_frame` exactly.

### Censoring — read before quoting a mean

A Fight clip the rule never flags has **no alarm time**. It is a missing
observation, right-censored at 4.800 s — not a large alarm time. Averaging over
only the clips that did alarm therefore **understates** real-world latency,
because the hardest clips are precisely the ones excluded. Detected and
censored counts are always reported side by side and never blended.

### Result — primary split (394 clips)

Of 200 Fight clips: **160 alarmed**, **40 censored** (recall 0.800, matching
the recorded aggregation result exactly).

Over the 160 detected clips:

| Statistic | Seconds |
|---|---:|
| mean | **1.272** |
| median | **0.533** |
| std dev | 1.155 |
| min | 0.533 |
| p25 | 0.533 |
| p75 | 1.600 |
| max | 4.800 |

Which window fired first, cumulative over the 160 detected clips:

| Alarmed by end of | Frames | Seconds | Clips | % of detected | % of all Fight |
|---|---:|---:|---:|---:|---:|
| window 0 | 15 | 0.533 | 82 | 51.3% | 41.0% |
| window 1 | 23 | 0.800 | 102 | 63.8% | 51.0% |
| window 2 | 31 | 1.067 | 111 | 69.4% | 55.5% |
| window 4 | 47 | 1.600 | 123 | 76.9% | 61.5% |
| window 8 | 79 | 2.667 | 140 | 87.5% | 70.0% |

Just over half of all detected fights are flagged by the **first** completed
window, 0.533 s in.

### False alarms are slower than true detections

Of 194 NonFight clips, 43 raised a false alarm (rate 0.222). Their timing:

| Statistic | False alarm (s) | True detection (s) |
|---|---:|---:|
| mean | 2.307 | 1.272 |
| median | 1.867 | 0.533 |
| p25 | 0.667 | 0.533 |
| p75 | 3.867 | 1.600 |

True fights are flagged roughly **3.5× sooner by median** than false alarms
arrive. This is an observation from the committed scores, not a tuned result —
but it suggests a latency-aware variant (require an early alarm, or require
persistence before escalating) could trade recall for precision along an axis
the current single-threshold rule does not use. That would be new research and
has not been attempted.

### Carve split, for reference

85 of 119 Fight clips alarmed (recall 0.714, matching
`frozen_aggregation.json`), mean 1.343 s, median 0.800 s, 34 censored.

### Exclusions

- The 6 confirmed-leaked clips: absent from the primary CSV, so never scored.
- 40 Fight clips (primary) / 34 (carve) that never alarm: excluded from the
  timing statistics, reported as censored.
- No clip was excluded for any other reason. Every clip in each CSV appears
  exactly once in the report.

### Reproduce

```bash
python tools/time_to_alarm.py --split primary \
    --json outputs/temporal_evaluation/time_to_alarm_primary.json
python tools/time_to_alarm.py --split carve
```

---

## 8. Remaining issues

Not addressed by Step 2; listed so they are not mistaken for solved.

1. **The clean baseline cannot be re-derived here** — needs `best.pt`.
2. **Single seed, single carve split.** No variance estimate for any metric.
3. **No confidence intervals.** n = 394 gives roughly ±3.7 points on accuracy
   at 95%. Cheap to add; not done.
4. **No external baseline comparison.** Only self-comparison across the
   project's own runs.
5. **Metrics not cross-checked** against scikit-learn.
6. **Carve split carries three decisions** (§6).
7. **GPU non-determinism uncontrolled.**
8. **Checkpoint identity unverifiable** — no hash was recorded.
