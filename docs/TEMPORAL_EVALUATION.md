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

Uncertainty for both, and what may be claimed when comparing them, is in
**`docs/SYSTEM_RESULT_UNCERTAINTY.md`** (Step 5). In short: the sliding-window
result is 0.7893 [0.7327, 0.8421] at the source-video level, the clean baseline
is 0.8325 [0.7924, 0.8661] uncorrected, and **no formal comparison between them
is possible** because the clean baseline's per-clip predictions were never
recovered.

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

Independently cross-checked against scikit-learn 1.9.0 in Step 4 (§9):
agreement to one unit in the last place on both AUCs, exact on everything
else. scikit-learn remains optional and is not a project dependency.

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

> **Step 5 correction.** The figures in this section are *conditional on
> detection*. The censoring-aware quantiles over **all 200** Fight clips are
> median **0.800 s** and p75 **3.733 s** — not 0.533 s and 1.600 s. Both are
> correct answers to different questions; see
> `docs/SYSTEM_RESULT_UNCERTAINTY.md` §6 before quoting either.

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

## 8. Confidence intervals

Added in Step 4. Computed by `tools/metric_confidence_intervals.py` from the
committed 394-clip Experiment-1 predictions. Nothing was retrained and no point
estimate changed - these are intervals around numbers that already existed.

### What the intervals mean

A 95% interval here describes **sampling variability of this evaluation set at
this fixed operating point**. Repeat the evaluation on a different sample of
clips from the same distribution and the metric would land inside the interval
about 95 times in 100.

They do **not** describe: generalisation to other footage or camera types;
uncertainty from the single training seed; or the model-selection bias
Experiment 1 carries by monitoring on its own reporting set (see
`docs/EXPERIMENT_REPRODUCIBILITY.md`). Those are separate and larger concerns
that no interval on a fixed prediction file can address.

### Tier 1 - Wilson score intervals, denominator fixed by the split

n = 394 (200 Fight, 194 NonFight), threshold 0.5.

| Metric | Point | 95% CI | Counts | Width |
|---|---:|---|---:|---:|
| accuracy | 0.8604 | [0.8227, 0.8912] | 339/394 | 6.85 pp |
| recall (sensitivity) | 0.8200 | [0.7609, 0.8671] | 164/200 | 10.62 pp |
| specificity | 0.9021 | [0.8521, 0.9364] | 175/194 | 8.43 pp |
| prevalence (Fight) | 0.5076 | [0.4584, 0.5567] | 200/394 | 9.83 pp |

Each is a count of successes out of a denominator fixed before any prediction
was made, so a binomial interval is appropriate.

**Method: Wilson score interval**, not Wald. Wald is symmetric, can fall outside
[0, 1], and collapses to zero width at 0 or 100 percent - which would be an
outright false claim at the extremes. The Wilson bounds are the two roots of
`(p_hat - p) / sqrt(p(1-p)/n) = +/- z`, computed in closed form with
`z = 1.9599639845400536` from `statistics.NormalDist().inv_cdf`. No scipy.

### Tier 2 - Wilson intervals, conditional on a model-chosen denominator

| Metric | Point | 95% CI | Counts | Width |
|---|---:|---|---:|---:|
| precision (PPV) | 0.8962 | [0.8435, 0.9325] | 164/183 | 8.90 pp |
| NPV | 0.8294 | [0.7728, 0.8741] | 175/211 | 10.13 pp |

Precision divides by the number of clips the model *chose* to call Fight. That
denominator is a random variable, not a design constant. Reporting a Wilson
interval conditional on the observed count is standard and defensible, but it
is a different object from the recall interval and is flagged
`denominator_is_random` in the machine-readable output.

### Tier 3 - percentile bootstrap; not binomial proportions

| Metric | Point | 95% CI |
|---|---:|---|
| F1 | 0.8564 | [0.8189, 0.8918] |
| ROC-AUC | 0.9430 | [0.9205, 0.9630] |
| PR-AUC | 0.9474 | [0.9268, 0.9657] |

10,000 resamples, seed 42, stratified by class so every resample keeps the
200/194 design. Percentile method, not BCa. Fully reproducible.

**No Wilson interval is claimed for any of these, and applying one would be
wrong.** F1 is a harmonic mean of two overlapping proportions with different
denominators. ROC-AUC and PR-AUC are rank statistics over all positive-negative
pairs, not counts out of a denominator. DeLong's variance estimator for ROC-AUC
was not implemented in this step.

### The independence assumption is violated

Every interval above assumes independent evaluation units. **The 394 clips are
not independent.**

| | |
|---|---:|
| Clips | 394 |
| Distinct source videos | **174** |
| Mean clips per source | 2.26 |
| Largest single source | 13 clips |
| Clips sharing a source with another clip | 274 (69.5%) |
| Singleton sources | 120 |
| Sources contributing to **both** classes | 25 |

Clips cut from one video share camera, scene, lighting and subjects, so their
outcomes are correlated and the effective sample size is below 394. The
clip-level intervals above are therefore **anti-conservative - too narrow**.

Grouping is **inferred** from the `<source_id>_<index>.avi` filename convention,
which matches all 394 primary and all 240 carve clips. It is not read from
dataset metadata; this repository holds no per-clip source-video record.
`rwf2000_config.SplitPolicy.validation_grouped_by_source_video` shows the
dataset authors treated source video as a real grouping, which is supporting
evidence, not proof.

To size the effect rather than assume it away, the same bootstrap was run
resampling whole source videos instead of clips:

| Metric | Clip-level (stratified) | Source-video-level (clustered) | Widening |
|---|---|---|---:|
| accuracy | [0.8274, 0.8934] 6.60 pp | [0.8145, 0.9042] 8.97 pp | **1.36x** |
| F1 | [0.8189, 0.8918] 7.29 pp | [0.7983, 0.9025] 10.42 pp | **1.43x** |
| ROC-AUC | [0.9205, 0.9630] 4.24 pp | [0.9162, 0.9661] 4.99 pp | **1.18x** |
| PR-AUC | [0.9268, 0.9657] 3.89 pp | [0.9145, 0.9698] 5.53 pp | **1.42x** |

**Any interval quoted in a write-up should be the clustered one, or the
clip-level one with this limitation stated alongside it.** Quoting the
clip-level interval alone overstates the precision of the result by roughly
20-45% depending on the metric.

### Reproduce

```bash
python tools/metric_confidence_intervals.py \
    --json outputs/step4_confidence_intervals/intervals.json
```

---

## 9. Independent metric cross-check

Added in Step 4. `src/temporal_metrics.py` reimplements ROC-AUC and PR-AUC in
numpy to avoid a scikit-learn dependency, and had never been compared against an
independent implementation.

`tools/crosscheck_metrics_sklearn.py` performs that comparison when
scikit-learn is importable, and reports SKIPPED (exit 0) when it is not.
scikit-learn is **not** in `requirements.txt` and is not installed in the
project venv; the test that exercises the comparison skips cleanly there.

**Result, against scikit-learn 1.9.0** (installed into an isolated directory so
the project environment was not modified):

| Metric | `temporal_metrics` | scikit-learn | Difference |
|---|---:|---:|---:|
| accuracy | 0.8604060913705583 | 0.8604060913705583 | 0 |
| precision | 0.8961748633879781 | 0.8961748633879781 | 0 |
| recall | 0.8200000000000000 | 0.8200000000000000 | 0 |
| F1 | 0.8563968668407310 | 0.8563968668407310 | 0 |
| ROC-AUC | 0.9430283505154640 | 0.9430283505154639 | +1.11e-16 |
| PR-AUC | 0.9473887310809866 | 0.9473887310809868 | -1.11e-16 |

Confusion matrix identical (TN 175 / FP 19 / FN 36 / TP 164). The two AUC
differences are one unit in the last place - floating-point summation order, not
a difference in definition.

**`temporal_metrics` is independently validated.** This closes the open item
carried since the Step 0 audit.

### This does not close the rounding limitation

Two different questions, and the cross-check answers only the first:

1. *Is the code correct?* - scikit-learn vs `temporal_metrics` on the **same**
   6-decimal data. Both see identical input, so agreement tests the code.
   **Answered: yes, to 1 ULP.**
2. *Can the original recorded value be recovered?* - either implementation vs
   the recorded 0.9432 / 0.9475. Both differ by ~1.7e-4 and ~1.1e-4 because
   `predictions.csv` stores 6 decimals, tying 105 of 394 rows (48 clips at
   exactly `1.000000`). That destroys ordering information a rank statistic
   needs. **Still unanswerable, and unaffected by any cross-check.**

Step 3 established that the recorded ROC-AUC lies inside the interval those
ties allow, [0.9428092784, 0.9432860825]. That finding stands unchanged.

---

## 10. Remaining issues

Not addressed by Step 2; listed so they are not mistaken for solved.

1. **The clean baseline cannot be re-derived here** — needs `best.pt`.
2. **Single seed, single carve split.** No variance estimate for any metric.
3. ~~**No confidence intervals.**~~ Closed in Step 4 (§8). The clip-level
   accuracy interval is [0.8227, 0.8912]; the source-video-clustered
   bootstrap is wider still.
4. **No external baseline comparison.** Only self-comparison across the
   project's own runs — and even the internal whole-clip vs sliding-window
   comparison cannot be tested formally (Step 5; the clean baseline's
   per-clip predictions were never recovered).
5. ~~**Metrics not cross-checked** against scikit-learn.~~ Closed in Step 4
   (§9): agreement to 1 ULP against scikit-learn 1.9.0.
6. **Carve split carries three decisions** (§6).
7. **GPU non-determinism uncontrolled.**
8. **Checkpoint identity unverifiable** — no hash was recorded.
9. **Evaluation units are not independent** (§8). 394 clips come from ~174
   source videos, so every clip-level interval understates uncertainty.
   Newly quantified in Step 4; not otherwise addressed.
10. **The source-video grouping is inferred from filenames**, not read from
    dataset metadata, which this repository does not hold.
