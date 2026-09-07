# Changelog

Engineering changes made after the Step 0 audit and golden-reference capture.
Research results are never edited; where a number changes, both the old and new
values are recorded.

Baseline for everything below: commit `a2422f1`, captured in full at
`outputs/golden_reference_a2422f1/` (source digests, environment, demo
outputs, test results, and measurements of known defects).

---

## Research programme closed - confirmatory fusion, Decision C

Commits `94f0b78` through `de3fdc3`. The 394-clip primary split is now
**permanently spent**; see `outputs/fusion/FINAL_experiment_lock_record.json`.

Four pre-registered results, two of them nulls:

**Paired clean-vs-sliding comparison** (`94f0b78`). Clip alignment was
*established*, not assumed: the recovered predictions carry no clip identifier,
so re-running the checkpoint locally and a 20,000-draw within-label permutation
test were used to prove the mapping (observed mean |difference| 4.48e-06 against
a best permutation of 2.21e-01). Overall accuracy difference is not
distinguishable (McNemar p = 0.0639), but Fight recall IS significantly worse
under sliding windows (p = 0.000195) while false alarms are slightly better.

**Aggregation selection** (`ac75b8c`). Six configurations at a matched
false-positive budget on the carve split. **No candidate promoted** - every
alternative was strictly worse than `max >= 0.14`, monotonically so as more
windows entered the statistic. Also found that top-k mean is not alarm-monotone:
it can retract a streaming alarm, measured on 2/3/5 clips for k = 2/3/5.

**Contamination discovery** (`c032b82`). A "fresh" validation carve drawn from
the untouched training remainder was source-disjoint from everything and
untouched by selection - but not by training. Measured memorization gap
**+0.244 recall** (0.9583 on training data vs 0.7143 on holdout). The experiment
was abandoned at 120/240 clips and no fusion number was produced from it.
Source-disjoint does not imply training-disjoint.

**Confirmatory fusion** (`202d979`, `604e8cc`, `5a8a03d`). Protocol frozen and
committed before primary was opened, with the percentile reference distributions
stored *inside* the artifact so the confirmatory run could not normalise primary
against itself. Result: **Decision C, no meaningful improvement.** B0 160/43/151/40
vs F2 160/46/148/40; exact McNemar b=25 c=22 p=0.770867; accuracy difference
-0.0076, source-clustered CI [-0.0463, +0.0309]. A development-set advantage of
+14 true positives at equal false positives did not transfer.

Full account: **`docs/CONFIRMATORY_FUSION_RESULT.md`**.

### Explainability, robustness and demo work

`6799594` Grad-CAM for the R3D-18 **violence decision** - the pre-existing
implementation explained a YOLO person detection, which cannot explain the
violence decision and was never wired into the runtime.

`6cbfcbe` the risk layer now declares itself unvalidated **in the data**, not
only in documentation: every emitted record carries `risk_level_is_validated:
false`. The four concerns - detection, evidence, confidence, risk interpretation
- are named apart with the empirical support each actually has.

`883053c` failure-mode characterization from development data only. Three
detection failure modes are class-dependent; missed fights have median
max-window score 0.041 against 0.886 for detected ones, so they are not marginal
cases a threshold change would recover.

`be28648`, `de3fdc3` saliency surfaced on demand in the dashboard (2.6 s/window,
measured), and the severity badge now carries its validation status - it
previously rendered "High" with nothing qualifying it.

No research number was edited. No threshold, feature, protocol or weight
changed.

---

## Step 5 - Uncertainty for the sliding-window system result

Branch `step5-sliding-window-uncertainty`.

Step 4 put intervals on Experiment 1's superseded whole-clip result. Step 5
applies the same machinery to the number the running system actually produces:
sliding-window accuracy **0.7893**. No retraining, no threshold change, no
aggregation-rule change, no split change.

New reference document: **`docs/SYSTEM_RESULT_UNCERTAINTY.md`** - which numbers
go in the paper and what may be claimed about them.

### The result is exactly recomputable without a checkpoint

`primary_window_scores.csv` + `frozen_aggregation.json` reproduce TP=160 FP=43
TN=151 FN=40 and accuracy 0.7893401015228426 exactly. The tool asserts this
first and refuses to report intervals around a point estimate it cannot
reproduce.

### Sliding-window intervals (n = 394)

| Metric | Point | 95% CI (clip-level) | Counts |
|---|---:|---|---:|
| accuracy | 0.7893 | [0.7464, 0.8267] | 311/394 |
| recall | 0.8000 | [0.7391, 0.8495] | 160/200 |
| specificity | 0.7784 | [0.7148, 0.8311] | 151/194 |
| precision | 0.7882 | [0.7269, 0.8388] | 160/203 |
| NPV | 0.7906 | [0.7274, 0.8423] | 151/191 |
| F1 (bootstrap) | 0.7940 | [0.7525, 0.8333] | - |
| ROC-AUC (bootstrap) | 0.8837 | [0.8505, 0.9140] | - |
| PR-AUC (bootstrap) | 0.8943 | [0.8641, 0.9225] | - |

ROC-AUC/PR-AUC here rank **max-of-17-windows**, the quantity the frozen rule
thresholds. They are **not comparable** with the clean baseline's whole-clip
ROC-AUC 0.9251, which ranks a different score.

### Source-video dependence applied directly

Unlike Experiment 1's `predictions.csv` (row index only, forcing a positional
inference in Step 4), the sliding-window scores carry clip names, so grouping
is applied without inference about ordering. 394 clips, 174 inferred sources.

| Metric | Clip-level | Source-video-level | Widening |
|---|---|---|---:|
| accuracy | 8.12 pp | 10.94 pp | 1.35x |
| F1 | 8.08 pp | 13.02 pp | 1.61x |
| ROC-AUC | 6.35 pp | 9.02 pp | 1.42x |
| PR-AUC | 5.84 pp | 9.84 pp | 1.68x |

Grouping remains **inferred** from the filename convention, not proven.

### Time to alarm - a material correction to Step 2's figures

Censoring is administrative and identical (all 40 non-detections at 4.800 s),
and lies entirely above the 80th percentile, so quantiles at or below the
detected fraction are **exact, not estimated**.

| | Over all 200 Fight clips | Conditional on the 160 detected |
|---|---:|---:|
| median | **0.800 s** | 0.533 s |
| q75 | **3.733 s** | 1.600 s |
| mean | *undefined under censoring* | 1.272 s |

**Step 2 reported the conditional figures.** The unconditional p75 is more than
twice the conditional one. Both are correct answers to different questions and
must be labelled. Bootstrap CI for the unconditional median: [0.800, 1.333]
clip-level, [0.533, 1.867] source-video-level.

The 40 censored clips are never imputed as 4.8-second alarms; a test constructs
a case where that imputation would change the answer and asserts the honest
path returns "not reached".

### Clean baseline - Wilson only

Its per-clip predictions were never recovered, so only counts are available.
Wilson intervals from those counts are a computation, not an invention:
accuracy 0.8325 [0.7924, 0.8661], recall 0.9100 [0.8622, 0.9423], specificity
0.7526 [0.6873, 0.8080], precision 0.7913 [0.7342, 0.8388], NPV 0.8902
[0.8332, 0.9294].

No bootstrap and no clustering correction are possible for it, so those
intervals are uncorrected and anti-conservative - and unlike the sliding-window
result, the size of that effect cannot be measured.

### 0.7893 vs 0.8325 cannot be formally compared

Same checkpoint, same 394 clips, different inference regime - exactly the
comparison worth making, and it is untestable. McNemar needs each clip's
outcome under both regimes; the clean baseline's per-clip predictions do not
exist here. An unpaired two-proportion test is also refused: the samples are
the identical clips scored twice, so it would report a meaningless p-value.
Pairing against Experiment 1 instead was considered and rejected - it would
confound a checkpoint change with a regime change, and its rows carry no clip
identifier.

The accuracy intervals overlap, which is **not** evidence of no difference.

### Added

- **`tools/sliding_window_uncertainty.py`**
- **`tests/test_sliding_window_uncertainty.py`** (46 tests, 16 subtests)
- **`docs/SYSTEM_RESULT_UNCERTAINTY.md`**

### Changed

- **`docs/TEMPORAL_EVALUATION.md`** - section 4 cross-links the new document;
  section 7 carries a Step 5 correction box distinguishing conditional from
  unconditional latency; remaining-issues entry 4 updated.

### Unchanged

`src/temporal_metrics.py`, `src/metric_intervals.py` (reused as-is),
`predictions.csv`, `temporal_risk/`, all historical result artifacts, the
golden reference, and the demo.

---

## Step 4 - Confidence intervals and independent metric cross-check

Branch `step4-confidence-intervals`.

Improves the statistical defensibility of the existing evidence without
changing the experiment. No retraining, no threshold change, no split change,
no point estimate altered.

### Confidence intervals

`tools/metric_confidence_intervals.py`, over the committed 394-clip
Experiment-1 predictions. Three tiers, deliberately kept apart:

**Wilson score intervals, denominator fixed by the split**

| Metric | Point | 95% CI | Counts |
|---|---:|---|---:|
| accuracy | 0.8604 | [0.8227, 0.8912] | 339/394 |
| recall | 0.8200 | [0.7609, 0.8671] | 164/200 |
| specificity | 0.9021 | [0.8521, 0.9364] | 175/194 |
| prevalence (Fight) | 0.5076 | [0.4584, 0.5567] | 200/394 |

**Wilson intervals conditional on a model-chosen denominator** - precision
0.8962 [0.8435, 0.9325] (164/183), NPV 0.8294 [0.7728, 0.8741] (175/211).
Flagged `denominator_is_random`, because the number of clips the model calls
Fight is not fixed by the design.

**Percentile bootstrap, not binomial proportions** - F1 0.8564
[0.8189, 0.8918], ROC-AUC 0.9430 [0.9205, 0.9630], PR-AUC 0.9474
[0.9268, 0.9657]. 10,000 resamples, seed 42, stratified by class.

No Wilson interval is claimed for F1 or either AUC, and the reason is recorded
in code (`NO_INTERVAL_CLAIMED`) and asserted by a test.

### Newly discovered: the evaluation units are not independent

The 394 primary clips come from only **174 distinct source videos** (mean 2.26
clips each, largest 13, 274 clips sharing a source, 25 sources contributing to
both classes). Clips cut from one video are correlated, so the effective sample
size is below 394 and every clip-level interval is anti-conservative.

Quantified rather than assumed away, by re-running the bootstrap over source
videos instead of clips:

| Metric | Clip-level | Source-video-level | Widening |
|---|---|---|---:|
| accuracy | 6.60 pp | 8.97 pp | 1.36x |
| F1 | 7.29 pp | 10.42 pp | 1.43x |
| ROC-AUC | 4.24 pp | 4.99 pp | 1.18x |
| PR-AUC | 3.89 pp | 5.53 pp | 1.42x |

Quoting a clip-level interval alone overstates precision by 20-45%. The
grouping is **inferred** from the `<source_id>_<index>.avi` filename convention
and cannot be confirmed against dataset metadata this repository does not hold.

### Independent metric cross-check

`tools/crosscheck_metrics_sklearn.py`. Optional: scikit-learn is not in
`requirements.txt`, is not installed in the project venv, and the tool reports
SKIPPED with exit 0 when it is absent.

Run against scikit-learn 1.9.0, installed into an isolated directory so the
project environment was untouched:

| Metric | `temporal_metrics` | scikit-learn | Difference |
|---|---:|---:|---:|
| accuracy | 0.8604060913705583 | 0.8604060913705583 | 0 |
| precision | 0.8961748633879781 | 0.8961748633879781 | 0 |
| recall | 0.8200000000000000 | 0.8200000000000000 | 0 |
| F1 | 0.8563968668407310 | 0.8563968668407310 | 0 |
| ROC-AUC | 0.9430283505154640 | 0.9430283505154639 | +1.11e-16 |
| PR-AUC | 0.9473887310809866 | 0.9473887310809868 | -1.11e-16 |

Confusion matrix identical. Agreement to one unit in the last place.
**`temporal_metrics` is now independently validated** - the open item carried
since the Step 0 audit.

### The rounding limitation is unchanged

The cross-check answers "is the code correct?" (yes). It does not and cannot
answer "can the original recorded value be recovered?" - both implementations
read the same 6-decimal file, which ties 105 of 394 rows and destroys the
ordering a rank statistic needs. Step 3's finding that the recorded 0.9432 lies
inside the achievable interval [0.9428092784, 0.9432860825] stands unchanged. A
test asserts the ties are still present and that no code path perturbs the
stored scores.

### Added

- **`src/metric_intervals.py`** - Wilson score intervals, percentile bootstrap
  (stratified and cluster variants), source-video clustering, and the recorded
  reasons for metrics that receive no binomial interval.
- **`tools/metric_confidence_intervals.py`**, **`tools/crosscheck_metrics_sklearn.py`**
- **`tests/test_metric_intervals.py`** (49 tests, 41 subtests; 2 skip without
  scikit-learn).

### Changed

- **`docs/TEMPORAL_EVALUATION.md`** - new sections 8 and 9; section 5's
  "not yet cross-checked" line corrected; remaining-issues list updated with two
  new entries for the independence finding.

### Deliberately not changed

`src/temporal_metrics.py` is left **byte-identical**. It produced every
historical number in this project; interval estimation is a separate concern
and lives in a new module. `predictions.csv`, `temporal_risk/`, all historical
result artifacts and the golden reference are untouched.

### Verified

Wilson implementation checked against its **definition** - both bounds solve
`(p_hat - p)/sqrt(p(1-p)/n) = +/- z` to 9 decimal places - rather than against
remembered table values, two of which turned out during development to be
continuity-corrected figures for a different estimator.

---

## Step 3 — Checkpoint identity and integrity

Branch `step3-checkpoint-identity`.

Closes the hole where a randomly initialised checkpoint could masquerade as the
trained model, and validates the historical Experiment-1 metrics from their raw
predictions. No model was retrained, no research result altered.

### Historical metric verification (zero cost, done first)

Recomputed Experiment 1's metrics from `outputs/temporal_violence/evaluation/predictions.csv`
(394 clips, 200 Fight / 194 NonFight) with this project's own
`temporal_metrics.binary_metrics`:

| Metric | Recomputed | Recorded | Difference |
|---|---:|---:|---:|
| accuracy | 0.8604060914 | 0.8604 | +0.0000060914 |
| precision (Fight) | 0.8961748634 | 0.8962 | −0.0000251366 |
| recall (Fight) | 0.8200000000 | 0.8200 | 0 |
| F1 (Fight) | 0.8563968668 | 0.8564 | −0.0000031332 |
| ROC-AUC | 0.9430283505 | 0.9432 | −0.0001716495 |
| PR-AUC | 0.9473887311 | 0.9475 | −0.0001112689 |

Confusion matrix reproduces **exactly** (TN 175 / FP 19 / FN 36 / TP 164), as do
both class supports.

The two rank metrics differ beyond rounding, and the cause was identified rather
than dismissed: `predictions.csv` stores probabilities to 6 decimals, tying 105
of 394 rows (48 clips all at `1.000000`). ROC-AUC depends on ordering, so the
true value is unrecoverable from the file; it must lie in
**[0.9428092784, 0.9432860825]**, and the recorded 0.9432 falls inside. The
recorded values are consistent with the raw predictions; the difference is
precision loss in the CSV, not an error in either.

Not yet an independent check of the metric code: scikit-learn is not installed
here, so `temporal_metrics` still has no cross-implementation comparison.

### Added

- **`src/checkpoint_identity.py`** — provenance guard and identity recording.
  Rejects on random initialisation, missing monitored value, missing metrics,
  missing required metadata, subset (smoke-test) runs, wrong class head, or
  epoch < 1. Explicitly does not use filename, file size, or `torch.load`
  succeeding as evidence.
- **`tools/verify_historical_metrics.py`** — the recomputation above, with
  tie-bound analysis for the rank metrics.
- **`tools/record_checkpoint.py`** — verify-then-record; writes
  `models/temporal_violence/CHECKPOINT.md` and `checkpoint_records.json`.
- **`models/temporal_violence/CHECKPOINT.md`** — documents `best.pt` as ABSENT,
  with its source, expected size, what it unblocks, and the caveat that no hash
  was ever recorded.
- **`tests/test_checkpoint_identity.py`** (36 tests + 8 subtests).
- **`docs/CHECKPOINT_INTEGRITY.md`**.

### Changed

- **`src/temporal_training.py`** — `Checkpointer.save` now records
  `training_provenance`, `protocol` and `saved_at`. This is the actual fix:
  before it, the information was never written, so no guard could read it.
- **`src/evaluate_temporal_baseline.py`** — verifies before loading; refuses to
  report metrics on a rejected checkpoint.
- **`tools/score_temporal_windows.py`** — same guard replaces the weaker
  Step-2 `is_task_specific` check.
- **`tests/test_temporal_reproducibility.py`** — 7 tests for the historical
  metric verification, including a hash pin on `predictions.csv`.

### Verified empirically

A real smoke-test checkpoint was generated (`--validate-pipeline`, random init,
4 clips, 132,743,435 bytes — within 0.3% of the authentic file's size):

- old guard: `is_task_specific = True` → **accepted**
- new guard: **rejected**, citing `training_provenance='random_init'` and both
  subset limits

Note `monitored_value` was `1.0`, so a missing-value check alone would not have
caught it.

### Checkpoint retrieval

**Not possible from this environment.** No Kaggle CLI, no credentials, no
`best.pt` anywhere on the machine, nothing bundled in the archives. kaggle.com
is reachable but the notebook is private. No replacement was created; the manual
procedure is documented in `docs/CHECKPOINT_INTEGRITY.md` §7.

### Untouched

`predictions.csv`, all historical result JSON, `temporal_risk/`, the golden
reference, and all Kaggle-derived records. New outputs go to
`outputs/step3_checkpoint_identity/`.

---

## Step 2 — Reproducible temporal protocol

Branch `step2-reproducible-temporal-protocol`.

Restores the experimental protocol that produced the reported temporal result
but existed only in Kaggle notebooks, and adds the missing window-scoring step.
No model was retrained, no architecture changed, no historical result altered.

### Added

- **`src/carved_validation.py`** — the carved-validation (leakage-safe)
  protocol. Builds the three-way `train_remainder` (1360) / `carve_validation`
  (240) / `primary_evaluation` (394) split, with `verify_disjoint()` as an
  auditable leakage check. Carve membership is **recovered** from
  `temporal_risk/carve_window_scores.csv`, not re-derived from a seed — the
  partition algorithm is not in the repository and inventing one would produce
  a different 240 clips and invalidate every number the carve split supports.
- **`tools/score_temporal_windows.py`** — reconstructs the missing
  `score_temporal_windows` step by composing existing modules. `--verify-geometry`
  checks window bounds against a committed CSV with no model and no weights.
  Refuses to emit probabilities without a task-specific checkpoint.
- **`tools/time_to_alarm.py`** — measures detection latency from committed
  scores. Right-censored Fight clips are reported separately, never averaged in.
- **`tests/test_carved_validation.py`** (25 tests) — split sizes, disjointness,
  leakage, name resolution, label agreement, trainer wiring.
- **`tests/test_temporal_reproducibility.py`** (25 tests) — window geometry vs
  the committed CSVs, the alarm definition, and agreement with recorded results.
- **`docs/EXPERIMENT_REPRODUCIBILITY.md`**, **`docs/TEMPORAL_EVALUATION.md`**,
  and this changelog.

### Changed

- **`src/temporal_training.py`** — new `protocol` field and `--protocol` flag
  with two named options. `build_dataloaders` branches on it. The default is
  `carved_validation`; selecting `experiment1_primary_monitor` prints a warning
  that its metrics are optimistically biased. The misleading
  `sizes["primary_evaluation_clips"]` key was replaced with `monitor_clips`
  plus an explicit `monitor_set` label, because under the carved protocol the
  monitor set is not the primary set.
- **`src/rwf2000_dataset.py`** — added `build_carved_train_dataset` and
  `build_carved_validation_dataset`.

### Preserved deliberately

- `experiment1_primary_monitor` is **not** deleted. It is the protocol behind
  `docs/temporal_baseline_results.md` and
  `rwf2000_config.SELECTED_DECISION_THRESHOLD`; removing it would make those
  documents unreproducible.
- `temporal_risk/` and all `docs/*_results.md` are untouched.

### Verified

- All 240 recorded carve clips resolve to real local files (235 by exact name,
  5 through Kaggle's deterministic `safe_relative_path` renaming).
- The three sets are pairwise disjoint; no confirmed-leaked clip appears in any.
- Recorded carve labels agree with the on-disk class directory for all 240.
- **Window geometry:** all 394 primary and all 240 carve clips use exactly the
  17 window bounds `TemporalClipBuffer(16, 8)` produces for 150 frames. This
  correspondence was previously assumed.
- Time-to-alarm independently reproduces the recorded aggregation counts
  (primary TP 160 / FP 43 / FN 40; carve TP 85 / FP 21 / FN 34) and the live
  demo's alarm frames for all three built-in clips.

### New result

Time to alarm on the primary split, over the 160 of 200 Fight clips that alarm
(40 right-censored, excluded and reported separately): **mean 1.272 s, median
0.533 s**, p75 1.600 s. 82 clips (51% of detected) alarm on the first completed
window. False alarms are slower — median 1.867 s vs 0.533 s.

This is the first measurement of the *benefit* side of the streaming
trade-off; previously only its accuracy cost (0.789 vs 0.832) was quantified.

### Still blocked

Training, window scoring, and re-deriving accuracy 0.8325 / ROC-AUC 0.9251 all
require `best.pt` (132.74 MB), which is not on this machine.

---

## Step 1 — Stale-track fix

Branch `step1-stale-track-fix`, commit `5031e53`.

`SimpleTracker` retains a track for `max_age` (30) frames after its last
detection with its bounding box frozen. `tools/run_demo.py` passed those
retained tracks to both the Phase-4 rule engine and the renderer, so departed
objects were scored as stationary and drawn as ghost boxes.
`src/end_to_end_fixture.py:123` already filtered them; the demo did not.

### Changed

- `src/tracker.py` — added `active_tracks()`.
- `tools/run_demo.py` — applies it before the rule snapshot and drawing.
- `tests/test_stale_tracks.py` — 11 regression tests.

### Effect on demo output

| Clip | Stationary | Proximity | Restricted | Crowding | Rule events |
|---|---|---|---|---|---|
| fight | 1255 → 106 | 193 → 48 | 0 → 0 | 1 → 6 | 1449 → 160 |
| nonfight | 466 → 78 | 28 → 8 | 11 → 11 | 2 → 1 | 507 → 98 |
| fp | 716 → 167 | 96 → 37 | 3 → 3 | 1 → 4 | 816 → 211 |

Crowding rises because `CrowdInteractionAnalyzer` latches; the inflated person
count previously kept it saturated. Restricted Area Entry is unchanged because
it is edge-triggered on a region crossing.

All temporal values unchanged on all three clips. Ghost boxes drawn: 0, against
2104 / 691 / 1361 before.

---

## Step 0 — Golden reference

No code changed. Captured `outputs/golden_reference_a2422f1/`: SHA-256 of all
85 tracked files, environment, commands, demo inputs and outputs, test results
(252 passed / 1 failed / 9 skipped), and measurements of the stale-track defect
and the false-alarm test failure. `verify_baseline.py` re-checks the tree
against it at any time.
