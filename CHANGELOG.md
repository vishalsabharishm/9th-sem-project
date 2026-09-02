# Changelog

Engineering changes made after the Step 0 audit and golden-reference capture.
Research results are never edited; where a number changes, both the old and new
values are recorded.

Baseline for everything below: commit `a2422f1`, captured in full at
`outputs/golden_reference_a2422f1/` (source digests, environment, demo
outputs, test results, and measurements of known defects).

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
