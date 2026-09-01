# Experiment reproducibility

What this repository can and cannot re-derive about the temporal violence
branch, and exactly how to run the correct protocol.

Status as of Step 2 (branch `step2-reproducible-temporal-protocol`).

---

## 1. The gap this document closes

Before Step 2, the repository could not reproduce the experiment that produced
its own headline temporal result:

| Needed to reproduce | Was it in the repo? |
|---|---|
| Carved-validation training protocol | **No** — only in Kaggle `notebook7bb9a86555` v3 |
| `score_temporal_windows` (sliding-window scoring) | **No** — only in Kaggle `notebookbce85b4c2b` |
| The trained checkpoint `best.pt` (132.74 MB) | **No** — never downloaded |
| Aggregation selection + application | Yes (`tools/select_temporal_aggregation.py`, `tools/apply_frozen_aggregation.py`) |
| Per-window scores | Yes, as committed CSVs (outputs, not code) |

`src/temporal_training.py` implemented a *different* protocol from the one
that produced the reported numbers — it monitored on the reporting set. Anyone
following the repository's own instructions would have reproduced the
superseded, biased Experiment 1 and got different numbers.

Step 2 restores the protocol and the scoring step. It does **not** retrain
anything and does **not** change any historical result.

---

## 2. The two protocols

Both are implemented; both are named; the correct one is the default.

### `carved_validation` — the correct, leakage-safe protocol

```
official train (1600)
    ├── train_remainder (1360)   <- gradient updates
    └── carve_validation (240)   <- early stopping AND threshold selection
official val (400)
    ├── leakage_excluded (6)     <- byte-identical duplicates of train clips
    └── primary_evaluation (394) <- touched ONCE, for final reporting
```

No clip used to make a choice is ever a clip a reported number is measured on.
`carved_validation.verify_disjoint()` asserts this and is exercised by
`tests/test_carved_validation.py`; the trainer refuses to start if it fails.

### `experiment1_primary_monitor` — the original/baseline protocol

```
official train (1600)            <- gradient updates
primary_evaluation (394)         <- early stopping AND reporting
```

Retained deliberately, not deleted: it is the protocol behind
`docs/temporal_baseline_results.md` and `rwf2000_config.SELECTED_DECISION_THRESHOLD`
(0.14), and removing it would make those documents unreproducible. Selecting it
prints a warning that its metrics are optimistically biased. It must not be
used for new reported numbers.

---

## 3. Recovered, not re-derived — and why

The 240-clip carve membership is **read from**
`temporal_risk/carve_window_scores.csv`, not recomputed from a seed.

The manifests record `carve_fraction = 0.15` and `carve_seed = 42`, and
`rwf2000_config.SplitPolicy` declares `validation_grouped_by_source_video =
True`. But the notebook that consumed those parameters is not in this
repository, so the exact partition algorithm is unknown. Re-deriving a split
from an unverifiable seed would produce a *different* 240 clips and silently
invalidate every number the carve split supports — the frozen aggregation rule,
its selection metrics, and the 0.16 decision threshold.

Reading the recorded membership reproduces the actual experiment exactly, and
is reproducible in the sense that matters: the same split, every run, on any
machine, verifiable against a committed artifact.

### Clip-name resolution

Six of the 240 recorded names do not match this machine's filenames. Both
causes are deterministic and both are handled without guessing:

| Mechanism | Count | Why |
|---|---:|---|
| `exact_filename` | 235 | names agree |
| `kaggle_safe_name_sha1` | 5 | Kaggle's extraction renamed over-long CJK names via `rwf2000_kaggle.safe_relative_path` (`<ascii-fragment>__<sha1[:12]>.avi`); this machine's extraction did not need to (`renamed_clips.json`: `recovered_under_safe_names: 0`) |
| `utf8_cp1252_roundtrip` | 0 here | one recorded name differs from a naive `Path.name` enumeration only by character encoding; retained because directory enumeration differs across platforms |

Resolution never falls back to fuzzy matching. An unresolved clip raises
`CarvedValidationError` rather than silently shrinking the split.

---

## 4. Documented uncertainty

Recorded honestly rather than guessed. None of these is resolved by Step 2.

1. **The partition algorithm is unknown.** `carve_fraction=0.15` and
   `carve_seed=42` are recorded, but not the code that used them. 240 is
   exactly 15% of 1600, which is consistent with several algorithms.

2. **Grouping by source video is unconfirmed.** Under a naive source-id
   heuristic (strip a trailing `_N` from the stem), 887 of 890 train groups
   fall entirely inside carve or entirely outside it — but 3 groups straddle.
   All 3 are non-ASCII names of the form `<title>_urlgot_<n>.avi`, where the
   heuristic is unreliable. This neither confirms nor refutes grouping, and
   the real algorithm cannot be recovered from the artifacts present.

3. **Kaggle-side preprocessing is inferred, not read.** The manifests record
   `clip_length=16`, `stride=8`, and the Kinetics-400 preprocessing constants,
   which match this project's modules exactly. The window geometry has now been
   *verified* (§6); the pixel-level preprocessing has not, because that needs
   the checkpoint to compare outputs.

4. **Checkpoint identity is unverified.** No hash of `best.pt` was recorded
   before it became unavailable, so a future download cannot be proven to be
   the same file that produced the reported metrics.

---

## 5. What can be run today, and what is blocked

| Step | Runnable here? | Command |
|---|---|---|
| Build the three-way split | **Yes** | `python -c "import sys;sys.path.insert(0,'src');from carved_validation import *;print(load_carved_split('data/rwf2000/RWF-2000').counts)"` |
| Verify leakage/disjointness | **Yes** | `python -m pytest tests/test_carved_validation.py` |
| Verify window geometry | **Yes** | `python tools/score_temporal_windows.py --verify-geometry` |
| Time-to-alarm | **Yes** | `python tools/time_to_alarm.py --split primary` |
| Aggregation selection + application | **Yes** | `python tools/select_temporal_aggregation.py` then `tools/apply_frozen_aggregation.py` |
| Produce window scores | **Blocked** | needs `--checkpoint`; `best.pt` absent |
| Train | **Blocked** | needs a GPU runtime |
| Re-derive accuracy 0.8325 / ROC-AUC 0.9251 | **Blocked** | needs `best.pt` |

`tools/score_temporal_windows.py` refuses to emit probabilities without a
task-specific checkpoint. A randomly initialised or Kinetics-400 model produces
numbers; those numbers are not violence probabilities, and substituting them
would be fabrication.

---

## 6. What Step 2 verified

Two claims that were previously assumed are now checked by tests:

**Window geometry.** All 394 primary and all 240 carve clips in the committed
CSVs use exactly the 17 window bounds that this project's
`TemporalClipBuffer(clip_length=16, stride=8)` produces for a 150-frame clip —
`(0,15), (8,23), … (128,143)`, one identical layout across every clip. This
closes the audit's open question about whether the recorded `first_frame` /
`last_frame` columns correspond to this project's buffering.

**Split integrity.** All 240 recorded carve clips resolve to real local files;
the three sets are pairwise disjoint; no confirmed-leaked clip appears in any
of them; recorded labels agree with the on-disk class directory for all 240.

---

## 7. Reproducing the full experiment

Steps 1–3 need a GPU. Steps 4–6 run anywhere.

```bash
# 1. Prepare the dataset (see docs/cloud_setup_rwf2000.md)

# 2. Train under the correct protocol
python src/temporal_training.py --train \
    --protocol carved_validation \
    --root data/rwf2000/RWF-2000 \
    --device cuda --amp \
    --checkpoint-dir models/temporal_violence

# 3. Score sliding windows with the resulting checkpoint
python tools/score_temporal_windows.py --split carve \
    --checkpoint models/temporal_violence/best.pt --device cuda \
    --output temporal_risk/carve_window_scores.regenerated.csv
python tools/score_temporal_windows.py --split primary \
    --checkpoint models/temporal_violence/best.pt --device cuda \
    --output temporal_risk/primary_window_scores.regenerated.csv

# 4. Select the aggregation rule on carve ONLY
python tools/select_temporal_aggregation.py

# 5. Apply it once to primary
python tools/apply_frozen_aggregation.py

# 6. Latency
python tools/time_to_alarm.py --split primary \
    --json outputs/temporal_evaluation/time_to_alarm_primary.json
```

Write regenerated scores to `*.regenerated.csv`. Do **not** overwrite the
committed CSVs: they are the recovered evidence behind every current number,
and the carve one is also the source of the split membership.

---

## 8. Reproducing the historical baseline for comparison

```bash
python src/temporal_training.py --train \
    --protocol experiment1_primary_monitor --device cuda --amp
```

This will print a warning. Its output is comparable to
`docs/temporal_baseline_results.md`, not to the clean baseline.

---

## 9. Determinism

| Element | Status |
|---|---|
| Split membership | Fully deterministic (read from a committed CSV) |
| Evaluation frame sampling | Deterministic — `SamplingConfig.eval_deterministic=True`, midpoint of each of 16 segments |
| Training augmentation | Seeded — `(seed, epoch, clip_index)`, `seed=42` |
| Torch / numpy / random seeding | `temporal_training.set_seed` |
| Window geometry | Deterministic, and verified against the committed CSVs |
| GPU kernel non-determinism | **Not controlled.** cuDNN autotuning and non-deterministic reductions mean a re-run on a GPU may differ in the last decimals. Not addressed by Step 2. |

---

## 10. Related documents

- `docs/TEMPORAL_EVALUATION.md` — evaluation and time-to-alarm details
- `docs/clean_baseline_results.md` — the recovered clean baseline (immutable)
- `docs/temporal_baseline_results.md` — Experiment 1 (superseded, biased)
- `docs/temporal_risk_integration.md` — checkpoint → risk assessment
- `docs/merge_manifest.md` — provenance of the recovered artifacts
- `CHANGELOG.md` — what each step changed
