# Checkpoint identity and integrity

How this project decides whether a file is genuinely a trained temporal model,
and how a checkpoint's identity is recorded so results can be tied to weights.

Added in Step 3. Companion to `docs/EXPERIMENT_REPRODUCIBILITY.md`.

---

## 1. The hole

`R3D18TemporalModel.load_checkpoint` sets `provenance` to
`checkpoint:<filename>` for **any** file it can load, and `is_task_specific`
returns True whenever provenance starts with `checkpoint`. That was the only
guard in front of `evaluate_temporal_baseline` and `score_temporal_windows`.

`temporal_training.py --validate-pipeline` writes a checkpoint from a
**randomly initialised** network trained for one epoch on four clips — to
`models/temporal_violence/best.pt`, the canonical path. Demonstrated on a real
smoke-test checkpoint during Step 3:

```
provenance       : checkpoint:best.pt
is_task_specific : True          <- the old guard accepts a random-init model
size             : 132,743,435 bytes
```

That file would have produced "metrics" and "violence probabilities" from
noise, at the exact path and near-exactly the size of the authentic 132.74 MB
checkpoint.

### Three things that are not evidence

| Not evidence | Why |
|---|---|
| The filename | `best.pt` is a path convention the smoke test also writes to |
| The file size | An untrained R3D-18 is the same size as a trained one — 132,743,435 vs 139,172,577 bytes |
| `torch.load` succeeding | Proves the file is a tensor archive, nothing more |

`src/checkpoint_identity.py` uses none of them.

---

## 2. What counts as evidence

Only metadata the training loop recorded about the run that produced the
weights. A checkpoint is **accepted** only when all of these hold:

| Check | Rejection reason |
|---|---|
| All of `state_dict`, `epoch`, `monitor`, `monitored_value`, `metrics`, `config`, `num_classes`, `class_labels` present | not written by this project's training loop |
| `monitored_value` is not None | the weights were never evaluated |
| `metrics` is not None | no evaluation recorded |
| `epoch >= 1` | no completed epoch |
| Initialisation is not random | outputs are noise |
| `config.limit_train_clips` / `limit_eval_clips` are None | smoke-test subset, not a reportable run |
| `num_classes == 2` and `class_labels == ("NonFight", "Fight")` | not this project's violence head |

**Warnings** (recorded, not fatal): a legacy checkpoint with no explicit
provenance field; a checkpoint whose protocol is `experiment1_primary_monitor`,
whose metrics are optimistically biased.

`monitored_value is None` alone would **not** have caught the smoke-test
checkpoint — it recorded `monitored_value = 1.0` (ROC-AUC on four clips). The
provenance and subset checks are what catch it. That is why the guard checks
all seven conditions rather than one.

---

## 3. Checkpoints are now self-describing

`Checkpointer.save` records three additional fields:

| Field | Value |
|---|---|
| `training_provenance` | the model wrapper's provenance at save time (`random_init`, `kinetics400_pretrained+untrained_head`, …) |
| `protocol` | `carved_validation` or `experiment1_primary_monitor` |
| `saved_at` | UTC timestamp |

This is the actual fix. Before it, the information simply was not written, so
no guard could have read it.

### Legacy checkpoints

The authentic `best.pt` predates these fields. Rejecting it for that would be
absurd, so when `training_provenance` is absent, initialisation is **inferred**
from `config.pretrained_backbone` — the smoke test sets it False, every real
run sets it True — and from the subset limits. The verdict sets
`inferred_initialisation = True` and emits a warning, so a reader can tell an
inferred judgement from a read one.

---

## 4. Where the guard runs

| Consumer | Behaviour |
|---|---|
| `src/evaluate_temporal_baseline.py` | verifies before loading; refuses to report metrics on a rejected checkpoint |
| `tools/score_temporal_windows.py` | verifies before loading; refuses to emit probabilities |
| `tools/record_checkpoint.py` | verifies before recording; `--record-anyway` documents a file *as rejected* |

Warnings are printed but do not block.

---

## 5. Identity recording

`tools/record_checkpoint.py` writes `models/temporal_violence/CHECKPOINT.md`
and `checkpoint_records.json`, capturing filename, SHA-256, size, modification
time, retrieval source, training provenance, protocol, monitored metric and
value, best epoch, and the training configuration.

Hashing is streamed (a 132 MB file never loads into memory) and deterministic:
identical bytes always produce the same digest regardless of filename or chunk
size, and the record is byte-identical across runs apart from `recorded_at`.

```bash
python tools/record_checkpoint.py --inspect models/temporal_violence/best.pt
python tools/record_checkpoint.py --checkpoint models/temporal_violence/best.pt \
    --retrieved-from "kaggle notebook7bb9a86555 v3, Output tab"
python tools/record_checkpoint.py --refresh
```

### Weights are not committed

`.gitignore` excludes `*.pt`, `*.pth`, `*.ckpt`, and no policy in this
repository permits committing weights. `CHECKPOINT.md` records identity so a
checkpoint obtained elsewhere can be **verified**, not so it can be
distributed. A test asserts no weight file is ever tracked by git.

---

## 6. Retrieval status of `best.pt`

**Not retrievable from this environment.** Established, not assumed:

| Check | Result |
|---|---|
| `best.pt` anywhere on this machine | none (only a smoke-test file created during Step 3 testing, since deleted) |
| Kaggle CLI installed | no (`ModuleNotFoundError: kaggle`) |
| `~/.kaggle/kaggle.json` | absent |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | unset |
| kaggle.com reachable | yes, but the notebook is private and its output requires authentication |
| Checkpoint bundled in `data/rwf2000_archives/` | no — 13 dataset archive parts only |

No replacement was created. `CHECKPOINT.md` documents exactly what is missing,
where it came from, and what it unblocks.

### The identity caveat that cannot be fixed retroactively

**No hash of `best.pt` was recorded before it became unavailable.** A future
download can be checked for internal consistency — provenance, protocol, epoch,
monitored value — and can be checked for *producing* the recorded metrics, but
it cannot be **proven** to be the same file that produced them. The recorded
size (139,172,577 bytes) is weak corroboration and is explicitly not treated as
evidence by the guard.

From Step 3 onward every checkpoint is hashed on arrival, so this gap does not
recur.

---

## 7. Manual retrieval procedure

1. Open Kaggle notebook `notebook7bb9a86555`, version 3 (session 344760335).
2. From the Output tab download `clean_experiment/training/checkpoints/best.pt`
   (expected ≈139,172,577 bytes).
3. Place it at `models/temporal_violence/best.pt` — do not commit it.
4. Record and verify:
   ```bash
   python tools/record_checkpoint.py --checkpoint models/temporal_violence/best.pt \
       --retrieved-from "kaggle notebook7bb9a86555 v3 Output tab, downloaded <date>"
   ```
5. Confirm the verdict is ACCEPTED and note any warnings (a legacy checkpoint
   will warn that initialisation was inferred).
6. Also download `clean_experiment/final_evaluation/predictions.csv` and
   `training_history.json` if available. **Do not overwrite** the existing
   `outputs/temporal_violence/evaluation/predictions.csv` — that is Experiment
   1's output, a different run. Save the clean-baseline files under a distinct
   name and compare hashes before treating either as canonical.
7. Once verified, `docs/EXPERIMENT_REPRODUCIBILITY.md` §7 becomes runnable end
   to end.
