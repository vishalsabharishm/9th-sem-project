# RWF-2000 Clean Baseline — Results (threshold 0.16)

**Status: CRITICAL VERIFIED CLEAN BASELINE — IMMUTABLE.** These numbers, this
checkpoint, and this threshold must not be modified or re-derived. This
document was reconstructed on 2026-08-25 by recovering the source notebook's
Output tab after the local repository was found to have no trace of this run
(not in the working tree, not in git history). Every number below was
transcribed directly from Kaggle notebook `notebook7bb9a86555`, **Version 3**
(session 344760335), output folder `clean_experiment/`. Nothing here is
estimated.

## Why this run exists — and why it supersedes Experiment 1

`docs/temporal_baseline_results.md` documents an earlier run
(`notebook5d98537e15`, threshold 0.14, accuracy 0.8604, ROC-AUC 0.9432) whose
own caveat says the threshold "was chosen on the same 394 clips the headline
metrics are reported on, so it is optimistically biased." This run is the
fix: a validation subset is **carved from train** (`validation_fraction =
0.15`, same 0.15 fraction the Phase B window-scoring manifest calls
`carve_fraction`), and every selection decision — early stopping *and*
decision threshold — is made on that carved subset. The 394-clip primary set
is touched exactly once, for final reporting, after every choice was already
frozen.

## 1. Configuration

Same architecture and hyperparameters as Experiment 1 (R3D-18,
Kinetics-400 pretrained, frozen stem/layer1/layer2, trainable
layer3/layer4/fc, AdamW head 1e-3 / backbone 1e-4, batch 8, cosine
annealing, mixed precision, seed 42). The one methodological change is
`holdout_validation: true` with a validation split carved from train, used
as the early-stopping and threshold-selection monitor instead of the
reporting set.

## 2. Training

- Completed epochs: **17** (early stopping on ROC-AUC, patience 5)
- **Best epoch: 12**
- Best monitor value (ROC-AUC on train-carved validation): **0.935342732134176**
- Total training time: 7081.72 s (~1 h 58 m)
- Monitor set: train-carved validation (never seen the 394-clip evaluation set)
- Checkpoint: `clean_experiment/training/checkpoints/best.pt` (132.74 MB)

## 3. Threshold selection — on the carved validation subset only

Selection rule: maximise recall subject to precision ≥ 0.80, FPR ≤ 0.20,
accuracy ≥ 0.8750.

| Threshold | Acc | Prec | Recall | F1 | Spec | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.14 | 0.8667 | 0.8175 | 0.9412 | 0.8750 | 0.7934 | 25 | 7 |
| 0.15 | 0.8667 | 0.8175 | 0.9412 | 0.8750 | 0.7934 | 25 | 7 |
| **0.16** | **0.8750** | **0.8296** | **0.9412** | **0.8819** | **0.8099** | **23** | **7** |
| 0.17 | 0.8708 | 0.8333 | 0.9244 | 0.8765 | 0.8182 | 22 | 9 |

**Selected: 0.16** — the operating point recorded in `frozen.json`. This
selection never touched the 394-clip primary set. Full 95-row sweep in
`clean_baseline_threshold_sweep.txt` (transcribed from
`clean_experiment/threshold_selection/threshold_sweep.txt`).

## 4. Final evaluation — 394-clip primary split, threshold 0.16

```
     class   precision    recall        f1   support
----------------------------------------------------
  NonFight      0.8902    0.7526    0.8156       194
     Fight      0.7913    0.9100    0.8465       200
----------------------------------------------------
  accuracy                          0.8325       394
```

| Metric | Value |
|---|---|
| Accuracy | **0.8325** (0.8324873096446701) |
| Precision (Fight) | **0.7913** (0.7913043478260869) |
| Recall (Fight) | **0.9100** |
| F1 (Fight) | **0.8465** (0.8465116279069768) |
| Specificity (NonFight recall) | **0.7526** |
| ROC-AUC | **0.9251** (0.9250644329896908) |
| PR-AUC | **0.9260** (0.926017126389842) |
| Inference | **31.01 ms/clip** (31.008600654800127, T4, batch 8, forward pass only) |

Confusion matrix, `Fight` positive:

|  | pred NonFight | pred Fight |
|---|---:|---:|
| **actual NonFight** | 146 | **FP = 48** |
| **actual Fight** | **FN = 18** | 182 |

This is the trade the "clean" threshold makes relative to Experiment 1's
biased 0.14: fewer false negatives on the true operating point (18 vs.
Experiment 1's 22 at its own threshold), at a cost the FP/FN split above
makes explicit — reported honestly rather than re-tuned after the fact.

## 5. Artifacts and provenance

```
clean_experiment/
  training/checkpoints/best.pt        132.74 MB   (epoch 12)
  training/training_history.json      20.11 kB
  threshold_selection/threshold_sweep.{csv,json,txt}
  final_evaluation/metrics.{json,csv}
  final_evaluation/classification_report.txt
  final_evaluation/predictions.csv
  final_evaluation/{confusion_matrix,roc_curve,training_curves}.png
  final_evaluation/spot_check.json
  frozen.json                          -- threshold + validation operating point
  independence_pre_training.json
  independence_post_training.json
```

Checkpoint (`best.pt`) was **not downloaded during this recovery pass** —
it is 132.74 MB, over the file-transfer tooling's per-file limit at the time.
*(Superseded: it was transferred on 2026-09-04 and verified — SHA-256
`a32271bf...`; see `docs/CHECKPOINT_INTEGRITY.md` and
`models/temporal_violence/checkpoint_records.json`.)* The headline numbers,
confusion matrix, and threshold-sweep table above are fully recovered and sufficient for documentation and for the
temporal-risk aggregation work (which consumes `carve_window_scores.csv` /
`primary_window_scores.csv`, not the checkpoint directly). Downloading
`best.pt` is only required if the demo needs to run live inference through
this exact checkpoint rather than using the precomputed window scores
already recovered in `temporal_risk/`.

## 6. Relationship to Phase B window-scoring

`temporal_risk/window_scoring_manifest.json` records this same checkpoint
(`clean_experiment/training/checkpoints/best.pt`) and this same threshold
(`frozen_threshold_for_downstream_use_only: 0.16`) as the source for the
per-window `fight_probability` scores in `carve_window_scores.csv` (240
clips) and `primary_window_scores.csv` (394 clips). Those two CSVs are what
`tools/select_temporal_aggregation.py` and `tools/apply_frozen_aggregation.py`
consume — the clip-level numbers in this document are the target the
temporal-aggregation pipeline must reproduce (or knowingly diverge from, with
the divergence explained) once window scores are aggregated back up to
clip level.
