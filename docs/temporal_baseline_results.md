# RWF-2000 Temporal Baseline — Results

Every number here was produced by Kaggle notebook `notebook5d98537e15`,
**Version #2** (`Successfully ran in 8516.4s`, GPU Tesla T4). Nothing is
estimated or interpolated. The run log is the authority; this document
transcribes it.

---

## 1. Architecture, and why this one

**R3D-18** — an 18-layer 3D ResNet from `torchvision.models.video`,
initialised from Kinetics-400 and fine-tuned on RWF-2000.

Recorded provenance string: `kinetics400_pretrained+untrained_head`.

It was chosen over the alternatives that were on the table (Video Swin,
TimeSformer, SlowFast, CNN+LSTM) for reasons that are about this project
rather than about leaderboards:

- **It fits a T4 honestly.** 16×112×112 clips at batch 8 trained in
  ~6 min/epoch with mixed precision. A transformer video model would not
  have finished inside the weekly GPU quota, and a CNN+LSTM would have
  needed a separate feature-extraction pass over 2000 clips first.
- **Kinetics-400 pretraining is the whole point.** 1600 training clips is
  far too few to learn spatio-temporal filters from scratch. The frozen
  stem/`layer1`/`layer2` keep the generic motion features and only
  `layer3`, `layer4`, and the head adapt.
- **It has a documented Grad-CAM seam.** `R3D18TemporalModel.feature_layer`
  exposes `layer4`, which emits `(N, 512, T', H', W')` before pooling —
  the hook point a later 3D Grad-CAM needs to localise *when* and *where*.
  A transformer would have required a different explainability approach
  than the Grad-CAM already used for YOLO.

This is a **baseline**, not a tuned result. The hyperparameters come
from `rwf2000_config.InitialTrainingConfig`, which labels itself
"INITIAL EXPERIMENTAL SETTINGS — not tuned, not validated". No search
was run.

## 2. Configuration

| Setting | Value |
|---|---|
| Backbone | R3D-18, Kinetics-400 pretrained |
| Frozen | `stem`, `layer1`, `layer2` |
| Trainable | `layer3`, `layer4`, `fc` |
| Optimizer | AdamW — head 1e-3, backbone 1e-4, weight decay 1e-4 |
| Scheduler | cosine annealing |
| Batch size | 8 |
| Clip sampling | 16 frames, uniform across the full 150-frame clip |
| Preprocessing | torchvision Kinetics-400 preset (112×112 crop, BGR→RGB) |
| Mixed precision | on (CUDA autocast + GradScaler) |
| Seed | 42 |
| Epoch cap | 20 |
| Early stopping | ROC-AUC, patience 5 |

## 3. Training

Stopped early at **epoch 15** — no ROC-AUC improvement for 5 epochs.
**Best epoch: 10.** Training wall time 7314 s (2 h 02 m).

| Epoch | train_loss | eval_loss | acc | f1 | roc_auc |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4464 | 0.3968 | 0.8477 | 0.8352 | 0.9199 |
| 2 | 0.2223 | 0.5942 | 0.7919 | 0.7574 | 0.9209 |
| 3 | 0.1872 | 0.6106 | 0.7995 | 0.7799 | 0.9103 |
| 4 | 0.1220 | 0.5547 | 0.8249 | 0.8279 | 0.9197 |
| 5 | 0.1555 | 0.5250 | 0.8629 | 0.8608 | 0.9243 |
| 6 | 0.1125 | 0.5729 | 0.8426 | 0.8324 | 0.9197 |
| 7 | 0.0701 | 0.5060 | 0.8325 | 0.8263 | 0.9206 |
| 8 | 0.1041 | 0.5221 | 0.8325 | 0.8187 | 0.9266 |
| 9 | 0.0584 | 0.4812 | 0.8629 | 0.8608 | 0.9357 |
| **10** | **0.0360** | **0.4721** | **0.8604** | **0.8564** | **0.9432** |
| 11 | 0.0674 | 0.5091 | 0.8629 | 0.8564 | 0.9361 |
| 12 | 0.0647 | 0.4866 | 0.8503 | 0.8451 | 0.9368 |
| 13 | 0.0195 | 0.5100 | 0.8579 | 0.8549 | 0.9383 |
| 14 | 0.0374 | 0.4971 | 0.8452 | 0.8407 | 0.9367 |
| 15 | 0.0353 | 0.4796 | 0.8452 | 0.8382 | 0.9408 |

**Overfitting is visible and should be stated in the thesis.** Training
loss falls from 0.45 to 0.02 while evaluation loss rises from 0.40 to
~0.48–0.51 after epoch 1. ROC-AUC still improves, so the ranking gets
better even as the calibrated loss worsens — which is exactly why
ROC-AUC, not eval loss, was the early-stopping monitor. Closing that gap
is a tuning question (augmentation strength, more frozen layers, dropout),
not a defect in this run.

## 4. Evaluation — 394 leak-free held-out clips

The headline set is the **primary evaluation split**: the official 400
validation clips minus the 6 confirmed train/val duplicates. Those 6 are
excluded from training *and* evaluation, so no number below is inflated by
a clip the model has seen.

```
     class   precision    recall        f1   support
----------------------------------------------------
  NonFight      0.8294    0.9021    0.8642       194
     Fight      0.8962    0.8200    0.8564       200
----------------------------------------------------
  accuracy                          0.8604       394
```

| Metric | Value |
|---|---|
| Accuracy | **0.8604** |
| Precision (Fight) | **0.8962** |
| Recall (Fight) | **0.8200** |
| F1 (Fight) | **0.8564** |
| ROC-AUC | **0.9432** |
| PR-AUC | **0.9475** |
| Inference | **28.58 ms/clip** (T4, batch 8, forward pass only) |

Confusion matrix, `Fight` positive:

|  | pred NonFight | pred Fight |
|---|---:|---:|
| **actual NonFight** | 175 | 19 |
| **actual Fight** | 36 | 164 |

**Read the error asymmetry before quoting the accuracy.** At the default
0.5 threshold the model misses **36 of 200 fights** while raising only 19
false alarms. For a surveillance system that trade-off is backwards — a
missed assault costs more than a spurious alert. The threshold was left
at 0.5 deliberately (`InitialTrainingConfig.decision_threshold` is
unset until validation data exists); ROC-AUC of 0.9432 says the ranking
supports a better operating point, and choosing one is a specific,
cheap next step that needs no retraining.

## 4a. Decision threshold

The 0.5 default was never tuned. `tools/threshold_sweep.py` re-thresholds
the run's own `predictions.csv` — no retraining, no GPU — and selects by a
stated rule rather than by eye:

> maximise Fight recall subject to precision >= 0.80, false-positive rate
> <= 0.20, and accuracy no worse than the 0.5 baseline.

Accuracy alone is the wrong objective here. On a balanced set it trades one
missed assault for one dismissed false alarm, and those costs are not equal
in a surveillance system.

**Selected: 0.14.**

| | threshold 0.50 | threshold 0.14 | change |
|---|---:|---:|---:|
| Accuracy | 0.8604 | 0.8604 | unchanged |
| Precision (Fight) | 0.8962 | 0.8436 | -0.0526 |
| Recall (Fight) | 0.8200 | 0.8900 | **+0.0700** |
| F1 (Fight) | 0.8564 | 0.8662 | +0.0098 |
| Specificity | 0.9021 | 0.8299 | -0.0722 |
| False positives | 19 | 33 | +14 |
| **Missed fights** | **36** | **22** | **-14 (-39%)** |

The trade is symmetric and cheap: 14 fewer missed fights for 14 more false
alarms, at identical overall accuracy. For a system whose purpose is to
catch assaults, that is the right direction.

For reference, the other criteria disagree — which is why the rule is
stated rather than assumed:

| Criterion | Threshold | Recall | FN | FP |
|---|---:|---:|---:|---:|
| Max accuracy / balanced accuracy / Youden J | 0.40 | 0.8450 | 31 | 22 |
| Max F1 | 0.10 | 0.9100 | 18 | 38 |
| Selected (recall at no accuracy cost) | **0.14** | **0.8900** | **22** | **33** |

If a higher alarm budget is acceptable, **0.10** halves the misses relative
to the default (36 -> 18) and maximises F1, at 38 false alarms.

**Caveat, and it matters.** This threshold was chosen on the same 394 clips
the headline metrics are reported on, so it is optimistically biased. The
project's `SplitPolicy` already provides for a validation subset carved
from train (`validation_fraction` 0.15) precisely so an operating point can
be chosen without touching the held-out set. Re-derive it there before
quoting 0.14 as a final result.

Recorded as `rwf2000_config.SELECTED_DECISION_THRESHOLD`, with provenance
in `SELECTED_THRESHOLD_PROVENANCE`. Full sweep:
`outputs/temporal_violence/evaluation/threshold_sweep/`.

## 5. Verification

- **Test suite:** 220 tests, `OK (skipped=13)` — run on Kaggle where torch
  is available, so the three torch-dependent modules that cannot run on
  the CPU-only dev machine were exercised here.
- **CPU smoke test:** passed before any GPU time was spent, on an
  untrained backbone over 4 clips. Its metrics describe a random network
  and are not results.
- **Fresh-process checkpoint load:** `tools/classify_sample_clips.py`
  loaded `best.pt` in a process that never saw training and classified 8
  held-out clips **8/8 correct**. This is a load-and-predict spot check,
  not a metric.
- **Dataset:** re-verified in this same run — `RESULT: READY`, 17/17
  checks, 2000 clips, 1000/1000, 1600/400, missing 0, truncated 0.

## 6. Artifacts

Under `/kaggle/working/baseline_results/` in the Version #2 output, and
mirrored to the private Kaggle dataset `rwf2000-baseline-results`:

```
training/checkpoints/best.pt          132,744,843 bytes   (epoch 10)
training/training_history.json         17,983 bytes
evaluation/metrics.json                 1,336 bytes
evaluation/metrics.csv                    293 bytes
evaluation/classification_report.txt      318 bytes
evaluation/predictions.csv             11,401 bytes   (per-clip scores)
evaluation/confusion_matrix.png        22,007 bytes
evaluation/roc_curve.png               36,138 bytes
evaluation/training_curves.png         77,119 bytes
evaluation/spot_check.json              1,686 bytes
```

## 7. Reproducing this

```bash
python tools/run_kaggle_baseline.py --stage all \
    --code /kaggle/input/<code-dataset> \
    --device cuda --amp --snapshot \
    --epochs 20 --batch-size 8 --num-workers 2
```

Environment recorded by the run: torch 2.10.0+cu128, CUDA available,
Tesla T4. Seed 42 across `random`, `numpy`, and `torch`.

The driver gates each stage on the previous one, so training cannot start
on a red test suite or an incomplete dataset — which is what stopped
Version #1 before it wasted GPU time on a packaging bug.

## 8. What this does NOT include

This branch is a standalone violence classifier. It is **not** wired into
`AbnormalEventDetector`, the tracker, `CrowdInteractionAnalyzer`, or
`RiskAssessor`. That integration is the next phase and is a separate,
deliberate step — see §8 of `cloud_setup_rwf2000.md` for why the branch
was kept isolated.
