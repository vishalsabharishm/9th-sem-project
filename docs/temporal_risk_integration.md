# Temporal risk integration

How the R3D-18 temporal violence branch goes from a trained checkpoint to a
`Temporal Violence Signal` event in the risk-assessment pipeline. Read
`docs/clean_baseline_results.md` first for the model itself; this document
covers everything downstream of that checkpoint.

## Pipeline overview

```
best.pt (clean-baseline checkpoint, epoch 12, threshold 0.16)
    |
    | sliding-window inference: clip_length=16, stride=8, 17 windows/clip
    | (Kaggle notebookbce85b4c2b -- "Phase B")
    v
temporal_risk/{carve,primary}_window_scores.csv   (per-window fight_probability)
    |
    | tools/select_temporal_aggregation.py  -- CARVE SPLIT ONLY (240 clips)
    v
temporal_risk/frozen_aggregation.json    (rule="max", threshold=0.14)
    |
    | tools/apply_frozen_aggregation.py  -- applies frozen rule to primary, no tuning
    v
temporal_risk/primary_aggregation_result.json   (honest generalization check)
    |
    | src/temporal_event_adapter.py: TemporalEventAdapter.evaluate_clip()
    v
EventDetection(event_type="Temporal Violence Signal", confidence=peak_prob, ...)
    |
    | src/risk_assessment.py: RiskAssessor.assess()
    v
RiskAssessment(risk_level="High", confidence_provenance="measured_model_probability")
```

`tools/run_demo.py` drives the whole right-hand side of this diagram (from
the CSVs onward) plus YOLO detection and tracking, on a real video, and
writes the annotated video / JSON / markdown you see in `outputs/demo/`.

## Why two splits, and why the order matters

`carve` (240 clips) and `primary` (394 clips) are disjoint. The discipline
that keeps the final numbers honest is: every decision that could overfit to
a specific set of clips is made on `carve`, and `primary` is touched exactly
once, to measure the already-frozen decision. This applies at two levels:

- **Threshold selection** (which produced the clean-baseline checkpoint
  itself): 0.16 was selected on a validation subset carved from *train*
  (see `docs/clean_baseline_results.md` section 3), never on `primary`.
- **Aggregation selection** (this document): the aggregation rule and its
  parameter were selected on `carve` (a held-out slice of clips, disjoint
  from both train and `primary`), then applied to `primary` unchanged.

Breaking this discipline is exactly what made Experiment 1's threshold
(0.14) "optimistically biased" per its own documentation
(`docs/temporal_baseline_results.md`). The project's explicit instruction
for this integration work was: never let a hyperparameter be chosen by
looking at the set it will be reported on. `tools/select_temporal_aggregation.py`
and `tools/apply_frozen_aggregation.py` are two separate scripts specifically
so that running the second one can never accidentally feed `primary` back
into a selection loop.

## Aggregation candidates considered

Four rule families, swept over `carve` (240 clips, 119 Fight / 121 NonFight):

| rule | parameterization | meaning |
|---|---|---|
| `any` | none (equivalent to `k_of_n` with k=1) | clip is Fight if any window's score >= the model's own frozen threshold (0.16) |
| `k_of_n` | k in 1..17 | clip is Fight if at least k windows score >= 0.16 |
| `max` | threshold in 0.01..0.99 | clip is Fight if the highest window score >= threshold |
| `mean` | threshold in 0.01..0.99 | clip is Fight if the average window score >= threshold |

`any`/`k_of_n` vote on binary per-window decisions using the model's own
calibrated 0.16 operating point; `max`/`mean` aggregate the continuous scores
first and sweep their own clip-level threshold, since averaging or maxing
changes the scale the 0.16 threshold was calibrated for.

Selection criterion: maximise recall subject to precision >= 0.80 and
FPR <= 0.20 on `carve`. Ties: fewer false positives, then higher F1, then
simpler rule family (`any` < `k_of_n` < `max` < `mean`).

**Selected: `max`, threshold 0.14.** (215 candidates considered, 175
satisfied the constraints; full ranked list in
`temporal_risk/frozen_aggregation.json`.)

Carve-split result at the selected rule: precision 0.8019, recall 0.7143,
F1 0.7556, FPR 0.174 (TP=85, FP=21, TN=100, FN=34).

Note the aggregation threshold (0.14) and Experiment 1's now-superseded
clip-level threshold (also 0.14) are numerically the same but **not the same
quantity** -- one thresholds a max-of-17-sliding-window score, the other
thresholded a single whole-clip forward pass. Treating them as
interchangeable would be a real mistake; they are kept clearly distinct in
this document and in code (`temporal_risk/frozen_aggregation.json`'s
`params.threshold` vs. `docs/temporal_baseline_results.json`'s
`threshold_selection.selected_threshold`).

## Applying the frozen rule to primary

`tools/apply_frozen_aggregation.py` performs **no tuning** -- it loads
`frozen_aggregation.json`, applies the `max >= 0.14` rule mechanically to
every clip in `primary_window_scores.csv`, and reports what happens:

| | carve (selection) | primary (frozen application) |
|---|---:|---:|
| precision | 0.8019 | 0.7882 |
| recall | 0.7143 | 0.8000 |
| F1 | 0.7556 | 0.7940 |
| FPR | 0.1736 | 0.2216 |
| accuracy | 0.7708 | 0.7893 |

Generalization from carve to primary: recall **+0.086**, precision
**-0.014**. Modest and in a defensible direction (recall improves, at a
small precision cost) -- not a red flag, and reported without adjustment.

### Why this is honestly below the whole-clip clean baseline

The clean baseline (`docs/clean_baseline_results.md`) reports accuracy 0.8325
and recall 0.91 on the *same 394 primary clips*, using **single-pass
whole-clip inference** (16 frames uniformly sampled across the full ~150-frame
clip). The sliding-window aggregation result above (accuracy 0.789, recall
0.80) is measurably lower. This is not a bug and not evidence the
integration is broken -- it is a real, expected cost of the streaming
regime: 16-frame sliding windows (stride 8) see less temporal context per
inference than one carefully-sampled whole-clip pass, and the aggregation
rule has to summarize 17 separate, noisier per-window opinions into one
decision instead of trusting a single confident one. The trade being made
explicitly here is: give up some accuracy in exchange for a signal that
updates continuously as a live video streams in, rather than only after an
entire clip has been observed. State this plainly in the final report and
in the presentation -- it is a more defensible finding than a suspiciously
perfect match would be.

## Provenance discipline in the adapter

`src/temporal_event_adapter.py` tags every confidence value it produces
`measured_model_probability`. `src/risk_assessment.py`'s
`_confidence_provenance()` tags the pre-existing Phase 4 rule-engine
confidences (0.9 for Stationary Person, 0.95 for Restricted Area Entry)
`declared_rule_constant`. This distinction is written into
`RiskAssessment.confidence_provenance` and appears in every JSON record and
every markdown report the pipeline produces, so a reader can never mistake
an engineer's fixed weight for a calibrated model output, or vice versa.

## What is NOT yet true

- Live inference (frames -> `TemporalInferenceEngine` -> window scores,
  bypassing the CSV) is supported by `TemporalEventAdapter`'s interface (it
  only needs a list of `WindowScore`-shaped values) but has not been
  exercised end-to-end, because `best.pt` (132.74 MB) has not been
  downloaded to this machine. `tools/run_demo.py` uses
  `PrecomputedWindowScoreSource` instead, reading real scores from
  `temporal_risk/primary_window_scores.csv`.
- The frozen aggregation rule was selected once, on one carve split
  (`carve_seed=42`). It has not been cross-validated across multiple carve
  seeds; that would be a reasonable robustness check for a longer project,
  but was out of scope given the review deadline.
