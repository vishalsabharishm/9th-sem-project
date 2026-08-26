# Merge manifest — temporal-risk integration work

An honest record of what was added or changed in this repository during the
temporal-risk recovery and integration pass (2026-08-25, two days before the
final review), and where each artifact actually came from. This document
exists because an earlier status description of this work (referencing a
completed "Phase A" merge and locally-present "clean baseline" artifacts)
did not match the repository's actual state when checked -- neither the
claimed new source/tool/doc files nor the Phase B `temporal_risk/` outputs
existed anywhere in the working tree or git history at the start of this
pass. Everything below was verified against the filesystem, not assumed.

## What was recovered (not authored) from Kaggle

These already existed as real completed runs on Kaggle; they were fetched
via browser automation (no direct network path from the recovery
environment to Kaggle's storage) and written into this repository for the
first time:

| file | source |
|---|---|
| `temporal_risk/carve_window_scores.csv` | Kaggle notebook `notebookbce85b4c2b` output, `temporal_risk/carve_window_scores.csv` (4080 rows, 240 clips x 17 windows -- verified) |
| `temporal_risk/primary_window_scores.csv` | same run, `primary_window_scores.csv` (6698 rows, 394 clips x 17 windows -- verified) |
| `temporal_risk/window_scoring_manifest.json` | same run's manifest |
| `docs/clean_baseline_results.md` + `temporal_risk/clean_baseline_*.json` | Kaggle notebook `notebook7bb9a86555` version 3, output folder `clean_experiment/` (`final_evaluation/metrics.json`, `final_evaluation/classification_report.txt`, `frozen.json`, `training/training_history.json`, `threshold_selection/threshold_sweep.txt`) -- this is the source of the threshold=0.16 / accuracy=0.8325 / ROC-AUC=0.9251 numbers |

The checkpoint itself, `clean_experiment/training/checkpoints/best.pt`
(132.74 MB), was **not** downloaded -- confirmed to exist and be that exact
size via the Kaggle UI, but not transferred. See
`docs/temporal_risk_integration.md`, "What is NOT yet true."

## What was newly written (in this pass)

| file | purpose |
|---|---|
| `tools/select_temporal_aggregation.py` | Sweeps any/k_of_n/max/mean aggregation candidates on `carve_window_scores.csv` only; writes `temporal_risk/frozen_aggregation.json`. |
| `tools/apply_frozen_aggregation.py` | Applies the frozen rule to `primary_window_scores.csv`, no tuning; writes `temporal_risk/primary_aggregation_result.json`. |
| `src/temporal_event_adapter.py` | `FrozenAggregationRule`, `PrecomputedWindowScoreSource`, `TemporalEventAdapter` -- turns window scores into an `EventDetection`, tagging provenance. |
| `src/risk_assessment.py` (edited) | Added `Temporal Violence Signal -> High` to the default risk mapping, added `RiskAssessment.confidence_provenance` and `_confidence_provenance()` to distinguish `measured_model_probability` from `declared_rule_constant`. Backward compatible: existing fields and behavior for the four pre-existing event types are unchanged; the new field defaults to `None` and the existing test suite (`tests/test_risk_assessment.py`, `tests/test_abnormal_events.py`) passes unmodified. |
| `tools/run_demo.py` | End-to-end demo runner: video -> YOLO -> tracking -> Phase 4 rules -> temporal signal -> risk assessment -> annotated video + JSON + markdown explanation. |
| `tests/test_temporal_event_adapter.py` | Unit tests for the aggregation-rule replay logic, plus two tests that run the adapter over the *actual* recovered `carve_window_scores.csv` and cross-check its output reproduces `frozen_aggregation.json`'s recorded confusion matrix exactly -- this is a correctness check against real data, not just synthetic examples. |
| `docs/clean_baseline_results.md` | Full write-up of the recovered clean-baseline run (see table above). |
| `docs/temporal_risk_integration.md` | Explains the whole pipeline from checkpoint to risk assessment, the carve/primary discipline, and why the primary-split aggregation result is honestly below the whole-clip clean baseline. |
| `docs/demo_instructions.md` | How to run `tools/run_demo.py` and what to point at during the review. |
| `README.md` (edited) | Corrected the stale "0.8604 / ROC-AUC 0.9432" status line (Experiment 1's biased numbers) to the clean baseline's "0.8325 / ROC-AUC 0.9251"; updated the temporal-branch section, phase-status section, risk-mapping list, and project-structure block to describe the now-completed integration instead of the old "not yet wired in" status. |

## What was verified, not just written

- `tools/select_temporal_aggregation.py` was actually run against the
  recovered `carve_window_scores.csv` (240 clips); its output
  (`frozen_aggregation.json`) reflects a real sweep of 215 candidates, not a
  hand-picked answer.
- `tools/apply_frozen_aggregation.py` was actually run against
  `primary_window_scores.csv` (394 clips); its output
  (`primary_aggregation_result.json`) is the real, once-only application of
  the frozen rule.
- The full pytest/unittest suite for the affected modules
  (`tests/test_abnormal_events.py`, `tests/test_risk_assessment.py`,
  `tests/test_temporal_event_adapter.py`, `tests/test_end_to_end_fixture.py`)
  was run in a sandboxed mirror of this repository (same source files,
  same `requirements.txt` versions where installable) -- 13 passed, 1
  skipped (the fixture test, which needs `data/person_fixture.mp4` and was
  not staged into that sandbox; it is present in this repository).
- `tools/run_demo.py` was actually executed against all three built-in
  clips (`fight`, `nonfight`, `fp`) in that same sandbox, producing real
  annotated videos, risk-assessment JSON, and explanation reports -- not
  just written and assumed to work. Those three output sets are included
  in this delivery under `outputs/demo/` as a working reference in case a
  live re-run during the review is undesirable.

## What is still open

See `docs/temporal_risk_integration.md`'s "What is NOT yet true" and this
project's own priority list: live on-device temporal inference (blocked on
downloading `best.pt`), a cross-validated aggregation-rule robustness check,
and the remaining documentation/PPT items for the final review.
