# Confirmatory fusion result — Decision C

**Status: CLOSED. The primary split is permanently spent.**

This document exists because the headline research result of the project lived
only in JSON artifacts and commit messages. Anyone reading the repository could
previously have missed it entirely.

---

## The question

Does decision-level fusion of interpretable spatial motion evidence with the
sliding-window R3D-18 violence score improve clip-level abnormal-event
classification over the temporal branch alone?

## The answer

**No meaningful improvement.** Under a protocol frozen and committed *before*
the evaluation set was opened, fusion did not beat the temporal-only incumbent.

| Candidate | TP | FP | TN | FN | Accuracy | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0 temporal-only** | 160 | 43 | 151 | 40 | **0.7893** | 0.8000 | 0.7784 | 0.7940 |
| **F2 fusion** | 160 | 46 | 148 | 40 | 0.7817 | 0.8000 | 0.7629 | 0.7882 |
| F1 or-gate | 160 | 43 | 151 | 40 | 0.7893 | 0.8000 | 0.7784 | 0.7940 |
| B1 spatial-only | 67 | 35 | 159 | 133 | 0.5736 | 0.3350 | 0.8196 | 0.4437 |

**Primary endpoint** — exact two-sided McNemar, α = 0.05:
`b = 25, c = 22, p = 0.770867`. Not significant; direction slightly favours B0.

**Effect size** — accuracy difference (F2 − B0) = **−0.0076**
clip-level 95% CI `[−0.0406, +0.0254]`, source-clustered `[−0.0463, +0.0309]`.
Both contain zero. The source-clustered interval is wider because 394 clips come
from only 174 source videos and are therefore not independent.

## Three findings worth more than the headline

**F2 traded evenly on Fight and lost on NonFight.** It recovered 13 Fight clips
B0 missed and lost 13 B0 caught — exactly offsetting, Fight-stratum `b=13 c=13
p=1.0`, recall unchanged at 0.8000. It then added 12 false alarms and removed 9.
The entire accuracy difference is those 3 net NonFight clips.

**F1 was inert.** Identical to B0, `b=0 c=0`. At a matched alarm budget the
calibrated OR threshold admits no primary clip the temporal rule had not already
fired on. It behaved the same way on development data.

**Spatial evidence is real but not complementary here.** B1 alone reaches
ROC-AUC **0.7058** on primary — far from chance. The spatial branch is not
uninformative; its signal simply did not survive fusion at a fixed operating
point.

## Why the development result did not transfer

On the 240-clip development carve, F2 showed **+14 true positives at identical
false positives**. On primary it showed none.

That gap is exactly what a calibration set overstates: F2's threshold was chosen
on those clips, so its development advantage was optimistically biased by
construction. The commit that froze the protocol said this figure "must not be
quoted as evidence"; primary confirmed the judgement. It is the cleanest
illustration in this project of why development numbers are not results.

## The contamination discovery

An earlier fusion attempt was built on a "fresh" carve drawn from the untouched
training remainder — source-disjoint from everything, untouched by *selection*.
It was not untouched by *training*. `clean_baseline_training_summary.json`
records `holdout_validation: true`, `validation_fraction: 0.15`, so the 240-clip
carve **is** the R3D holdout and the remainder is the training set.

The signature, same frozen rule throughout:

| Set | Role vs R3D training | Recall | FPR |
|---|---|---:|---:|
| fresh carve, fit split | **training data** | **0.9583** | 0.0577 |
| existing 240-clip carve | **holdout** | 0.7143 | 0.1736 |
| primary 394 | held-out evaluation | 0.8000 | 0.2216 |

A **+0.244 recall gap**. Source-disjointness guards against near-duplicate
leakage between splits; it cannot undo gradient updates. That experiment was
abandoned at 120/240 clips, the eval split was never read, and no fusion number
was produced from it. See `outputs/fusion/fusion_blocked_contamination.md`.

This is arguably the most transferable finding in the project: **a split can be
source-disjoint and still not be training-disjoint.**

## What this does and does not establish

**Establishes:** this frozen fusion rule, at this operating point, does not
improve on the temporal-only detector on this split.

**Does not establish:** that spatial motion is uninformative (B1's AUC says
otherwise); that fusion is impossible with a different feature, operating point
or combination rule; equivalence — non-significance is not proof, and the
intervals admit roughly four points either way; anything about other datasets,
real-time operation, or the risk layer.

## Scope

**Clip-level offline confirmatory classification.** Not a causal mid-clip alarm,
not real-time, not early warning. The spatial feature is a whole-clip aggregate.
Measured throughput is ~34 s per 5 s clip on CPU. The earlier temporal
early-alarm analysis is a separate result and is not merged with this one.

Camera motion was **not** compensated and remains an unresolved confounder.

## Provenance

| Artifact | SHA-256 |
|---|---|
| Primary result | `7d53b487c31ef700b1fce91fdbccac5b3ad050cc6579316be2cc7e4693084670` |
| Protocol (internal) | `ff2d78c5ae064e16b1144b74193b40eb66d71f63069c8594faf13238fb6384cd` |
| Protocol (file) | `e443325a0bcb6b8177c8a24b0da7cce8b40be5ac3a11b5ad21ba3c422f2f7b72` |
| Primary window scores | `4ac143db48f64970dc987f8f469d2f79a71ab4dd0a7310ed13ad9e0f8e64be4d` |
| R3D-18 checkpoint | `a32271bfb5a9c273f84672b16ecd5fa351fdd617d39a2c5c73f5c667255a559a` |
| YOLOv8s | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` |

Statistics: exact McNemar on discordant pairs; paired percentile bootstrap,
10,000 resamples, seed 42, at clip and source-video level; Wilson intervals for
binomial proportions. Source grouping via `metric_intervals.source_video_id`.

Every reported number was independently recomputed from the per-clip table
rather than read back from summary fields — sample counts, all four confusion
matrices, every metric, `b`/`c`, the exact p-value, and every F2 score from the
stored development references. **All checks passed.**

## Artifact map

| File | Contents |
|---|---|
| `outputs/fusion/FINAL_confirmatory_primary_evaluation.json` | the result, per-clip table included |
| `outputs/fusion/FINAL_experiment_lock_record.json` | closeout; `primary_split_status: PERMANENTLY SPENT` |
| `outputs/fusion/FINAL_primary_spatial_scores.json` | primary spatial features |
| `temporal_risk/frozen_confirmatory_fusion_protocol.json` | **the authoritative protocol** |
| `temporal_risk/frozen_fusion_protocol.json` | **SUPERSEDED v1** — fresh-carve based, invalid; do not cite |
| `outputs/fusion/fusion_blocked_contamination.md` | the contamination discovery |
| `outputs/fusion/development_failure_modes.json` | failure characterization |
| `tools/run_confirmatory_primary_evaluation.py` | the one-shot runner |

> **Naming hazard.** `frozen_fusion_protocol.json` and
> `frozen_confirmatory_fusion_protocol.json` sit in the same directory and
> differ by one word. The first is the **superseded** v1, calibrated on a carve
> later proved to be training data. It has already been mis-cited once. The
> authoritative protocol is identified by `protocol_sha256 = ff2d78c5…` and by
> its four thresholds (0.14 / 0.023440 / 0.441209 / 0.491335). Always verify by
> SHA, never by filename.

## The lock

The 394-clip primary split has been used for its single authorised confirmatory
evaluation. It must never be used again for development, tuning, model
selection, threshold search, feature design, or any further experiment. Any
future confirmatory claim requires a new dataset and a newly frozen protocol.
