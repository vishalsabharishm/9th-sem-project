# Fusion validation blocked — the fresh carve is R3D training data

**Status: fusion parked. No fusion result is claimed.**
Discovered after the fit split was scored, before any eval-split result was read.

---

## 1. The finding

The fresh validation carve was drawn from the untouched remainder of the
RWF-2000 training split. It is untouched by *selection* — never used for
checkpoint or aggregation choices — and it is source-disjoint from both the
240-clip carve and the held-out primary split. All of that is true and was
verified.

It is **not** untouched by *training*. `temporal_risk/clean_baseline_training_summary.json`
records:

```
holdout_validation   : true
validation_fraction  : 0.15
monitor_set          : "train-carved validation"
```

1600 train clips, 15% held out = the 240-clip carve. The remaining **1360 clips
are the R3D-18 training set**, and the fresh carve was drawn entirely from them.
So every temporal score on the fresh carve is a prediction on data the model
was fitted to.

Source-disjointness does not help here. It protects against *near-duplicate
leakage between splits*; it cannot undo the fact that these specific clips
carried gradient updates.

## 2. The empirical signature

The frozen rule `max >= 0.14`, unchanged, applied to three sets that differ only
in their relationship to R3D training:

| Set | Role w.r.t. R3D training | Recall | FPR | TP | FP | TN | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| fresh carve, fit split (n=100) | **training data** | **0.9583** | **0.0577** | 46 | 3 | 49 | 2 |
| existing 240-clip carve | **holdout** (the monitor set) | 0.7143 | 0.1736 | 85 | 21 | 100 | 34 |
| primary 394 (recorded, not re-read) | held-out evaluation | 0.8000 | 0.2216 | 160 | 43 | 151 | 40 |

**Memorization gap: +0.244 recall** against the holdout, with false-positive
rate a third of it. This is the signature of evaluating a model on its own
training data, and it is large.

## 3. Why this invalidates the fusion experiment

Two independent reasons, either sufficient on its own.

**No headroom.** On the fit split the temporal branch misses **2 Fight clips out
of 48**, and the matched alarm budget is **3 false positives**. Extrapolating to
the 117 Fight clips of the eval split, the temporal branch would miss roughly
five. The frozen protocol requires 6–10 net discordant clips for exact McNemar
to reach significance. The experiment could not have produced a significant
result in either direction — it was degenerate before it was run. That is stop
condition 6 (sample size statistically inadequate).

**Structurally unfair comparison.** YOLOv8s is an off-the-shelf COCO detector
that has never seen RWF-2000, so the spatial branch is *not* memorizing these
clips. Fusion would therefore have pitted a memorized temporal branch against a
non-memorized spatial one. Any measured fusion effect would confound
"does spatial evidence add value" with "spatial evidence cannot compete with
memorization", and the direction of that bias is against fusion. Even a null
result would have been uninterpretable.

Evaluation-split scoring was **halted at 120/240 clips** rather than completed.
Finishing it would have produced a number that could not be honestly
interpreted, and the eval split was never read.

## 4. What is *not* invalidated

The spatial characterization stands. It was measured on the **existing 240-clip
carve**, which is R3D holdout data, and YOLO was never trained on any of it:

| Measurement (existing carve, held-out) | Result |
|---|---|
| `speed_mean` AUC, all 235 analyzable clips | 0.855 [0.803, 0.903] |
| scale-normalised `speed`, all 235 | 0.806 [0.745, 0.867] |
| `speed_mean`, high-quality subset (n=140) | 0.914 [0.863, 0.954] |
| source-clustered bootstrap | [0.773, 0.921] |
| **missed Fights vs correctly-rejected NonFight** | **0.867 [0.793, 0.926]** |
| same, high-quality subset | 0.920 [0.855, 0.970] |

The last two are the complementarity result: within the region where the
temporal branch is wrong, motion still separates the clips it should have caught
from the ones it correctly rejected. That evidence is unaffected by this
contamination and remains the reason fusion is worth pursuing.

The unresolved confounder also stands: **camera motion is not compensated**, and
RWF-2000 Fight clips may simply be filmed differently. Scale normalisation and
the quality-subset probe reduce but do not eliminate this.

## 5. Why no uncontaminated set remains

| Candidate set | Blocked by |
|---|---|
| Train remainder (1360 clips) | **R3D training data** — memorized |
| Existing 240-clip carve | spent three times for selection (best-epoch, 215-candidate sweep, v2 protocol) |
| Primary 394 | reserved for confirmatory evaluation; out of scope for this task |

There is no set on which fusion parameters can be both *fitted* and *evaluated*
without either memorization or selection contamination. This is a property of
the data available, not a solvable engineering problem, and it is not resolved
by relaxing the standard.

## 6. The one valid path forward

Fit fusion parameters on the **existing 240-clip carve** — it is spent for
*selection*, but it is genuine R3D **holdout** data, which is the property that
matters here — then evaluate **once** on the primary 394 clips under a
pre-frozen protocol.

The selection contamination of the carve would bias *carve* estimates
optimistically, but no carve estimate would be reported: the confirmatory number
would come from primary, which is untouched. That is precisely what the primary
split was reserved for.

This requires primary access and is therefore explicitly out of scope for the
current task. It should be a separate, deliberately authorised phase, with the
protocol frozen and committed before primary is opened.

## 7. What was kept

The fresh carve manifest, its builder, the scorer and the frozen fusion protocol
are retained. They are correct and reusable: the manifest construction, the
source-grouped splitting, the provenance exclusion of the 27 non-ASCII clips,
and the calibration machinery are all independent of this contamination. What
changed is only the claim that the fresh carve can serve as a validation set for
a branch trained on it.

The fit-split scores are retained as the evidence for the memorization gap
measured in §2.
