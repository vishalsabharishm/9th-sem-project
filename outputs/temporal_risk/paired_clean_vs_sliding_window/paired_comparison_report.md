# Paired comparison — clean baseline vs sliding window

Two inference regimes, the **same clean-baseline checkpoint**, the **same 394
primary clips**. The comparison isolates the inference regime, not the model.

Alignment empirically established, not assumed — see §1. Reads only; no
committed evaluation artifact was modified.

Produced by four new repository files:

| File | Role |
|---|---|
| `tools/recover_clean_baseline_alignment.py` | establishes the clip alignment by local re-inference |
| `tools/paired_clean_vs_sliding.py` | paired statistics, disagreements, early detection |
| `tools/alignment_evidence.py` | seeded permutation test and tie-invariance proof |
| `tests/test_paired_comparison.py` | pins all of the above (18 tests) |

---

## 1. Alignment — established, not assumed

The recovered clean-baseline `predictions.csv` carries
`index,true_label,fight_probability,predicted_label` and **no clip
identifier**. Row order alone is not evidence, so the alignment was tested.

**Hypothesis.** `src/evaluate_temporal_baseline.py` writes exactly that header,
iterates the primary evaluation dataset with `shuffle=False`, and that dataset
comes from `rwf2000_splits` sorted relative paths. So recovered row *i* should
be sorted primary clip *i*.

**Test.** The recovered checkpoint was re-run locally, whole-clip, in known clip
order, and the resulting probability sequence compared position by position.

| Evidence | Value |
|---|---|
| Clean predictions SHA-256 | `ee6c2699…a1a58ce3` (verified) |
| Checkpoint SHA-256 | `a32271bf…55a559a` (verified) |
| Label mismatches | **0 / 394** |
| Exact agreement at six decimals | **284 / 394** |
| Mean absolute difference | **4.48e-06** |
| Max absolute difference | **3.16e-04** |
| Rows over 1e-3 tolerance | **0** |

**Permutation test** (`tools/alignment_evidence.py`, seed 42, 20,000 draws).
Clips are re-assigned at random *within* the label blocks — the hard null, since
the recovered file already fixes the label sequence, so only label-consistent
mappings compete. Those permutations give mean |difference| between 2.2066e-01
and 3.0544e-01, mean 2.6686e-01. The hypothesised alignment scores 4.4756e-06:
**59,625× better than a typical wrong mapping**, and better than **every one**
of the 20,000 permutations. Permutation p = 5.0e-05 (add-one estimator), bounded
by the number of permutations rather than by the data.

**Residual ambiguity is provably harmless.** 99 adjacent clip pairs have local
probabilities within 1e-6, so value-matching alone could not separate them. Of
those, 8 are cross-label and therefore inadmissible (the recovered file states
each row's label, and 0 mismatched); the remaining 91 all share the same label
*and* the same clean decision.

Adjacent pairs understate the freedom, though — a run of mutually-close clips
can be permuted arbitrarily, not merely swapped pairwise. The check is therefore
run over the transitive closure: **2,920 admissible swaps** across all tie
clusters, of which **0** would move a single cell.

The argument, machine-checked in `alignment_evidence.json`: a swap of clips *i*
and *j* moves clean decision *dᵢ* onto clip *j* and *dⱼ* onto clip *i*, while
the sliding-window outcome stays with the clip. Admissibility forces
*labelᵢ = labelⱼ*, so a clean decision's correctness does not change when it
moves between them. Hence when *dᵢ = dⱼ* the contributed cells are identical
before and after:

```
before:  (clean correct, sliding correct) + (clean correct, sliding wrong)
after :  (clean correct, sliding wrong)   + (clean correct, sliding correct)
```

So the discordant counts `b` and `c` — and every McNemar p-value derived from
them — are **invariant** under every admissible swap. The paired tests below do
not depend on resolving those ties.

**Coverage.** Both regimes cover exactly the same 394 clips; labels agree for
every clip; no clip appears in one and not the other.

**Checkpoint provenance.** `window_scoring_manifest.json` records the scoring
checkpoint as `notebook7bb9a86555/clean_experiment/training/checkpoints/best.pt`
— the clean baseline. It is **not** the Experiment-1 checkpoint
(`notebook5d98537e15`), which is not used anywhere in this analysis.

---

## 2. The two regimes are not two thresholds

| | Clean baseline | Sliding window |
|---|---|---|
| Input | 16 frames sampled across all 150 | 17 windows of 16 consecutive frames, stride 8 |
| Statistic | one clip-level probability | **maximum** of 17 window probabilities |
| Threshold | 0.16 | 0.14 |

The thresholds are numerically close, but they act on **different random
variables**. The maximum of 17 draws is stochastically larger than a single
draw, so 0.14-on-a-maximum is a materially more permissive operating point than
0.16-on-a-single-value. Reading the comparison as "0.16 vs 0.14" would be wrong.

---

## 3. Confusion matrices and metrics

| | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| Clean baseline | 182 | 48 | 146 | 18 |
| Sliding window | 160 | 43 | 151 | 40 |

| Metric | Clean | Sliding | Difference |
|---|---:|---:|---:|
| accuracy | 0.8325 | 0.7893 | **−0.0431** |
| precision | 0.7913 | 0.7882 | −0.0031 |
| recall | 0.9100 | 0.8000 | **−0.1100** |
| specificity | 0.7526 | 0.7784 | **+0.0258** |
| F1 | 0.8465 | 0.7940 | −0.0525 |

Wilson 95% intervals on accuracy: clean [0.7924, 0.8661], sliding
[0.7464, 0.8267]. These are *marginal* intervals and overlap — but overlap of
marginal intervals is not a paired test and is not used to draw a conclusion.

---

## 4. Paired tests

### Overall accuracy

|  | Sliding correct | Sliding wrong |
|---|---:|---:|
| **Clean correct** | 282 | 46 |
| **Clean wrong** | 29 | 37 |

McNemar exact, two-sided: **p = 0.0639** (b = 46, c = 29, 75 discordant).

**Not distinguishable at the 5% level.** The direction favours the clean
baseline, but the evidence does not exclude chance.

Paired bootstrap for the accuracy difference (sliding − clean), 10,000
resamples, seed 42:

| Level | 95% CI |
|---|---|
| clip-level | [−0.0863, **+0.0000**] |
| source-video-level | [−0.0969, **+0.0083**] |

Both intervals include zero. The source-video interval — which respects the
fact that the 394 clips come from ~174 source videos — is wider, as expected.

### Fight clips only (n = 200): does each regime detect the fight?

|  | Sliding detects | Sliding misses |
|---|---:|---:|
| **Clean detects** | 154 | 28 |
| **Clean misses** | 6 | 12 |

McNemar exact: **p = 0.000195** (b = 28, c = 6).

**The sliding-window regime misses significantly more fights.** 28 fights the
whole-clip pass catches are missed by the windowed rule, against only 6 the
other way. This is the clearest single finding in the analysis.

### NonFight clips only (n = 194): does each regime correctly reject?

|  | Sliding rejects | Sliding false-alarms |
|---|---:|---:|
| **Clean rejects** | 128 | 18 |
| **Clean false-alarms** | 23 | 25 |

McNemar exact: **p = 0.533** (b = 18, c = 23).

Not distinguishable, and the direction slightly **favours** the sliding window
(43 false alarms vs 48).

### Why the overall test is null

The two strata move in opposite directions. A large, significant recall loss
(−0.11) is partly offset by a smaller, non-significant specificity gain
(+0.026). Aggregating them into overall accuracy averages away a real effect —
which is why the stratified tests matter more here than the headline one.

---

## 5. Early detection

From the same committed window scores. Censoring is administrative and
identical: every non-detection is censored at 4.800 s. That is the completion
time of the **final** window (frames 128–143, so (143 + 1) / 30), not a time
beyond it — so a clip that genuinely alarms on the last window carries the same
timestamp, and **3 of the 160 detections do**. Those 3 are observed detections,
not imputations. **The 4.8 s censoring time is never imputed as a detection
time**, and censored clips are excluded from every latency statistic below.

| | Value |
|---|---:|
| Fight clips | 200 |
| Detected | 160 |
| Not detected (right-censored) | 40 |
| Detected on the **first** completed window | **82** (51% of detections) |

| Statistic | Conditional on detection | Over all 200 Fight clips |
|---|---:|---:|
| q25 | — | 0.533 s |
| median | 0.533 s | **0.800 s** |
| p75 | 1.600 s | **3.733 s** |
| mean | 1.272 s | *undefined under censoring* |

Quantiles at or below the detected fraction (0.80) are **exact**, not
estimated: all three fall strictly below 4.800 s, so no censored clip could
change them regardless of when its alarm would have arrived. Statistics above
that fraction — including the conditional mean — are not identified under
censoring and are reported as such.

False alarms: 43, median 1.867 s, mean 2.307 s — later than true detections
(median 0.533 s conditional).

**This is the operational counterweight to the recall loss.** The whole-clip
regime produces no decision at all until the clip ends; the windowed regime
flags 41% of all Fight clips within 0.533 s and half within 0.800 s. That
capability does not exist in the clean baseline at any accuracy.

---

## 6. Disagreements

75 clips where exactly one regime is correct. Full detail in
`disagreement_table.csv`.

| True label | Clean right / sliding wrong | Clean wrong / sliding right |
|---|---:|---:|
| Fight | **28** | 6 |
| NonFight | 18 | **23** |
| **Total** | **46** | **29** |

The asymmetry is concentrated in the Fight stratum. On NonFight clips the two
regimes disagree almost symmetrically, slightly favouring the sliding window.

**No causal claim is made from this table.** It shows *where* the regimes
differ, not *why*. Attributing the pattern to any specific mechanism — window
length, aggregation rule, temporal localisation of the violent action — would
need a targeted experiment that this analysis does not perform.

---

## 7. Answers to the research questions

**1. Is the accuracy difference statistically distinguishable under a paired
test?** No. McNemar exact p = 0.0639; both bootstrap CIs include zero. The
point estimate favours the clean baseline by 4.3 points, but the data do not
exclude chance at the 5% level.

**2. Does the sliding window improve or worsen Fight recall?** **Worsens it,
significantly.** Recall 0.91 → 0.80; paired p = 0.000195. This is the one
clearly established difference.

**3. Does it improve or worsen false alarms on NonFight?** Slightly improves
them (48 → 43 false alarms, specificity +0.026), but **not significantly**
(p = 0.533). Treat as "no detectable difference".

**4. Does the temporal regime provide an operational benefit through earlier
alarms?** Yes, and it is not a matter of degree — the clean baseline emits **no
decision at all** until the clip ends. The windowed regime alarms on 82 of 200
Fight clips at 0.533 s and reaches its median at 0.800 s. Whether that benefit
outweighs 22 additional missed fights is a deployment judgement, not a
statistical one.

**5. Evidence supporting keeping sliding-window inference.** Its overall
accuracy is not statistically distinguishable from the clean baseline; its
false-alarm behaviour is no worse and possibly slightly better; and it is the
only regime that can produce a mid-clip alarm at all. The system's purpose is
streaming surveillance, which the whole-clip regime cannot serve.

**6. Evidence arguing for further calibration/aggregation research.** The
recall loss is real and significant, and it is a property of the *aggregation
rule*, not of the model — both regimes share weights. `max ≥ 0.14` was selected
once on one carve split under a precision/FPR constraint, never cross-validated
across seeds, and the alternatives (`k_of_n`, `mean`) were swept only at that
one operating point. Recovering some of the 28 lost detections without
surrendering the specificity gain is a well-posed, unexplored question.

**7. Conclusions NOT justified by these experiments.**

- That the sliding-window regime "is worse". Overall accuracy is not
  distinguishable; only recall is.
- That whole-clip inference is significantly better overall. p = 0.0639 is not
  significance, and reporting it as a trend would be reading a null result as
  positive.
- Any causal explanation of the disagreement pattern.
- Any claim about generalisation beyond these 394 clips, which come from ~174
  source videos and one training seed.
- That specificity genuinely improved — the direction is favourable but the
  test is null.
- Anything about the deployed end-to-end system: this compares the temporal
  branch alone, with no detector, tracker, rule engine or risk mapping involved.

---

## 8. Reproducing this analysis

```bash
# 1. Establish the clip alignment (needs the recovered checkpoint; ~11 min CPU)
python tools/recover_clean_baseline_alignment.py     --clean-predictions <path to recovered predictions.csv>     --checkpoint models/temporal_violence/best.pt

# 2. Paired statistics                     -> paired_comparison.json, disagreement_table.csv
python tools/paired_clean_vs_sliding.py

# 3. Alignment evidence (~1 s, no checkpoint) -> alignment_evidence.json
python tools/alignment_evidence.py

# 4. Verify
python -m pytest tests/test_paired_comparison.py
```

Steps 2 and 3 are byte-deterministic — verified by re-running each three times
and comparing SHA-256. Every random procedure is seeded (bootstrap 42,
permutation 42) and the seed is recorded in its own output. Step 1 depends on
the checkpoint, which is deliberately not in the repository; the tests skip
cleanly when its output is absent.

---

## 9. Limitations

1. **One checkpoint, one seed.** No variance estimate over training runs.
2. **Non-independent clips.** 394 clips from ~174 inferred source videos; the
   source-video bootstrap accounts for this, the Wilson intervals do not.
3. **Source grouping is inferred** from the `<id>_<n>.avi` filename convention,
   not read from dataset metadata.
4. **Alignment is empirically established, not certified.** The permutation
   evidence is overwhelming and the residual ambiguity is provably table-
   invariant, but the original Kaggle run's clip order was never recorded.
5. **The comparison is of regimes at their frozen operating points**, not of
   the best achievable version of either.
