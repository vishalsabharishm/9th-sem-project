# What the frozen spatial feature actually measures

This note exists because the frozen protocol's own prose describes the spatial
feature more loosely than the implementation computes it. The implementation is
correct in the only sense that matters for this project — it is the computation
that produced every frozen threshold and every locked result — but a reader who
takes the prose literally will form the wrong picture of the numerator.

Nothing here changes any number. This is a terminology correction.

## The short version

The feature is:

```
spatial = mean(displacement samples) / mean(person-group diagonal)     [ratio of means]
```

The **denominator** is what the prose says: the mean, over frames containing at
least one person, of the diagonal of the box enclosing all person boxes in that
frame.

The **numerator** is *not* a frame-to-frame speed. Each displacement sample is
the Euclidean distance between a track's centroid **now** and that same track's
centroid **two appearances earlier in its own history** — skipping one
intervening appearance.

The frozen protocol
(`temporal_risk/frozen_confirmatory_fusion_protocol.json`) describes it as
*"Euclidean centroid displacement in raw image pixels between consecutive
frames of the same track"*. That sentence describes a one-transition
displacement. The code computes a two-appearance one.

## Why the code does this

`BehaviorAnalyzer.get_previous_bbox(id)` returns `history[-2]`, not
`history[-1]`:

```python
def get_previous_bbox(self, object_id):
    history = self._bbox_history.get(object_id, [])
    if len(history) < 2:
        return None
    return history[-2]
```

The frozen spatial scorer calls it **before** `update_history` has appended the
current frame:

```python
for track in persons:
    previous = behaviour.get_previous_bbox(track.id)      # history[-2]
    if previous is not None:
        speeds.append(behaviour._movement_pixels(previous, track.bbox))
...
behaviour.update_history(snapshot, ...)                   # current frame appended LAST
```

At the moment of the call, `history[-1]` is the track's bbox at its most recent
*previous* appearance and `history[-2]` is the appearance before that. The
sample therefore spans from appearance *k−2* to the current appearance *k*.

## Exactly what this implies

Verified empirically; see `tests/test_operational_fusion.py`,
`SpatialHistoryOffsetTests`.

| Question | Answer |
|---|---|
| Which history element is used? | `history[-2]`, read before the current frame is appended |
| How many transitions are spanned? | Two *appearances* of that track. Under continuous tracking that is two frame transitions; a track moving 10 px/frame yields samples of **20.0 px**, not 10.0 |
| Does it depend on warm-up? | Yes. A track produces no sample at its 1st or 2nd appearance (`len(history) < 2` → `None`). A track appearing *k* times contributes **max(k − 2, 0)** samples |
| Is the span always 2 wall-clock frames? | **No.** History records only frames where the track was matched to a detection. Across a tracker dropout the same two-appearance gap can span many more wall-clock frames — e.g. a track seen at frames 0, 1, 5 gives, at frame 5, a sample from frame 0: a 5-frame span |
| Do non-person tracks interfere? | No. History is keyed per track id, so a car in the frame does not perturb a person's samples |
| Does the ratio use exactly these samples? | Yes. The numerator is the mean of exactly this sample set; the denominator is computed independently from per-frame group geometry |

So the accurate one-line description is:

> mean centroid displacement computed over the implementation's two-appearance
> history offset, normalised by the mean person-group diagonal (ratio of means)

Acceptable shorter forms: *"two-appearance centroid displacement"*,
*"displacement over the implementation's two-frame history offset"*.

Do **not** describe it as one-frame speed, consecutive-frame displacement, or
frame-to-frame velocity. The code does not compute those.

## Why it is not being fixed

Because it is not a defect in the result — only in the description of it.

The same computation, with the same offset, generated all of:

- the development carve spatial distribution (221 defined values) that is the
  percentile reference stored inside the frozen protocol;
- the calibrated thresholds `theta_s`, `theta_or` and `theta_f`;
- `outputs/fusion/FINAL_primary_spatial_scores.json`;
- the confirmatory result in
  `outputs/fusion/FINAL_confirmatory_primary_evaluation.json`.

Every threshold and every percentile is therefore expressed in these units, and
the comparison between development and primary is internally consistent.
Changing the offset now would put new video on a different scale from the frozen
thresholds while leaving the thresholds unchanged — which would silently
invalidate comparability with the locked artifacts. That is a far worse outcome
than an imprecise sentence.

The magnitude of the effect is also bounded and uninteresting: for smooth
motion a two-appearance displacement is roughly twice a one-appearance one, a
near-constant scale factor that the percentile transform and the calibrated
thresholds absorb. It does not change any ranking, and the confirmatory result
(Decision C, no statistically significant improvement from fusion) does not
depend on it.

## Why the frozen prose was not corrected in place

The wording lives in `temporal_risk/frozen_confirmatory_fusion_protocol.json`,
whose SHA-256 is recorded in `outputs/fusion/FINAL_experiment_lock_record.json`
and checked at load time by both the research tool and
`src/operational_fusion.py`. Editing a single character of that file changes its
hash and would make every integrity check fail.

The three tools that generate these artifacts are equally untouchable — each has
its own SHA-256 recorded inside the artifact it produced:

| Tool | Hash recorded in |
|---|---|
| `tools/run_confirmatory_primary_evaluation.py` | `FINAL_confirmatory_primary_evaluation.json` |
| `tools/calibrate_fusion_protocol.py` | `frozen_confirmatory_fusion_protocol.json` |
| `tools/score_fresh_carve.py` | `fresh_carve_fit_scores.json` |

Editing the prose inside `calibrate_fusion_protocol.py` would additionally mean
re-running it no longer reproduces the frozen protocol byte-for-byte, breaking
the reproducibility claim to fix a comment.

So the artifacts and their generators are left exactly as they are, and this
note is the correction of record. The operational implementation
(`src/operational_fusion.py`) reproduces the frozen behaviour exactly — pinned
by an equivalence test against the research tool on real video — and describes
it accurately in its own docstrings.

## For the write-up

Describe the feature as measured over a two-appearance history offset, or state
the discrepancy explicitly. A reader who inspects `get_previous_bbox` will find
it, and it is better volunteered than discovered.
