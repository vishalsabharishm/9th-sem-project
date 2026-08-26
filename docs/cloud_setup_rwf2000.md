# RWF-2000 Cloud Dataset Setup

Procedure for acquiring, verifying, and inspecting RWF-2000 in the cloud-GPU
environment used for fine-tuning the learned temporal violence/action branch.

**Scope.** RWF-2000 trains only that one branch. It is not the dataset for the
project overall and is not responsible for crowd or crowd-interaction anomaly
detection, which remains the existing geometric `CrowdInteractionAnalyzer`.

**This document does not train anything.** It ends with an inspected dataset and
generated metadata, for review before any training is scheduled.

---

## 1. Verified source

| Field | Value |
|---|---|
| Record | <https://zenodo.org/records/15687512> |
| Version DOI | `10.5281/zenodo.15687512` |
| Concept DOI | `10.5281/zenodo.15687511` |
| Depositor | Cheng, Ming — Duke Kunshan University, ORCID `0000-0002-4733-3596` |
| Zenodo license | CC-BY-4.0 |
| Archive | 13 × `.7z` parts, 9,399,888,359 B (8.75 GiB) |

The depositor is the first author of the RWF-2000 ICPR 2021 paper, and Duke
Kunshan is the SMIIP Lab's institution.

**Do not substitute an unofficial mirror.** Third-party copies on other dataset
hosts relicense the data in ways that conflict with the original terms.

### Licensing position

The Zenodo deposit is CC-BY-4.0. The official repository additionally forbids
redistribution and commercial use without SMIIP Lab approval. We comply with the
**stricter union** of both:

- Academic, non-commercial use only.
- Do not redistribute the videos — do not commit them, re-upload them, or
  publish clips. On Kaggle, keep any uploaded dataset **private**.
- Respect the clause about mental health and personal privacy when choosing
  frames for the thesis or demo.
- Cite both the paper and the dataset DOI.

```bibtex
@INPROCEEDINGS{9412502,
  author={Cheng, Ming and Cai, Kunjing and Li, Ming},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  title={RWF-2000: An Open Large Scale Video Database for Violence Detection},
  year={2021}, pages={4183-4190},
  doi={10.1109/ICPR48806.2021.9412502}}
```

> Cheng, M. (2021). *RWF2000 — A Large Scale Video Database for Violence
> Detection* [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.15687512>

---

## 2. Environment comparison

| | **Kaggle Notebooks** | **Google Colab (free)** |
|---|---|---|
| GPU | P100 16 GB, or 2× T4 16 GB; 32 GB RAM | Usually T4, **not guaranteed** — verify with `nvidia-smi` each session |
| Quota | ~30 h/week GPU, published | Dynamic, unpublished, varies by account and demand |
| Session limit | 12 h | 12 h, plus ~90 min idle timeout |
| Working disk | `/kaggle/working` **20 GB, persistent** | ~40 GB local, **erased at session end** |
| Dataset persistence | Kaggle Datasets mounted read-only, survive sessions | Requires Google Drive (free tier 15 GB total) |
| Checkpoint retrieval | Notebook output / dataset download | Save to mounted Drive |

**Recommendation: Kaggle.**

The deciding factor is dataset persistence. The archive is 8.75 GiB and extracts
to roughly 12 GB. On Colab that is erased when the session ends, so every
training session would re-download ~9.4 GiB. Kaggle lets the extracted dataset be
attached read-only across sessions, so the download happens **once**. Kaggle also
publishes its quota and allocates a GPU deterministically, which matters for a
project that must report reproducible conditions.

**Storage caveat.** Peak usage during extraction is ~21 GB (archives + extracted),
which exceeds the 20 GB `/kaggle/working` quota. Extract into `/kaggle/temp`
instead — it is ephemeral but larger — and keep `/kaggle/working` for the small
outputs (metadata JSON, later checkpoints ≈130 MB).

---

## 3. Procedure

### Step 0 — session setup

```bash
nvidia-smi                        # confirm which GPU was allocated; record it
apt-get -qq install -y p7zip-full # multi-part 7z extraction
```

Pin versions to match local development so a checkpoint loads without surprises:

```bash
pip -q install "torch==2.13.0" "torchvision==0.28.0"
python -c "import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.cuda.is_available())"
```

### Step 1 — project code

Bring `src/` into the session (clone the repo, or upload `src/` as a private
Kaggle Dataset). Required modules:

```
src/config.py                    src/dataset_inspection.py
src/video_loader.py              src/rwf2000_config.py
src/prepare_rwf2000_dataset.py
```

### Step 2 — run the whole preparation sequence

One command performs acquisition → checksum verification → extraction →
layout discovery → inspection → leakage checks → metadata generation:

```bash
python src/prepare_rwf2000_dataset.py \
    --root      /kaggle/temp/rwf2000 \
    --archives  /kaggle/temp/rwf2000_archives \
    --metadata  /kaggle/working/rwf2000_metadata
```

The script is idempotent. Parts whose published MD5 already matches are not
re-downloaded, and extraction is skipped when the target already holds videos, so
interrupting and re-running is safe. Extraction only begins once **all 13**
checksums verify; if any part fails, the script stops with a non-zero exit and
nothing is extracted.

Expect roughly: download 10–20 min on Kaggle bandwidth, extraction a few minutes,
inspection a few minutes (metadata probing only — videos are never fully decoded).

### Step 3 — read the output

The script prints observed values only:

- discovered split and class directory names (**expect `val`, not `test`**)
- readable video count, per-split and per-class counts
- resolution / fps / frame-count distributions, duration min-mean-max
- unreadable files and unexpected formats
- cross-split filename collisions and duplicate groups
- `NOTE:` lines wherever observations differ from published figures

Published figures are used strictly as a cross-check and are never substituted
for observations. A `NOTE:` line is a prompt for human review, not a failure.

### Step 4 — persist the dataset (one time)

Create a **private** Kaggle Dataset from the extracted folder so later sessions
mount it read-only at `/kaggle/input/...` without re-downloading. Keep it private:
redistribution is not permitted.

### Step 5 — retrieve metadata

`/kaggle/working/rwf2000_metadata/` contains `rwf2000_metadata.json` (summary plus
per-video records) and `reproducibility.json`. Download both and keep them with
the project — they are the observed evidence for the thesis.

---

## 4. Split policy

**The official split is preserved. It is never randomly re-split.**

The dataset authors allocated clips so that clips sharing a source video are not
spread across splits, and the predefined split has been checked for leakage. A
random re-split would reintroduce exactly that leakage and inflate reported
accuracy.

Encoded in `rwf2000_config.SplitPolicy`:

| Setting | Value |
|---|---|
| `use_official_split` | `True` |
| `random_resplit_allowed` | `False` |
| `official_test_is_untouched` | `True` — held out for final evaluation only |
| `validation_carved_from` | `"train"` |
| `validation_fraction` | `0.15` |
| `validation_grouped_by_source_video` | `True` |

Threshold selection and early stopping use the validation subset carved from
**train**. The official held-out split is touched **once**, for final reporting.

If source-video identity cannot be recovered from filenames, group-aware
splitting is not possible — in that case state the limitation explicitly in the
thesis rather than silently using a random split.

---

## 5. After inspection

Review the observed statistics **before** any training. In particular confirm:

1. Discovered split/class names match what the loader will expect.
2. Observed class counts, and whether the set is actually balanced (this decides
   whether `class_weighting` stays `None`).
3. Observed fps and frame counts, which determine the temporal sampling stride —
   the current `SamplingConfig` assumes 150 frames at 30 fps.
4. Zero unreadable files, or a documented list of any exclusions.
5. Zero cross-split duplicates, or an investigation of any found.

Downloading pretrained R3D-18 Kinetics weights and starting fine-tuning are
**separate steps requiring separate approval**.

---

## 6. Kaggle extraction: the filename-length problem

A plain `7z x` of this archive **cannot** fully succeed on Kaggle. Twenty-four
`train/Train_Fight` clips have CJK filenames of 256–508 UTF-8 bytes, and every
Linux filesystem caps a single path component at **255 bytes** (`NAME_MAX`).
7-Zip reports:

```text
ERROR: Can not open output file : File name too long
```

`800 − 24 = 776`, which is exactly the `Train_Fight` count a naive extraction
produces.

**This is not the Windows `MAX_PATH` problem.** `MAX_PATH` limits the *total
path* to 260 characters and is escapable with a `\?\` prefix (that is what
`video_loader.long_path` does, and it affects only 11 files). `NAME_MAX` limits
one *component* to 255 bytes and has **no escape on any Linux filesystem** — the
clips must be written under different names.

### Three failure modes to distinguish

| Symptom | Cause | Fix |
|---|---|---|
| `File name too long`, `Train_Fight` = 776 | 24 names exceed `NAME_MAX` | Re-extract those 24 under safe names |
| `Train_NonFight` < 800, `val/` empty | Multi-volume 7z aborted — a part is missing or the session died | All 13 parts must sit in one directory |
| `FileNotFoundError: .../RWF-2000/train/Train_Fight` | Archive has a top-level `RWF-2000/`, so `-o.../RWF-2000` nests it twice | Use `resolve_dataset_root()`, never a hardcoded path |

### Procedure

**One cell.** `kaggle_bootstrap_cell.py` at the repository root is a generated,
self-contained cell: paste its entire contents into the notebook and run it.
The project modules it needs are embedded as a compressed payload, so it needs
no internet, no `pip install`, and no access to the local source tree. Only
`cv2` and `numpy` are used, both preinstalled on Kaggle. **py7zr is never
used** — a Kaggle session frequently has no route to PyPI.

Regenerate the cell after changing any `src/` module it embeds:

```bash
python tools/build_kaggle_cell.py
```

The embedded module list is computed from the real import graph, so it cannot
drift; `tests/test_kaggle_cell.py` fails if the committed cell is stale.

The cell, in order:

1. Removes a previous **failed** extraction at the old path (never the current
   target, and never a complete dataset).
2. Unpacks the embedded modules and puts them on `sys.path`.
3. Finds a 7-Zip binary, installing `p7zip` only if one is absent.
4. Finds the parts anywhere under `/kaggle/input`, whatever the dataset slug.
5. Verifies all 13 are present and match `ARCHIVE_MD5`, and **refuses to
   extract** otherwise — a missing volume is what truncates an extraction.
6. Reads the archive table of contents and predicts, from UTF-8 byte length,
   which entries cannot be written.
7. Bulk-extracts with `7z x`, tolerating the expected long-name errors.
8. Recovers whatever the bulk pass did **not** land, using 7-Zip alone: all of
   them are requested in a single `7z x -so` call; 7-Zip emits them
   concatenated in archive order, and the table of contents gives each exact
   byte length, so the stream is split back apart and every slice is verified
   against its recorded **CRC32**. Buffering is capped so a pathological case
   cannot exhaust RAM. If verification ever fails, a slower per-entry pass
   takes over.
9. Resolves the real dataset root, reconciling nested `RWF-2000/RWF-2000`.
10. Reconciles disk against the archive table, reporting **missing** and
    **truncated** files separately.
11. Runs the readiness validation and prints one verdict.

Recovery is driven by what is genuinely **absent**, not by the name-length
prediction alone. On a filesystem that happens to accept the long names,
nothing is recovered and no clip is written twice — writing a clip under both
its original and a sanitised name would silently inflate the class counts.

Renaming is safe: labels come from the **directory**, never the filename, and
every recorded leakage exclusion is a short ASCII name in `val/`, so no recorded
path is ever renamed. Every rename is recorded in
`rwf2000_metadata/renamed_clips.json`, written outside the dataset tree.

Re-running is safe: a complete tree is detected and left alone; an incomplete or
truncated one is rebuilt.

### Validation

```python
!python src/prepare_rwf2000_dataset.py --root /kaggle/working/rwf2000 \
    --archives /kaggle/input/<your-dataset-slug> --skip-download
```

then the readiness gate:

```python
import sys; sys.path.insert(0, "src")
from pathlib import Path
from rwf2000_validation import validate_dataset
from prepare_rwf2000_dataset import resolve_dataset_root

root = resolve_dataset_root(Path("/kaggle/working/rwf2000"))
print(validate_dataset(root).to_text())
```

`validate_dataset` checks that both classes exist and are non-empty, that every
clip opens in OpenCV, that frame count / fps / resolution are readable, that
labels collapse to exactly `Fight` and `NonFight`, that there are no cross-split
filename collisions, that confirmed leakage still matches the recorded six
pairs, and that the 400 / 6 / 394 partition holds. It prints `RESULT: READY`
only when all of that passes.
