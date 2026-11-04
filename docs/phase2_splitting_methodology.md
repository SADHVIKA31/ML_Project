# Phase 2 – Leakage-Safe Dataset Splitting: Methodology

> **Branch:** `sadhvika_phase2`
> **Responsibility:** Sadhvika – Blockchain / Security Review

---

## Why This Matters

Suppose two nearly identical frames from the same underwater video appear in
both the **training** and **testing** datasets.

The model may memorise visual patterns from training and simply *recognise*
those patterns in test frames rather than genuinely learning to identify fish
species.  This inflates test accuracy and gives a false sense of model quality.

This phase prevents that problem entirely.

---

## The Core Principle

> **Split by video group, not by individual image.**

Every image that belongs to the same source video or trajectory must land in
the **same** split (train, val, or test).  No group is ever divided across
two splits.

---

## Pipeline Overview

```
data/processed/            (Phase 1 output)
        │
        ▼
extract_video_groups.py    Step 1-2: group images by source video id
        │
        ▼
data/interim/video_groups.json
        │
        ▼
split_dataset.py           Steps 3-4: split GROUPS (not images) → train/val/test
        │
        ▼
data/splits/
  train.csv  |  val.csv  |  test.csv  |  split_summary.json
        │
        ▼
verify_leakage.py          Steps 5-6: 3-layer leakage verification
        │
        ▼
reports/
  leakage_report.json  |  leakage_report.md
```

---

## Step-by-Step Methodology

### Step 1 – Extract Video/Trajectory Groups

**Script:** `scripts/extract_video_groups.py`

The Fish4Knowledge dataset encodes the source video in the filename:

```
fish_image_<video_id>_<frame_id>.png
```

The script uses a regex to extract `video_id` from the filename.
If no video id is found in the filename, it falls back to EXIF metadata.
Images with no detectable trajectory are placed in singleton groups.

**Output:** `data/interim/video_groups.json`
```json
{
  "vid_00001": ["Clownfish/fish_image_00001_0001.png", "Clownfish/fish_image_00001_0002.png"],
  "vid_00002": ["Shark/fish_image_00002_0001.png"],
  ...
}
```

---

### Steps 2–3 – Split by Video Groups

**Script:** `scripts/split_dataset.py`

**Algorithm:**
1. For each class, collect all video group IDs associated with that class.
2. Shuffle the group IDs using a fixed random seed (`--seed 42` by default).
3. Assign groups (not individual images) to splits using the target ratio:
   - **70%** of groups → Training
   - **15%** of groups → Validation
   - **15%** of groups → Testing
4. All images belonging to an assigned group go into that split exclusively.

**Key guarantee:** Because assignment is at the group level, no image from
video X can appear in both train and test.

**Output:**
```
data/splits/
  train.csv           # class, video_id, image_path, filename
  val.csv
  test.csv
  split_summary.json  # per-class counts + actual ratios
```

---

### Steps 4–5 – Verify Leakage

**Script:** `scripts/verify_leakage.py`

Three independent checks are performed:

| # | Check | Method | What it catches |
|---|-------|--------|-----------------|
| 1 | **Video-Group Isolation** | Set intersection of `video_id` values | Any group accidentally split across train/val/test |
| 2 | **Exact Duplicate** | MD5 hash of every image file | Byte-identical frames that appear in multiple splits |
| 3 | **Near-Duplicate** *(optional)* | Perceptual hash (pHash), Hamming distance ≤ 8 | Visually similar frames (slightly different encoding, brightness, crop) |

A clean run produces:
```
✅ Video-Group Isolation : PASS
✅ Exact Duplicates      : PASS
✅ Near-Duplicates       : PASS
```

**Output:**
```
reports/leakage_report.json
reports/leakage_report.md
```

---

## Reproducing the Splits

```bash
# 1. Install Phase 2 dependencies
pip install -r requirements.txt -r requirements_phase2.txt

# 2. Group images by source video
python scripts/extract_video_groups.py \
    --dataset data/processed \
    --out     data/interim/video_groups.json

# 3. Generate leakage-safe splits (CSV manifests)
python scripts/split_dataset.py \
    --dataset data/processed \
    --groups  data/interim/video_groups.json \
    --out     data/splits \
    --train   0.70 --val 0.15 --test 0.15 \
    --seed    42

# 4. Verify no leakage (add --phash for near-duplicate check)
python scripts/verify_leakage.py \
    --splits data/splits \
    --out    reports \
    --phash
```

---

## Assumptions & Limitations

| Assumption | Detail |
|------------|--------|
| Filename encodes video id | True for Fish4Knowledge (`fish_image_VVVVV_FFFF.*`) |
| pHash threshold = 8 | Chosen empirically; may need tuning per dataset |
| Singleton groups | Images with no detectable video id are treated as individual groups and split individually |
| Seed = 42 | Fixed for reproducibility; change with `--seed` |

---

## Deliverables Checklist

- [x] `data/splits/train.csv` – Training split manifest
- [x] `data/splits/val.csv`   – Validation split manifest
- [x] `data/splits/test.csv`  – Testing split manifest
- [x] `data/splits/split_summary.json` – Per-class split statistics
- [x] `scripts/extract_video_groups.py` – Video grouping code
- [x] `scripts/split_dataset.py` – Split generation code
- [x] `scripts/verify_leakage.py` – Leakage verification code
- [x] `reports/leakage_report.md` – Leakage verification results
- [x] `docs/phase2_splitting_methodology.md` – This document

---

*Branch: `sadhvika_phase2` → PR target: `develop`*
