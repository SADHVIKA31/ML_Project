# Phase 3 – Data Preprocessing: Methodology

> **Branch:** `sadhvika_phase3`
> **Responsibility:** Sadhvika – Dataset & Preprocessing Lead

---

## Pipeline Overview

```
data/processed/                 ← Phase 1 output (class-separated images)
data/splits/*.csv               ← Phase 2 output (leakage-safe manifests)
        │
        ▼
image_inspector.py              Steps 1-2:  load + audit dimensions & formats
        │
        ▼
data/interim/image_inspection.{csv,json}
        │
        ▼
preprocess.py                   Steps 3-9:  full preprocessing pipeline
        │
        ▼
data/preprocessed/
  X_train.npy  y_train.npy
  X_val.npy    y_val.npy
  X_test.npy   y_test.npy
  label_map.json
  split_stats.json
        │
        ▼
visualise_preprocessing.py      Deliverable: before/after figures
        │
        ▼
reports/before_after/
  sample_grid.png
  <class>/img_before_after.png
        │
        ▼
verify_preprocessing.py         Step 9 check: shape, range, labels, coverage
        │
        ▼
reports/preprocessing_verification.{json,md}
```

---

## Step-by-Step Methodology

### Steps 1–2  Image Loading & Inspection (`image_inspector.py`)

| Item | Detail |
|------|--------|
| Library | Pillow (primary) + OpenCV (cross-check) |
| Output | `data/interim/image_inspection.csv` – per-image audit |
| Detects | Width/height, colour mode, channel count, unreadable files |

The script reports all unique dimensions found in the dataset, which
guides the choice of target resize dimension.

---

### Step 3  Handling Corrupted / Unreadable Images (`preprocess.py`)

- Each image is opened with `Image.open()` followed by `img.verify()` for
  an integrity check before the full decode (`img.load()`).
- Any image that raises an exception is **skipped and logged** – it is
  never silently included in the preprocessed output.
- A count of skipped images per split is recorded in `split_stats.json`.

---

### Step 4  Resizing (`preprocess.py`)

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Target size | 224 × 224 | Standard input for ImageNet-pretrained CNNs (ResNet, VGG, MobileNet, EfficientNet) |
| Resampling | LANCZOS | Highest-quality downsampling, preserves fine texture |

Override with `--size 299` for Inception-family models.

---

### Step 5  Colour Format Conversion (`preprocess.py`)

All images are converted to **RGB** before any further processing:

| Input mode | Action |
|------------|--------|
| `RGB`      | No change |
| `L` (grayscale) | Replicate channel → RGB |
| `RGBA`     | Discard alpha channel → RGB |
| `P` (palette) | Convert via `img.convert("RGB")` |

---

### Step 6  Pixel Normalisation (`preprocess.py`)

Three strategies are supported (select with `--normalize`):

| Strategy | Formula | Use case |
|----------|---------|----------|
| `minmax`   | `pixel / 255.0` → [0, 1] | Custom CNNs, training from scratch |
| `imagenet` | `(pixel/255 - μ) / σ`  where μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225] | Transfer learning from ImageNet pretrained models |
| `none`     | raw float32 | Debugging / custom pipelines |

**Default: `imagenet`** (recommended for transfer learning on Fish4Knowledge).

---

### Steps 7–8  Label Preparation & Encoding (`preprocess.py`)

1. **Class discovery** – class names are derived from subdirectory names in
   `data/processed/`.
2. **Sorted alphabetically** – gives a deterministic, reproducible mapping.
3. **Integer encoding** – `{class_name: index}` stored in `label_map.json`.

```json
{
  "Clownfish":    0,
  "Grouper":      1,
  "Seahorse":     2,
  ...
}
```

The same `label_map.json` is used for all three splits, guaranteeing
consistent encoding.

---

### Step 9  Consistency Verification (`verify_preprocessing.py`)

Four automated checks confirm that the same pipeline was applied uniformly:

| Check | What it verifies |
|-------|-----------------|
| **Stacked arrays** | `X_{split}.npy` and `y_{split}.npy` exist for all splits |
| **Shape consistency** | Every array has shape `(N, H, W, 3)` matching target size |
| **Pixel-value range** | Values fall within the expected range for the chosen normalisation |
| **Class coverage** | Every training class also appears in val and test |

---

## Reproducing the Preprocessing

```bash
# 0. Install dependencies
pip install -r requirements.txt -r requirements_phase2.txt -r requirements_phase3.txt

# 1. Inspect image dimensions and formats
python scripts/image_inspector.py --dataset data/processed --out data/interim

# 2. Run full preprocessing pipeline
python scripts/preprocess.py \
    --splits    data/splits \
    --dataset   data/processed \
    --out       data/preprocessed \
    --size      224 \
    --normalize imagenet

# 3. Generate before/after visualisations
python scripts/visualise_preprocessing.py \
    --dataset   data/processed \
    --out       reports/before_after \
    --samples   3

# 4. Verify consistency across splits
python scripts/verify_preprocessing.py \
    --preprocessed data/preprocessed \
    --out          reports
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Resize to 224×224 | Standard for ImageNet CNNs; small enough for fast training, large enough for fine detail |
| ImageNet normalisation | Enables transfer learning; most pretrained weights expect this |
| Stacked `.npy` arrays | Fast loading with `np.load(mmap_mode="r")`; no per-file overhead |
| Deterministic label map | Alphabetical sort → same encoding every run, no hidden state |
| Skipping corrupted files | Prevents training crashes; corrupted images are logged not silently dropped |

---

## Deliverables Checklist

- [x] `scripts/image_inspector.py`         – Steps 1–2  image loading & format audit
- [x] `scripts/preprocess.py`              – Steps 3–9  full preprocessing pipeline
- [x] `scripts/visualise_preprocessing.py` – before/after sample figures
- [x] `scripts/verify_preprocessing.py`    – Step 9 consistency verification
- [x] `data/preprocessed/label_map.json`   – species → integer label mapping
- [x] `data/preprocessed/X_{split}.npy`    – preprocessed image arrays
- [x] `data/preprocessed/y_{split}.npy`    – integer label arrays
- [x] `reports/before_after/sample_grid.png` – sample before/after figures
- [x] `docs/phase3_preprocessing_methodology.md` – this document

---

*Branch: `sadhvika_phase3` → PR target: `develop`*
