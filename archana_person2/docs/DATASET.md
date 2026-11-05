# Dataset Documentation

**Author:** Archana Ulloju (Person 2)  
**Project:** Machine Learning for Robust Aquatic Biodiversity Monitoring  
**Last Updated:** 2026-11-05 (Phase 12)

---

## 1. Dataset: Fish4Knowledge

| Attribute | Detail |
|-----------|--------|
| **Name** | Fish4Knowledge Underwater Fish Dataset |
| **Source** | Kaggle — https://www.kaggle.com/datasets/markdaniellampa/fish-dataset |
| **Origin** | Recorded from fixed underwater cameras in the coastal waters of Taiwan |
| **Total images** | ~27,370 labelled JPEG images |
| **Species (classes)** | Multiple aquatic fish species (exact count subject to Person 1's audit) |
| **Image resolution** | Variable; standardised to 224×224 px by preprocessing pipeline |
| **Colour space** | BGR (OpenCV default) → converted to RGB for feature extraction |
| **Annotation** | Each image is labelled with a single species class name |

> [!IMPORTANT]
> **Dataset availability on this machine: NOT PRESENT.**
> The Fish4Knowledge images are not committed to Git (gitignored).
> Person 1 (Sadhvika) is responsible for downloading and preprocessing the dataset.
> All Person 2 test results currently use **synthetic fixtures only**.

---

## 2. Dataset Availability Status

| File / Directory | Purpose | Status |
|-----------------|---------|--------|
| `data/raw/` | Raw downloaded images | ❌ Empty |
| `data/processed/` | Class-separated images (Person 1) | ❌ Empty |
| `data/splits/train.csv` | Official training split manifest | ❌ MISSING |
| `data/splits/val.csv` | Official validation split manifest | ❌ MISSING |
| `data/splits/test.csv` | Official test split manifest | ❌ MISSING |
| `data/preprocessed/X_train.npy` | Preprocessed training arrays | ❌ MISSING |
| `data/preprocessed/y_train.npy` | Training labels | ❌ MISSING |
| `data/preprocessed/label_map.json` | Class name → integer mapping | ❌ MISSING |
| `archana_person2/artifacts/fixtures/synthetic_dataset/` | Synthetic test data (5 classes, 150 images) | ✅ Present |

---

## 3. Split Manifests (Person 1)

Person 1 (Sadhvika) generates official train/val/test split manifests as CSV files.
Each row contains:

| Column | Type | Description |
|--------|------|-------------|
| `image_path` | `str` | Relative path to image file |
| `label` | `str` | Species class name |
| `label_idx` | `int` | Integer label (0-indexed) |
| `video_id` | `str` | Source video identifier |
| `split` | `str` | `"train"` / `"val"` / `"test"` |

> [!NOTE]
> Splits are partitioned **by video_id** to prevent temporal leakage. Frames from
> the same video clip cannot appear in both training and test sets.

---

## 4. Split Sizes (Nominal — Pending Real Data)

Actual split counts depend on Person 1's preprocessing run. Expected proportions:

| Split | Proportion | Expected Count |
|-------|-----------|----------------|
| Train | 70% | ~19,159 images |
| Validation | 15% | ~4,106 images |
| Test | 15% | ~4,106 images |

---

## 5. Synthetic Fixture Dataset (Used in All Current Tests)

When the real Fish4Knowledge dataset is unavailable, Person 2's pipeline falls back
to a deterministic synthetic dataset:

| Attribute | Value |
|-----------|-------|
| **Classes** | 5 (Species_01 … Species_05) |
| **Total samples** | 150 (30 per class) |
| **Train / Val / Test** | 90 / 30 / 30 |
| **Image format** | NumPy random uint8 RGB arrays (64×64 px) |
| **Purpose** | Software correctness testing only |

> [!CAUTION]
> Synthetic dataset results (accuracy, F1, etc.) must **never** be reported as
> real-world model performance in the academic project report.
> Real model performance can only be measured after running the full pipeline with
> the actual Fish4Knowledge images.

---

## 6. Preprocessing Pipeline (Person 1)

Person 1's preprocessing pipeline produces the files consumed by Person 2:

```
1. Download raw images
   python scripts/download_dataset.py --source kaggle --dest data/raw

2. Organise into species folders
   python scripts/organise_dataset.py --src data/raw --dest data/processed

3. Generate split manifests (video-stratified)
   python scripts/split_dataset.py \
       --dataset data/processed \
       --out data/splits \
       --seed 42

4. Preprocess images → NumPy arrays (ImageNet normalisation)
   python scripts/preprocess.py \
       --splits data/splits \
       --dataset data/processed \
       --out data/preprocessed
```

After step 4, the following files are ready for Person 2:

```
data/preprocessed/
├── X_train.npy        ← float32, ImageNet-normalised, shape (N_train, H, W, 3)
├── X_val.npy
├── X_test.npy
├── y_train.npy        ← int64 label indices
├── y_val.npy
├── y_test.npy
└── label_map.json     ← {"Species_Name": int_index, ...}
```

---

## 7. Preprocessing Compatibility Note

> [!WARNING]
> Person 1's `.npy` arrays use **ImageNet mean/std normalisation** (float32).
> Person 2's feature extractors (Color Histogram, HOG) expect **raw uint8 RGB**.
>
> When extracting features from Person 1's preprocessed NumPy arrays, inverse-normalise
> before feature extraction. When predicting directly from image files via
> `predict_image()` or `AquaticPredictor.predict_image_path()`, this is handled
> automatically (OpenCV reads raw pixels → BGR → RGB → uint8 feature extraction).
>
> **Inverse-normalisation formula:**
> ```python
> mean = np.array([0.485, 0.456, 0.406])
> std  = np.array([0.229, 0.224, 0.225])
> img_uint8 = np.clip((img_float * std + mean) * 255, 0, 255).astype(np.uint8)
> ```

---

## 8. Person 2 Label Map (Runtime)

Person 2's pipeline loads the label map from `data/preprocessed/label_map.json`.
If that file is unavailable, the synthetic fixture class names are used instead:

```json
{
  "Species_01": 0,
  "Species_02": 1,
  "Species_03": 2,
  "Species_04": 3,
  "Species_05": 4
}
```

When real data is available, this will reflect actual Fish4Knowledge species names.

---

## 9. How to Trigger Real-Data Tests

After Person 1's preprocessing pipeline has run:

```bash
# 4 previously-skipped tests will now execute automatically
python -m pytest archana_person2/tests/test_validation.py -v -k "RealData"
```

The tests check:
1. `data/preprocessed/X_train.npy` exists and is a valid array.
2. `data/preprocessed/label_map.json` is valid JSON with at least 2 classes.
3. `data/splits/train.csv` has the expected columns.
4. Feature extraction on a real image produces the expected dimensionality.

---

*Dataset documentation by Archana Ulloju (Person 2) — November 2026*
