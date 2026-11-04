# Dataset Handover Package – Phase 5

> **Author:** Sadhvika (Dataset & Preprocessing Lead)
> **Branch:** `sadhvika_phase5`
> **Status:** ✅ Ready for team consumption
> **Date:** 2026-11-05

---

## 🎯 Purpose

This document is the **official dataset handover** from the Dataset &
Preprocessing team (Sadhvika) to all other team members.

After reading this document you will be able to:
- Load the dataset into your training/evaluation code with one import
- Understand the split strategy and why it matters
- Understand the preprocessing pipeline and label encoding
- Verify your environment is correctly set up

---

## 📦 What Has Been Prepared (Phases 1–4)

| Phase | Deliverable | Location |
|-------|-------------|----------|
| Phase 1 | Downloaded + organised dataset | `data/processed/<ClassName>/` |
| Phase 2 | Leakage-safe train/val/test splits | `data/splits/train.csv`, `val.csv`, `test.csv` |
| Phase 3 | Preprocessed NumPy arrays + label map | `data/preprocessed/` |
| Phase 4 | Full EDA reports and observations | `reports/eda/` |

---

## 📂 Dataset Structure

```
data/
├── processed/                   ← Phase 1: class-separated images
│   ├── <ClassName_1>/
│   │   ├── fish_image_00001_0001.jpg
│   │   └── ...
│   ├── <ClassName_2>/
│   └── ...
│
├── splits/                      ← Phase 2: leakage-safe manifests
│   ├── train.csv                   columns: class, video_id, image_path, filename
│   ├── val.csv
│   ├── test.csv
│   └── split_summary.json
│
└── preprocessed/                ← Phase 3: ready-to-train arrays
    ├── X_train.npy              shape: (N, 224, 224, 3)  dtype: float32
    ├── y_train.npy              shape: (N,)              dtype: int32
    ├── X_val.npy
    ├── y_val.npy
    ├── X_test.npy
    ├── y_test.npy
    ├── label_map.json           {class_name: integer_index}
    └── split_stats.json
```

---

## 🏷️ Label Mapping

Labels are integer-encoded in **alphabetical order** of the class name.

```json
{
  "ClassName_A": 0,
  "ClassName_B": 1,
  "ClassName_C": 2,
  ...
}
```

The authoritative file is: `data/preprocessed/label_map.json`

> ⚠️ **Never hardcode class indices.** Always load from `label_map.json`.

---

## 🔄 Split Ratios

| Split | Ratio | Guarantee |
|-------|-------|-----------|
| Train | 70% | No video-group overlap with val/test |
| Val   | 15% | No video-group overlap with train/test |
| Test  | 15% | No video-group overlap with train/val |

**Key guarantee:** Images from the same source video are **never** split
across train and test. This prevents temporal/near-duplicate leakage.

**Seed:** 42 (fixed for full reproducibility)

---

## ⚙️ Preprocessing Pipeline (Phase 3)

Every image has been processed identically across all three splits:

| Step | Operation | Detail |
|------|-----------|--------|
| 1 | Load | Pillow + OpenCV |
| 2 | Validate | Skip corrupted / unreadable |
| 3 | Resize | 224 × 224 px (LANCZOS) |
| 4 | Convert | All modes → RGB |
| 5 | Normalise | ImageNet mean/std z-score |

**ImageNet normalisation:**
```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
pixel_normalised = (pixel / 255 - mean) / std
```

> ✅ This is the standard for ImageNet pretrained models (ResNet, EfficientNet, ViT…)

---

## 🚀 Loading the Dataset (Quick Start)

### Option A — NumPy arrays (fastest, recommended)

```python
from scripts.dataset_loader import DatasetLoader

loader   = DatasetLoader()
X_train, y_train = loader.load_numpy("train")
X_val,   y_val   = loader.load_numpy("val")
X_test,  y_test  = loader.load_numpy("test")
label_map        = loader.get_label_map()   # {class_name: int}
num_classes      = loader.num_classes()     # e.g. 23
```

### Option B — PyTorch Dataset

```python
from scripts.dataset_loader import PyTorchFishDataset
from torch.utils.data import DataLoader

train_ds = PyTorchFishDataset("train")
val_ds   = PyTorchFishDataset("val")
test_ds  = PyTorchFishDataset("test")

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)
test_dl  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4)

for images, labels in train_dl:
    # images: (B, 3, H, W) float32  — PyTorch channel-first convention
    # labels: (B,)          int64
    pass
```

### Option C — Keras / TensorFlow generator

```python
from scripts.dataset_loader import keras_generator

train_gen = keras_generator("train", batch_size=32, shuffle=True)
val_gen   = keras_generator("val",   batch_size=32, shuffle=False)

model.fit(
    train_gen,
    steps_per_epoch = len(train_records) // 32,
    validation_data = val_gen,
    validation_steps= len(val_records) // 32,
    epochs = 50,
)
```

### Lazy loading (memory-efficient)

```python
loader  = DatasetLoader()
records = loader.load_split_records("train")  # list of dicts

for record in records:
    img_array = loader.load_and_preprocess(record["image_path"])
    label     = record["label_idx"]
```

---

## ✅ Verifying Your Setup

Before starting model training, run the integration check:

```bash
python scripts/verify_integration.py

# With PyTorch check:
python scripts/verify_integration.py --pytorch

# With Keras check:
python scripts/verify_integration.py --keras
```

Expected output:
```
✅  Directories present
✅  Split CSV manifests readable
✅  Preprocessed arrays present
✅  Label map consistent
✅  DatasetLoader end-to-end
→ INTEGRATION VERIFIED ✅  Dataset is ready for model training.
```

---

## 📋 Per-Team-Member Quick Reference

### Archan (Frontend / Model UI)
- Use `DatasetLoader.get_label_map()` to get `{class_name: index}` for
  display labels in the UI.
- The class names come directly from `data/preprocessed/label_map.json`.

### Pragnya (Backend / Training Pipeline)
- Load data with `DatasetLoader.load_numpy()` or `PyTorchFishDataset`.
- Use **X_train / y_train** for training, **X_val / y_val** for validation
  callback, **X_test / y_test** for final evaluation only.
- **Never use test data during training or hyperparameter tuning.**
- Class weights for imbalance (see EDA report):
  ```python
  from sklearn.utils.class_weight import compute_class_weight
  import numpy as np
  _, y_train = loader.load_numpy("train")
  weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
  class_weights = dict(enumerate(weights))
  ```

### Nithya (Testing / QA)
- Use `loader.load_numpy("test")` for all final evaluations.
- **The test split must NEVER be touched until the model is finalised.**
- Run `verify_integration.py` to confirm environment before every test run.
- Report any dataset-related issues to Sadhvika on the `sadhvika_phase5` branch.

---

## 🐛 Raising Dataset Issues

If you encounter any dataset-related problem during model development:

1. Check `reports/eda/eda_observations.md` — it may already be documented.
2. Open a GitHub Issue with label `dataset` and tag `@SADHVIKA31`.
3. Or raise it directly on this PR / in the team channel.

Common issues and fixes:

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: X_train.npy` | Run `python scripts/preprocess.py` |
| `FileNotFoundError: train.csv` | Run `python scripts/split_dataset.py` |
| `KeyError: class not in label_map` | Re-run `preprocess.py` after adding new classes |
| Shape mismatch in model | Check `loader.target_size` matches model input (default 224×224) |
| Wrong pixel range | Confirm `normalize="imagenet"` in DatasetLoader |

---

## 📖 Related Documentation

| Document | Location |
|----------|----------|
| Phase 1 methodology | `docs/` *(in Phase 1 PR)* |
| Phase 2 splitting methodology | [`docs/phase2_splitting_methodology.md`](../docs/phase2_splitting_methodology.md) |
| Phase 3 preprocessing methodology | [`docs/phase3_preprocessing_methodology.md`](../docs/phase3_preprocessing_methodology.md) |
| Phase 4 EDA methodology | [`docs/phase4_eda_methodology.md`](../docs/phase4_eda_methodology.md) |
| EDA observations & conclusions | [`reports/eda/eda_observations.md`](../reports/eda/eda_observations.md) |

---

*Handover complete. Branch: `sadhvika_phase5` → PR target: `develop`*
