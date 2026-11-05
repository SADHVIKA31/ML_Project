# Integration Guide — Phase 9

**Person 2: Archana Ulloju**  
**Project: Machine Learning for Robust Aquatic Biodiversity Monitoring**  
**Branch:** `archana/phase-09-integration`

---

## 1. Overview

Phase 9 integrates Person 2's classical ML components with the team's project
infrastructure. The integration bridge connects:

| Team Member | Role | Key Components |
|-------------|------|----------------|
| **Sadhvika (Person 1)** | Dataset & Preprocessing | `scripts/dataset_loader.py` — canonical DatasetLoader, split manifests (`data/splits/*.csv`), preprocessed NumPy arrays (`data/preprocessed/`), `label_map.json` |
| **Archana (Person 2)** | Classical ML Models | Feature extraction (Color Hist, HOG, Combined), Logistic Regression, Random Forest, SVM, Evaluation, Comparison, Prediction pipeline |

The integration bridge is implemented in **`archana_person2/src/integration.py`**.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PROJECT ROOT (ML_Project/)                       │
│                                                                         │
│  Person 1 (Sadhvika)                  Person 2 (Archana)                │
│  ─────────────────────                ─────────────────────             │
│  scripts/dataset_loader.py            archana_person2/src/              │
│    DatasetLoader                 ◄─── integration.py  (BRIDGE)          │
│    PyTorchFishDataset                   │                               │
│    keras_generator                      │                               │
│                                         ▼                               │
│  data/                                feature_extractor.py              │
│    splits/train.csv                   color_histogram.py                │
│    splits/val.csv          ──────►    hog_features.py                   │
│    splits/test.csv                    dataset_loader.py (Person2)       │
│    preprocessed/X_*.npy               logistic_regression.py            │
│    preprocessed/y_*.npy               random_forest.py                  │
│    preprocessed/label_map.json        svm_classifier.py                 │
│                                       evaluation.py                      │
│                                       comparison.py                      │
│                                       predict.py  (AquaticPredictor)    │
│                                                                         │
│                            archana_person2/artifacts/                   │
│                                models/logistic_regression_pipeline.joblib│
│                                models/random_forest_model.joblib         │
│                                models/svm_model.joblib                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

```
Input Image (JPEG / PNG / BMP / TIFF)
        │
        ▼
  OpenCV BGR read → RGB conversion
        │
        ▼
  AquaticFeatureExtractor.extract_image(image, representation)
        │
        ├── "color_hist" → ColorHistogramExtractor  → 1,024-dim vector
        ├── "hog"        → HOGFeatureExtractor       → 6,084-dim vector
        └── "combined"   → Color Hist + HOG concat   → 7,108-dim vector
        │
        ▼
  reshape to (1, D) float32
        │
        ▼
  sklearn Pipeline / estimator (loaded from .joblib)
    ├── StandardScaler.transform(X)   [fitted on training data]
    └── Classifier.predict(X)
        │
        ├── Logistic Regression → predict_proba() ✅
        ├── SVM (Platt scaling) → predict_proba() ✅
        └── Random Forest       → predict_proba() ❌ (not exposed)
        │
        ▼
  Result dict
    {
      "predicted_class" : "Species_03",
      "predicted_index" : 2,
      "confidence"      : 0.9134,
      "probabilities"   : {"Species_01": 0.03, ...},
      "feature_vector"  : np.ndarray (shape [D]),
      "representation"  : "combined",
      "model_name"      : "svm",
      "image_path"      : "/path/to/fish.jpg"
    }
```

---

## 4. Integration Module API

### `archana_person2/src/integration.py`

#### Top-level functions

| Function | Description |
|----------|-------------|
| `load_classical_predictor(model_name, representation, ...)` | Factory — returns a ready AquaticPredictor |
| `predict_image(image_path, model_name, representation, ...)` | One-call: load + predict from image file |
| `predict_array(image, model_name, representation, ...)` | One-call: load + predict from numpy array |
| `get_dataset_info()` | Cross-member dataset metadata (bridges to Sadhvika's DatasetLoader) |
| `verify_classical_integration(verbose)` | Smoke-test of all integration touch-points |

#### `ClassicalMLPipeline` façade

```python
from archana_person2.src.integration import ClassicalMLPipeline

# Instantiate with desired model
pipeline = ClassicalMLPipeline(model_name="svm", representation="combined")

# Predict from file
result = pipeline.predict("/path/to/fish.jpg")
print(result["predicted_class"])   # e.g. "Species_03"
print(result["confidence"])        # e.g. 0.91

# Predict from numpy array
import numpy as np
image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
result = pipeline.predict_array(image)

# Properties
print(pipeline.model_name)     # "svm"
print(pipeline.feature_dim)    # 7108
print(pipeline.class_names)    # ["Species_01", ...]
```

---

## 5. Model Selection Guide

| Model | File | Feature Dim | Proba Support | Best For |
|-------|------|------------|--------------|----------|
| `logistic_regression` | `logistic_regression_pipeline.joblib` | 7,108 | ✅ | Fast inference, interpretable |
| `random_forest` | `random_forest_model.joblib` | 7,108 | ❌ | Non-linear, feature importance |
| `svm` | `svm_model.joblib` | 7,108 | ✅ | Highest accuracy (Platt scaling) |

**Recommendation**: Use `svm` with `combined` features (default) for best accuracy.

---

## 6. Connecting to Sadhvika's Dataset Pipeline

### Label Map

```python
# Option A: via integration bridge (recommended)
from archana_person2.src.integration import get_dataset_info
info = get_dataset_info()
label_map = info["label_map"]    # {class_name: int}
num_classes = info["num_classes"]

# Option B: directly via Sadhvika's DatasetLoader
from scripts.dataset_loader import DatasetLoader
loader = DatasetLoader()
label_map = loader.get_label_map()
```

### Loading Preprocessed Data for Batch Evaluation

```python
from scripts.dataset_loader import DatasetLoader
from archana_person2.src.feature_extractor import AquaticFeatureExtractor

# Load split manifests (from Person 1's pipeline)
loader = DatasetLoader()
records = loader.load_split_records("test")     # list of dicts

# Extract features (Person 2's pipeline)
extractor = AquaticFeatureExtractor()
X, y = [], []
for record in records:
    img = loader.load_and_preprocess(record["image_path"])  # float32, normalised
    # NOTE: Person 2's extractors expect uint8 RGB — convert accordingly:
    import numpy as np
    img_uint8 = np.clip(
        (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255,
        0, 255,
    ).astype(np.uint8)
    feats = extractor.extract_image(img_uint8, representation="combined")
    X.append(feats)
    y.append(record["label_idx"])
```

> **Note on preprocessing compatibility:** Sadhvika's pipeline normalises pixel
> values to ImageNet mean/std (float32). Person 2's feature extractors (Color
> Histogram, HOG) expect raw uint8 RGB images. When loading from Sadhvika's
> preprocessed NumPy arrays, inverse-normalise before feature extraction.
> When loading directly from image files via `predict_image()` or
> `AquaticPredictor.predict_image_path()`, the internal pipeline handles this
> correctly (OpenCV reads raw BGR → RGB conversion → uint8 feature extraction).

---

## 7. Quick Start

### Predict a single image

```bash
# Using the CLI script
python archana_person2/scripts/predict_image.py \
    --image path/to/fish.jpg \
    --model svm \
    --representation combined

# With JSON output
python archana_person2/scripts/predict_image.py \
    --image path/to/fish.jpg --json
```

### Programmatic prediction

```python
from archana_person2.src.integration import predict_image

result = predict_image(
    image_path="path/to/fish.jpg",
    model_name="svm",
    representation="combined",
)
print(f"Prediction: {result['predicted_class']} (confidence: {result['confidence']:.2%})")
```

### Verify integration health

```bash
python -c "
from archana_person2.src.integration import verify_classical_integration
verify_classical_integration(verbose=True)
"
```

---

## 8. Running Tests

```bash
# Phase 9 integration tests only
python -m pytest archana_person2/tests/test_integration.py -v

# Full test suite (Phases 1–9)
python -m pytest archana_person2/tests/ -v

# Run with environment variables to suppress matplotlib warnings
$env:MPLBACKEND="Agg"; $env:MPLCONFIGDIR="c:\Users\ulloj\ML_Project\.mplcache"; \
python -m pytest archana_person2/tests/ -v
```

---

## 9. Dependencies

### Person 2 Dependencies (required)

| Package | Purpose |
|---------|---------|
| `scikit-learn` | LR, RF, SVM classifiers + StandardScaler |
| `numpy` | Feature arrays |
| `opencv-python` (`cv2`) | Image loading + preprocessing |
| `joblib` | Model serialization |
| `pyyaml` | Config loading |
| `scikit-image` | HOG feature extraction |

### Person 1 Dependencies (required for real data)

| Package | Purpose |
|---------|---------|
| `Pillow` | Image loading in Sadhvika's DatasetLoader |
| `numpy` | NumPy arrays |
| `tqdm` | Progress bars |

### Optional

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | PyTorch Dataset wrapper (Person 1) |
| `tensorflow` | Keras generator (Person 1) |
| `matplotlib` | Plots (Phase 7 comparison) |

---

## 10. Known Limitations

| Limitation | Detail |
|-----------|--------|
| **Real dataset not committed** | Fish4Knowledge images (~27,370) are git-ignored. Models trained on synthetic fixtures are for software testing only. |
| **Preprocessing incompatibility** | Sadhvika's arrays are ImageNet-normalised float32; Person 2's extractors need uint8 RGB. Inverse-normalise when going from `.npy` arrays to feature extraction. |
| **Random Forest probabilities** | RF does not expose a calibrated probability distribution in Phase 8–9. Use LR or SVM for confidence scores. |
| **No deep learning integration** | No DL members (Pragnya, Nithya, Archan) have committed code to `develop` as of Phase 9. Integration with DL models deferred to Phase 10+. |
| **No REST API** | No Flask/FastAPI wrapper exists yet. CLI script (`predict_image.py`) is the current user-facing interface. |
| **No frontend** | No web UI exists yet. |

---

## 11. File Structure (Phase 9 additions)

```
archana_person2/
├── src/
│   └── integration.py          ← NEW: integration bridge module
├── tests/
│   └── test_integration.py     ← NEW: 33 integration tests
└── docs/
    └── INTEGRATION.md          ← NEW: this document
```

---

## 12. Next Steps (Phase 10)

- Integrate with deep learning members' models when available on `develop`.
- Create a unified REST API (`scripts/api.py` or `app.py`) exposing both
  classical ML and DL predictions under a common interface.
- Add a cross-member evaluation notebook comparing classical vs. DL accuracy.
- Extend `verify_classical_integration()` to include DL model checks.

---

*Phase 9 integration implemented by Archana Ulloju (Person 2) — November 2026.*
