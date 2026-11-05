# Prediction Pipeline — Phase 8

**Person 2: Archana Ulloju**
**Project: Machine Learning for Robust Aquatic Biodiversity Monitoring**

---

## Overview

Phase 8 adds a **reusable, end-to-end prediction pipeline** that:

1. Loads any of the three trained classical ML models (Logistic Regression, Random Forest, SVM).
2. Applies the **exact same preprocessing and feature extraction** used during training.
3. Accepts a new underwater image (file path or NumPy array).
4. Returns the predicted aquatic species name and, where supported, a full probability distribution.

No model retraining is performed — this module is inference-only.

---

## File Structure

```
archana_person2/
├── src/
│   └── predict.py                  ← Reusable AquaticPredictor class
├── scripts/
│   └── predict_image.py            ← CLI script for single-image prediction
├── tests/
│   └── test_prediction.py          ← 20 unit tests (synthetic fixtures only)
└── docs/
    └── PREDICTION.md               ← This document
```

---

## Module: `archana_person2/src/predict.py`

### Class: `AquaticPredictor`

```python
from archana_person2.src.predict import AquaticPredictor

predictor = AquaticPredictor(
    model_name="svm",          # "logistic_regression" | "random_forest" | "svm"
    representation="combined", # "color_hist" | "hog" | "combined"
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"svm"` | Which trained model to load |
| `representation` | `str \| None` | `None` → `"combined"` | Feature representation |
| `model_path` | `str \| None` | Auto-detected | Explicit `.joblib` file path |
| `config_path` | `str \| None` | Auto-detected | Path to `config.yaml` |

#### Methods

| Method | Description |
|--------|-------------|
| `predict_array(image)` | Predict from a NumPy RGB uint8 array |
| `predict_image_path(image_path)` | Predict from an image file on disk |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `feature_dim` | `int` | Extracted feature vector length |
| `class_names` | `List[str]` | Sorted species names known to the model |

#### Return Value (both predict methods)

```python
{
    "predicted_class": "Species_03",     # Most likely species name
    "predicted_index": 2,                # Integer label index
    "confidence": 0.9134,               # Max probability (None if unsupported)
    "probabilities": {                   # Full distribution (None for RF)
        "Species_01": 0.03,
        "Species_02": 0.05,
        "Species_03": 0.91,
        ...
    },
    "feature_vector": np.ndarray,        # Extracted feature vector (shape [D])
    "representation": "combined",        # Feature type used
    "model_name": "svm",                # Model identifier
    "image_path": "/path/to/fish.jpg",  # (predict_image_path only)
}
```

---

## Probability Support

| Model | Confidence | Probabilities |
|-------|-----------|---------------|
| Logistic Regression | ✅ Yes | ✅ Yes |
| SVM (with Platt scaling) | ✅ Yes | ✅ Yes |
| Random Forest | ❌ Not exposed | ❌ Not exposed |

> **Note**: Random Forest has a `predict_proba` method, but Phase 8 deliberately restricts it to Logistic Regression and SVM — the two models where probability calibration is well-defined in the Phase 3 and Phase 5 implementations.

---

## CLI Script: `predict_image.py`

### Basic Usage

```bash
# Using SVM with combined features (default)
python archana_person2/scripts/predict_image.py \
    --image path/to/fish.jpg

# Using Logistic Regression with color histogram
python archana_person2/scripts/predict_image.py \
    --image path/to/fish.jpg \
    --model logistic_regression \
    --representation color_hist

# Output top-5 species and confidence bars
python archana_person2/scripts/predict_image.py \
    --image path/to/fish.jpg \
    --model svm \
    --top-k 5

# Machine-readable JSON output
python archana_person2/scripts/predict_image.py \
    --image path/to/fish.jpg \
    --json
```

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--image` | ✅ Yes | — | Input image path |
| `--model` | No | `svm` | Model to use |
| `--representation` | No | `combined` | Feature type |
| `--model-path` | No | Auto | Explicit `.joblib` path |
| `--config` | No | Auto | Path to `config.yaml` |
| `--top-k` | No | `3` | Number of top species to display |
| `--json` | No | Off | Output JSON instead of human-readable text |

### Example Output

```
────────────────────────────────────────────────────────────
  Image        : /path/to/fish.jpg
  Model        : svm
  Representation: combined
  Feature Dim  : 7108
────────────────────────────────────────────────────────────
  ► Predicted Species : Species_03
  ► Label Index       : 2
  ► Confidence        : 0.9134  (91.3%)

  Top-3 Species:
     1. Species_03                   0.9134  |████████████████    |
     2. Species_07                   0.0531  |█                   |
     3. Species_12                   0.0217  |                    |
────────────────────────────────────────────────────────────
```

---

## Error Handling

| Situation | Exception Raised |
|-----------|-----------------|
| Unknown `model_name` | `ValueError` |
| Unknown `representation` | `ValueError` |
| Model `.joblib` file not found | `FileNotFoundError` |
| Image file not found | `FileNotFoundError` |
| File is not a valid image | `ValueError` |
| `None` passed as image array | `ValueError` |
| Wrong number of channels (e.g. RGBA) | `ValueError` |
| Empty array | `ValueError` |

---

## Supported Feature Representations

| Name | Dimensions | Description |
|------|-----------|-------------|
| `color_hist` | 1,024 | RGB + HSV color histograms (8 bins each channel) |
| `hog` | 6,084 | Histogram of Oriented Gradients (16×16 cells, 9 orientations) |
| `combined` | 7,108 | Concatenation of Color Histogram + HOG |

> **Important**: Always use the same `representation` that was used when the model was trained. All three saved models were trained with `combined` features by default.

---

## Supported Models

| Model Name | File | Phase |
|------------|------|-------|
| `logistic_regression` | `logistic_regression_pipeline.joblib` | Phase 3 |
| `random_forest` | `random_forest_model.joblib` | Phase 4 |
| `svm` | `svm_model.joblib` | Phase 5 |

Default model directory: `archana_person2/artifacts/models/`

---

## Testing

Phase 8 includes **20 unit tests** in `archana_person2/tests/test_prediction.py`.

All tests use **synthetic data only** — no real Fish4Knowledge images are required.

```bash
# Run only Phase 8 tests
python -m pytest archana_person2/tests/test_prediction.py -v

# Run full test suite
python -m pytest archana_person2/tests/ -v
```

### Test Coverage

| Test Group | Tests |
|------------|-------|
| Initialisation (valid/invalid inputs) | 5 |
| `predict_array` correctness | 10 |
| `predict_image_path` correctness | 3 |
| Property accessors | 2 |
| **Total** | **20** |

---

## Data Provenance Note

> All Phase 8 tests use synthetic fixtures only.
> Predictions on real Fish4Knowledge images must be performed using the full dataset pipeline (Phase 1–7) with the official Person 1 split manifests.
> Synthetic results are **not** suitable for the final academic report.

---

## Integration with Other Phases

```
Phase 2  → AquaticFeatureExtractor (feature extraction — reused by predict.py)
Phase 3  → logistic_regression_pipeline.joblib
Phase 4  → random_forest_model.joblib
Phase 5  → svm_model.joblib
Phase 6  → Evaluation framework (separate from inference)
Phase 7  → Model comparison (separate from inference)
Phase 8  → AquaticPredictor (this phase — inference only)
```

---

*Phase 8 implemented by Archana Ulloju (Person 2) — November 2026.*
