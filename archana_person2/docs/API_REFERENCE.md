# API Reference — Person 2 Classical ML Pipeline

**Author:** Archana Ulloju (Person 2)  
**Project:** Machine Learning for Robust Aquatic Biodiversity Monitoring  
**Package root:** `archana_person2.src`  
**Last Updated:** 2026-11-05 (Phase 12)

---

## Quick Import

```python
# Integration bridge — recommended entry point
from archana_person2.src.integration import (
    ClassicalMLPipeline,
    load_classical_predictor,
    predict_image,
    predict_array,
    get_dataset_info,
    verify_classical_integration,
)

# Core classes
from archana_person2.src.predict import AquaticPredictor
from archana_person2.src.feature_extractor import AquaticFeatureExtractor
from archana_person2.src.evaluation import CommonModelEvaluator
from archana_person2.src.comparison import ModelComparator

# Models
from archana_person2.src.logistic_regression import AquaticLogisticRegression
from archana_person2.src.random_forest import AquaticRandomForest
from archana_person2.src.svm_classifier import AquaticSVMClassifier

# Feature extractors
from archana_person2.src.color_histogram import ColorHistogramExtractor
from archana_person2.src.hog_features import HOGFeatureExtractor

# Dataset loader
from archana_person2.src.dataset_loader import Person2DatasetLoader
```

---

## Module: `integration.py`

The recommended public interface. All functions are importable at the package root.

---

### `ClassicalMLPipeline`

High-level façade combining feature extraction + model loading + prediction.

```python
class ClassicalMLPipeline:
    def __init__(
        self,
        model_name: str = "svm",
        representation: str = "combined",
        model_path: str | None = None,
        config_path: str | None = None,
    ): ...
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"svm"` | `"logistic_regression"`, `"random_forest"`, or `"svm"` |
| `representation` | `str` | `"combined"` | `"color_hist"`, `"hog"`, or `"combined"` |
| `model_path` | `str\|None` | Auto | Explicit `.joblib` file path; auto-detected if `None` |
| `config_path` | `str\|None` | Auto | Path to `config.yaml`; auto-detected if `None` |

**Methods**

| Method | Signature | Returns |
|--------|-----------|---------|
| `predict` | `(image_path: str) -> dict` | Prediction result dict |
| `predict_array` | `(image: np.ndarray) -> dict` | Prediction result dict |

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `model_name` | `str` | Active model name |
| `feature_dim` | `int` | Feature vector length (1024 / 6084 / 7108) |
| `class_names` | `list[str]` | Sorted list of species known to the model |

**Example**

```python
pipeline = ClassicalMLPipeline(model_name="svm", representation="combined")
result = pipeline.predict("path/to/fish.jpg")
print(result["predicted_class"])   # "Dascyllus_aruanus"
print(f"{result['confidence']:.1%}")  # "91.3%"
```

---

### `predict_image()`

One-shot function: load model + extract features + predict from image file.

```python
def predict_image(
    image_path: str,
    model_name: str = "svm",
    representation: str = "combined",
    model_path: str | None = None,
    config_path: str | None = None,
) -> dict: ...
```

**Returns:** prediction result dict (see [Result Dict](#prediction-result-dict) below).

---

### `predict_array()`

One-shot function: load model + extract features + predict from NumPy array.

```python
def predict_array(
    image: np.ndarray,
    model_name: str = "svm",
    representation: str = "combined",
    model_path: str | None = None,
    config_path: str | None = None,
) -> dict: ...
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `np.ndarray` | uint8 RGB array, shape `(H, W, 3)` |

---

### `load_classical_predictor()`

Factory — returns a configured `AquaticPredictor` instance.

```python
def load_classical_predictor(
    model_name: str = "svm",
    representation: str = "combined",
    model_path: str | None = None,
    config_path: str | None = None,
) -> AquaticPredictor: ...
```

---

### `get_dataset_info()`

Returns cross-member dataset metadata by bridging to Sadhvika's `DatasetLoader`.

```python
def get_dataset_info() -> dict: ...
```

**Returns**

```python
{
  "num_classes":  5,                          # int
  "class_names":  ["Species_01", ...],        # list[str]
  "label_map":    {"Species_01": 0, ...},     # dict[str, int]
  "split_counts": {"train": 90, "val": 30, "test": 30},
  "source":       "synthetic" | "real"
}
```

---

### `verify_classical_integration()`

Smoke-test of all integration touch-points. Returns `True` if all pass.

```python
def verify_classical_integration(verbose: bool = False) -> bool: ...
```

---

## Module: `predict.py`

---

### `AquaticPredictor`

Low-level inference class. Loads one model and performs predictions.

```python
class AquaticPredictor:
    def __init__(
        self,
        model_name: str = "svm",
        representation: str | None = None,   # None → "combined"
        model_path: str | None = None,
        config_path: str | None = None,
    ): ...
```

**Methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `predict_array` | `(image: np.ndarray) -> dict` | Predict from uint8 RGB array |
| `predict_image_path` | `(image_path: str) -> dict` | Predict from image file on disk |

**Properties:** `feature_dim: int`, `class_names: list[str]`

---

## Prediction Result Dict

Both `predict_array` and `predict_image_path` return this structure:

```python
{
  "predicted_class": "Species_03",  # str — most likely species name
  "predicted_index": 2,             # int — integer label index
  "confidence":      0.9134,        # float | None — max(probabilities), None for RF
  "probabilities": {                # dict[str, float] | None — full distribution
      "Species_01": 0.03,           #   None for Random Forest
      "Species_02": 0.05,
      "Species_03": 0.9134,
      ...
  },
  "feature_vector":  np.ndarray,   # shape [D] float32
  "representation":  "combined",   # str
  "model_name":      "svm",        # str
  "image_path":      "/path/...",  # str (predict_image_path only)
}
```

**Probability support by model:**

| Model | `confidence` | `probabilities` |
|-------|-------------|-----------------|
| Logistic Regression | ✅ float | ✅ dict |
| SVM (CalibratedClassifierCV) | ✅ float | ✅ dict |
| Random Forest | `None` | `None` |

---

## Module: `feature_extractor.py`

---

### `AquaticFeatureExtractor`

Unified feature extraction across all three representations.

```python
class AquaticFeatureExtractor:
    def extract_image(
        self,
        image: np.ndarray,          # uint8 RGB, shape (H, W, 3)
        representation: str = "combined",
    ) -> np.ndarray: ...            # float32, shape (D,)

    def extract_batch(
        self,
        images: list[np.ndarray],
        representation: str = "combined",
    ) -> np.ndarray: ...            # float32, shape (N, D)
```

**Feature dimensions:**

| `representation` | Dimension |
|-----------------|-----------|
| `"color_hist"` | 1,024 |
| `"hog"` | 6,084 |
| `"combined"` | 7,108 |

---

## Module: `color_histogram.py`

### `ColorHistogramExtractor`

```python
class ColorHistogramExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray: ...
    # Returns float32 1D array of length 1024
    # RGB histogram: 8×8×8 bins, L1-normalised (512 values)
    # HSV histogram: 8×8×8 bins, L1-normalised (512 values)
    # Concatenated: [RGB_hist | HSV_hist]
```

---

## Module: `hog_features.py`

### `HOGFeatureExtractor`

```python
class HOGFeatureExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray: ...
    # Returns float32 1D array of length 6084
    # Resize to 224×224 → grayscale → skimage.feature.hog
    # orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), block_norm="L2-Hys"
```

---

## Module: `evaluation.py`

### `CommonModelEvaluator`

```python
class CommonModelEvaluator:
    def __init__(self, class_names: list[str] | None = None): ...

    def evaluate_predictions(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
    ) -> dict: ...

    def evaluate_model(
        self,
        model,              # fitted sklearn estimator with .predict()
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict: ...
```

**Return dict keys:** `"metrics"`, `"confusion_matrix"`, `"classification_report"`

---

## Module: `comparison.py`

### `ModelComparator`

```python
class ModelComparator:
    def compare_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        representations: list[str] | None = None,  # default: all three
    ) -> dict: ...

    def plot_comparison(
        self,
        results: dict,
        output_dir: str = "archana_person2/artifacts/plots/",
    ) -> None: ...

    def save_report(
        self,
        results: dict,
        output_path: str,
    ) -> None: ...
```

---

## Module: `dataset_loader.py`

### `Person2DatasetLoader`

```python
class Person2DatasetLoader:
    def load_split(
        self,
        split: str,              # "train" | "val" | "test"
        use_synthetic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...   # (images, labels)

    def get_label_map(self) -> dict[str, int]: ...
    def get_class_names(self) -> list[str]: ...
    def get_num_classes(self) -> int: ...
```

---

## Error Catalogue

| Exception | Raised by | Cause |
|-----------|-----------|-------|
| `ValueError` | `AquaticPredictor.__init__` | Unknown `model_name` or `representation` |
| `FileNotFoundError` | `AquaticPredictor.__init__` | `.joblib` model file not found |
| `FileNotFoundError` | `predict_image_path` | Image file does not exist |
| `ValueError` | `predict_image_path` | File is not a readable image |
| `ValueError` | `predict_array` | `None`, wrong channels (not 3), or empty array |
| `AttributeError` | package `__getattr__` | Symbol not in public API |

---

## Package Metadata

```python
import archana_person2.src as pkg

pkg.__version__          # "1.9.0"
pkg.__author__           # "Archana Ulloju"
pkg.__role__             # "Person 2 -- Classical ML and Feature Extraction Lead"
pkg.__phases_complete__  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

---

*API reference by Archana Ulloju (Person 2) — November 2026*
