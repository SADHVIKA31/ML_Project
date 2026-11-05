# Machine Learning for Robust Aquatic Biodiversity Monitoring

> **Team project — Bachelor of Technology (B.Tech)**  
> Dataset: Fish4Knowledge (~27,370 underwater images, multiple aquatic species)  
> Repository: https://github.com/SADHVIKA31/ML_Project.git  
> Active branch: `develop` — **never merge directly into `main`**

---

## Project Overview

This project builds a complete, end-to-end machine learning system to automatically
identify aquatic species from underwater camera footage.  The system combines classical
handcrafted-feature models (Member 2) with dataset preprocessing and deep-learning
components contributed by other team members.

**Primary application:** Automated fish-species recognition for ecological monitoring,
conservation assessment, and marine biodiversity research.

---

## Team Members and Responsibilities

| Member | Role | Branch prefix |
|--------|------|---------------|
| **Sadhvika (Person 1)** | Dataset collection, preprocessing, train/val/test split generation | `sadhvika_phase*` |
| **Archana Ulloju (Person 2)** | Classical ML models, feature extraction, evaluation, comparison, prediction pipeline, integration | `archana/phase-*` |

---

## Project Objectives

1. Load and understand the Fish4Knowledge aquatic dataset.
2. Extract robust handcrafted feature representations (Color Histogram, HOG, Combined).
3. Train and evaluate three classical ML classifiers (Logistic Regression, Random Forest, SVM).
4. Build a reusable, leakage-safe evaluation and comparison framework.
5. Provide a reusable single-image prediction pipeline.
6. Integrate all Member 2 components through a single public API bridge.
7. Validate correctness through comprehensive automated testing.
8. Document the full system for academic reporting.

---

## Repository Structure

```
ML_Project/
├── data/
│   ├── raw/                    ← Downloaded raw Fish4Knowledge files
│   ├── interim/                ← Intermediate processing artefacts
│   ├── processed/              ← Class-separated images (Person 1)
│   ├── splits/                 ← train.csv / val.csv / test.csv (Person 1)
│   └── preprocessed/           ← X_train.npy, y_train.npy, label_map.json (Person 1)
│
├── scripts/                    ← Shared / Person 1 scripts
│   ├── dataset_loader.py       ← Canonical DatasetLoader (Person 1)
│   ├── download_dataset.py
│   ├── organise_dataset.py
│   ├── split_dataset.py
│   └── preprocess.py
│
├── archana_person2/            ← Person 2 (Archana) — Classical ML pipeline
│   ├── src/                    ← All importable source modules
│   │   ├── __init__.py         ← Public API + lazy imports
│   │   ├── dataset_loader.py   ← Person2DatasetLoader
│   │   ├── color_histogram.py  ← ColorHistogramExtractor
│   │   ├── hog_features.py     ← HOGFeatureExtractor
│   │   ├── feature_extractor.py← AquaticFeatureExtractor (combined)
│   │   ├── logistic_regression.py ← AquaticLogisticRegression
│   │   ├── random_forest.py    ← AquaticRandomForest
│   │   ├── svm_classifier.py   ← AquaticSVMClassifier
│   │   ├── evaluation.py       ← CommonModelEvaluator
│   │   ├── comparison.py       ← ModelComparator
│   │   ├── predict.py          ← AquaticPredictor (inference pipeline)
│   │   └── integration.py      ← ClassicalMLPipeline (public API bridge)
│   ├── scripts/                ← CLI entry-points
│   │   ├── predict_image.py    ← Single-image prediction CLI
│   │   ├── train_logistic_regression.py
│   │   ├── train_random_forest.py
│   │   ├── train_svm.py
│   │   ├── evaluate_models.py
│   │   ├── run_model_comparison.py
│   │   ├── run_feature_extraction.py
│   │   └── inspect_dataset.py
│   ├── tests/                  ← Automated test suite (164 tests)
│   │   ├── test_comparison.py
│   │   ├── test_evaluation.py
│   │   ├── test_feature_extraction.py
│   │   ├── test_integration.py
│   │   ├── test_models.py
│   │   ├── test_prediction.py
│   │   └── test_validation.py
│   ├── docs/                   ← Phase-level documentation
│   │   ├── ARCHITECTURE.md
│   │   ├── DATASET.md
│   │   ├── MODELS.md
│   │   ├── API_REFERENCE.md
│   │   ├── INSTALLATION.md
│   │   ├── FEATURE_EXTRACTION.md
│   │   ├── LOGISTIC_REGRESSION.md
│   │   ├── RANDOM_FOREST.md
│   │   ├── SVM.md
│   │   ├── EVALUATION.md
│   │   ├── MODEL_COMPARISON.md
│   │   ├── PREDICTION.md
│   │   ├── INTEGRATION.md
│   │   └── VALIDATION_REPORT.md
│   ├── artifacts/
│   │   ├── features/           ← Cached .npz feature arrays
│   │   ├── models/             ← Serialised .joblib model files (git-ignored)
│   │   ├── metrics/            ← JSON metric reports
│   │   ├── plots/              ← Confusion matrices, bar charts
│   │   └── fixtures/           ← Synthetic test datasets
│   └── config/
│       └── config.yaml         ← Centralised configuration
│
├── docs/                       ← Shared / Person 1 methodology docs
├── reports/                    ← EDA reports (Person 1)
├── notebooks/                  ← Exploratory notebooks
├── requirements.txt            ← Shared dependencies
├── requirements_person2.txt    ← Person 2 dependencies
└── README.md                   ← This file
```

---

## Quick Start — Person 2 (Classical ML Pipeline)

### 1. Install dependencies

```bash
# Person 2 dependencies only
pip install -r requirements_person2.txt

# Full project dependencies
pip install -r requirements.txt
```

### 2. Predict a single underwater image

```bash
# Default: SVM model with combined features
python archana_person2/scripts/predict_image.py --image path/to/fish.jpg

# Choose model and feature type
python archana_person2/scripts/predict_image.py \
    --image path/to/fish.jpg \
    --model logistic_regression \
    --representation color_hist

# JSON output
python archana_person2/scripts/predict_image.py --image path/to/fish.jpg --json
```

### 3. Run the full test suite

```powershell
# Windows PowerShell
$env:PYTHONDONTWRITEBYTECODE="1"
$env:MPLBACKEND="Agg"
$env:MPLCONFIGDIR="c:\Users\ulloj\ML_Project\.mplcache"
python -m pytest archana_person2/tests/ -v --tb=short
```

Expected: **160 passed, 4 skipped, 0 failed**  
(4 skipped = real Fish4Knowledge data not yet available on this machine)

### 4. Programmatic API

```python
from archana_person2.src.integration import predict_image, ClassicalMLPipeline

# One-shot prediction
result = predict_image("path/to/fish.jpg", model_name="svm", representation="combined")
print(result["predicted_class"])   # e.g. "Dascyllus_aruanus"
print(f"{result['confidence']:.1%}")

# Reusable pipeline object
pipeline = ClassicalMLPipeline(model_name="svm", representation="combined")
result = pipeline.predict("path/to/fish.jpg")
```

### 5. Train models (requires real data from Person 1)

```bash
python archana_person2/scripts/train_logistic_regression.py
python archana_person2/scripts/train_random_forest.py
python archana_person2/scripts/train_svm.py
```

### 6. Run evaluation and comparison

```bash
python archana_person2/scripts/evaluate_models.py
python archana_person2/scripts/run_model_comparison.py
```

---

## Person 1 — Dataset Pipeline Quick Start

```bash
# Download dataset (requires Kaggle API key at ~/.kaggle/kaggle.json)
python scripts/download_dataset.py --source kaggle --dest data/raw

# Organise into class folders
python scripts/organise_dataset.py --src data/raw --dest data/processed

# Generate train/val/test split manifests
python scripts/split_dataset.py --dataset data/processed --out data/splits --seed 42

# Preprocess images → NumPy arrays
python scripts/preprocess.py --splits data/splits --dataset data/processed --out data/preprocessed
```

Dataset reference: [Fish4Knowledge on Kaggle](https://www.kaggle.com/datasets/markdaniellampa/fish-dataset)

---

## Phase 11 Test Results (Latest — 2026-11-05)

> [!IMPORTANT]
> All test results below are from **synthetic fixtures only**.  
> Real Fish4Knowledge validation is **PENDING** — the dataset is not yet available on this machine.  
> Do not use these numbers as real-world model accuracy in academic reports.

| Test File | Collected | Passed | Skipped | Failed |
|-----------|-----------|--------|---------|--------|
| `test_comparison.py` | 4 | 4 | 0 | 0 |
| `test_evaluation.py` | 15 | 15 | 0 | 0 |
| `test_feature_extraction.py` | 8 | 8 | 0 | 0 |
| `test_integration.py` | 33 | 33 | 0 | 0 |
| `test_models.py` | 14 | 14 | 0 | 0 |
| `test_prediction.py` | 20 | 20 | 0 | 0 |
| `test_validation.py` | 70 | 66 | 4 | 0 |
| **TOTAL** | **164** | **160** | **4** | **0** |

Scikit-learn version: 1.9.1 · Python 3.14.4 · pytest 9.2.0 · Runtime ~6 s

---

## Branch Workflow

```
feature branch  ──PR──►  develop  ──PR──►  main
                                   (never directly)
```

- All work must be committed on a feature branch and merged into `develop`.
- **Never push directly to `main`.**
- Current develop HEAD: `9261634`

---

## Documentation Index

| Document | Contents |
|----------|----------|
| [`archana_person2/docs/ARCHITECTURE.md`](archana_person2/docs/ARCHITECTURE.md) | System architecture, data flow, module relationships |
| [`archana_person2/docs/DATASET.md`](archana_person2/docs/DATASET.md) | Dataset source, class labels, preprocessing pipeline |
| [`archana_person2/docs/MODELS.md`](archana_person2/docs/MODELS.md) | ML algorithms, training, evaluation, hyperparameters |
| [`archana_person2/docs/API_REFERENCE.md`](archana_person2/docs/API_REFERENCE.md) | All public classes, methods, and return values |
| [`archana_person2/docs/INSTALLATION.md`](archana_person2/docs/INSTALLATION.md) | Installation, environment setup, troubleshooting |
| [`archana_person2/docs/FEATURE_EXTRACTION.md`](archana_person2/docs/FEATURE_EXTRACTION.md) | Color Histogram, HOG, Combined feature specification |
| [`archana_person2/docs/INTEGRATION.md`](archana_person2/docs/INTEGRATION.md) | Integration bridge, Sadhvika's DatasetLoader interface |
| [`archana_person2/docs/PREDICTION.md`](archana_person2/docs/PREDICTION.md) | Prediction pipeline, AquaticPredictor, CLI usage |
| [`archana_person2/docs/VALIDATION_REPORT.md`](archana_person2/docs/VALIDATION_REPORT.md) | Phase 11 test results, bug fixes, real-data status |

---

*Project maintained by Archana Ulloju (Person 2) · November 2026*
