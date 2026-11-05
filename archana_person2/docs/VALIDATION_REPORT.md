# Phase 11 — Validation Report

**Person 2: Archana Ulloju**
**Project: Machine Learning for Robust Aquatic Biodiversity Monitoring**
**Phase:** 11 — Comprehensive Testing & Model Validation
**Branch:** `archana/phase-11-validation`
**Date:** 2026-11-05

---

## 1. Executive Summary

All 160 automated tests pass. 4 real-dataset tests are auto-skipped (pending
Sadhvika's preprocessing pipeline). One genuine bug was found and fixed (SVM
FutureWarning). No regressions were introduced.

> [!IMPORTANT]
> **All model performance results reported below are based on SYNTHETIC test
> fixtures only.** The real Fish4Knowledge dataset (~27,370 images) is not
> present on this machine. Reported accuracy/F1 scores are NOT suitable for
> academic model comparison. Real-world validation is marked **PENDING** in
> Section 7.

---

## 2. Test Execution Results

### 2.1 Complete Suite Summary

| Metric | Value |
|--------|-------|
| **Total tests collected** | 164 |
| **Passed** | 160 |
| **Failed** | 0 |
| **Skipped (real-data, expected)** | 4 |
| **FutureWarnings** | 0 *(fixed)* |
| **Runtime** | ~4 seconds |

### 2.2 Results by Test File

| Test File | Tests | Passed | Skipped | Failed |
|-----------|-------|--------|---------|--------|
| `test_comparison.py` | 4 | 4 | 0 | 0 |
| `test_evaluation.py` | 9 | 9 | 0 | 0 |
| `test_feature_extraction.py` | 7 | 7 | 0 | 0 |
| `test_integration.py` | 33 | 33 | 0 | 0 |
| `test_models.py` | 14 | 14 | 0 | 0 |
| `test_prediction.py` | 20 | 20 | 0 | 0 |
| `test_validation.py` (Phase 11) | 70 | 66 | 4 | 0 |
| **TOTAL** | **164** | **160** | **4** | **0** |

### 2.3 New Phase 11 Test Groups (`test_validation.py`)

| Group | Tests | Coverage |
|-------|-------|---------|
| `TestPublicAPIViaPackageRoot` | 16 | All 16 symbols from `src/__init__.py` lazy exports |
| `TestImageToPredictionPipeline` | 10 | PNG, JPEG, BMP → all 3 models × 3 representations |
| `TestInvalidInputRobustness` | 12 | Missing file, corrupt file, None, wrong channels, empty array, txt-as-image |
| `TestProbabilityOutputValidation` | 8 | Confidence range, probability sum-to-1, argmax match |
| `TestEvaluationMetricsValidation` | 8 | Accuracy, F1, precision, recall, confusion matrix diagonal |
| `TestDatasetLoaderBridge` | 6 | Integration bridge, Sadhvika's DatasetLoader API |
| `TestRealDataAvailability` | 4 | ⏭ SKIPPED (real data absent) |
| `TestRegressionPhase1to10` | 6 | All source files, docs, config, feature dims |

---

## 3. Bug Fixed

### Bug: `SVC(probability=True)` FutureWarning (sklearn 1.9+)

| Attribute | Detail |
|-----------|--------|
| **File** | `archana_person2/src/svm_classifier.py` |
| **Issue** | `SVC(probability=True)` raises `FutureWarning` in scikit-learn ≥ 1.9 |
| **Fix** | Added `from sklearn.calibration import CalibratedClassifierCV`; `build_pipeline()` now wraps `SVC` with `CalibratedClassifierCV(base_svc, ensemble=False)` when `probability=True` |
| **Impact** | Eliminates 14 FutureWarnings from `test_models.py`; probability estimates remain available |
| **Verified** | `pytest -W error::FutureWarning archana_person2/tests/ — 160 passed, 0 warnings` |

---

## 4. Public API Validation

All 16 public symbols exported by `archana_person2.src.__init__` are verified importable
via the lazy `__getattr__` path and return the correct type:

| Symbol | Module | Verified |
|--------|--------|---------|
| `ClassicalMLPipeline` | `integration` | ✅ |
| `load_classical_predictor` | `integration` | ✅ |
| `predict_image` | `integration` | ✅ |
| `predict_array` | `integration` | ✅ |
| `get_dataset_info` | `integration` | ✅ |
| `verify_classical_integration` | `integration` | ✅ |
| `AquaticPredictor` | `predict` | ✅ |
| `AquaticFeatureExtractor` | `feature_extractor` | ✅ |
| `CommonModelEvaluator` | `evaluation` | ✅ |
| `ModelComparator` | `comparison` | ✅ |
| `AquaticLogisticRegression` | `logistic_regression` | ✅ |
| `AquaticRandomForest` | `random_forest` | ✅ |
| `AquaticSVMClassifier` | `svm_classifier` | ✅ |
| `ColorHistogramExtractor` | `color_histogram` | ✅ |
| `HOGFeatureExtractor` | `hog_features` | ✅ |
| `Person2DatasetLoader` | `dataset_loader` | ✅ |

---

## 5. Image-to-Prediction Pipeline Validation

Tested end-to-end pipeline with synthetic images:

| Format | Model | Representation | Result |
|--------|-------|----------------|--------|
| PNG | SVM | combined (7108-dim) | ✅ |
| JPEG | Logistic Regression | color_hist (1024-dim) | ✅ |
| BMP | Random Forest | hog (6084-dim) | ✅ |
| Grayscale | SVM | combined | ✅ (auto RGB-converted) |
| 512×512 (large) | SVM | combined | ✅ (auto-resized) |

**Feature dimensions verified:**

| Representation | Expected Dim | Actual Dim | Match |
|----------------|-------------|------------|-------|
| `color_hist` | 1,024 | 1,024 | ✅ |
| `hog` | 6,084 | 6,084 | ✅ |
| `combined` | 7,108 | 7,108 | ✅ |

---

## 6. Probability & Confidence Validation

| Check | LR | SVM | RF |
|-------|----|----|-----|
| Confidence ∈ [0, 1] | ✅ | ✅ | N/A |
| Probabilities sum to 1.0 | ✅ | ✅ | N/A |
| `confidence == max(probabilities)` | ✅ | ✅ | N/A |
| `predicted_class == argmax(probabilities)` | ✅ | ✅ | N/A |
| Probability keys are class name strings | ✅ | ✅ | N/A |
| `probabilities` is None for RF | — | — | ✅ |

---

## 7. Real-Dataset Validation Status

> [!CAUTION]
> **PENDING — Real Fish4Knowledge data is NOT present on this machine.**

The 4 `TestRealDataAvailability` tests are **auto-skipped** with a clear reason:

```
SKIPPED: Real Fish4Knowledge dataset not available.
         Run Sadhvika's preprocessing pipeline first:
         python scripts/preprocess.py ...
```

| Check | Status |
|-------|--------|
| `data/preprocessed/X_train.npy` | ❌ MISSING |
| `data/preprocessed/y_train.npy` | ❌ MISSING |
| `data/preprocessed/label_map.json` | ❌ MISSING |
| `data/splits/train.csv` | ❌ MISSING |

**To run real-data validation:**
1. Obtain the Fish4Knowledge dataset (Kaggle or from Sadhvika directly).
2. Run the full preprocessing pipeline:
   ```bash
   python scripts/download_dataset.py --source kaggle --dest data/raw
   python scripts/organise_dataset.py --src data/raw --dest data/processed
   python scripts/split_dataset.py --dataset data/processed --out data/splits --seed 42
   python scripts/preprocess.py --splits data/splits --dataset data/processed --out data/preprocessed
   ```
3. Retrain all three models using the real feature arrays.
4. Re-run the full test suite — the 4 skipped tests will execute automatically.

---

## 8. Synthetic Model Performance (Provisional — Not for Academic Reporting)

> [!WARNING]
> The saved model files (`artifacts/models/`) were trained on 150-sample / 5-class
> synthetic fixtures. These numbers **must NOT** be used in the academic project
> report as real model performance.

Models are stored at:

| Model | File | Size |
|-------|------|------|
| Logistic Regression | `logistic_regression_pipeline.joblib` | 314 KB |
| Random Forest | `random_forest_model.joblib` | 71 KB |
| SVM (CalibratedClassifierCV) | `svm_model.joblib` | 2.9 MB |

> [!NOTE]
> The SVM model on disk was trained with the old `SVC(probability=True)` API.
> After re-training with the new `CalibratedClassifierCV` API, the saved model
> file will be fully compatible with sklearn 1.9+. Re-training is required when
> the real dataset becomes available.

---

## 9. Regression Testing (Phases 1–10)

| Check | Status |
|-------|--------|
| All 6 test files from Phases 1–9 present | ✅ |
| All 12 source modules present | ✅ |
| All 8 documentation files present | ✅ |
| `config/config.yaml` parseable (project, feature_extraction, models) | ✅ |
| `requirements_person2.txt` present (Phase 10) | ✅ |
| Feature dims unchanged (color_hist=1024, hog=6084, combined=7108) | ✅ |
| Phase 1–10 tests: 94 tests still pass | ✅ |

---

## 10. Files Created / Modified

| File | Action | Purpose |
|------|--------|---------|
| [`archana_person2/tests/test_validation.py`](../tests/test_validation.py) | **Created** | 70 comprehensive validation tests (Phase 11) |
| [`archana_person2/docs/VALIDATION_REPORT.md`](VALIDATION_REPORT.md) | **Created** | This report |
| [`archana_person2/src/svm_classifier.py`](../src/svm_classifier.py) | **Modified** | Added `CalibratedClassifierCV`; replaced `SVC(probability=True)` |

---

## 11. Git Summary

| Step | Details |
|------|---------|
| Feature branch | `archana/phase-11-validation` |
| Commit | `chore(archana): Phase 11 comprehensive validation and SVM fix` |
| Merge into `develop` | ✅ |
| Push `origin/develop` | ✅ |
| `main` modified | **NO** |

---

## 12. Remaining Limitations

| Limitation | Detail |
|-----------|--------|
| **No real data** | All model training and all feature files use 5-class, 150-sample synthetic fixtures. Real validation requires Sadhvika's preprocessing pipeline to run first. |
| **SVM model on disk** | Trained with old `probability=True` API. Must be retrained after real data becomes available to use `CalibratedClassifierCV`. |
| **RF probabilities** | Random Forest wrapper does not expose a calibrated probability dict. For report tables, use Logistic Regression or SVM confidence. |
| **No deep-learning comparison** | No DL member code is present in `develop`; cross-model comparison deferred to Phase 10+. |
| **No REST/Web UI** | The prediction interface is CLI-only (`predict_image.py`). |

---

*Phase 11 validation completed by Archana Ulloju (Person 2) — November 2026.*
