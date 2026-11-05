# System Architecture — Person 2 Classical ML Pipeline

**Author:** Archana Ulloju (Person 2)  
**Project:** Machine Learning for Robust Aquatic Biodiversity Monitoring  
**Last Updated:** 2026-11-05 (Phase 12)

---

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       ML_Project (Repository Root)                        │
│                                                                           │
│  ┌──────────────────────────┐      ┌──────────────────────────────────┐  │
│  │  Person 1 — Sadhvika     │      │  Person 2 — Archana Ulloju       │  │
│  │  Dataset & Preprocessing │      │  Classical ML Pipeline           │  │
│  │                          │      │                                  │  │
│  │  scripts/dataset_loader  │◄─────│  src/integration.py              │  │
│  │  scripts/preprocess.py   │      │  src/dataset_loader.py           │  │
│  │  scripts/split_dataset   │      │                                  │  │
│  │                          │      │  Feature Extraction              │  │
│  │  data/splits/            │─────►│    color_histogram.py            │  │
│  │    train.csv             │      │    hog_features.py               │  │
│  │    val.csv               │      │    feature_extractor.py          │  │
│  │    test.csv              │      │                                  │  │
│  │                          │      │  Classifiers                     │  │
│  │  data/preprocessed/      │─────►│    logistic_regression.py        │  │
│  │    X_train.npy           │      │    random_forest.py              │  │
│  │    y_train.npy           │      │    svm_classifier.py             │  │
│  │    label_map.json        │      │                                  │  │
│  └──────────────────────────┘      │  Evaluation & Comparison         │  │
│                                    │    evaluation.py                 │  │
│                                    │    comparison.py                 │  │
│                                    │                                  │  │
│                                    │  Inference                       │  │
│                                    │    predict.py (AquaticPredictor) │  │
│                                    └──────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Dependency Graph

```
integration.py  (ClassicalMLPipeline — public entry point)
      │
      ├── predict.py           (AquaticPredictor)
      │       └── feature_extractor.py  (AquaticFeatureExtractor)
      │               ├── color_histogram.py  (ColorHistogramExtractor)
      │               └── hog_features.py     (HOGFeatureExtractor)
      │
      ├── logistic_regression.py  (AquaticLogisticRegression)
      ├── random_forest.py        (AquaticRandomForest)
      ├── svm_classifier.py       (AquaticSVMClassifier)
      │
      ├── evaluation.py    (CommonModelEvaluator)
      ├── comparison.py    (ModelComparator)
      └── dataset_loader.py (Person2DatasetLoader)
              └── scripts/dataset_loader.py  (Sadhvika's DatasetLoader — optional)
```

---

## 3. Data Flow: Image → Prediction

```
 Input Image (JPEG / PNG / BMP / TIFF)
         │
         ▼
   cv2.imread → BGR-to-RGB conversion → uint8 [H×W×3]
         │
         ▼
   AquaticFeatureExtractor.extract_image(image, representation)
         │
         ├── "color_hist"  → ColorHistogramExtractor  → 1,024-dim float32 vector
         │                      RGB histogram: 8×8×8 bins, L1-normalised
         │                      HSV histogram: 8×8×8 bins, L1-normalised
         │                      Concatenated: [512 RGB | 512 HSV]
         │
         ├── "hog"         → HOGFeatureExtractor       → 6,084-dim float32 vector
         │                      Resize to 224×224 → grayscale
         │                      skimage.feature.hog(orientations=9,
         │                        pixels_per_cell=(16,16), cells_per_block=(2,2),
         │                        block_norm="L2-Hys")
         │                      Shape: 13×13×4×9 = 6,084
         │
         └── "combined"    → Concatenate(color_hist, hog) → 7,108-dim float32 vector
         │
         ▼
   reshape to (1, D) float32
         │
         ▼
   sklearn Pipeline (loaded from .joblib)
         ├── StandardScaler.transform(X)      ← fitted on training data only
         └── Classifier.predict(X) / predict_proba(X)
                 ├── LogisticRegression   → predict_proba ✅
                 ├── SVM (CalibratedClassifierCV) → predict_proba ✅
                 └── RandomForest        → predict_proba ❌ (not exposed)
         │
         ▼
   Prediction Result dict
         {
           "predicted_class" : "Dascyllus_aruanus",
           "predicted_index" : 2,
           "confidence"      : 0.9134,          # max(probabilities), None for RF
           "probabilities"   : {"cls_1": 0.03, ...},  # None for RF
           "feature_vector"  : np.ndarray,       # shape [D]
           "representation"  : "combined",
           "model_name"      : "svm",
           "image_path"      : "/path/to/fish.jpg"
         }
```

---

## 4. Training Data Flow (Batch)

```
data/splits/train.csv  ─────────────────────────┐
data/splits/val.csv    ─────────────────────────┤
data/splits/test.csv   ─────────────────────────┤
                                                 │
                                                 ▼
                               Person2DatasetLoader / Sadhvika's DatasetLoader
                                                 │
                               for each image path in split:
                                 cv2.imread → uint8 RGB
                                                 │
                                                 ▼
                               AquaticFeatureExtractor.extract_batch()
                               → X_train (N_train × D), y_train
                               → X_val   (N_val   × D), y_val
                               → X_test  (N_test  × D), y_test
                               → cached as .npz in artifacts/features/
                                                 │
                                                 ▼
                         ┌────────────────────────────────────┐
                         │  Hyperparameter search on X_val    │  ← validation only
                         │  Select best C / kernel / n_trees  │
                         └────────────────────────────────────┘
                                                 │
                                                 ▼
                         StandardScaler.fit(X_train)  → transform X_train, X_val, X_test
                                                 │
                                                 ▼
                         Classifier.fit(X_train_scaled, y_train)
                                                 │
                                                 ▼
                         Classifier.predict(X_test_scaled)  ← test evaluated ONCE
                                                 │
                                                 ▼
                         CommonModelEvaluator.evaluate_predictions()
                         → accuracy, macro-F1, precision, recall, confusion matrix
                                                 │
                                                 ▼
                         joblib.dump(pipeline, artifacts/models/*.joblib)
```

---

## 5. File and Module Index

| File | Class / Function | Phase | Description |
|------|-----------------|-------|-------------|
| `src/__init__.py` | — | 1–10 | Lazy public API exports |
| `src/dataset_loader.py` | `Person2DatasetLoader` | 1 | Dataset loading, split reading |
| `src/color_histogram.py` | `ColorHistogramExtractor` | 2 | RGB+HSV histogram features |
| `src/hog_features.py` | `HOGFeatureExtractor` | 2 | HOG gradient features |
| `src/feature_extractor.py` | `AquaticFeatureExtractor` | 2 | Unified feature extraction |
| `src/logistic_regression.py` | `AquaticLogisticRegression` | 3 | LR training + evaluation |
| `src/random_forest.py` | `AquaticRandomForest` | 4 | RF training + evaluation |
| `src/svm_classifier.py` | `AquaticSVMClassifier` | 5 | SVM training + evaluation |
| `src/evaluation.py` | `CommonModelEvaluator` | 6 | Standard metrics framework |
| `src/comparison.py` | `ModelComparator` | 7 | Cross-model benchmarking |
| `src/predict.py` | `AquaticPredictor` | 8 | Inference pipeline |
| `src/integration.py` | `ClassicalMLPipeline` | 9 | Public API bridge |

---

## 6. Configuration

All runtime parameters are centralised in [`archana_person2/config/config.yaml`](../config/config.yaml).

Key sections:

| Section | Key Parameters |
|---------|---------------|
| `project` | `random_seed: 42`, contributor info |
| `dataset` | `splits_dir`, `preprocessed_dir`, `synthetic_dir`, `target_size: [224, 224]` |
| `feature_extraction.color_histogram` | `bins_rgb: [8,8,8]`, `bins_hsv: [8,8,8]`, `normalize: true` |
| `feature_extraction.hog` | `orientations: 9`, `pixels_per_cell: [16,16]`, `cells_per_block: [2,2]` |
| `models` | `output_dir`, hyperparameters for each model |
| `evaluation` | `primary_metric: macro_f1` |

---

## 7. Artifact Directory Layout

```
archana_person2/artifacts/
├── features/
│   ├── features_train_color_hist.npz   ← cached training color histograms
│   ├── features_train_hog.npz          ← cached training HOG features
│   ├── features_train_combined.npz     ← cached combined (7108-dim)
│   ├── features_val_*.npz              ← validation split caches
│   └── features_test_*.npz             ← test split caches
├── models/                             ← git-ignored, trained model files
│   ├── logistic_regression_pipeline.joblib
│   ├── random_forest_model.joblib
│   └── svm_model.joblib
├── metrics/                            ← JSON reports, CSV summaries
├── plots/                              ← confusion matrices, bar charts
└── fixtures/
    └── synthetic_dataset/              ← deterministic synthetic test data
```

---

## 8. Leakage-Safety Guarantees

| Guarantee | Mechanism |
|-----------|-----------|
| `StandardScaler` fitted only on training data | `scaler.fit(X_train)` only; `transform()` applied to val and test |
| Hyperparameter selection uses validation, never test | All grid search loops use `X_val` accuracy/F1 |
| Test set evaluated exactly once | Called only after best model is selected |
| Split partitioning by video_id | Prevents temporal leakage (same fish video in train and test) |
| Synthetic fixtures never mixed with real results | Separate `synthetic_dir`; real data files are gitignored |

---

*Architecture documented by Archana Ulloju (Person 2) — November 2026*
