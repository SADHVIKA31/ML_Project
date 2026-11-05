# ML Models — Algorithms, Training & Evaluation

**Author:** Archana Ulloju (Person 2)  
**Project:** Machine Learning for Robust Aquatic Biodiversity Monitoring  
**Last Updated:** 2026-11-05 (Phase 12)

---

## 1. Overview

Person 2 implements three classical supervised classifiers, all wrapped in
leakage-safe scikit-learn Pipelines:

| Model | Phase | File | sklearn Class |
|-------|-------|------|---------------|
| Logistic Regression | 3 | `src/logistic_regression.py` | `LogisticRegression` |
| Random Forest | 4 | `src/random_forest.py` | `RandomForestClassifier` |
| Support Vector Machine | 5 | `src/svm_classifier.py` | `SVC` + `CalibratedClassifierCV` |

All three models:
- Are wrapped in `sklearn.pipeline.Pipeline([('scaler', StandardScaler()), ('classifier', ...)])`.
- Scale features using `StandardScaler` **fitted on training data only**.
- Select hyperparameters using the **validation split only** (test split never touched during tuning).
- Evaluate final performance on the **test split exactly once**.
- Are serialised to `.joblib` files in `archana_person2/artifacts/models/`.

---

## 2. Logistic Regression (Phase 3)

### Algorithm

Multinomial Logistic Regression with softmax output and $L_2$ regularisation.

$$P(Y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$

Loss function:

$$\mathcal{L} = -\sum_{i=1}^N \sum_{k=1}^K \mathbb{I}(y_i = k) \log P(Y = k \mid \mathbf{x}_i) + \frac{1}{2C}\sum_k \|\mathbf{w}_k\|_2^2$$

### Hyperparameters

| Parameter | Search Grid | Default |
|-----------|-------------|---------|
| `C` (inverse regularisation) | `[0.01, 0.1, 1.0, 5.0, 10.0]` | `1.0` |
| `solver` | `lbfgs` | `lbfgs` |
| `max_iter` | `1000` | `1000` |
| `multi_class` | `multinomial` | `multinomial` |

**Selection criterion:** Validation Macro-F1.

### Key Class: `AquaticLogisticRegression`

```python
from archana_person2.src.logistic_regression import AquaticLogisticRegression

lr = AquaticLogisticRegression()
lr.fit(X_train, y_train, X_val, y_val)      # fits + selects best C on val
metrics = lr.evaluate(X_test, y_test)        # single test evaluation
lr.save("artifacts/models/logistic_regression_pipeline.joblib")
```

### Saved Artifact

`archana_person2/artifacts/models/logistic_regression_pipeline.joblib`

---

## 3. Random Forest (Phase 4)

### Algorithm

Bootstrap ensemble of $B$ independently trained decision trees:

$$\hat{Y}(\mathbf{x}) = \arg\max_{k} \frac{1}{B} \sum_{b=1}^B \mathbb{I}(T_b(\mathbf{x}) = k)$$

Each tree uses:
- **Bagging:** Each tree trains on a bootstrap sample of training data.
- **Random subspace:** At each split, only $\sqrt{D}$ features are evaluated.
- **Gini impurity** as the split criterion.

### Hyperparameters

| Parameter | Search Grid | Best |
|-----------|-------------|------|
| `n_estimators` | `[50, 100]` | `100` |
| `max_depth` | `[10, 20, None]` | `20` |
| `min_samples_split` | `[2, 5]` | `5` |
| `min_samples_leaf` | `[1, 2]` | `2` |

**Selection criterion:** Validation Macro-F1.

### Key Class: `AquaticRandomForest`

```python
from archana_person2.src.random_forest import AquaticRandomForest

rf = AquaticRandomForest()
rf.fit(X_train, y_train, X_val, y_val)
metrics = rf.evaluate(X_test, y_test)
rf.save("artifacts/models/random_forest_model.joblib")
```

### Saved Artifact

`archana_person2/artifacts/models/random_forest_model.joblib`

---

## 4. Support Vector Machine (Phase 5)

### Algorithm

Maximum-margin classification in a Reproducing Kernel Hilbert Space:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T\phi(\mathbf{x}_i)+b) \ge 1-\xi_i, \quad \xi_i \ge 0$$

**Kernels evaluated:**

| Kernel | Formula | Use case |
|--------|---------|---------|
| Linear | $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$ | High-dim features (D=7108), interpretable |
| RBF | $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$ | Non-linear morphological patterns |
| Polynomial | $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma\mathbf{x}_i^T\mathbf{x}_j + r)^d$ | Polynomial feature interactions |

### Probability Estimation

Probability calibration uses **Platt scaling** via `CalibratedClassifierCV`:

```python
from sklearn.calibration import CalibratedClassifierCV
base_svc = SVC(kernel=kernel, C=C, gamma=gamma)
calibrated = CalibratedClassifierCV(base_svc, ensemble=False)
```

> [!NOTE]
> `SVC(probability=True)` is deprecated in scikit-learn ≥ 1.9. Person 2 uses
> `CalibratedClassifierCV` as the leakage-safe, forward-compatible alternative.
> This fix was applied in Phase 11.

### Hyperparameters

| Parameter | Search Grid | Default |
|-----------|-------------|---------|
| `kernel` | `["linear", "rbf", "poly"]` | `rbf` |
| `C` | `[0.1, 1.0, 10.0]` | `1.0` |
| `gamma` | `["scale", "auto", 0.001]` | `scale` |

**Selection criterion:** Validation Macro-F1.

### Key Class: `AquaticSVMClassifier`

```python
from archana_person2.src.svm_classifier import AquaticSVMClassifier

svm = AquaticSVMClassifier()
svm.fit(X_train, y_train, X_val, y_val)
metrics = svm.evaluate(X_test, y_test)
svm.save("artifacts/models/svm_model.joblib")
```

### Saved Artifact

`archana_person2/artifacts/models/svm_model.joblib`

---

## 5. Evaluation Framework (Phase 6)

### Class: `CommonModelEvaluator`

Provides a standardised, identical evaluation procedure for all three models.

```python
from archana_person2.src.evaluation import CommonModelEvaluator

evaluator = CommonModelEvaluator(class_names=["cls_A", "cls_B", ...])

# Evaluate from raw predictions
result = evaluator.evaluate_predictions(y_true, y_pred)

# Evaluate a fitted sklearn model directly
result = evaluator.evaluate_model(fitted_pipeline, X_test, y_test)
```

**Return value structure:**

```python
{
  "metrics": {
      "accuracy": 0.9400,
      "macro_precision": 0.9455,
      "macro_recall": 0.9400,
      "macro_f1": 0.9397,
      "weighted_f1": 0.9397
  },
  "confusion_matrix": [[...], ...],       # K×K list of lists
  "classification_report": {              # per-class precision/recall/F1
      "Species_01": {"precision": ..., "recall": ..., "f1-score": ..., "support": ...},
      ...
  }
}
```

### Primary Metric: Macro-F1

$$\text{Macro-F1} = \frac{1}{K}\sum_{k=1}^K \text{F1}_k = \frac{1}{K}\sum_{k=1}^K \frac{2 \cdot \text{Prec}_k \cdot \text{Rec}_k}{\text{Prec}_k + \text{Rec}_k}$$

Macro-F1 is preferred over accuracy because it:
- Treats every species with **equal weight** regardless of class frequency.
- Prevents dominant species from masking poor performance on rare aquatic species.

---

## 6. Model Comparison Framework (Phase 7)

### Class: `ModelComparator`

Compares all three models across all three feature representations (9 configurations):

```python
from archana_person2.src.comparison import ModelComparator

comparator = ModelComparator()
results = comparator.compare_all(X_test, y_test)
comparator.plot_comparison(results, output_dir="artifacts/plots/")
comparator.save_report(results, "artifacts/metrics/comparison_report.json")
```

### Benchmark Results (Synthetic — NOT for Academic Reporting)

> [!WARNING]
> The table below was produced on the **5-class / 150-sample synthetic dataset**.
> These numbers verify software correctness only. Do not cite them in the academic report.

| Model | Feature | Accuracy | Macro-F1 | Log Loss | Train Time |
|-------|---------|----------|----------|----------|------------|
| Logistic Regression | color_hist | 1.0000 | 1.0000 | 0.0030 | 0.05 s |
| Random Forest | color_hist | 1.0000 | 1.0000 | 0.0250 | 0.12 s |
| SVM (Linear) | color_hist | 1.0000 | 1.0000 | 0.3068 | 0.01 s |
| Logistic Regression | hog | 0.9200 | 0.9195 | 0.4500 | 0.12 s |
| Random Forest | hog | 1.0000 | 1.0000 | 0.1713 | 0.14 s |
| SVM (Linear) | hog | 0.9400 | 0.9397 | 0.3349 | 0.03 s |
| Logistic Regression | combined | 1.0000 | 1.0000 | 0.0888 | 0.06 s |
| Random Forest | combined | 1.0000 | 1.0000 | 0.0988 | 0.15 s |
| SVM (Linear) | combined | 1.0000 | 1.0000 | 0.1673 | 0.04 s |

---

## 7. Training Scripts

| Script | Purpose |
|--------|---------|
| `archana_person2/scripts/train_logistic_regression.py` | Train and save LR model |
| `archana_person2/scripts/train_random_forest.py` | Train and save RF model |
| `archana_person2/scripts/train_svm.py` | Train and save SVM model |
| `archana_person2/scripts/evaluate_models.py` | Evaluate all 3 models |
| `archana_person2/scripts/run_model_comparison.py` | 9-way benchmark + plots |

```bash
# Train all three models (requires real data)
python archana_person2/scripts/train_logistic_regression.py
python archana_person2/scripts/train_random_forest.py
python archana_person2/scripts/train_svm.py

# Evaluate and compare
python archana_person2/scripts/evaluate_models.py
python archana_person2/scripts/run_model_comparison.py
```

---

## 8. Known Limitations

| Limitation | Detail |
|-----------|--------|
| No real data | All saved models are trained on 5-class synthetic fixtures. Real validation is PENDING. |
| RF probability | RandomForest's `predict_proba` is not exposed in the AquaticPredictor. Use LR or SVM for confidence scores. |
| SVM model on disk | Saved model was trained with old `SVC(probability=True)` API. Must be retrained after real data is available to use `CalibratedClassifierCV`. |
| No deep-learning comparison | No DL model code is present in `develop`. Cross-architecture comparison deferred. |

---

*Models documented by Archana Ulloju (Person 2) — November 2026*
