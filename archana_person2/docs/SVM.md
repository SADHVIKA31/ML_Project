# Support Vector Machine (SVM) Classifier Specification — Person 2

**Author:** Archana Ulloju (Person 2 - Classical ML & Feature Extraction Lead)  
**Project:** Machine Learning for Robust Aquatic Biodiversity Monitoring  
**Target Integration Branch:** `develop`  
**Phase:** Phase 5 — Support Vector Machine (SVM) Implementation & Evaluation  

---

## 1. Algorithmic Formulation

Support Vector Classification (SVC) constructs maximum-margin separating hyperplanes in high-dimensional reproducing kernel Hilbert spaces (RKHS). For multi-class aquatic species monitoring, the one-vs-rest (or one-vs-one) formulation optimizes:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i$$

subject to:
$$y_i (\mathbf{w}^T \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad \forall i \in \{1, \dots, N\}$$

where:
- $C > 0$ denotes the regularization penalty trading off margin maximization against training constraint slack.
- $\phi(\mathbf{x})$ denotes non-linear mapping induced by Mercer kernel functions $K(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$.

### Evaluated Kernels:
1. **Linear Kernel:**
   $$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$$
   Computationally efficient baseline; highly effective in very high-dimensional feature spaces ($D = 7,108$).
2. **Radial Basis Function (RBF) Kernel:**
   $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2), \quad \gamma > 0$$
   Captures complex local aquatic morphology and non-linear feature interactions.
3. **Polynomial Kernel:**
   $$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + \text{coef0})^d, \quad d \in \{2, 3\}$$
   Models polynomial combinations of color and gradient features.

---

## 2. Leakage-Safe Pipeline Architecture

1. **StandardScaler Scaling:**
   - SVM margins are strictly scale-dependent. Unnormalized HOG and Color Histogram features would distort hyperplane orientation.
   - `StandardScaler` calculates mean $\mu_{\text{train}}$ and variance $\sigma^2_{\text{train}}$ strictly on training samples.
   - Validation and test splits are transformed using frozen training statistics.
2. **Scikit-Learn Pipeline Encapsulation:**
   - Scaler and `SVC` classifier are chained inside `Pipeline([('scaler', StandardScaler()), ('classifier', SVC(...))])`.
   - Probability estimation (`probability=True`) uses Platt's scaling with internal cross-validation.
3. **Validation-Set Selection:**
   - Kernel comparison and hyperparameter searches evaluate performance strictly on the validation set.
   - The test set is evaluated exactly once after model selection to guarantee unbiased generalisation estimation.
   - Selection metric: **Validation Macro-F1** (addressing severe class imbalance).

---

## 3. Experimental Deliverables

- **Model Pipeline:** `archana_person2/artifacts/models/svm_model.joblib`
- **Kernel Comparison Experiments:** `archana_person2/artifacts/metrics/svm_kernel_experiments.json`
- **Hyperparameter Tuning Log:** `archana_person2/artifacts/metrics/svm_experiments.json`
- **Evaluation Metrics Report:** `archana_person2/artifacts/metrics/svm_metrics.json`
- **Prediction Logs:** `archana_person2/artifacts/metrics/svm_predictions.csv`
- **Confusion Matrix:** `archana_person2/artifacts/plots/svm_confusion_matrix.png`
