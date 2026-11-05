# Logistic Regression Baseline Specification – Person 2

**Author:** Archana Ulloju (Person 2 - Classical ML & Feature Extraction Lead)  
**Project:** Machine Learning for Robust Aquatic Biodiversity Monitoring  
**Target Integration Branch:** `develop`  
**Phase:** Phase 3 – Logistic Regression Implementation & Evaluation  

---

## 1. Algorithmic Formulation

Multinomial Logistic Regression models the posterior probability of species class $k \in \{1, \dots, K\}$ given handcrafted feature representation $\mathbf{x} \in \mathbb{R}^D$ using the softmax link function:

$$P(Y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$

Objective function minimized with $L_2$ Tikhonov regularization:
$$\min_{\mathbf{w}, b} -\sum_{i=1}^N \sum_{k=1}^K \mathbb{I}(y_i = k) \log P(Y = k \mid \mathbf{x}_i) + \frac{1}{2C} \sum_{k=1}^K \|\mathbf{w}_k\|_2^2$$

where $C > 0$ controls the inverse regularization strength.

---

## 2. Pipeline Architecture & Leakage Prevention

1. **StandardScaler Preprocessing:**
   - Features (e.g. HOG and Color Histogram) exhibit disparate numerical scales and variances.
   - The `StandardScaler` calculates mean $\mu_{\text{train}}$ and variance $\sigma^2_{\text{train}}$ strictly on training samples.
   - Test samples are transformed using frozen training parameters, eliminating transductive data leakage.
2. **Encapsulation via Scikit-Learn Pipeline:**
   - Preprocessing and classification are chained in a single serializable `Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(...))])`.
3. **Hyperparameter Selection:**
   - $C \in \{0.01, 0.1, 1.0, 5.0, 10.0\}$ is tuned strictly on the validation split.
   - Selection criterion: Validation **Macro-F1**. The test split is kept isolated.

---

## 3. Artifact Deliverables

- **Model Pipeline:** `archana_person2/artifacts/models/logistic_regression_pipeline.joblib`
- **Metric Report:** `archana_person2/artifacts/metrics/logistic_regression_metrics.json`
- **Prediction Logs:** `archana_person2/artifacts/metrics/logistic_regression_predictions.csv`
- **Confusion Matrix:** `archana_person2/artifacts/plots/logistic_regression_confusion_matrix.png`
