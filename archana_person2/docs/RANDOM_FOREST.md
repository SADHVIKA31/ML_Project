# Random Forest Classifier Specification – Person 2

**Author:** Archana Ulloju (Person 2 - Classical ML & Feature Extraction Lead)  
**Project:** Machine Learning for Robust Aquatic Biodiversity Monitoring  
**Target Integration Branch:** `develop`  
**Phase:** Phase 4 – Random Forest Implementation & Evaluation  

---

## 1. Algorithmic Formulation

Random Forest is a bootstrap ensemble of $B$ decision trees $\{T_b\}_{b=1}^B$:
$$\hat{Y}(\mathbf{x}) = \arg\max_{k \in \{1,\dots,K\}} \frac{1}{B} \sum_{b=1}^B \mathbb{I}(T_b(\mathbf{x}) = k)$$

### Key Characteristics:
- **Bootstrap Aggregation (Bagging):** Reduces model variance across noisy underwater images.
- **Random Subspace Projection:** At each candidate split, only a random subset of $m \approx \sqrt{D}$ features is evaluated, decorrelating individual trees.
- **Gini Impurity Criterion:**
  $$I_G(p) = 1 - \sum_{k=1}^K p_k^2$$
- **Mean Decrease in Impurity (Feature Importance):**
  $$I(\text{feature } j) = \sum_{b=1}^B \sum_{t \in T_b: v(s_t)=j} \Delta I_G(s_t, t)$$

---

## 2. Hyperparameter Search Grid

Hyperparameter experiments are evaluated strictly on the **validation split** (zero test leakage):
- `n_estimators`: $[50, 100]$
- `max_depth`: $[10, 20, \text{None}]$
- `min_samples_split`: $[2, 5]$
- `min_samples_leaf`: $[1, 2]$
- Metric for selection: **Validation Macro-F1**.

---

## 3. Artifact Deliverables

- **Model Artifact:** `archana_person2/artifacts/models/random_forest_model.joblib`
- **Experiment Records:** `archana_person2/artifacts/metrics/random_forest_experiments.json`
- **Evaluation Report:** `archana_person2/artifacts/metrics/random_forest_metrics.json`
- **Prediction Logs:** `archana_person2/artifacts/metrics/random_forest_predictions.csv`
- **Confusion Matrix:** `archana_person2/artifacts/plots/random_forest_confusion_matrix.png`
- **Feature Importance Chart:** `archana_person2/artifacts/plots/random_forest_feature_importance.png`
