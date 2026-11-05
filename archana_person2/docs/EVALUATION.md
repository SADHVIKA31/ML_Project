# Common Model Evaluation Framework Specification — Person 2

**Author:** Archana Ulloju (Person 2 - Classical ML & Feature Extraction Lead)  
**Project:** Machine Learning for Robust Aquatic Biodiversity Monitoring  
**Target Integration Branch:** `develop`  
**Phase:** Phase 6 — Common Model Evaluation Framework  

---

## 1. Overview and Purpose

The **Common Model Evaluation Framework** provides an objective, standardized, and reproducible evaluation pipeline for all classical machine learning models in this project:
1. **Multinomial Logistic Regression** (StandardScaler + $L_2$ Regularized Logistic Regression)
2. **Random Forest Classifier** (Bagged Decision Tree Ensemble with Gini Impurity)
3. **Support Vector Machine (SVM)** (StandardScaler + Maximum-Margin SVC with Platt Probability Scaling)

Every model is evaluated on the exact same test feature vectors, using the exact same ground-truth labels, under identical metric formulations.

---

## 2. Beginner-Friendly Guide to Machine Learning Evaluation

### 1. What is Model Evaluation?
Model evaluation is the formal assessment of a trained machine learning model's predictive capability and generalisation performance on independent, unseen data.

### 2. Why Do We Evaluate Machine Learning Models?
Training accuracy can be deceptively optimistic because models can memorize sample patterns (overfitting). Evaluation on a strictly isolated test set ensures the model has learned true underlying morphological representations rather than noise.

### 3. What is Accuracy?
$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
Accuracy measures overall correctness. However, in underwater aquatic biodiversity datasets with severe class imbalance (e.g. abundant common species vs. rare endangered fish), accuracy is misleading: a naive classifier predicting only the majority class can yield high accuracy while completely failing on rare species.

### 4. What is Precision?
$$\text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}$$
Precision answers: *"When the model predicts species $k$, how often is it actually species $k$?"* High precision means low false-positive rates (avoiding false alarms).

### 5. What is Recall?
$$\text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$$
Recall (sensitivity) answers: *"Out of all true instances of species $k$ in the water, what proportion did the model correctly detect?"* High recall is crucial for monitoring rare aquatic species.

### 6. What is F1 Score?
$$\text{F1}_k = 2 \times \frac{\text{Precision}_k \times \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
The harmonic mean of precision and recall. It penalizes extreme imbalances between precision and recall.

### 7. What is Macro Averaging?
$$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k$$
Macro averaging computes the unweighted arithmetic mean across all species, treating every class with equal importance regardless of frequency. **Macro-F1 is the PRIMARY metric for this project** because it prevents dominant species from masking poor performance on minority species.

### 8. What is Weighted Averaging?
$$\text{Weighted-F1} = \sum_{k=1}^K \left(\frac{N_k}{N}\right) \text{F1}_k$$
Weighted averaging weights each species' score by its true support $N_k$.

### 9. What is a Confusion Matrix?
A $K \times K$ table where entry $C_{i, j}$ represents the number of samples known to belong to class $i$ that were predicted as class $j$. Diagonal cells represent correct predictions (true positives); off-diagonal cells highlight systematic inter-species misclassifications.

### 10. What is a Classification Report?
A consolidated tabular summary listing precision, recall, F1-score, and support ($N$) for each individual species class, alongside global macro and weighted averages.

### 11. Why Must All Models Use the Same Test Dataset?
Comparing models evaluated on different splits or different subsets introduces sample bias. Fair benchmarking requires identical test instances, identical image preconditioning, and identical class indices.

### 12. Why Should We Avoid Data Leakage?
Data leakage occurs when information from outside the training split (validation or test distributions) inadvertently influences model training. Examples include computing `StandardScaler` mean and variance across the full dataset or using frames from the same video in both training and test sets. In our pipeline, all transformers fit strictly on `X_train`, and splits are partitioned by `video_id`.

### 13. How Do We Execute the Evaluation Script?
```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONDONTWRITEBYTECODE="1"
$env:MPLBACKEND="Agg"
python archana_person2/scripts/evaluate_models.py
```

### 14. How Do We Interpret the Generated Results?
- Check **Macro-F1**: Indicates balanced multi-species identification.
- Check **Confusion Matrices** (`archana_person2/artifacts/plots/confusion_matrix_*.png`): Identify if visually similar species are frequently confused.
- Check **Log Loss**: Measures the quality of predicted probabilities (lower is better).
- Check **ROC-AUC (Macro OvR)**: Measures separation capability across decision thresholds.

---

## 3. Dataset Authenticity & Experimental Limitations

> [!WARNING]
> **Provisional Software Verification Status:**
> The current test evaluation results (Accuracy = 1.0000, Macro-F1 = 1.0000) were obtained on the **synthetic test fixture** (150 images across 5 classes).
> - Because raw datasets are not checked into Git, these metrics verify that the **software architecture, class order, and metric functions operate flawlessly**.
> - These results **must not** be reported as real-world ecological accuracy in final project submissions. Real empirical evaluation will be conducted once the full ~27,370 image Fish4Knowledge dataset is downloaded.

---

## 4. Generated Artifacts

- **Detailed Evaluation Metrics JSON:** `archana_person2/artifacts/metrics/evaluation_results.json`
- **Summary Comparison CSV:** `archana_person2/artifacts/metrics/evaluation_summary.csv`
- **Classification Reports JSON:** `archana_person2/artifacts/metrics/classification_report.json`
- **Visualizations:**
  - `archana_person2/artifacts/plots/confusion_matrix_logistic_regression.png`
  - `archana_person2/artifacts/plots/confusion_matrix_random_forest.png`
  - `archana_person2/artifacts/plots/confusion_matrix_svm.png`
  - `archana_person2/artifacts/plots/model_evaluation_metrics_summary.png`
  - `archana_person2/artifacts/plots/per_class_performance.png`
