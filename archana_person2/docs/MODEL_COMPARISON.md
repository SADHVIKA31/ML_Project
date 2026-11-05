# Classical Model Comparison & Benchmarking Specification — Person 2

**Author:** Archana Ulloju (Person 2 - Classical ML & Feature Extraction Lead)  
**Project:** Machine Learning for Robust Aquatic Biodiversity Monitoring  
**Target Integration Branch:** `develop`  
**Phase:** Phase 7 — Model Comparison and Visualization  

---

## 1. Executive Summary

This study conducts a systematic empirical benchmark across all three classical machine learning baselines:
1. **Multinomial Logistic Regression** with $L_2$ Regularization and StandardScaler.
2. **Random Forest Classifier** with Bagged Decision Trees and Gini Impurity.
3. **Support Vector Machine (SVM)** with Maximum-Margin Hyperplanes and Platt Probability Scaling.

To evaluate representation impact, each model was evaluated across **three distinct feature spaces**:
- **Color Histogram** ($1,024$ dimensions: RGB + HSV joint color distribution).
- **HOG — Histogram of Oriented Gradients** ($6,084$ dimensions: local edge and morphological gradients).
- **Combined Representation** ($7,108$ dimensions: multi-modal concatenated representation).

Every configuration was evaluated on the **exact same test split** with frozen preprocessing parameters.

---

## 2. Comprehensive 9-Way Benchmark Table

| Model | Feature Representation | Dimensions | Accuracy | Macro Precision | Macro Recall | Macro-F1 (Primary) | Weighted-F1 | Log Loss | Train Time | Inference Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Color Histogram | 1,024 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 0.0030 | 0.05s | 32.6 ms |
| **Random Forest** | Color Histogram | 1,024 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 0.0250 | 0.12s | 40.2 ms |
| **SVM (Linear)** | Color Histogram | 1,024 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 0.3068 | 0.01s | 26.5 ms |
| **Logistic Regression** | HOG | 6,084 | 0.9200 | 0.9285 | 0.9200 | **0.9195** | 0.9195 | 0.4500 | 0.12s | 23.7 ms |
| **Random Forest** | HOG | 6,084 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 0.1713 | 0.14s | 37.2 ms |
| **SVM (Linear)** | HOG | 6,084 | 0.9400 | 0.9455 | 0.9400 | **0.9397** | 0.9397 | 0.3349 | 0.03s | 40.1 ms |
| **Logistic Regression** | Combined | 7,108 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 0.0888 | 0.06s | 25.9 ms |
| **Random Forest** | Combined | 7,108 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 0.0988 | 0.15s | 36.6 ms |
| **SVM (Linear)** | Combined | 7,108 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 0.1673 | 0.04s | 44.0 ms |

---

## 3. In-Depth Comparative Analysis

### 3.1 Feature Representation Modality Impact
1. **HOG Modality ($D = 6,084$):**
   - HOG captures geometric edge orientations and contour silhouettes.
   - When evaluated on HOG alone, linear boundaries struggle slightly with visually similar silhouette profiles:
     - Logistic Regression achieves $0.9195$ Macro-F1 ($92.0\%$ Accuracy).
     - SVM achieves $0.9397$ Macro-F1 ($94.0\%$ Accuracy).
     - Random Forest successfully leverages non-linear decision splits to maintain $1.0000$ Macro-F1.
2. **Color Histogram Modality ($D = 1,024$):**
   - Color distribution provides strong discriminative signal across the distinct species pigmentation patterns.
   - High computational efficiency with rapid training ($<0.05$s).
3. **Combined Modality ($D = 7,108$):**
   - **Crucial finding:** Combining Color Histograms with HOG restores perfect multi-class discrimination across **all three classifiers**, raising Logistic Regression from $0.9195 \to 1.0000$ and SVM from $0.9397 \to 1.0000$.
   - Multi-modal fusion provides complementary spatial gradient and chromatic cues, compensating for the weaknesses of individual feature descriptors.

### 3.2 Model Trade-off Analysis
1. **Random Forest:**
   - **Strengths:** Most robust across all representations ($1.0000$ Macro-F1 on all three modalities, including HOG alone). Invariant to feature scaling.
   - **Weaknesses:** Slightly higher training latency ($0.15$s) due to bagging 100 decision trees.
2. **Support Vector Machine (Linear):**
   - **Strengths:** Fastest training latency ($0.01$s - $0.04$s), maximum-margin decision hyperplanes.
   - **Weaknesses:** Requires probability calibration via Platt scaling which increases inference latency ($44.0$ ms).
3. **Logistic Regression:**
   - **Strengths:** Highly compact, fast inference, lowest cross-entropy log loss on color features ($0.0030$).
   - **Weaknesses:** Requires linear separability, which drops performance on pure gradient features ($0.9195$ Macro-F1 on HOG alone).

---

## 4. Best Classical ML Baseline Identification

### Top Ranked Baselines:
1. **Primary Operational Baseline (Multi-Modal):**
   - **Logistic Regression (Combined Representation)**: Achieves **1.0000 Macro-F1**, $1.0000$ Accuracy, low Log Loss ($0.0888$), and fastest combined inference ($25.9$ ms for 50 samples / $0.52$ ms per sample).
2. **Most Robust Classifier Across Single Modalities:**
   - **Random Forest**: The only classifier maintaining **1.0000 Macro-F1 across all three modalities** (Color Hist, HOG, and Combined).

---

## 5. Experimental Validity & Dataset Caveat

> [!IMPORTANT]
> **Provisional Software Verification Status:**
> As verified in the Phase 5 Data Provenance Audit, these results were computed on the **synthetic software verification dataset** (150 images across 5 classes).
> All pipeline code, feature extractors, and comparative scripts are verified and 100% ready.
> When the full ~27,370 image Fish4Knowledge dataset is downloaded, executing `python archana_person2/scripts/run_model_comparison.py` will update this document with full-scale empirical benchmark figures.

---

## 6. Generated Visualizations

- `archana_person2/artifacts/plots/model_performance_comparison.png`
- `archana_person2/artifacts/plots/accuracy_comparison.png`
- `archana_person2/artifacts/plots/macro_f1_comparison.png`
- `archana_person2/artifacts/plots/feature_representation_impact.png`
- `archana_person2/artifacts/plots/inference_latency_comparison.png`
