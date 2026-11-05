# Dataset Integration and Understanding Report – Person 2

**Author:** Archana Ulloju (Person 2 - Classical ML & Feature Extraction Lead)  
**Project:** Machine Learning for Robust Aquatic Biodiversity Monitoring  
**Target Integration Branch:** `develop`  
**Phase:** Phase 1 – Dataset Understanding & Repository Inspection  

---

## 1. Executive Summary

This document specifies the integration between the dataset preparation completed by **Person 1 (Sadhvika)** and the downstream classical machine learning pipeline implemented by **Person 2 (Archana Ulloju)**. 

Our proposal evaluates classical machine learning models (Logistic Regression, Random Forest, Support Vector Machine) alongside deep transfer learning architectures on aquatic species monitoring. The dataset requirements demand strict temporal/spatial leakage prevention and standardized multi-channel image representation.

---

## 2. Dataset Architecture & Split Guarantees

### 2.1 File Organization
The authoritative dataset layout established in Phase 1-5 handover:
- `data/processed/<ClassName>/`: Clean, validated RGB fish images.
- `data/splits/train.csv`: Training manifest (70%).
- `data/splits/val.csv`: Validation manifest (15%).
- `data/splits/test.csv`: Independent evaluation test manifest (15%).
- `data/preprocessed/label_map.json`: Authoritative alphabetical integer class mapping.

### 2.2 Leakage Prevention Guarantee
- **Video-Group Disjointness:** Images originate from underwater continuous surveillance cameras. Splitting naively by image index or random permutation introduces near-duplicate temporal leakage (successive frames of the same fish).
- **Guarantee:** All splits are partitioned strictly by `video_id`. No `video_id` present in `train.csv` exists in `val.csv` or `test.csv`.
- **Random Seed:** 42 is enforced globally across all modules.

---

## 3. Person 2 Integration Strategy

1. **Dataset Loader Module (`Person2DatasetLoader`):**
   - Automatically detects official split CSV manifests (`data/splits/train.csv`, `val.csv`, `test.csv`).
   - Resolves image paths relative to repository root.
   - Converts BGR to RGB and resizes uniformly to 224×224 px (standard proposal input).
   - Validates image dimensions, channels, and corrupt headers.
2. **Offline Synthetic Fixture:**
   - Because raw datasets (~27,370 images) are git-ignored and distributed separately via shared storage / Kaggle credentials, `Person2DatasetLoader` includes a zero-dependency synthetic aquatic dataset generator.
   - The fixture simulates underwater turbidity, illumination gradients, and distinct morphological classes, allowing 100% of the unit test suite and integration pipeline to be validated reproducibly without blocking on manual data downloads.
3. **Primary Metric:**
   - **Macro-F1** is selected as the primary performance metric due to extreme class imbalance inherent in aquatic surveillance datasets.

---

## 4. Verification Check
Running `python archana_person2/scripts/inspect_dataset.py` produces:
- `archana_person2/artifacts/metrics/dataset_statistics.json`
- `archana_person2/artifacts/plots/class_distribution.png`
- Audit log verifying zero leakage across splits.
