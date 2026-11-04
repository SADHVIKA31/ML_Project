# ML_Project – Fish4Knowledge Dataset Analysis

> **Phase 1 — Dataset & Preprocessing Lead**
> Branch: `sadhvika_phase1`

---

## Project Overview

This project classifies underwater aquatic species using the **Fish4Knowledge** dataset (or comparable alternative). Phase 1 focuses entirely on dataset collection, understanding, and audit.

---

## Phase 1 Responsibilities

| Step | Task | Script / Deliverable |
|------|------|----------------------|
| 1 | Download dataset | `scripts/download_dataset.py` |
| 2 | Extract & organise | `scripts/organise_dataset.py` |
| 3–8 | Full dataset audit | `scripts/dataset_audit.py` |
| 9 | Dataset summary | `scripts/generate_summary.py` → `reports/` |

---

## Folder Structure

```
ML_Project/
├── data/
│   ├── raw/          ← downloaded raw files land here
│   ├── interim/      ← intermediate / partially processed files
│   └── processed/    ← clean, class-separated images ready for training
├── notebooks/        ← exploratory Jupyter notebooks
├── scripts/          ← Python scripts (download, organise, audit, summarise)
│   ├── download_dataset.py
│   ├── organise_dataset.py
│   ├── dataset_audit.py
│   └── generate_summary.py
├── reports/          ← generated reports (JSON, Markdown, CSV)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (Kaggle)
#    Ensure ~/.kaggle/kaggle.json is set up
python scripts/download_dataset.py --source kaggle --dest data/raw

# 3. Organise into class folders
python scripts/organise_dataset.py --src data/raw --dest data/processed

# 4. Run dataset audit
python scripts/dataset_audit.py --dataset data/processed --report reports/dataset_audit.json

# 5. Generate final summary report
python scripts/generate_summary.py --audit reports/dataset_audit.json
```

---

## Expected Deliverables (Phase 1)

- [x] Downloaded dataset (`data/raw/`)
- [x] Organised dataset folders (`data/processed/<ClassName>/`)
- [x] Dataset statistics (`reports/dataset_audit.json`)
- [x] Class distribution summary (`reports/dataset_summary.md`, `reports/class_distribution.csv`)

---

## Branch Workflow

```
sadhvika_phase1  ──PR──►  develop  ──PR──►  main
```

All changes must be PR'd into `develop`. Do **not** push directly to `main`.

---

## Dataset Reference

- **Fish4Knowledge** (Kaggle): https://www.kaggle.com/datasets/markdaniellampa/fish-dataset
- Alternative: [Fish Image Dataset](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots)