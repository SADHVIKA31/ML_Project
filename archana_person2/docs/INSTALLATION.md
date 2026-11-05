# Installation & Environment Setup

**Author:** Archana Ulloju (Person 2)  
**Project:** Machine Learning for Robust Aquatic Biodiversity Monitoring  
**Last Updated:** 2026-11-05 (Phase 12)

---

## 1. Requirements

### System Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| Python | 3.10+ | 3.14.4 |
| OS | Windows 10 / Linux / macOS | Windows 11 |
| RAM | 4 GB | 16 GB |
| Disk | 500 MB (code + synthetic data) | — |
| Disk (with real data) | ~5 GB | — |

### Python Packages — Person 2

Listed in [`requirements_person2.txt`](../../requirements_person2.txt):

```
numpy>=1.26.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
scipy>=1.11.0
opencv-python>=4.8.0
Pillow>=10.0.0
joblib>=1.3.0
PyYAML>=6.0.1
matplotlib>=3.8.0      # optional — Phase 7 comparison plots
seaborn>=0.13.0        # optional — Phase 7 comparison plots
```

> [!IMPORTANT]
> **`pandas` is intentionally excluded.**
> Python 3.14 has a known C-extension deadlock with pandas on Windows.
> All CSV and JSON I/O in Person 2's pipeline uses Python stdlib `csv`, `json`,
> and `numpy` instead.

---

## 2. Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/SADHVIKA31/ML_Project.git
cd ML_Project
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Person 2 dependencies

```bash
pip install -r requirements_person2.txt
```

### Step 4 — (Optional) Install full project dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Verify installation

```bash
python -c "
import sklearn; print('sklearn:', sklearn.__version__)
import cv2;     print('opencv:', cv2.__version__)
import skimage; print('skimage:', skimage.__version__)
import numpy;   print('numpy:', numpy.__version__)
print('All Person 2 dependencies OK.')
"
```

---

## 3. Environment Variables

Set these before running any Python script or pytest on Windows:

```powershell
# PowerShell
$env:PYTHONDONTWRITEBYTECODE = "1"    # suppress .pyc file creation
$env:MPLBACKEND = "Agg"               # non-interactive matplotlib backend
$env:MPLCONFIGDIR = "c:\Users\ulloj\ML_Project\.mplcache"  # writable cache dir
```

```bash
# Linux / macOS
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/path/to/ML_Project/.mplcache
```

---

## 4. Running the Test Suite

```powershell
# Full Person 2 test suite (164 tests, ~6 seconds)
$env:PYTHONDONTWRITEBYTECODE="1"; $env:MPLBACKEND="Agg"; $env:MPLCONFIGDIR="c:\Users\ulloj\ML_Project\.mplcache"
python -m pytest archana_person2/tests/ -v --tb=short

# With FutureWarning check (strict mode)
python -m pytest archana_person2/tests/ -v --tb=short -W error::FutureWarning

# Run a single test file
python -m pytest archana_person2/tests/test_validation.py -v

# Run only real-data tests (auto-skipped if data is unavailable)
python -m pytest archana_person2/tests/test_validation.py -v -k "RealData"
```

**Expected result:** `160 passed, 4 skipped, 0 failed`  
(4 skipped = `TestRealDataAvailability`, auto-skipped when Fish4Knowledge data is absent)

---

## 5. Running Scripts

### Single-image prediction

```powershell
python archana_person2/scripts/predict_image.py --image path/to/fish.jpg
python archana_person2/scripts/predict_image.py --image path/to/fish.jpg --model logistic_regression --representation color_hist
python archana_person2/scripts/predict_image.py --image path/to/fish.jpg --json
```

### Feature extraction

```powershell
python archana_person2/scripts/run_feature_extraction.py
```

### Training models (requires real data)

```powershell
python archana_person2/scripts/train_logistic_regression.py
python archana_person2/scripts/train_random_forest.py
python archana_person2/scripts/train_svm.py
```

### Evaluation and comparison

```powershell
python archana_person2/scripts/evaluate_models.py
python archana_person2/scripts/run_model_comparison.py
```

### Dataset inspection

```powershell
python archana_person2/scripts/inspect_dataset.py
```

### Integration smoke-test

```powershell
python -c "
from archana_person2.src.integration import verify_classical_integration
verify_classical_integration(verbose=True)
"
```

---

## 6. Configuration

Edit [`archana_person2/config/config.yaml`](../config/config.yaml) to change:

- Dataset paths (`dataset.splits_dir`, `dataset.preprocessed_dir`)
- Feature extraction parameters (bin counts, HOG cell sizes)
- Model hyperparameters (C, n_estimators, kernel)
- Output directories for models, metrics, plots

---

## 7. Troubleshooting

### `ModuleNotFoundError: archana_person2`

Ensure you are running from the project root (`ML_Project/`), not from a subdirectory:

```bash
cd c:\Users\ulloj\ML_Project
python -m pytest archana_person2/tests/ -v
```

### `FileNotFoundError: svm_model.joblib`

Models are gitignored. Train models first:

```bash
python archana_person2/scripts/train_svm.py
```

Or run tests — they use synthetic fixtures and don't require pre-trained models.

### matplotlib backend error on Windows

Set the backend before importing matplotlib:

```powershell
$env:MPLBACKEND = "Agg"
```

### `FutureWarning: SVC(probability=True)` on sklearn ≥ 1.9

This was fixed in Phase 11. Ensure you have the latest `develop` branch:

```bash
git checkout develop
git pull origin develop
```

### `pandas` import error

`pandas` is intentionally not installed. Person 2's pipeline does not use pandas.
If another team member's code requires it, install it separately:

```bash
pip install pandas
```

### Python 3.14 — `EPERM mkdir` error with `pnpm` / `npm`

This is a transient sandbox error. Retry the command. If it persists, run with elevated permissions.

---

## 8. Kaggle API Setup (for dataset download)

To download the Fish4Knowledge dataset, configure your Kaggle API key:

1. Go to https://www.kaggle.com/settings → API → Create New Token.
2. Save `kaggle.json` to `~/.kaggle/kaggle.json` (Linux/macOS) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/macOS).
4. Run: `python scripts/download_dataset.py --source kaggle --dest data/raw`

---

*Installation guide by Archana Ulloju (Person 2) — November 2026*
