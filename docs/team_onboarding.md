# Team Onboarding Guide – Getting Started with the Dataset

> **Quick-start for:** Archan, Pragnya, Nithya
> **Dataset lead:** Sadhvika

---

## Step 1 — Clone the repo and check out develop

```bash
git clone https://github.com/SADHVIKA31/ML_Project.git
cd ML_Project
git checkout develop
git pull origin develop
```

---

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt \
            -r requirements_phase2.txt \
            -r requirements_phase3.txt \
            -r requirements_phase4.txt
```

For PyTorch training:
```bash
pip install torch torchvision
```

For Keras/TF training:
```bash
pip install tensorflow
```

---

## Step 3 — Get the dataset data files

> ⚠️ Large binary files (images and `.npy` arrays) are **not committed to Git**
> (they are in `.gitignore`). You need to obtain them separately.

**Option A — Run the full pipeline yourself (takes ~30-60 min):**

```bash
# 1. Download dataset (requires Kaggle API key)
python scripts/download_dataset.py --source kaggle --dest data/raw

# 2. Organise into class folders
python scripts/organise_dataset.py --src data/raw --dest data/processed

# 3. Group images by source video
python scripts/extract_video_groups.py --dataset data/processed --out data/interim/video_groups.json

# 4. Generate leakage-safe splits
python scripts/split_dataset.py --dataset data/processed --groups data/interim/video_groups.json --out data/splits --seed 42

# 5. Preprocess images
python scripts/preprocess.py --splits data/splits --dataset data/processed --out data/preprocessed --size 224 --normalize imagenet
```

**Option B — Ask Sadhvika for the pre-processed files:**
Sadhvika will share `data/preprocessed/` and `data/splits/` via Google Drive / shared storage.
Copy the received files into your local `ML_Project/` directory.

---

## Step 4 — Verify your setup

```bash
python scripts/verify_integration.py
```

You should see:
```
✅  Directories present
✅  Split CSV manifests readable
✅  Preprocessed arrays present
✅  Label map consistent
✅  DatasetLoader end-to-end
→ INTEGRATION VERIFIED ✅
```

If any check fails, refer to `docs/phase5_dataset_handover.md` → Raising Dataset Issues.

---

## Step 5 — Load data in your training script

```python
# ALWAYS use DatasetLoader — never load files manually
from scripts.dataset_loader import DatasetLoader

loader = DatasetLoader()
loader.print_summary()          # print dataset info

X_train, y_train = loader.load_numpy("train")
X_val,   y_val   = loader.load_numpy("val")
X_test,  y_test  = loader.load_numpy("test")
label_map        = loader.get_label_map()
num_classes      = loader.num_classes()
```

---

## Key Rules

> [!IMPORTANT]
> - **Never** load images manually from `data/processed/` in your training code.  
>   Always go through `DatasetLoader` to guarantee consistent preprocessing.
> - **Never** train or tune on `X_test / y_test`. It is for final evaluation only.
> - **Always** use `label_map.json` for class indices — never hardcode numbers.
> - **Always** use the same `seed=42` splits — never re-split the data yourself.

---

## Full documentation

See [`docs/phase5_dataset_handover.md`](phase5_dataset_handover.md) for complete details.
