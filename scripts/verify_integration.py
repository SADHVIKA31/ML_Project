"""
verify_integration.py
---------------------
Phase 5 – Integration Verification Script.

Run this script to confirm the dataset handover is complete and that
your environment is correctly configured to consume the dataset.

Checks
------
1. Required directories exist
2. Split CSV manifests are present and readable
3. Preprocessed NumPy arrays are present and have correct shape
4. label_map.json is present and consistent with split data
5. DatasetLoader works end-to-end
6. (Optional) PyTorch Dataset loads a single batch
7. (Optional) Keras generator yields a valid batch

Usage
-----
  python scripts/verify_integration.py
  python scripts/verify_integration.py --pytorch
  python scripts/verify_integration.py --keras

Expected output on a correct environment
-----------------------------------------
  ✅ Directories present
  ✅ Split CSV manifests readable
  ✅ Preprocessed arrays present
  ✅ Label map consistent
  ✅ DatasetLoader end-to-end
  ✅ [optional] PyTorch batch
  ✅ [optional] Keras batch
  → INTEGRATION VERIFIED ✅
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from project root
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

REQUIRED_DIRS = [
    "data/processed",
    "data/splits",
    "data/preprocessed",
    "reports/eda",
]

REQUIRED_SPLIT_CSVS = [
    "data/splits/train.csv",
    "data/splits/val.csv",
    "data/splits/test.csv",
    "data/splits/split_summary.json",
]

REQUIRED_NPY = [
    "data/preprocessed/X_train.npy",
    "data/preprocessed/y_train.npy",
    "data/preprocessed/X_val.npy",
    "data/preprocessed/y_val.npy",
    "data/preprocessed/X_test.npy",
    "data/preprocessed/y_test.npy",
    "data/preprocessed/label_map.json",
]

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def _check(label: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    msg  = f"  {icon}  {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)
    return ok


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_directories(root: Path) -> bool:
    missing = [d for d in REQUIRED_DIRS if not (root / d).exists()]
    if missing:
        return _check("Directories present", False,
                      f"Missing: {', '.join(missing)}")
    return _check("Directories present", True)


def check_split_csvs(root: Path) -> bool:
    missing = [f for f in REQUIRED_SPLIT_CSVS if not (root / f).exists()]
    if missing:
        return _check("Split CSV manifests readable", False,
                      f"Missing: {', '.join(missing)}")
    # Quick readability check
    import csv
    for f in REQUIRED_SPLIT_CSVS:
        p = root / f
        if p.suffix == ".csv":
            try:
                with open(p, newline="") as fh:
                    rows = list(csv.DictReader(fh))
                if not rows:
                    return _check("Split CSV manifests readable", False,
                                  f"{f} is empty")
            except Exception as e:
                return _check("Split CSV manifests readable", False, str(e))
    return _check("Split CSV manifests readable", True,
                  f"train / val / test CSVs present and non-empty")


def check_preprocessed_arrays(root: Path) -> bool:
    import numpy as np
    missing = [f for f in REQUIRED_NPY if not (root / f).exists()]
    if missing:
        return _check("Preprocessed arrays present", False,
                      f"Missing: {', '.join(missing)}")
    # Shape check on train
    try:
        X = np.load(root / "data/preprocessed/X_train.npy", mmap_mode="r")
        if X.ndim != 4 or X.shape[3] != 3:
            return _check("Preprocessed arrays present", False,
                          f"X_train shape unexpected: {X.shape}")
        return _check("Preprocessed arrays present", True,
                      f"X_train shape = {X.shape}  dtype={X.dtype}")
    except Exception as e:
        return _check("Preprocessed arrays present", False, str(e))


def check_label_map(root: Path) -> bool:
    import csv, json
    lm_path = root / "data/preprocessed/label_map.json"
    if not lm_path.exists():
        return _check("Label map consistent", False, "label_map.json missing")
    with open(lm_path) as f:
        label_map = json.load(f)
    if not label_map:
        return _check("Label map consistent", False, "label_map.json is empty")
    # Check class names appear in train CSV
    train_csv = root / "data/splits/train.csv"
    if train_csv.exists():
        with open(train_csv, newline="") as f:
            csv_classes = {row["class"] for row in csv.DictReader(f)}
        missing = csv_classes - set(label_map.keys())
        if missing:
            return _check("Label map consistent", False,
                          f"Classes in train CSV missing from label_map: {missing}")
    return _check("Label map consistent", True,
                  f"{len(label_map)} classes mapped")


def check_datasetloader(root: Path) -> bool:
    try:
        from scripts.dataset_loader import DatasetLoader
        loader = DatasetLoader(
            splits_dir    = root / "data/splits",
            preprocessed  = root / "data/preprocessed",
            label_map_path= root / "data/preprocessed/label_map.json",
        )
        label_map = loader.get_label_map()
        split_info = loader.split_info()
        return _check("DatasetLoader end-to-end", True,
                      f"{len(label_map)} classes  "
                      f"train={split_info.get('train', {}).get('num_images')} "
                      f"val={split_info.get('val', {}).get('num_images')} "
                      f"test={split_info.get('test', {}).get('num_images')}")
    except Exception as e:
        return _check("DatasetLoader end-to-end", False, str(e))


def check_pytorch(root: Path) -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        print(f"  {WARN} PyTorch not installed — skipping (pip install torch)")
        return True

    try:
        from torch.utils.data import DataLoader
        from scripts.dataset_loader import PyTorchFishDataset
        ds = PyTorchFishDataset("train", loader=__import__(
            "scripts.dataset_loader", fromlist=["DatasetLoader"]
        ).DatasetLoader(
            splits_dir     = root / "data/splits",
            preprocessed   = root / "data/preprocessed",
            label_map_path = root / "data/preprocessed/label_map.json",
        ))
        dl = DataLoader(ds, batch_size=4, shuffle=False)
        X_batch, y_batch = next(iter(dl))
        return _check("PyTorch Dataset batch", True,
                      f"X={tuple(X_batch.shape)}  y={tuple(y_batch.shape)}")
    except Exception as e:
        return _check("PyTorch Dataset batch", False, str(e))


def check_keras(root: Path) -> bool:
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print(f"  {WARN} TensorFlow not installed — skipping (pip install tensorflow)")
        return True

    try:
        from scripts.dataset_loader import keras_generator, DatasetLoader
        loader = DatasetLoader(
            splits_dir     = root / "data/splits",
            preprocessed   = root / "data/preprocessed",
            label_map_path = root / "data/preprocessed/label_map.json",
        )
        gen = keras_generator("train", batch_size=4, shuffle=False, loader=loader)
        X_batch, y_batch = next(gen)
        return _check("Keras generator batch", True,
                      f"X={X_batch.shape}  y={y_batch.shape}")
    except Exception as e:
        return _check("Keras generator batch", False, str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify dataset integration")
    parser.add_argument("--root",    default=".",       help="Project root directory")
    parser.add_argument("--pytorch", action="store_true", help="Run PyTorch check")
    parser.add_argument("--keras",   action="store_true", help="Run Keras check")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print("\n" + "=" * 60)
    print("  DATASET INTEGRATION VERIFICATION")
    print(f"  Project root: {root}")
    print("=" * 60 + "\n")

    results = [
        check_directories(root),
        check_split_csvs(root),
        check_preprocessed_arrays(root),
        check_label_map(root),
        check_datasetloader(root),
    ]

    if args.pytorch:
        results.append(check_pytorch(root))
    if args.keras:
        results.append(check_keras(root))

    print()
    all_passed = all(results)
    if all_passed:
        print("  → INTEGRATION VERIFIED ✅  Dataset is ready for model training.\n")
    else:
        print("  → INTEGRATION INCOMPLETE ❌  Fix the failing checks above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
