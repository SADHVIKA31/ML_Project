"""
dataset_loader.py
-----------------
Phase 5 – Handover Utility: Shared dataset loader for ALL team members.

This is the SINGLE canonical way to load the prepared dataset into any
training or evaluation script. All team members (Archan, Pragnya, Nithya)
MUST use this module to load data — this guarantees consistent splits,
label encoding, and preprocessing across all models.

Features
--------
- Load train / val / test splits from CSV manifests (Phase 2)
- Load preprocessed NumPy arrays (Phase 3)
- Lazy image loading with on-the-fly preprocessing (Phase 3 pipeline)
- Returns label map (Phase 3)
- PyTorch Dataset-compatible (optional, if torch is installed)
- TensorFlow/Keras generator-compatible (optional)

Usage — NumPy arrays (fastest, recommended)
-------------------------------------------
    from scripts.dataset_loader import DatasetLoader

    loader = DatasetLoader()
    X_train, y_train = loader.load_numpy("train")
    X_val,   y_val   = loader.load_numpy("val")
    X_test,  y_test  = loader.load_numpy("test")
    label_map        = loader.get_label_map()

Usage — Lazy image loading (memory-efficient for large datasets)
----------------------------------------------------------------
    loader = DatasetLoader()
    train_records = loader.load_split_records("train")  # list of dicts
    for record in train_records:
        img_array = loader.load_and_preprocess(record["image_path"])
        label     = record["label_idx"]

Usage — PyTorch Dataset
-----------------------
    from scripts.dataset_loader import PyTorchFishDataset
    from torch.utils.data import DataLoader

    train_ds = PyTorchFishDataset("train")
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

Usage — Keras generator
------------------------
    from scripts.dataset_loader import keras_generator

    train_gen = keras_generator("train", batch_size=32)
    val_gen   = keras_generator("val",   batch_size=32, shuffle=False)

Requirements: numpy, Pillow
Optional:     torch (PyTorch), tensorflow (Keras)
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Default paths (resolve relative to project root)
# ---------------------------------------------------------------------------

_HERE         = Path(__file__).resolve().parent.parent   # project root
SPLITS_DIR    = _HERE / "data" / "splits"
PREPROCESSED  = _HERE / "data" / "preprocessed"
LABEL_MAP_PATH = PREPROCESSED / "label_map.json"

# Preprocessing defaults (must match Phase 3 settings)
DEFAULT_SIZE      = (224, 224)
DEFAULT_NORMALIZE = "imagenet"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """
    Canonical dataset loader for the Fish4Knowledge project.

    Parameters
    ----------
    splits_dir    : Path to directory containing train/val/test.csv
    preprocessed  : Path to directory containing X_<split>.npy, y_<split>.npy
    label_map_path: Path to label_map.json
    target_size   : (H, W) for lazy loading
    normalize     : "imagenet" | "minmax" | "none"
    """

    def __init__(
        self,
        splits_dir:    str | Path = SPLITS_DIR,
        preprocessed:  str | Path = PREPROCESSED,
        label_map_path: str | Path = LABEL_MAP_PATH,
        target_size:   tuple[int, int] = DEFAULT_SIZE,
        normalize:     str = DEFAULT_NORMALIZE,
    ):
        self.splits_dir     = Path(splits_dir)
        self.preprocessed   = Path(preprocessed)
        self.label_map_path = Path(label_map_path)
        self.target_size    = target_size
        self.normalize      = normalize

        self._label_map: dict[str, int] | None = None
        self._inv_label_map: dict[int, str] | None = None

    # -------------------------------------------------------------------
    # Label map
    # -------------------------------------------------------------------

    def get_label_map(self) -> dict[str, int]:
        """Return {class_name: integer_index} mapping."""
        if self._label_map is None:
            if not self.label_map_path.exists():
                raise FileNotFoundError(
                    f"label_map.json not found at {self.label_map_path}\n"
                    "Run Phase 3 preprocessing first:\n"
                    "  python scripts/preprocess.py"
                )
            with open(self.label_map_path) as f:
                self._label_map = json.load(f)
        return self._label_map

    def get_class_name(self, idx: int) -> str:
        """Return class name for a given integer index."""
        if self._inv_label_map is None:
            self._inv_label_map = {v: k for k, v in self.get_label_map().items()}
        return self._inv_label_map.get(idx, f"unknown_{idx}")

    def num_classes(self) -> int:
        return len(self.get_label_map())

    # -------------------------------------------------------------------
    # NumPy loading (fastest)
    # -------------------------------------------------------------------

    def load_numpy(
        self, split: str, mmap: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load pre-computed stacked NumPy arrays for a split.

        Parameters
        ----------
        split : "train" | "val" | "test"
        mmap  : If True, use memory-mapping (reduces RAM usage for large datasets)

        Returns
        -------
        X : float32 array of shape (N, H, W, 3)
        y : int32  array of shape (N,)
        """
        X_path = self.preprocessed / f"X_{split}.npy"
        y_path = self.preprocessed / f"y_{split}.npy"

        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Preprocessed arrays not found for split '{split}'.\n"
                f"Expected:\n  {X_path}\n  {y_path}\n"
                "Run Phase 3 preprocessing:\n"
                "  python scripts/preprocess.py --splits data/splits "
                "--dataset data/processed --out data/preprocessed"
            )

        mode = "r" if mmap else None
        X = np.load(X_path, mmap_mode=mode)
        y = np.load(y_path, mmap_mode=mode)
        return X, y

    # -------------------------------------------------------------------
    # CSV record loading (lazy, memory-efficient)
    # -------------------------------------------------------------------

    def load_split_records(self, split: str) -> list[dict]:
        """
        Load image path records from a split CSV manifest.

        Each record is a dict:
            {class, video_id, image_path, filename, label_idx}

        Parameters
        ----------
        split : "train" | "val" | "test"
        """
        csv_path = self.splits_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {csv_path}\n"
                "Run Phase 2 splitting:\n"
                "  python scripts/split_dataset.py"
            )

        label_map = self.get_label_map()
        records = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                row["label_idx"] = label_map.get(row["class"], -1)
                records.append(row)
        return records

    # -------------------------------------------------------------------
    # On-the-fly image loading + preprocessing
    # -------------------------------------------------------------------

    def load_and_preprocess(self, image_path: str | Path) -> np.ndarray:
        """
        Load a single image, apply the Phase 3 preprocessing pipeline,
        and return a float32 array of shape (H, W, 3).

        This is identical to the preprocessing applied by preprocess.py —
        use this for inference on new images.
        """
        path = Path(image_path)
        img  = Image.open(path).convert("RGB")
        img  = img.resize(self.target_size, Image.Resampling.LANCZOS)
        arr  = np.array(img, dtype=np.float32)

        if self.normalize == "minmax":
            arr = arr / 255.0
        elif self.normalize == "imagenet":
            arr = arr / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

        return arr

    # -------------------------------------------------------------------
    # Split statistics
    # -------------------------------------------------------------------

    def split_info(self) -> dict:
        """Return a summary of all three splits."""
        info: dict[str, dict] = {}
        for split in ("train", "val", "test"):
            X_path = self.preprocessed / f"X_{split}.npy"
            if X_path.exists():
                X = np.load(X_path, mmap_mode="r")
                info[split] = {
                    "num_images": len(X),
                    "shape":      list(X.shape),
                    "dtype":      str(X.dtype),
                }
            else:
                csv_path = self.splits_dir / f"{split}.csv"
                if csv_path.exists():
                    with open(csv_path, newline="") as f:
                        n = sum(1 for _ in csv.DictReader(f))
                    info[split] = {"num_images": n, "source": "csv_only"}
                else:
                    info[split] = {"num_images": "not found"}
        return info

    def print_summary(self):
        """Print a human-readable dataset summary to stdout."""
        label_map = self.get_label_map()
        split_info = self.split_info()

        print("=" * 60)
        print("  FISH4KNOWLEDGE DATASET — HANDOVER SUMMARY")
        print("=" * 60)
        print(f"  Classes     : {len(label_map)}")
        print(f"  Normalize   : {self.normalize}")
        print(f"  Target size : {self.target_size}")
        print()
        print(f"  {'Split':<10} {'Images':>8} {'Shape'}")
        print(f"  {'-'*40}")
        for split, info in split_info.items():
            shape = info.get("shape", "—")
            n     = info.get("num_images", "—")
            print(f"  {split:<10} {str(n):>8}   {shape}")
        print()
        print(f"  Label Map:")
        for cls, idx in sorted(label_map.items(), key=lambda x: x[1]):
            print(f"    {idx:3d} → {cls}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# PyTorch Dataset (optional)
# ---------------------------------------------------------------------------

class PyTorchFishDataset:
    """
    PyTorch-compatible Dataset for the Fish4Knowledge project.

    Usage
    -----
        from scripts.dataset_loader import PyTorchFishDataset
        from torch.utils.data import DataLoader

        ds = PyTorchFishDataset("train")
        dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
        for images, labels in dl:
            ...  # images: (B, C, H, W) float32 tensor
    """

    def __init__(
        self,
        split:   str,
        loader:  DatasetLoader | None = None,
        use_npy: bool = True,
    ):
        self.loader  = loader or DatasetLoader()
        self.split   = split
        self.use_npy = use_npy

        if use_npy:
            self._X, self._y = self.loader.load_numpy(split)
        else:
            self._records = self.loader.load_split_records(split)

    def __len__(self) -> int:
        return len(self._X) if self.use_npy else len(self._records)

    def __getitem__(self, idx: int):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is not installed. pip install torch")

        if self.use_npy:
            # (H, W, C) → (C, H, W) for PyTorch convention
            img = torch.from_numpy(self._X[idx].copy()).permute(2, 0, 1)
            lbl = torch.tensor(int(self._y[idx]), dtype=torch.long)
        else:
            rec = self._records[idx]
            arr = self.loader.load_and_preprocess(rec["image_path"])
            img = torch.from_numpy(arr).permute(2, 0, 1)
            lbl = torch.tensor(rec["label_idx"], dtype=torch.long)

        return img, lbl


# ---------------------------------------------------------------------------
# Keras / TensorFlow generator (optional)
# ---------------------------------------------------------------------------

def keras_generator(
    split:      str,
    batch_size: int = 32,
    shuffle:    bool = True,
    loader:     DatasetLoader | None = None,
) -> Generator:
    """
    Infinite Keras-compatible generator that yields (X_batch, y_batch).

    Usage
    -----
        gen = keras_generator("train", batch_size=32)
        model.fit(gen, steps_per_epoch=len(train_records) // 32, epochs=50)
    """
    loader  = loader or DatasetLoader()
    records = loader.load_split_records(split)

    if shuffle:
        import random
        random.shuffle(records)

    n = len(records)
    idx = 0
    while True:
        batch_records = records[idx: idx + batch_size]
        if len(batch_records) < batch_size:
            if shuffle:
                import random
                random.shuffle(records)
            idx = 0
            batch_records = records[:batch_size]

        X_batch = np.stack([loader.load_and_preprocess(r["image_path"]) for r in batch_records])
        y_batch = np.array([r["label_idx"] for r in batch_records], dtype=np.int32)

        idx = (idx + batch_size) % n
        yield X_batch, y_batch


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = DatasetLoader()
    loader.print_summary()

    print("\n[TEST] Loading NumPy arrays ...")
    try:
        X_train, y_train = loader.load_numpy("train")
        print(f"  X_train : {X_train.shape}  dtype={X_train.dtype}")
        print(f"  y_train : {y_train.shape}  classes={set(y_train.tolist())}")
        print(f"  Pixel range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    except FileNotFoundError as e:
        print(f"  [INFO] {e}")

    print("\n[TEST] Label map:")
    for cls, idx in list(loader.get_label_map().items())[:5]:
        print(f"  {idx} → {cls}")
    print("  ...")
