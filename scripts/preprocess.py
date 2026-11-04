"""
preprocess.py
-------------
Phase 3 – Steps 3-9: Full image preprocessing pipeline.

Responsibilities
----------------
3.  Handle corrupted / unreadable images (skip with logging).
4.  Resize images to a consistent target size (default 224×224 for CNNs).
5.  Convert images to RGB format (strip alpha, convert grayscale).
6.  Normalise pixel values to [0, 1] float32 (ImageNet mean/std optional).
7.  Prepare class labels from directory names.
8.  Encode species labels into integer indices and save label_map.json.
9.  Apply identical preprocessing across train / val / test splits.

Outputs
-------
  data/preprocessed/
    train/  val/  test/          (NumPy .npy arrays per class, or flat arrays)
  data/preprocessed/label_map.json
  data/preprocessed/split_stats.json
  reports/preprocessing_summary.md

Usage
-----
  # Preprocess all three splits
  python scripts/preprocess.py \\
      --splits    data/splits \\
      --dataset   data/processed \\
      --out       data/preprocessed \\
      --size      224 \\
      --normalize imagenet          # choices: minmax | imagenet | none

  # Quick test on processed folder directly (no split CSVs needed)
  python scripts/preprocess.py \\
      --dataset   data/processed \\
      --out       data/preprocessed \\
      --size      224

Requirements: pip install opencv-python Pillow numpy tqdm
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ImageNet channel-wise mean and std (RGB order)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

logging.basicConfig(
    format="%(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 3: Load & validate a single image
# ---------------------------------------------------------------------------

def load_image(path: Path) -> Image.Image | None:
    """
    Load an image safely.  Returns a PIL Image or None if unreadable.
    Handles corrupted files, truncated images, and format errors.
    """
    try:
        img = Image.open(path)
        img.verify()          # detect corruption early
    except (UnidentifiedImageError, Exception) as e:
        log.warning("Corrupted / unreadable – skipping: %s  [%s]", path.name, e)
        return None

    # Re-open after verify (PIL closes stream after verify)
    try:
        img = Image.open(path)
        img.load()            # force full decode
        return img
    except Exception as e:
        log.warning("Cannot decode – skipping: %s  [%s]", path.name, e)
        return None


# ---------------------------------------------------------------------------
# Step 4-6: Resize + colour convert + normalise
# ---------------------------------------------------------------------------

def preprocess_image(
    img: Image.Image,
    target_size: tuple[int, int],
    normalize: str,          # "minmax" | "imagenet" | "none"
) -> np.ndarray:
    """
    Returns a float32 NumPy array of shape (H, W, 3).

    Steps:
      4. Resize  → target_size using high-quality LANCZOS resampling
      5. Convert → RGB  (strips alpha, converts grayscale/palette modes)
      6. Normalise pixel values
    """
    # --- Step 5: convert to RGB ---
    if img.mode != "RGB":
        img = img.convert("RGB")

    # --- Step 4: resize ---
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    # --- numpy cast ---
    arr = np.array(img, dtype=np.float32)   # shape (H, W, 3), range [0, 255]

    # --- Step 6: normalise ---
    if normalize == "minmax":
        arr = arr / 255.0                                        # → [0, 1]
    elif normalize == "imagenet":
        arr = arr / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD              # ImageNet Z-score
    # normalize == "none" → keep raw uint8-equivalent floats

    return arr


# ---------------------------------------------------------------------------
# Step 7-8: Label preparation
# ---------------------------------------------------------------------------

def build_label_map(class_names: list[str]) -> dict[str, int]:
    """Map sorted class names → integer indices (0-based, alphabetical order)."""
    return {cls: idx for idx, cls in enumerate(sorted(class_names))}


# ---------------------------------------------------------------------------
# Step 9: Process a full split
# ---------------------------------------------------------------------------

def process_split(
    records: list[dict],
    dataset_dir: Path,
    out_dir: Path,
    split_name: str,
    label_map: dict[str, int],
    target_size: tuple[int, int],
    normalize: str,
    save_npy: bool,
) -> dict:
    """
    Process every image in a split's record list.

    Returns stats dict: {total, processed, skipped, per_class}
    """
    split_out = out_dir / split_name
    split_out.mkdir(parents=True, exist_ok=True)

    # Collect per-class arrays if saving flat .npy
    class_images: dict[str, list[np.ndarray]] = {}
    class_labels: dict[str, list[int]]        = {}

    skipped = 0
    processed = 0

    for record in tqdm(records, desc=f"  {split_name:5s}", leave=False):
        cls_name  = record["class"]
        img_path  = Path(record["image_path"])

        if not img_path.exists():
            log.warning("File missing: %s", img_path)
            skipped += 1
            continue

        img = load_image(img_path)
        if img is None:
            skipped += 1
            continue

        arr = preprocess_image(img, target_size, normalize)
        label = label_map[cls_name]

        if save_npy:
            # Save individual file: data/preprocessed/<split>/<class>/<stem>.npy
            dest_dir = split_out / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            np.save(dest_dir / (img_path.stem + ".npy"), arr)
        else:
            class_images.setdefault(cls_name, []).append(arr)
            class_labels.setdefault(cls_name, []).append(label)

        processed += 1

    if not save_npy:
        # Save one stacked array per split: X_<split>.npy, y_<split>.npy
        all_X = np.stack([img for imgs in class_images.values() for img in imgs])
        all_y = np.array([lbl for lbls in class_labels.values() for lbl in lbls], dtype=np.int32)
        np.save(out_dir / f"X_{split_name}.npy", all_X)
        np.save(out_dir / f"y_{split_name}.npy", all_y)
        log.info("Saved X_%s.npy (%s) and y_%s.npy", split_name, all_X.shape, split_name)

    per_class = {cls: len(imgs) for cls, imgs in class_images.items()} if not save_npy \
                else {}

    return {
        "total":     len(records),
        "processed": processed,
        "skipped":   skipped,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess image dataset for ML training")
    parser.add_argument("--splits",    default="data/splits",       help="Split directory (from Phase 2)")
    parser.add_argument("--dataset",   default="data/processed",    help="Organised class-folder dataset")
    parser.add_argument("--out",       default="data/preprocessed", help="Output directory")
    parser.add_argument("--size",      type=int, default=224,       help="Target image size in pixels (default 224)")
    parser.add_argument("--normalize", default="imagenet",
                        choices=["minmax", "imagenet", "none"],
                        help="Pixel normalisation strategy (default imagenet)")
    parser.add_argument("--save-npy",  action="store_true",
                        help="Save individual .npy files instead of stacked X/y arrays")
    parser.add_argument("--reports",   default="reports",           help="Reports output directory")
    args = parser.parse_args()

    splits_dir  = Path(args.splits)
    dataset_dir = Path(args.dataset)
    out_dir     = Path(args.out)
    target_size = (args.size, args.size)

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Discover class names ---
    if dataset_dir.exists():
        class_names = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    else:
        sys.exit(f"[ERROR] Dataset directory not found: {dataset_dir}")

    if not class_names:
        sys.exit(f"[ERROR] No class sub-directories found in {dataset_dir}")

    # --- Step 7-8: Build & save label map ---
    label_map = build_label_map(class_names)
    label_map_path = out_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    log.info("Label map saved → %s  (%d classes)", label_map_path, len(label_map))
    log.info("Label map: %s", label_map)

    # --- Step 9: Process each split ---
    split_stats = {}

    if splits_dir.exists() and (splits_dir / "train.csv").exists():
        for split_name in ("train", "val", "test"):
            csv_path = splits_dir / f"{split_name}.csv"
            if not csv_path.exists():
                log.warning("Split CSV not found: %s — skipping", csv_path)
                continue
            records = _read_csv(csv_path)
            log.info("Processing split '%s' (%d records) ...", split_name, len(records))
            stats = process_split(
                records, dataset_dir, out_dir, split_name,
                label_map, target_size, args.normalize, args.save_npy
            )
            split_stats[split_name] = stats
    else:
        # Fallback: process entire dataset as a single "all" split
        log.warning("No split CSVs found under %s — processing full dataset as 'all'.", splits_dir)
        all_images = [
            p for p in dataset_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        records = [{"class": p.parent.name, "image_path": str(p)} for p in all_images]
        stats = process_split(
            records, dataset_dir, out_dir, "all",
            label_map, target_size, args.normalize, args.save_npy
        )
        split_stats["all"] = stats

    # --- Save split stats ---
    stats_path = out_dir / "split_stats.json"
    full_stats = {
        "target_size":   [args.size, args.size],
        "normalize":     args.normalize,
        "num_classes":   len(label_map),
        "split_stats":   split_stats,
    }
    with open(stats_path, "w") as f:
        json.dump(full_stats, f, indent=2)

    # --- Console summary ---
    print("\n" + "=" * 60)
    print("  PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Target size  : {args.size}×{args.size}")
    print(f"  Normalize    : {args.normalize}")
    print(f"  Classes      : {len(label_map)}")
    print(f"  {'Split':<10} {'Total':>8} {'Processed':>10} {'Skipped':>8}")
    print(f"  {'-'*40}")
    for sname, s in split_stats.items():
        print(f"  {sname:<10} {s['total']:>8} {s['processed']:>10} {s['skipped']:>8}")
    print("=" * 60)
    print(f"\n[DONE] Preprocessed data → {out_dir.resolve()}")
    print(f"[DONE] Label map         → {label_map_path.resolve()}")
    print(f"[DONE] Stats             → {stats_path.resolve()}")


if __name__ == "__main__":
    main()
