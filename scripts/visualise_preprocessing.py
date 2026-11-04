"""
visualise_preprocessing.py
--------------------------
Phase 3 – Deliverable: Sample before-and-after preprocessing images.

Picks a random sample of images from each class (or from split CSVs),
applies the full preprocessing pipeline, and saves side-by-side comparison
figures to reports/before_after/.

Usage:
    python scripts/visualise_preprocessing.py \\
        --dataset   data/processed \\
        --out       reports/before_after \\
        --size      224 \\
        --normalize imagenet \\
        --samples   3          # images per class

Requirements: pip install Pillow numpy matplotlib tqdm
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for servers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ImageNet stats (must match preprocess.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing (mirrors preprocess.py exactly – kept standalone for clarity)
# ---------------------------------------------------------------------------

def _apply_preprocessing(
    img: Image.Image,
    target_size: tuple[int, int],
    normalize: str,
) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    if normalize == "minmax":
        arr = arr / 255.0
    elif normalize == "imagenet":
        arr = arr / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def _denormalize_for_display(arr: np.ndarray, normalize: str) -> np.ndarray:
    """Bring normalised array back to [0,1] uint8 for matplotlib display."""
    if normalize == "imagenet":
        arr = arr * IMAGENET_STD + IMAGENET_MEAN
    arr = np.clip(arr, 0, 1)
    return arr


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def save_before_after(
    img_path: Path,
    out_dir: Path,
    target_size: tuple[int, int],
    normalize: str,
):
    try:
        original = Image.open(img_path)
        original.load()
    except Exception as e:
        print(f"[WARN] Cannot open {img_path.name}: {e}")
        return

    preprocessed = _apply_preprocessing(original, target_size, normalize)
    display_arr  = _denormalize_for_display(preprocessed.copy(), normalize)

    class_name = img_path.parent.name
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Class: {class_name}  |  File: {img_path.name}", fontsize=11, fontweight="bold")

    # --- Before ---
    axes[0].imshow(original.convert("RGB"))
    axes[0].set_title(
        f"BEFORE\n{original.width}×{original.height}  mode={original.mode}",
        fontsize=9,
    )
    axes[0].axis("off")

    # --- After ---
    axes[1].imshow(display_arr)
    axes[1].set_title(
        f"AFTER (preprocessed)\n"
        f"{target_size[0]}×{target_size[1]}  normalize={normalize}\n"
        f"pixel range [{preprocessed.min():.2f}, {preprocessed.max():.2f}]",
        fontsize=9,
    )
    axes[1].axis("off")

    plt.tight_layout()
    out_path = out_dir / class_name / (img_path.stem + "_before_after.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_grid(samples: list[tuple[Path, str]], out_dir: Path, target_size, normalize):
    """Save a single composite grid of all selected before/after samples."""
    n = len(samples)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]           # make iterable

    fig.suptitle("Before / After Preprocessing — Sample Grid", fontsize=13, fontweight="bold")

    for row, (img_path, _cls) in enumerate(samples):
        try:
            original = Image.open(img_path)
            original.load()
        except Exception:
            continue

        preprocessed = _apply_preprocessing(original, target_size, normalize)
        display_arr  = _denormalize_for_display(preprocessed.copy(), normalize)

        axes[row][0].imshow(original.convert("RGB"))
        axes[row][0].set_title(
            f"[BEFORE] {_cls}/{img_path.name}\n{original.width}×{original.height}", fontsize=8
        )
        axes[row][0].axis("off")

        axes[row][1].imshow(display_arr)
        axes[row][1].set_title(
            f"[AFTER] {target_size[0]}×{target_size[1]}  {normalize}\n"
            f"range [{preprocessed.min():.2f}, {preprocessed.max():.2f}]",
            fontsize=8,
        )
        axes[row][1].axis("off")

    plt.tight_layout()
    grid_path = out_dir / "sample_grid.png"
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Sample grid saved → {grid_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate before/after preprocessing visuals")
    parser.add_argument("--dataset",    default="data/processed",     help="Organised dataset dir")
    parser.add_argument("--out",        default="reports/before_after", help="Output directory")
    parser.add_argument("--size",       type=int, default=224,         help="Target resize dimension")
    parser.add_argument("--normalize",  default="imagenet",
                        choices=["minmax", "imagenet", "none"],        help="Normalisation strategy")
    parser.add_argument("--samples",    type=int, default=2,           help="Samples per class (default 2)")
    parser.add_argument("--seed",       type=int, default=42,          help="Random seed")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_dir     = Path(args.out)
    target_size = (args.size, args.size)

    if not dataset_dir.exists():
        sys.exit(f"[ERROR] Dataset directory not found: {dataset_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        sys.exit("[ERROR] No class subdirectories found.")

    all_samples: list[tuple[Path, str]] = []

    for cls_dir in class_dirs:
        images = [
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not images:
            continue
        selected = random.sample(images, min(args.samples, len(images)))
        for img_path in selected:
            save_before_after(img_path, out_dir, target_size, args.normalize)
            all_samples.append((img_path, cls_dir.name))

    save_grid(all_samples[:min(10, len(all_samples))], out_dir, target_size, args.normalize)

    print(f"[DONE] Individual figures → {out_dir.resolve()}")
    print(f"[INFO] Total figures generated: {len(all_samples)}")


if __name__ == "__main__":
    main()
