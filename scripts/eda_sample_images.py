"""
eda_sample_images.py
--------------------
Phase 4 – Task 7: Display sample images from each species in a grid.

Generates
---------
  reports/eda/sample_images_grid.png     – N samples per class, all classes
  reports/eda/sample_per_class/          – one grid PNG per class (more samples)

Usage
-----
  python scripts/eda_sample_images.py \\
      --dataset  data/processed \\
      --out      reports/eda \\
      --samples  4         # images per class in the main grid
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
BACKGROUND = "#0f0f1a"
TEXT_COLOR = "#ecf0f1"


def _pick_samples(cls_dir: Path, n: int, seed: int) -> list[Path]:
    images = [
        p for p in cls_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    random.seed(seed)
    return random.sample(images, min(n, len(images)))


def _safe_open(path: Path):
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main sample grid (all classes × N samples)
# ---------------------------------------------------------------------------

def plot_main_grid(dataset_dir: Path, out_path: Path, samples_per_class: int, seed: int):
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        return

    n_classes = len(class_dirs)
    n_cols    = samples_per_class
    n_rows    = n_classes

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.2, n_rows * 2.2 + 0.5),
    )
    fig.patch.set_facecolor(BACKGROUND)

    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for row, cls_dir in enumerate(class_dirs):
        samples = _pick_samples(cls_dir, samples_per_class, seed)
        for col in range(n_cols):
            ax = axes[row][col]
            ax.set_facecolor(BACKGROUND)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col < len(samples):
                img = _safe_open(samples[col])
                if img:
                    ax.imshow(img)
                    if col == 0:
                        ax.set_ylabel(cls_dir.name, fontsize=7, color=TEXT_COLOR,
                                      rotation=0, labelpad=70, va="center")
            else:
                ax.set_visible(False)

    fig.suptitle(
        f"Sample Images — {n_classes} Species  |  {samples_per_class} samples each",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )
    plt.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[DONE] Sample grid → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Per-class individual grids (more images per class)
# ---------------------------------------------------------------------------

def plot_per_class_grids(dataset_dir: Path, out_dir: Path, samples: int, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    for cls_dir in sorted(dataset_dir.iterdir()):
        if not cls_dir.is_dir():
            continue

        picked = _pick_samples(cls_dir, samples, seed)
        if not picked:
            continue

        n_cols = min(4, len(picked))
        n_rows = (len(picked) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        fig.patch.set_facecolor(BACKGROUND)

        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for i, ax in enumerate(axes_flat):
            ax.set_facecolor(BACKGROUND)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if i < len(picked):
                img = _safe_open(picked[i])
                if img:
                    ax.imshow(img)
                    ax.set_title(picked[i].name, fontsize=6, color=TEXT_COLOR, pad=2)
            else:
                ax.set_visible(False)

        fig.suptitle(f"Class: {cls_dir.name}  ({len(picked)} samples)",
                     fontsize=11, fontweight="bold", color=TEXT_COLOR)
        plt.tight_layout(pad=0.5)
        plt.savefig(out_dir / f"{cls_dir.name}_samples.png",
                    dpi=120, bbox_inches="tight", facecolor=BACKGROUND)
        plt.close(fig)

    print(f"[DONE] Per-class grids → {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Display sample images from each species")
    parser.add_argument("--dataset",  default="data/processed",   help="Organised dataset dir")
    parser.add_argument("--out",      default="reports/eda",       help="Output directory")
    parser.add_argument("--samples",  type=int, default=4,         help="Samples per class (default 4)")
    parser.add_argument("--seed",     type=int, default=42,        help="Random seed")
    parser.add_argument("--per-class-samples", type=int, default=8, help="Samples for per-class grids")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_dir     = Path(args.out)

    if not dataset_dir.exists():
        sys.exit(f"[ERROR] Dataset directory not found: {dataset_dir}")

    plot_main_grid(dataset_dir, out_dir / "sample_images_grid.png",
                   args.samples, args.seed)
    plot_per_class_grids(dataset_dir, out_dir / "sample_per_class",
                         args.per_class_samples, args.seed)


if __name__ == "__main__":
    main()
