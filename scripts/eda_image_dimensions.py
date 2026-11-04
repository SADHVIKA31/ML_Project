"""
eda_image_dimensions.py
-----------------------
Phase 4 – Task 6: Analyse image dimensions and resolutions across the dataset.

Generates
---------
  reports/eda/dimension_scatter.png       – width vs height scatter
  reports/eda/dimension_histogram.png     – width + height histograms
  reports/eda/aspect_ratio_dist.png       – aspect-ratio distribution
  reports/eda/resolution_summary.json     – numeric summary

Usage
-----
  python scripts/eda_image_dimensions.py \\
      --dataset data/processed \\
      --out     reports/eda
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
BACKGROUND = "#0f0f1a"
TEXT_COLOR = "#ecf0f1"


def _apply_dark(fig, *axes):
    fig.patch.set_facecolor(BACKGROUND)
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")


def _collect_dimensions(dataset_dir: Path):
    widths, heights, classes = [], [], []
    for cls_dir in sorted(dataset_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in cls_dir.rglob("*"):
            if not (img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS):
                continue
            try:
                with Image.open(img_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
                    classes.append(cls_dir.name)
            except Exception:
                pass

    try:
        from tqdm import tqdm  # noqa
    except ImportError:
        pass

    return widths, heights, classes


# ---------------------------------------------------------------------------
# Scatter: width vs height
# ---------------------------------------------------------------------------

def plot_scatter(widths, heights, classes, out_path):
    unique_classes = sorted(set(classes))
    cmap = plt.get_cmap("tab20")
    color_map = {cls: cmap(i / max(len(unique_classes), 1)) for i, cls in enumerate(unique_classes)}
    colors = [color_map[c] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 7))
    _apply_dark(fig, ax)

    ax.scatter(widths, heights, c=colors, alpha=0.4, s=12, edgecolors="none")

    # Diagonal (square images)
    max_dim = max(max(widths), max(heights))
    ax.plot([0, max_dim], [0, max_dim], "--", color="#e74c3c", linewidth=1.2,
            label="Square (W=H)", alpha=0.8)

    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Dimensions: Width vs Height\n(each dot = one image)", fontsize=12,
                 fontweight="bold")

    # Annotate stats
    stats_text = (
        f"N={len(widths)}\n"
        f"W: [{min(widths)}, {max(widths)}]\n"
        f"H: [{min(heights)}, {max(heights)}]"
    )
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=8,
            color=TEXT_COLOR, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", edgecolor="#444", alpha=0.8))

    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor=TEXT_COLOR, fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[DONE] Scatter → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Histograms: width and height
# ---------------------------------------------------------------------------

def plot_histograms(widths, heights, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _apply_dark(fig, ax1, ax2)

    bins = min(50, len(set(widths)))

    ax1.hist(widths, bins=bins, color="#3498db", edgecolor="none", alpha=0.85)
    ax1.axvline(np.mean(widths), color="#e74c3c", linewidth=1.5, linestyle="--",
                label=f"Mean = {np.mean(widths):.0f}px")
    ax1.axvline(np.median(widths), color="#2ecc71", linewidth=1.5, linestyle="-.",
                label=f"Median = {np.median(widths):.0f}px")
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Count")
    ax1.set_title("Width Distribution")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor=TEXT_COLOR, fontsize=8)

    ax2.hist(heights, bins=bins, color="#9b59b6", edgecolor="none", alpha=0.85)
    ax2.axvline(np.mean(heights), color="#e74c3c", linewidth=1.5, linestyle="--",
                label=f"Mean = {np.mean(heights):.0f}px")
    ax2.axvline(np.median(heights), color="#2ecc71", linewidth=1.5, linestyle="-.",
                label=f"Median = {np.median(heights):.0f}px")
    ax2.set_xlabel("Height (px)")
    ax2.set_ylabel("Count")
    ax2.set_title("Height Distribution")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor=TEXT_COLOR, fontsize=8)

    fig.suptitle("Image Dimension Distributions", fontsize=13, fontweight="bold",
                 color=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[DONE] Histograms → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Aspect ratio distribution
# ---------------------------------------------------------------------------

def plot_aspect_ratio(widths, heights, out_path):
    ratios = [w / h for w, h in zip(widths, heights) if h > 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark(fig, ax)

    ax.hist(ratios, bins=min(60, len(set(ratios))), color="#e67e22", edgecolor="none", alpha=0.85)
    ax.axvline(1.0,  color="#2ecc71",  linewidth=1.5, linestyle="--", label="Square (1:1)")
    ax.axvline(4/3,  color="#3498db",  linewidth=1.5, linestyle="--", label="4:3")
    ax.axvline(16/9, color="#e74c3c",  linewidth=1.5, linestyle="--", label="16:9")
    ax.axvline(np.mean(ratios), color="#f39c12", linewidth=1.5, linestyle="-.",
               label=f"Mean = {np.mean(ratios):.2f}")

    ax.set_xlabel("Aspect Ratio (Width / Height)")
    ax.set_ylabel("Count")
    ax.set_title("Aspect Ratio Distribution", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor=TEXT_COLOR, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[DONE] Aspect ratio → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse image dimensions and resolutions")
    parser.add_argument("--dataset", default="data/processed", help="Organised dataset dir")
    parser.add_argument("--out",     default="reports/eda",    help="Output directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_dir     = Path(args.out)

    if not dataset_dir.exists():
        sys.exit(f"[ERROR] Dataset directory not found: {dataset_dir}")

    print("[INFO] Collecting image dimensions ...")
    widths, heights, classes = _collect_dimensions(dataset_dir)

    if not widths:
        sys.exit("[ERROR] No readable images found.")

    ratios = [w / h for w, h in zip(widths, heights) if h > 0]

    # Numeric summary
    summary = {
        "total_images":     len(widths),
        "width":            {"min": min(widths),  "max": max(widths),  "mean": round(float(np.mean(widths)), 2),  "median": float(np.median(widths)),  "std": round(float(np.std(widths)), 2)},
        "height":           {"min": min(heights), "max": max(heights), "mean": round(float(np.mean(heights)), 2), "median": float(np.median(heights)), "std": round(float(np.std(heights)), 2)},
        "aspect_ratio":     {"min": round(min(ratios), 3), "max": round(max(ratios), 3), "mean": round(float(np.mean(ratios)), 3)},
        "unique_dimensions": len(set(zip(widths, heights))),
        "all_same_size":    len(set(zip(widths, heights))) == 1,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "resolution_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_scatter(widths, heights, classes, out_dir / "dimension_scatter.png")
    plot_histograms(widths, heights, out_dir / "dimension_histogram.png")
    plot_aspect_ratio(widths, heights, out_dir / "aspect_ratio_dist.png")

    print(f"\n[SUMMARY]")
    print(f"  Images          : {summary['total_images']}")
    print(f"  Unique dims     : {summary['unique_dimensions']}")
    print(f"  Width range     : [{summary['width']['min']}, {summary['width']['max']}]")
    print(f"  Height range    : [{summary['height']['min']}, {summary['height']['max']}]")
    print(f"  Aspect ratio    : mean={summary['aspect_ratio']['mean']}")
    print(f"[DONE] JSON → {summary_path.resolve()}")


if __name__ == "__main__":
    main()
