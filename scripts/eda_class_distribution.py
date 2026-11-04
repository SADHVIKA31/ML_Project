"""
eda_class_distribution.py
--------------------------
Phase 4 – Tasks 3 & 4: Visualise class distribution and identify
majority / minority classes.

Generates
---------
  reports/eda/class_distribution_bar.png    – horizontal bar chart
  reports/eda/class_distribution_pie.png    – pie chart (top-N + Others)
  reports/eda/class_imbalance_heatmap.png   – imbalance severity heatmap

Usage
-----
  python scripts/eda_class_distribution.py \\
      --dataset data/processed \\
      --out     reports/eda \\
      --top-n   15
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Colour palette
MAJORITY_COLOR  = "#2ecc71"   # green
MINORITY_COLOR  = "#e74c3c"   # red
NEUTRAL_COLOR   = "#3498db"   # blue
BACKGROUND      = "#0f0f1a"   # dark
TEXT_COLOR      = "#ecf0f1"


def _count_images(dataset_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = Counter()
    for cls_dir in dataset_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        n = sum(
            1 for p in cls_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        counts[cls_dir.name] = n
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _apply_dark_style(fig, ax):
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")


# ---------------------------------------------------------------------------
# Chart 1 – Horizontal bar chart
# ---------------------------------------------------------------------------

def plot_bar_chart(counts: dict[str, int], out_path: Path, top_n: int = 30):
    items    = list(counts.items())[:top_n]
    classes  = [k for k, _ in items]
    values   = [v for _, v in items]
    total    = sum(counts.values())
    n        = len(items)

    # Colour: top 3 green, bottom 3 red, rest blue
    colors = []
    for i in range(n):
        if i < 3:
            colors.append(MAJORITY_COLOR)
        elif i >= n - 3:
            colors.append(MINORITY_COLOR)
        else:
            colors.append(NEUTRAL_COLOR)

    fig_height = max(8, n * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    _apply_dark_style(fig, ax)

    bars = ax.barh(range(n), values, color=colors, edgecolor="none", height=0.65)

    # Annotate count + percentage
    for i, (bar, val) in enumerate(zip(bars, values)):
        pct = val / total * 100
        ax.text(val + max(values) * 0.005, i, f"{val}  ({pct:.1f}%)",
                va="center", ha="left", fontsize=8, color=TEXT_COLOR)

    ax.set_yticks(range(n))
    ax.set_yticklabels(classes, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Images", fontsize=10)
    ax.set_title(
        f"Class Distribution — Top {n} Species\n"
        f"Total: {total} images  |  {len(counts)} classes",
        fontsize=12, fontweight="bold", pad=15,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=MAJORITY_COLOR, label="Majority (Top 3)"),
        Patch(facecolor=NEUTRAL_COLOR,  label="Normal"),
        Patch(facecolor=MINORITY_COLOR, label="Minority (Bottom 3)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              facecolor="#1a1a2e", edgecolor="#444", labelcolor=TEXT_COLOR,
              fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[DONE] Bar chart → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Chart 2 – Pie chart (top-N + Others)
# ---------------------------------------------------------------------------

def plot_pie_chart(counts: dict[str, int], out_path: Path, top_n: int = 10):
    items = list(counts.items())
    top   = items[:top_n]
    rest  = items[top_n:]

    labels = [k for k, _ in top]
    sizes  = [v for _, v in top]

    if rest:
        labels.append("Others")
        sizes.append(sum(v for _, v in rest))

    cmap   = plt.get_cmap("tab20")
    colors = [cmap(i / len(labels)) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(10, 8))
    _apply_dark_style(fig, ax)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%",
        colors=colors, startangle=140,
        wedgeprops=dict(edgecolor="#0f0f1a", linewidth=1.5),
        pctdistance=0.82,
    )
    for at in autotexts:
        at.set_color(TEXT_COLOR)
        at.set_fontsize(8)

    ax.legend(
        wedges, labels,
        title="Species", title_fontsize=9,
        loc="center left", bbox_to_anchor=(1, 0.5),
        facecolor="#1a1a2e", edgecolor="#444",
        labelcolor=TEXT_COLOR, fontsize=8,
    )

    total = sum(sizes)
    ax.set_title(
        f"Class Distribution (Pie)\nTop {top_n} species + Others  |  Total: {total}",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=20,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[DONE] Pie chart → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Chart 3 – Imbalance severity heatmap
# ---------------------------------------------------------------------------

def plot_imbalance_heatmap(counts: dict[str, int], out_path: Path):
    if not counts:
        return

    classes = list(counts.keys())
    values  = list(counts.values())
    total   = sum(values)
    mean    = total / len(values)

    # Imbalance score per class relative to mean
    scores  = [(v - mean) / mean for v in values]

    # Reshape to roughly square grid
    n       = len(classes)
    ncols   = min(10, n)
    nrows   = (n + ncols - 1) // ncols
    # Pad
    pad     = nrows * ncols - n
    scores_padded  = scores  + [0] * pad
    classes_padded = classes + [""] * pad

    grid        = np.array(scores_padded).reshape(nrows, ncols)
    label_grid  = np.array(classes_padded).reshape(nrows, ncols)

    fig, ax = plt.subplots(figsize=(max(10, ncols * 1.4), max(5, nrows * 1.2)))
    _apply_dark_style(fig, ax)

    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Relative deviation from mean", fraction=0.03)

    ax.set_xticks([])
    ax.set_yticks([])

    for r in range(nrows):
        for c in range(ncols):
            lbl = label_grid[r, c]
            if lbl:
                ax.text(c, r, lbl, ha="center", va="center",
                        fontsize=max(5, 9 - nrows // 2), color="black",
                        fontweight="bold", wrap=True)

    ax.set_title(
        "Class Imbalance Heatmap\nGreen = over-represented  |  Red = under-represented",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=15,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[DONE] Heatmap → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise class distribution")
    parser.add_argument("--dataset", default="data/processed", help="Organised dataset dir")
    parser.add_argument("--out",     default="reports/eda",    help="Output directory")
    parser.add_argument("--top-n",   type=int, default=20,     help="Max classes to show in bar chart")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_dir     = Path(args.out)

    if not dataset_dir.exists():
        sys.exit(f"[ERROR] Dataset directory not found: {dataset_dir}")

    counts = _count_images(dataset_dir)
    if not counts:
        sys.exit("[ERROR] No images found.")

    plot_bar_chart(counts, out_dir / "class_distribution_bar.png",    top_n=args.top_n)
    plot_pie_chart(counts, out_dir / "class_distribution_pie.png",    top_n=10)
    plot_imbalance_heatmap(counts, out_dir / "class_imbalance_heatmap.png")

    print(f"\n[INFO] All class-distribution charts saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
