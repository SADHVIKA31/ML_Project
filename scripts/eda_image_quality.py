"""
eda_image_quality.py
--------------------
Phase 4 – Tasks 8, 9, 10: Examine image quality, detect illumination problems,
turbidity, blur, colour distortion, and occlusion indicators.

Metrics computed per image
--------------------------
  - Brightness (mean pixel intensity)       → illumination analysis
  - Contrast   (std of pixel intensities)   → contrast analysis
  - Blur score (Laplacian variance)         → sharpness / blur detection
  - Saturation (mean HSV S channel)         → colour distortion proxy
  - Dark/bright fraction                    → over/under exposure flags

Thresholds (configurable)
--------------------------
  BLUR_THRESHOLD        : images with Laplacian variance < threshold are blurry
  DARK_THRESHOLD        : mean brightness < threshold → dark / under-exposed
  BRIGHT_THRESHOLD      : mean brightness > threshold → over-exposed
  LOW_CONTRAST_THRESHOLD: contrast std < threshold    → low contrast

Outputs
-------
  reports/eda/image_quality_analysis.csv
  reports/eda/image_quality_analysis.json
  reports/eda/quality_brightness_hist.png
  reports/eda/quality_blur_hist.png
  reports/eda/quality_flags_summary.png

Usage
-----
  python scripts/eda_image_quality.py \\
      --dataset data/processed \\
      --out     reports/eda \\
      --max-images 1000     # optional cap for speed
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageStat

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Quality thresholds
BLUR_THRESHOLD         = 100.0   # Laplacian variance below this → blurry
DARK_THRESHOLD         = 50.0    # mean brightness 0-255 below this → dark
BRIGHT_THRESHOLD       = 220.0   # mean brightness above this → overexposed
LOW_CONTRAST_THRESHOLD = 20.0    # std of brightness below this → low contrast

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


# ---------------------------------------------------------------------------
# Per-image quality metrics
# ---------------------------------------------------------------------------

def _laplacian_variance(img_array: np.ndarray) -> float:
    """Blur detection: high variance = sharp, low variance = blurry."""
    gray = np.mean(img_array, axis=2) if img_array.ndim == 3 else img_array
    # Approximate Laplacian with simple kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.ndimage import convolve
    laplacian = convolve(gray.astype(np.float32), kernel)
    return float(np.var(laplacian))


def _mean_saturation(img: Image.Image) -> float:
    """Mean HSV saturation – low saturation = colour-desaturated / turbid."""
    try:
        hsv = img.convert("HSV")
        s_channel = np.array(hsv)[:, :, 1]
        return float(np.mean(s_channel))
    except Exception:
        return 0.0


def analyse_image(path: Path) -> dict:
    record = {
        "path":        str(path),
        "filename":    path.name,
        "class":       path.parent.name,
        "width":       None,
        "height":      None,
        "brightness":  None,
        "contrast":    None,
        "blur_score":  None,
        "saturation":  None,
        "is_dark":     False,
        "is_bright":   False,
        "is_blurry":   False,
        "is_low_contrast": False,
        "readable":    True,
        "error":       None,
    }
    try:
        img = Image.open(path).convert("RGB")
        img.load()
        arr = np.array(img, dtype=np.float32)
        gray = np.mean(arr, axis=2)

        record["width"]      = img.width
        record["height"]     = img.height
        record["brightness"] = round(float(np.mean(gray)), 2)
        record["contrast"]   = round(float(np.std(gray)), 2)
        record["saturation"] = round(_mean_saturation(img), 2)

        try:
            record["blur_score"] = round(_laplacian_variance(arr), 2)
        except Exception:
            record["blur_score"] = None

        record["is_dark"]         = record["brightness"] is not None and record["brightness"] < DARK_THRESHOLD
        record["is_bright"]       = record["brightness"] is not None and record["brightness"] > BRIGHT_THRESHOLD
        record["is_blurry"]       = record["blur_score"] is not None and record["blur_score"] < BLUR_THRESHOLD
        record["is_low_contrast"] = record["contrast"]   is not None and record["contrast"]   < LOW_CONTRAST_THRESHOLD

    except Exception as e:
        record["readable"] = False
        record["error"]    = str(e)

    return record


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_hist(values, title, xlabel, out_path, threshold=None, threshold_label="", color="#3498db"):
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark(fig, ax)
    ax.hist(values, bins=60, color=color, edgecolor="none", alpha=0.85)
    if threshold is not None:
        ax.axvline(threshold, color="#e74c3c", linewidth=1.8, linestyle="--",
                   label=f"{threshold_label} = {threshold}")
    ax.axvline(np.mean(values), color="#2ecc71", linewidth=1.5, linestyle="-.",
               label=f"Mean = {np.mean(values):.1f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Image Count")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor=TEXT_COLOR, fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[DONE] {out_path.name} → {out_path.resolve()}")


def _plot_flags_summary(flags: dict, out_path: Path, total: int):
    labels = list(flags.keys())
    counts = list(flags.values())
    pcts   = [c / total * 100 for c in counts]
    colors = ["#e74c3c", "#e67e22", "#9b59b6", "#3498db"]

    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_dark(fig, ax)

    bars = ax.bar(labels, counts, color=colors[:len(labels)], edgecolor="none", width=0.5)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, color=TEXT_COLOR)

    ax.set_ylabel("Number of Images")
    ax.set_title(
        f"Image Quality Issues — {total} images total",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"[DONE] Quality flags → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse image quality metrics")
    parser.add_argument("--dataset",    default="data/processed", help="Organised dataset dir")
    parser.add_argument("--out",        default="reports/eda",    help="Output directory")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Cap number of images analysed (0 = all)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_dir     = Path(args.out)

    if not dataset_dir.exists():
        sys.exit(f"[ERROR] Dataset directory not found: {dataset_dir}")

    # Collect image paths
    all_paths = [
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if args.max_images and args.max_images < len(all_paths):
        import random; random.seed(42)
        all_paths = random.sample(all_paths, args.max_images)
        print(f"[INFO] Capped to {len(all_paths)} images (--max-images {args.max_images})")

    if not all_paths:
        sys.exit("[ERROR] No images found.")

    try:
        from tqdm import tqdm
        it = tqdm(all_paths, desc="Analysing quality")
    except ImportError:
        it = all_paths

    print(f"[INFO] Analysing {len(all_paths)} images ...")
    records = [analyse_image(p) for p in it]

    readable = [r for r in records if r["readable"]]

    # Save CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "image_quality_analysis.csv"
    fieldnames = ["path", "filename", "class", "width", "height",
                  "brightness", "contrast", "blur_score", "saturation",
                  "is_dark", "is_bright", "is_blurry", "is_low_contrast",
                  "readable", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Flags
    flags = {
        "Dark / Under-exposed":  sum(1 for r in readable if r["is_dark"]),
        "Over-exposed":          sum(1 for r in readable if r["is_bright"]),
        "Blurry":                sum(1 for r in readable if r["is_blurry"]),
        "Low Contrast":          sum(1 for r in readable if r["is_low_contrast"]),
    }

    # Numeric summary
    brightness_vals = [r["brightness"] for r in readable if r["brightness"] is not None]
    blur_vals       = [r["blur_score"] for r in readable if r["blur_score"] is not None]
    sat_vals        = [r["saturation"] for r in readable if r["saturation"] is not None]

    summary = {
        "total_analysed":   len(records),
        "readable":         len(readable),
        "unreadable":       len(records) - len(readable),
        "quality_flags":    flags,
        "flag_percentages": {k: round(v / len(readable) * 100, 2) if readable else 0
                             for k, v in flags.items()},
        "brightness": {
            "mean":   round(float(np.mean(brightness_vals)), 2) if brightness_vals else None,
            "std":    round(float(np.std(brightness_vals)), 2) if brightness_vals else None,
            "min":    round(min(brightness_vals), 2) if brightness_vals else None,
            "max":    round(max(brightness_vals), 2) if brightness_vals else None,
            "threshold_dark":   DARK_THRESHOLD,
            "threshold_bright": BRIGHT_THRESHOLD,
        },
        "blur": {
            "mean":      round(float(np.mean(blur_vals)), 2) if blur_vals else None,
            "std":       round(float(np.std(blur_vals)), 2) if blur_vals else None,
            "threshold": BLUR_THRESHOLD,
        },
        "saturation": {
            "mean": round(float(np.mean(sat_vals)), 2) if sat_vals else None,
            "std":  round(float(np.std(sat_vals)), 2) if sat_vals else None,
        },
        "thresholds_used": {
            "BLUR_THRESHOLD":          BLUR_THRESHOLD,
            "DARK_THRESHOLD":          DARK_THRESHOLD,
            "BRIGHT_THRESHOLD":        BRIGHT_THRESHOLD,
            "LOW_CONTRAST_THRESHOLD":  LOW_CONTRAST_THRESHOLD,
        },
    }

    json_path = out_dir / "image_quality_analysis.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    if brightness_vals:
        _plot_hist(brightness_vals,
                   "Brightness Distribution (Mean Pixel Intensity 0-255)",
                   "Mean Brightness", out_dir / "quality_brightness_hist.png",
                   threshold=DARK_THRESHOLD, threshold_label="Dark threshold",
                   color="#f39c12")
    if blur_vals:
        _plot_hist(blur_vals,
                   "Blur Score Distribution (Laplacian Variance)\nLow = Blurry",
                   "Blur Score (Laplacian Variance)",
                   out_dir / "quality_blur_hist.png",
                   threshold=BLUR_THRESHOLD, threshold_label="Blur threshold",
                   color="#9b59b6")

    _plot_flags_summary(flags, out_dir / "quality_flags_summary.png", len(readable))

    # Console summary
    print("\n" + "=" * 60)
    print("  IMAGE QUALITY ANALYSIS SUMMARY")
    print("=" * 60)
    for flag, count in flags.items():
        pct = count / len(readable) * 100 if readable else 0
        icon = "⚠️ " if pct > 10 else "✅ "
        print(f"  {icon}{flag:<30}: {count:5d}  ({pct:.1f}%)")
    print("=" * 60)
    print(f"\n[DONE] CSV  → {csv_path.resolve()}")
    print(f"[DONE] JSON → {json_path.resolve()}")


if __name__ == "__main__":
    main()
