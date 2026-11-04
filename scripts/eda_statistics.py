"""
eda_statistics.py
-----------------
Phase 4 – Tasks 1, 2, 5: Dataset statistics, per-species image counts,
and class-imbalance metrics.

Outputs
-------
  reports/eda/dataset_statistics.json
  reports/eda/dataset_statistics.md

Usage
-----
  python scripts/eda_statistics.py --dataset data/processed --out reports/eda
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_images(dataset_dir: Path) -> list[tuple[str, Path]]:
    """Return [(class_name, path), ...] for every image found."""
    records = []
    for cls_dir in sorted(dataset_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in cls_dir.rglob("*"):
            if img.is_file() and img.suffix.lower() in SUPPORTED_EXTENSIONS:
                records.append((cls_dir.name, img))
    return records


def _imbalance_metrics(counts: dict[str, int]) -> dict:
    """Compute class-imbalance statistics."""
    values = list(counts.values())
    if not values:
        return {}

    total      = sum(values)
    n_classes  = len(values)
    max_count  = max(values)
    min_count  = min(values)
    mean_count = total / n_classes

    # Imbalance ratio
    ir = max_count / max(min_count, 1)

    # Gini coefficient (measure of inequality, 0 = perfect balance, 1 = max imbalance)
    sorted_vals = sorted(values)
    gini_num = sum((2 * i - n_classes - 1) * v for i, v in enumerate(sorted_vals, 1))
    gini = abs(gini_num) / (n_classes * total) if total else 0

    # Coefficient of variation
    variance = sum((v - mean_count) ** 2 for v in values) / n_classes
    std = variance ** 0.5
    cv = std / mean_count if mean_count else 0

    # Majority / minority classes (top-3 and bottom-3)
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    majority = [{"class": k, "count": v} for k, v in sorted_items[:3]]
    minority = [{"class": k, "count": v} for k, v in sorted_items[-3:]]

    # Severity label
    if ir > 20:
        severity = "SEVERE (>20x)"
    elif ir > 10:
        severity = "HIGH (10-20x)"
    elif ir > 5:
        severity = "MODERATE (5-10x)"
    elif ir > 2:
        severity = "MILD (2-5x)"
    else:
        severity = "BALANCED (<2x)"

    return {
        "total_images":       total,
        "num_classes":        n_classes,
        "max_per_class":      max_count,
        "min_per_class":      min_count,
        "mean_per_class":     round(mean_count, 2),
        "std_per_class":      round(std, 2),
        "imbalance_ratio":    round(ir, 2),
        "gini_coefficient":   round(gini, 4),
        "coeff_of_variation": round(cv, 4),
        "imbalance_severity": severity,
        "majority_classes":   majority,
        "minority_classes":   minority,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_statistics(dataset_dir: Path, out_dir: Path):
    records = _collect_images(dataset_dir)
    if not records:
        sys.exit(f"[ERROR] No images found under {dataset_dir}")

    # Per-class counts
    counts: dict[str, int] = Counter(cls for cls, _ in records)
    sorted_counts = dict(sorted(counts.items(), key=lambda x: -x[1]))

    imbalance = _imbalance_metrics(counts)

    # Per-class percentage
    total = imbalance["total_images"]
    per_class_detail = {
        cls: {
            "count":      cnt,
            "percentage": round(cnt / total * 100, 2),
            "rank":       rank,
        }
        for rank, (cls, cnt) in enumerate(sorted(counts.items(), key=lambda x: -x[1]), 1)
    }

    report = {
        "dataset_path":    str(dataset_dir.resolve()),
        "summary":         imbalance,
        "per_class":       per_class_detail,
        "per_class_counts": sorted_counts,
    }

    # --- Save JSON ---
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dataset_statistics.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- Save Markdown ---
    md_lines = [
        "# Dataset Statistics Report",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Images | **{imbalance['total_images']}** |",
        f"| Total Classes | **{imbalance['num_classes']}** |",
        f"| Max Images / Class | {imbalance['max_per_class']} |",
        f"| Min Images / Class | {imbalance['min_per_class']} |",
        f"| Mean Images / Class | {imbalance['mean_per_class']} |",
        f"| Std Dev | {imbalance['std_per_class']} |",
        f"| Imbalance Ratio (max/min) | **{imbalance['imbalance_ratio']}x** |",
        f"| Gini Coefficient | {imbalance['gini_coefficient']} |",
        f"| Coefficient of Variation | {imbalance['coeff_of_variation']} |",
        f"| Imbalance Severity | **{imbalance['imbalance_severity']}** |",
        "",
        "## Class Distribution (Ranked)",
        "",
        "| Rank | Class | Count | % of Total |",
        "|------|-------|-------|------------|",
    ]

    for cls, detail in sorted(per_class_detail.items(), key=lambda x: x[1]["rank"]):
        md_lines.append(
            f"| {detail['rank']} | {cls} | {detail['count']} | {detail['percentage']}% |"
        )

    md_lines += [
        "",
        "## Class Imbalance Analysis",
        "",
        f"**Majority Classes (Top 3):**",
    ]
    for item in imbalance["majority_classes"]:
        md_lines.append(f"- {item['class']}: {item['count']} images")

    md_lines += ["", f"**Minority Classes (Bottom 3):**"]
    for item in imbalance["minority_classes"]:
        md_lines.append(f"- {item['class']}: {item['count']} images")

    sev = imbalance["imbalance_severity"]
    if "SEVERE" in sev or "HIGH" in sev:
        md_lines += [
            "",
            "> ⚠️ **Significant class imbalance detected.**",
            "> Recommended mitigations:",
            "> - Weighted cross-entropy loss",
            "> - Oversampling minority classes (SMOTE / augmentation)",
            "> - Undersampling majority classes",
        ]
    elif "MODERATE" in sev:
        md_lines += ["", "> ⚡ Moderate imbalance — consider class-weighted training."]
    else:
        md_lines += ["", "> ✅ Dataset is reasonably balanced."]

    md_lines += ["", "---", "*Auto-generated by `eda_statistics.py`*"]

    md_path = out_dir / "dataset_statistics.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # --- Console summary ---
    print("\n" + "=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total images      : {imbalance['total_images']}")
    print(f"  Total classes     : {imbalance['num_classes']}")
    print(f"  Imbalance ratio   : {imbalance['imbalance_ratio']}x  ({imbalance['imbalance_severity']})")
    print(f"  Gini coefficient  : {imbalance['gini_coefficient']}")
    print("=" * 60)
    print(f"\n[DONE] JSON → {json_path.resolve()}")
    print(f"[DONE] MD   → {md_path.resolve()}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Compute EDA statistics and class distribution")
    parser.add_argument("--dataset", default="data/processed", help="Organised dataset dir")
    parser.add_argument("--out",     default="reports/eda",    help="Output directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        sys.exit(f"[ERROR] Dataset directory not found: {dataset_dir}")

    compute_statistics(dataset_dir, Path(args.out))


if __name__ == "__main__":
    main()
