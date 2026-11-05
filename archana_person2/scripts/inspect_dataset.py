"""
Dataset Inspection Script - Person 2 (Archana Ulloju).

Audits official dataset or synthetic fallback, checks video-leakage guarantees,
computes class distribution statistics, and saves charts and JSON reports.
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure archana_person2 and workspace root are in Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from archana_person2.src.dataset_loader import Person2DatasetLoader


def inspect_dataset():
    loader = Person2DatasetLoader()
    metrics_dir = project_root / "archana_person2" / "artifacts" / "metrics"
    plots_dir = project_root / "archana_person2" / "artifacts" / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 1: DATASET INSPECTION & UNDERSTANDING")
    print("=" * 60)

    # Check if official split manifests exist
    official_train = project_root / "data" / "splits" / "train.csv"
    using_synthetic = not official_train.exists()

    if using_synthetic:
        print("[INFO] Official data/splits/train.csv not found on disk.")
        print("[INFO] Generating synthetic test fixture to verify pipeline integrity...")
        fixture_meta = loader.create_synthetic_fixture(num_classes=5, samples_per_class=30)
        print(f"[INFO] Synthetic fixture created: {fixture_meta['total_samples']} samples across {fixture_meta['num_classes']} classes.")

    # Load records for each split
    train_records = loader.load_split_records("train")
    val_records = loader.load_split_records("val")
    test_records = loader.load_split_records("test")

    label_map = loader.get_label_map()
    inv_label_map = loader.get_inverse_label_map()

    # Video leakage check
    train_vids = {r["video_id"] for r in train_records}
    val_vids = {r["video_id"] for r in val_records}
    test_vids = {r["video_id"] for r in test_records}

    train_val_leakage = train_vids.intersection(val_vids)
    train_test_leakage = train_vids.intersection(test_vids)
    val_test_leakage = val_vids.intersection(test_vids)

    leakage_safe = (len(train_val_leakage) == 0 and len(train_test_leakage) == 0 and len(val_test_leakage) == 0)

    # Class distributions
    train_counts = Counter(r["class_name"] for r in train_records)
    val_counts = Counter(r["class_name"] for r in val_records)
    test_counts = Counter(r["class_name"] for r in test_records)
    total_counts = Counter()
    for c in [train_counts, val_counts, test_counts]:
        total_counts.update(c)

    # Image dimension check on a sample
    image_sizes = []
    for r in train_records[:10]:
        img = loader.load_and_validate_image(r["image_path"])
        if img is not None:
            image_sizes.append(list(img.shape))

    stats = {
        "dataset_type": "synthetic_fixture" if using_synthetic else "official_dataset",
        "official_splits_present": not using_synthetic,
        "num_classes": len(label_map),
        "classes": list(label_map.keys()),
        "total_samples": len(train_records) + len(val_records) + len(test_records),
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "leakage_verification": {
            "train_videos_count": len(train_vids),
            "val_videos_count": len(val_vids),
            "test_videos_count": len(test_vids),
            "train_val_overlap": len(train_val_leakage),
            "train_test_overlap": len(train_test_leakage),
            "val_test_overlap": len(val_test_leakage),
            "is_leakage_safe": leakage_safe,
        },
        "class_distribution": {
            cls: {
                "train": train_counts.get(cls, 0),
                "val": val_counts.get(cls, 0),
                "test": test_counts.get(cls, 0),
                "total": total_counts.get(cls, 0),
            }
            for cls in label_map.keys()
        },
        "sample_image_dimensions": image_sizes[:3],
    }

    # Save metrics JSON
    stats_path = metrics_dir / "dataset_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[SAVED] Dataset statistics -> {stats_path}")

    # Generate Class Distribution Visualization
    classes = list(stats["class_distribution"].keys())
    tr_vals = [stats["class_distribution"][c]["train"] for c in classes]
    vl_vals = [stats["class_distribution"][c]["val"] for c in classes]
    ts_vals = [stats["class_distribution"][c]["test"] for c in classes]

    plt.figure(figsize=(10, 5))
    x = range(len(classes))
    width = 0.25

    plt.bar([i - width for i in x], tr_vals, width=width, label="Train", color="#1f77b4")
    plt.bar(x, vl_vals, width=width, label="Val", color="#ff7f0e")
    plt.bar([i + width for i in x], ts_vals, width=width, label="Test", color="#2ca02c")

    plt.xlabel("Aquatic Species Class", fontsize=11, fontweight="bold")
    plt.ylabel("Sample Count", fontsize=11, fontweight="bold")
    plt.title("Aquatic Biodiversity Dataset - Class Distribution per Split", fontsize=12, fontweight="bold")
    plt.xticks(x, classes, rotation=30, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plot_path = plots_dir / "class_distribution.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[SAVED] Class distribution chart -> {plot_path}")

    print("\n[VERIFICATION SUMMARY]")
    print(f"Total Samples: {stats['total_samples']} (Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)})")
    print(f"Num Classes: {stats['num_classes']}")
    print(f"Leakage Safe (video disjoint): {leakage_safe}")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    inspect_dataset()
