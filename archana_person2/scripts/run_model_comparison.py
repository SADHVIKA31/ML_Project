"""
Model Comparison and Visualization Script - Person 2 (Archana Ulloju).

Runs the full 9-way benchmark:
3 Classical Models (Logistic Regression, Random Forest, SVM)
X 3 Feature Modalities (Color Histogram, HOG, Combined)

Generates:
- model_comparison.json
- model_comparison.csv
- best_baseline.json
- High-resolution comparative plots
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from archana_person2.src.comparison import ModelComparator


def plot_performance_comparison(records: List[Dict[str, Any]], save_path: Path):
    """Grouped bar chart for all models using the combined feature representation."""
    combined_records = [r for r in records if r["representation"] == "combined"]
    metric_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    metric_labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro-F1 (Primary)", "Weighted-F1"]

    x = np.arange(len(metric_labels))
    width = 0.25
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (rec, color) in enumerate(zip(combined_records, colors)):
        vals = [rec[k] for k in metric_keys]
        rects = ax.bar(x + (idx - 1) * width, vals, width, label=rec["model"], color=color, alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Classical ML Performance Comparison (Combined Representation)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.20)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_across_representations(
    records: List[Dict[str, Any]],
    metric_key: str,
    metric_name: str,
    save_path: Path,
):
    """Bar chart comparing a single metric across models and feature representations."""
    models = ["Logistic Regression", "Random Forest", "Support Vector Machine (SVM)"]
    reps = ["color_hist", "hog", "combined"]
    rep_labels = ["Color Histogram", "HOG Features", "Combined (Color+HOG)"]

    x = np.arange(len(models))
    width = 0.25
    colors = ["#3498db", "#9b59b6", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (rep, rep_label, color) in enumerate(zip(reps, rep_labels, colors)):
        vals = []
        for m in models:
            match = next((r for r in records if r["model"] == m and r["representation"] == rep), None)
            vals.append(match[metric_key] if match else 0.0)

        rects = ax.bar(x + (idx - 1) * width, vals, width, label=rep_label, color=color, alpha=0.9)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8,
            )

    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f"{metric_name} Across Feature Representations", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.20)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_computational_latency(records: List[Dict[str, Any]], save_path: Path):
    """Plots training time and inference latency across models."""
    combined_records = [r for r in records if r["representation"] == "combined"]
    models = [r["model"] for r in combined_records]
    train_times = [r["train_time_sec"] for r in combined_records]
    infer_times = [r["infer_time_ms"] for r in combined_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training Time
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    bars1 = ax1.bar(models, train_times, color=colors, alpha=0.85, width=0.5)
    ax1.set_ylabel("Training Time (seconds)", fontsize=11)
    ax1.set_title("Model Training Latency (Combined)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.2f}s", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # Inference Latency
    bars2 = ax2.bar(models, infer_times, color=colors, alpha=0.85, width=0.5)
    ax2.set_ylabel("Test Set Inference Time (ms)", fontsize=11)
    ax2.set_title("Model Inference Latency (50 test samples)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}ms", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("PHASE 7: MODEL COMPARISON & BENCHMARKING (9 CONFIGURATIONS)")
    print("=" * 70)

    metrics_dir = project_root / "archana_person2" / "artifacts" / "metrics"
    plots_dir = project_root / "archana_person2" / "artifacts" / "plots"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    comparator = ModelComparator()

    print("\n[1/4] Running 9-way benchmark across models and representations...")
    comparator.run_full_comparison()

    print("\n[2/4] Saving benchmark comparison artifacts...")
    json_path, csv_path, best_path = comparator.save_benchmark_artifacts(metrics_dir)
    print(f"      Full comparison JSON: {json_path}")
    print(f"      Tabular summary CSV : {csv_path}")
    print(f"      Best baseline JSON  : {best_path}")

    # Display console comparison table
    records = comparator.get_comparison_table_records()
    print("\n" + "-" * 95)
    print(f"{'Model':<28} | {'Representation':<12} | {'Accuracy':<8} | {'Macro-F1':<8} | {'Weighted-F1':<11} | {'Train(s)':<8}")
    print("-" * 95)
    for r in records:
        print(f"{r['model']:<28} | {r['representation']:<12} | {r['accuracy']:<8.4f} | {r['macro_f1']:<8.4f} | {r['weighted_f1']:<11.4f} | {r['train_time_sec']:<8.2f}")
    print("-" * 95)

    best = comparator.best_baseline_
    print(f"\n[BEST BASELINE IDENTIFIED]: {best['model_name']} ({best['representation']})")
    print(f"      Macro-F1: {best['macro_f1']:.4f} | Accuracy: {best['accuracy']:.4f}")
    print(f"      Rationale: {best['selection_rationale']}")

    print("\n[3/4] Generating comparative visualizations...")
    plot_performance_comparison(records, plots_dir / "model_performance_comparison.png")
    print(f"      Saved -> {plots_dir / 'model_performance_comparison.png'}")

    plot_metric_across_representations(records, "accuracy", "Accuracy", plots_dir / "accuracy_comparison.png")
    print(f"      Saved -> {plots_dir / 'accuracy_comparison.png'}")

    plot_metric_across_representations(records, "macro_f1", "Macro-F1", plots_dir / "macro_f1_comparison.png")
    print(f"      Saved -> {plots_dir / 'macro_f1_comparison.png'}")

    plot_metric_across_representations(records, "macro_f1", "Macro-F1 Impact", plots_dir / "feature_representation_impact.png")
    print(f"      Saved -> {plots_dir / 'feature_representation_impact.png'}")

    plot_computational_latency(records, plots_dir / "inference_latency_comparison.png")
    print(f"      Saved -> {plots_dir / 'inference_latency_comparison.png'}")

    print("\n" + "=" * 70)
    print("PHASE 7: MODEL COMPARISON & VISUALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
