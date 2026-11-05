"""
Common Model Evaluation Pipeline - Person 2 (Archana Ulloju).

Evaluates all three classical machine learning baselines:
1. Logistic Regression (StandardScaler + Multinomial LR Pipeline)
2. Random Forest (Ensemble Trees with Bagging)
3. Support Vector Machine (StandardScaler + SVC Pipeline)

Ensures:
- Identical test feature split (no test data in training/validation).
- Identical ground-truth labels and sample IDs.
- Consistent class ordering across all confusion matrices and reports.
- Comprehensive metric logging (Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Log Loss, ROC-AUC).
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from archana_person2.src.feature_extractor import AquaticFeatureExtractor
from archana_person2.src.logistic_regression import AquaticLogisticRegression
from archana_person2.src.random_forest import AquaticRandomForest
from archana_person2.src.svm_classifier import AquaticSVMClassifier
from archana_person2.src.evaluation import CommonModelEvaluator


def plot_confusion_matrix(cm_list: List[List[int]], classes: List[str], title: str, save_path: Path):
    """Plots and saves a single confusion matrix with consistent styling."""
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_metrics_summary(models_data: Dict[str, Dict[str, Any]], save_path: Path):
    """Plots a grouped bar chart comparing all models across key summary metrics."""
    metrics_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    metric_labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro-F1 (Primary)", "Weighted-F1"]
    model_names = list(models_data.keys())

    x = np.arange(len(metric_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    for idx, (m_name, color) in enumerate(zip(model_names, colors)):
        vals = [models_data[m_name]["metrics"][k] for k in metrics_keys]
        rects = ax.bar(x + (idx - 1) * width, vals, width, label=m_name, color=color, alpha=0.9)
        # Add value labels above bars
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
    ax.set_title("Classical ML Models Evaluation Summary (Test Set)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(models_data: Dict[str, Dict[str, Any]], classes: List[str], save_path: Path):
    """Plots per-class F1-scores across all three models."""
    x = np.arange(len(classes))
    width = 0.25
    model_names = list(models_data.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    for idx, (m_name, color) in enumerate(zip(model_names, colors)):
        report = models_data[m_name]["classification_report"]
        f1_vals = [report.get(cls_name, {}).get("f1-score", 0.0) for cls_name in classes]
        ax.bar(x + (idx - 1) * width, f1_vals, width, label=m_name, color=color, alpha=0.9)

    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("Per-Class F1-Score Comparison Across Models", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("PHASE 6: COMMON MODEL EVALUATION FRAMEWORK EXECUTION")
    print("=" * 70)

    # 1. Paths setup
    models_dir = project_root / "archana_person2" / "artifacts" / "models"
    metrics_dir = project_root / "archana_person2" / "artifacts" / "metrics"
    plots_dir = project_root / "archana_person2" / "artifacts" / "plots"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load identical test features and labels
    print("\n[1/5] Loading identical test features and ground-truth labels...")
    extractor = AquaticFeatureExtractor()
    X_test, y_test, test_ids, _ = extractor.extract_dataset("test", "combined")
    
    inv_label_map = extractor.loader.get_inverse_label_map()
    classes = [inv_label_map[i] for i in sorted(inv_label_map.keys())]
    labels = list(range(len(classes)))
    print(f"      Test set samples: {len(y_test)}, Features dimension: {X_test.shape[1]}")
    print(f"      Classes ({len(classes)}): {classes}")

    # 3. Instantiate Common Evaluator
    evaluator = CommonModelEvaluator(class_names=classes, labels=labels)

    # 4. Load models
    print("\n[2/5] Loading trained classical model artifacts...")
    model_paths = {
        "Logistic Regression": models_dir / "logistic_regression_pipeline.joblib",
        "Random Forest": models_dir / "random_forest_model.joblib",
        "Support Vector Machine (SVM)": models_dir / "svm_model.joblib",
    }

    loaded_models = {}
    for m_name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found for {m_name} at: {path}")
        if m_name == "Logistic Regression":
            loaded_models[m_name] = AquaticLogisticRegression.load(path)
        elif m_name == "Random Forest":
            loaded_models[m_name] = AquaticRandomForest.load(path)
        elif m_name == "Support Vector Machine (SVM)":
            loaded_models[m_name] = AquaticSVMClassifier.load(path)
        print(f"      Loaded: {m_name} from {path.name}")

    # 5. Evaluate all models through the common framework
    print("\n[3/5] Evaluating all models through common pipeline...")
    evaluation_results: Dict[str, Dict[str, Any]] = {}
    summary_rows = []

    for m_name, model in loaded_models.items():
        print(f"\n      --- Evaluating {m_name} ---")
        eval_res = evaluator.evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            sample_ids=test_ids,
            model_name=m_name,
        )
        evaluation_results[m_name] = eval_res
        metrics = eval_res["metrics"]
        prob_metrics = eval_res["probability_metrics"]

        print(f"          Accuracy         : {metrics['accuracy']:.4f}")
        print(f"          Macro Precision  : {metrics['macro_precision']:.4f}")
        print(f"          Macro Recall     : {metrics['macro_recall']:.4f}")
        print(f"          Macro-F1 (Primary): {metrics['macro_f1']:.4f}")
        print(f"          Weighted-F1      : {metrics['weighted_f1']:.4f}")

        if prob_metrics.get("available"):
            ll = prob_metrics.get("log_loss")
            roc = prob_metrics.get("roc_auc_macro_ovr")
            ll_str = f"{ll:.4f}" if ll is not None else "N/A"
            roc_str = f"{roc:.4f}" if roc is not None else "N/A"
            print(f"          Log Loss         : {ll_str}")
            print(f"          ROC-AUC (Macro)  : {roc_str}")
        else:
            print(f"          Prob Metrics     : Unavailable ({prob_metrics.get('reason')})")

        summary_rows.append({
            "model_name": m_name,
            "test_samples": eval_res["num_samples"],
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "weighted_precision": metrics["weighted_precision"],
            "macro_recall": metrics["macro_recall"],
            "weighted_recall": metrics["weighted_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "log_loss": prob_metrics.get("log_loss"),
            "roc_auc_macro_ovr": prob_metrics.get("roc_auc_macro_ovr"),
        })

    # 6. Save results
    print("\n[4/5] Saving evaluation results and summary tables...")
    
    # 6.1 Full evaluation JSON
    eval_json_path = metrics_dir / "evaluation_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"      Saved detailed results -> {eval_json_path}")

    # 6.2 Summary CSV
    summary_csv_path = metrics_dir / "evaluation_summary.csv"
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model_name", "test_samples", "accuracy",
            "macro_precision", "weighted_precision",
            "macro_recall", "weighted_recall",
            "macro_f1", "weighted_f1",
            "log_loss", "roc_auc_macro_ovr",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"      Saved summary CSV -> {summary_csv_path}")

    # 6.3 Classification reports JSON
    reports_json_path = metrics_dir / "classification_report.json"
    classification_reports = {
        m_name: res["classification_report"] for m_name, res in evaluation_results.items()
    }
    with open(reports_json_path, "w", encoding="utf-8") as f:
        json.dump(classification_reports, f, indent=2)
    print(f"      Saved classification reports -> {reports_json_path}")

    # 7. Generate plots
    print("\n[5/5] Generating evaluation plots and visualizations...")
    
    # 7.1 Confusion Matrices for all 3 models
    plot_confusion_matrix(
        evaluation_results["Logistic Regression"]["confusion_matrix"],
        classes,
        f"Logistic Regression Confusion Matrix\nMacro-F1: {evaluation_results['Logistic Regression']['metrics']['macro_f1']:.4f}",
        plots_dir / "confusion_matrix_logistic_regression.png",
    )
    print(f"      Saved -> {plots_dir / 'confusion_matrix_logistic_regression.png'}")

    plot_confusion_matrix(
        evaluation_results["Random Forest"]["confusion_matrix"],
        classes,
        f"Random Forest Confusion Matrix\nMacro-F1: {evaluation_results['Random Forest']['metrics']['macro_f1']:.4f}",
        plots_dir / "confusion_matrix_random_forest.png",
    )
    print(f"      Saved -> {plots_dir / 'confusion_matrix_random_forest.png'}")

    plot_confusion_matrix(
        evaluation_results["Support Vector Machine (SVM)"]["confusion_matrix"],
        classes,
        f"SVM Confusion Matrix\nMacro-F1: {evaluation_results['Support Vector Machine (SVM)']['metrics']['macro_f1']:.4f}",
        plots_dir / "confusion_matrix_svm.png",
    )
    print(f"      Saved -> {plots_dir / 'confusion_matrix_svm.png'}")

    # 7.2 Model Evaluation Summary Bar Chart
    summary_plot_path = plots_dir / "model_evaluation_metrics_summary.png"
    plot_model_metrics_summary(evaluation_results, summary_plot_path)
    print(f"      Saved -> {summary_plot_path}")

    # 7.3 Per-Class Performance Comparison Chart
    per_class_plot_path = plots_dir / "per_class_performance.png"
    plot_per_class_f1(evaluation_results, classes, per_class_plot_path)
    print(f"      Saved -> {per_class_plot_path}")

    print("\n" + "=" * 70)
    print("PHASE 6: COMMON MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
