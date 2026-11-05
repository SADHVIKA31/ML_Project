"""
Training and Evaluation Script for Support Vector Machine (SVM) - Person 2 (Archana Ulloju).

Executes:
1. Feature loading (combined representation: Color Histogram + HOG).
2. Kernel comparison experiments (Linear vs RBF vs Polynomial) evaluated on validation set.
3. Hyperparameter tuning (C, gamma, degree) evaluated strictly on validation set.
4. Final unbiased evaluation on the isolated test set.
5. Prediction logging, confusion matrix plotting, and model serialization.
"""

import sys
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from archana_person2.src.feature_extractor import AquaticFeatureExtractor
from archana_person2.src.svm_classifier import AquaticSVMClassifier


def train_and_evaluate(representation: str = "combined"):
    print("=" * 65)
    print(f"PHASE 5: SVM CLASSIFIER TRAINING & EVALUATION ({representation.upper()})")
    print("=" * 65)

    extractor = AquaticFeatureExtractor()
    models_dir = project_root / "archana_person2" / "artifacts" / "models"
    metrics_dir = project_root / "archana_person2" / "artifacts" / "metrics"
    plots_dir = project_root / "archana_person2" / "artifacts" / "plots"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load extracted features
    print("[1/6] Loading extracted features and labels...")
    X_train, y_train, train_ids, _ = extractor.extract_dataset("train", representation)
    X_val, y_val, val_ids, _ = extractor.extract_dataset("val", representation)
    X_test, y_test, test_ids, _ = extractor.extract_dataset("test", representation)
    print(f"      Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 2. Kernel comparison experiment on validation split
    print("\n[2/6] Running kernel comparison experiments on validation split...")
    clf = AquaticSVMClassifier(random_state=42)
    kernel_results = clf.compare_kernels(
        X_train, y_train, X_val, y_val,
        kernels=["linear", "rbf", "poly"],
        c_values=[0.1, 1.0, 10.0],
    )
    best_kernel_setup = kernel_results["best_kernel_setup"]
    print(f"      Best Kernel: {best_kernel_setup['kernel'].upper()} (C={best_kernel_setup['C']}, Val Macro-F1: {best_kernel_setup['val_macro_f1']:.4f})")

    kernel_exp_path = metrics_dir / "svm_kernel_experiments.json"
    with open(kernel_exp_path, "w", encoding="utf-8") as f:
        json.dump(kernel_results, f, indent=2)
    print(f"      Kernel experiments saved -> {kernel_exp_path}")

    # 3. Full hyperparameter search on validation split
    print("\n[3/6] Running validation grid search (no test leakage)...")
    param_grid = {
        "kernel": ["linear", "rbf", "poly"],
        "C": [0.01, 0.1, 1.0, 5.0, 10.0],
        "gamma": ["scale", "auto"],
        "degree": [2, 3],
    }
    best_params = clf.tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid=param_grid)
    print(f"      Best overall hyperparameters: {best_params} (Val Macro-F1: {best_params['best_val_macro_f1']:.4f})")

    exp_path = metrics_dir / "svm_experiments.json"
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_params": best_params,
            "total_trials": len(clf.experiment_results_),
            "trials": clf.experiment_results_,
        }, f, indent=2)
    print(f"      Hyperparameter experiments saved -> {exp_path}")

    # 4. Final evaluation on test split
    print("\n[4/6] Evaluating on independent test set...")
    label_map = extractor.loader.get_label_map()
    inv_label_map = extractor.loader.get_inverse_label_map()
    classes = [inv_label_map[i] for i in sorted(inv_label_map.keys())]

    metrics = clf.evaluate(X_test, y_test, class_names=classes)
    metrics["selected_representation"] = representation
    metrics["best_kernel_setup"] = best_kernel_setup
    metrics["best_hyperparameters"] = best_params

    metrics_path = metrics_dir / "svm_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"      Test Accuracy : {metrics['accuracy']:.4f}")
    print(f"      Test Macro-F1 : {metrics['macro_f1']:.4f}  (Primary Metric)")
    print(f"      Test Weighted-F1: {metrics['weighted_f1']:.4f}")
    print(f"      Metrics saved -> {metrics_path}")

    # 5. Export sample predictions CSV
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)
    csv_path = metrics_dir / "svm_predictions.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["sample_id", "true_class", "pred_class", "correct", "confidence"]
        writer.writerow(header)
        for s_id, true_idx, pred_idx, probs in zip(test_ids, y_test, test_preds, test_probs):
            true_cls = inv_label_map.get(true_idx, f"Class_{true_idx}")
            pred_cls = inv_label_map.get(pred_idx, f"Class_{pred_idx}")
            conf = float(np.max(probs))
            writer.writerow([s_id, true_cls, pred_cls, true_idx == pred_idx, f"{conf:.4f}"])
    print(f"      Predictions saved -> {csv_path}")

    # 6. Plot Confusion Matrix
    print("\n[5/6] Plotting confusion matrix...")
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"SVM Confusion Matrix ({clf.kernel.upper()}, C={clf.C})\nMacro-F1: {metrics['macro_f1']:.4f}",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    cm_path = plots_dir / "svm_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Confusion matrix plot saved -> {cm_path}")

    # 7. Serialize trained model
    print("\n[6/6] Saving trained model pipeline...")
    model_path = models_dir / "svm_model.joblib"
    clf.save(model_path)
    print(f"      Model artifact saved -> {model_path}")

    print("\n" + "=" * 65)
    print("PHASE 5: SVM CLASSIFIER TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    train_and_evaluate(representation="combined")
