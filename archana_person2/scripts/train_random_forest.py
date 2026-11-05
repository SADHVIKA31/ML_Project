"""
Training and Evaluation Script for Random Forest - Person 2 (Archana Ulloju).
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
from archana_person2.src.random_forest import AquaticRandomForest


def train_and_evaluate(representation: str = "combined"):
    print("=" * 65)
    print(f"PHASE 4: RANDOM FOREST TRAINING & EVALUATION ({representation.upper()})")
    print("=" * 65)

    extractor = AquaticFeatureExtractor()
    models_dir = project_root / "archana_person2" / "artifacts" / "models"
    metrics_dir = project_root / "archana_person2" / "artifacts" / "metrics"
    plots_dir = project_root / "archana_person2" / "artifacts" / "plots"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load extracted features
    print("[1/5] Loading extracted features and labels...")
    X_train, y_train, train_ids, _ = extractor.extract_dataset("train", representation)
    X_val, y_val, val_ids, _ = extractor.extract_dataset("val", representation)
    X_test, y_test, test_ids, _ = extractor.extract_dataset("test", representation)
    print(f"      Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 2. Hyperparameter tuning on validation split
    print("\n[2/5] Running validation grid search (no test leakage)...")
    clf = AquaticRandomForest(random_state=42)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    best_params = clf.tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid=param_grid)
    print(f"      Best params: {best_params} (Val Macro-F1: {best_params['best_val_macro_f1']:.4f})")

    # Save hyperparameter search log
    exp_path = metrics_dir / "random_forest_experiments.json"
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_params": best_params,
            "total_trials": len(clf.experiment_results_),
            "trials": clf.experiment_results_,
        }, f, indent=2)
    print(f"      Hyperparameter experiments saved -> {exp_path}")

    # 3. Final evaluation on test split
    print("\n[3/5] Evaluating on independent test set...")
    label_map = extractor.loader.get_label_map()
    inv_label_map = extractor.loader.get_inverse_label_map()
    classes = [inv_label_map[i] for i in sorted(inv_label_map.keys())]

    metrics = clf.evaluate(X_test, y_test, target_names=classes)
    metrics["selected_representation"] = representation
    metrics["best_hyperparameters"] = best_params

    print("\n--- TEST PERFORMANCE REPORT ---")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:       {metrics['macro_recall']:.4f}")
    print(f"  Macro-F1 (PRIMARY): {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1:        {metrics['weighted_f1']:.4f}")

    # 4. Save artifacts
    print("\n[4/5] Saving model artifact and metric reports...")
    model_path = models_dir / "random_forest_model.joblib"
    clf.save(str(model_path))
    print(f"      Model artifact saved -> {model_path}")

    metrics_path = metrics_dir / "random_forest_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Metrics report saved -> {metrics_path}")

    # 5. Save predictions CSV, confusion matrix, and feature importance plot
    print("\n[5/5] Generating predictions CSV, confusion matrix, and feature importances...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    pred_csv_path = metrics_dir / "random_forest_predictions.csv"
    with open(pred_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["sample_id", "true_label", "true_class", "pred_label", "pred_class", "confidence"]
        header.extend([f"prob_{c}" for c in classes])
        writer.writerow(header)

        for i in range(len(test_ids)):
            true_idx = int(y_test[i])
            pred_idx = int(y_pred[i])
            conf = float(np.max(y_proba[i]))
            row = [
                test_ids[i],
                true_idx,
                inv_label_map.get(true_idx, f"Class_{true_idx}"),
                pred_idx,
                inv_label_map.get(pred_idx, f"Class_{pred_idx}"),
                f"{conf:.4f}",
            ]
            row.extend([f"{p:.4f}" for p in y_proba[i]])
            writer.writerow(row)
    print(f"      Predictions CSV saved -> {pred_csv_path}")

    # Confusion matrix plot
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title("Random Forest - Test Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Species Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Species Label", fontsize=11, fontweight="bold")
    plt.tight_layout()

    cm_path = plots_dir / "random_forest_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"      Confusion matrix chart saved -> {cm_path}")

    # Feature importances plot (Top 25 features)
    importances = clf.get_feature_importances()
    top_k = min(25, len(importances))
    top_indices = np.argsort(importances)[::-1][:top_k]
    top_scores = importances[top_indices]

    # Distinguish Color Hist vs HOG in combined representation
    col_dim = extractor.color_extractor.feature_dim
    feat_labels = []
    for idx in top_indices:
        if idx < col_dim:
            feat_labels.append(f"ColorHist_{idx}")
        else:
            feat_labels.append(f"HOG_{idx - col_dim}")

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(top_k)
    plt.barh(y_pos, top_scores[::-1], color="#2ca02c", align="center")
    plt.yticks(y_pos, feat_labels[::-1], fontsize=9)
    plt.xlabel("Gini Feature Importance", fontsize=11, fontweight="bold")
    plt.title(f"Random Forest - Top {top_k} Informative Features ({representation.upper()})", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    fi_path = plots_dir / "random_forest_feature_importance.png"
    plt.savefig(fi_path, dpi=200)
    plt.close()
    print(f"      Feature importance chart saved -> {fi_path}")

    print("\n" + "=" * 65)
    print("RANDOM FOREST TRAINING & EVALUATION COMPLETED!")
    print("=" * 65)
    return metrics


if __name__ == "__main__":
    train_and_evaluate(representation="combined")
