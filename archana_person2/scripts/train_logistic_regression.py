"""
Training and Evaluation Script for Logistic Regression - Person 2 (Archana Ulloju).
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
from archana_person2.src.logistic_regression import AquaticLogisticRegression


def train_and_evaluate(representation: str = "combined"):
    print("=" * 65)
    print(f"PHASE 3: LOGISTIC REGRESSION TRAINING & EVALUATION ({representation.upper()})")
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
    print("\n[2/5] Tuning regularization parameter C on validation set (leakage-safe)...")
    clf = AquaticLogisticRegression(max_iter=1000, random_state=42)
    best_params = clf.tune_hyperparameters(
        X_train, y_train, X_val, y_val,
        c_candidates=[0.01, 0.1, 1.0, 5.0, 10.0]
    )
    print(f"      Best C selected: {best_params['C']} (Val Macro-F1: {best_params['best_val_macro_f1']:.4f})")

    # 3. Final evaluation on test split
    print("\n[3/5] Evaluating on independent test set...")
    label_map = extractor.loader.get_label_map()
    inv_label_map = extractor.loader.get_inverse_label_map()
    classes = [inv_label_map[i] for i in sorted(inv_label_map.keys())]

    metrics = clf.evaluate(X_test, y_test, target_names=classes)
    metrics["selected_representation"] = representation
    metrics["validation_tuning_results"] = clf.validation_results_

    print("\n--- TEST PERFORMANCE REPORT ---")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
    print(f"  Macro-F1 (PRIMARY): {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1:      {metrics['weighted_f1']:.4f}")

    # 4. Save artifacts
    print("\n[4/5] Saving model pipeline and metric reports...")
    model_path = models_dir / "logistic_regression_pipeline.joblib"
    clf.save(str(model_path))
    print(f"      Trained pipeline saved -> {model_path}")

    metrics_path = metrics_dir / "logistic_regression_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Metrics report saved -> {metrics_path}")

    # 5. Save predictions CSV and confusion matrix plot
    print("\n[5/5] Generating predictions CSV and confusion matrix plot...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    pred_csv_path = metrics_dir / "logistic_regression_predictions.csv"
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
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Logistic Regression - Test Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    # Label counts inside matrix cells
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

    cm_path = plots_dir / "logistic_regression_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"      Confusion matrix chart saved -> {cm_path}")

    print("\n" + "=" * 65)
    print("LOGISTIC REGRESSION TRAINING & EVALUATION COMPLETED!")
    print("=" * 65)
    return metrics


if __name__ == "__main__":
    train_and_evaluate(representation="combined")
