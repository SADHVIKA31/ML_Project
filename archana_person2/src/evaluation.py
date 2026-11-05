"""
Common Model Evaluation Framework - Person 2 (Archana Ulloju).

Provides reusable, standardized evaluation metrics, classification reports,
confusion matrices, and optional probability metrics for all classical ML models:
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

Ensures consistent class ordering, rigorous input validation, and zero test leakage.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def validate_evaluation_inputs(
    y_true: Union[np.ndarray, List[Any]],
    y_pred: Union[np.ndarray, List[Any]],
    y_prob: Optional[Union[np.ndarray, List[List[float]]]] = None,
    sample_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Validates evaluation inputs against common edge cases:
    - Empty inputs
    - Mismatched lengths
    - Invalid probability dimensions or out-of-bound values
    - Duplicate sample IDs
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0:
        raise ValueError("Ground-truth labels (y_true) cannot be empty.")
    if len(y_pred_arr) == 0:
        raise ValueError("Predicted labels (y_pred) cannot be empty.")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Mismatched lengths between y_true ({len(y_true_arr)}) and y_pred ({len(y_pred_arr)})."
        )

    if sample_ids is not None:
        if len(sample_ids) != len(y_true_arr):
            raise ValueError(
                f"Mismatched lengths between sample_ids ({len(sample_ids)}) and y_true ({len(y_true_arr)})."
            )
        if len(set(sample_ids)) != len(sample_ids):
            duplicates = [sid for sid in set(sample_ids) if sample_ids.count(sid) > 1]
            raise ValueError(f"Duplicate sample IDs detected in evaluation set: {duplicates[:5]}")

    y_prob_arr: Optional[np.ndarray] = None
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)
        if y_prob_arr.ndim != 2:
            raise ValueError(f"Probability array must be 2-dimensional (samples, classes). Got ndim={y_prob_arr.ndim}.")
        if y_prob_arr.shape[0] != len(y_true_arr):
            raise ValueError(
                f"Mismatched probability samples ({y_prob_arr.shape[0]}) and y_true ({len(y_true_arr)})."
            )
        if np.any(np.isnan(y_prob_arr)) or np.any(np.isinf(y_prob_arr)):
            raise ValueError("Probability array contains NaN or infinite values.")
        if np.any(y_prob_arr < -1e-5) or np.any(y_prob_arr > 1.0 + 1e-5):
            raise ValueError("Probability values must lie strictly in [0.0, 1.0].")

    return y_true_arr, y_pred_arr, y_prob_arr


def compute_standard_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Computes standard classification metrics with fixed zero_division=0.
    Primary Metric: Macro-F1 (unweighted mean across species).
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    return {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "weighted_precision": weighted_prec,
        "macro_recall": macro_rec,
        "weighted_recall": weighted_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def compute_classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generates per-class precision, recall, F1, and support."""
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return report


def compute_confusion_matrix_list(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> List[List[int]]:
    """Generates confusion matrix preserving consistent class order."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm.tolist()


def compute_probability_metrics(
    y_true: np.ndarray,
    y_prob: Optional[np.ndarray],
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Computes optional probability-based metrics (Log Loss and Multi-class ROC-AUC).
    Safely degrades and reports unavailability reason when preconditions are not met.
    Never fabricates fake metric values.
    """
    if y_prob is None:
        return {
            "available": False,
            "reason": "Model or evaluation run did not supply predicted class probabilities.",
        }

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {
            "available": False,
            "reason": "At least two distinct classes must be present in y_true to evaluate probability metrics.",
        }

    results: Dict[str, Any] = {"available": True}

    # 1. Log Loss
    try:
        # Clip probabilities for numerical stability
        eps = 1e-15
        clipped_probs = np.clip(y_prob, eps, 1 - eps)
        clipped_probs /= clipped_probs.sum(axis=1, keepdims=True)
        loss = float(log_loss(y_true, clipped_probs, labels=labels))
        results["log_loss"] = loss
    except Exception as e:
        results["log_loss"] = None
        results["log_loss_error"] = str(e)

    # 2. Multi-class ROC-AUC (One-vs-Rest)
    try:
        # Check if number of columns matches number of labels
        if labels is not None and y_prob.shape[1] == len(labels):
            auc = float(roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="macro",
            ))
            results["roc_auc_macro_ovr"] = auc
        else:
            auc = float(roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
            ))
            results["roc_auc_macro_ovr"] = auc
    except Exception as e:
        results["roc_auc_macro_ovr"] = None
        results["roc_auc_error"] = str(e)

    return results


class CommonModelEvaluator:
    """
    Reusable evaluation engine applied across all classical machine learning models.
    Guarantees:
    - Same test samples and split.
    - Same ground-truth labels.
    - Same class alphabetical index mapping.
    - Identical metric formulations.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
    ):
        self.class_names = class_names
        self.labels = labels if labels is not None else (
            list(range(len(class_names))) if class_names is not None else None
        )

    def evaluate_predictions(
        self,
        y_true: Union[np.ndarray, List[Any]],
        y_pred: Union[np.ndarray, List[Any]],
        y_prob: Optional[Union[np.ndarray, List[List[float]]]] = None,
        sample_ids: Optional[List[str]] = None,
        model_name: str = "Classical Model",
    ) -> Dict[str, Any]:
        """Evaluates raw predicted classes and probabilities against ground truth."""
        y_true_v, y_pred_v, y_prob_v = validate_evaluation_inputs(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            sample_ids=sample_ids,
        )

        standard = compute_standard_metrics(y_true_v, y_pred_v, labels=self.labels)
        report = compute_classification_report_dict(
            y_true_v,
            y_pred_v,
            labels=self.labels,
            target_names=self.class_names,
        )
        cm = compute_confusion_matrix_list(y_true_v, y_pred_v, labels=self.labels)
        prob_metrics = compute_probability_metrics(y_true_v, y_prob_v, labels=self.labels)

        evaluation_output = {
            "model_name": model_name,
            "num_samples": int(len(y_true_v)),
            "metrics": standard,
            "probability_metrics": prob_metrics,
            "confusion_matrix": cm,
            "classification_report": report,
            "class_names": self.class_names,
            "labels": self.labels,
        }
        return evaluation_output

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sample_ids: Optional[List[str]] = None,
        model_name: str = "Classical Model",
    ) -> Dict[str, Any]:
        """Executes model inference and calculates the common evaluation metrics."""
        y_pred = model.predict(X_test)

        y_prob: Optional[np.ndarray] = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                y_prob = None

        result = self.evaluate_predictions(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            sample_ids=sample_ids,
            model_name=model_name,
        )
        return result
