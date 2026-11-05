"""
Automated Unit Tests for Common Model Evaluation Framework - Person 2 (Archana Ulloju).

Tests:
1. Accuracy calculation with verified hand-computed values.
2. Macro and Weighted Precision.
3. Macro and Weighted Recall.
4. Macro and Weighted F1 score (Macro-F1 as primary metric).
5. Classification report formatting and per-class values.
6. Confusion matrix values and consistent class ordering.
7. Optional probability metrics (Log Loss and Multi-class ROC-AUC).
8. Graceful handling of unavailable probability metrics without fabrication.
9. Robust input validation: empty inputs, length mismatches, invalid probabilities, duplicate IDs.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from archana_person2.src.evaluation import (
    validate_evaluation_inputs,
    compute_standard_metrics,
    compute_classification_report_dict,
    compute_confusion_matrix_list,
    compute_probability_metrics,
    CommonModelEvaluator,
)


@pytest.fixture
def manual_toy_example():
    """
    Small, manually verifiable 3-class test case.
    Class 0: 2 samples, 1 TP, 1 FN (pred 1). Precision=0.5, Recall=0.5, F1=0.5
    Class 1: 2 samples, 2 TP, 0 FN (plus 1 FP from class 0). Precision=2/3, Recall=1.0, F1=0.8
    Class 2: 2 samples, 1 TP, 1 FN (pred 0). Precision=1.0, Recall=0.5, F1=2/3
    """
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    y_pred = np.array([0, 1, 1, 1, 2, 0], dtype=np.int32)
    class_names = ["Species_01", "Species_02", "Species_03"]
    labels = [0, 1, 2]
    return y_true, y_pred, class_names, labels


class TestMetricCalculations:
    def test_accuracy(self, manual_toy_example):
        y_true, y_pred, _, labels = manual_toy_example
        metrics = compute_standard_metrics(y_true, y_pred, labels=labels)
        expected_acc = 4.0 / 6.0
        assert np.isclose(metrics["accuracy"], expected_acc, atol=1e-4)

    def test_precision_macro_and_weighted(self, manual_toy_example):
        y_true, y_pred, _, labels = manual_toy_example
        metrics = compute_standard_metrics(y_true, y_pred, labels=labels)
        # Macro Precision: (0.5 + 2/3 + 1.0) / 3 = 13/18 ~= 0.7222
        expected_prec = 13.0 / 18.0
        assert np.isclose(metrics["macro_precision"], expected_prec, atol=1e-4)
        assert np.isclose(metrics["weighted_precision"], expected_prec, atol=1e-4)

    def test_recall_macro_and_weighted(self, manual_toy_example):
        y_true, y_pred, _, labels = manual_toy_example
        metrics = compute_standard_metrics(y_true, y_pred, labels=labels)
        # Macro Recall: (0.5 + 1.0 + 0.5) / 3 = 2/3 ~= 0.6667
        expected_rec = 2.0 / 3.0
        assert np.isclose(metrics["macro_recall"], expected_rec, atol=1e-4)
        assert np.isclose(metrics["weighted_recall"], expected_rec, atol=1e-4)

    def test_f1_macro_and_weighted(self, manual_toy_example):
        y_true, y_pred, _, labels = manual_toy_example
        metrics = compute_standard_metrics(y_true, y_pred, labels=labels)
        # Macro F1: (0.5 + 0.8 + 2/3) / 3 = (59/30) / 3 = 59/90 ~= 0.6556
        expected_f1 = 59.0 / 90.0
        assert np.isclose(metrics["macro_f1"], expected_f1, atol=1e-4)
        assert np.isclose(metrics["weighted_f1"], expected_f1, atol=1e-4)

    def test_classification_report(self, manual_toy_example):
        y_true, y_pred, class_names, labels = manual_toy_example
        report = compute_classification_report_dict(
            y_true, y_pred, labels=labels, target_names=class_names
        )
        assert "Species_01" in report
        assert "Species_02" in report
        assert "Species_03" in report
        assert np.isclose(report["Species_01"]["precision"], 0.5, atol=1e-4)
        assert np.isclose(report["Species_02"]["recall"], 1.0, atol=1e-4)
        assert report["Species_01"]["support"] == 2

    def test_confusion_matrix_values_and_ordering(self, manual_toy_example):
        y_true, y_pred, _, labels = manual_toy_example
        cm = compute_confusion_matrix_list(y_true, y_pred, labels=labels)
        expected_cm = [
            [1, 1, 0],
            [0, 2, 0],
            [1, 0, 1],
        ]
        assert cm == expected_cm


class TestProbabilityMetrics:
    def test_valid_probabilities(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        # High confidence predictions
        y_prob = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.85, 0.1, 0.05],
            [0.15, 0.75, 0.1],
            [0.1, 0.1, 0.8],
        ])
        prob_res = compute_probability_metrics(y_true, y_prob, labels=[0, 1, 2])
        assert prob_res["available"] is True
        assert prob_res["log_loss"] is not None
        assert prob_res["roc_auc_macro_ovr"] is not None
        assert 0.0 <= prob_res["log_loss"] <= 1.0
        assert 0.8 <= prob_res["roc_auc_macro_ovr"] <= 1.0

    def test_missing_probabilities_graceful_degradation(self):
        y_true = np.array([0, 1, 0])
        prob_res = compute_probability_metrics(y_true, None)
        assert prob_res["available"] is False
        assert "reason" in prob_res

    def test_single_class_probability_graceful_degradation(self):
        y_true = np.array([0, 0, 0])
        y_prob = np.array([[1.0], [1.0], [1.0]])
        prob_res = compute_probability_metrics(y_true, y_prob)
        assert prob_res["available"] is False
        assert "reason" in prob_res


class TestInputValidationAndRobustness:
    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_evaluation_inputs([], [])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="Mismatched lengths"):
            validate_evaluation_inputs([0, 1], [0])

    def test_duplicate_sample_ids_raise(self):
        y_true = [0, 1]
        y_pred = [0, 1]
        sample_ids = ["img_001", "img_001"]
        with pytest.raises(ValueError, match="Duplicate sample IDs detected"):
            validate_evaluation_inputs(y_true, y_pred, sample_ids=sample_ids)

    def test_invalid_probability_dimension_raises(self):
        y_true = [0, 1]
        y_pred = [0, 1]
        y_prob_1d = [0.5, 0.5]
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            validate_evaluation_inputs(y_true, y_pred, y_prob=y_prob_1d)

    def test_out_of_bounds_probability_raises(self):
        y_true = [0, 1]
        y_pred = [0, 1]
        y_prob_bad = [[1.5, -0.5], [0.5, 0.5]]
        with pytest.raises(ValueError, match="strictly in"):
            validate_evaluation_inputs(y_true, y_pred, y_prob=y_prob_bad)


class TestCommonModelEvaluator:
    def test_evaluator_end_to_end(self, manual_toy_example):
        y_true, y_pred, class_names, labels = manual_toy_example
        evaluator = CommonModelEvaluator(class_names=class_names, labels=labels)

        res = evaluator.evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            model_name="TestClassifier",
        )

        assert res["model_name"] == "TestClassifier"
        assert res["num_samples"] == 6
        assert "metrics" in res
        assert "accuracy" in res["metrics"]
        assert "macro_f1" in res["metrics"]
        assert "confusion_matrix" in res
        assert "classification_report" in res
        assert res["class_names"] == class_names
