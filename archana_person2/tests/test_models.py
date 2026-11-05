"""
Unit Tests for Machine Learning Models - Person 2 (Archana Ulloju).

Tests Logistic Regression, Random Forest, and Support Vector Machine pipelines,
ensuring proper fit/transform, evaluation metric calculation, and joblib serialization
without data leakage.
"""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from archana_person2.src.logistic_regression import AquaticLogisticRegression
from archana_person2.src.random_forest import AquaticRandomForest
from archana_person2.src.svm_classifier import AquaticSVMClassifier


@pytest.fixture
def synthetic_data():
    """Generates synthetic feature matrices and labels for multi-class testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, 20).astype(np.float32)
    # 3 classes: 0, 1, 2
    y = np.array([0, 1, 2] * 20, dtype=np.int32)
    return X[:40], y[:40], X[40:50], y[40:50], X[50:], y[50:]


class TestLogisticRegression:
    def test_pipeline_build_and_fit(self, synthetic_data):
        X_train, y_train, _, _, X_test, y_test = synthetic_data
        clf = AquaticLogisticRegression(C=1.0, max_iter=200, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1, 2})

        probs = clf.predict_proba(X_test)
        assert probs.shape == (len(X_test), 3)
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    def test_hyperparameter_tuning(self, synthetic_data):
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        clf = AquaticLogisticRegression(max_iter=200, random_state=42)
        best_params = clf.tune_hyperparameters(
            X_train, y_train, X_val, y_val, c_candidates=[0.1, 1.0]
        )
        assert "C" in best_params
        assert clf.C in [0.1, 1.0]
        assert len(clf.validation_results_) == 2

    def test_evaluation_metrics(self, synthetic_data):
        X_train, y_train, _, _, X_test, y_test = synthetic_data
        clf = AquaticLogisticRegression(C=1.0, max_iter=200, random_state=42)
        clf.fit(X_train, y_train)

        target_names = ["Species_01", "Species_02", "Species_03"]
        metrics = clf.evaluate(X_test, y_test, target_names=target_names)

        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "confusion_matrix" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert len(metrics["confusion_matrix"]) == 3

    def test_save_load(self, synthetic_data):
        X_train, y_train, _, _, X_test, _ = synthetic_data
        clf = AquaticLogisticRegression(C=0.5, max_iter=200, random_state=42)
        clf.fit(X_train, y_train)
        preds_orig = clf.predict(X_test)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_lr.joblib")
            clf.save(save_path)
            loaded = AquaticLogisticRegression.load(save_path)
            preds_loaded = loaded.predict(X_test)
            assert np.array_equal(preds_orig, preds_loaded)
            assert loaded.C == 0.5


class TestRandomForest:
    def test_model_build_and_fit(self, synthetic_data):
        X_train, y_train, _, _, X_test, y_test = synthetic_data
        rf = AquaticRandomForest(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1, 2})

        probs = rf.predict_proba(X_test)
        assert probs.shape == (len(X_test), 3)
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    def test_feature_importances(self, synthetic_data):
        X_train, y_train, _, _, _, _ = synthetic_data
        rf = AquaticRandomForest(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        importances = rf.get_feature_importances()
        assert len(importances) == X_train.shape[1]
        assert np.isclose(np.sum(importances), 1.0, atol=1e-5)

    def test_hyperparameter_tuning(self, synthetic_data):
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        rf = AquaticRandomForest(random_state=42)
        grid = {
            "n_estimators": [5, 10],
            "max_depth": [3, 5],
        }
        best_params = rf.tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid=grid)
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert len(rf.experiment_results_) == 4

    def test_evaluation_metrics(self, synthetic_data):
        X_train, y_train, _, _, X_test, y_test = synthetic_data
        rf = AquaticRandomForest(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        target_names = ["Species_01", "Species_02", "Species_03"]
        metrics = rf.evaluate(X_test, y_test, target_names=target_names)

        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "confusion_matrix" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_save_load(self, synthetic_data):
        X_train, y_train, _, _, X_test, _ = synthetic_data
        rf = AquaticRandomForest(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        preds_orig = rf.predict(X_test)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_rf.joblib")
            rf.save(save_path)
            loaded = AquaticRandomForest.load(save_path)
            preds_loaded = loaded.predict(X_test)
            assert np.array_equal(preds_orig, preds_loaded)
            assert loaded.n_estimators == 10


class TestSVMClassifier:
    def test_pipeline_build_and_fit(self, synthetic_data):
        X_train, y_train, _, _, X_test, y_test = synthetic_data
        svm = AquaticSVMClassifier(kernel="rbf", C=1.0, random_state=42)
        svm.fit(X_train, y_train)

        preds = svm.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1, 2})

        probs = svm.predict_proba(X_test)
        assert probs.shape == (len(X_test), 3)
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    def test_kernel_comparison(self, synthetic_data):
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        svm = AquaticSVMClassifier(random_state=42)
        results = svm.compare_kernels(
            X_train, y_train, X_val, y_val,
            kernels=["linear", "rbf", "poly"],
            c_values=[0.1, 1.0],
        )
        assert "best_kernel_setup" in results
        assert "all_kernel_results" in results
        assert len(results["all_kernel_results"]) == 6
        assert len(svm.kernel_comparison_results_) == 6

    def test_hyperparameter_tuning(self, synthetic_data):
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        svm = AquaticSVMClassifier(random_state=42)
        grid = {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1.0],
            "gamma": ["scale"],
            "degree": [3],
        }
        best_params = svm.tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid=grid)
        assert "kernel" in best_params
        assert "C" in best_params
        assert len(svm.experiment_results_) == 4

    def test_evaluation_metrics(self, synthetic_data):
        X_train, y_train, _, _, X_test, y_test = synthetic_data
        svm = AquaticSVMClassifier(kernel="linear", C=1.0, random_state=42)
        svm.fit(X_train, y_train)

        class_names = ["Species_01", "Species_02", "Species_03"]
        metrics = svm.evaluate(X_test, y_test, class_names=class_names)

        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "weighted_f1" in metrics
        assert "confusion_matrix" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert len(metrics["confusion_matrix"]) == 3

    def test_save_load(self, synthetic_data):
        X_train, y_train, _, _, X_test, _ = synthetic_data
        svm = AquaticSVMClassifier(kernel="rbf", C=2.5, random_state=42)
        svm.fit(X_train, y_train)
        preds_orig = svm.predict(X_test)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_svm.joblib")
            svm.save(save_path)
            loaded = AquaticSVMClassifier.load(save_path)
            preds_loaded = loaded.predict(X_test)
            assert np.array_equal(preds_orig, preds_loaded)
            assert loaded.C == 2.5
            assert loaded.kernel == "rbf"
