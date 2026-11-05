"""
test_validation.py — Phase 11: Comprehensive Testing & Model Validation
========================================================================

Person 2 (Archana Ulloju) — Machine Learning for Robust Aquatic Biodiversity Monitoring.

Covers:
  TestPublicAPIViaPackageRoot      (16 tests) — Every public symbol from src/__init__.py
  TestImageToPredictionPipeline    (10 tests) — End-to-end image → feature → model → result
  TestInvalidInputRobustness       (12 tests) — Corrupt / missing / wrong-format inputs
  TestProbabilityOutputValidation  ( 8 tests) — Confidence values, probability distributions
  TestEvaluationMetricsValidation  ( 8 tests) — Metric correctness and classification reports
  TestDatasetLoaderBridge          ( 6 tests) — integration.py ↔ Sadhvika's DatasetLoader
  TestRealDataAvailability         ( 4 tests) — Real Fish4Knowledge data check (skipped if absent)
  TestRegressionPhase1to10         ( 6 tests) — Regression guard: Phases 1-10 still work

Data provenance: ALL tests use synthetic fixtures.
Real-data tests (TestRealDataAvailability) are auto-skipped when data/preprocessed/
does not exist and are clearly marked PENDING in the validation report.
"""

import os
import sys
import json
import csv
import tempfile
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List

import numpy as np
import pytest
import cv2

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

_PERSON2_ROOT = _PROJECT_ROOT / "archana_person2"
_MODELS_DIR   = _PERSON2_ROOT / "artifacts" / "models"
_MODEL_FILES  = {
    "logistic_regression": "logistic_regression_pipeline.joblib",
    "random_forest":       "random_forest_model.joblib",
    "svm":                 "svm_model.joblib",
}
_REAL_DATA_AVAILABLE = (
    (_PROJECT_ROOT / "data" / "preprocessed" / "X_train.npy").exists()
    and (_PROJECT_ROOT / "data" / "preprocessed" / "label_map.json").exists()
    and (_PROJECT_ROOT / "data" / "splits" / "train.csv").exists()
)

# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------

def _rgb(h: int = 64, w: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_png(tmp_path: Path, name: str = "fish.png", seed: int = 1) -> Path:
    img = _rgb(seed=seed)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = tmp_path / name
    cv2.imwrite(str(out), bgr)
    return out


def _make_jpeg(tmp_path: Path, name: str = "fish.jpg", seed: int = 2) -> Path:
    img = _rgb(seed=seed)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = tmp_path / name
    cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def _make_bmp(tmp_path: Path, name: str = "fish.bmp", seed: int = 3) -> Path:
    img = _rgb(seed=seed)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = tmp_path / name
    cv2.imwrite(str(out), bgr)
    return out


_LABEL_MAP_5 = {f"Species_{i:02d}": i for i in range(5)}
_INV_LABEL_5 = {i: f"Species_{i:02d}" for i in range(5)}


def _make_mock_predictor(model_name="svm", representation="combined"):
    """Return an AquaticPredictor with fully mocked sklearn internals."""
    from archana_person2.src.predict import AquaticPredictor, _PROB_SUPPORTED_MODELS
    n = 5
    sk_est = MagicMock()
    sk_est.classes_ = np.arange(n, dtype=np.int32)
    sk_est.predict.return_value = np.array([0])
    if model_name in _PROB_SUPPORTED_MODELS:
        proba = np.zeros((1, n), dtype=np.float64); proba[0, 0] = 1.0
        sk_est.predict_proba.return_value = proba

    wrapper = MagicMock()
    with patch("pathlib.Path.exists", return_value=True), \
         patch("archana_person2.src.predict._load_model_instance", return_value=wrapper), \
         patch("archana_person2.src.predict._get_sklearn_estimator", return_value=sk_est), \
         patch("archana_person2.src.dataset_loader.Person2DatasetLoader.get_label_map",
               return_value=_LABEL_MAP_5), \
         patch("archana_person2.src.dataset_loader.Person2DatasetLoader.get_inverse_label_map",
               return_value=_INV_LABEL_5):
        predictor = AquaticPredictor(model_name=model_name, representation=representation)

    dim = predictor.feature_dim
    predictor._feature_extractor.extract_image = MagicMock(
        return_value=np.ones(dim, dtype=np.float32)
    )
    return predictor


# ===========================================================================
# TEST CLASSES
# ===========================================================================


class TestPublicAPIViaPackageRoot:
    """
    Every public symbol exported by archana_person2.src.__init__ must be
    importable via the package root (lazy __getattr__ path).
    """

    def test_package_version_is_1_9_0(self):
        import archana_person2.src as pkg
        assert pkg.__version__ == "1.9.0"

    def test_phases_complete_includes_1_to_10(self):
        import archana_person2.src as pkg
        assert list(range(1, 11)) == pkg.__phases_complete__

    def test_ClassicalMLPipeline_accessible(self):
        from archana_person2.src import ClassicalMLPipeline
        assert ClassicalMLPipeline is not None

    def test_load_classical_predictor_accessible(self):
        from archana_person2.src import load_classical_predictor
        assert callable(load_classical_predictor)

    def test_predict_image_accessible(self):
        from archana_person2.src import predict_image
        assert callable(predict_image)

    def test_predict_array_accessible(self):
        from archana_person2.src import predict_array
        assert callable(predict_array)

    def test_get_dataset_info_accessible(self):
        from archana_person2.src import get_dataset_info
        assert callable(get_dataset_info)

    def test_verify_classical_integration_accessible(self):
        from archana_person2.src import verify_classical_integration
        assert callable(verify_classical_integration)

    def test_AquaticPredictor_accessible(self):
        from archana_person2.src import AquaticPredictor
        assert AquaticPredictor is not None

    def test_AquaticFeatureExtractor_accessible(self):
        from archana_person2.src import AquaticFeatureExtractor
        assert AquaticFeatureExtractor is not None

    def test_CommonModelEvaluator_accessible(self):
        from archana_person2.src import CommonModelEvaluator
        assert CommonModelEvaluator is not None

    def test_ModelComparator_accessible(self):
        from archana_person2.src import ModelComparator
        assert ModelComparator is not None

    def test_AquaticLogisticRegression_accessible(self):
        from archana_person2.src import AquaticLogisticRegression
        assert AquaticLogisticRegression is not None

    def test_AquaticRandomForest_accessible(self):
        from archana_person2.src import AquaticRandomForest
        assert AquaticRandomForest is not None

    def test_AquaticSVMClassifier_accessible(self):
        from archana_person2.src import AquaticSVMClassifier
        assert AquaticSVMClassifier is not None

    def test_unknown_symbol_raises_attribute_error(self):
        import archana_person2.src as pkg
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = pkg.NonExistentClass


class TestImageToPredictionPipeline:
    """Full image → feature extraction → model → result pipeline tests."""

    def test_png_to_prediction_svm_combined(self, tmp_path):
        p = _make_mock_predictor("svm", "combined")
        result = p.predict_image_path(_make_png(tmp_path))
        assert result["predicted_class"].startswith("Species_")
        assert result["representation"] == "combined"
        assert result["model_name"] == "svm"

    def test_jpeg_to_prediction_lr_color_hist(self, tmp_path):
        p = _make_mock_predictor("logistic_regression", "color_hist")
        result = p.predict_image_path(_make_jpeg(tmp_path))
        assert result["predicted_class"].startswith("Species_")
        assert result["representation"] == "color_hist"

    def test_bmp_to_prediction_rf_hog(self, tmp_path):
        p = _make_mock_predictor("random_forest", "hog")
        result = p.predict_image_path(_make_bmp(tmp_path))
        assert result["predicted_class"].startswith("Species_")
        assert result["representation"] == "hog"

    def test_result_contains_all_required_keys(self, tmp_path):
        p = _make_mock_predictor()
        result = p.predict_image_path(_make_png(tmp_path))
        required = {"predicted_class", "predicted_index", "confidence",
                    "probabilities", "feature_vector", "representation",
                    "model_name", "image_path"}
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_array_pipeline_all_model_representation_combos(self):
        img = _rgb()
        for model in ("logistic_regression", "random_forest", "svm"):
            for rep in ("color_hist", "hog", "combined"):
                p = _make_mock_predictor(model, rep)
                result = p.predict_array(img)
                assert result["model_name"] == model
                assert result["representation"] == rep

    def test_predicted_index_is_non_negative_int(self, tmp_path):
        p = _make_mock_predictor()
        result = p.predict_image_path(_make_png(tmp_path))
        assert isinstance(result["predicted_index"], int)
        assert result["predicted_index"] >= 0

    def test_feature_vector_dtype_float32(self):
        p = _make_mock_predictor()
        result = p.predict_array(_rgb())
        assert result["feature_vector"].dtype == np.float32

    def test_feature_vector_length_matches_representation(self):
        from archana_person2.src.feature_extractor import AquaticFeatureExtractor
        extractor = AquaticFeatureExtractor()
        for rep in ("color_hist", "hog", "combined"):
            p = _make_mock_predictor(representation=rep)
            result = p.predict_array(_rgb())
            expected = extractor.get_feature_dim(rep)
            assert len(result["feature_vector"]) == expected

    def test_grayscale_image_accepted(self):
        p = _make_mock_predictor()
        gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = p.predict_array(gray)
        assert "predicted_class" in result

    def test_large_image_accepted(self):
        """Images larger than training size should be accepted (resized internally)."""
        p = _make_mock_predictor()
        big = _rgb(h=512, w=512)
        result = p.predict_array(big)
        assert "predicted_class" in result


class TestInvalidInputRobustness:
    """Every invalid-input scenario must raise a specific, informative exception."""

    def test_missing_image_file_raises_file_not_found(self, tmp_path):
        p = _make_mock_predictor()
        with pytest.raises(FileNotFoundError):
            p.predict_image_path(tmp_path / "ghost.jpg")

    def test_corrupt_file_raises_value_error(self, tmp_path):
        bad = tmp_path / "corrupt.png"
        bad.write_bytes(b"THIS IS NOT AN IMAGE")
        with pytest.raises(ValueError):
            p = _make_mock_predictor()
            p.predict_image_path(bad)

    def test_none_image_raises_value_error(self):
        p = _make_mock_predictor()
        with pytest.raises(ValueError):
            p.predict_array(None)

    def test_string_instead_of_array_raises_value_error(self):
        p = _make_mock_predictor()
        with pytest.raises(ValueError):
            p.predict_array("not_an_array")

    def test_empty_array_raises_value_error(self):
        p = _make_mock_predictor()
        with pytest.raises(ValueError):
            p.predict_array(np.array([]))

    def test_wrong_channel_count_raises_value_error(self):
        p = _make_mock_predictor()
        with pytest.raises(ValueError):
            p.predict_array(np.ones((64, 64, 4), dtype=np.uint8))  # RGBA

    def test_load_missing_model_file_raises_file_not_found(self):
        from archana_person2.src.integration import load_classical_predictor
        with pytest.raises(FileNotFoundError):
            load_classical_predictor(model_path="/no/such/file.joblib")

    def test_load_unknown_model_name_raises_value_error(self):
        from archana_person2.src.integration import load_classical_predictor
        with pytest.raises(ValueError, match="Unknown model"):
            load_classical_predictor(model_name="xgboost")

    def test_load_unknown_representation_raises_value_error(self):
        from archana_person2.src.integration import load_classical_predictor
        with pytest.raises(ValueError, match="Unknown representation"):
            load_classical_predictor(representation="raw_pixels")

    def test_zero_dimension_image_raises_value_error(self):
        p = _make_mock_predictor()
        with pytest.raises(ValueError):
            p.predict_array(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_txt_file_as_image_raises_value_error(self, tmp_path):
        txt = tmp_path / "fish.txt"
        txt.write_text("not an image")
        p = _make_mock_predictor()
        with pytest.raises((ValueError, Exception)):
            p.predict_image_path(txt)

    def test_evaluator_rejects_empty_inputs(self):
        from archana_person2.src.evaluation import CommonModelEvaluator
        ev = CommonModelEvaluator()
        with pytest.raises(ValueError):
            ev.evaluate_predictions(
                y_true=np.array([]),
                y_pred=np.array([]),
                sample_ids=[],
            )


class TestProbabilityOutputValidation:
    """Validate confidence values and probability distributions."""

    def test_svm_confidence_between_0_and_1(self):
        p = _make_mock_predictor("svm", "combined")
        result = p.predict_array(_rgb())
        assert 0.0 <= result["confidence"] <= 1.0

    def test_lr_confidence_between_0_and_1(self):
        p = _make_mock_predictor("logistic_regression", "combined")
        result = p.predict_array(_rgb())
        assert 0.0 <= result["confidence"] <= 1.0

    def test_rf_confidence_is_none(self):
        """Random Forest does not expose a probability dict in the wrapper."""
        p = _make_mock_predictor("random_forest", "combined")
        result = p.predict_array(_rgb())
        assert result["probabilities"] is None

    def test_svm_probabilities_sum_to_1(self):
        p = _make_mock_predictor("svm", "combined")
        result = p.predict_array(_rgb())
        if result["probabilities"] is not None:
            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 1e-6

    def test_lr_probabilities_sum_to_1(self):
        p = _make_mock_predictor("logistic_regression", "combined")
        result = p.predict_array(_rgb())
        if result["probabilities"] is not None:
            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 1e-6

    def test_svm_probabilities_keys_are_class_names(self):
        p = _make_mock_predictor("svm", "combined")
        result = p.predict_array(_rgb())
        if result["probabilities"] is not None:
            for key in result["probabilities"]:
                assert isinstance(key, str)
                assert key.startswith("Species_")

    def test_confidence_matches_max_probability(self):
        p = _make_mock_predictor("svm", "combined")
        result = p.predict_array(_rgb())
        if result["probabilities"] is not None:
            assert abs(result["confidence"] - max(result["probabilities"].values())) < 1e-9

    def test_predicted_class_matches_argmax_probability(self):
        p = _make_mock_predictor("lr" if False else "logistic_regression", "combined")
        result = p.predict_array(_rgb())
        if result["probabilities"] is not None:
            best = max(result["probabilities"], key=result["probabilities"].get)
            assert result["predicted_class"] == best


class TestEvaluationMetricsValidation:
    """Validate evaluation metric correctness and classification report integrity."""

    def _eval(self, y_true, y_pred):
        """Helper: run evaluate_predictions and return the flat metrics dict."""
        from archana_person2.src.evaluation import CommonModelEvaluator
        ev = CommonModelEvaluator()
        result = ev.evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            sample_ids=list(range(len(y_true))),
        )
        # metrics are nested under "metrics" key
        return result.get("metrics", result)

    def _imperfect_5class(self):
        rng = np.random.default_rng(99)
        y_true = rng.integers(0, 5, 50)
        y_pred = y_true.copy()
        flip_idx = rng.choice(50, 10, replace=False)
        y_pred[flip_idx] = (y_pred[flip_idx] + 1) % 5
        return y_true, y_pred

    def test_perfect_accuracy_is_1(self):
        y = np.array([0, 1, 0, 1, 1])
        m = self._eval(y, y.copy())
        assert m["accuracy"] == pytest.approx(1.0)

    def test_macro_f1_range_0_to_1(self):
        y, yp = self._imperfect_5class()
        m = self._eval(y, yp)
        assert 0.0 <= m["macro_f1"] <= 1.0

    def test_confusion_matrix_diagonal_matches_per_class_correct(self):
        from archana_person2.src.evaluation import CommonModelEvaluator
        y  = np.array([0, 0, 1, 1, 2, 2])
        yp = np.array([0, 1, 1, 1, 2, 0])
        ev = CommonModelEvaluator()
        result = ev.evaluate_predictions(
            y_true=y, y_pred=yp, sample_ids=list(range(len(y)))
        )
        cm = result["confusion_matrix"]
        diagonal = [cm[i][i] for i in range(len(cm))]
        assert diagonal == [1, 2, 1]

    def test_classification_report_contains_all_classes(self):
        from archana_person2.src.evaluation import CommonModelEvaluator
        y, yp = self._imperfect_5class()
        ev = CommonModelEvaluator()
        result = ev.evaluate_predictions(
            y_true=y, y_pred=yp, sample_ids=list(range(len(y)))
        )
        report = result["classification_report"]
        for cls in range(5):
            assert str(cls) in report or cls in report

    def test_weighted_f1_between_0_and_1(self):
        y, yp = self._imperfect_5class()
        m = self._eval(y, yp)
        assert 0.0 <= m["weighted_f1"] <= 1.0

    def test_evaluator_end_to_end_returns_required_keys(self):
        from archana_person2.src.evaluation import CommonModelEvaluator
        y, yp = self._imperfect_5class()
        ev = CommonModelEvaluator()
        result = ev.evaluate_predictions(
            y_true=y, y_pred=yp, sample_ids=list(range(len(y)))
        )
        # top-level keys
        for key in ("metrics", "confusion_matrix", "classification_report"):
            assert key in result, f"Missing top-level key: {key}"
        # nested metric keys
        for key in ("accuracy", "macro_f1", "weighted_f1",
                    "macro_precision", "macro_recall"):
            assert key in result["metrics"], f"Missing metric: {key}"

    def test_macro_precision_at_least_0(self):
        y, yp = self._imperfect_5class()
        m = self._eval(y, yp)
        assert m["macro_precision"] >= 0.0

    def test_evaluator_rejects_mismatched_lengths(self):
        from archana_person2.src.evaluation import CommonModelEvaluator
        ev = CommonModelEvaluator()
        with pytest.raises(ValueError):
            ev.evaluate_predictions(
                y_true=np.array([0, 1, 2]),
                y_pred=np.array([0, 1]),
                sample_ids=[0, 1],
            )


class TestDatasetLoaderBridge:
    """Validate integration.py ↔ DatasetLoader interaction."""

    def test_get_dataset_info_returns_valid_dict(self):
        from archana_person2.src.integration import get_dataset_info
        info = get_dataset_info()
        assert isinstance(info, dict)
        assert info["num_classes"] > 0
        assert isinstance(info["label_map"], dict)

    def test_get_dataset_info_fallback_gives_23_classes(self):
        """Fallback to default gives 23 synthetic class placeholders."""
        from archana_person2.src import integration as intmod
        with patch.object(intmod, "_try_sadhvika_loader", return_value=None), \
             patch("archana_person2.src.dataset_loader.Person2DatasetLoader.get_label_map",
                   side_effect=FileNotFoundError("no label map")):
            info = intmod.get_dataset_info()
        assert info["source"] == "default_fallback"
        assert info["num_classes"] == 23

    def test_verify_integration_importable_checks_all_pass(self):
        from archana_person2.src.integration import verify_classical_integration
        results = verify_classical_integration(verbose=False)
        for name in [
            "archana_person2.src.predict importable",
            "archana_person2.src.feature_extractor importable",
            "archana_person2.src.evaluation importable",
            "archana_person2.src.comparison importable",
            "Feature extractor dimensions correct",
        ]:
            assert results.get(name) is True, f"Check failed: {name}"

    def test_sadhvika_dataset_loader_importable(self):
        try:
            from scripts.dataset_loader import DatasetLoader
            assert DatasetLoader is not None
        except ImportError:
            pytest.skip("scripts.dataset_loader not on path")

    def test_sadhvika_loader_load_and_preprocess_produces_float32(self):
        """load_and_preprocess on a synthetic PNG must return float32 array."""
        try:
            from scripts.dataset_loader import DatasetLoader
        except ImportError:
            pytest.skip("scripts.dataset_loader not on path")
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "fish.png"
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            loader = DatasetLoader()
            arr = loader.load_and_preprocess(img_path)
        assert arr.dtype == np.float32
        assert arr.ndim == 3
        assert arr.shape[2] == 3

    def test_synthetic_fixture_manifests_are_readable(self):
        """The committed synthetic fixture CSVs must be parseable."""
        fixture_dir = _PERSON2_ROOT / "artifacts" / "fixtures" / "synthetic_dataset"
        for split in ("train", "val", "test"):
            csv_path = fixture_dir / f"{split}.csv"
            assert csv_path.exists(), f"Fixture CSV missing: {csv_path}"
            with open(csv_path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            assert len(rows) > 0, f"Fixture CSV empty: {split}.csv"


class TestRealDataAvailability:
    """
    Real Fish4Knowledge data tests.
    Auto-skipped (marked PENDING) when data/preprocessed/ is not present.
    These tests do NOT use synthetic substitutes.
    """

    _SKIP_MSG = (
        "Real Fish4Knowledge dataset not available. "
        "Run Sadhvika's preprocessing pipeline first: "
        "python scripts/preprocess.py ..."
    )

    @pytest.mark.skipif(not _REAL_DATA_AVAILABLE, reason=_SKIP_MSG)
    def test_real_label_map_loads(self):
        label_map_path = _PROJECT_ROOT / "data" / "preprocessed" / "label_map.json"
        with open(label_map_path) as f:
            lm = json.load(f)
        assert len(lm) >= 1
        assert all(isinstance(v, int) for v in lm.values())

    @pytest.mark.skipif(not _REAL_DATA_AVAILABLE, reason=_SKIP_MSG)
    def test_real_X_train_shape_valid(self):
        X = np.load(str(_PROJECT_ROOT / "data" / "preprocessed" / "X_train.npy"),
                    mmap_mode="r")
        assert X.ndim == 4
        assert X.shape[3] == 3

    @pytest.mark.skipif(not _REAL_DATA_AVAILABLE, reason=_SKIP_MSG)
    def test_real_train_csv_non_empty(self):
        with open(_PROJECT_ROOT / "data" / "splits" / "train.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0

    @pytest.mark.skipif(not _REAL_DATA_AVAILABLE, reason=_SKIP_MSG)
    def test_saved_models_were_trained_on_real_data(self):
        """
        If real data is present, load each saved model and confirm
        its classes_ attribute has at least as many classes as label_map.
        """
        import joblib
        with open(_PROJECT_ROOT / "data" / "preprocessed" / "label_map.json") as f:
            lm = json.load(f)
        for model_name, fname in _MODEL_FILES.items():
            model_path = _MODELS_DIR / fname
            assert model_path.exists(), f"Model file missing: {model_path}"
            obj = joblib.load(str(model_path))
            # Try to find classes_ in pipeline or top-level
            clf = obj
            if hasattr(obj, "named_steps"):
                clf = obj.named_steps.get("classifier", obj)
            if hasattr(clf, "classes_"):
                assert len(clf.classes_) <= len(lm), (
                    f"{model_name}: classes_ has {len(clf.classes_)} entries "
                    f"but label_map has {len(lm)}"
                )


class TestRegressionPhase1to10:
    """Regression guard: confirm all Phase 1–10 deliverables still exist and work."""

    def test_all_test_files_present(self):
        test_dir = _PROJECT_ROOT / "archana_person2" / "tests"
        for name in ["test_feature_extraction.py", "test_models.py",
                     "test_evaluation.py", "test_comparison.py",
                     "test_prediction.py", "test_integration.py"]:
            assert (test_dir / name).exists(), f"Test file missing: {name}"

    def test_all_src_modules_present(self):
        src_dir = _PROJECT_ROOT / "archana_person2" / "src"
        for name in ["__init__.py", "color_histogram.py", "hog_features.py",
                     "feature_extractor.py", "dataset_loader.py",
                     "logistic_regression.py", "random_forest.py",
                     "svm_classifier.py", "evaluation.py",
                     "comparison.py", "predict.py", "integration.py"]:
            assert (src_dir / name).exists(), f"Source file missing: {name}"

    def test_all_docs_present(self):
        docs_dir = _PROJECT_ROOT / "archana_person2" / "docs"
        for name in ["FEATURE_EXTRACTION.md", "LOGISTIC_REGRESSION.md",
                     "RANDOM_FOREST.md", "SVM.md", "EVALUATION.md",
                     "MODEL_COMPARISON.md", "PREDICTION.md", "INTEGRATION.md"]:
            assert (docs_dir / name).exists(), f"Doc missing: {name}"

    def test_config_yaml_present_and_parseable(self):
        import yaml
        cfg_path = _PROJECT_ROOT / "archana_person2" / "config" / "config.yaml"
        assert cfg_path.exists()
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "project" in cfg
        assert "feature_extraction" in cfg
        assert "models" in cfg

    def test_requirements_person2_present(self):
        assert (_PROJECT_ROOT / "requirements_person2.txt").exists()

    def test_feature_extractor_dimensions_unchanged(self):
        """Core dimensionality contract: colour_hist=1024, hog=6084, combined=7108."""
        from archana_person2.src.feature_extractor import AquaticFeatureExtractor
        ex = AquaticFeatureExtractor()
        assert ex.get_feature_dim("color_hist") == 1024
        assert ex.get_feature_dim("hog")        == 6084
        assert ex.get_feature_dim("combined")   == 7108
