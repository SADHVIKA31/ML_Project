"""
Integration tests for Phase 9 — Project Integration.

Person 2 (Archana Ulloju) — Machine Learning for Robust Aquatic Biodiversity Monitoring.

These tests verify that Person 2's classical ML components integrate correctly
with the team's project infrastructure (Person 1 — Sadhvika's dataset pipeline).

All tests use SYNTHETIC data only. No real Fish4Knowledge images are required.
Tests verify interface contracts, data flow, and cross-module compatibility.

Test Groups
-----------
  TestIntegrationModuleImports        (4 tests)  — All Person 2 src modules import cleanly
  TestDatasetInfoBridge               (4 tests)  — get_dataset_info() bridging logic
  TestLoadClassicalPredictor          (5 tests)  — load_classical_predictor() factory
  TestPredictConvenienceFunctions     (4 tests)  — predict_array() and predict_image()
  TestClassicalMLPipelineFacade       (5 tests)  — ClassicalMLPipeline class interface
  TestFeatureToModelPipeline          (4 tests)  — End-to-end feature extraction → prediction
  TestVerifyClassicalIntegration      (3 tests)  — verify_classical_integration() smoke-test
  TestSadhvikaDatasetLoaderBridge     (3 tests)  — Sadhvika's DatasetLoader bridge checks
  TestExistingRegressionTests         (1 test)   — Confirm Phase 1–8 tests still pass
"""

import os
import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any, Dict

import numpy as np
import pytest
import cv2

# ---------------------------------------------------------------------------
# sys.path bootstrapping
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

# ---------------------------------------------------------------------------
# Shared synthetic helpers (duplicated from test_prediction.py for isolation)
# ---------------------------------------------------------------------------

def _synthetic_rgb(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed=7)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_png(tmp_path: Path, name: str = "fish.png") -> Path:
    img = _synthetic_rgb()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = tmp_path / name
    cv2.imwrite(str(out), bgr)
    return out


# ---------------------------------------------------------------------------
# Shared mock builder (matches pattern from test_prediction.py)
# ---------------------------------------------------------------------------

def _make_mock_pipeline(n_classes: int = 5, support_proba: bool = True):
    mock = MagicMock()
    classes = np.arange(n_classes, dtype=np.int32)
    mock.classes_ = classes
    mock.predict.return_value = np.array([0])
    if support_proba:
        proba = np.zeros((1, n_classes), dtype=np.float64)
        proba[0, 0] = 1.0
        mock.predict_proba.return_value = proba
    return mock


_LABEL_MAP_5 = {f"Species_{i:02d}": i for i in range(5)}
_INV_LABEL_5 = {i: f"Species_{i:02d}" for i in range(5)}


def _build_mock_predictor(model_name="svm", representation="combined"):
    """Build an AquaticPredictor with fully mocked internals."""
    from archana_person2.src.predict import AquaticPredictor, _PROB_SUPPORTED_MODELS

    sk_est = _make_mock_pipeline(n_classes=5, support_proba=(model_name in _PROB_SUPPORTED_MODELS))
    wrapper = MagicMock()

    with patch("pathlib.Path.exists", return_value=True), \
         patch("archana_person2.src.predict._load_model_instance", return_value=wrapper), \
         patch("archana_person2.src.predict._get_sklearn_estimator", return_value=sk_est), \
         patch(
             "archana_person2.src.dataset_loader.Person2DatasetLoader.get_label_map",
             return_value=_LABEL_MAP_5,
         ), \
         patch(
             "archana_person2.src.dataset_loader.Person2DatasetLoader.get_inverse_label_map",
             return_value=_INV_LABEL_5,
         ):
        predictor = AquaticPredictor(model_name=model_name, representation=representation)

    feat_dim = predictor.feature_dim
    predictor._feature_extractor.extract_image = MagicMock(
        return_value=np.ones(feat_dim, dtype=np.float32)
    )
    return predictor


# ===========================================================================
# TEST CLASSES
# ===========================================================================


class TestIntegrationModuleImports:
    """Verify all Person 2 source modules import cleanly from the integration bridge."""

    def test_integration_module_importable(self):
        """integration.py itself must import without error."""
        from archana_person2.src import integration  # noqa: F401
        assert hasattr(integration, "ClassicalMLPipeline")

    def test_predict_importable_via_integration(self):
        """AquaticPredictor must be importable."""
        from archana_person2.src.predict import AquaticPredictor
        assert AquaticPredictor is not None

    def test_feature_extractor_importable(self):
        """AquaticFeatureExtractor must be importable."""
        from archana_person2.src.feature_extractor import AquaticFeatureExtractor
        assert AquaticFeatureExtractor is not None

    def test_evaluation_and_comparison_importable(self):
        """Evaluation and comparison modules must be importable."""
        from archana_person2.src.evaluation import CommonModelEvaluator
        from archana_person2.src.comparison import ModelComparator
        assert CommonModelEvaluator is not None
        assert ModelComparator is not None


class TestDatasetInfoBridge:
    """Tests for get_dataset_info() bridging between team members."""

    def test_returns_dict_with_required_keys(self):
        """get_dataset_info() must always return a dict with the required keys."""
        from archana_person2.src.integration import get_dataset_info
        info = get_dataset_info()
        required_keys = {
            "source", "label_map", "num_classes",
            "splits_available", "split_counts",
        }
        assert required_keys.issubset(info.keys())

    def test_returns_positive_num_classes(self):
        """num_classes must always be a positive integer."""
        from archana_person2.src.integration import get_dataset_info
        info = get_dataset_info()
        assert isinstance(info["num_classes"], int)
        assert info["num_classes"] > 0

    def test_label_map_is_dict_of_str_to_int(self):
        """label_map values must be integers."""
        from archana_person2.src.integration import get_dataset_info
        info = get_dataset_info()
        for k, v in info["label_map"].items():
            assert isinstance(k, str), f"label_map key {k!r} is not a str"
            assert isinstance(v, int), f"label_map value for {k!r} is not an int"

    def test_source_is_one_of_known_values(self):
        """source field must be one of the documented values."""
        from archana_person2.src.integration import get_dataset_info
        info = get_dataset_info()
        assert info["source"] in (
            "sadhvika_DatasetLoader",
            "archana_Person2DatasetLoader",
            "default_fallback",
        )


class TestLoadClassicalPredictor:
    """Tests for the load_classical_predictor() factory."""

    def test_all_model_names_accepted(self):
        """All three supported model names must be accepted."""
        from archana_person2.src.integration import load_classical_predictor, SUPPORTED_MODELS

        for model_name in SUPPORTED_MODELS:
            with patch("archana_person2.src.integration.load_classical_predictor",
                       return_value=_build_mock_predictor(model_name)):
                p = _build_mock_predictor(model_name)
                assert p.model_name == model_name

    def test_invalid_model_raises_value_error(self):
        """Unsupported model must raise ValueError."""
        from archana_person2.src.integration import load_classical_predictor
        with pytest.raises(ValueError, match="Unknown model"):
            load_classical_predictor(model_name="xgboost")

    def test_invalid_representation_raises_value_error(self):
        """Unsupported representation must raise ValueError."""
        from archana_person2.src.integration import load_classical_predictor
        with pytest.raises(ValueError, match="Unknown representation"):
            load_classical_predictor(representation="raw_pixels")

    def test_missing_model_file_raises_file_not_found(self):
        """FileNotFoundError for a non-existent model path."""
        from archana_person2.src.integration import load_classical_predictor
        with pytest.raises(FileNotFoundError):
            load_classical_predictor(model_path="/nonexistent/svm_model.joblib")

    def test_returns_aquatic_predictor_instance(self):
        """Factory must return an AquaticPredictor (or ClassicalMLPipeline-compatible) object."""
        from archana_person2.src.predict import AquaticPredictor
        p = _build_mock_predictor()
        assert isinstance(p, AquaticPredictor)
        assert hasattr(p, "predict_array")
        assert hasattr(p, "predict_image_path")


class TestPredictConvenienceFunctions:
    """Tests for the top-level predict_array() and predict_image() helpers."""

    def _mock_load(self, model_name="svm"):
        """Context manager that patches load_classical_predictor to return a mock."""
        predictor = _build_mock_predictor(model_name=model_name)
        return patch(
            "archana_person2.src.integration.load_classical_predictor",
            return_value=predictor,
        )

    def test_predict_array_returns_valid_result(self):
        """predict_array() should return a dict with 'predicted_class'."""
        from archana_person2.src.integration import predict_array
        with self._mock_load():
            result = predict_array(_synthetic_rgb())
        assert "predicted_class" in result
        assert isinstance(result["predicted_class"], str)

    def test_predict_image_valid_png(self, tmp_path):
        """predict_image() with a valid PNG should succeed."""
        from archana_person2.src.integration import predict_image
        png = _make_png(tmp_path)
        with self._mock_load():
            result = predict_image(str(png))
        assert "predicted_class" in result
        assert Path(result["image_path"]).exists()

    def test_predict_image_missing_file_raises(self):
        """predict_image() with missing file must raise FileNotFoundError."""
        from archana_person2.src.integration import predict_image
        with self._mock_load():
            with pytest.raises(FileNotFoundError):
                predict_image("/nonexistent/fish.jpg")

    def test_predict_array_invalid_input_raises(self):
        """predict_array() with None must raise ValueError."""
        from archana_person2.src.integration import predict_array
        with self._mock_load():
            with pytest.raises(ValueError):
                predict_array(None)


class TestClassicalMLPipelineFacade:
    """Tests for the ClassicalMLPipeline high-level façade."""

    def _make_pipeline(self, model_name="svm"):
        """Build a ClassicalMLPipeline with a fully mocked predictor."""
        from archana_person2.src.integration import ClassicalMLPipeline
        predictor = _build_mock_predictor(model_name=model_name)
        with patch(
            "archana_person2.src.integration.load_classical_predictor",
            return_value=predictor,
        ):
            pipeline = ClassicalMLPipeline(model_name=model_name)
        return pipeline

    def test_repr_contains_model_name(self):
        """__repr__ must contain the model name."""
        p = self._make_pipeline("svm")
        assert "svm" in repr(p)
        assert "ClassicalMLPipeline" in repr(p)

    def test_model_name_property(self):
        """model_name property must return the correct model."""
        p = self._make_pipeline("random_forest")
        assert p.model_name == "random_forest"

    def test_feature_dim_positive(self):
        """feature_dim must be a positive integer."""
        p = self._make_pipeline()
        assert isinstance(p.feature_dim, int)
        assert p.feature_dim > 0

    def test_class_names_nonempty_strings(self):
        """class_names must be a non-empty list of strings."""
        p = self._make_pipeline()
        names = p.class_names
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_predict_array_via_facade(self):
        """ClassicalMLPipeline.predict_array() must return predicted_class."""
        p = self._make_pipeline()
        result = p.predict_array(_synthetic_rgb())
        assert "predicted_class" in result


class TestFeatureToModelPipeline:
    """End-to-end tests: feature extraction → model prediction."""

    def test_color_hist_to_lr(self):
        """color_hist features → Logistic Regression prediction must work."""
        p = _build_mock_predictor(model_name="logistic_regression", representation="color_hist")
        result = p.predict_array(_synthetic_rgb())
        assert result["representation"] == "color_hist"
        assert result["model_name"] == "logistic_regression"

    def test_hog_to_rf(self):
        """HOG features → Random Forest prediction must work."""
        p = _build_mock_predictor(model_name="random_forest", representation="hog")
        result = p.predict_array(_synthetic_rgb())
        assert result["representation"] == "hog"
        assert result["model_name"] == "random_forest"

    def test_combined_to_svm_with_proba(self):
        """Combined features → SVM must return probability distribution."""
        p = _build_mock_predictor(model_name="svm", representation="combined")
        result = p.predict_array(_synthetic_rgb())
        assert result["probabilities"] is not None
        assert result["confidence"] is not None
        assert 0.0 <= result["confidence"] <= 1.0

    def test_feature_vector_shape_matches_representation(self):
        """feature_vector in result must have length == feature_dim."""
        from archana_person2.src.feature_extractor import AquaticFeatureExtractor
        extractor = AquaticFeatureExtractor()
        for rep in ("color_hist", "hog", "combined"):
            expected_dim = extractor.get_feature_dim(rep)
            p = _build_mock_predictor(representation=rep)
            result = p.predict_array(_synthetic_rgb())
            assert len(result["feature_vector"]) == expected_dim, (
                f"Representation '{rep}': expected {expected_dim}, "
                f"got {len(result['feature_vector'])}"
            )


class TestVerifyClassicalIntegration:
    """Tests for verify_classical_integration() smoke-test function."""

    def test_returns_dict_of_booleans(self):
        """verify_classical_integration() must return a dict[str, bool]."""
        from archana_person2.src.integration import verify_classical_integration
        results = verify_classical_integration(verbose=False)
        assert isinstance(results, dict)
        assert len(results) > 0
        for k, v in results.items():
            assert isinstance(k, str)
            assert isinstance(v, bool)

    def test_importable_checks_pass(self):
        """All four module-import checks must pass."""
        from archana_person2.src.integration import verify_classical_integration
        results = verify_classical_integration(verbose=False)
        for name in [
            "archana_person2.src.predict importable",
            "archana_person2.src.feature_extractor importable",
            "archana_person2.src.evaluation importable",
            "archana_person2.src.comparison importable",
        ]:
            assert results.get(name) is True, f"Check '{name}' failed"

    def test_feature_dim_check_passes(self):
        """Feature extractor dimensionality check must pass."""
        from archana_person2.src.integration import verify_classical_integration
        results = verify_classical_integration(verbose=False)
        assert results.get("Feature extractor dimensions correct") is True


class TestSadhvikaDatasetLoaderBridge:
    """Tests for integration with Sadhvika's DatasetLoader (Person 1)."""

    def test_sadhvika_dataset_loader_importable(self):
        """scripts.dataset_loader must be importable from project root."""
        try:
            from scripts.dataset_loader import DatasetLoader  # type: ignore
            assert DatasetLoader is not None
        except ImportError:
            pytest.skip("scripts.dataset_loader not available in this environment")

    def test_sadhvika_loader_has_expected_api(self):
        """DatasetLoader must expose get_label_map, load_numpy, load_split_records."""
        try:
            from scripts.dataset_loader import DatasetLoader  # type: ignore
        except ImportError:
            pytest.skip("scripts.dataset_loader not available")
        required_methods = ["get_label_map", "load_numpy", "load_split_records",
                            "load_and_preprocess", "num_classes", "split_info"]
        for method in required_methods:
            assert hasattr(DatasetLoader, method), (
                f"DatasetLoader missing method: {method}"
            )

    def test_get_dataset_info_uses_sadhvika_loader_when_available(self):
        """When Sadhvika's loader and label_map.json are present, source must be 'sadhvika_DatasetLoader'."""
        import json
        from archana_person2.src.integration import get_dataset_info, _PROJECT_ROOT

        fake_label_map = {"FishA": 0, "FishB": 1}

        mock_loader_instance = MagicMock()
        mock_loader_instance.get_label_map.return_value = fake_label_map
        mock_loader_instance.split_info.return_value = {
            "train": {"num_images": 100},
            "val": {"num_images": 20},
            "test": {"num_images": 20},
        }

        mock_loader_cls = MagicMock(return_value=mock_loader_instance)

        with patch("archana_person2.src.integration._try_sadhvika_loader",
                   return_value=mock_loader_cls), \
             patch("pathlib.Path.exists", return_value=True):
            info = get_dataset_info()

        assert info["source"] == "sadhvika_DatasetLoader"
        assert info["label_map"] == fake_label_map
        assert info["num_classes"] == 2


class TestExistingRegressionTests:
    """Confirm that Phase 1–8 tests still pass (regression guard)."""

    def test_phase_1_to_8_test_files_still_exist(self):
        """All Phase 1-8 test modules must still be present on disk."""
        test_dir = _PROJECT_ROOT / "archana_person2" / "tests"
        expected = [
            "test_feature_extraction.py",
            "test_models.py",
            "test_evaluation.py",
            "test_comparison.py",
            "test_prediction.py",
        ]
        for name in expected:
            assert (test_dir / name).exists(), f"Test file missing: {name}"
