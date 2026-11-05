"""
Unit tests for the Phase 8 Prediction Pipeline.

Person 2 (Archana Ulloju) — Machine Learning for Robust Aquatic Biodiversity Monitoring.

All tests use SYNTHETIC data only. No real Fish4Knowledge images are used.
Tests verify software correctness: loading, feature extraction, prediction shape,
error handling, and interface contracts.

Tests
-----
  TestAquaticPredictorInit
      - test_valid_model_names_accepted
      - test_invalid_model_name_raises_value_error
      - test_invalid_representation_raises_value_error
      - test_missing_model_file_raises_file_not_found
      - test_repr_string

  TestPredictArray
      - test_predict_array_returns_expected_keys
      - test_predict_array_rgb_uint8
      - test_predict_array_greyscale_converted
      - test_predict_array_all_models_and_representations
      - test_predict_array_probabilities_for_lr_and_svm
      - test_predict_array_rf_no_probabilities_key_is_none
      - test_predict_array_invalid_none_raises
      - test_predict_array_invalid_non_ndarray_raises
      - test_predict_array_empty_array_raises
      - test_predict_array_wrong_channels_raises

  TestPredictImagePath
      - test_predict_image_path_valid_png
      - test_predict_image_path_missing_file_raises
      - test_predict_image_path_corrupt_file_raises

  TestPredictorProperties
      - test_feature_dim_matches_representation
      - test_class_names_is_nonempty_list_of_strings
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any

import numpy as np
import pytest
import cv2

# ---------------------------------------------------------------------------
# sys.path bootstrapping (consistent with other test files in this project)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Suppress matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------

def _synthetic_rgb(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a deterministic random RGB uint8 image."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_png(tmp_path: Path, name: str = "test.png") -> Path:
    """Write a synthetic image to disk and return its path."""
    img = _synthetic_rgb()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = tmp_path / name
    cv2.imwrite(str(out), bgr)
    return out


# ---------------------------------------------------------------------------
# Fixtures: mock trained sklearn Pipelines / estimators
# ---------------------------------------------------------------------------

def _make_mock_pipeline(n_classes: int = 5, support_proba: bool = True):
    """Return a mock sklearn Pipeline-like object."""
    mock = MagicMock()
    classes = np.arange(n_classes, dtype=np.int32)
    mock.classes_ = classes
    mock.predict.return_value = np.array([0])
    if support_proba:
        proba = np.zeros((1, n_classes), dtype=np.float64)
        proba[0, 0] = 1.0
        mock.predict_proba.return_value = proba
    else:
        del mock.predict_proba
    return mock


def _make_mock_rf_model(n_classes: int = 5):
    """Return a mock RandomForestClassifier-like object (no predict_proba in public API)."""
    mock = MagicMock()
    mock.classes_ = np.arange(n_classes, dtype=np.int32)
    mock.predict.return_value = np.array([2])
    # RF does have predict_proba, but AquaticPredictor only checks _PROB_SUPPORTED_MODELS
    proba = np.zeros((1, n_classes), dtype=np.float64)
    proba[0, 2] = 1.0
    mock.predict_proba.return_value = proba
    return mock


# ---------------------------------------------------------------------------
# Factory: build AquaticPredictor with mocked internals
# ---------------------------------------------------------------------------

def _build_predictor(
    model_name: str = "svm",
    representation: str = "combined",
    n_classes: int = 5,
):
    """
    Instantiate AquaticPredictor without loading a real file from disk.

    We patch:
      1. ``pathlib.Path.exists`` → True, so the model-file check passes.
      2. ``_load_model_instance`` → returns a MagicMock wrapper.
      3. ``_get_sklearn_estimator`` → returns a mock sklearn estimator.
      4. ``Person2DatasetLoader.get_label_map`` and
         ``Person2DatasetLoader.get_inverse_label_map`` → synthetic maps
         so no filesystem access is required for label loading.
    """
    from archana_person2.src.predict import AquaticPredictor, _PROB_SUPPORTED_MODELS

    # Build synthetic label maps
    _label_map = {f"Species_{i:02d}": i for i in range(n_classes)}
    _inv_label_map = {i: f"Species_{i:02d}" for i in range(n_classes)}

    # Set up mock sklearn estimator
    if model_name in _PROB_SUPPORTED_MODELS:
        sk_est = _make_mock_pipeline(n_classes=n_classes, support_proba=True)
    else:
        sk_est = _make_mock_rf_model(n_classes=n_classes)

    wrapper = MagicMock()

    with patch("pathlib.Path.exists", return_value=True), \
         patch("archana_person2.src.predict._load_model_instance", return_value=wrapper), \
         patch("archana_person2.src.predict._get_sklearn_estimator", return_value=sk_est), \
         patch(
             "archana_person2.src.dataset_loader.Person2DatasetLoader.get_label_map",
             return_value=_label_map,
         ), \
         patch(
             "archana_person2.src.dataset_loader.Person2DatasetLoader.get_inverse_label_map",
             return_value=_inv_label_map,
         ):
        predictor = AquaticPredictor(
            model_name=model_name,
            representation=representation,
        )

    # Inject deterministic feature extractor behaviour
    feat_dim = predictor.feature_dim
    predictor._feature_extractor.extract_image = MagicMock(
        return_value=np.ones(feat_dim, dtype=np.float32)
    )
    return predictor


# ===========================================================================
# TEST CLASSES
# ===========================================================================

class TestAquaticPredictorInit:
    """Tests for AquaticPredictor initialisation."""

    def test_valid_model_names_accepted(self):
        """All three supported model names should be accepted without error."""
        from archana_person2.src.predict import SUPPORTED_MODELS

        for m in SUPPORTED_MODELS:
            p = _build_predictor(model_name=m)
            assert p.model_name == m

    def test_invalid_model_name_raises_value_error(self):
        """An unrecognised model name must raise ValueError."""
        from archana_person2.src.predict import AquaticPredictor

        with pytest.raises(ValueError, match="Unsupported model"):
            AquaticPredictor(model_name="xgboost")

    def test_invalid_representation_raises_value_error(self):
        """An unrecognised representation must raise ValueError."""
        from archana_person2.src.predict import AquaticPredictor

        with patch("pathlib.Path.exists", return_value=True), \
             patch("archana_person2.src.predict._load_model_instance", return_value=MagicMock()), \
             patch("archana_person2.src.predict._get_sklearn_estimator", return_value=_make_mock_pipeline()):
            with pytest.raises(ValueError, match="Unsupported representation"):
                AquaticPredictor(model_name="svm", representation="raw_pixels")

    def test_missing_model_file_raises_file_not_found(self):
        """FileNotFoundError is raised when the .joblib file does not exist."""
        from archana_person2.src.predict import AquaticPredictor

        with pytest.raises(FileNotFoundError):
            AquaticPredictor(
                model_name="svm",
                model_path="/nonexistent/path/svm_model.joblib",
            )

    def test_repr_string(self):
        """__repr__ must return a string containing the model name."""
        p = _build_predictor(model_name="random_forest")
        r = repr(p)
        assert "random_forest" in r
        assert "AquaticPredictor" in r


class TestPredictArray:
    """Tests for AquaticPredictor.predict_array."""

    def test_predict_array_returns_expected_keys(self):
        """Result dict must contain all required keys."""
        p = _build_predictor()
        result = p.predict_array(_synthetic_rgb())
        required = {
            "predicted_class", "predicted_index", "confidence",
            "probabilities", "feature_vector", "representation", "model_name",
        }
        assert required.issubset(result.keys())

    def test_predict_array_rgb_uint8(self):
        """Standard RGB uint8 input should produce a valid prediction."""
        p = _build_predictor(model_name="svm")
        result = p.predict_array(_synthetic_rgb())
        assert isinstance(result["predicted_class"], str)
        assert isinstance(result["predicted_index"], int)

    def test_predict_array_greyscale_converted(self):
        """Greyscale (H×W) input should be converted to 3-channel and not raise."""
        p = _build_predictor()
        grey = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = p.predict_array(grey)
        assert isinstance(result["predicted_class"], str)

    def test_predict_array_all_models_and_representations(self):
        """All 9 combinations of (model, representation) should return valid results."""
        from archana_person2.src.predict import SUPPORTED_MODELS, SUPPORTED_REPRESENTATIONS
        img = _synthetic_rgb()
        for m in SUPPORTED_MODELS:
            for rep in SUPPORTED_REPRESENTATIONS:
                p = _build_predictor(model_name=m, representation=rep)
                result = p.predict_array(img)
                assert result["model_name"] == m
                assert result["representation"] == rep

    def test_predict_array_probabilities_for_lr_and_svm(self):
        """Logistic Regression and SVM should return a non-None probabilities dict."""
        for m in ("logistic_regression", "svm"):
            p = _build_predictor(model_name=m)
            result = p.predict_array(_synthetic_rgb())
            assert result["probabilities"] is not None, f"{m} should expose probabilities"
            assert isinstance(result["confidence"], float)
            assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_array_rf_no_probabilities_key_is_none(self):
        """Random Forest is NOT in _PROB_SUPPORTED_MODELS, so probabilities must be None."""
        p = _build_predictor(model_name="random_forest")
        result = p.predict_array(_synthetic_rgb())
        # probabilities key must exist but value should be None for RF
        assert "probabilities" in result
        assert result["probabilities"] is None
        assert result["confidence"] is None

    def test_predict_array_invalid_none_raises(self):
        """Passing None as image must raise ValueError."""
        p = _build_predictor()
        with pytest.raises(ValueError):
            p.predict_array(None)

    def test_predict_array_invalid_non_ndarray_raises(self):
        """Passing a list instead of an ndarray must raise ValueError."""
        p = _build_predictor()
        with pytest.raises(ValueError):
            p.predict_array([[1, 2, 3], [4, 5, 6]])

    def test_predict_array_empty_array_raises(self):
        """Passing an empty ndarray must raise ValueError."""
        p = _build_predictor()
        with pytest.raises(ValueError):
            p.predict_array(np.array([]))

    def test_predict_array_wrong_channels_raises(self):
        """Passing a 4-channel (RGBA) image must raise ValueError."""
        p = _build_predictor()
        with pytest.raises(ValueError):
            p.predict_array(np.zeros((64, 64, 4), dtype=np.uint8))


class TestPredictImagePath:
    """Tests for AquaticPredictor.predict_image_path."""

    def test_predict_image_path_valid_png(self, tmp_path):
        """Should load a real PNG from disk and return a valid prediction."""
        p = _build_predictor()
        png_path = _make_png(tmp_path)
        result = p.predict_image_path(str(png_path))
        assert "image_path" in result
        assert Path(result["image_path"]).exists()
        assert isinstance(result["predicted_class"], str)

    def test_predict_image_path_missing_file_raises(self):
        """FileNotFoundError for a non-existent path."""
        p = _build_predictor()
        with pytest.raises(FileNotFoundError):
            p.predict_image_path("/nonexistent/path/fish.jpg")

    def test_predict_image_path_corrupt_file_raises(self, tmp_path):
        """ValueError for a file that is not a valid image."""
        p = _build_predictor()
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"this is not an image")
        with pytest.raises(ValueError, match="Could not read image"):
            p.predict_image_path(str(corrupt))


class TestPredictorProperties:
    """Tests for AquaticPredictor property accessors."""

    def test_feature_dim_matches_representation(self):
        """feature_dim must equal what AquaticFeatureExtractor reports."""
        from archana_person2.src.feature_extractor import AquaticFeatureExtractor
        extractor = AquaticFeatureExtractor()
        for rep in ("color_hist", "hog", "combined"):
            p = _build_predictor(representation=rep)
            expected = extractor.get_feature_dim(rep)
            assert p.feature_dim == expected, f"Mismatch for rep='{rep}'"

    def test_class_names_is_nonempty_list_of_strings(self):
        """class_names must return a non-empty list of strings."""
        p = _build_predictor(n_classes=5)
        names = p.class_names
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)
