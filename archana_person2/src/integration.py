"""
integration.py — Project-level integration bridge for Person 2 (Archana Ulloju).

Machine Learning for Robust Aquatic Biodiversity Monitoring
===========================================================

This module is the **single entry point** connecting:

  Person 1 (Sadhvika) — Dataset & Preprocessing
    scripts.DatasetLoader   → canonical split manifests, label_map.json,
                               preprocessed NumPy arrays

  Person 2 (Archana)   — Classical ML Models & Feature Extraction
    AquaticPredictor        → feature extraction + model inference
    AquaticFeatureExtractor → Color Hist / HOG / Combined features
    Evaluation & Comparison → CommonModelEvaluator, BenchmarkEngine

Design
------
- NO retraining performed in this module.
- Falls back to synthetic fixtures when the real dataset is absent
  (identical to the behaviour adopted in Phases 1–8).
- All public functions are safe to call without the real Fish4Knowledge
  dataset installed — only inference on new images requires saved models.

Public API
----------
ClassicalMLPipeline        : High-level façade for prediction & evaluation.
load_classical_predictor() : Factory; returns a ready-to-use AquaticPredictor.
predict_image()            : One-call convenience wrapper.
get_dataset_info()         : Cross-member dataset metadata summary.
verify_classical_integration() : Smoke-test of all integration touch-points.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root resolution (works from any cwd)
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent          # archana_person2/src/
_PERSON2_ROOT = _MODULE_DIR.parent                     # archana_person2/
_PROJECT_ROOT = _PERSON2_ROOT.parent                   # ML_Project/

# Ensure project root is importable so that scripts.dataset_loader works
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Suppress matplotlib font cache (Windows compatibility)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

# ---------------------------------------------------------------------------
# Supported model / representation constants (re-export for convenience)
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = ("logistic_regression", "random_forest", "svm")
SUPPORTED_REPRESENTATIONS = ("color_hist", "hog", "combined")

_DEFAULT_MODEL = "svm"
_DEFAULT_REPRESENTATION = "combined"

_MODELS_DIR = _PERSON2_ROOT / "artifacts" / "models"
_MODEL_FILES: Dict[str, str] = {
    "logistic_regression": "logistic_regression_pipeline.joblib",
    "random_forest": "random_forest_model.joblib",
    "svm": "svm_model.joblib",
}


# ===========================================================================
# Dataset bridge helpers
# ===========================================================================

def _try_sadhvika_loader():
    """
    Attempt to import and return Sadhvika's DatasetLoader.

    Returns the DatasetLoader *class* (not an instance) on success,
    or None if the module is not importable (e.g. Pillow not installed or
    the scripts package is not on sys.path).
    """
    try:
        from scripts.dataset_loader import DatasetLoader  # type: ignore
        return DatasetLoader
    except ImportError:
        return None


def get_dataset_info() -> Dict[str, Any]:
    """
    Return cross-member dataset metadata by bridging to Sadhvika's DatasetLoader.

    When the preprocessed data is not present (common during development /
    CI), falls back to Archana's Person2DatasetLoader which uses synthetic
    fixtures.

    Returns
    -------
    dict with keys:
        source          : "sadhvika_DatasetLoader" | "archana_Person2DatasetLoader" | "default_fallback"
        label_map       : {class_name: int}
        num_classes     : int
        splits_available: list[str]  — which of train/val/test are readable
        split_counts    : dict       — {"train": N, "val": N, "test": N}
        preprocessed_dir: str | None
        splits_dir      : str | None
    """
    info: Dict[str, Any] = {
        "source": None,
        "label_map": {},
        "num_classes": 0,
        "splits_available": [],
        "split_counts": {},
        "preprocessed_dir": str(_PROJECT_ROOT / "data" / "preprocessed"),
        "splits_dir": str(_PROJECT_ROOT / "data" / "splits"),
    }

    # --- Attempt Sadhvika's DatasetLoader first ---
    DatasetLoaderCls = _try_sadhvika_loader()
    if DatasetLoaderCls is not None:
        label_map_path = _PROJECT_ROOT / "data" / "preprocessed" / "label_map.json"
        if label_map_path.exists():
            try:
                loader = DatasetLoaderCls()
                label_map = loader.get_label_map()
                info["source"] = "sadhvika_DatasetLoader"
                info["label_map"] = label_map
                info["num_classes"] = len(label_map)

                split_info = loader.split_info()
                for split in ("train", "val", "test"):
                    si = split_info.get(split)
                    if si:
                        info["splits_available"].append(split)
                        info["split_counts"][split] = si.get("num_images", 0)

                logger.info("Dataset info obtained from Sadhvika's DatasetLoader.")
                return info
            except Exception as exc:
                logger.warning("Sadhvika DatasetLoader failed: %s — falling back.", exc)

    # --- Fallback to Archana's Person2DatasetLoader ---
    try:
        from archana_person2.src.dataset_loader import Person2DatasetLoader
        p2_loader = Person2DatasetLoader()
        label_map = p2_loader.get_label_map()
        info["source"] = "archana_Person2DatasetLoader"
        info["label_map"] = label_map
        info["num_classes"] = len(label_map)

        for split in ("train", "val", "test"):
            try:
                records = p2_loader.load_split_records(split)
                if records:
                    info["splits_available"].append(split)
                    info["split_counts"][split] = len(records)
            except Exception:
                pass

        logger.info("Dataset info obtained from Archana's Person2DatasetLoader.")
        return info
    except Exception as exc:
        logger.warning("Person2DatasetLoader also failed: %s", exc)

    # --- Absolute fallback: return skeleton ---
    info["source"] = "default_fallback"
    info["label_map"] = {f"Species_{i:02d}": i for i in range(23)}
    info["num_classes"] = 23
    return info


# ===========================================================================
# Predictor factory
# ===========================================================================

def load_classical_predictor(
    model_name: str = _DEFAULT_MODEL,
    representation: str = _DEFAULT_REPRESENTATION,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> "AquaticPredictor":  # noqa: F821
    """
    Factory function that returns a fully-loaded :class:`AquaticPredictor`.

    This is the **recommended way** for other team members to obtain a
    predictor — it centralises all path resolution and validation.

    Parameters
    ----------
    model_name : "logistic_regression" | "random_forest" | "svm"
    representation : "color_hist" | "hog" | "combined"
    model_path : explicit path to .joblib file (optional; auto-detected)
    config_path : path to config.yaml (optional; auto-detected)

    Returns
    -------
    AquaticPredictor (ready for inference)

    Raises
    ------
    ValueError        — unknown model_name or representation
    FileNotFoundError — model file not found
    """
    from archana_person2.src.predict import AquaticPredictor

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {SUPPORTED_MODELS}"
        )
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Supported: {SUPPORTED_REPRESENTATIONS}"
        )

    # Auto-resolve model path
    if model_path is None:
        model_path = str(_MODELS_DIR / _MODEL_FILES[model_name])

    predictor = AquaticPredictor(
        model_name=model_name,
        representation=representation,
        model_path=model_path,
        config_path=config_path,
    )
    logger.info("Loaded predictor: %r", predictor)
    return predictor


# ===========================================================================
# One-call convenience wrappers
# ===========================================================================

def predict_image(
    image_path: Union[str, Path],
    model_name: str = _DEFAULT_MODEL,
    representation: str = _DEFAULT_REPRESENTATION,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One-call convenience: load a model and predict the species in *image_path*.

    Internally calls :func:`load_classical_predictor` then
    ``predictor.predict_image_path()``.

    Parameters
    ----------
    image_path    : path to an underwater image (JPEG, PNG, BMP, TIFF …)
    model_name    : which model to use (default: "svm")
    representation: feature type (default: "combined")
    model_path    : optional explicit .joblib path
    config_path   : optional config.yaml path

    Returns
    -------
    Same dict as :meth:`AquaticPredictor.predict_image_path`.

    Raises
    ------
    FileNotFoundError — image or model not found
    ValueError        — invalid image or unsupported input
    """
    predictor = load_classical_predictor(
        model_name=model_name,
        representation=representation,
        model_path=model_path,
        config_path=config_path,
    )
    return predictor.predict_image_path(image_path)


def predict_array(
    image: np.ndarray,
    model_name: str = _DEFAULT_MODEL,
    representation: str = _DEFAULT_REPRESENTATION,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One-call convenience: predict from an already-loaded RGB uint8 numpy array.

    Parameters
    ----------
    image         : RGB uint8 numpy array (H, W, 3)
    model_name    : which model to use (default: "svm")
    representation: feature type (default: "combined")
    model_path    : optional explicit .joblib path
    config_path   : optional config.yaml path

    Returns
    -------
    Same dict as :meth:`AquaticPredictor.predict_array`.
    """
    predictor = load_classical_predictor(
        model_name=model_name,
        representation=representation,
        model_path=model_path,
        config_path=config_path,
    )
    return predictor.predict_array(image)


# ===========================================================================
# Integration smoke-test
# ===========================================================================

def verify_classical_integration(verbose: bool = True) -> Dict[str, bool]:
    """
    Run a lightweight smoke-test of all Person 2 integration touch-points.

    Checks (all using synthetic data — no real dataset required):
      1. archana_person2.src.predict is importable
      2. archana_person2.src.feature_extractor is importable
      3. archana_person2.src.evaluation is importable
      4. archana_person2.src.comparison is importable
      5. AquaticFeatureExtractor extracts correct dimensions
      6. get_dataset_info() returns a valid dict
      7. Saved model files exist (optional — only if training was run)
      8. scripts.dataset_loader (Sadhvika) is importable

    Parameters
    ----------
    verbose : if True, print a human-readable report to stdout

    Returns
    -------
    dict[str, bool] — {check_name: passed}
    """
    results: Dict[str, bool] = {}

    def _check(name: str, fn) -> bool:
        try:
            fn()
            results[name] = True
            if verbose:
                print(f"  ✅  {name}")
            return True
        except Exception as exc:
            results[name] = False
            if verbose:
                print(f"  ❌  {name}  —  {exc}")
            return False

    if verbose:
        print("\n" + "=" * 60)
        print("  PERSON 2 CLASSICAL ML INTEGRATION VERIFICATION")
        print("=" * 60 + "\n")

    # 1. Module imports
    _check(
        "archana_person2.src.predict importable",
        lambda: __import__("archana_person2.src.predict", fromlist=["AquaticPredictor"]),
    )
    _check(
        "archana_person2.src.feature_extractor importable",
        lambda: __import__(
            "archana_person2.src.feature_extractor",
            fromlist=["AquaticFeatureExtractor"],
        ),
    )
    _check(
        "archana_person2.src.evaluation importable",
        lambda: __import__(
            "archana_person2.src.evaluation",
            fromlist=["CommonModelEvaluator"],
        ),
    )
    _check(
        "archana_person2.src.comparison importable",
        lambda: __import__(
            "archana_person2.src.comparison",
            fromlist=["ModelComparator"],
        ),
    )

    # 2. Feature extractor dimensionality
    def _check_feature_dims():
        from archana_person2.src.feature_extractor import AquaticFeatureExtractor
        extractor = AquaticFeatureExtractor()
        expected = {"color_hist": 1024, "hog": 6084, "combined": 7108}
        for rep, dim in expected.items():
            actual = extractor.get_feature_dim(rep)
            assert actual == dim, f"'{rep}' dim: expected {dim}, got {actual}"

    _check("Feature extractor dimensions correct", _check_feature_dims)

    # 3. Dataset info bridge
    def _check_dataset_info():
        info = get_dataset_info()
        assert isinstance(info, dict)
        assert "label_map" in info
        assert "num_classes" in info
        assert info["num_classes"] > 0

    _check("get_dataset_info() returns valid dict", _check_dataset_info)

    # 4. Saved model files exist
    def _check_models():
        missing = []
        for model_name, fname in _MODEL_FILES.items():
            path = _MODELS_DIR / fname
            if not path.exists():
                missing.append(fname)
        if missing:
            raise FileNotFoundError(f"Missing model files: {missing}")

    _check("Saved model files present", _check_models)

    # 5. Sadhvika DatasetLoader importable
    def _check_sadhvika():
        DatasetLoaderCls = _try_sadhvika_loader()
        if DatasetLoaderCls is None:
            raise ImportError("scripts.dataset_loader not importable")

    _check("scripts.dataset_loader (Sadhvika) importable", _check_sadhvika)

    # 6. Config file present
    def _check_config():
        cfg = _PERSON2_ROOT / "config" / "config.yaml"
        if not cfg.exists():
            raise FileNotFoundError(f"config.yaml not found at {cfg}")

    _check("archana_person2/config/config.yaml present", _check_config)


    all_passed = all(results.values())
    if verbose:
        print()
        if all_passed:
            print("  → PERSON 2 INTEGRATION VERIFIED ✅\n")
        else:
            n_fail = sum(1 for v in results.values() if not v)
            print(f"  → {n_fail} CHECK(S) FAILED ❌\n")

    return results


# ===========================================================================
# High-level façade class
# ===========================================================================

class ClassicalMLPipeline:
    """
    High-level façade exposing the full Person 2 classical ML pipeline
    to other team members.

    This class hides the internal module structure and provides a simple,
    stable interface for prediction, feature extraction, and evaluation.

    Example
    -------
    >>> pipeline = ClassicalMLPipeline(model_name="svm")
    >>> result = pipeline.predict("/path/to/fish.jpg")
    >>> print(result["predicted_class"], result["confidence"])
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        representation: str = _DEFAULT_REPRESENTATION,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_name     : "logistic_regression" | "random_forest" | "svm"
        representation : "color_hist" | "hog" | "combined"
        model_path     : optional explicit .joblib path
        config_path    : optional config.yaml path
        """
        self._predictor = load_classical_predictor(
            model_name=model_name,
            representation=representation,
            model_path=model_path,
            config_path=config_path,
        )

    # ------------------------------------------------------------------

    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Predict aquatic species from an image file path."""
        return self._predictor.predict_image_path(image_path)

    def predict_array(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict aquatic species from an RGB uint8 numpy array."""
        return self._predictor.predict_array(image)

    @property
    def model_name(self) -> str:
        return self._predictor.model_name

    @property
    def representation(self) -> str:
        return self._predictor.representation

    @property
    def feature_dim(self) -> int:
        return self._predictor.feature_dim

    @property
    def class_names(self) -> List[str]:
        return self._predictor.class_names

    def __repr__(self) -> str:
        return (
            f"ClassicalMLPipeline("
            f"model='{self.model_name}', "
            f"representation='{self.representation}', "
            f"classes={len(self.class_names)}, "
            f"feature_dim={self.feature_dim})"
        )
