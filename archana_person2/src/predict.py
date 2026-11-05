"""
Reusable Prediction Pipeline - Person 2 (Archana Ulloju).

Loads a trained Logistic Regression, Random Forest, or SVM model,
applies exactly the same preprocessing and feature extraction used
during training, and returns the predicted species label together
with per-class probability (where the model supports it).

Usage
-----
>>> from archana_person2.src.predict import AquaticPredictor
>>> predictor = AquaticPredictor(model_name="svm", representation="combined")
>>> result = predictor.predict_image_path("path/to/fish_image.jpg")
>>> print(result["predicted_class"], result["confidence"])
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

from archana_person2.src.feature_extractor import AquaticFeatureExtractor
from archana_person2.src.dataset_loader import Person2DatasetLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = ("logistic_regression", "random_forest", "svm")
SUPPORTED_REPRESENTATIONS = ("color_hist", "hog", "combined")

# Default model filenames on disk
_MODEL_FILENAMES: Dict[str, str] = {
    "logistic_regression": "logistic_regression_pipeline.joblib",
    "random_forest": "random_forest_model.joblib",
    "svm": "svm_model.joblib",
}

# Models that natively support predict_proba without special handling
_PROB_SUPPORTED_MODELS = {"logistic_regression", "svm"}

# Default feature representation used during training for each model
_DEFAULT_REPRESENTATION: Dict[str, str] = {
    "logistic_regression": "combined",
    "random_forest": "combined",
    "svm": "combined",
}


# ---------------------------------------------------------------------------
# Loader helpers (lazy import to avoid circular imports)
# ---------------------------------------------------------------------------

def _load_model_instance(model_name: str, model_path: str) -> Any:
    """
    Load the correct wrapper class for *model_name* from *model_path*.

    Returns the wrapper instance (AquaticLogisticRegression,
    AquaticRandomForest, or AquaticSVMClassifier) so that callers can
    access class-level metadata (.pipeline / .model) if needed.
    """
    if model_name == "logistic_regression":
        from archana_person2.src.logistic_regression import AquaticLogisticRegression
        return AquaticLogisticRegression.load(model_path)
    elif model_name == "random_forest":
        from archana_person2.src.random_forest import AquaticRandomForest
        return AquaticRandomForest.load(model_path)
    elif model_name == "svm":
        from archana_person2.src.svm_classifier import AquaticSVMClassifier
        return AquaticSVMClassifier.load(model_path)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {SUPPORTED_MODELS}"
        )


def _get_sklearn_estimator(wrapper: Any, model_name: str):
    """Return the raw sklearn estimator (Pipeline or RandomForestClassifier)."""
    if model_name in ("logistic_regression", "svm"):
        return wrapper.pipeline
    else:  # random_forest
        return wrapper.model


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class AquaticPredictor:
    """
    End-to-end prediction pipeline for aquatic species classification.

    Parameters
    ----------
    model_name : str
        One of ``'logistic_regression'``, ``'random_forest'``, ``'svm'``.
    representation : str, optional
        Feature representation to extract — ``'color_hist'``, ``'hog'``,
        or ``'combined'``.  Defaults to the representation used during
        training (``'combined'`` for all three models).
    model_path : str, optional
        Absolute path to the saved ``.joblib`` file.  If *None* the
        default artifact directory is used.
    config_path : str, optional
        Path to ``config.yaml``.  Auto-detected when *None*.

    Raises
    ------
    ValueError
        If *model_name* or *representation* is not recognised.
    FileNotFoundError
        If the model file cannot be located on disk.
    """

    def __init__(
        self,
        model_name: str = "svm",
        representation: Optional[str] = None,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        model_name = model_name.lower().strip()
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Choose from: {SUPPORTED_MODELS}"
            )

        self.model_name = model_name
        self.representation = (
            representation.lower().strip()
            if representation is not None
            else _DEFAULT_REPRESENTATION[model_name]
        )
        if self.representation not in SUPPORTED_REPRESENTATIONS:
            raise ValueError(
                f"Unsupported representation '{self.representation}'. "
                f"Choose from: {SUPPORTED_REPRESENTATIONS}"
            )

        # Resolve project root from this file's location
        self._project_root = Path(__file__).resolve().parent.parent.parent

        # Config
        if config_path is None:
            config_path = str(
                self._project_root / "archana_person2" / "config" / "config.yaml"
            )
        self._config_path = config_path
        with open(config_path, "r", encoding="utf-8") as fh:
            self._config = yaml.safe_load(fh)

        # Resolve model path
        if model_path is None:
            models_dir = self._project_root / self._config.get(
                "models", {}
            ).get("output_dir", "archana_person2/artifacts/models")
            model_path = str(models_dir / _MODEL_FILENAMES[model_name])

        self._model_path = model_path
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}\n"
                "Train and save the model first using the corresponding "
                "training script."
            )

        # Load model
        logger.info("Loading model '%s' from %s …", model_name, self._model_path)
        self._wrapper = _load_model_instance(model_name, self._model_path)
        self._estimator = _get_sklearn_estimator(self._wrapper, model_name)

        # Feature extractor (uses same config → same preprocessing params)
        self._feature_extractor = AquaticFeatureExtractor(
            config_path=config_path
        )

        # Label map (integer index → species name)
        loader = Person2DatasetLoader(config_path=config_path)
        self._inv_label_map: Dict[int, str] = loader.get_inverse_label_map()

        # Inspect classes known to the estimator
        if hasattr(self._estimator, "classes_"):
            self._classes_: np.ndarray = self._estimator.classes_
        else:
            self._classes_ = np.array(
                sorted(self._inv_label_map.keys()), dtype=np.int32
            )

        logger.info(
            "AquaticPredictor ready — model='%s', representation='%s', classes=%d",
            model_name,
            self.representation,
            len(self._classes_),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_array(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict the aquatic species from an **RGB** uint8 numpy array.

        Parameters
        ----------
        image : np.ndarray
            RGB image as returned by ``cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)``.
            Any spatial size is accepted; resizing is handled internally.

        Returns
        -------
        dict with keys:
            ``predicted_class`` : str  — species name
            ``predicted_index``  : int  — integer label
            ``confidence``       : float | None — maximum probability (None if unsupported)
            ``probabilities``    : dict[str, float] | None — full distribution (None if unsupported)
            ``feature_vector``   : np.ndarray — extracted feature vector (shape [D])
            ``representation``   : str — feature representation used
            ``model_name``       : str — model identifier

        Raises
        ------
        ValueError
            If the image array is invalid (wrong dtype, shape, or empty).
        RuntimeError
            If the model returns an unexpected output format.
        """
        # --- Validate input array ---
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("image must be a non-None numpy ndarray.")
        if image.ndim == 2:
            # Greyscale → replicate to 3 channels
            image = np.stack([image, image, image], axis=-1)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must have shape (H, W, 3), got {image.shape}."
            )
        if image.size == 0:
            raise ValueError("image array is empty (zero-size).")

        # Ensure uint8
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # --- Feature extraction ---
        feature_vector = self._feature_extractor.extract_image(
            image, representation=self.representation
        )
        X = feature_vector.reshape(1, -1).astype(np.float32)

        # --- Predict ---
        pred_index_arr = self._estimator.predict(X)
        pred_index = int(pred_index_arr[0])
        predicted_class = self._inv_label_map.get(pred_index, f"unknown_{pred_index}")

        # --- Probabilities (optional) ---
        probabilities: Optional[Dict[str, float]] = None
        confidence: Optional[float] = None

        if self.model_name in _PROB_SUPPORTED_MODELS and hasattr(
            self._estimator, "predict_proba"
        ):
            try:
                proba_arr = self._estimator.predict_proba(X)[0]
                probabilities = {
                    self._inv_label_map.get(int(cls_idx), f"unknown_{cls_idx}"): float(p)
                    for cls_idx, p in zip(self._classes_, proba_arr)
                }
                confidence = float(proba_arr.max())
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "predict_proba failed for model '%s': %s",
                    self.model_name,
                    exc,
                )

        return {
            "predicted_class": predicted_class,
            "predicted_index": pred_index,
            "confidence": confidence,
            "probabilities": probabilities,
            "feature_vector": feature_vector,
            "representation": self.representation,
            "model_name": self.model_name,
        }

    def predict_image_path(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Predict the aquatic species from an image file on disk.

        Parameters
        ----------
        image_path : str | Path
            Path to a valid image file (JPEG, PNG, BMP, TIFF, …).

        Returns
        -------
        Same dict as :meth:`predict_array`, plus:
            ``image_path`` : str — resolved absolute path to the input file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file cannot be read as a valid image.
        """
        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            raise ValueError(
                f"Could not read image from '{image_path}'. "
                "Check that it is a valid image file."
            )

        # Convert BGR → RGB (consistent with training pipeline)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        result = self.predict_array(rgb)
        result["image_path"] = str(image_path)
        return result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the extracted feature vector."""
        return self._feature_extractor.get_feature_dim(self.representation)

    @property
    def class_names(self) -> List[str]:
        """Sorted list of class (species) names known to this predictor."""
        return [
            self._inv_label_map.get(int(idx), f"unknown_{idx}")
            for idx in sorted(self._inv_label_map.keys())
        ]

    def __repr__(self) -> str:
        return (
            f"AquaticPredictor("
            f"model='{self.model_name}', "
            f"representation='{self.representation}', "
            f"classes={len(self._inv_label_map)}, "
            f"feature_dim={self.feature_dim})"
        )
