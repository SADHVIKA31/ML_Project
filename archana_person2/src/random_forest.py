"""
Random Forest Classifier Baseline - Person 2 (Archana Ulloju).

Implements ensemble decision trees with bagging, random feature subspace selection,
validation-set hyperparameter tuning, Gini impurity-based feature importance extraction,
and Macro-F1 evaluation.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


class AquaticRandomForest:
    """
    Random Forest Classifier for aquatic species monitoring.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs

        self.model: Optional[RandomForestClassifier] = None
        self.best_params_: Dict[str, Any] = {}
        self.experiment_results_: List[Dict[str, Any]] = []

    def _build_model(
        self,
        n_estimators: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
    ) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_estimators if n_estimators is not None else self.n_estimators,
            max_depth=max_depth if max_depth is not None else self.max_depth,
            min_samples_split=min_samples_split if min_samples_split is not None else self.min_samples_split,
            min_samples_leaf=min_samples_leaf if min_samples_leaf is not None else self.min_samples_leaf,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "AquaticRandomForest":
        """Fits the Random Forest model on training samples."""
        if self.model is None:
            self.model = self._build_model()
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distributions."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict_proba(X)

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Hyperparameter tuning evaluated strictly on the validation set.
        Primary metric: Macro-F1.
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 150],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            }

        best_score = -1.0
        best_params = {}
        all_experiments = []

        # Grid search over candidate parameters
        import itertools
        keys = list(param_grid.keys())
        combinations = list(itertools.product(*[param_grid[k] for k in keys]))

        for combo in combinations:
            params = dict(zip(keys, combo))
            m = self._build_model(**params)
            m.fit(X_train, y_train)

            val_preds = m.predict(X_val)
            val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            val_acc = accuracy_score(y_val, val_preds)

            record = {
                "params": params,
                "val_macro_f1": float(val_macro_f1),
                "val_accuracy": float(val_acc),
            }
            all_experiments.append(record)

            if val_macro_f1 > best_score:
                best_score = val_macro_f1
                best_params = params

        self.best_params_ = {**best_params, "best_val_macro_f1": float(best_score)}
        self.experiment_results_ = all_experiments

        # Update instance params and retrain with best configuration
        self.n_estimators = best_params.get("n_estimators", self.n_estimators)
        self.max_depth = best_params.get("max_depth", self.max_depth)
        self.min_samples_split = best_params.get("min_samples_split", self.min_samples_split)
        self.min_samples_leaf = best_params.get("min_samples_leaf", self.min_samples_leaf)

        self.model = self._build_model(**best_params)
        self.model.fit(X_train, y_train)

        return self.best_params_

    def get_feature_importances(self) -> np.ndarray:
        """Returns Gini feature importance array."""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            raise RuntimeError("Model is not fitted or has no feature_importances_.")
        return self.model.feature_importances_

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate Random Forest model on given dataset.
        Computes Accuracy, Macro/Weighted Precision, Macro/Weighted Recall,
        Macro/Weighted F1, confusion matrix, and classification report.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        y_pred = self.model.predict(X)

        acc = accuracy_score(y, y_pred)
        macro_p = precision_score(y, y_pred, average="macro", zero_division=0)
        weighted_p = precision_score(y, y_pred, average="weighted", zero_division=0)
        macro_r = recall_score(y, y_pred, average="macro", zero_division=0)
        weighted_r = recall_score(y, y_pred, average="weighted", zero_division=0)
        macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y, y_pred)
        cls_rep = classification_report(
            y,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "model_name": "Random Forest",
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
            },
            "num_samples": int(len(y)),
            "accuracy": float(acc),
            "macro_precision": float(macro_p),
            "weighted_precision": float(weighted_p),
            "macro_recall": float(macro_r),
            "weighted_recall": float(weighted_r),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": cls_rep,
        }
        return metrics

    def save(self, filepath: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save unfitted model.")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath: str) -> "AquaticRandomForest":
        """Load trained model from disk."""
        instance = cls()
        instance.model = joblib.load(filepath)
        instance.n_estimators = instance.model.n_estimators
        instance.max_depth = instance.model.max_depth
        instance.min_samples_split = instance.model.min_samples_split
        instance.min_samples_leaf = instance.model.min_samples_leaf
        instance.random_state = instance.model.random_state
        return instance
