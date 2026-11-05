"""
Logistic Regression Classifier Baseline - Person 2 (Archana Ulloju).

Implements multi-class Logistic Regression inside a scikit-learn Pipeline with
StandardScaler. Preprocessing is strictly fitted on training data.
Tuning is conducted on the validation split (never on the test split).
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


class AquaticLogisticRegression:
    """
    Multi-class Logistic Regression with StandardScaler pipeline for aquatic species classification.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
    ):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.class_weight = class_weight
        self.pipeline: Optional[Pipeline] = None
        self.best_params_: Dict[str, Any] = {}
        self.validation_results_: List[Dict[str, Any]] = []

    def build_pipeline(self, C: Optional[float] = None) -> Pipeline:
        """Constructs an sklearn Pipeline with StandardScaler and LogisticRegression."""
        c_val = C if C is not None else self.C
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_val,
                    max_iter=self.max_iter,
                    solver=self.solver,
                    random_state=self.random_state,
                    class_weight=self.class_weight,
                ),
            ),
        ])
        return pipe

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "AquaticLogisticRegression":
        """Fits preprocessing scaler and model strictly on training samples."""
        if self.pipeline is None:
            self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate class predictions using the fitted pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probabilities using the fitted pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")
        return self.pipeline.predict_proba(X)

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        c_candidates: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Hyperparameter selection evaluated STRICTLY on the validation split.
        The test split is NEVER touched during tuning.
        Primary metric: Macro-F1.
        """
        if c_candidates is None:
            c_candidates = [0.01, 0.1, 1.0, 5.0, 10.0]

        best_score = -1.0
        best_c = self.C
        results = []

        for c_val in c_candidates:
            pipe = self.build_pipeline(C=c_val)
            pipe.fit(X_train, y_train)

            val_preds = pipe.predict(X_val)
            val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            val_acc = accuracy_score(y_val, val_preds)

            entry = {
                "C": c_val,
                "val_macro_f1": float(val_macro_f1),
                "val_accuracy": float(val_acc),
            }
            results.append(entry)

            if val_macro_f1 > best_score:
                best_score = val_macro_f1
                best_c = c_val

        self.C = best_c
        self.best_params_ = {"C": best_c, "best_val_macro_f1": float(best_score)}
        self.validation_results_ = results

        # Retrain on train split using selected best C
        self.pipeline = self.build_pipeline(C=best_c)
        self.pipeline.fit(X_train, y_train)

        return self.best_params_

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the pipeline and compute all required metrics:
        - Accuracy, Macro/Weighted Precision, Macro/Weighted Recall, Macro/Weighted F1
        - Confusion Matrix and Per-Class Report
        """
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet.")

        y_pred = self.pipeline.predict(X)

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
            "model_name": "Logistic Regression",
            "hyperparameters": {"C": self.C, "max_iter": self.max_iter, "solver": self.solver},
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
        """Persist trained pipeline to disk via joblib."""
        if self.pipeline is None:
            raise RuntimeError("Cannot save unfitted pipeline.")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, filepath)

    @classmethod
    def load(cls, filepath: str) -> "AquaticLogisticRegression":
        """Load trained pipeline from disk."""
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        lr_step = instance.pipeline.named_steps.get("classifier")
        if lr_step is not None:
            instance.C = lr_step.C
            instance.max_iter = lr_step.max_iter
            instance.solver = lr_step.solver
            instance.random_state = lr_step.random_state
        return instance
