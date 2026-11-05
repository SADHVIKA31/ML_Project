"""
Support Vector Machine (SVM) Classifier Baseline - Person 2 (Archana Ulloju).

Implements multi-class Support Vector Classification inside a scikit-learn Pipeline
with StandardScaler. Preprocessing is strictly fitted on training data to prevent
data leakage.
Includes kernel experimentation (Linear, RBF, Polynomial) and validation-set hyperparameter
tuning (evaluating Macro-F1, never touching test data).
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


class AquaticSVMClassifier:
    """
    Support Vector Machine Classifier with StandardScaler pipeline for aquatic species monitoring.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        probability: bool = True,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.probability = probability
        self.random_state = random_state
        self.class_weight = class_weight

        self.pipeline: Optional[Pipeline] = None
        self.best_params_: Dict[str, Any] = {}
        self.kernel_comparison_results_: List[Dict[str, Any]] = []
        self.experiment_results_: List[Dict[str, Any]] = []

    def build_pipeline(
        self,
        kernel: Optional[str] = None,
        C: Optional[float] = None,
        gamma: Optional[Union[str, float]] = None,
        degree: Optional[int] = None,
        coef0: Optional[float] = None,
    ) -> Pipeline:
        """
        Constructs an sklearn Pipeline with StandardScaler and SVC.
        StandardScaler ensures invariant feature scale across heterogeneous descriptors.
        """
        k = kernel if kernel is not None else self.kernel
        c_val = C if C is not None else self.C
        g = gamma if gamma is not None else self.gamma
        d = degree if degree is not None else self.degree
        c0 = coef0 if coef0 is not None else self.coef0

        svc = SVC(
            kernel=k,
            C=c_val,
            gamma=g,
            degree=d,
            coef0=c0,
            probability=self.probability,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", svc),
        ])
        return pipe

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "AquaticSVMClassifier":
        """Fits StandardScaler and SVC strictly on training data."""
        if self.pipeline is None:
            self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels using the fitted pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts posterior probabilities using the fitted pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")
        return self.pipeline.predict_proba(X)

    def compare_kernels(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        kernels: Optional[List[str]] = None,
        c_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Compares Linear, RBF, and Polynomial kernels across candidate C values on the validation set.
        Guarantees zero test-set leakage.
        """
        if kernels is None:
            kernels = ["linear", "rbf", "poly"]
        if c_values is None:
            c_values = [0.1, 1.0, 10.0]

        results = []
        best_score = -1.0
        best_kernel_setup: Dict[str, Any] = {}

        for k in kernels:
            for c_val in c_values:
                # Degree 3 for poly, gamma 'scale'
                pipe = self.build_pipeline(kernel=k, C=c_val, gamma="scale", degree=3)
                pipe.fit(X_train, y_train)

                val_preds = pipe.predict(X_val)
                val_macro_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
                val_acc = float(accuracy_score(y_val, val_preds))
                val_weighted_f1 = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))

                entry = {
                    "kernel": k,
                    "C": c_val,
                    "gamma": "scale",
                    "degree": 3 if k == "poly" else None,
                    "val_macro_f1": val_macro_f1,
                    "val_weighted_f1": val_weighted_f1,
                    "val_accuracy": val_acc,
                }
                results.append(entry)

                if val_macro_f1 > best_score:
                    best_score = val_macro_f1
                    best_kernel_setup = entry

        self.kernel_comparison_results_ = results
        return {
            "best_kernel_setup": best_kernel_setup,
            "all_kernel_results": results,
        }

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Hyperparameter grid search evaluated strictly on the validation split.
        Zero test-set leakage. Selects candidate optimizing Macro-F1.
        """
        if param_grid is None:
            param_grid = {
                "kernel": ["linear", "rbf", "poly"],
                "C": [0.01, 0.1, 1.0, 5.0, 10.0],
                "gamma": ["scale", "auto"],
                "degree": [2, 3],
            }

        kernels = param_grid.get("kernel", ["rbf"])
        c_list = param_grid.get("C", [1.0])
        gamma_list = param_grid.get("gamma", ["scale"])
        degree_list = param_grid.get("degree", [3])

        best_score = -1.0
        best_cfg: Dict[str, Any] = {}
        experiments = []

        for k in kernels:
            for c_val in c_list:
                gammas_to_try = gamma_list if k in ["rbf", "poly"] else ["scale"]
                for g in gammas_to_try:
                    degrees_to_try = degree_list if k == "poly" else [3]
                    for d in degrees_to_try:
                        pipe = self.build_pipeline(kernel=k, C=c_val, gamma=g, degree=d)
                        pipe.fit(X_train, y_train)

                        val_preds = pipe.predict(X_val)
                        val_macro_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
                        val_acc = float(accuracy_score(y_val, val_preds))
                        val_weighted_f1 = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))

                        record = {
                            "kernel": k,
                            "C": c_val,
                            "gamma": g if k in ["rbf", "poly"] else None,
                            "degree": d if k == "poly" else None,
                            "val_macro_f1": val_macro_f1,
                            "val_weighted_f1": val_weighted_f1,
                            "val_accuracy": val_acc,
                        }
                        experiments.append(record)

                        if val_macro_f1 > best_score:
                            best_score = val_macro_f1
                            best_cfg = {
                                "kernel": k,
                                "C": c_val,
                                "gamma": g,
                                "degree": d,
                                "best_val_macro_f1": val_macro_f1,
                            }

        self.best_params_ = best_cfg
        self.experiment_results_ = experiments

        # Update model state to best hyperparameter configuration
        self.kernel = best_cfg["kernel"]
        self.C = best_cfg["C"]
        self.gamma = best_cfg.get("gamma", "scale")
        self.degree = best_cfg.get("degree", 3)

        # Refit final pipeline on training set
        self.pipeline = self.build_pipeline(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
        )
        self.pipeline.fit(X_train, y_train)

        return self.best_params_

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the fitted pipeline on an evaluation split (validation or test).
        Computes Accuracy, Macro/Weighted Precision, Recall, Macro/Weighted F1,
        per-class classification report, and confusion matrix.
        """
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")

        preds = self.pipeline.predict(X)

        acc = float(accuracy_score(y, preds))
        macro_prec = float(precision_score(y, preds, average="macro", zero_division=0))
        weighted_prec = float(precision_score(y, preds, average="weighted", zero_division=0))
        macro_rec = float(recall_score(y, preds, average="macro", zero_division=0))
        weighted_rec = float(recall_score(y, preds, average="weighted", zero_division=0))
        macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y, preds, average="weighted", zero_division=0))

        cm = confusion_matrix(y, preds).tolist()
        report = classification_report(
            y,
            preds,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        return {
            "model_name": "Support Vector Machine (SVM)",
            "hyperparameters": {
                "kernel": self.kernel,
                "C": self.C,
                "gamma": self.gamma,
                "degree": self.degree,
            },
            "num_samples": len(y),
            "accuracy": acc,
            "macro_precision": macro_prec,
            "weighted_precision": weighted_prec,
            "macro_recall": macro_rec,
            "weighted_recall": weighted_rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": cm,
            "classification_report": report,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes the fitted pipeline and configuration."""
        if self.pipeline is None:
            raise RuntimeError("No trained pipeline to save.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "degree": self.degree,
            "best_params": self.best_params_,
            "random_state": self.random_state,
        }, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "AquaticSVMClassifier":
        """Deserializes a saved AquaticSVMClassifier pipeline."""
        data = joblib.load(filepath)
        instance = cls(
            kernel=data.get("kernel", "rbf"),
            C=data.get("C", 1.0),
            gamma=data.get("gamma", "scale"),
            degree=data.get("degree", 3),
            random_state=data.get("random_state", 42),
        )
        instance.pipeline = data["pipeline"]
        instance.best_params_ = data.get("best_params", {})
        return instance
