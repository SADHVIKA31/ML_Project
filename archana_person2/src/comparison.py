"""
Model Comparison and Benchmarking Engine - Person 2 (Archana Ulloju).

Compares all classical machine learning models:
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

Across all three handcrafted feature representations:
1. Color Histogram (1,024 dims - RGB + HSV)
2. Histogram of Oriented Gradients (6,084 dims - shape/edges)
3. Combined Color Histogram + HOG (7,108 dims - multi-modal)

Evaluates on identical test splits, measures computational latency (training and inference),
and objectively determines the best classical baseline based on Macro-F1.
"""

import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from archana_person2.src.feature_extractor import AquaticFeatureExtractor
from archana_person2.src.logistic_regression import AquaticLogisticRegression
from archana_person2.src.random_forest import AquaticRandomForest
from archana_person2.src.svm_classifier import AquaticSVMClassifier
from archana_person2.src.evaluation import CommonModelEvaluator


class ModelComparator:
    """
    Automated benchmark suite comparing classical ML models and feature representations.
    """

    SUPPORTED_MODELS = ["Logistic Regression", "Random Forest", "Support Vector Machine (SVM)"]
    SUPPORTED_REPRESENTATIONS = ["color_hist", "hog", "combined"]

    def __init__(self, extractor: Optional[AquaticFeatureExtractor] = None):
        self.extractor = extractor or AquaticFeatureExtractor()
        inv_map = self.extractor.loader.get_inverse_label_map()
        self.class_names = [inv_map[i] for i in sorted(inv_map.keys())]
        self.labels = list(range(len(self.class_names)))
        self.evaluator = CommonModelEvaluator(class_names=self.class_names, labels=self.labels)

        self.benchmark_results_: List[Dict[str, Any]] = []
        self.best_baseline_: Optional[Dict[str, Any]] = None

    def _instantiate_model(self, model_name: str, representation: str) -> Any:
        """Instantiates fresh model configured appropriately for the representation."""
        if model_name == "Logistic Regression":
            return AquaticLogisticRegression(C=1.0, max_iter=500, random_state=42)
        elif model_name == "Random Forest":
            return AquaticRandomForest(n_estimators=100, max_depth=20, random_state=42)
        elif model_name == "Support Vector Machine (SVM)":
            # Linear kernel is optimal in high-dimensional feature spaces (>=1024)
            return AquaticSVMClassifier(kernel="linear", C=1.0, random_state=42)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def evaluate_configuration(
        self,
        model_name: str,
        representation: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Trains and benchmarks a single model on a specific feature representation.
        Measures training latency and test inference latency.
        """
        model = self._instantiate_model(model_name, representation)

        # 1. Measure Training Time
        t0_train = time.perf_counter()
        model.fit(X_train, y_train)
        t_train_sec = time.perf_counter() - t0_train

        # 2. Measure Test Inference Time
        t0_infer = time.perf_counter()
        eval_result = self.evaluator.evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            sample_ids=test_ids,
            model_name=f"{model_name} [{representation}]",
        )
        t_infer_ms = (time.perf_counter() - t0_infer) * 1000.0

        record = {
            "model_name": model_name,
            "representation": representation,
            "feature_dim": int(X_train.shape[1]),
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "train_time_sec": round(t_train_sec, 4),
            "infer_time_ms": round(t_infer_ms, 2),
            "infer_time_per_sample_ms": round(t_infer_ms / max(len(y_test), 1), 4),
            "metrics": eval_result["metrics"],
            "probability_metrics": eval_result["probability_metrics"],
            "confusion_matrix": eval_result["confusion_matrix"],
            "classification_report": eval_result["classification_report"],
        }
        return record

    def run_full_comparison(
        self,
        models: Optional[List[str]] = None,
        representations: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes complete benchmark across all model-representation permutations.
        """
        models = models or self.SUPPORTED_MODELS
        representations = representations or self.SUPPORTED_REPRESENTATIONS

        all_results: List[Dict[str, Any]] = []

        # Load representations once per modality
        feature_cache = {}
        for rep in representations:
            X_tr, y_tr, tr_ids, _ = self.extractor.extract_dataset("train", rep)
            X_te, y_te, te_ids, _ = self.extractor.extract_dataset("test", rep)
            feature_cache[rep] = (X_tr, y_tr, X_te, y_te, te_ids)

        for rep in representations:
            X_tr, y_tr, X_te, y_te, te_ids = feature_cache[rep]
            for m_name in models:
                res = self.evaluate_configuration(
                    model_name=m_name,
                    representation=rep,
                    X_train=X_tr,
                    y_train=y_tr,
                    X_test=X_te,
                    y_test=y_te,
                    test_ids=te_ids,
                )
                all_results.append(res)

        self.benchmark_results_ = all_results
        self.best_baseline_ = self.identify_best_baseline(all_results)
        return all_results

    @staticmethod
    def identify_best_baseline(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identifies the best classical baseline model.
        Ranking hierarchy:
        1. Macro-F1 (primary metric - highest)
        2. Accuracy (highest)
        3. Log loss (lowest, if available)
        4. Inference latency (lowest)
        """
        if not results:
            raise ValueError("Results list cannot be empty when identifying best baseline.")

        def sort_key(entry: Dict[str, Any]) -> Tuple[float, float, float, float]:
            macro_f1 = entry["metrics"]["macro_f1"]
            acc = entry["metrics"]["accuracy"]
            prob = entry.get("probability_metrics", {})
            ll = prob.get("log_loss")
            neg_ll = -ll if ll is not None else -999.0
            neg_latency = -entry["infer_time_ms"]
            return (macro_f1, acc, neg_ll, neg_latency)

        sorted_results = sorted(results, key=sort_key, reverse=True)
        best = sorted_results[0]
        return {
            "model_name": best["model_name"],
            "representation": best["representation"],
            "macro_f1": best["metrics"]["macro_f1"],
            "accuracy": best["metrics"]["accuracy"],
            "weighted_f1": best["metrics"]["weighted_f1"],
            "train_time_sec": best["train_time_sec"],
            "infer_time_ms": best["infer_time_ms"],
            "log_loss": best["probability_metrics"].get("log_loss"),
            "selection_rationale": (
                f"Selected as the top performing baseline based on primary metric Macro-F1 "
                f"({best['metrics']['macro_f1']:.4f}) and accuracy ({best['metrics']['accuracy']:.4f})."
            ),
        }

    def get_comparison_table_records(self) -> List[Dict[str, Any]]:
        """Returns flat tabular rows suitable for CSV or Markdown table generation."""
        records = []
        for r in self.benchmark_results_:
            metrics = r["metrics"]
            prob = r["probability_metrics"]
            records.append({
                "model": r["model_name"],
                "representation": r["representation"],
                "feature_dim": r["feature_dim"],
                "accuracy": round(metrics["accuracy"], 4),
                "macro_precision": round(metrics["macro_precision"], 4),
                "weighted_precision": round(metrics["weighted_precision"], 4),
                "macro_recall": round(metrics["macro_recall"], 4),
                "weighted_recall": round(metrics["weighted_recall"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "weighted_f1": round(metrics["weighted_f1"], 4),
                "log_loss": round(prob["log_loss"], 4) if prob.get("log_loss") is not None else "N/A",
                "roc_auc": round(prob["roc_auc_macro_ovr"], 4) if prob.get("roc_auc_macro_ovr") is not None else "N/A",
                "train_time_sec": r["train_time_sec"],
                "infer_time_ms": r["infer_time_ms"],
            })
        return records

    def save_benchmark_artifacts(
        self,
        output_dir: Path,
    ) -> Tuple[Path, Path, Path]:
        """Saves model_comparison.json, model_comparison.csv, and best_baseline.json."""
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "model_comparison.json"
        csv_path = output_dir / "model_comparison.csv"
        best_path = output_dir / "best_baseline.json"

        # 1. Full JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "benchmark_results": self.benchmark_results_,
                "best_baseline": self.best_baseline_,
                "total_configurations_evaluated": len(self.benchmark_results_),
            }, f, indent=2)

        # 2. Tabular CSV
        records = self.get_comparison_table_records()
        if records:
            fieldnames = list(records[0].keys())
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in records:
                    writer.writerow(row)

        # 3. Best Baseline JSON
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(self.best_baseline_, f, indent=2)

        return json_path, csv_path, best_path
