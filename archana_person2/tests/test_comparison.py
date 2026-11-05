"""
Automated Unit Tests for Model Comparison and Benchmarking - Person 2 (Archana Ulloju).

Tests:
1. Best baseline identification logic based primarily on Macro-F1.
2. Multi-metric tie-breaking hierarchy (Macro-F1 -> Accuracy -> Log Loss -> Latency).
3. Benchmark table record generation and schema validation.
4. Artifact saving (JSON, CSV, best_baseline metadata) in temporary directory.
5. Error handling for empty or malformed benchmark inputs.
"""

import os
import tempfile
import json
import csv
import pytest
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from archana_person2.src.comparison import ModelComparator


@pytest.fixture
def mock_benchmark_results():
    """Generates synthetic benchmark outputs across models with distinct metric values."""
    return [
        {
            "model_name": "Logistic Regression",
            "representation": "color_hist",
            "feature_dim": 1024,
            "train_samples": 50,
            "test_samples": 50,
            "train_time_sec": 0.05,
            "infer_time_ms": 12.0,
            "infer_time_per_sample_ms": 0.24,
            "metrics": {
                "accuracy": 0.88,
                "macro_precision": 0.89,
                "weighted_precision": 0.89,
                "macro_recall": 0.88,
                "weighted_recall": 0.88,
                "macro_f1": 0.88,
                "weighted_f1": 0.88,
            },
            "probability_metrics": {"log_loss": 0.45, "roc_auc_macro_ovr": 0.95},
            "confusion_matrix": [[10, 0], [2, 8]],
            "classification_report": {},
        },
        {
            "model_name": "Random Forest",
            "representation": "combined",
            "feature_dim": 7108,
            "train_samples": 50,
            "test_samples": 50,
            "train_time_sec": 0.20,
            "infer_time_ms": 10.0,
            "infer_time_per_sample_ms": 0.20,
            "metrics": {
                "accuracy": 0.96,
                "macro_precision": 0.96,
                "weighted_precision": 0.96,
                "macro_recall": 0.96,
                "weighted_recall": 0.96,
                "macro_f1": 0.96,
                "weighted_f1": 0.96,
            },
            "probability_metrics": {"log_loss": 0.12, "roc_auc_macro_ovr": 0.99},
            "confusion_matrix": [[10, 0], [0, 10]],
            "classification_report": {},
        },
        {
            "model_name": "Support Vector Machine (SVM)",
            "representation": "combined",
            "feature_dim": 7108,
            "train_samples": 50,
            "test_samples": 50,
            "train_time_sec": 0.15,
            "infer_time_ms": 25.0,
            "infer_time_per_sample_ms": 0.50,
            "metrics": {
                "accuracy": 0.96,
                "macro_precision": 0.96,
                "weighted_precision": 0.96,
                "macro_recall": 0.96,
                "weighted_recall": 0.96,
                "macro_f1": 0.96,
                "weighted_f1": 0.96,
            },
            "probability_metrics": {"log_loss": 0.20, "roc_auc_macro_ovr": 0.98},
            "confusion_matrix": [[10, 0], [0, 10]],
            "classification_report": {},
        },
    ]


class TestBestBaselineIdentification:
    def test_identifies_highest_macro_f1(self, mock_benchmark_results):
        # RF and SVM have higher Macro-F1 (0.96) than LR (0.88)
        best = ModelComparator.identify_best_baseline(mock_benchmark_results)
        assert best["macro_f1"] == 0.96
        # Tie-breaker between RF and SVM (both 0.96): RF has lower log_loss (0.12 vs 0.20)
        assert best["model_name"] == "Random Forest"
        assert best["representation"] == "combined"
        assert "selection_rationale" in best

    def test_empty_results_raise_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelComparator.identify_best_baseline([])


class TestComparisonRecordsAndArtifacts:
    def test_comparison_table_records_structure(self, mock_benchmark_results):
        comparator = ModelComparator.__new__(ModelComparator)
        comparator.benchmark_results_ = mock_benchmark_results

        records = comparator.get_comparison_table_records()
        assert len(records) == 3

        r0 = records[0]
        expected_keys = [
            "model", "representation", "feature_dim", "accuracy",
            "macro_precision", "weighted_precision", "macro_recall",
            "weighted_recall", "macro_f1", "weighted_f1",
            "log_loss", "roc_auc", "train_time_sec", "infer_time_ms",
        ]
        for k in expected_keys:
            assert k in r0

    def test_save_benchmark_artifacts(self, mock_benchmark_results):
        comparator = ModelComparator.__new__(ModelComparator)
        comparator.benchmark_results_ = mock_benchmark_results
        comparator.best_baseline_ = ModelComparator.identify_best_baseline(mock_benchmark_results)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            json_p, csv_p, best_p = comparator.save_benchmark_artifacts(out_path)

            assert json_p.exists()
            assert csv_p.exists()
            assert best_p.exists()

            # Verify JSON content
            with open(json_p, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert "benchmark_results" in data
                assert len(data["benchmark_results"]) == 3

            # Verify CSV content
            with open(csv_p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 3
                assert rows[0]["model"] == "Logistic Regression"

            # Verify Best Baseline JSON
            with open(best_p, "r", encoding="utf-8") as f:
                best_data = json.load(f)
                assert best_data["model_name"] == "Random Forest"
                assert best_data["macro_f1"] == 0.96
