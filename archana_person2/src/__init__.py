"""
Person 2: Archana Ulloju
Project: Machine Learning for Robust Aquatic Biodiversity Monitoring

Contribution:
  - Feature Extraction (Color Histogram, HOG, Combined)
  - Classical ML Models: Logistic Regression, Random Forest, SVM
  - Common Evaluation Framework
  - Model Comparison & Visualisation
  - Reusable Prediction Pipeline (AquaticPredictor)
  - Project Integration Bridge (ClassicalMLPipeline)

Phases Completed: 1-10

Public API
----------
  from archana_person2.src.integration import (
      ClassicalMLPipeline,
      load_classical_predictor,
      predict_image,
      predict_array,
      get_dataset_info,
      verify_classical_integration,
  )
  from archana_person2.src.predict import AquaticPredictor
  from archana_person2.src.feature_extractor import AquaticFeatureExtractor
  from archana_person2.src.evaluation import CommonModelEvaluator
  from archana_person2.src.comparison import ModelComparator
"""

__author__ = "Archana Ulloju"
__role__ = "Person 2 -- Classical ML and Feature Extraction Lead"
__version__ = "1.9.0"
__phases_complete__ = list(range(1, 11))  # Phases 1-10


def __getattr__(name: str):
    """Lazy attribute access for public symbols -- avoids circular imports."""
    _lazy_map = {
        # Integration bridge (Phase 9)
        "ClassicalMLPipeline":          "archana_person2.src.integration",
        "load_classical_predictor":     "archana_person2.src.integration",
        "predict_image":                "archana_person2.src.integration",
        "predict_array":                "archana_person2.src.integration",
        "get_dataset_info":             "archana_person2.src.integration",
        "verify_classical_integration": "archana_person2.src.integration",
        # Core prediction (Phase 8)
        "AquaticPredictor":             "archana_person2.src.predict",
        # Feature extraction (Phase 2)
        "AquaticFeatureExtractor":      "archana_person2.src.feature_extractor",
        "ColorHistogramExtractor":      "archana_person2.src.color_histogram",
        "HOGFeatureExtractor":          "archana_person2.src.hog_features",
        # Evaluation (Phase 6)
        "CommonModelEvaluator":         "archana_person2.src.evaluation",
        # Comparison (Phase 7)
        "ModelComparator":              "archana_person2.src.comparison",
        # Models (Phases 3-5)
        "AquaticLogisticRegression":    "archana_person2.src.logistic_regression",
        "AquaticRandomForest":          "archana_person2.src.random_forest",
        "AquaticSVMClassifier":         "archana_person2.src.svm_classifier",
        # Dataset (Phase 1)
        "Person2DatasetLoader":         "archana_person2.src.dataset_loader",
    }
    if name in _lazy_map:
        import importlib
        module = importlib.import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(
        f"module 'archana_person2.src' has no attribute {name!r}"
    )
