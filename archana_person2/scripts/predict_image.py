"""
predict_image.py — Command-line prediction script.

Person 2 (Archana Ulloju) — Machine Learning for Robust Aquatic Biodiversity Monitoring.

Loads a trained model (Logistic Regression, Random Forest, or SVM),
extracts features from a user-supplied underwater image, and prints
the predicted species label with an optional confidence score.

Usage
-----
  python archana_person2/scripts/predict_image.py \\
      --image path/to/fish.jpg \\
      --model svm \\
      --representation combined

  python archana_person2/scripts/predict_image.py \\
      --image path/to/fish.jpg \\
      --model logistic_regression \\
      --representation color_hist \\
      --model-path archana_person2/artifacts/models/logistic_regression_pipeline.joblib

Arguments
---------
  --image           Path to the input image (required).
  --model           Model to use: logistic_regression | random_forest | svm  (default: svm)
  --representation  Feature type: color_hist | hog | combined  (default: combined)
  --model-path      Optional explicit path to .joblib model file.
  --config          Optional path to config.yaml.
  --top-k           Number of top species to display (default: 3).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path regardless of invocation directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Suppress matplotlib font cache (Windows compatibility)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s — %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_image.py",
        description=(
            "Predict aquatic species from an underwater image using a "
            "trained classical ML model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", required=True, metavar="PATH",
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--model", default="svm",
        choices=["logistic_regression", "random_forest", "svm"],
        help="Model to use for prediction (default: svm).",
    )
    parser.add_argument(
        "--representation", default="combined",
        choices=["color_hist", "hog", "combined"],
        help="Feature representation (default: combined).",
    )
    parser.add_argument(
        "--model-path", default=None, metavar="PATH",
        help="Explicit path to a saved .joblib model file (optional).",
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="Path to config.yaml (optional; auto-detected by default).",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, metavar="K",
        help="Number of top species predictions to display (default: 3).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results in JSON format.",
    )
    return parser


def _print_human_readable(result: dict, top_k: int) -> None:
    """Pretty-print the prediction result to stdout."""
    sep = "─" * 60
    print(sep)
    print(f"  Image        : {result.get('image_path', 'array input')}")
    print(f"  Model        : {result['model_name']}")
    print(f"  Representation: {result['representation']}")
    print(f"  Feature Dim  : {len(result['feature_vector'])}")
    print(sep)
    print(f"  ► Predicted Species : {result['predicted_class']}")
    print(f"  ► Label Index       : {result['predicted_index']}")
    if result["confidence"] is not None:
        print(f"  ► Confidence        : {result['confidence']:.4f}  ({result['confidence']*100:.1f}%)")
    else:
        print("  ► Confidence        : N/A (model does not support probabilities)")

    # Top-K species
    if result["probabilities"] and top_k > 1:
        top_entries = sorted(
            result["probabilities"].items(), key=lambda kv: kv[1], reverse=True
        )[:top_k]
        print(f"\n  Top-{top_k} Species:")
        for rank, (species, prob) in enumerate(top_entries, start=1):
            bar = "█" * int(prob * 20)
            print(f"    {rank:2}. {species:<30s} {prob:.4f}  |{bar:<20}|")
    print(sep)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Import here (after sys.path is configured)
    from archana_person2.src.predict import AquaticPredictor

    # Build predictor
    try:
        predictor = AquaticPredictor(
            model_name=args.model,
            representation=args.representation,
            model_path=args.model_path,
            config_path=args.config,
        )
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # Run prediction
    try:
        result = predictor.predict_image_path(args.image)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.json:
        # Serialise numpy arrays for JSON
        output = {k: v for k, v in result.items() if k != "feature_vector"}
        output["feature_dim"] = len(result["feature_vector"])
        print(json.dumps(output, indent=2))
    else:
        _print_human_readable(result, top_k=args.top_k)


if __name__ == "__main__":
    main()
