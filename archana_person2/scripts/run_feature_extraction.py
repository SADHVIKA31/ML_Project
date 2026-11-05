"""
Runner Script for Feature Extraction - Person 2 (Archana Ulloju).

Extracts and caches Color Histograms, HOG, and Combined features across train,
validation, and test splits.
"""

import sys
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from archana_person2.src.feature_extractor import AquaticFeatureExtractor


def main():
    print("=" * 65)
    print("PHASE 2: HANDCRAFTED FEATURE EXTRACTION PIPELINE")
    print("=" * 65)

    extractor = AquaticFeatureExtractor()
    representations = ["color_hist", "hog", "combined"]
    splits = ["train", "val", "test"]

    results = {}

    for rep in representations:
        print(f"\n--- Extracting Representation: {rep.upper()} (Dim: {extractor.get_feature_dim(rep)}) ---")
        results[rep] = {}
        for sp in splits:
            X, y, sample_ids, video_ids = extractor.extract_dataset(
                split=sp,
                representation=rep,
                force_recompute=True,
            )
            print(f"  [{sp.upper():5s}] X: {X.shape}, y: {y.shape}, unique classes: {len(set(y))}")
            results[rep][sp] = {
                "shape_X": list(X.shape),
                "shape_y": list(y.shape),
                "num_samples": len(sample_ids),
            }

    print("\n" + "=" * 65)
    print("FEATURE EXTRACTION COMPLETED SUCCESSFULLY FOR ALL CONFIGURATIONS!")
    print("=" * 65)
    return results


if __name__ == "__main__":
    main()
