"""
Unit Tests for Feature Extraction - Person 2 (Archana Ulloju).
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from archana_person2.src.color_histogram import ColorHistogramExtractor
from archana_person2.src.hog_features import HOGFeatureExtractor
from archana_person2.src.feature_extractor import AquaticFeatureExtractor


class TestColorHistogram:
    def test_dimensions_rgb(self):
        extractor = ColorHistogramExtractor(channels="rgb", bins_rgb=(8, 8, 8))
        assert extractor.feature_dim == 512
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        feats = extractor.extract(img)
        assert feats.shape == (512,)
        assert np.isclose(np.sum(feats), 1.0, atol=1e-3)

    def test_dimensions_hsv(self):
        extractor = ColorHistogramExtractor(channels="hsv", bins_hsv=(8, 8, 8))
        assert extractor.feature_dim == 512
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        feats = extractor.extract(img)
        assert feats.shape == (512,)

    def test_dimensions_both(self):
        extractor = ColorHistogramExtractor(channels="both", bins_rgb=(8, 8, 8), bins_hsv=(8, 8, 8))
        assert extractor.feature_dim == 1024
        img = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        feats = extractor.extract(img)
        assert feats.shape == (1024,)
        assert feats.dtype == np.float32

    def test_invalid_image_raises(self):
        extractor = ColorHistogramExtractor()
        with pytest.raises(ValueError):
            extractor.extract(np.zeros((0, 0, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            extractor.extract(np.zeros((50, 50), dtype=np.uint8))  # 2D grayscale instead of 3D


class TestHOGFeatures:
    def test_dimensions(self):
        extractor = HOGFeatureExtractor(
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            target_size=(224, 224),
        )
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        feats = extractor.extract(img)
        assert len(feats) == extractor.feature_dim
        assert feats.dtype == np.float32

    def test_invalid_image_raises(self):
        extractor = HOGFeatureExtractor()
        with pytest.raises(ValueError):
            extractor.extract(np.zeros((0, 0, 3), dtype=np.uint8))


class TestUnifiedFeatureExtractor:
    def test_combined_representation(self):
        extractor = AquaticFeatureExtractor()
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        f_col = extractor.extract_image(img, "color_hist")
        f_hog = extractor.extract_image(img, "hog")
        f_comb = extractor.extract_image(img, "combined")

        assert len(f_comb) == len(f_col) + len(f_hog)
        assert np.allclose(f_comb[:len(f_col)], f_col)
        assert np.allclose(f_comb[len(f_col):], f_hog)

    def test_extract_dataset_caching(self):
        extractor = AquaticFeatureExtractor()
        X1, y1, ids1, vids1 = extractor.extract_dataset(split="train", representation="color_hist", force_recompute=True)
        assert X1.ndim == 2
        assert len(y1) == len(X1)
        assert len(ids1) == len(X1)
        assert len(vids1) == len(X1)

        # Second call should load from cache
        X2, y2, ids2, vids2 = extractor.extract_dataset(split="train", representation="color_hist", force_recompute=False)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)
        assert ids1 == ids2
