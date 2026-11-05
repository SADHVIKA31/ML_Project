"""
Unified Feature Extractor - Person 2 (Archana Ulloju).

Combines Color Histograms and HOG features into three representations:
1. 'color_hist'
2. 'hog'
3. 'combined' (Color Histogram + HOG)

Guarantees sample ordering, label alignment, caching, and compressed serialization.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import yaml

from archana_person2.src.color_histogram import ColorHistogramExtractor
from archana_person2.src.hog_features import HOGFeatureExtractor
from archana_person2.src.dataset_loader import Person2DatasetLoader


class AquaticFeatureExtractor:
    """
    Unified feature extraction manager for classical ML baselines.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        if config_path is None:
            config_path = str(self.project_root / "archana_person2" / "config" / "config.yaml")

        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        fe_cfg = self.config.get("feature_extraction", {})
        ds_cfg = self.config.get("dataset", {})
        self.target_size = tuple(ds_cfg.get("target_size", [224, 224]))

        # Initialize sub-extractors
        ch_cfg = fe_cfg.get("color_histogram", {})
        self.color_extractor = ColorHistogramExtractor(
            channels=ch_cfg.get("channels", "both"),
            bins_rgb=ch_cfg.get("bins_rgb", [8, 8, 8]),
            bins_hsv=ch_cfg.get("bins_hsv", [8, 8, 8]),
            normalize=ch_cfg.get("normalize", True),
            target_size=self.target_size,
        )

        hog_cfg = fe_cfg.get("hog", {})
        self.hog_extractor = HOGFeatureExtractor(
            orientations=hog_cfg.get("orientations", 9),
            pixels_per_cell=hog_cfg.get("pixels_per_cell", [16, 16]),
            cells_per_block=hog_cfg.get("cells_per_block", [2, 2]),
            block_norm=hog_cfg.get("block_norm", "L2-Hys"),
            target_size=self.target_size,
        )

        self.output_dir = self.project_root / fe_cfg.get("output_dir", "archana_person2/artifacts/features")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_features = fe_cfg.get("cache_features", True)
        self.loader = Person2DatasetLoader(str(self.config_path))

    def get_feature_dim(self, representation: str) -> int:
        """Returns feature vector length for the specified representation."""
        rep = representation.lower()
        if rep == "color_hist":
            return self.color_extractor.feature_dim
        elif rep == "hog":
            return self.hog_extractor.feature_dim
        elif rep == "combined":
            return self.color_extractor.feature_dim + self.hog_extractor.feature_dim
        else:
            raise ValueError(f"Unknown representation '{representation}'. Choose 'color_hist', 'hog', or 'combined'.")

    def extract_image(self, image: np.ndarray, representation: str = "combined") -> np.ndarray:
        """Extract features from a single RGB image numpy array."""
        rep = representation.lower()
        if rep == "color_hist":
            return self.color_extractor.extract(image)
        elif rep == "hog":
            return self.hog_extractor.extract(image)
        elif rep == "combined":
            f_col = self.color_extractor.extract(image)
            f_hog = self.hog_extractor.extract(image)
            return np.hstack([f_col, f_hog]).astype(np.float32)
        else:
            raise ValueError(f"Unknown representation '{representation}'")

    def extract_dataset(
        self,
        split: str = "train",
        representation: str = "combined",
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Extract features for an entire split ('train', 'val', or 'test').
        Returns:
            X: np.ndarray of shape (N, D)
            y: np.ndarray of shape (N,)
            sample_ids: list of sample identifiers
            video_ids: list of video group identifiers
        """
        cache_file = self.output_dir / f"features_{split}_{representation}.npz"
        meta_file = self.output_dir / f"features_{split}_{representation}_meta.json"

        # Check cache
        if self.cache_features and not force_recompute and cache_file.exists():
            data = np.load(cache_file, allow_pickle=True)
            X = data["X"]
            y = data["y"]
            sample_ids = list(data["sample_ids"])
            video_ids = list(data["video_ids"])
            return X, y, sample_ids, video_ids

        records = self.loader.load_split_records(split)
        if not records:
            raise RuntimeError(f"No records found for split '{split}'. Ensure dataset manifests or fixtures exist.")

        X_list = []
        y_list = []
        sample_ids = []
        video_ids = []

        expected_dim = self.get_feature_dim(representation)

        for rec in records:
            img_path = rec["image_path"]
            img = self.loader.load_and_validate_image(img_path)
            if img is None:
                continue

            feats = self.extract_image(img, representation=representation)
            if len(feats) != expected_dim:
                continue

            X_list.append(feats)
            y_list.append(rec["label_idx"])
            sample_ids.append(rec["sample_id"])
            video_ids.append(rec["video_id"])

        if not X_list:
            raise RuntimeError(f"Zero valid samples extracted for split '{split}'!")

        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int32)

        # Save to compressed NPZ
        np.savez_compressed(
            cache_file,
            X=X,
            y=y,
            sample_ids=np.array(sample_ids, dtype=object),
            video_ids=np.array(video_ids, dtype=object),
        )

        # Save extraction metadata
        meta = {
            "split": split,
            "representation": representation,
            "num_samples": int(X.shape[0]),
            "feature_dim": int(X.shape[1]),
            "label_count": len(np.unique(y)),
            "cache_file": str(cache_file.name),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return X, y, sample_ids, video_ids
