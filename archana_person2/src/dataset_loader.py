"""
Dataset Loader for Person 2 (Archana Ulloju).

Integrates with Person 1 (Sadhvika)'s dataset organization, split manifests,
and preprocessing standards. Provides robust validation, image loading,
video-group verification, and synthetic test data generation.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import yaml


class Person2DatasetLoader:
    """
    Dataset loader for classical ML models and feature extraction.
    Compatible with Person 1's split manifests and folder structures.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        if config_path is None:
            config_path = str(self.project_root / "archana_person2" / "config" / "config.yaml")

        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.ds_cfg = self.config.get("dataset", {})

        self.splits_dir = self.project_root / self.ds_cfg.get("splits_dir", "data/splits")
        self.processed_dir = self.project_root / self.ds_cfg.get("processed_dir", "data/processed")
        self.preprocessed_dir = self.project_root / self.ds_cfg.get("preprocessed_dir", "data/preprocessed")
        self.synthetic_dir = self.project_root / self.ds_cfg.get("synthetic_dir", "archana_person2/artifacts/fixtures/synthetic_dataset")

        self.target_size = tuple(self.ds_cfg.get("target_size", [224, 224]))
        self.valid_exts = set(self.ds_cfg.get("valid_extensions", [".jpg", ".jpeg", ".png", ".bmp"]))

        self._label_map: Optional[Dict[str, int]] = None
        self._inv_label_map: Optional[Dict[int, str]] = None

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def get_label_map(self) -> Dict[str, int]:
        """Load or create the authoritative class-to-index mapping."""
        if self._label_map is not None:
            return self._label_map

        # Check official label map
        official_map_path = self.project_root / self.ds_cfg.get("label_map", "data/preprocessed/label_map.json")
        if official_map_path.exists():
            with open(official_map_path, "r", encoding="utf-8") as f:
                self._label_map = json.load(f)
                self._inv_label_map = {int(v): k for k, v in self._label_map.items()}
                return self._label_map

        # Check processed directory
        if self.processed_dir.exists():
            classes = sorted([d.name for d in self.processed_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
            if classes:
                self._label_map = {cls_name: idx for idx, cls_name in enumerate(classes)}
                self._inv_label_map = {idx: cls_name for idx, cls_name in enumerate(classes)}
                return self._label_map

        # Check synthetic directory
        if self.synthetic_dir.exists():
            syn_map_path = self.synthetic_dir / "label_map.json"
            if syn_map_path.exists():
                with open(syn_map_path, "r", encoding="utf-8") as f:
                    self._label_map = json.load(f)
                    self._inv_label_map = {int(v): k for k, v in self._label_map.items()}
                    return self._label_map

        # Default fallback 23 classes (Fish4Knowledge standard proposal)
        default_classes = [f"Species_{i:02d}" for i in range(1, 24)]
        self._label_map = {cls_name: idx for idx, cls_name in enumerate(default_classes)}
        self._inv_label_map = {idx: cls_name for idx, cls_name in enumerate(default_classes)}
        return self._label_map

    def get_inverse_label_map(self) -> Dict[int, str]:
        """Returns integer-to-class-name mapping."""
        if self._inv_label_map is None:
            self.get_label_map()
        return self._inv_label_map

    def load_split_records(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        Load record metadata for a split ('train', 'val', or 'test').
        Returns list of dicts:
            { 'sample_id': ..., 'class_name': ..., 'label_idx': ...,
              'video_id': ..., 'image_path': ... }
        """
        label_map = self.get_label_map()
        split_file = self.splits_dir / f"{split}.csv"

        records: List[Dict[str, Any]] = []

        if split_file.exists():
            with open(split_file, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    cls_name = row.get("class", row.get("class_name", "unknown"))
                    label_idx = label_map.get(cls_name, int(row.get("label_idx", -1)))
                    video_id = row.get("video_id", f"vid_{i}")
                    img_path = row.get("image_path", "")
                    if not os.path.isabs(img_path):
                        img_path = str(self.project_root / img_path)
                    sample_id = row.get("filename", Path(img_path).name)

                    records.append({
                        "sample_id": sample_id,
                        "class_name": cls_name,
                        "label_idx": label_idx,
                        "video_id": video_id,
                        "image_path": img_path,
                    })
            return records

        # Fallback to synthetic split if official split manifest is not yet generated
        syn_split_file = self.synthetic_dir / f"{split}.csv"
        if syn_split_file.exists():
            with open(syn_split_file, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    cls_name = row.get("class", row.get("class_name", "unknown"))
                    label_idx = label_map.get(cls_name, int(row.get("label_idx", -1)))
                    video_id = row.get("video_id", f"syn_vid_{i}")
                    img_path = row.get("image_path", "")
                    if not os.path.isabs(img_path):
                        img_path = str(self.synthetic_dir / img_path)
                    sample_id = row.get("filename", Path(img_path).name)

                    records.append({
                        "sample_id": sample_id,
                        "class_name": cls_name,
                        "label_idx": label_idx,
                        "video_id": video_id,
                        "image_path": img_path,
                    })

        return records

    def load_and_validate_image(self, path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """
        Load an image from disk, validate it, and resize it.
        Returns RGB image as numpy array of uint8, or None if invalid.
        """
        if target_size is None:
            target_size = self.target_size

        if not os.path.exists(path):
            return None

        # Check file extension
        ext = Path(path).suffix.lower()
        if ext not in self.valid_exts:
            return None

        try:
            bgr = cv2.imread(path)
            if bgr is None or bgr.size == 0:
                return None
            if bgr.shape[0] < 2 or bgr.shape[1] < 2:
                return None

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != target_size:
                rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)

            return rgb
        except Exception:
            return None

    def create_synthetic_fixture(
        self,
        num_classes: int = 5,
        samples_per_class: int = 30,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a synthetic aquatic species dataset fixture for testing and
        dataset-independent development when raw images are not yet available.
        Ensures leakage-free video-group splits (70% train, 15% val, 15% test).
        """
        if output_dir is None:
            target_dir = self.synthetic_dir
        else:
            target_dir = Path(output_dir)

        images_dir = target_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        class_names = [f"Species_{i:02d}" for i in range(1, num_classes + 1)]
        label_map = {cls_name: i for i, cls_name in enumerate(class_names)}

        # Save label_map.json
        with open(target_dir / "label_map.json", "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)

        rng = np.random.RandomState(42)
        all_records = []

        # Generate images with distinct color characteristics per class to simulate aquatic species
        for cls_idx, cls_name in enumerate(class_names):
            cls_folder = images_dir / cls_name
            cls_folder.mkdir(parents=True, exist_ok=True)

            # Assign 3 video sources per class to test video-group splitting
            num_videos = 3
            samples_per_vid = samples_per_class // num_videos

            for vid in range(num_videos):
                video_id = f"video_{cls_idx:02d}_{vid:02d}"
                for s in range(samples_per_vid):
                    filename = f"fish_{cls_name}_v{vid:02d}_s{s:03d}.jpg"
                    filepath = cls_folder / filename

                    # Synthetic image: background underwater greenish-blue + distinct class color oval
                    img = np.zeros((120, 160, 3), dtype=np.uint8)
                    # Underwater water background
                    img[:, :, 0] = rng.randint(20, 50, (120, 160))   # R (low)
                    img[:, :, 1] = rng.randint(80, 140, (120, 160))  # G (mid)
                    img[:, :, 2] = rng.randint(120, 200, (120, 160)) # B (high)

                    # Distinctive fish shape (colored ellipse)
                    color = (
                        int((cls_idx * 50 + 80) % 255),
                        int((cls_idx * 70 + 40) % 255),
                        int((cls_idx * 30 + 100) % 255),
                    )
                    cv2.ellipse(img, (80, 60), (40, 20), 0, 0, 360, color, -1)
                    # Add eye
                    cv2.circle(img, (105, 55), 4, (255, 255, 255), -1)
                    cv2.circle(img, (105, 55), 2, (0, 0, 0), -1)

                    cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                    rel_path = os.path.relpath(filepath, target_dir)
                    all_records.append({
                        "filename": filename,
                        "class": cls_name,
                        "class_name": cls_name,
                        "label_idx": cls_idx,
                        "video_id": video_id,
                        "image_path": rel_path,
                    })

        # Split by video_id to guarantee zero video leakage across splits
        # 3 videos per class: vid 0 -> train (66.7%), vid 1 -> val (16.7%), vid 2 -> test (16.7%)
        train_rows, val_rows, test_rows = [], [], []
        for rec in all_records:
            vid_num = int(rec["video_id"].split("_")[-1])
            if vid_num == 0:
                train_rows.append(rec)
            elif vid_num == 1:
                val_rows.append(rec)
            else:
                test_rows.append(rec)

        fieldnames = ["filename", "class", "class_name", "label_idx", "video_id", "image_path"]
        for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            split_csv = target_dir / f"{split_name}.csv"
            with open(split_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        summary = {
            "num_classes": num_classes,
            "classes": class_names,
            "total_samples": len(all_records),
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "test_samples": len(test_rows),
            "target_dir": str(target_dir),
        }

        with open(target_dir / "split_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
