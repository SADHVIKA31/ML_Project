"""
Color Histogram Feature Extraction Module - Person 2 (Archana Ulloju).

Extracts normalized 3D color histograms from RGB and HSV color spaces for underwater
aquatic biodiversity imagery.
"""

from typing import Tuple, Optional, Union
import numpy as np
import cv2


class ColorHistogramExtractor:
    """
    Extracts multi-channel color histograms in RGB, HSV, or combined spaces.
    Underwater imagery exhibits unique absorption properties (red light absorbed fast,
    blue/green dominance); HSV captures hue and saturation independent of illumination.
    """

    def __init__(
        self,
        channels: str = "both",
        bins_rgb: Tuple[int, int, int] = (8, 8, 8),
        bins_hsv: Tuple[int, int, int] = (8, 8, 8),
        normalize: bool = True,
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.channels = channels.lower()
        if self.channels not in ("rgb", "hsv", "both"):
            raise ValueError(f"channels must be 'rgb', 'hsv', or 'both', got {channels}")

        self.bins_rgb = tuple(bins_rgb)
        self.bins_hsv = tuple(bins_hsv)
        self.normalize = normalize
        self.target_size = target_size

        # Compute deterministic feature dimensionality
        dim_rgb = int(np.prod(self.bins_rgb))
        dim_hsv = int(np.prod(self.bins_hsv))
        if self.channels == "rgb":
            self.feature_dim = dim_rgb
        elif self.channels == "hsv":
            self.feature_dim = dim_hsv
        else:
            self.feature_dim = dim_rgb + dim_hsv

    def _extract_3d_hist(self, img_channels: np.ndarray, bins: Tuple[int, int, int], ranges: list) -> np.ndarray:
        """Helper to extract and normalize a 3D histogram using OpenCV."""
        hist = cv2.calcHist([img_channels], [0, 1, 2], None, list(bins), ranges)
        hist = hist.flatten().astype(np.float32)

        if self.normalize:
            norm = np.linalg.norm(hist, ord=1)
            if norm > 1e-7:
                hist = hist / norm
            else:
                hist = np.zeros_like(hist)
        return hist

    def extract(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        Extract color histogram feature vector from an RGB image array or image path.
        Returns 1D numpy array of float32.
        """
        if isinstance(image, str):
            bgr = cv2.imread(image)
            if bgr is None or bgr.size == 0:
                raise ValueError(f"Could not read image from path: {image}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if image.size == 0 or image.ndim != 3:
                raise ValueError(f"Invalid image array with shape {image.shape}")
            rgb = image
        else:
            raise TypeError(f"Expected np.ndarray or str path, got {type(image)}")

        # Resize to standardized dimensions
        if rgb.shape[:2] != self.target_size:
            rgb = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_AREA)

        features = []

        if self.channels in ("rgb", "both"):
            # RGB ranges: 0-256 for all 3 channels
            rgb_ranges = [0, 256, 0, 256, 0, 256]
            hist_rgb = self._extract_3d_hist(rgb, self.bins_rgb, rgb_ranges)
            features.append(hist_rgb)

        if self.channels in ("hsv", "both"):
            # Convert RGB to HSV
            # Hue: 0-180 in OpenCV 8-bit, Saturation: 0-256, Value: 0-256
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            hsv_ranges = [0, 180, 0, 256, 0, 256]
            hist_hsv = self._extract_3d_hist(hsv, self.bins_hsv, hsv_ranges)
            features.append(hist_hsv)

        feature_vector = np.concatenate(features, axis=0).astype(np.float32)
        assert len(feature_vector) == self.feature_dim, (
            f"Expected dim {self.feature_dim}, got {len(feature_vector)}"
        )
        return feature_vector
