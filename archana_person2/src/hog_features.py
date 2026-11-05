"""
Histogram of Oriented Gradients (HOG) Feature Extraction - Person 2 (Archana Ulloju).

Extracts structural and edge gradient orientation descriptors for fish morphology
recognition using scikit-image.
"""

from typing import Tuple, Union
import numpy as np
import cv2
from skimage.feature import hog


class HOGFeatureExtractor:
    """
    Extracts Histogram of Oriented Gradients (HOG) descriptors.
    HOG captures local shape, edge gradients, and contours, which are invariant
    to illumination changes and essential for recognizing fish anatomical structures
    (fins, body outline, caudal tail shape).
    """

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (16, 16),
        cells_per_block: Tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.orientations = orientations
        self.pixels_per_cell = tuple(pixels_per_cell)
        self.cells_per_block = tuple(cells_per_block)
        self.block_norm = block_norm
        self.target_size = target_size

        # Compute deterministic feature dimensionality
        # n_blocks_x = (W / px_x) - cx + 1
        # n_blocks_y = (H / px_y) - cy + 1
        # total_features = n_blocks_y * n_blocks_x * cx * cy * orientations
        n_blocks_y = (self.target_size[0] // self.pixels_per_cell[0]) - self.cells_per_block[0] + 1
        n_blocks_x = (self.target_size[1] // self.pixels_per_cell[1]) - self.cells_per_block[1] + 1
        self.feature_dim = (
            n_blocks_y * n_blocks_x * self.cells_per_block[0] * self.cells_per_block[1] * self.orientations
        )

    def extract(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        Extract HOG feature vector from an RGB image array or image path.
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

        # Convert to Grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Extract HOG features
        hog_vector = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            visualize=False,
            feature_vector=True,
        ).astype(np.float32)

        assert len(hog_vector) == self.feature_dim, (
            f"Expected HOG dim {self.feature_dim}, got {len(hog_vector)}"
        )
        return hog_vector
