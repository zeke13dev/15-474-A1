"""P1: the uncompressed reference texture."""

import numpy as np
from PIL import Image

from sampling import BilinearSampler


class Texture(BilinearSampler):
    """A texture stored as a full (height, width, 3) float32 array in [0, 1]."""

    def __init__(self, filename: str):
        self.texels = self._read_pixels(filename)
        self.height, self.width = self.texels.shape[:2]

    @staticmethod
    def _read_pixels(filename: str) -> np.ndarray:
        # Normalized floats keep blending and color distances from overflowing.
        with Image.open(filename) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    def _sample_raw(self, x: int, y: int) -> np.ndarray:
        # NumPy images are indexed [row, column]; x is the column.
        return self.texels[y, x]
