"""P2: S3TC / DXT1 (BC1) block compression."""

import numpy as np

from sampling import BilinearSampler
from texture import Texture

BLOCK_SIZE = 4
TEXELS_PER_BLOCK = BLOCK_SIZE * BLOCK_SIZE
INDEX_BITS = 2  # each texel stores a 2-bit palette index
INDEX_MASK = (1 << INDEX_BITS) - 1

# RGB565: 5 bits red, 6 bits green, 5 bits blue.
RGB565_LEVELS = np.array([31, 63, 31], dtype=np.float32)
RGB565_SHIFTS = (11, 5, 0)


def pack_565(color) -> int:
    """Quantize an RGB color in [0, 1] to a 16-bit RGB565 code."""
    quantized = np.rint(np.clip(color, 0, 1) * RGB565_LEVELS).astype(int)
    return sum(int(q) << shift for q, shift in zip(quantized, RGB565_SHIFTS))


def unpack_565(code) -> np.ndarray:
    """Decode a 16-bit RGB565 code back to RGB in [0, 1].

    Uses the ``q / levels`` approximation permitted by the assignment.
    """
    code = int(code)
    quantized = [(code >> shift) & int(level)
                 for shift, level in zip(RGB565_SHIFTS, RGB565_LEVELS)]
    return np.array(quantized, dtype=np.float32) / RGB565_LEVELS


def palette_from(endpoints) -> np.ndarray:
    """Build the 4-color palette interpolated between two RGB565 endpoints."""
    c0, c1 = (unpack_565(code) for code in endpoints)
    return np.array([c0, c1, (2 * c0 + c1) / 3, (c0 + 2 * c1) / 3])


def pack_indices(indices) -> int:
    """Pack 16 palette indices (row-major) into one 32-bit word."""
    return sum(int(index) << (INDEX_BITS * i) for i, index in enumerate(indices))


def unpack_indices(word) -> list:
    """Inverse of ``pack_indices``."""
    word = int(word)
    return [(word >> (INDEX_BITS * i)) & INDEX_MASK for i in range(TEXELS_PER_BLOCK)]


class S3TC(BilinearSampler):
    """A texture compressed to 8 bytes per 4x4 block.

    Per block we store two RGB565 endpoints (``endpoints``, uint16 x 2) and
    sixteen 2-bit palette indices packed into one uint32 (``indices``).
    """

    def __init__(self, texture: Texture):
        self.compress(texture)

    # ----------------------------------------------------------------- encode

    def compress(self, texture: Texture) -> None:
        """Encode every 4x4 block of ``texture``, replacing any prior state."""
        if texture.width % BLOCK_SIZE or texture.height % BLOCK_SIZE:
            raise ValueError(
                f"S3TC currently requires width and height divisible by {BLOCK_SIZE}"
            )
        self.width, self.height = texture.width, texture.height
        self.block_cols = self.width // BLOCK_SIZE
        self.block_rows = self.height // BLOCK_SIZE

        block_count = self.block_rows * self.block_cols
        self.endpoints = np.empty((block_count, 2), dtype=np.uint16)
        self.indices = np.empty(block_count, dtype=np.uint32)

        for block, (by, bx) in enumerate(self._block_positions()):
            pixels = self._block_slice(texture.texels, by, bx).reshape(TEXELS_PER_BLOCK, 3)
            self.endpoints[block], self.indices[block] = self._encode_block(pixels)

    @staticmethod
    def _choose_endpoints(pixels) -> list:
        """Pick the two endpoint colors for a block of (16, 3) pixels.

        Simple baseline: the corners of the block's per-channel RGB bounding
        box. These need not be colors actually present in the block.
        """
        return [pack_565(pixels.max(axis=0)), pack_565(pixels.min(axis=0))]

    @staticmethod
    def _nearest_palette_indices(pixels, palette) -> np.ndarray:
        """For each pixel, the index of the closest palette entry (squared RGB distance)."""
        distances = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

    def _encode_block(self, pixels):
        endpoints = self._choose_endpoints(pixels)
        indices = self._nearest_palette_indices(pixels, palette_from(endpoints))
        if endpoints[0] == endpoints[1]:
            # A constant block: every texel is endpoint 0.
            indices[:] = 0
        return endpoints, pack_indices(indices)

    # ----------------------------------------------------------------- decode

    def _sample_raw(self, x: int, y: int) -> np.ndarray:
        block = (y // BLOCK_SIZE) * self.block_cols + (x // BLOCK_SIZE)
        local = (y % BLOCK_SIZE) * BLOCK_SIZE + (x % BLOCK_SIZE)
        index = (int(self.indices[block]) >> (INDEX_BITS * local)) & INDEX_MASK
        return palette_from(self.endpoints[block])[index]

    def reconstruct(self) -> np.ndarray:
        """Decode the full image; equivalent to ``sample()`` at every texel center."""
        result = np.empty((self.height, self.width, 3), dtype=np.float32)
        for block, (by, bx) in enumerate(self._block_positions()):
            palette = palette_from(self.endpoints[block])
            pixels = palette[unpack_indices(self.indices[block])]
            self._block_slice(result, by, bx)[:] = pixels.reshape(BLOCK_SIZE, BLOCK_SIZE, 3)
        return result

    @property
    def nbytes(self) -> int:
        """Size of the stored representation: exactly 8 bytes per block."""
        return self.endpoints.nbytes + self.indices.nbytes

    # ---------------------------------------------------------------- helpers

    def _block_positions(self):
        """Yield (block_row, block_col) in the same row-major order as storage."""
        for by in range(self.block_rows):
            for bx in range(self.block_cols):
                yield by, bx

    @staticmethod
    def _block_slice(image, by: int, bx: int):
        """View of the 4x4 pixel region for block (by, bx)."""
        y0, x0 = by * BLOCK_SIZE, bx * BLOCK_SIZE
        return image[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE]
