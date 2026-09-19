import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from texture import Texture
from s3tc import S3TC


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def texture(self, pixels):
        path = Path(self.directory.name) / 'texture.png'
        Image.fromarray(pixels).save(path)
        return Texture(str(path))

    def test_centers_edges_and_midpoint(self):
        texture = self.texture(np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ], dtype=np.uint8))
        for y in range(2):
            for x in range(2):
                np.testing.assert_allclose(texture.sample((x + .5) / 2, (y + .5) / 2),
                                           texture.texels[y, x])
        np.testing.assert_allclose(texture.sample(.5, .5), [.5, .5, .5])
        np.testing.assert_allclose(texture.sample(0, 0), [1, 0, 0])
        np.testing.assert_allclose(texture.sample(1, 1), [1, 1, 1])
        np.testing.assert_allclose(texture.sample(-1, 2), [0, 0, 1])

    def test_matches_pytorch_on_rectangular_image(self):
        rng = np.random.default_rng(42)
        texture = self.texture(rng.integers(0, 256, (8, 12, 3), dtype=np.uint8))
        uv = rng.uniform(-.1, 1.1, (100, 2)).astype(np.float32)
        image = torch.from_numpy(texture.texels).permute(2, 0, 1).unsqueeze(0)
        coords = torch.from_numpy(uv * 2 - 1).reshape(1, 100, 1, 2)
        expected = F.grid_sample(image, coords, mode='bilinear', padding_mode='border',
                                 align_corners=False)[0, :, :, 0].T.numpy()
        actual = np.array([texture.sample(float(u), float(v)) for u, v in uv])
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_single_texel_and_rgb_conversion(self):
        texture = self.texture(np.array([[128]], dtype=np.uint8))
        np.testing.assert_allclose(texture.sample(.8, .2), np.full(3, 128 / 255))

    def test_s3tc_known_palette_and_packed_indices(self):
        # Row-major values select palette entries [1, 3, 2, 0].
        gray = np.tile(np.array([0, 85, 170, 255], dtype=np.uint8), (4, 1))
        pixels = np.repeat(gray[:, :, None], 3, axis=2)
        compressed = S3TC(self.texture(pixels))
        np.testing.assert_array_equal(compressed.endpoints[0], [65535, 0])
        expected_bits = sum(index << (2 * i)
                            for i, index in enumerate([1, 3, 2, 0] * 4))
        self.assertEqual(int(compressed.indices[0]), expected_bits)
        np.testing.assert_allclose(compressed.reconstruct(), pixels / 255, atol=1e-7)
        for y in range(4):
            for x in range(4):
                np.testing.assert_allclose(compressed.sample((x + .5) / 4, (y + .5) / 4),
                                           pixels[y, x] / 255, atol=1e-7)
        self.assertEqual(compressed.nbytes, 8)

    def test_blocks_and_bilinear_crossing(self):
        pixels = np.zeros((8, 12, 3), dtype=np.uint8)
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255],
                           [255, 255, 0], [255, 0, 255], [0, 255, 255]])
        for block, color in enumerate(colors):
            by, bx = divmod(block, 3)
            pixels[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4] = color
        texture = self.texture(pixels)
        compressed = S3TC(texture)
        np.testing.assert_allclose(compressed.reconstruct(), texture.texels)
        for y in range(8):
            for x in range(12):
                np.testing.assert_allclose(compressed.sample((x + .5) / 12, (y + .5) / 8),
                                           texture.texels[y, x])
        # Intersection of FOUR separately encoded blocks.
        np.testing.assert_allclose(compressed.sample(4 / 12, 4 / 8), [.75, .5, .25])
        self.assertEqual(compressed.nbytes, 48)
        # Recompressing replaces, rather than appends to, old state.
        compressed.compress(self.texture(np.zeros((4, 4, 3), dtype=np.uint8)))
        self.assertEqual(compressed.nbytes, 8)
        np.testing.assert_array_equal(compressed.sample(1, 1), [0, 0, 0])

    def test_s3tc_rejects_partial_blocks(self):
        with self.assertRaisesRegex(ValueError, 'divisible by 4'):
            S3TC(self.texture(np.zeros((5, 4, 3), dtype=np.uint8)))


if __name__ == '__main__':
    unittest.main()
