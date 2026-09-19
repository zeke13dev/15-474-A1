import unittest

import torch

from neural import ColorMLP, FeatureGrid, NeuralTexture, reconstruct, texel_centers, train


class FeatureGridTests(unittest.TestCase):
    def test_output_shape(self):
        grid = FeatureGrid(resolutions=(16, 32, 64), feat_dim=2)
        out = grid(torch.rand(100, 2))
        self.assertEqual(grid.out_dim, 6)
        self.assertEqual(out.shape, (100, 6))
        self.assertEqual(grid.num_params, 2 * (16 ** 2 + 32 ** 2 + 64 ** 2))

    def test_cell_center_returns_stored_feature(self):
        grid = FeatureGrid(resolutions=(4,), feat_dim=3)
        x, y = 1, 2
        uv = torch.tensor([[(x + 0.5) / 4, (y + 0.5) / 4]])
        # The tensor is indexed [row, column], i.e. [y, x].
        expected = grid.grids[0][0, :, y, x]
        torch.testing.assert_close(grid(uv)[0], expected)

    def test_midpoint_blends_two_cells(self):
        grid = FeatureGrid(resolutions=(4,), feat_dim=1)
        with torch.no_grad():
            grid.grids[0].zero_()
            grid.grids[0][0, 0, 0, 0] = 1.0
            grid.grids[0][0, 0, 0, 1] = 3.0
        # Halfway between the centers of cells (0, 0) and (1, 0) on row 0.
        uv = torch.tensor([[1 / 4, 0.5 / 4]])
        torch.testing.assert_close(grid(uv)[0, 0], torch.tensor(2.0))

    def test_border_clamps_instead_of_fading_to_zero(self):
        grid = FeatureGrid(resolutions=(4,), feat_dim=1)
        with torch.no_grad():
            grid.grids[0].fill_(5.0)
        uv = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
        torch.testing.assert_close(grid(uv), torch.full((3, 1), 5.0))

    def test_gradients_reach_every_level(self):
        grid = FeatureGrid(resolutions=(8, 16), feat_dim=2)
        grid(torch.rand(50, 2)).sum().backward()
        for level in grid.grids:
            self.assertIsNotNone(level.grad)
            self.assertGreater(level.grad.abs().sum().item(), 0)

    def test_matches_numpy_texture_sampler(self):
        # A 1-level grid with 3 features is just a texture; it must agree with P1.
        import numpy as np
        from texture import Texture
        from PIL import Image
        import tempfile
        from pathlib import Path

        rng = np.random.default_rng(1)
        pixels = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.png"
            Image.fromarray(pixels).save(path)
            texture = Texture(str(path))

        grid = FeatureGrid(resolutions=(8,), feat_dim=3)
        with torch.no_grad():
            grid.grids[0].copy_(torch.from_numpy(texture.texels).permute(2, 0, 1)[None])
        uv = rng.uniform(-0.1, 1.1, (64, 2)).astype(np.float32)
        expected = np.array([texture.sample(float(u), float(v)) for u, v in uv])
        actual = grid(torch.from_numpy(uv)).detach().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)


class ColorMLPTests(unittest.TestCase):
    def test_output_shape_and_range(self):
        mlp = ColorMLP(in_dim=6)
        out = mlp(torch.randn(100, 6) * 10)
        self.assertEqual(out.shape, (100, 3))
        self.assertTrue(((out >= 0) & (out <= 1)).all())

    def test_param_count_matches_architecture(self):
        # Linear(in, 64) + Linear(64, 64) + Linear(64, 3), weights plus biases.
        in_dim = 8
        expected = (in_dim * 64 + 64) + (64 * 64 + 64) + (64 * 3 + 3)
        self.assertEqual(ColorMLP(in_dim).num_params, expected)

    def test_gradients_flow_from_color_back_to_grid(self):
        grid = FeatureGrid(resolutions=(8, 16), feat_dim=2)
        mlp = ColorMLP(grid.out_dim)
        color = mlp(grid(torch.rand(32, 2)))
        color.sum().backward()
        for level in grid.grids:
            self.assertGreater(level.grad.abs().sum().item(), 0)
        for p in mlp.parameters():
            self.assertIsNotNone(p.grad)


class TrainingTests(unittest.TestCase):
    def test_texel_centers_are_row_major_and_centered(self):
        coords = texel_centers(width=4, height=2)
        self.assertEqual(coords.shape, (8, 2))
        torch.testing.assert_close(coords[0], torch.tensor([0.125, 0.25]))
        torch.testing.assert_close(coords[1], torch.tensor([0.375, 0.25]))   # next column
        torch.testing.assert_close(coords[4], torch.tensor([0.125, 0.75]))   # next row

    def test_overfits_a_small_texture(self):
        import numpy as np
        rng = np.random.default_rng(3)
        texels = rng.random((8, 8, 3), dtype=np.float32)
        model = NeuralTexture(resolutions=(8,), feat_dim=4)
        history = train(model, texels, device="cpu", steps=300, batch_size=64)
        image = reconstruct(model, width=8, height=8, device="cpu")
        self.assertEqual(image.shape, (8, 8, 3))
        self.assertGreater(history[-1], history[0])
        mse = np.mean((image - texels) ** 2)
        self.assertGreater(-10 * np.log10(mse), 30)

    def test_sizes(self):
        small = NeuralTexture(resolutions=(64,), feat_dim=2)
        self.assertEqual(small.grid.num_params, 64 * 64 * 2)
        self.assertEqual(small.nbytes, small.num_params * 4)


class QuantizationTests(unittest.TestCase):
    def test_roundtrip_error_bounded_by_half_step(self):
        from quantize import quantize_uint8
        x = torch.randn(1000) * 3
        stored = quantize_uint8(x)
        x_hat = stored.dequantize()
        self.assertEqual(stored.q.dtype, torch.uint8)
        torch.testing.assert_close(x_hat.min(), x.min())
        torch.testing.assert_close(x_hat.max(), x.max())
        self.assertLessEqual((x - x_hat).abs().max().item(), stored.scale / 2 + 1e-6)
        self.assertEqual(stored.nbytes, 1000 + 8)

    def test_constant_array(self):
        from quantize import quantize_uint8
        stored = quantize_uint8(torch.full((10,), 0.7))
        torch.testing.assert_close(stored.dequantize(), torch.full((10,), 0.7))
        self.assertEqual(int(stored.q.max()), 0)

    def test_quantize_model_size_quality_and_reload(self):
        import numpy as np
        from quantize import quantize_model
        rng = np.random.default_rng(5)
        texels = rng.random((8, 8, 3), dtype=np.float32)
        model = NeuralTexture(resolutions=(8,), feat_dim=4)
        train(model, texels, device="cpu", steps=300, batch_size=64)
        before = reconstruct(model, 8, 8, "cpu")
        stored = quantize_model(model)
        after = reconstruct(model, 8, 8, "cpu")
        # 1 byte per grid value + (lo, scale), MLP still float32.
        self.assertEqual(stored.nbytes, model.grid.num_params + 8 + model.mlp.num_params * 4)
        self.assertLess(stored.nbytes, model.nbytes)
        psnr = lambda img: -10 * np.log10(np.mean((img - texels) ** 2))
        self.assertGreater(psnr(after), psnr(before) - 3)   # small, bounded quality loss
        # Decoding the stored form into a fresh model reproduces the same image.
        fresh = stored.load_into(NeuralTexture(resolutions=(8,), feat_dim=4))
        np.testing.assert_allclose(reconstruct(fresh, 8, 8, "cpu"), after, atol=1e-6)

    def test_quantized_mlp_is_smaller(self):
        from quantize import quantize_model
        model = NeuralTexture(resolutions=(8,), feat_dim=2)
        full = quantize_model(model, quantize_mlp=True).nbytes
        self.assertLess(full, model.grid.num_params + 8 + model.mlp.num_params * 4)

if __name__ == "__main__":
    unittest.main()
