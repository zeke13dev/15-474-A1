# A1: Neural Texture Compression

15-474 Assignment 1. A bilinear texture sampler (P1), an S3TC/DXT1 block
compressor (P2), a multi-resolution feature grid plus tiny MLP fit to a single
texture (P3-P5), experiments across three architectures and six textures
(P6, P8), and post-training 8-bit quantization (P7).

## Layout

| file | purpose |
|---|---|
| `sampling.py` | P1 bilinear `sample(u, v)` mixin shared by every representation |
| `texture.py` | uncompressed reference texture |
| `s3tc.py` | P2 DXT1 block compression: RGB565 endpoints, 2-bit indices, 8 bytes per 4x4 block |
| `neural.py` | P3 `FeatureGrid`, P4 `ColorMLP`, P5 `NeuralTexture`, `train`, `reconstruct` |
| `quantize.py` | P7 uniform 8-bit quantization and the stored `QuantizedTexture` form |
| `metrics.py` | PSNR, raw size, high-frequency energy fraction |
| `utils.py` | device selection and PNG writing |
| `baseline.py` | CLI: S3TC on one texture |
| `train.py` | CLI: fit one architecture to one texture |
| `experiments.py` | CLI: S3TC + all three architectures + quantization on a set of textures |
| `plot.py` | figures and summary table from the saved results |
| `test_sampling.py`, `test_neural.py` | unit tests (22) |
| `textures/` | the three provided textures and the three self-sourced ones |
| `results/` | one folder of reconstructions per texture, `results.json`, and the figures |
| `notes/` | debugging notes from the P1/P2 checkpoint |

## Setup

```sh
python -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

PyTorch picks `cuda`, then `mps`, then `cpu`.

## Reproduce

```sh
venv/bin/python -m unittest test_sampling test_neural
venv/bin/python experiments.py                                # gradient, bricks, clouds
venv/bin/python experiments.py --textures sky sunset grass    # merged into the same results.json
venv/bin/python plot.py                                       # figures into results/
```

The full set of nine fits takes about two minutes on an M-series Mac.
`experiments.py` writes every reconstruction (`<arch>.png`, `<arch>-q8.png`,
`s3tc.png`) into `results/<texture>/` and merges each texture's PSNR curve,
sizes, and quantized PSNR/size into `results/results.json`. `plot.py` reads
only the JSON, so figures can be regenerated without retraining.

Single runs:

```sh
venv/bin/python baseline.py textures/bricks.png --output bricks-s3tc.png
venv/bin/python train.py textures/bricks.png --arch large --output bricks-large.png
```

## Conventions

- Texel (x, y) is centered at ((x + 0.5) / W, (y + 0.5) / H). Coordinates
  outside [0, 1] clamp to the border. The neural grid uses
  `grid_sample(align_corners=False, padding_mode="border")`, which matches
  the NumPy sampler exactly (tested).
- PSNR uses MAX = 1 on float images before PNG rounding.
- Size is the count of stored numbers times bytes per number: 4 for float32,
  1 for 8-bit values plus 8 bytes of (lo, scale) per quantized array. S3TC is
  W * H / 2 bytes. Compression factor is raw bytes / stored bytes.
- Training: Adam, lr 1e-2, 16,384 random texel centers per step, 2000 steps,
  MSE loss. The final PSNR is measured on the full reconstruction, not the
  last minibatch.
