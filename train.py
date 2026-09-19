"""Fit a NeuralTexture to one image and report PSNR and size."""

import argparse
from pathlib import Path

from metrics import psnr, raw_rgb_bytes
from neural import NeuralTexture, reconstruct, train
from texture import Texture
from utils import get_device, write_pixels

# P6 architectures: (resolutions, feat_dim). The MLP is always 2 x 64.
ARCHITECTURES = {
    "small": ((64,), 2),
    "medium": ((16, 32, 64), 2),
    "large": ((16, 32, 64, 128), 4),
}


def parse_args():
    parser = argparse.ArgumentParser(description="P5 neural texture training")
    parser.add_argument("texture")
    parser.add_argument("--arch", choices=ARCHITECTURES, default="large")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", default="neural.png")
    parser.add_argument("--device", default=get_device())
    args = parser.parse_args()
    if Path(args.texture).resolve() == Path(args.output).resolve():
        parser.error("Choose an output path different from the original texture")
    return args


def main():
    args = parse_args()
    texture = Texture(args.texture)
    resolutions, feat_dim = ARCHITECTURES[args.arch]
    model = NeuralTexture(resolutions, feat_dim)

    train(model, texture.texels, args.device, steps=args.steps, log_every=200)
    image = reconstruct(model, texture.width, texture.height, args.device)
    write_pixels(image, args.output)

    raw_bytes = raw_rgb_bytes(texture.width, texture.height)
    print(f"Architecture: {args.arch}  grid {model.grid.num_params} + mlp {model.mlp.num_params} params")
    print(f"PSNR: {psnr(image, texture.texels):.2f} dB")
    print(f"Raw RGB: {raw_bytes} bytes; neural (float32): {model.nbytes} bytes")
    print(f"Compressed/raw ratio: {model.nbytes / raw_bytes:.4f}")
    print(f"Compression factor: {raw_bytes / model.nbytes:.2f}x")
    print(f"Reconstruction: {args.output}")


if __name__ == "__main__":
    main()
