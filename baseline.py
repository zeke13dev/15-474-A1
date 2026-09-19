"""Run the P1/P2 S3TC baseline on one texture and report quality and size."""

import argparse
from pathlib import Path

from metrics import psnr, raw_rgb_bytes
from s3tc import S3TC
from texture import Texture
from utils import get_device, write_pixels


def parse_args():
    parser = argparse.ArgumentParser(description="P1/P2 texture compression baseline")
    parser.add_argument("texture", help="Input image; dimensions must be divisible by 4")
    parser.add_argument("--output", default="reconstruction.png")
    args = parser.parse_args()
    if Path(args.texture).resolve() == Path(args.output).resolve():
        parser.error("Choose an output path different from the original texture")
    return args


def main():
    args = parse_args()
    texture = Texture(args.texture)
    compressed = S3TC(texture)
    reconstruction = compressed.reconstruct()
    write_pixels(reconstruction, args.output)

    raw_bytes = raw_rgb_bytes(texture.width, texture.height)
    print(f"Device available for neural tasks: {get_device()}")
    print(f"PSNR: {psnr(reconstruction, texture.texels):.2f} dB")
    print(f"Raw RGB: {raw_bytes} bytes; S3TC: {compressed.nbytes} bytes")
    print(f"Compressed/raw ratio: {compressed.nbytes / raw_bytes:.4f}")
    print(f"Compression factor: {raw_bytes / compressed.nbytes:.2f}x")
    print(f"Reconstruction: {args.output}")


if __name__ == "__main__":
    main()
