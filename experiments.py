"""P6: fit every architecture to every texture, plus the S3TC baseline.

Writes results/<texture>/<arch>.png reconstructions and merges each texture's
PSNR curves and sizes into results/results.json, so plots can be regenerated
without retraining and separate runs accumulate in one place.
"""

import argparse
import json
from pathlib import Path

from metrics import high_frequency_fraction, psnr, raw_rgb_bytes
from neural import NeuralTexture, reconstruct, train
from quantize import quantize_model
from s3tc import S3TC
from texture import Texture
from utils import get_device, write_pixels
from train import ARCHITECTURES

TEXTURES = ("gradient", "bricks", "clouds")


def run_s3tc(texture: Texture, out_dir: Path) -> dict:
    compressed = S3TC(texture)
    image = compressed.reconstruct()
    write_pixels(image, out_dir / "s3tc.png")
    return {"psnr": psnr(image, texture.texels), "bytes": compressed.nbytes}


def run_neural(texture: Texture, arch: str, device: str, steps: int, out_dir: Path) -> dict:
    resolutions, feat_dim = ARCHITECTURES[arch]
    model = NeuralTexture(resolutions, feat_dim)
    history = train(model, texture.texels, device, steps=steps)
    image = reconstruct(model, texture.width, texture.height, device)
    write_pixels(image, out_dir / f"{arch}.png")
    result = {
        "psnr": psnr(image, texture.texels),
        "bytes": model.nbytes,
        "grid_params": model.grid.num_params,
        "mlp_params": model.mlp.num_params,
        "history": history,
    }

    # P7: quantize the grids to 8 bits and measure again with the same decoder.
    stored = quantize_model(model)
    quantized = reconstruct(model, texture.width, texture.height, device)
    write_pixels(quantized, out_dir / f"{arch}-q8.png")
    result["quantized"] = {"psnr": psnr(quantized, texture.texels), "bytes": stored.nbytes}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--textures", nargs="+", default=list(TEXTURES))
    parser.add_argument("--texture-dir", default="textures")
    parser.add_argument("--results", default="results")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default=get_device())
    args = parser.parse_args()

    results_dir = Path(args.results)
    results = {}
    for name in args.textures:
        texture = Texture(str(Path(args.texture_dir) / f"{name}.png"))
        out_dir = results_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "width": texture.width,
            "height": texture.height,
            "raw_bytes": raw_rgb_bytes(texture.width, texture.height),
            "high_freq_fraction": high_frequency_fraction(texture.texels),
            "s3tc": run_s3tc(texture, out_dir),
        }
        print(f"{name:9s} s3tc    {entry['s3tc']['psnr']:6.2f} dB  {entry['s3tc']['bytes']:7d} B")
        for arch in ARCHITECTURES:
            entry[arch] = run_neural(texture, arch, args.device, args.steps, out_dir)
            r, q = entry[arch], entry[arch]["quantized"]
            print(f"{name:9s} {arch:7s} {r['psnr']:6.2f} dB  {r['bytes']:7d} B"
                  f"   -> 8-bit {q['psnr']:6.2f} dB  {q['bytes']:7d} B")
        results[name] = entry

    # Merge into any existing results so separate runs accumulate in one file.
    path = results_dir / "results.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(results)
    path.write_text(json.dumps(merged))
    print(f"Wrote {path} ({', '.join(merged)})")


if __name__ == "__main__":
    main()
