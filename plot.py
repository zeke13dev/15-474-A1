"""P6 figures from results/results.json: PSNR curves and size-vs-quality."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARCH_COLORS = {"small": "#2a78d6", "medium": "#eb6834", "large": "#1baf7a"}
S3TC_COLOR = "#52514e"
TEXTURE_MARKERS = {"gradient": "o", "bricks": "s", "clouds": "^",
                   "sky": "v", "sunset": "D", "grass": "P"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(results, path):
    textures = list(results)
    ncols = min(3, len(textures))
    nrows = -(-len(textures) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows),
                             sharey=True, squeeze=False)
    axes = axes.reshape(-1)
    for ax in axes[len(textures):]:
        ax.set_visible(False)
    for ax, name in zip(axes, textures):
        entry = results[name]
        for arch, color in ARCH_COLORS.items():
            ax.plot(entry[arch]["history"], color=color, linewidth=1.6, label=arch.capitalize())
        ax.axhline(entry["s3tc"]["psnr"], color=S3TC_COLOR, linestyle="--", linewidth=1.4, label="S3TC")
        ax.set_title(name, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("training step", color=MUTED)
        style(ax)
    for row in range(nrows):
        axes[row * ncols].set_ylabel("PSNR (dB)", color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("PSNR over training, three architectures per texture", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_size_vs_quality(results, path, quantized_key=None):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for name, marker in TEXTURE_MARKERS.items():
        if name not in results:
            continue
        entry = results[name]
        for arch, color in ARCH_COLORS.items():
            kb = entry[arch]["bytes"] / 1024
            ax.scatter(kb, entry[arch]["psnr"], marker=marker, s=60, color=color, zorder=3)
            if quantized_key and quantized_key in entry[arch]:
                q = entry[arch][quantized_key]
                ax.scatter(q["bytes"] / 1024, q["psnr"], marker=marker, s=60,
                           facecolors="none", edgecolors=color, linewidths=1.6, zorder=3)
                ax.plot([kb, q["bytes"] / 1024], [entry[arch]["psnr"], q["psnr"]],
                        color=color, linewidth=0.8, alpha=0.5)
        ax.scatter(entry["s3tc"]["bytes"] / 1024, entry["s3tc"]["psnr"],
                   marker=marker, s=60, color=S3TC_COLOR, zorder=3)
    # Legends: color = architecture, marker = texture.
    from matplotlib.lines import Line2D
    arch_handles = [Line2D([], [], marker="o", linestyle="", color=c, label=a.capitalize())
                    for a, c in ARCH_COLORS.items()]
    arch_handles.append(Line2D([], [], marker="o", linestyle="", color=S3TC_COLOR, label="S3TC"))
    if quantized_key:
        arch_handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                                   color=INK, label="8-bit quantized"))
    tex_handles = [Line2D([], [], marker=m, linestyle="", color=INK, label=n)
                   for n, m in TEXTURE_MARKERS.items() if n in results]
    # Both legends outside the plot area so they never cover a point.
    first = ax.legend(handles=arch_handles, frameon=False, fontsize=9,
                      loc="upper left", bbox_to_anchor=(1.01, 1.0), title="method")
    ax.add_artist(first)
    ax.legend(handles=tex_handles, frameon=False, fontsize=9,
              loc="upper left", bbox_to_anchor=(1.01, 0.5), title="texture")
    ax.set_xscale("log")
    ax.set_xlabel("representation size (KB, log scale)", color=MUTED)
    ax.set_ylabel("PSNR (dB)", color=MUTED)
    ax.set_title("Size vs. quality: neural fits against S3TC", loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_frequency(results, path):
    """Final PSNR against each texture's high-frequency energy fraction."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, entry in results.items():
        hf = entry["high_freq_fraction"]
        for arch, color in ARCH_COLORS.items():
            ax.scatter(hf, entry[arch]["psnr"], marker=TEXTURE_MARKERS.get(name, "o"),
                       s=60, color=color, zorder=3)
        ax.scatter(hf, entry["s3tc"]["psnr"], marker=TEXTURE_MARKERS.get(name, "o"),
                   s=60, color=S3TC_COLOR, zorder=3)
        ax.annotate(name, (hf, entry["large"]["psnr"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=9, color=MUTED)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=a.capitalize())
               for a, c in ARCH_COLORS.items()]
    handles.append(Line2D([], [], marker="o", linestyle="", color=S3TC_COLOR, label="S3TC"))
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")
    ax.set_xscale("log")
    ax.set_xlabel("fraction of energy above 32 cycles/image (log scale)", color=MUTED)
    ax.set_ylabel("final PSNR (dB)", color=MUTED)
    ax.set_title("Quality falls as high-frequency content rises", loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reconstructions(results, results_dir, path, crop=None):
    """Original | S3TC | Small | Medium | Large for every texture.

    ``crop`` = (y, x, size) shows a zoomed region instead of the full image.
    """
    from PIL import Image
    columns = ["original", "s3tc", "small", "medium", "large"]
    names = list(results)
    fig, axes = plt.subplots(len(names), len(columns),
                             figsize=(2.2 * len(columns), 2.3 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        for col, method in enumerate(columns):
            ax = axes[row][col]
            file = Path("textures") / f"{name}.png" if method == "original" \
                else Path(results_dir) / name / f"{method}.png"
            image = Image.open(file)
            if crop:
                y, x, size = crop
                image = image.crop((x, y, x + size, y + size)).resize((256, 256), Image.NEAREST)
            ax.imshow(image)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                label = method.upper() if method == "s3tc" else method.capitalize()
                if method != "original":
                    label += f"  {results[name][method]['psnr']:.1f} dB"
                ax.set_title(label, fontsize=9, color=INK)
            elif method != "original":
                ax.set_title(f"{results[name][method]['psnr']:.1f} dB", fontsize=9, color=MUTED)
        axes[row][0].set_ylabel(name, fontsize=10, color=INK)
    title = "Reconstructions" + (f", {crop[2]}px crop at ({crop[1]}, {crop[0]})" if crop else "")
    fig.suptitle(title, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_table(results):
    print(f"{'texture':9s} {'method':8s} {'PSNR':>7s} {'KB':>8s} {'ratio':>7s} {'factor':>7s}")
    for name, entry in results.items():
        raw = entry["raw_bytes"]
        for method in ["s3tc", "small", "medium", "large"]:
            r = entry[method]
            print(f"{name:9s} {method:8s} {r['psnr']:7.2f} {r['bytes'] / 1024:8.1f} "
                  f"{r['bytes'] / raw:7.4f} {raw / r['bytes']:6.1f}x")
            if "quantized" in r:
                q = r["quantized"]
                print(f"{'':9s} {method + '-q8':8s} {q['psnr']:7.2f} {q['bytes'] / 1024:8.1f} "
                      f"{q['bytes'] / raw:7.4f} {raw / q['bytes']:6.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--quantized-key", default="quantized",
                        help="results key holding post-quantization psnr/bytes (P7); '' to hide")
    args = parser.parse_args()
    results_dir = Path(args.results)
    results = json.loads((results_dir / "results.json").read_text())
    plot_curves(results, results_dir / "psnr_curves.png")
    plot_size_vs_quality(results, results_dir / "size_vs_quality.png", args.quantized_key or None)
    plot_frequency(results, results_dir / "psnr_vs_frequency.png")
    plot_reconstructions(results, results_dir, results_dir / "reconstructions.png")
    plot_reconstructions(results, results_dir, results_dir / "crops.png", crop=(192, 192, 96))
    print_table(results)
    print()
    for name, entry in results.items():
        print(f"{name:9s} high-frequency fraction {entry['high_freq_fraction']:.4f}")


if __name__ == "__main__":
    main()
