"""Evaluation helpers shared by the S3TC baseline and later neural runs."""

import numpy as np


def psnr(reconstruction, target) -> float:
    """PSNR in dB for images normalized to [0, 1] (MAX = 1)."""
    mse = np.mean((np.asarray(reconstruction, dtype=np.float64) - target) ** 2)
    return float("inf") if mse == 0 else -10 * np.log10(mse)


def raw_rgb_bytes(width: int, height: int) -> int:
    """Size of the uncompressed 8-bit RGB texture."""
    return width * height * 3


def high_frequency_fraction(texels, cutoff_cycles: float = 32.0) -> float:
    """Fraction of spectral energy above ``cutoff_cycles`` per image width.

    Ignores the DC term. A 64-cell grid can represent at most 32 cycles across
    the image (its Nyquist limit), so the default measures how much of the
    texture the Small model cannot resolve at all.
    """
    gray = np.asarray(texels, dtype=np.float64).mean(axis=2)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2
    h, w = gray.shape
    fy, fx = np.mgrid[-h // 2:h - h // 2, -w // 2:w - w // 2]
    radius = np.sqrt((fy * w / h) ** 2 + fx ** 2)      # cycles per image width
    total = spectrum.sum()
    return float(spectrum[radius > cutoff_cycles].sum() / total) if total else 0.0
