"""Small helpers shared by the command-line scripts."""

import numpy as np
import torch
from PIL import Image


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_pixels(pixels, filename) -> None:
    """Save a float image in [0, 1] as an 8-bit PNG."""
    as_uint8 = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(as_uint8).save(filename)
