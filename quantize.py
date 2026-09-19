"""P7: post-training uniform 8-bit quantization of the stored representation."""

from dataclasses import dataclass

import torch

from neural import NeuralTexture

LEVELS = 256                   # 8 bits
BYTES_PER_RANGE = 2 * 4        # each quantized array also stores (lo, scale) as float32


@dataclass
class QuantizedArray:
    """The stored form of one array: uint8 indices plus the range to decode them."""
    q: torch.Tensor            # uint8, same shape as the original
    lo: float
    scale: float

    def dequantize(self, dtype=torch.float32) -> torch.Tensor:
        return self.lo + self.q.to(dtype) * self.scale

    @property
    def nbytes(self) -> int:
        return self.q.numel() + BYTES_PER_RANGE


def quantize_uint8(x: torch.Tensor) -> QuantizedArray:
    """Round ``x`` to 256 evenly spaced levels between its own min and max."""
    lo, hi = x.min().item(), x.max().item()
    scale = (hi - lo) / (LEVELS - 1)
    if scale == 0:                        # constant array: every value is lo
        return QuantizedArray(torch.zeros_like(x, dtype=torch.uint8), lo, 0.0)
    q = torch.round((x - lo) / scale).clamp(0, LEVELS - 1).to(torch.uint8)
    return QuantizedArray(q, lo, scale)


@dataclass
class QuantizedTexture:
    """A NeuralTexture with its grids (and optionally MLP) stored as 8-bit arrays."""
    grids: list                # QuantizedArray per level
    mlp: list                  # QuantizedArray per parameter, or float32 tensors

    @property
    def nbytes(self) -> int:
        grid_bytes = sum(g.nbytes for g in self.grids)
        mlp_bytes = sum(p.nbytes if isinstance(p, QuantizedArray) else p.numel() * 4
                        for p in self.mlp)
        return grid_bytes + mlp_bytes

    @torch.no_grad()
    def load_into(self, model: NeuralTexture) -> NeuralTexture:
        """Decode the stored arrays into ``model``'s parameters, in place."""
        for grid, stored in zip(model.grid.grids, self.grids):
            grid.data.copy_(stored.dequantize().to(grid.device))
        for p, stored in zip(model.mlp.parameters(), self.mlp):
            value = stored.dequantize() if isinstance(stored, QuantizedArray) else stored
            p.data.copy_(value.to(p.device))
        return model


@torch.no_grad()
def quantize_model(model: NeuralTexture, quantize_mlp: bool = False) -> QuantizedTexture:
    """Build the 8-bit representation of ``model`` and load its decoded values
    back into ``model`` so rendering afterward uses the quantized weights."""
    stored = QuantizedTexture(
        grids=[quantize_uint8(grid.data.cpu()) for grid in model.grid.grids],
        mlp=[quantize_uint8(p.data.cpu()) if quantize_mlp else p.data.cpu().clone()
             for p in model.mlp.parameters()],
    )
    stored.load_into(model)
    return stored
