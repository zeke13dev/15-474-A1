"""P3-P5: feature grid, decoder MLP, and the training loop that fits them."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGrid(nn.Module):
    """Learnable feature vectors on several grids, blended bilinearly at (u, v).

    Each level stores a (1, feat_dim, R, R) tensor: an R x R grid of
    feat_dim-dimensional features. Looking up a coordinate interpolates the
    four surrounding cells at every level and concatenates the results, so the
    output for one coordinate has ``feat_dim * len(resolutions)`` values.
    """

    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=2, init_scale=1e-2):
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.feat_dim = feat_dim
        # ParameterList registers each grid so the optimizer and .to(device) see it.
        self.grids = nn.ParameterList(
            nn.Parameter(torch.randn(1, feat_dim, R, R) * init_scale)
            for R in self.resolutions
        )
        self.out_dim = feat_dim * len(self.resolutions)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """Map (N, 2) coordinates in [0, 1] to (N, out_dim) interpolated features."""
        n = uv.shape[0]
        # grid_sample wants (1, H_out, W_out, 2) in [-1, 1], x before y.
        # Treat the N samples as an N x 1 output image.
        coords = (uv.to(torch.float32) * 2 - 1).view(1, n, 1, 2)

        features = []
        for grid in self.grids:
            sampled = F.grid_sample(
                grid,
                coords,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )  # (1, feat_dim, N, 1)
            features.append(sampled[0, :, :, 0].T)  # (N, feat_dim)
        return torch.cat(features, dim=1)

    @property
    def num_params(self) -> int:
        return sum(grid.numel() for grid in self.grids)


class ColorMLP(nn.Module):
    """Decode a concatenated feature vector to an RGB color in [0, 1].

    Fixed architecture from the assignment: two hidden layers of width 64 with
    ReLU, then a 3-channel output squashed by a sigmoid. This is the only work
    the renderer runs per texel.
    """
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(N, in_dim) features -> (N, 3) RGB."""
        return self.net(features)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class NeuralTexture(nn.Module):
    """Grid + MLP: the compressed representation of one texture."""

    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=2):
        super().__init__()
        self.grid = FeatureGrid(resolutions, feat_dim)
        self.mlp = ColorMLP(self.grid.out_dim)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.grid(uv))

    @property
    def num_params(self) -> int:
        return self.grid.num_params + self.mlp.num_params

    @property
    def nbytes(self) -> int:
        """Size in float32; P7 quantization will shrink this."""
        return self.num_params * 4


def texel_centers(width: int, height: int) -> torch.Tensor:
    """All H*W texel-center coordinates as (N, 2) in [0, 1], row-major."""
    u = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    v = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    vv, uu = torch.meshgrid(v, u, indexing="ij")     # (H, W) each
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)


def build_dataset(texels: np.ndarray, device: str):
    """Pair every texel center with its ground-truth color. Built once, fixed."""
    height, width = texels.shape[:2]
    coords = texel_centers(width, height).to(device)
    target = torch.from_numpy(texels).reshape(-1, 3).to(device)
    return coords, target


def train(model: NeuralTexture, texels: np.ndarray, device: str,
          steps: int = 2000, batch_size: int = 16_384, lr: float = 1e-2,
          seed: int = 0, log_every: int = 0):
    """Fit ``model`` to ``texels`` with MSE. Returns per-step PSNR (dB)."""
    torch.manual_seed(seed)
    model.to(device)
    coords, target = build_dataset(texels, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = min(batch_size, len(coords))

    history = []
    for step in range(steps):
        batch = torch.randint(len(coords), (batch_size,), device=device)
        loss = torch.mean((model(coords[batch]) - target[batch]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        psnr = -10 * torch.log10(loss).item()
        history.append(psnr)
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:5d}  loss {loss.item():.6f}  psnr {psnr:.2f} dB")
    return history


@torch.no_grad()
def reconstruct(model: NeuralTexture, width: int, height: int,
                device: str, chunk: int = 65_536) -> np.ndarray:
    """Decode the model at every texel center into an (H, W, 3) float32 image."""
    model.eval()
    coords = texel_centers(width, height).to(device)
    colors = torch.cat([model(coords[i:i + chunk]) for i in range(0, len(coords), chunk)])
    model.train()
    return colors.reshape(height, width, 3).cpu().numpy()
