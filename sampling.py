"""P1: bilinear texture sampling shared by every representation."""

from math import floor


class BilinearSampler:
    """Mixin that turns integer texel lookups into continuous (u, v) sampling.

    Subclasses provide ``width``, ``height``, and ``_sample_raw(x, y)``, which
    returns the stored color of the integer texel at column ``x``, row ``y``.
    """

    def sample(self, u: float, v: float):
        """Return the bilinearly blended color at texture coordinate (u, v).

        Texel (x, y) is centered at ((x + 0.5) / width, (y + 0.5) / height).
        Coordinates outside [0, 1] clamp to the border texels, matching
        ``grid_sample(..., align_corners=False, padding_mode="border")``.
        """
        x = self._clamp(u * self.width - 0.5, self.width)
        y = self._clamp(v * self.height - 0.5, self.height)

        # Integer neighbors on either side, and the fractional position between them.
        x0, y0 = floor(x), floor(y)
        x1, y1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1)
        s, t = x - x0, y - y0

        c00 = self._sample_raw(x0, y0)
        c10 = self._sample_raw(x1, y0)
        c01 = self._sample_raw(x0, y1)
        c11 = self._sample_raw(x1, y1)
        return (
            (1 - s) * (1 - t) * c00
            + s * (1 - t) * c10
            + (1 - s) * t * c01
            + s * t * c11
        )

    @staticmethod
    def _clamp(coordinate: float, size: int) -> float:
        """Clamp a continuous texel-space coordinate to [0, size - 1]."""
        return min(max(coordinate, 0.0), size - 1)
