"""Per-point error heatmap export.

A single Chamfer number hides *where* a scanner fails. Exporting the
reconstruction coloured by distance-to-reference turns the metric into
something you can rotate in Meshlab and immediately see: floaters go red,
faithful surface stays blue.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geometry import write_ply_colored


# A compact blue→green→yellow→red ramp (perceptually ordered, no external deps).
_STOPS = np.array(
    [[49, 54, 149], [69, 117, 180], [116, 173, 209], [171, 217, 233],
     [255, 255, 191], [253, 174, 97], [244, 109, 67], [215, 48, 39], [165, 0, 38]],
    dtype=np.float64,
)


def _colormap(t: np.ndarray) -> np.ndarray:
    """Map ``t`` in [0, 1] to RGB uint8 via linear interpolation over _STOPS."""
    t = np.clip(t, 0.0, 1.0)
    pos = t * (len(_STOPS) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(_STOPS) - 1)
    frac = (pos - lo)[:, None]
    rgb = _STOPS[lo] * (1 - frac) + _STOPS[hi] * frac
    return rgb.astype(np.uint8)


def export_error_heatmap(
    path: str | Path,
    reconstruction: np.ndarray,
    reference: np.ndarray,
    *,
    max_error_mm: float,
) -> None:
    """Colour each reconstructed point by its distance to the reference.

    ``max_error_mm`` is the top of the colour scale — set it to the vertical's
    pass threshold so anything red is out of tolerance at a glance.
    """
    recon = np.asarray(reconstruction, dtype=np.float64)
    tree = cKDTree(np.asarray(reference, dtype=np.float64))
    dist, _ = tree.query(recon, workers=-1)
    colors = _colormap(dist / max_error_mm if max_error_mm > 0 else dist * 0)
    write_ply_colored(path, recon, colors)
