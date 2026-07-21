"""Recover metric scale for an up-to-scale reconstruction.

Monocular SfM/GS reconstructions are only defined up to a global scale, so an
*absolute* error measurement is impossible until that scale is fixed. There are
two honest ways to do it, both supported here:

1. **Scale bar** — place an object of certified length in the scene, mark its
   two endpoints in the reconstruction, and rescale so their distance matches
   the true length. This fixes scale independently of the reference, so the
   subsequent rigid alignment measures a *genuine* absolute error.

2. **Calibration standard** — scan a certified artefact (e.g. a ceramic
   ball-bar or a gauge sphere of known diameter). Fitting the sphere gives both
   the metric scale and a direct form/roundness error you can report on its own.

Do NOT recover scale by similarity-aligning to the reference and reading off the
Umeyama scale factor — that folds scale error into the alignment and makes the
absolute number a tautology. That path is deliberately not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScaleResult:
    factor: float               # multiply reconstruction coords by this
    measured: float             # measured quantity before scaling (recon units)
    reference: float            # certified quantity (metric units)
    residual: float | None = None  # fit residual for the standard, if applicable


def scale_from_bar(
    endpoint_a: np.ndarray,
    endpoint_b: np.ndarray,
    true_length: float,
) -> ScaleResult:
    """Metric scale from two marked endpoints of a scale bar of known length."""
    a = np.asarray(endpoint_a, dtype=np.float64)
    b = np.asarray(endpoint_b, dtype=np.float64)
    measured = float(np.linalg.norm(a - b))
    if measured <= 0:
        raise ValueError("scale-bar endpoints are coincident")
    if true_length <= 0:
        raise ValueError("true_length must be positive")
    return ScaleResult(factor=true_length / measured, measured=measured, reference=true_length)


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares sphere fit. Returns (centre, radius, rms_residual).

    Uses the standard linear formulation: for each point
    ``x^2+y^2+z^2 = 2 c·x + (r^2 - |c|^2)``, solved for ``[cx, cy, cz, g]``.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 4:
        raise ValueError("need at least 4 (N, 3) points to fit a sphere")

    a = np.hstack([2.0 * pts, np.ones((pts.shape[0], 1))])
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    centre = sol[:3]
    radius = float(np.sqrt(sol[3] + centre @ centre))

    residuals = np.linalg.norm(pts - centre, axis=1) - radius
    rms = float(np.sqrt((residuals ** 2).mean()))
    return centre, radius, rms


def scale_from_standard(points: np.ndarray, true_diameter: float) -> ScaleResult:
    """Metric scale from a scanned gauge sphere of certified diameter.

    The fit residual (in reconstruction units, before scaling) is a direct,
    reference-free measure of the scanner's form error on a known-perfect
    surface — worth reporting on its own.
    """
    _, radius, rms = fit_sphere(points)
    measured_diameter = 2.0 * radius
    if measured_diameter <= 0:
        raise ValueError("degenerate sphere fit")
    return ScaleResult(
        factor=true_diameter / measured_diameter,
        measured=measured_diameter,
        reference=true_diameter,
        residual=rms,
    )
