"""Rigid and similarity registration of a reconstruction onto a reference.

The choice of alignment mode is the single most important methodological
decision in the whole E1 spike, so it lives in its own module with a loud
docstring:

- ``similarity`` (7-DoF: rotation + translation + *scale*) makes the two shapes
  the same size before measuring, so the residual reflects **shape fidelity
  only**. It *hides* absolute-scale error. Use it to answer "is the geometry
  the right shape?" — NOT "is it metrically correct?".

- ``rigid`` (6-DoF: rotation + translation, scale fixed at 1.0) measures the
  **true absolute error**, including any scale error. This is the rigorous E1
  test, and it is only meaningful when the reconstruction is already in metric
  units (see ``scale.py``). If you feed it an up-to-scale reconstruction the
  numbers will be meaningless — that is by design, not a bug.

Both estimators are closed-form (Umeyama, 1991) and are optionally refined with
a point-to-point ICP pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Transform:
    """A similarity transform ``y = s * R @ x + t``."""

    scale: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation


def umeyama(
    source: np.ndarray,
    target: np.ndarray,
    *,
    with_scale: bool,
) -> Transform:
    """Least-squares similarity transform mapping ``source`` onto ``target``.

    ``source`` and ``target`` are (N, 3) arrays of *corresponding* points. When
    ``with_scale`` is False the scale is fixed to 1.0 (rigid alignment).

    Implements Umeyama (1991), "Least-squares estimation of transformation
    parameters between two point patterns", including the reflection-avoiding
    sign correction.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be matching (N, 3) arrays")

    n = source.shape[0]
    mu_src = source.mean(axis=0)
    mu_dst = target.mean(axis=0)
    src_c = source - mu_src
    dst_c = target - mu_dst

    # Cross-covariance and its SVD.
    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)

    # Correct for a possible reflection so R is a proper rotation.
    s = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1] = -1.0
    rotation = u @ np.diag(s) @ vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        scale = float((d * s).sum() / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0

    translation = mu_dst - scale * (rotation @ mu_src)
    return Transform(scale=scale, rotation=rotation, translation=translation)


def icp_refine(
    source: np.ndarray,
    target: np.ndarray,
    init: Transform,
    *,
    with_scale: bool,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
    max_pair_distance: float | None = None,
) -> Transform:
    """Point-to-point ICP starting from ``init``.

    Reconstruction and reference are generally *not* in correspondence, so we
    alternate nearest-neighbour matching with an Umeyama solve. ``with_scale``
    must match the alignment mode used for ``init`` — never let ICP silently
    re-introduce a scale degree of freedom into a rigid (absolute) evaluation.
    """
    source = np.asarray(source, dtype=np.float64)
    tree = cKDTree(np.asarray(target, dtype=np.float64))

    transform = init
    prev_rmse = np.inf
    for _ in range(max_iterations):
        moved = transform.apply(source)
        dist, idx = tree.query(moved, workers=-1)

        mask = slice(None)
        if max_pair_distance is not None:
            mask = dist <= max_pair_distance
            if mask.sum() < 3:
                break  # too few inliers to solve a stable transform

        transform = umeyama(source[mask], np.asarray(target)[idx[mask]], with_scale=with_scale)

        rmse = float(np.sqrt((dist[mask] ** 2).mean()))
        if abs(prev_rmse - rmse) < tolerance:
            break
        prev_rmse = rmse

    return transform


def align(
    source: np.ndarray,
    target: np.ndarray,
    *,
    mode: str,
    refine_icp: bool = True,
    max_pair_distance: float | None = None,
) -> Transform:
    """Align ``source`` to ``target`` under the given mode.

    ``mode`` is ``"rigid"`` (absolute error; scale fixed) or ``"similarity"``
    (shape-only; scale free). Alignment is initialised with a coarse Umeyama
    fit on nearest-neighbour correspondences, then optionally ICP-refined.
    """
    if mode not in ("rigid", "similarity"):
        raise ValueError(f"mode must be 'rigid' or 'similarity', got {mode!r}")
    with_scale = mode == "similarity"
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # Coarse init: align centroids (and, in similarity mode, gross scale via RMS
    # radius) so the nearest-neighbour correspondence search starts from roughly
    # overlapping clouds instead of garbage matches. Rotation is left to ICP.
    mu_src, mu_dst = source.mean(axis=0), target.mean(axis=0)
    rms_src = np.sqrt(((source - mu_src) ** 2).sum(axis=1).mean())
    rms_dst = np.sqrt(((target - mu_dst) ** 2).sum(axis=1).mean())
    s0 = float(rms_dst / rms_src) if (with_scale and rms_src > 0) else 1.0
    init = Transform(scale=s0, rotation=np.eye(3), translation=mu_dst - s0 * mu_src)

    if not refine_icp:
        return init
    return icp_refine(
        source, target, init,
        with_scale=with_scale, max_pair_distance=max_pair_distance,
    )
