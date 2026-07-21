"""Absolute error metrics between a reconstruction and a reference surface.

Everything here is in the *metric units of the reference* (millimetres, if the
reference is in millimetres). The metrics deliberately mirror the DTU
accuracy/completeness convention so the numbers are comparable with the GS
papers — the difference is that we report them as *absolute* distances on a
real calibrated object, which is precisely the benchmark→reality gap the E1
spike exists to measure.

- **Accuracy**: how far each reconstructed point is from the reference. High
  accuracy error = the scanner invented surface that is not there.
- **Completeness**: how far each reference point is from the reconstruction.
  High completeness error = the scanner missed real surface.
- **Chamfer**: the symmetric mean of the two — the single headline number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class DirectionalError:
    """One-directional nearest-neighbour distance statistics (metric units)."""

    mean: float
    median: float
    rms: float
    p90: float
    p95: float
    p99: float
    max: float
    # Fraction of points within a tolerance, keyed by the tolerance in mm.
    within: dict[float, float] = field(default_factory=dict)


@dataclass
class ErrorReport:
    accuracy: DirectionalError       # reconstruction -> reference
    completeness: DirectionalError   # reference -> reconstruction
    chamfer_mean: float              # 0.5 * (acc.mean + comp.mean)
    chamfer_median: float
    unit: str = "mm"


def _nn_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distance from every ``query`` point to ``reference``."""
    tree = cKDTree(np.asarray(reference, dtype=np.float64))
    dist, _ = tree.query(np.asarray(query, dtype=np.float64), workers=-1)
    return dist


def _summarise(dist: np.ndarray, tolerances: tuple[float, ...]) -> DirectionalError:
    dist = np.asarray(dist, dtype=np.float64)
    return DirectionalError(
        mean=float(dist.mean()),
        median=float(np.median(dist)),
        rms=float(np.sqrt((dist ** 2).mean())),
        p90=float(np.percentile(dist, 90)),
        p95=float(np.percentile(dist, 95)),
        p99=float(np.percentile(dist, 99)),
        max=float(dist.max()),
        within={float(t): float((dist <= t).mean()) for t in tolerances},
    )


def compute_error(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    *,
    tolerances: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0),
    unit: str = "mm",
) -> ErrorReport:
    """Compute accuracy, completeness and Chamfer error.

    Both point sets must already be **aligned and in metric units**. Points are
    typically dense uniform samples of each surface (see ``geometry.sample_surface``).
    ``tolerances`` are the pass thresholds reported as "% of surface within X mm".
    """
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if reconstruction.ndim != 2 or reconstruction.shape[1] != 3:
        raise ValueError("reconstruction must be an (N, 3) array")
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference must be an (M, 3) array")

    acc = _summarise(_nn_distances(reconstruction, reference), tolerances)
    comp = _summarise(_nn_distances(reference, reconstruction), tolerances)
    return ErrorReport(
        accuracy=acc,
        completeness=comp,
        chamfer_mean=0.5 * (acc.mean + comp.mean),
        chamfer_median=0.5 * (acc.median + comp.median),
        unit=unit,
    )


@dataclass
class Verdict:
    passed: bool
    threshold_mm: float
    metric: str
    value_mm: float
    reason: str


def evaluate_verdict(
    report: ErrorReport,
    *,
    threshold_mm: float,
    metric: str = "chamfer_median",
) -> Verdict:
    """Turn the error report into a go/no-go verdict for the target vertical.

    ``metric`` selects the number compared against ``threshold_mm``. The default
    is the median Chamfer distance — robust to a handful of outlier floaters,
    which is the honest choice for "is the bulk of the surface trustworthy?".
    Use ``"accuracy.p95"`` style dotted paths for stricter, tail-sensitive
    verticals (e.g. dental, where a single 1 mm spike can ruin a fit).
    """
    value = _resolve_metric(report, metric)
    passed = value <= threshold_mm
    reason = (
        f"{metric} = {value:.4f} {report.unit} "
        f"{'≤' if passed else '>'} threshold {threshold_mm:.4f} {report.unit}"
    )
    return Verdict(
        passed=passed,
        threshold_mm=threshold_mm,
        metric=metric,
        value_mm=value,
        reason=reason,
    )


def _resolve_metric(report: ErrorReport, metric: str) -> float:
    """Resolve a dotted metric path like ``accuracy.p95`` or ``chamfer_median``."""
    obj: object = report
    for part in metric.split("."):
        if isinstance(obj, dict):
            obj = obj[float(part)]
        else:
            obj = getattr(obj, part)
    if not isinstance(obj, (int, float)):
        raise ValueError(f"metric path {metric!r} did not resolve to a number")
    return float(obj)
