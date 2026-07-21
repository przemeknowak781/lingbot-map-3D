"""Unit tests for the load-bearing math: alignment, metrics, scale, geometry.

These run without a GPU, COLMAP or any GS backend — they validate the parts of
the harness that decide the E1 verdict. Run with ``pytest`` or directly:
``python tests/test_core.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e1_harness import align, geometry, metrics, scale as scale_mod  # noqa: E402


def _rotation(ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


# --------------------------------------------------------------------------- #
def test_umeyama_recovers_similarity_exactly():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(200, 3))
    R = _rotation(0.3, -0.7, 1.1)
    s, t = 2.5, np.array([4.0, -1.0, 0.5])
    dst = s * (src @ R.T) + t

    tf = align.umeyama(src, dst, with_scale=True)
    assert abs(tf.scale - s) < 1e-9, tf.scale
    assert np.allclose(tf.rotation, R, atol=1e-9)
    assert np.allclose(tf.apply(src), dst, atol=1e-8)


def test_umeyama_rigid_ignores_scale():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(200, 3))
    R = _rotation(0.1, 0.2, -0.3)
    dst = 3.0 * (src @ R.T) + 1.0  # scaled, but we fit rigidly

    tf = align.umeyama(src, dst, with_scale=False)
    assert tf.scale == 1.0
    assert np.allclose(tf.rotation, R, atol=1e-9)  # rotation still recovered


def test_align_icp_registers_shape():
    # A structurally asymmetric surface z = f(x, y) over an asymmetric domain, so
    # rotation is well constrained (unlike an isotropic blob). This mirrors the
    # real case: reconstruction and reference are roughly pre-aligned and, after
    # metric scaling, at the same scale — ICP only needs to polish the pose.
    rng = np.random.default_rng(3)
    x = rng.uniform(-3, 6, 4000)
    y = rng.uniform(-2, 2, 4000)
    z = 0.4 * np.sin(x) + 0.3 * np.cos(1.7 * y) + 0.05 * x
    ref = np.column_stack([x, y, z])

    R = _rotation(0.08, 0.12, -0.1)               # modest misalignment
    moved = 1.3 * (ref @ R.T) + np.array([1.0, -0.5, 0.7])

    tf = align.align(moved, ref, mode="similarity")
    err = metrics.compute_error(tf.apply(moved), ref)
    assert err.chamfer_mean < 1e-3, err.chamfer_mean


# --------------------------------------------------------------------------- #
def test_metrics_zero_on_identical():
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(500, 3))
    err = metrics.compute_error(pts, pts)
    assert err.chamfer_mean == 0.0
    assert err.accuracy.within[0.05] == 1.0


def test_metrics_known_offset():
    # Two parallel planes 0.2 mm apart → every nearest distance is exactly 0.2.
    rng = np.random.default_rng(5)
    grid = np.column_stack([rng.uniform(-5, 5, 4000), rng.uniform(-5, 5, 4000), np.zeros(4000)])
    shifted = grid + np.array([0.0, 0.0, 0.2])
    err = metrics.compute_error(shifted, grid)
    assert abs(err.accuracy.mean - 0.2) < 1e-6, err.accuracy.mean
    assert abs(err.accuracy.p95 - 0.2) < 1e-6
    # 0.2 mm error is within the 0.25 mm tolerance but not within 0.1 mm.
    assert err.accuracy.within[0.25] == 1.0
    assert err.accuracy.within[0.1] == 0.0


def test_verdict_pass_fail():
    rng = np.random.default_rng(6)
    grid = np.column_stack([rng.uniform(-5, 5, 2000), rng.uniform(-5, 5, 2000), np.zeros(2000)])
    err = metrics.compute_error(grid + [0, 0, 0.08], grid)
    assert metrics.evaluate_verdict(err, threshold_mm=0.1).passed
    assert not metrics.evaluate_verdict(err, threshold_mm=0.05).passed


# --------------------------------------------------------------------------- #
def test_scale_from_bar():
    a, b = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 2.0])  # 2 recon-units
    res = scale_mod.scale_from_bar(a, b, true_length=50.0)        # = 50 mm
    assert abs(res.factor - 25.0) < 1e-12


def test_sphere_fit_and_standard_scale():
    rng = np.random.default_rng(7)
    centre, radius = np.array([1.0, -2.0, 0.5]), 0.4
    dirs = rng.normal(size=(3000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = centre + radius * dirs
    c_hat, r_hat, rms = scale_mod.fit_sphere(pts)
    assert np.allclose(c_hat, centre, atol=1e-6)
    assert abs(r_hat - radius) < 1e-6
    assert rms < 1e-6
    # Certified diameter 25.4 mm; measured diameter 0.8 recon-units → ×31.75.
    res = scale_mod.scale_from_standard(pts, true_diameter=25.4)
    assert abs(res.factor - 25.4 / 0.8) < 1e-4


# --------------------------------------------------------------------------- #
def test_surface_sampling_stays_on_plane():
    # Unit square in z=0, two triangles.
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = geometry.Mesh(vertices=verts, faces=faces)
    pts = geometry.sample_surface(mesh, 5000, seed=0)
    assert np.allclose(pts[:, 2], 0.0)
    assert pts[:, 0].min() >= -1e-9 and pts[:, 0].max() <= 1 + 1e-9
    assert len(pts) == 5000


def test_ply_roundtrip(tmp_path=None):
    import tempfile

    d = Path(tempfile.mkdtemp())
    pts = np.random.default_rng(8).normal(size=(100, 3))
    colors = np.zeros((100, 3), dtype=np.uint8)
    path = d / "cloud.ply"
    geometry.write_ply_colored(path, pts, colors)
    loaded = geometry.load_mesh(path)
    assert loaded.vertices.shape == (100, 3)
    assert np.allclose(loaded.vertices, pts, atol=1e-5)


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
