"""End-to-end E1 pipeline: images → reconstruction → absolute error → verdict."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import align, metrics, scale as scale_mod
from .config import Config
from .geometry import Mesh, load_and_sample, load_mesh, sample_surface
from .heatmap import export_error_heatmap
from .reconstruct import BACKENDS, reconstruct
from .report import build_report, write_reports
from .sfm import run_colmap


def _resolve_scale(cfg: Config, recon_points: np.ndarray) -> scale_mod.ScaleResult | None:
    s = cfg.scale
    if s.method == "none":
        return None
    if s.method == "bar":
        if not (s.endpoint_a and s.endpoint_b and s.true_length_mm):
            raise ValueError("scale.method 'bar' needs endpoint_a, endpoint_b, true_length_mm")
        return scale_mod.scale_from_bar(s.endpoint_a, s.endpoint_b, s.true_length_mm)
    if s.method == "standard":
        if not (s.standard_points_path and s.true_diameter_mm):
            raise ValueError("scale.method 'standard' needs standard_points_path, true_diameter_mm")
        pts = load_mesh(s.standard_points_path).vertices
        return scale_mod.scale_from_standard(pts, s.true_diameter_mm)
    raise ValueError(f"unknown scale.method {s.method!r}")


def run(cfg: Config, *, skip_reconstruction: Path | None = None) -> dict:
    """Run the full pipeline and return the report dict.

    ``skip_reconstruction`` lets you point straight at an already-produced mesh
    (e.g. re-scoring a backend output) and skip the COLMAP + GS stages.
    """
    if not cfg.reference_mesh:
        raise ValueError("reference_mesh is required — the certified ground-truth surface (mm).")
    if skip_reconstruction is None and not cfg.image_dir:
        raise ValueError("image_dir is required unless you re-score an existing mesh.")

    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # 1–2. SfM + reconstruction (external tools) — or reuse an existing mesh.
    if skip_reconstruction is not None:
        mesh_path = skip_reconstruction
        backend_spec = BACKENDS[cfg.backend]
    else:
        print("[1/5] COLMAP SfM …")
        sfm = run_colmap(cfg.image_dir, work / "sfm", colmap_bin=cfg.colmap_bin, matcher=cfg.matcher)
        print(f"[2/5] Reconstruction backend: {cfg.backend} …")
        result = reconstruct(cfg.backend, sfm.sparse_dir, sfm.image_dir, work / "recon",
                             command_template=cfg.command_template)
        mesh_path, backend_spec = result.mesh_path, result.backend

    # 3. Sample both surfaces uniformly by area.
    print("[3/5] Sampling surfaces …")
    recon_mesh: Mesh = load_mesh(mesh_path)
    recon = sample_surface(recon_mesh, cfg.n_samples, seed=cfg.seed)
    reference = load_and_sample(cfg.reference_mesh, cfg.n_samples, seed=cfg.seed)

    # 4. Metric scale, then align.
    print("[4/5] Scaling + aligning …")
    scale_result = _resolve_scale(cfg, recon)
    if scale_result is not None:
        recon = recon * scale_result.factor
    if cfg.align_mode == "rigid" and scale_result is None and cfg.scale.method == "none":
        print("  ! rigid mode with no metric scale — result is only valid if the "
              "backend already outputs metric units.")

    transform = align.align(recon, reference, mode=cfg.align_mode)
    recon_aligned = transform.apply(recon)

    # 5. Error, heatmap, verdict, reports.
    print("[5/5] Measuring error …")
    error = metrics.compute_error(recon_aligned, reference,
                                  tolerances=cfg.tolerances_mm, unit="mm")
    verdict = metrics.evaluate_verdict(error, threshold_mm=cfg.threshold_mm,
                                       metric=cfg.verdict_metric)
    export_error_heatmap(work / "error_heatmap.ply", recon_aligned, reference,
                         max_error_mm=cfg.threshold_mm)

    report = build_report(backend=backend_spec, align_mode=cfg.align_mode,
                          scale=scale_result, error=error, verdict=verdict,
                          n_samples=cfg.n_samples)
    json_path, md_path = write_reports(report, work)
    print(f"\n  → {json_path}\n  → {md_path}\n  → {work / 'error_heatmap.ply'}")
    print(f"\n  VERDICT: {'PASS ✅' if verdict.passed else 'FAIL ❌'} — {verdict.reason}")
    return report
