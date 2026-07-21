"""Surface-reconstruction backend adapter.

The E1 spike measures *how accurate the best available method actually is on a
real calibrated object*. For the spike we deliberately run whatever method has
the strongest published DTU accuracy — even research/non-commercial code — since
we are only measuring the ceiling, not shipping it. Licensing only matters once
we commit to productising a method (see the decision document, edge E2).

Backends are invoked by subprocess against their own repo/checkout, so this file
stays free of any GPL/Inria-licensed code. Each backend is expected to consume a
COLMAP sparse model and emit a mesh (PLY/OBJ). Register a backend by adding an
entry to ``BACKENDS`` with the command template to run.

Recommended spike order (by published DTU mean Chamfer, lower = better):
    PGSR (0.52 mm) · GausSurf (0.52 mm) · RaDe-GS (0.68) · GOF (0.74) · 2DGS (0.80)
Also run classical COLMAP dense MVS + Poisson as the non-GS baseline — it is the
BSD-clean fallback and the honest yardstick GS must beat to justify itself.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BackendSpec:
    name: str
    commercial_clean: bool     # True only if the method's code is safe to ship
    note: str


# Provenance/licensing metadata, surfaced in the report so the accuracy ceiling
# is never confused with a shippable component.
BACKENDS: dict[str, BackendSpec] = {
    "pgsr":       BackendSpec("PGSR", False, "Inria-derived code; research-only. Accuracy ceiling only."),
    "gaussurf":   BackendSpec("GausSurf", False, "Verify license per-repo before any product use."),
    "rade-gs":    BackendSpec("RaDe-GS", False, "Verify license per-repo."),
    "gof":        BackendSpec("GOF", False, "Confirmed Inria/MPII non-commercial license."),
    "2dgs":       BackendSpec("2D Gaussian Splatting", False, "Verify license per-repo."),
    "colmap-mvs": BackendSpec("COLMAP dense MVS + Poisson", True, "BSD — clean baseline and shippable fallback."),
}


@dataclass
class ReconstructionResult:
    mesh_path: Path
    backend: BackendSpec


def reconstruct(
    backend: str,
    sparse_dir: str | Path,
    image_dir: str | Path,
    out_dir: str | Path,
    *,
    command_template: str | None = None,
) -> ReconstructionResult:
    """Run a reconstruction backend and return the produced mesh path.

    ``command_template`` is a shell template with ``{sparse}``, ``{images}`` and
    ``{out}`` placeholders pointing at the backend's own entry point, e.g.::

        python /repos/PGSR/train.py -s {sparse} --images {images} -m {out}

    We keep the exact command external and user-supplied so no third-party
    training code is vendored into this (otherwise clean) repository.
    """
    spec = BACKENDS.get(backend)
    if spec is None:
        raise ValueError(f"unknown backend {backend!r}; known: {', '.join(BACKENDS)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if command_template is None:
        raise ValueError(
            f"backend {backend!r} needs a --command-template pointing at your "
            f"local {spec.name} checkout (kept external for license hygiene)."
        )

    cmd = command_template.format(sparse=sparse_dir, images=image_dir, out=out_dir)
    print("  $", cmd)
    subprocess.run(cmd, shell=True, check=True)

    meshes = sorted([*out_dir.glob("**/*.ply"), *out_dir.glob("**/*.obj")], key=lambda p: p.stat().st_mtime)
    if not meshes:
        raise FileNotFoundError(f"{spec.name} produced no .ply/.obj mesh under {out_dir}")
    return ReconstructionResult(mesh_path=meshes[-1], backend=spec)
