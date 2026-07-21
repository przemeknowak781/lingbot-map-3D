"""COLMAP Structure-from-Motion wrapper (external tool).

This stage recovers camera poses and a sparse point cloud from the input
images. COLMAP is BSD-licensed, so it is safe to depend on in a commercial
product — it is the clean geometric backbone flagged in the decision document.

This module only *drives* a COLMAP binary via subprocess; it does not vendor or
reimplement it. It is intentionally thin: the spike's scientific value is in the
error measurement, not in re-solving SfM.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SfmResult:
    sparse_dir: Path       # COLMAP sparse model (cameras/images/points3D)
    database: Path
    image_dir: Path


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_colmap(
    image_dir: str | Path,
    work_dir: str | Path,
    *,
    colmap_bin: str = "colmap",
    matcher: str = "exhaustive",
    single_camera: bool = True,
) -> SfmResult:
    """Run COLMAP feature extraction, matching and sparse mapping.

    ``matcher`` is ``"exhaustive"`` (best for the tens-to-hundreds of images a
    small-object turntable capture produces) or ``"sequential"`` for video.
    ``single_camera`` shares intrinsics across all frames — correct when every
    photo comes from the same phone/lens, and it materially stabilises the solve.
    """
    if shutil.which(colmap_bin) is None:
        raise FileNotFoundError(
            f"COLMAP binary {colmap_bin!r} not found on PATH. "
            "Install COLMAP (https://colmap.github.io) or pass --colmap-bin."
        )

    image_dir = Path(image_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    _run([
        colmap_bin, "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--ImageReader.camera_model", "OPENCV",
    ])
    _run([colmap_bin, f"{matcher}_matcher", "--database_path", str(database)])
    _run([
        colmap_bin, "mapper",
        "--database_path", str(database),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
    ])
    return SfmResult(sparse_dir=sparse_dir / "0", database=database, image_dir=image_dir)
