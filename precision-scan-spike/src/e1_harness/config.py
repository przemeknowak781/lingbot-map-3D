"""Run configuration for the E1 validation harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScaleConfig:
    # One of: "bar", "standard", "none".
    #   bar      -> needs endpoint_a, endpoint_b (recon coords) and true_length_mm
    #   standard -> needs true_diameter_mm and standard_points_path (a PLY of the
    #               scanned gauge-sphere region)
    #   none     -> reconstruction is assumed already metric (a metric backend)
    method: str = "none"
    endpoint_a: list[float] | None = None
    endpoint_b: list[float] | None = None
    true_length_mm: float | None = None
    true_diameter_mm: float | None = None
    standard_points_path: str | None = None


@dataclass
class Config:
    # Inputs
    image_dir: str
    reference_mesh: str            # certified ground-truth surface, in mm
    work_dir: str = "runs/e1"

    # Reconstruction backend (see reconstruct.BACKENDS)
    backend: str = "colmap-mvs"
    command_template: str | None = None
    colmap_bin: str = "colmap"
    matcher: str = "exhaustive"

    # Evaluation
    #   mode "rigid"      -> ABSOLUTE error (requires metric scale; the real E1 test)
    #   mode "similarity" -> shape-only error (scale free; diagnostic, NOT the verdict)
    align_mode: str = "rigid"
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    n_samples: int = 500_000
    tolerances_mm: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)

    # Verdict
    verdict_metric: str = "chamfer_median"
    threshold_mm: float = 0.1      # default: dental-grade sub-0.1 mm target
    seed: int = 0

    @staticmethod
    def load(path: str | Path) -> "Config":
        data = _read_structured(Path(path))
        scale = ScaleConfig(**data.pop("scale", {}))
        if "tolerances_mm" in data:
            data["tolerances_mm"] = tuple(data["tolerances_mm"])
        return Config(scale=scale, **data)


def _read_structured(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # optional dependency

            return yaml.safe_load(text)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyYAML not installed — either `pip install pyyaml` or use a .json config."
            ) from exc
    return json.loads(text)
