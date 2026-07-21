"""Command-line entry point for the E1 validation harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="e1",
        description="E1 — measure the ABSOLUTE accuracy of a GS reconstruction "
                    "against a calibrated reference object.",
    )
    parser.add_argument("--config", required=True, help="Path to a .yaml/.json run config")
    parser.add_argument("--reuse-mesh", default=None,
                        help="Skip COLMAP+backend; score this existing mesh instead")
    parser.add_argument("--threshold-mm", type=float, default=None,
                        help="Override the pass/fail threshold from the config")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    if args.threshold_mm is not None:
        cfg.threshold_mm = args.threshold_mm

    report = run(cfg, skip_reconstruction=Path(args.reuse_mesh) if args.reuse_mesh else None)
    return 0 if report["verdict"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
