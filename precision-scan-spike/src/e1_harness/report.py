"""Assemble the machine-readable and human-readable run reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .metrics import ErrorReport, Verdict
from .reconstruct import BackendSpec
from .scale import ScaleResult


def build_report(
    *,
    backend: BackendSpec,
    align_mode: str,
    scale: ScaleResult | None,
    error: ErrorReport,
    verdict: Verdict,
    n_samples: int,
) -> dict:
    report = {
        "verdict": {
            "passed": verdict.passed,
            "metric": verdict.metric,
            "value_mm": round(verdict.value_mm, 5),
            "threshold_mm": verdict.threshold_mm,
            "reason": verdict.reason,
        },
        "align_mode": align_mode,
        "absolute": align_mode == "rigid",
        "backend": {
            "name": backend.name,
            "commercial_clean": backend.commercial_clean,
            "note": backend.note,
        },
        "scale": None if scale is None else {
            "factor": round(scale.factor, 8),
            "measured": round(scale.measured, 6),
            "reference": scale.reference,
            "form_residual_mm": None if scale.residual is None else round(scale.residual, 5),
        },
        "error_mm": {
            "chamfer_mean": round(error.chamfer_mean, 5),
            "chamfer_median": round(error.chamfer_median, 5),
            "accuracy": _dir(error.accuracy),
            "completeness": _dir(error.completeness),
        },
        "n_samples": n_samples,
    }
    return report


def _dir(d) -> dict:
    out = {k: round(v, 5) for k, v in asdict(d).items() if k != "within"}
    out["within_pct"] = {f"{k:g}mm": round(100 * v, 2) for k, v in d.within.items()}
    return out


def write_reports(report: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "e1_report.json"
    md_path = out_dir / "e1_report.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_render_markdown(report))
    return json_path, md_path


def _render_markdown(r: dict) -> str:
    v = r["verdict"]
    badge = "✅ PASS" if v["passed"] else "❌ FAIL"
    warn = "" if r["absolute"] else (
        "\n> ⚠️ **similarity mode** — scale was fitted away, so this measures "
        "*shape only*, not absolute accuracy. Not a valid E1 verdict.\n"
    )
    b = r["backend"]
    clean = "clean ✅" if b["commercial_clean"] else "research-only ⚠️"
    acc, comp = r["error_mm"]["accuracy"], r["error_mm"]["completeness"]
    lines = [
        "# E1 — absolute accuracy validation",
        "",
        f"## {badge}  ·  {v['metric']} = {v['value_mm']} mm  (threshold {v['threshold_mm']} mm)",
        warn,
        f"- **Backend:** {b['name']} — {clean} ({b['note']})",
        f"- **Alignment:** {r['align_mode']} ({'absolute' if r['absolute'] else 'shape-only'})",
        f"- **Samples:** {r['n_samples']:,} per surface",
    ]
    if r["scale"]:
        s = r["scale"]
        lines.append(f"- **Metric scale:** ×{s['factor']} (measured {s['measured']} → {s['reference']} mm)")
        if s["form_residual_mm"] is not None:
            lines.append(f"- **Gauge form residual:** {s['form_residual_mm']} mm (reference-free)")
    lines += [
        "",
        "| metric | mean | median | p95 | p99 | max |",
        "|---|--:|--:|--:|--:|--:|",
        f"| accuracy (recon→ref) | {acc['mean']} | {acc['median']} | {acc['p95']} | {acc['p99']} | {acc['max']} |",
        f"| completeness (ref→recon) | {comp['mean']} | {comp['median']} | {comp['p95']} | {comp['p99']} | {comp['max']} |",
        "",
        "**Surface within tolerance (accuracy):** "
        + ", ".join(f"{k} → {p}%" for k, p in acc["within_pct"].items()),
        "",
    ]
    return "\n".join(lines)
