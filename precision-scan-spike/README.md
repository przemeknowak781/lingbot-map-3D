# E1 — absolute-accuracy validation harness

> **The one question this answers:** on a real, calibrated small object, how far
> — in millimetres — is a surface-aligned Gaussian-Splatting reconstruction from
> ground truth, and does it clear the target vertical's tolerance?

This is the **desktop validation spike** from the project decision document
(edge **E1**). Before investing in a mobile, on-device product, we need one hard
number: does GS reconstruction reach *measurement-class* accuracy on a physical
object — not a relative-scale benchmark like DTU. The whole product thesis rests
on that number, so we measure it first, cheaply, on a workstation.

It is **not** the product. It runs on a desktop GPU (e.g. an RTX 4090), uses
whatever reconstruction method has the best published accuracy (even
research-only code — we're measuring the ceiling, not shipping it), and reports
absolute error against a certified reference.

## Why this and not a DTU number

The research is clear: top surface-aligned GS methods reach ~0.52 mm mean
Chamfer **on the DTU benchmark** — but that is a *relative-scale* metric on a
fixed dataset. Real-world GS geometric error has been reported an order of
magnitude worse. E1 closes that benchmark→reality gap for *our* object, *our*
capture, in real millimetres. A PASS is the green light for E1; a FAIL means the
central differentiator is not defensible and we should stop before building the
app.

## What it measures

| Metric | Meaning |
|---|---|
| **Accuracy** (recon → ref) | how far reconstructed surface is from truth — high = invented geometry / floaters |
| **Completeness** (ref → recon) | how far true surface is from the reconstruction — high = missed surface |
| **Chamfer** | symmetric mean of the two — the headline number |
| **within-tolerance %** | fraction of surface inside 0.05 / 0.1 / 0.25 / 0.5 / 1.0 mm |
| **form residual** | (gauge-sphere scale mode) reference-free roundness error |

Plus a per-point **error heatmap** PLY (red = out of tolerance) and a go/no-go
**verdict** against your vertical's threshold.

## The methodology that makes the number honest

Two alignment modes, and the difference is the whole point:

- **`rigid`** (6-DoF) — scale fixed at 1.0. Measures **true absolute error**,
  including scale error. This is the real E1 verdict. Only valid when the
  reconstruction is in metric units (see scale below).
- **`similarity`** (7-DoF) — scale fitted away. Measures **shape only**; it
  *hides* scale error. A diagnostic, never the verdict. Reports say so loudly.

Because monocular SfM/GS is only defined up to scale, `rigid` mode needs a
**metric anchor**:

- `bar` — two endpoints of a scale bar of known length, or
- `standard` — a scanned gauge sphere of certified diameter (also yields a
  reference-free form error).

Recovering scale by similarity-aligning to the reference is deliberately **not**
supported — it would fold scale error into the alignment and make the absolute
number a tautology.

## Install

```bash
cd precision-scan-spike
pip install -r requirements.txt        # numpy + scipy is enough for the core
# optional: pip install trimesh open3d pyyaml   # more formats + YAML configs
```

External tools, invoked by subprocess (kept out of this repo for license
hygiene — see the decision doc, edge E2):

- **COLMAP** (BSD) — SfM poses. `https://colmap.github.io`
- A **reconstruction backend** checkout (PGSR / GausSurf / 2DGS / …) *or*
  COLMAP dense MVS for the BSD-clean baseline.

## GUI (Windows-friendly)

A local, instrument-style control panel — no cloud, standard-library server, no
extra pip installs:

```bash
python gui/server.py          # opens http://localhost:8000 in your browser
```

On Windows just double-click **`run_gui.bat`**. The GUI lets you pick the
images/reference with **native OS file dialogs**, choose a backend, run, watch
the **live log**, read the **PASS/FAIL verdict** and metrics, and orbit the
**3D error heatmap** (reconstruction coloured by distance to reference — red is
out of tolerance). Tip: point *Re-score existing mesh* at a `.ply` to skip
COLMAP entirely and just measure an already-produced mesh.

> The 3D viewer loads three.js from a CDN on first use (needs internet once);
> everything else runs fully offline. The verdict/metrics work even if the
> viewer can't load.

## Run (CLI)

```bash
cp config.example.yaml config.yaml     # edit paths, backend, scale, threshold
python -m e1_harness.cli --config config.yaml
```

Re-score an existing mesh without re-running COLMAP + the backend:

```bash
python -m e1_harness.cli --config config.yaml --reuse-mesh path/to/mesh.ply
```

Exit code is `0` on PASS, `1` on FAIL — wire it straight into CI. Outputs land
in `work_dir/`: `e1_report.json`, `e1_report.md`, `error_heatmap.ply`.

## Pipeline

```
images ─▶ COLMAP SfM ─▶ backend (GS/MVS) ─▶ mesh
                                              │
reference mesh (mm) ──────────────┐           ▼
                                  │      area-uniform sampling
                          metric scale (bar / gauge sphere)
                                  ▼
                        rigid / similarity align
                                  ▼
              accuracy · completeness · Chamfer · heatmap · VERDICT
```

`sfm.py` and `reconstruct.py` are thin subprocess adapters; the scientific value
lives in `align.py`, `metrics.py`, `scale.py`, `geometry.py`, which are pure
numpy/scipy and fully unit-tested.

## Tests

```bash
python tests/test_core.py     # or: pytest tests/
```

Covers exact recovery of a known similarity transform, ICP registration,
zero-error and known-offset metrics, pass/fail verdict logic, scale-bar and
gauge-sphere scale recovery, area-weighted surface sampling, and PLY round-trip
— all without a GPU or external tools.

## Interpreting the verdict

- **PASS** at your vertical's threshold (e.g. dental ≤0.1 mm, reverse-eng
  ≤0.25–0.5 mm) → E1 is defensible for that vertical; proceed toward the product.
- **FAIL** → check the heatmap. Systematic (whole surface off) points at a scale
  or pose problem; localized red points at floaters or missed detail. If it
  fails even at the accuracy ceiling with a top backend, the differentiator does
  not hold — stop, per the decision document's kill criteria.

## Layout

```
src/e1_harness/
  align.py        rigid/similarity Umeyama + ICP
  metrics.py      accuracy / completeness / Chamfer / verdict
  scale.py        metric scale from bar or gauge sphere
  geometry.py     PLY/OBJ IO + area-uniform surface sampling
  heatmap.py      per-point error → coloured PLY
  sfm.py          COLMAP wrapper (subprocess)
  reconstruct.py  backend adapter + license metadata (subprocess)
  report.py       JSON + Markdown report
  pipeline.py     orchestration
  cli.py          `python -m e1_harness.cli`
gui/
  server.py       stdlib local server + native file pickers
  index.html      instrument-style UI + 3D heatmap viewer (three.js)
run_gui.bat       Windows double-click launcher
tests/test_core.py
scripts/capture_guide.md
```

This directory is self-contained and carries no dependency on the surrounding
repository — it is meant to be lifted into its own project once the E1 result is in.
