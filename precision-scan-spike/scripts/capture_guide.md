# Capture guide — calibrated small-object scan for E1

The E1 verdict is only as trustworthy as the capture. The goal is a photo set
that lets COLMAP solve poses cleanly and a **certified reference** to measure
against in real millimetres.

## What you need in frame
- **The object**, filling most of the frame (small objects need macro/close focus).
- **A metric anchor** — one of:
  - a **gauge sphere / ball-bar** of certified diameter (best: also gives a
    reference-free form error), or
  - a **scale bar** / calibrated ruler of known length.
- **Texture**: matte, feature-rich surfaces solve best. Matting spray on shiny
  or transparent parts is standard practice — untextured/specular surfaces are
  the #1 cause of a failed solve.

## Shooting
- 60–200 photos, orbiting the object at 2–3 elevations, ~10–15° between shots.
- Keep the object still; move the camera (turntable is fine if the background is
  masked, otherwise the background must move *with* the object — don't mix).
- Lock exposure and focus. Diffuse, even lighting; avoid moving shadows.
- Sharp frames only — motion blur poisons SfM.

## Ground truth
The `reference_mesh` must be an independent, higher-accuracy measurement of the
**same physical object**, in millimetres — e.g. a structured-light/industrial
scan or a CAD model for a machined part. Without a trustworthy reference there is
no absolute error to report, only a shape-consistency diagnostic.

## Then
```
cp config.example.yaml config.yaml   # edit paths, scale, threshold
python -m e1_harness.cli --config config.yaml
```
Open `runs/<name>/error_heatmap.ply` in MeshLab/CloudCompare: red = out of
tolerance. Read the verdict in `e1_report.md`.
