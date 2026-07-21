"""Mesh / point-cloud IO and surface sampling.

Chamfer-style metrics are only fair when both surfaces are sampled *uniformly by
area* — otherwise a mesh with many tiny triangles in one region biases the
distance. So we always sample meshes to dense point clouds before measuring.

If ``trimesh`` or ``open3d`` are installed we use them (they read every format
under the sun); otherwise a small built-in reader covers ASCII / little-endian
binary PLY and OBJ, which is enough for COLMAP and most GS exporters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray            # (N, 3)
    faces: np.ndarray | None = None  # (M, 3) int, or None for a pure point cloud


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_mesh(path: str | Path) -> Mesh:
    """Load a mesh or point cloud from PLY / OBJ (and anything trimesh handles)."""
    path = Path(path)
    try:  # Prefer a real library when available.
        import trimesh

        loaded = trimesh.load(path, process=False, force="mesh")
        faces = np.asarray(loaded.faces, dtype=np.int64) if getattr(loaded, "faces", None) is not None and len(loaded.faces) else None
        return Mesh(vertices=np.asarray(loaded.vertices, dtype=np.float64), faces=faces)
    except Exception:
        pass

    suffix = path.suffix.lower()
    if suffix == ".ply":
        return _load_ply(path)
    if suffix == ".obj":
        return _load_obj(path)
    raise ValueError(f"unsupported mesh format {suffix!r}; install trimesh for more")


def _load_obj(path: Path) -> Mesh:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            _, x, y, z, *_ = line.split()
            verts.append((float(x), float(y), float(z)))
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            for i in range(1, len(idx) - 1):  # fan-triangulate polygons
                faces.append((idx[0], idx[i], idx[i + 1]))
    return Mesh(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64) if faces else None,
    )


_PLY_DTYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _load_ply(path: Path) -> Mesh:
    with open(path, "rb") as fh:
        raw = fh.read()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii", errors="replace")
    body = raw[header_end:]

    fmt = "ascii"
    elements: list[tuple[str, int, list[tuple]]] = []
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property" and elements:
            elements[-1][2].append(tuple(parts[1:]))

    if fmt == "ascii":
        return _parse_ply_ascii(body.decode("ascii", errors="replace"), elements)
    if fmt == "binary_little_endian":
        return _parse_ply_binary(body, elements, endian="<")
    raise ValueError(f"unsupported PLY format {fmt!r}")


def _parse_ply_ascii(body: str, elements) -> Mesh:
    tokens = body.split()
    pos = 0
    vertices: np.ndarray | None = None
    faces: list[list[int]] = []
    for name, count, props in elements:
        if name == "vertex":
            names = [p[-1] for p in props]
            xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
            width = len(props)
            block = np.array(tokens[pos:pos + count * width], dtype=np.float64).reshape(count, width)
            vertices = block[:, [xi, yi, zi]]
            pos += count * width
        else:
            for _ in range(count):
                n = int(tokens[pos]); pos += 1
                idx = [int(t) for t in tokens[pos:pos + n]]; pos += n
                for i in range(1, n - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return Mesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64) if faces else None)


def _parse_ply_binary(body: bytes, elements, endian: str) -> Mesh:
    offset = 0
    vertices: np.ndarray | None = None
    faces: list[list[int]] = []
    for name, count, props in elements:
        is_list = any(p[0] == "list" for p in props)
        if not is_list:
            dtype = np.dtype([(p[-1], endian + _PLY_DTYPES[p[1]]) for p in props])
            block = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
            offset += dtype.itemsize * count
            if name == "vertex":
                vertices = np.stack([block["x"], block["y"], block["z"]], axis=1).astype(np.float64)
        else:
            lp = next(p for p in props if p[0] == "list")
            count_dt = np.dtype(endian + _PLY_DTYPES[lp[1]])
            index_dt = np.dtype(endian + _PLY_DTYPES[lp[2]])
            for _ in range(count):
                n = int(np.frombuffer(body, dtype=count_dt, count=1, offset=offset)[0])
                offset += count_dt.itemsize
                idx = np.frombuffer(body, dtype=index_dt, count=n, offset=offset)
                offset += index_dt.itemsize * n
                for i in range(1, n - 1):
                    faces.append([int(idx[0]), int(idx[i]), int(idx[i + 1])])
    return Mesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64) if faces else None)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_surface(mesh: Mesh, n_samples: int, *, seed: int = 0) -> np.ndarray:
    """Uniformly sample ``n_samples`` points by triangle area.

    Falls back to the raw vertices when the mesh has no faces (already a point
    cloud). Sampling makes accuracy/completeness comparable across meshes with
    wildly different tessellation density.
    """
    if mesh.faces is None or len(mesh.faces) == 0:
        return _resample_points(mesh.vertices, n_samples, seed=seed)

    v = mesh.vertices
    tris = v[mesh.faces]                      # (M, 3, 3)
    ab = tris[:, 1] - tris[:, 0]
    ac = tris[:, 2] - tris[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(ab, ac), axis=1)
    total = areas.sum()
    if total <= 0:
        return _resample_points(v, n_samples, seed=seed)

    rng = np.random.default_rng(seed)
    face_idx = rng.choice(len(mesh.faces), size=n_samples, p=areas / total)
    u = rng.random(n_samples)
    w = rng.random(n_samples)
    over = u + w > 1.0                        # fold back into the triangle
    u[over], w[over] = 1.0 - u[over], 1.0 - w[over]

    origin = tris[face_idx, 0]
    return origin + u[:, None] * ab[face_idx] + w[:, None] * ac[face_idx]


def _resample_points(points: np.ndarray, n_samples: int, *, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= n_samples:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=n_samples, replace=False)]


def load_and_sample(path: str | Path, n_samples: int, *, seed: int = 0) -> np.ndarray:
    return sample_surface(load_mesh(path), n_samples, seed=seed)


# --------------------------------------------------------------------------- #
# Writing (heatmap)
# --------------------------------------------------------------------------- #
def write_ply_colored(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write an ASCII PLY point cloud with per-point RGB (uint8)."""
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header",
    ]
    body = (
        f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}"
        for p, c in zip(points, colors)
    )
    Path(path).write_text("\n".join(lines) + "\n" + "\n".join(body) + "\n")
