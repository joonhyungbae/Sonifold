"""
Isotropic remeshing of genus 3–6 meshes for quality comparable to sphere/torus.

- Loads genus 3–6 from data/experiments (triple_torus_genus3 … hex_torus_genus6).
- Target edge length = mean edge length of torus (from data/experiments/torus_genus1.obj).
- Uses PyMeshLab isotropic explicit remeshing; falls back to trimesh if needed.
- Verifies: mean aspect ratio < 5, area CV < 0.3, no angles < 10° or > 150°, Euler = 2 - 2g.
- Saves remeshed OBJ to data/results/remeshed_genus/{mesh_id}_remeshed.obj.

Usage (from project root):
  pip install pymeshlab   # recommended
  python scripts/analysis/remesh_genus_meshes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = root / "data" / "experiments"
REMESHED_DIR = root / "data" / "results" / "remeshed_genus"

# Genus 3–6 only (mesh_id in data/experiments)
GENUS_REMESH_IDS = [
    ("triple_torus_genus3", 3),
    ("quad_torus_genus4", 4),
    ("penta_torus_genus5", 5),
    ("hex_torus_genus6", 6),
]

TARGET_VERTICES_APPROX = 5000
MAX_ASPECT_MEAN = 5.0
MAX_AREA_CV = 0.3
MIN_ANGLE_DEG = 10.0
MAX_ANGLE_DEG = 150.0


def _edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unique edge lengths from triangle mesh."""
    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    lengths = np.linalg.norm(vertices[e1] - vertices[e0], axis=1)
    # unique edges: (i,j) with i < j
    edges = np.sort(np.column_stack([e0, e1]), axis=1)
    keys = edges[:, 0].astype(np.int64) * (edges[:, 0].max() + 1) + edges[:, 1]
    _, idx = np.unique(keys, return_index=True)
    return lengths[idx]


def _aspect_ratios(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    a = np.linalg.norm(v1 - v0, axis=1)
    b = np.linalg.norm(v2 - v1, axis=1)
    c = np.linalg.norm(v0 - v2, axis=1)
    sides = np.stack([a, b, c], axis=1)
    longest = np.max(sides, axis=1)
    shortest = np.min(sides, axis=1)
    return np.where(shortest > 1e-12, longest / shortest, np.nan)


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _angles_deg(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    a = np.linalg.norm(v1 - v0, axis=1)
    b = np.linalg.norm(v2 - v1, axis=1)
    c = np.linalg.norm(v0 - v2, axis=1)
    cos0 = np.clip((b**2 + c**2 - a**2) / (2 * b * c + 1e-12), -1.0, 1.0)
    cos1 = np.clip((a**2 + c**2 - b**2) / (2 * a * c + 1e-12), -1.0, 1.0)
    cos2 = np.clip((a**2 + b**2 - c**2) / (2 * a * b + 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(np.concatenate([cos0, cos1, cos2])))


def _euler(V: np.ndarray, F: np.ndarray) -> int:
    nv = V.shape[0]
    nf = F.shape[0]
    e0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    e1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    edges = np.sort(np.column_stack([e0, e1]), axis=1)
    ne = len(np.unique(edges[:, 0] * (edges[:, 0].max() + 1) + edges[:, 1]))
    return int(nv - ne + nf)


def expected_euler(genus: int) -> int:
    return 2 - 2 * genus


def get_torus_mean_edge() -> float:
    import trimesh
    path = EXPERIMENTS_DIR / "torus_genus1.obj"
    if not path.exists():
        raise FileNotFoundError("Torus mesh not found: {}".format(path))
    m = trimesh.load(str(path), force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int32)
    lengths = _edge_lengths(V, F)
    return float(np.mean(lengths))


def verify_quality(V: np.ndarray, F: np.ndarray, genus: int) -> tuple[bool, dict]:
    """Returns (ok, metrics_dict)."""
    aspect = _aspect_ratios(V, F)
    aspect_valid = aspect[~np.isnan(aspect)]
    areas = _triangle_areas(V, F)
    angles = _angles_deg(V, F)
    euler = _euler(V, F)
    expected = expected_euler(genus)

    aspect_mean = float(np.mean(aspect_valid)) if len(aspect_valid) > 0 else np.nan
    area_cv = float(np.std(areas) / np.mean(areas)) if np.mean(areas) > 0 else np.nan
    angle_lt_10 = int(np.sum(angles < MIN_ANGLE_DEG))
    angle_gt_150 = int(np.sum(angles > MAX_ANGLE_DEG))
    euler_ok = euler == expected

    metrics = {
        "n_vertices": V.shape[0],
        "n_faces": F.shape[0],
        "aspect_mean": aspect_mean,
        "area_cv": area_cv,
        "angle_lt_10": angle_lt_10,
        "angle_gt_150": angle_gt_150,
        "euler": euler,
        "euler_expected": expected,
        "euler_ok": euler_ok,
    }
    ok = (
        aspect_mean < MAX_ASPECT_MEAN
        and area_cv < MAX_AREA_CV
        and angle_lt_10 == 0
        and angle_gt_150 == 0
        and euler_ok
    )
    return ok, metrics


def remesh_pymeshlab(V: np.ndarray, F: np.ndarray, target_edge: float, iterations: int = 10):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(V, F)
    ms.add_mesh(m)
    # targetlen: absolute if AbsoluteValue exists, else percentage of bbox diagonal
    try:
        targetlen_arg = pymeshlab.AbsoluteValue(target_edge)
    except AttributeError:
        diag = float(np.linalg.norm(np.ptp(V, axis=0)))
        if diag < 1e-12:
            diag = 1.0
        pct = 100.0 * target_edge / diag
        targetlen_arg = pymeshlab.PercentageValue(pct)
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations,
        targetlen=targetlen_arg,
    )
    out = ms.current_mesh()
    return np.asarray(out.vertex_matrix(), dtype=np.float64), np.asarray(out.face_matrix(), dtype=np.int32)


def remesh_one(mesh_id: str, genus: int, target_edge: float) -> tuple[np.ndarray, np.ndarray] | None:
    import trimesh
    path = EXPERIMENTS_DIR / (mesh_id + ".obj")
    if not path.exists():
        print("Skip {}: file not found".format(path), file=sys.stderr)
        return None
    m = trimesh.load(str(path), force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int32)
    if F.shape[1] != 3:
        print("Skip {}: not triangular".format(mesh_id), file=sys.stderr)
        return None

    try:
        Vr, Fr = remesh_pymeshlab(V, F, target_edge)
    except Exception as e:
        print("Remesh failed for {}: {}".format(mesh_id, e), file=sys.stderr)
        return None

    ok, metrics = verify_quality(Vr, Fr, genus)
    print(
        "  {} -> V={} F={} aspect_mean={:.3f} area_cv={:.3f} angle_lt_10={} angle_gt_150={} Euler {} (expected {}) {}".format(
            mesh_id,
            metrics["n_vertices"],
            metrics["n_faces"],
            metrics["aspect_mean"],
            metrics["area_cv"],
            metrics["angle_lt_10"],
            metrics["angle_gt_150"],
            metrics["euler"],
            metrics["euler_expected"],
            "OK" if ok else "FAIL",
        ),
        file=sys.stderr,
    )
    if not ok:
        print("  WARNING: quality checks not all passed; saving anyway.", file=sys.stderr)
    return Vr, Fr


def main() -> int:
    try:
        import trimesh
    except ImportError:
        print("trimesh required: pip install trimesh", file=sys.stderr)
        return 1
    try:
        import pymeshlab
    except ImportError:
        print("pymeshlab required for isotropic remeshing: pip install pymeshlab", file=sys.stderr)
        return 1

    REMESHED_DIR.mkdir(parents=True, exist_ok=True)
    target_edge = get_torus_mean_edge()
    print("Torus mean edge length = {:.6f}".format(target_edge), file=sys.stderr)

    for mesh_id, genus in GENUS_REMESH_IDS:
        result = remesh_one(mesh_id, genus, target_edge)
        if result is None:
            continue
        Vr, Fr = result
        out_path = REMESHED_DIR / (mesh_id + "_remeshed.obj")
        import trimesh
        out_mesh = trimesh.Trimesh(vertices=Vr, faces=Fr)
        out_mesh.export(str(out_path))
        print("Saved {}".format(out_path), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
