"""
Cotangent-weight Laplacian negativity check for genus-sequence meshes.

For each of the 7 genus-sequence meshes (genus 0–6) at original resolution and
at remeshed resolutions (~10k, ~20k vertices):
1. Assemble the cotangent-weight matrix (same code as precompute.eigensolver).
2. Count off-diagonal entries: negative (acute, normal) vs positive (obtuse angles > π/2, can cause spurious eigenvalues).
3. Report total off-diagonal, n_negative, n_positive, pct_negative, pct_positive, max magnitudes.

Saves: data/results/cotangent_weight_negativity.csv

Usage (from project root):
  python scripts/analysis/cotangent_weight_negativity_check.py           # original + remeshed
  REMESH=0 python scripts/analysis/cotangent_weight_negativity_check.py  # original only (fast)
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

try:
    import gpytoolbox as gpy
except ImportError:
    gpy = None

EXPERIMENTS_DIR = root / "data" / "experiments"
RESULTS_DIR = root / "data" / "results"
OUT_CSV = RESULTS_DIR / "cotangent_weight_negativity.csv"

MESHES = [
    ("sphere_genus0", 0),
    ("torus_genus1", 1),
    ("double_torus_genus2", 2),
    ("triple_torus_genus3", 3),
    ("quad_torus_genus4", 4),
    ("penta_torus_genus5", 5),
    ("hex_torus_genus6", 6),
]


def assemble_cotangent_laplacian(V: np.ndarray, F: np.ndarray) -> csr_matrix:
    """Same assembly as precompute.eigensolver: intrinsic cotangent Laplacian."""
    if gpy is None:
        raise ImportError("gpytoolbox required. pip install gpytoolbox")
    n_verts = V.shape[0]
    F = np.asarray(F, dtype=np.int32)
    l_sq = gpy.halfedge_lengths_squared(V, F)
    L = gpy.cotangent_laplacian_intrinsic(l_sq, F, n=n_verts)
    if not isinstance(L, csr_matrix):
        L = csr_matrix(L)
    return L


def negativity_stats(L: csr_matrix) -> dict:
    """
    From cotangent Laplacian L (L_ij = -0.5*(cot α + cot β) for i≠j):
    - Acute angles → cot > 0 → L_ij < 0 (normal).
    - Obtuse angles (> π/2) → cot < 0 → L_ij can be positive (can cause spurious eigenvalues).
    Returns total_off_diagonal, n_negative, n_positive, pct_negative, pct_positive, max magnitudes.
    """
    L_coo = L.tocoo()
    row, col, data = L_coo.row, L_coo.col, np.asarray(L_coo.data).ravel()
    off_diag = row != col
    if not np.any(off_diag):
        return {
            "total_off_diagonal": 0,
            "n_negative": 0,
            "n_positive": 0,
            "pct_negative": 0.0,
            "pct_positive": 0.0,
            "max_neg_magnitude": 0.0,
            "max_pos_magnitude": 0.0,
        }
    off_data = data[off_diag]
    total = int(off_diag.sum())
    neg = off_data < 0
    pos = off_data > 0
    n_neg = int(neg.sum())
    n_pos = int(pos.sum())
    pct_neg = 100.0 * n_neg / total if total else 0.0
    pct_pos = 100.0 * n_pos / total if total else 0.0
    neg_vals = off_data[neg]
    pos_vals = off_data[pos]
    max_neg_mag = float(np.max(np.abs(neg_vals))) if len(neg_vals) > 0 else 0.0
    max_pos_mag = float(np.max(pos_vals)) if len(pos_vals) > 0 else 0.0
    return {
        "total_off_diagonal": total,
        "n_negative": n_neg,
        "n_positive": n_pos,
        "pct_negative": pct_neg,
        "pct_positive": pct_pos,
        "max_neg_magnitude": max_neg_mag,
        "max_pos_magnitude": max_pos_mag,
    }


def load_mesh(mesh_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = EXPERIMENTS_DIR / (mesh_id + ".obj")
    if not path.exists():
        return None
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int32)
    if F.shape[1] != 3:
        return None
    return V, F


def _edge_lengths(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    e0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    e1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    lengths = np.linalg.norm(V[e1] - V[e0], axis=1)
    edges = np.sort(np.column_stack([e0, e1]), axis=1)
    keys = edges[:, 0].astype(np.int64) * (edges[:, 0].max() + 1) + edges[:, 1]
    _, idx = np.unique(keys, return_index=True)
    return lengths[idx]


def remesh_pymeshlab(V: np.ndarray, F: np.ndarray, target_edge: float, iterations: int = 10):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(V, F)
    ms.add_mesh(m)
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


def _euler(V: np.ndarray, F: np.ndarray) -> int:
    nv, nf = V.shape[0], F.shape[0]
    e0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    e1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    edges = np.sort(np.column_stack([e0, e1]), axis=1)
    ne = len(np.unique(edges[:, 0] * (edges[:, 0].max() + 1) + edges[:, 1]))
    return int(nv - ne + nf)


def expected_euler(genus: int) -> int:
    return 2 - 2 * genus


def build_resolutions(
    mesh_id: str, genus: int, do_remesh: bool = True
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Original + optionally remeshed_10k (0.7× edge), remeshed_20k (0.5× edge). Skip if Euler not preserved."""
    out = []
    V, F = load_mesh(mesh_id)
    if V is None:
        return out
    if _euler(V, F) != expected_euler(genus):
        return out
    out.append(("original", V.copy(), F.copy()))
    if not do_remesh:
        return out
    mean_edge = float(np.mean(_edge_lengths(V, F)))
    for label, scale in [("remeshed_10k", 0.7), ("remeshed_20k", 0.5)]:
        try:
            Vr, Fr = remesh_pymeshlab(V, F, mean_edge * scale)
        except Exception:
            continue
        if _euler(Vr, Fr) != expected_euler(genus):
            continue
        out.append((label, Vr, Fr))
    return out


def run():
    if gpy is None:
        print("gpytoolbox required. pip install gpytoolbox", file=sys.stderr)
        sys.exit(1)

    do_remesh = os.environ.get("REMESH", "1").strip().lower() not in ("0", "false", "no")
    if not do_remesh:
        print("REMESH=0: skipping remeshed resolutions (original only).", file=sys.stderr)

    rows = []
    for mesh_id, genus in MESHES:
        resolutions = build_resolutions(mesh_id, genus, do_remesh=do_remesh)
        if not resolutions:
            print(f"[SKIP] {mesh_id}: no resolutions (missing file or Euler check failed)", file=sys.stderr)
            continue
        for resolution, V, F in resolutions:
            try:
                L = assemble_cotangent_laplacian(V, F)
            except Exception as e:
                print(f"[ERROR] {mesh_id} {resolution}: {e}", file=sys.stderr)
                continue
            stats = negativity_stats(L)
            n_verts = V.shape[0]
            rows.append({
                "mesh_id": mesh_id,
                "genus": genus,
                "resolution": resolution,
                "n_vertices": n_verts,
                "total_off_diagonal": stats["total_off_diagonal"],
                "n_negative": stats["n_negative"],
                "n_positive": stats["n_positive"],
                "pct_negative": round(stats["pct_negative"], 4),
                "pct_positive": round(stats["pct_positive"], 4),
                "max_neg_magnitude": round(stats["max_neg_magnitude"], 8),
                "max_pos_magnitude": round(stats["max_pos_magnitude"], 8),
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mesh_id", "genus", "resolution", "n_vertices",
        "total_off_diagonal", "n_negative", "n_positive", "pct_negative", "pct_positive",
        "max_neg_magnitude", "max_pos_magnitude",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")
    for r in rows:
        print(
            f"  {r['mesh_id']} {r['resolution']}: n_verts={r['n_vertices']}, "
            f"off_diag={r['total_off_diagonal']}, neg={r['n_negative']} ({r['pct_negative']}%), "
            f"pos={r['n_positive']} ({r['pct_positive']}% obtuse), "
            f"max_neg={r['max_neg_magnitude']}, max_pos={r['max_pos_magnitude']}"
        )


if __name__ == "__main__":
    run()
