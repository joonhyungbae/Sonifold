"""
Mesh quality analysis for remeshed genus 3–6 meshes only.

Loads from data/results/remeshed_genus/*_remeshed.obj, reuses metrics from
scripts/mesh_quality_analysis.py, writes data/results/remeshed_genus/mesh_quality_report.csv.

Usage (from project root):
  python scripts/analysis/mesh_quality_remeshed.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
REMESHED_DIR = root / "data" / "results" / "remeshed_genus"

# Reuse analysis logic from mesh_quality_analysis (same directory)
from mesh_quality_analysis import (
    compute_angles_deg,
    compute_aspect_ratios,
    compute_edge_lengths,
    compute_triangle_areas,
    euler_characteristic,
    expected_euler_for_genus,
)

REMESHED_MESH_IDS = [
    ("triple_torus_genus3_remeshed", 3),
    ("quad_torus_genus4_remeshed", 4),
    ("penta_torus_genus5_remeshed", 5),
    ("hex_torus_genus6_remeshed", 6),
]


def load_remeshed_mesh(mesh_id: str):
    import trimesh
    path = REMESHED_DIR / (mesh_id + ".obj")
    if not path.exists():
        return None
    try:
        return trimesh.load(str(path), force="mesh")
    except Exception as e:
        print("WARNING: Failed to load {}: {}".format(path, e), file=sys.stderr)
        return None


def analyze_one(mesh_id: str, genus: int):
    mesh = load_remeshed_mesh(mesh_id)
    if mesh is None:
        return None
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if faces.shape[1] != 3:
        return None
    n_vertices = len(vertices)
    n_faces = len(faces)
    n_edges = mesh.edges_unique.shape[0]

    aspect = compute_aspect_ratios(vertices, faces)
    aspect_valid = aspect[~np.isnan(aspect)]
    areas = compute_triangle_areas(vertices, faces)
    edge_len = compute_edge_lengths(mesh)
    angles = compute_angles_deg(vertices, faces)

    area_mean = float(np.mean(areas))
    area_std = float(np.std(areas))
    area_cv = area_std / area_mean if area_mean > 0 else float("nan")
    angle_deg_min = float(np.min(angles))
    angle_deg_max = float(np.max(angles))
    degenerate_low = int(np.sum(angles < 10))
    degenerate_high = int(np.sum(angles > 150))
    euler = euler_characteristic(mesh)
    expected = expected_euler_for_genus(genus)
    euler_ok = euler == expected

    if len(aspect_valid) == 0:
        aspect_mean = aspect_median = aspect_min = aspect_max = aspect_std = float("nan")
    else:
        aspect_mean = float(np.mean(aspect_valid))
        aspect_median = float(np.median(aspect_valid))
        aspect_min = float(np.min(aspect_valid))
        aspect_max = float(np.max(aspect_valid))
        aspect_std = float(np.std(aspect_valid))

    edge_mean = float(np.mean(edge_len))
    edge_std = float(np.std(edge_len))
    edge_cv = edge_std / edge_mean if edge_mean > 0 else float("nan")

    return {
        "mesh": mesh_id,
        "genus": genus,
        "n_vertices": n_vertices,
        "n_faces": n_faces,
        "n_edges": n_edges,
        "aspect_mean": aspect_mean,
        "aspect_median": aspect_median,
        "aspect_min": aspect_min,
        "aspect_max": aspect_max,
        "aspect_std": aspect_std,
        "area_mean": area_mean,
        "area_std": area_std,
        "area_min": float(np.min(areas)),
        "area_max": float(np.max(areas)),
        "area_cv": area_cv,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "edge_min": float(np.min(edge_len)),
        "edge_max": float(np.max(edge_len)),
        "edge_cv": edge_cv,
        "angle_mean": float(np.mean(angles)),
        "angle_std": float(np.std(angles)),
        "angle_min": angle_deg_min,
        "angle_max": angle_deg_max,
        "angle_lt_10_count": degenerate_low,
        "angle_gt_150_count": degenerate_high,
        "euler": euler,
        "euler_expected": expected,
        "euler_ok": euler_ok,
    }


def main() -> int:
    try:
        import trimesh
    except ImportError:
        print("trimesh required: pip install trimesh", file=sys.stderr)
        return 1

    REMESHED_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for mesh_id, genus in REMESHED_MESH_IDS:
        rec = analyze_one(mesh_id, genus)
        if rec is None:
            print("WARNING: Skipping {}".format(mesh_id), file=sys.stderr)
            continue
        rows.append(rec)

    if not rows:
        print("No remeshed meshes found in {}".format(REMESHED_DIR), file=sys.stderr)
        return 1

    out_csv = REMESHED_DIR / "mesh_quality_report.csv"
    fieldnames = [k for k in rows[0]]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Saved {}".format(out_csv), file=sys.stderr)
    for r in rows:
        euler_str = "ok" if r["euler_ok"] else "MISMATCH"
        print("  {} genus={} V={} aspect_mean={:.3f} area_cv={:.3f} Euler {}".format(
            r["mesh"], r["genus"], r["n_vertices"], r["aspect_mean"], r["area_cv"], euler_str), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
