"""
Mesh quality analysis for genus 0–6 surfaces.

Computes per-mesh quality metrics (aspect ratio, area distribution, edge lengths,
angles, Euler characteristic) and writes CSV/MD reports plus a 2-panel PDF figure.
Used to preempt reviewer questions on mesh quality (e.g. Section 5.3 genus–β₀).

Usage:
  python scripts/mesh_quality_analysis.py

Outputs:
  - data/results/mesh_quality_report.csv
  - data/results/mesh_quality_report.md
  - figures/fig_mesh_quality.pdf
  - Summary and optional WARNING to stdout.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent.parent
MESH_DIR = root / "data" / "meshes"
EXPERIMENTS_DIR = root / "data" / "experiments"
RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"

# Genus-sequence meshes (genus 0 through 6)
MESH_NAMES = [
    "sphere",       # g=0
    "torus",        # g=1
    "double_torus", # g=2
    "genus3",
    "genus4",
    "genus5",
    "genus6",
]

# Fallback: data/experiments filenames when data/meshes not used
MESH_NAME_TO_EXPERIMENT = {
    "sphere": "sphere_genus0",
    "torus": "torus_genus1",
    "double_torus": "double_torus_genus2",
    "genus3": "triple_torus_genus3",
    "genus4": "quad_torus_genus4",
    "genus5": "penta_torus_genus5",
    "genus6": "hex_torus_genus6",
}


def load_mesh(name: str):
    """
    Load a mesh from data/meshes by name. Tries .obj then .ply.
    If not found, tries data/experiments with genus-sequence filenames.
    Returns a trimesh.Trimesh or None if not found/loadable.
    """
    import trimesh

    def try_load(path: Path):
        if not path.exists():
            return None
        try:
            geom = trimesh.load(str(path), force="mesh")
            if hasattr(geom, "vertices"):
                return geom
            if hasattr(geom, "geometry") and geom.geometry:
                return list(geom.geometry.values())[0]
            return None
        except Exception as e:
            print(f"WARNING: Failed to load {path}: {e}", file=sys.stderr)
            return None

    for ext in (".obj", ".ply"):
        path = MESH_DIR / f"{name}{ext}"
        out = try_load(path)
        if out is not None:
            return out

    exp_name = MESH_NAME_TO_EXPERIMENT.get(name)
    if exp_name and EXPERIMENTS_DIR.exists():
        for ext in (".obj", ".ply"):
            path = EXPERIMENTS_DIR / f"{exp_name}{ext}"
            out = try_load(path)
            if out is not None:
                return out
    return None


def edge_length(vertices: np.ndarray, e0: np.ndarray, e1: np.ndarray) -> np.ndarray:
    """Return lengths of edges from vertex indices e0, e1."""
    return np.linalg.norm(vertices[e1] - vertices[e0], axis=1)


def compute_aspect_ratios(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    For each triangle, aspect ratio = longest edge / shortest edge.
    Returns array of shape (n_faces,).
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    a = np.linalg.norm(v1 - v0, axis=1)
    b = np.linalg.norm(v2 - v1, axis=1)
    c = np.linalg.norm(v0 - v2, axis=1)
    sides = np.stack([a, b, c], axis=1)
    longest = np.max(sides, axis=1)
    shortest = np.min(sides, axis=1)
    # Avoid division by zero for degenerate triangles
    aspect = np.where(shortest > 0, longest / shortest, np.nan)
    return aspect


def compute_triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area of each triangle (vectorized)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def compute_edge_lengths(mesh) -> np.ndarray:
    """All unique edge lengths."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    edges = np.asarray(mesh.edges_unique, dtype=np.int32)
    return edge_length(verts, edges[:, 0], edges[:, 1])


def compute_angles_deg(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    All triangle angles in degrees, shape (n_faces * 3,) or (n_faces, 3).
    Uses law of cosines.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    a = np.linalg.norm(v1 - v0, axis=1)
    b = np.linalg.norm(v2 - v1, axis=1)
    c = np.linalg.norm(v0 - v2, axis=1)
    # Angles at vertices 0, 1, 2 (sides opposite: a->0, b->1, c->2)
    # cos A = (b² + c² - a²) / (2 b c)
    cos0 = np.clip((b**2 + c**2 - a**2) / (2 * b * c + 1e-12), -1.0, 1.0)
    cos1 = np.clip((a**2 + c**2 - b**2) / (2 * a * c + 1e-12), -1.0, 1.0)
    cos2 = np.clip((a**2 + b**2 - c**2) / (2 * a * b + 1e-12), -1.0, 1.0)
    ang0 = np.degrees(np.arccos(cos0))
    ang1 = np.degrees(np.arccos(cos1))
    ang2 = np.degrees(np.arccos(cos2))
    return np.concatenate([ang0, ang1, ang2])


def euler_characteristic(mesh) -> int:
    """V - E + F for the mesh."""
    return int(mesh.vertices.shape[0] - mesh.edges_unique.shape[0] + mesh.faces.shape[0])


def expected_euler_for_genus(genus: int) -> int:
    """Expected Euler characteristic for closed orientable surface: 2 - 2*g."""
    return 2 - 2 * genus


def analyze_mesh(name: str, genus: int):
    """
    Load one mesh and compute all quality metrics.
    Returns a dict of metrics or None if mesh not found.
    """
    mesh = load_mesh(name)
    if mesh is None:
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if faces.shape[1] != 3:
        print(f"WARNING: {name} is not a triangle mesh (faces shape {faces.shape}), skipping.", file=sys.stderr)
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
        "mesh": name,
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
        # Keep raw arrays for figure
        "_aspect": aspect_valid,
        "_areas": areas,
        "_area_mean": area_mean,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    """Write metrics to CSV (exclude internal keys starting with _)."""
    if not rows:
        return
    fieldnames = [k for k in rows[0] if not k.startswith("_")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_md(rows: list[dict], path: Path) -> None:
    """Write a markdown table for quick inspection."""
    if not rows:
        return
    exclude = {"_aspect", "_areas", "_area_mean"}
    keys = [k for k in rows[0] if not k.startswith("_") and k not in exclude]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k)
            if isinstance(v, float) and not np.isnan(v):
                cells.append(f"{v:.4g}")
            elif isinstance(v, bool):
                cells.append("yes" if v else "no")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_figure(rows: list[dict], path: Path) -> None:
    """Two-panel figure: aspect ratio box plot, normalized area box plot (x = genus)."""
    if not rows:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    genera = [r["genus"] for r in rows]
    aspect_data = [r["_aspect"] for r in rows]
    # Normalized areas (by mean area per mesh)
    norm_area_data = [r["_areas"] / r["_area_mean"] for r in rows]

    positions = np.arange(len(rows))
    bp1 = ax1.boxplot(aspect_data, positions=positions, widths=0.5, patch_artist=True)
    bp2 = ax2.boxplot(norm_area_data, positions=positions, widths=0.5, patch_artist=True)

    for b in bp1["boxes"]:
        b.set_facecolor("white")
        b.set_edgecolor("black")
    for b in bp2["boxes"]:
        b.set_facecolor("white")
        b.set_edgecolor("black")

    ax1.set_xticks(positions)
    ax1.set_xticklabels([str(g) for g in genera])
    ax1.set_xlabel("Genus")
    ax1.set_ylabel("Aspect ratio (longest/shortest edge)")
    ax1.set_title("Triangle aspect ratio")

    ax2.set_xticks(positions)
    ax2.set_xticklabels([str(g) for g in genera])
    ax2.set_xlabel("Genus")
    ax2.set_ylabel("Area / mean area")
    ax2.set_title("Normalized triangle area")

    for ax in (ax1, ax2):
        ax.set_axisbelow(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    try:
        import trimesh
    except ImportError:
        print("trimesh not found. Install with: pip install trimesh", file=sys.stderr)
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, name in enumerate(MESH_NAMES):
        genus = i  # sphere=0, torus=1, ..., genus6=6
        rec = analyze_mesh(name, genus)
        if rec is None:
            print(f"WARNING: Skipping {name} (file not found or not loadable).", file=sys.stderr)
            continue
        rows.append(rec)

    if not rows:
        print("No meshes could be loaded. Place OBJ/PLY files in data/meshes/.", file=sys.stderr)
        return 1

    write_csv(rows, RESULTS_DIR / "mesh_quality_report.csv")
    write_md(rows, RESULTS_DIR / "mesh_quality_report.md")
    plot_figure(rows, FIGURES_DIR / "fig_mesh_quality.pdf")

    # Warnings
    for r in rows:
        if r.get("aspect_mean", 0) > 5 or r.get("area_cv", 0) > 1.0:
            print(
                f"WARNING: Mesh '{r['mesh']}' has mean aspect ratio = {r.get('aspect_mean', 0):.2f} "
                f"or area CV = {r.get('area_cv', 0):.2f}. Consider remeshing for better quality.",
                file=sys.stderr,
            )

    # Stdout summary
    print("Mesh quality summary (genus sequence)")
    print("-" * 60)
    for r in rows:
        euler_str = "ok" if r["euler_ok"] else "MISMATCH"
        print(
            f"  {r['mesh']:14} genus={r['genus']}  V={r['n_vertices']} F={r['n_faces']} E={r['n_edges']}  "
            f"aspect_mean={r['aspect_mean']:.3f}  area_CV={r['area_cv']:.3f}  Euler {euler_str}"
        )
    print("-" * 60)
    print(f"Reports: {RESULTS_DIR / 'mesh_quality_report.csv'}, {RESULTS_DIR / 'mesh_quality_report.md'}")
    print(f"Figure: {FIGURES_DIR / 'fig_mesh_quality.pdf'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
