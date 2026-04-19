"""
Eigenvalue convergence test under mesh refinement — ALL genus 0–6.

Runs the same convergence test as eigenvalue_convergence_test.py on all seven
genus-sequence meshes (genus 0 through 6) to show robustness. Resolutions:
original (~5k), isotropic remesh ~10k (0.7× edge), ~20k (0.5× edge). After each
remesh, Euler characteristic χ = V−E+F = 2−2g is verified; if not preserved,
that resolution is reported and skipped.

Usage (from project root):
  conda activate sonifold
  python scripts/eigenvalue_convergence_test_all_genus.py

Outputs:
  - data/results/eigenvalue_convergence_test_all_genus.csv
  - figures/fig_eigenvalue_convergence_all_genus.pdf
  - Summary to stdout with max % change in H and mean β₀; flags if H >5% or mean β₀ >30%.
"""

from __future__ import annotations

import os
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import cupy  # noqa: F401
    os.environ["USE_GPU"] = "1"
    _USE_GPU = True
except ImportError:
    os.environ["USE_GPU"] = "0"
    _USE_GPU = False

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from analysis.spectral_descriptors import compute_beta0_stats, spectral_entropy
from precompute.eigensolver import compute_eigen

EXPERIMENTS_DIR = root / "data" / "experiments"
RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"
OUT_CSV = RESULTS_DIR / "eigenvalue_convergence_test_all_genus.csv"
OUT_FIG = FIGURES_DIR / "fig_eigenvalue_convergence_all_genus.pdf"

# All seven genus-sequence meshes (genus 0 through 6)
MESHES = [
    ("sphere_genus0", 0),
    ("torus_genus1", 1),
    ("double_torus_genus2", 2),
    ("triple_torus_genus3", 3),
    ("quad_torus_genus4", 4),
    ("penta_torus_genus5", 5),
    ("hex_torus_genus6", 6),
]

K = 50
N_BETA0_TRIALS = 50
RNG = np.random.default_rng(42)


def _edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unique edge lengths."""
    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    lengths = np.linalg.norm(vertices[e1] - vertices[e0], axis=1)
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


def mesh_quality_metrics(V: np.ndarray, F: np.ndarray) -> dict:
    aspect = _aspect_ratios(V, F)
    aspect_valid = aspect[~np.isnan(aspect)]
    areas = _triangle_areas(V, F)
    mean_ar = float(np.mean(aspect_valid)) if len(aspect_valid) > 0 else np.nan
    area_cv = float(np.std(areas) / np.mean(areas)) if np.mean(areas) > 0 else np.nan
    return {"mean_aspect_ratio": mean_ar, "area_cv": area_cv}


def verify_euler(V: np.ndarray, F: np.ndarray, genus: int) -> bool:
    return _euler(V, F) == expected_euler(genus)


def run_one_resolution(
    mesh_name: str,
    genus: int,
    resolution_label: str,
    V: np.ndarray,
    F: np.ndarray,
) -> dict | None:
    """Compute eigenvalues, H, beta0, quality for one (mesh, resolution)."""
    n_verts = V.shape[0]
    n_faces = F.shape[0]
    quality = mesh_quality_metrics(V, F)

    try:
        evals, evecs = compute_eigen(V, F, N=K)
    except Exception as e:
        print(f"  [ERROR] {mesh_name} {resolution_label}: eigensolver failed: {e}", file=sys.stderr)
        return None

    evals = np.asarray(evals, dtype=np.float64).ravel()
    if len(evals) < K:
        print(f"  [WARN] {mesh_name} {resolution_label}: only {len(evals)} eigenvalues", file=sys.stderr)
    evals = evals[:K]

    H = spectral_entropy(evals)
    mean_beta0, std_beta0 = compute_beta0_stats(V, F, evecs, min(K, evecs.shape[0]), N_BETA0_TRIALS, RNG)

    row = {
        "mesh_name": mesh_name,
        "genus": genus,
        "resolution": resolution_label,
        "num_vertices": n_verts,
        "n_vertices": n_verts,
        "n_faces": n_faces,
        "mean_aspect_ratio": round(quality["mean_aspect_ratio"], 4) if not np.isnan(quality["mean_aspect_ratio"]) else "",
        "area_cv": round(quality["area_cv"], 4) if not np.isnan(quality["area_cv"]) else "",
        "spectral_entropy_H": round(H, 6),
        "mean_beta0": round(mean_beta0, 4),
        "std_beta0": round(std_beta0, 4),
    }
    for i in range(len(evals)):
        row[f"lambda_{i+1}"] = round(float(evals[i]), 8)
    for i in range(len(evals), K):
        row[f"lambda_{i+1}"] = ""
    return row


def build_resolutions(mesh_id: str, genus: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """
    Returns [(resolution_label, V, F), ...] for original, ~10k, ~20k.
    After each remesh, Euler χ = 2−2g is verified; if not preserved, that
    resolution is reported and skipped (not added to the list).
    """
    out = []
    V, F = load_mesh(mesh_id)
    if V is None:
        return out

    if not verify_euler(V, F, genus):
        print(
            f"  [SKIP] {mesh_id} original: Euler χ = V−E+F = {_euler(V, F)} "
            f"(expected {expected_euler(genus)} = 2−2g). Skipping mesh.",
            file=sys.stderr,
        )
        return out

    out.append(("original", V.copy(), F.copy()))
    mean_edge = float(np.mean(_edge_lengths(V, F)))

    for label, scale in [("remeshed_10k", 0.7), ("remeshed_20k", 0.5)]:
        try:
            Vr, Fr = remesh_pymeshlab(V, F, mean_edge * scale)
        except Exception as e:
            print(f"  [WARN] Remeshing {mesh_id} → {label} failed: {e}. Skipping this resolution.", file=sys.stderr)
            continue
        if not verify_euler(Vr, Fr, genus):
            print(
                f"  [SKIP] {mesh_id} {label}: Euler χ = {_euler(Vr, Fr)} "
                f"(expected {expected_euler(genus)}). Remeshing failed — skipping this resolution.",
                file=sys.stderr,
            )
            continue
        out.append((label, Vr, Fr))

    return out


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    lambda_cols = [f"lambda_{i+1}" for i in range(K)]
    fieldnames = [
        "genus", "resolution", "num_vertices", "mean_aspect_ratio",
        "spectral_entropy_H", "mean_beta0", "std_beta0",
    ] + lambda_cols
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV}", file=sys.stderr)


def plot_figure(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_genus = {}
    for r in rows:
        g = r["genus"]
        if g not in by_genus:
            by_genus[g] = []
        by_genus[g].append(r)

    genus_order = sorted(by_genus.keys())
    res_order = ["original", "remeshed_10k", "remeshed_20k"]
    colors = ["C0", "C1", "C2"]
    markers = ["o", "s", "^"]

    n_rows = len(genus_order)
    if n_rows == 0:
        print("No rows for figure.", file=sys.stderr)
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_height = max(8, 1.8 * n_rows)
    fig = plt.figure(figsize=(10, fig_height))
    row_h = 1.0 / n_rows
    margin = 0.02
    left_w = 0.48
    right_w = 0.48
    gap = 0.04

    for row_i, genus in enumerate(genus_order):
        res_rows = by_genus[genus]
        res_rows = sorted(
            res_rows,
            key=lambda r: res_order.index(r["resolution"]) if r["resolution"] in res_order else 99,
        )
        y_bottom = 1.0 - (row_i + 1) * row_h + margin
        y_height = row_h - 2 * margin

        ax_left = fig.add_axes([margin, y_bottom, left_w - margin, y_height])
        ax_right = fig.add_axes([0.5 + gap, y_bottom, right_w - gap, y_height])

        x = np.arange(1, K + 1)
        for i, r in enumerate(res_rows):
            lam = [r.get(f"lambda_{j+1}") for j in range(K)]
            lam = [float(v) for v in lam if v != "" and v is not None]
            if not lam:
                continue
            y = np.array(lam[: min(len(x), len(lam))])
            x_k = x[: len(y)]
            ax_left.stem(
                x_k, y,
                linefmt=colors[i % len(colors)] + "-",
                markerfmt=markers[i % len(markers)],
                basefmt=" ",
                label=r["resolution"] + f" (n={r['num_vertices']})",
            )
        ax_left.set_xlabel("$k$")
        ax_left.set_ylabel("$\\lambda_k$")
        ax_left.set_title(f"Genus {genus}")
        ax_left.legend(loc="upper right", fontsize=6)
        ax_left.grid(True, alpha=0.3)

        labels = [r["resolution"] for r in res_rows]
        H_vals = [r.get("spectral_entropy_H") or 0 for r in res_rows]
        b0_vals = [r.get("mean_beta0") or 0 for r in res_rows]
        x_pos = np.arange(len(labels))
        w = 0.35
        ax_right.bar(x_pos - w / 2, H_vals, w, label="$H$", color="steelblue")
        ax_right_twin = ax_right.twinx()
        ax_right_twin.bar(x_pos + w / 2, b0_vals, w, label=r"mean $\beta_0$", color="coral", alpha=0.8)
        ax_right.set_xticks(x_pos)
        ax_right.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        ax_right.set_ylabel("$H$")
        ax_right_twin.set_ylabel(r"mean $\beta_0$")
        ax_right.legend(loc="upper left", fontsize=6)
        ax_right_twin.legend(loc="upper right", fontsize=6)

    plt.savefig(OUT_FIG, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_FIG}", file=sys.stderr)


def print_summary(rows: list[dict]) -> None:
    """Per genus: max % change in H and mean_β₀ (original vs 20k). Flag if H >5% or mean_β₀ >30%."""
    by_genus = {}
    for r in rows:
        g = r["genus"]
        if g not in by_genus:
            by_genus[g] = []
        by_genus[g].append(r)

    res_order = ["original", "remeshed_10k", "remeshed_20k"]
    print("\n--- Eigenvalue convergence summary (all genus) ---")
    for genus in sorted(by_genus.keys()):
        res_rows = by_genus[genus]
        res_rows = sorted(
            res_rows,
            key=lambda r: res_order.index(r["resolution"]) if r["resolution"] in res_order else 99,
        )
        orig = next((r for r in res_rows if r["resolution"] == "original"), None)
        fine = next((r for r in res_rows if r["resolution"] == "remeshed_20k"), None)
        if orig is None or fine is None:
            print(f"  Genus {genus}: missing original or remeshed_20k — skipping summary")
            continue

        H_orig = orig.get("spectral_entropy_H")
        H_fine = fine.get("spectral_entropy_H")
        b0_orig = orig.get("mean_beta0")
        b0_fine = fine.get("mean_beta0")

        H_change_pct = None
        if H_orig is not None and H_fine is not None and abs(H_fine) > 1e-20:
            H_change_pct = 100.0 * abs(H_fine - H_orig) / abs(H_fine)

        b0_change_pct = None
        if b0_orig is not None and b0_fine is not None and abs(b0_fine) > 1e-20:
            b0_change_pct = 100.0 * abs(b0_fine - b0_orig) / abs(b0_fine)

        flags = []
        if H_change_pct is not None and H_change_pct > 5.0:
            flags.append("H change >5%")
        if b0_change_pct is not None and b0_change_pct > 30.0:
            flags.append("mean β₀ change >30%")

        line = f"  Genus {genus}:"
        if H_change_pct is not None:
            line += f" max % change in H = {H_change_pct:.2f}%"
        if b0_change_pct is not None:
            line += f", max % change in mean β₀ = {b0_change_pct:.2f}%"
        if flags:
            line += f"  [FLAG: {', '.join(flags)}]"
        print(line)
    print("---\n")


def main() -> int:
    try:
        import trimesh
    except ImportError:
        print("trimesh required: pip install trimesh", file=sys.stderr)
        return 1
    try:
        import pymeshlab
    except ImportError:
        print("pymeshlab required: pip install pymeshlab", file=sys.stderr)
        return 1

    if _USE_GPU:
        print("Eigenvalue backend: GPU (CuPy).", file=sys.stderr)
    else:
        print("Eigenvalue backend: CPU (CuPy not installed).", file=sys.stderr)

    all_rows = []
    for mesh_id, genus in MESHES:
        resolutions = build_resolutions(mesh_id, genus)
        for res_label, V, F in resolutions:
            row = run_one_resolution(mesh_id, genus, res_label, V, F)
            if row is not None:
                all_rows.append(row)
                print(
                    f"  {mesh_id} {res_label}: n={row['num_vertices']} aspect={row.get('mean_aspect_ratio')} "
                    f"H={row['spectral_entropy_H']} mean_β0={row['mean_beta0']}",
                    file=sys.stderr,
                )

    if not all_rows:
        print("No results.", file=sys.stderr)
        return 1

    write_csv(all_rows)
    plot_figure(all_rows)
    print_summary(all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
