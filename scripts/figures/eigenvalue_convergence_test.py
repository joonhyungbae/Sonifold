"""
Eigenvalue convergence test under mesh refinement.

Tests whether cotangent-weight Laplace–Beltrami eigenvalues (and hence spectral entropy H
and β₀) are stable under isotropic remeshing. Uses genus 4 (worst quality) and genus 2
(double torus, control). For each mesh: original (~5k), remeshed ~10k, remeshed ~20k.
Same LB assembly as the rest of the pipeline (gpytoolbox cotangent + Voronoi mass).

Usage (from project root):
  # Recommended: use sonifold conda env (GPU + pymeshlab included).
  #   conda activate sonifold
  #   python scripts/eigenvalue_convergence_test.py
  # Or create env from repo: conda env create -f environment.yml && conda activate sonifold
  #
  # Otherwise: pip install pymeshlab trimesh gpytoolbox scipy numpy matplotlib
  # Optional GPU: pip install cupy-cuda12x (CUDA 12) or cupy-cuda11x (CUDA 11).
  python scripts/eigenvalue_convergence_test.py

Outputs:
  - data/results/eigenvalue_convergence_test.csv
  - figures/fig_eigenvalue_convergence.pdf
  - Summary of relative changes to stdout.
"""

from __future__ import annotations

import os
import csv
import sys
from pathlib import Path

import numpy as np

# Prefer GPU for eigenvalue solve when CuPy is available (much faster for ~10k–20k vertices)
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
OUT_CSV = RESULTS_DIR / "eigenvalue_convergence_test.csv"
OUT_FIG = FIGURES_DIR / "fig_eigenvalue_convergence.pdf"

# Genus 4 (worst quality), genus 2 (control)
MESHES = [
    ("quad_torus_genus4", 4),
    ("double_torus_genus2", 2),
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
    """Returns [(resolution_label, V, F), ...] for original, ~10k, ~20k."""
    out = []
    V, F = load_mesh(mesh_id)
    if V is None:
        return out

    out.append(("original", V.copy(), F.copy()))
    mean_edge = float(np.mean(_edge_lengths(V, F)))

    try:
        V2, F2 = remesh_pymeshlab(V, F, mean_edge * 0.7)
        out.append(("remeshed_10k", V2, F2))
        V3, F3 = remesh_pymeshlab(V, F, mean_edge * 0.5)
        out.append(("remeshed_20k", V3, F3))
    except Exception as e:
        print(f"  [WARN] Remeshing {mesh_id} failed: {e}. Using original only.", file=sys.stderr)

    return out


def verify_euler(V: np.ndarray, F: np.ndarray, genus: int) -> bool:
    return _euler(V, F) == expected_euler(genus)


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    # Columns: mesh_name, genus, resolution, n_vertices, n_faces, mean_aspect_ratio, area_cv,
    #          lambda_1..lambda_50, spectral_entropy_H, mean_beta0, std_beta0
    lambda_cols = [f"lambda_{i+1}" for i in range(K)]
    fieldnames = [
        "mesh_name", "genus", "resolution", "n_vertices", "n_faces",
        "mean_aspect_ratio", "area_cv",
    ] + lambda_cols + ["spectral_entropy_H", "mean_beta0", "std_beta0"]
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

    # Group by mesh: need 3 resolutions each for genus 4 and genus 2
    by_mesh = {}
    for r in rows:
        key = (r["mesh_name"], r["genus"])
        if key not in by_mesh:
            by_mesh[key] = []
        by_mesh[key].append(r)

    # Order: genus 4 first, then genus 2
    mesh_order = [(m, g) for m, g in MESHES if (m, g) in by_mesh]
    if not mesh_order:
        print("No rows for figure.", file=sys.stderr)
        return

    res_order = ["original", "remeshed_10k", "remeshed_20k"]
    colors = ["C0", "C1", "C2"]
    markers = ["o", "s", "^"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # Left: two panels — genus 4 stems (main), genus 2 stems (below or inset). Right: one bar chart for both meshes.
    fig = plt.figure(figsize=(10, 8))
    ax_stem_g4 = fig.add_axes([0.08, 0.55, 0.38, 0.38])   # left top: genus 4 eigenvalues
    ax_stem_g2 = fig.add_axes([0.08, 0.08, 0.38, 0.38])   # left bottom: genus 2 eigenvalues
    ax_bars = fig.add_axes([0.55, 0.15, 0.42, 0.75])     # right: H and mean β₀ for both meshes

    # Left top: genus 4 stem overlay
    if (MESHES[0][0], MESHES[0][1]) in by_mesh:
        mesh_name, genus = MESHES[0]
        res_rows = by_mesh[(mesh_name, genus)]
        res_rows = sorted(res_rows, key=lambda r: res_order.index(r["resolution"]) if r["resolution"] in res_order else 99)
        x = np.arange(1, K + 1)
        for i, r in enumerate(res_rows):
            lam = [r.get(f"lambda_{j+1}") for j in range(K)]
            lam = [float(v) for v in lam if v != "" and v is not None]
            if not lam:
                continue
            y = np.array(lam[: min(len(x), len(lam))])
            x_k = x[: len(y)]
            ax_stem_g4.stem(
                x_k, y,
                linefmt=colors[i % len(colors)] + "-",
                markerfmt=markers[i % len(markers)],
                basefmt=" ",
                label=r["resolution"] + f" (n={r['n_vertices']})",
            )
        ax_stem_g4.set_xlabel("$k$")
        ax_stem_g4.set_ylabel("$\\lambda_k$")
        ax_stem_g4.set_title(f"Genus 4 (quad torus)")
        ax_stem_g4.legend(loc="upper right", fontsize=7)
        ax_stem_g4.grid(True, alpha=0.3)

    # Left bottom: genus 2 stem overlay
    if len(MESHES) > 1 and (MESHES[1][0], MESHES[1][1]) in by_mesh:
        mesh_name, genus = MESHES[1]
        res_rows = by_mesh[(mesh_name, genus)]
        res_rows = sorted(res_rows, key=lambda r: res_order.index(r["resolution"]) if r["resolution"] in res_order else 99)
        x = np.arange(1, K + 1)
        for i, r in enumerate(res_rows):
            lam = [r.get(f"lambda_{j+1}") for j in range(K)]
            lam = [float(v) for v in lam if v != "" and v is not None]
            if not lam:
                continue
            y = np.array(lam[: min(len(x), len(lam))])
            x_k = x[: len(y)]
            ax_stem_g2.stem(
                x_k, y,
                linefmt=colors[i % len(colors)] + "-",
                markerfmt=markers[i % len(markers)],
                basefmt=" ",
                label=r["resolution"] + f" (n={r['n_vertices']})",
            )
        ax_stem_g2.set_xlabel("$k$")
        ax_stem_g2.set_ylabel("$\\lambda_k$")
        ax_stem_g2.set_title("Genus 2 (double torus)")
        ax_stem_g2.legend(loc="upper right", fontsize=7)
        ax_stem_g2.grid(True, alpha=0.3)

    # Right: bar chart — H and mean β₀ at each resolution for BOTH meshes
    labels = []
    H_vals = []
    b0_vals = []
    for mesh_name, genus in mesh_order:
        res_rows = by_mesh[(mesh_name, genus)]
        res_rows = sorted(res_rows, key=lambda r: res_order.index(r["resolution"]) if r["resolution"] in res_order else 99)
        for r in res_rows:
            labels.append(f"g{genus} {r['resolution']}")
            H_vals.append(r.get("spectral_entropy_H") or 0)
            b0_vals.append(r.get("mean_beta0") or 0)
    x_pos = np.arange(len(labels))
    w = 0.35
    ax_bars.bar(x_pos - w / 2, H_vals, w, label="$H$ (spectral entropy)", color="steelblue")
    ax_bars_twin = ax_bars.twinx()
    ax_bars_twin.bar(x_pos + w / 2, b0_vals, w, label=r"mean $\beta_0$", color="coral", alpha=0.8)
    ax_bars.set_xticks(x_pos)
    ax_bars.set_xticklabels(labels, rotation=25, ha="right")
    ax_bars.set_ylabel("Spectral entropy $H$")
    ax_bars_twin.set_ylabel(r"Mean $\beta_0$")
    ax_bars.set_title("Spectral entropy $H$ and mean $\\beta_0$ by mesh and resolution")
    ax_bars.legend(loc="upper left", fontsize=8)
    ax_bars_twin.legend(loc="upper right", fontsize=8)

    plt.savefig(OUT_FIG, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_FIG}", file=sys.stderr)


def print_summary(rows: list[dict]) -> None:
    """Max relative change in eigenvalues; relative change in H and mean_beta0 (coarse vs fine)."""
    by_mesh = {}
    for r in rows:
        key = (r["mesh_name"], r["genus"])
        if key not in by_mesh:
            by_mesh[key] = []
        by_mesh[key].append(r)

    res_order = ["original", "remeshed_10k", "remeshed_20k"]
    print("\n--- Eigenvalue convergence summary ---")
    for (mesh_name, genus), res_rows in by_mesh.items():
        res_rows = sorted(res_rows, key=lambda r: res_order.index(r["resolution"]) if r["resolution"] in res_order else 99)
        if len(res_rows) < 2:
            continue
        coarse = res_rows[0]
        fine = res_rows[-1]
        lam_c = np.array([float(coarse.get(f"lambda_{j+1}") or np.nan) for j in range(K)])
        lam_f = np.array([float(fine.get(f"lambda_{j+1}") or np.nan) for j in range(K)])
        valid = np.isfinite(lam_c) & np.isfinite(lam_f) & (lam_f > 1e-20)
        if np.any(valid):
            rel = np.abs(lam_f[valid] - lam_c[valid]) / lam_f[valid]
            print(f"  {mesh_name} (genus {genus}): max |λ_k^fine - λ_k^coarse|/λ_k^fine = {np.max(rel):.4f} (over {np.sum(valid)} modes)")
        H_c = coarse.get("spectral_entropy_H")
        H_f = fine.get("spectral_entropy_H")
        if H_c is not None and H_f is not None and H_f != 0:
            print(f"    H: relative change (coarse→fine) = {abs(H_f - H_c) / abs(H_f):.4f}")
        b0_c = coarse.get("mean_beta0")
        b0_f = fine.get("mean_beta0")
        if b0_c is not None and b0_f is not None and b0_f != 0:
            print(f"    mean β₀: relative change = {abs(b0_f - b0_c) / abs(b0_f):.4f}")
        # Paper sentence (genus 4: aspect coarse vs fine)
        ar_c = coarse.get("mean_aspect_ratio")
        ar_f = fine.get("mean_aspect_ratio")
        if isinstance(ar_c, (int, float)) and isinstance(ar_f, (int, float)):
            print(f"    Aspect ratio: {ar_c:.1f} → {ar_f:.1f}")
    print("---")
    print("\nSuggested sentence for paper (if stable):")
    print('  "Remeshing genus 4 from ~5k to ~20k vertices reduced mean aspect ratio while changing')
    print('   spectral entropy by less than X%% and mean β₀ by less than Y%%, indicating that the')
    print('   qualitative spectral properties are robust to mesh quality in this range."')
    print("  (Replace X, Y with the relative changes printed above.)")


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
        print("Eigenvalue backend: CPU (CuPy not installed; pip install cupy-cuda12x for GPU).", file=sys.stderr)

    all_rows = []
    for mesh_id, genus in MESHES:
        resolutions = build_resolutions(mesh_id, genus)
        for res_label, V, F in resolutions:
            if not verify_euler(V, F, genus):
                print(f"  [WARN] {mesh_id} {res_label}: Euler V-E+F = {_euler(V, F)} (expected {expected_euler(genus)})", file=sys.stderr)
            row = run_one_resolution(mesh_id, genus, res_label, V, F)
            if row is not None:
                all_rows.append(row)
                print(
                    f"  {mesh_id} {res_label}: n={row['n_vertices']} aspect={row.get('mean_aspect_ratio')} "
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
