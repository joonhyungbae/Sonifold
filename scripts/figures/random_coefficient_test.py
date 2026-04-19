"""
Random coefficient test: for each mesh, sample random coefficients a N=1000 times without audio
to obtain β₀ distribution. Measures "mesh-intrinsic β₀ distribution" independently of audio.

Usage:
  python scripts/random_coefficient_test.py

Prerequisites:
  - Eigenbasis for genus sequence mesh in data/experiments/eigen/ or data/eigen/
    (sphere_genus0, torus_genus1, double_torus_genus2, triple_torus_genus3, quad_torus_genus4)

Output:
  - data/results/random_coefficient_beta0.csv
  - data/results/random_coefficient_summary.csv
  - figures/fig_random_coefficient.pdf
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

# Genus sequence: same 5 meshes as results_genus_extended.csv
GENUS_SEQUENCE = [
    ("sphere_genus0", 0),
    ("torus_genus1", 1),
    ("double_torus_genus2", 2),
    ("triple_torus_genus3", 3),
    ("quad_torus_genus4", 4),
]

EIGEN_EXPERIMENTS = root / "data" / "experiments" / "eigen"
EIGEN_MAIN = root / "data" / "eigen"
RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"
OUT_BETA0_CSV = RESULTS_DIR / "random_coefficient_beta0.csv"
OUT_SUMMARY_CSV = RESULTS_DIR / "random_coefficient_summary.csv"
OUT_FIG = FIGURES_DIR / "fig_random_coefficient.pdf"

K = 50
N_TRIALS = 1000

# Distribution names (requirement 3)
DIST_UNIFORM_SPHERE = "uniform_sphere"   # (a) randn then L2 normalize
DIST_DIRICHLET = "dirichlet"              # (b) Dirichlet(1,...,1)
DIST_NORMAL_L2 = "normal_L2"              # (c) Standard normal then L2 normalize


def _mesh_alt_id(mesh_id: str) -> str:
    """experiments mesh_id -> data/eigen short name (sphere, torus, double_torus)."""
    for suffix in ["_genus0", "_genus1", "_genus2", "_genus3", "_genus4"]:
        if mesh_id.endswith(suffix):
            return mesh_id[: -len(suffix)]
    return mesh_id


def load_eigenbasis(mesh_id: str, genus: int):
    """
    Load V, F, evecs (K, n_vertices), evals for mesh.
    Try data/experiments/eigen first, then data/eigen with short name for genus 0,1,2.
    Returns None if not found or fewer than K modes.
    """
    for base, ext in [(EIGEN_EXPERIMENTS, ".npz"), (EIGEN_EXPERIMENTS, ".json")]:
        path = base / (mesh_id + ext)
        if path.exists():
            break
        path = None
    if path is None and genus <= 2:
        alt = _mesh_alt_id(mesh_id)
        for base, ext in [(EIGEN_MAIN, ".npz"), (EIGEN_MAIN, ".json")]:
            path = base / (alt + ext)
            if path.exists():
                break
            path = None
    if path is None:
        return None

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        V = np.asarray(data["vertices"], dtype=np.float64)
        F = np.asarray(data["faces"], dtype=np.int32)
        evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
        evals = (
            np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
            if "eigenvalues" in data.files
            else None
        )
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        V = np.array(data["vertices"], dtype=np.float64)
        F = np.array(data["faces"], dtype=np.int32)
        evecs = np.array(data["eigenvectors"], dtype=np.float64)
        evals = np.array(data.get("eigenvalues", []), dtype=np.float64)

    n_verts = V.shape[0]
    if evecs.shape[0] == n_verts:
        evecs = evecs[:, :K].T
    else:
        evecs = evecs[:K, :]
    if evecs.shape[0] < K:
        return None
    if evals is not None and len(evals) >= K:
        evals = evals[:K]
    else:
        evals = None
    return V, F, evecs, evals


def sample_coefficients(distribution: str, K: int, rng: np.random.Generator) -> np.ndarray:
    """Sample coefficient vector a of length K from given distribution."""
    if distribution == DIST_UNIFORM_SPHERE or distribution == DIST_NORMAL_L2:
        a = rng.standard_normal(K)
        n = np.linalg.norm(a)
        if n < 1e-20:
            a = rng.standard_normal(K)
            n = np.linalg.norm(a)
        return a / n
    if distribution == DIST_DIRICHLET:
        a = rng.dirichlet(np.ones(K))
        return np.asarray(a, dtype=np.float64)
    raise ValueError("Unknown distribution: " + distribution)


def run_one_trial(V, F, evecs, a: np.ndarray) -> int:
    """Compute scalar field f = sum a_k phi_k and return beta0."""
    f = compute_scalar_field(evecs, a)
    m = compute_topology_metrics(V, F, f)
    return m.beta0


def main():
    rng = np.random.default_rng(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows_beta0 = []
    for mesh_name, genus in tqdm(GENUS_SEQUENCE, desc="mesh"):
        out = load_eigenbasis(mesh_name, genus)
        if out is None:
            print(f"Skip {mesh_name}: no eigenbasis with K>={K}", file=sys.stderr)
            continue
        V, F, evecs, _ = out
        evecs = evecs[:K, :]

        for distribution in [DIST_UNIFORM_SPHERE, DIST_DIRICHLET, DIST_NORMAL_L2]:
            for trial in range(N_TRIALS):
                a = sample_coefficients(distribution, K, rng)
                beta0 = run_one_trial(V, F, evecs, a)
                rows_beta0.append({
                    "mesh_name": mesh_name,
                    "genus": genus,
                    "distribution": distribution,
                    "trial": trial,
                    "beta0": beta0,
                })

    with open(OUT_BETA0_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mesh_name", "genus", "distribution", "trial", "beta0"])
        w.writeheader()
        w.writerows(rows_beta0)
    print(f"Saved {OUT_BETA0_CSV}", file=sys.stderr)

    # Summary: mean, std, median, q25, q75 per (mesh_name, genus, distribution)
    summary = []
    from itertools import groupby
    key_fn = lambda r: (r["mesh_name"], r["genus"], r["distribution"])
    sorted_rows = sorted(rows_beta0, key=key_fn)
    for key, group in groupby(sorted_rows, key=key_fn):
        mesh_name, genus, distribution = key
        beta0_arr = np.array([r["beta0"] for r in group], dtype=np.float64)
        summary.append({
            "mesh_name": mesh_name,
            "genus": genus,
            "distribution": distribution,
            "mean_beta0": round(float(np.mean(beta0_arr)), 4),
            "std_beta0": round(float(np.std(beta0_arr)), 4),
            "median_beta0": round(float(np.median(beta0_arr)), 4),
            "q25_beta0": round(float(np.percentile(beta0_arr, 25)), 4),
            "q75_beta0": round(float(np.percentile(beta0_arr, 75)), 4),
        })

    with open(OUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mesh_name", "genus", "distribution",
                "mean_beta0", "std_beta0", "median_beta0", "q25_beta0", "q75_beta0",
            ],
        )
        w.writeheader()
        w.writerows(summary)
    print(f"Saved {OUT_SUMMARY_CSV}", file=sys.stderr)

    # Figure: box plot, x=genus, color=distribution (same style as fig_genus_beta0)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from matplotlib.patches import Patch

    plt.rc("font", family="serif")
    fig, ax = plt.subplots(figsize=(6.5, 4))

    dist_order = [DIST_UNIFORM_SPHERE, DIST_DIRICHLET, DIST_NORMAL_L2]
    dist_labels = {
        "uniform_sphere": r"Uniform($S^{K-1}$)",
        "dirichlet": "Dirichlet(1,…,1)",
        "normal_L2": "Normal L2",
    }
    dist_colors = {"uniform_sphere": "C0", "dirichlet": "C1", "normal_L2": "C2"}

    genera = sorted(set(r["genus"] for r in rows_beta0))
    if not genera:
        print("No data for figure.", file=sys.stderr)
        return 0

    by_genus_dist = defaultdict(list)
    for r in rows_beta0:
        by_genus_dist[(r["genus"], r["distribution"])].append(r["beta0"])

    n_genus = len(genera)
    n_dist = len(dist_order)
    box_width = 0.6 / n_dist
    bp_data = []
    box_positions = []
    for i, g in enumerate(genera):
        for j, dist in enumerate(dist_order):
            data = by_genus_dist.get((g, dist), [])
            if not data:
                continue
            bp_data.append(data)
            pos = i + (j - n_dist / 2 + 0.5) * box_width
            box_positions.append(pos)

    bp = ax.boxplot(
        bp_data,
        positions=box_positions,
        widths=box_width * 0.8,
        patch_artist=True,
        showfliers=False,
    )
    for idx, (g, dist) in enumerate([(g, d) for g in genera for d in dist_order]):
        if idx < len(bp["boxes"]):
            bp["boxes"][idx].set_facecolor(dist_colors.get(dist, "gray"))
            bp["boxes"][idx].set_alpha(0.7)

    ax.set_xticks(range(n_genus))
    ax.set_xticklabels([str(g) for g in genera])
    ax.set_xlabel("Genus")
    ax.set_ylabel(r"$\beta_0$")
    ax.set_title(r"$\beta_0$ distribution: random coefficients (N=1000 per mesh×distribution)")
    legend_elements = [
        Patch(facecolor=dist_colors[d], alpha=0.7, label=dist_labels.get(d, d))
        for d in dist_order
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved {OUT_FIG}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
