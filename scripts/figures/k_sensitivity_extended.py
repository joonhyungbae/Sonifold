"""
K-sensitivity extension: sphere mesh + A3(piano), K ∈ {10, 50, 100, 200}.
Load eigenbasis from data/experiments/eigen; if fewer than 200, compute with eigsh.
Measure β₀, A_ratio, per-frame compute time (ms), then save CSV and PDF.

Run from project root: python scripts/k_sensitivity_extended.py
Output: data/results/k_sensitivity_extended.csv, figures/fig_k_sensitivity.pdf
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

EXPERIMENTS_DIR = root / "data" / "experiments"
EIGEN_DIR = root / "data" / "experiments" / "eigen"
RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"
OUT_CSV = RESULTS_DIR / "k_sensitivity_extended.csv"
OUT_PDF = FIGURES_DIR / "fig_k_sensitivity.pdf"

MESH_ID = "sphere_genus0"
AUDIO_ID = "A3"
STRATEGY = "direct"
K_VALUES = [10, 50, 100, 200]
N_EIGEN_MAX = 200


def load_mesh(mesh_id: str):
    """Load mesh vertices and faces from data/experiments/{mesh_id}.obj."""
    import trimesh
    obj_path = EXPERIMENTS_DIR / (mesh_id + ".obj")
    if not obj_path.exists():
        raise FileNotFoundError(obj_path)
    m = trimesh.load(obj_path, force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int32)
    return V, F


def load_eigen(mesh_id: str, n_eigen: int):
    """
    Load eigenbasis from data/experiments/eigen/{mesh_id}.npz or .json.
    Returns (V, F, evals, evecs) with evals/evecs having at most n_eigen modes.
    If file has fewer than n_eigen modes, returns what is available (caller may then compute).
    """
    V, F = load_mesh(mesh_id)
    npz_path = EIGEN_DIR / (mesh_id + ".npz")
    json_path = EIGEN_DIR / (mesh_id + ".json")

    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
        evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
        if "vertices" in data.files and "faces" in data.files:
            V = np.asarray(data["vertices"], dtype=np.float64)
            F = np.asarray(data["faces"], dtype=np.int32)
        n_use = min(len(evals), evecs.shape[0], n_eigen)
        return V, F, evals[:n_use], evecs[:n_use]

    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        evecs = np.array(data["eigenvectors"], dtype=np.float64)
        evals = np.array(data.get("eigenvalues", []), dtype=np.float64)
        if "vertices" in data and "faces" in data:
            V = np.array(data["vertices"], dtype=np.float64)
            F = np.array(data["faces"], dtype=np.int32)
        if len(evals) == 0 and evecs.shape[0] > 0:
            evals = np.zeros(evecs.shape[0], dtype=np.float64)
        n_use = min(evecs.shape[0], len(evals), n_eigen)
        return V, F, evals[:n_use], evecs[:n_use]

    return V, F, np.array([]), np.array([]).reshape(0, V.shape[0])


def compute_eigen_if_needed(mesh_id: str, n_eigen: int):
    """
    Load mesh + eigen; if fewer than n_eigen modes, compute with precompute.eigensolver.
    Returns (V, F, evals, evecs) with at least min(n_eigen, available) modes.
    """
    V, F, evals, evecs = load_eigen(mesh_id, n_eigen)
    if evecs.shape[0] >= n_eigen:
        return V, F, evals[:n_eigen], evecs[:n_eigen]

    try:
        from precompute.eigensolver import compute_eigen
    except Exception as e:
        raise FileNotFoundError(
            f"Need {n_eigen} modes but only {evecs.shape[0]} in {EIGEN_DIR}; eigensolver failed: {e}"
        ) from e

    print(f"Computing {n_eigen} eigenmodes for {mesh_id} (had {evecs.shape[0]})...", file=sys.stderr)
    evals_new, evecs_new = compute_eigen(V, F, N=min(n_eigen, N_EIGEN_MAX))
    return V, F, evals_new, evecs_new


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sig, sr = get_audio(AUDIO_ID)
    mag, _ = compute_fft(sig, sr)

    K_max = max(K_VALUES)
    V, F, evals_full, evecs_full = compute_eigen_if_needed(MESH_ID, K_max)

    rows = []
    for K in K_VALUES:
        if K > evecs_full.shape[0]:
            print(f"Skipping K={K} (only {evecs_full.shape[0]} modes)", file=sys.stderr)
            continue
        evals = evals_full[:K]
        evecs = evecs_full[:K]

        t0 = time.perf_counter()
        coef = map_fft_to_coefficients(mag, K, strategy=STRATEGY, eigenvalues=evals)
        f = compute_scalar_field(evecs, coef)
        m = compute_topology_metrics(V, F, f)
        elapsed = time.perf_counter() - t0

        rows.append({
            "K": K,
            "beta0": m.beta0,
            "A_ratio": m.A_ratio,
            "compute_time_ms": round(elapsed * 1000.0, 4),
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["K", "beta0", "A_ratio", "compute_time_ms"])
        w.writeheader()
        w.writerows(rows)
    print("Saved", OUT_CSV, file=sys.stderr)

    # Figures: K vs beta0, K vs A_ratio, K vs compute_time_ms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    K_list = [r["K"] for r in rows]
    beta0_list = [r["beta0"] for r in rows]
    A_ratio_list = [r["A_ratio"] for r in rows]
    time_ms_list = [r["compute_time_ms"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].plot(K_list, beta0_list, "o-", color="C0", linewidth=2, markersize=8)
    axes[0].set_xlabel("$K$ (number of eigenmodes)")
    axes[0].set_ylabel(r"$\beta_0$")
    axes[0].set_title("Visual complexity")
    axes[0].set_xticks(K_list)

    axes[1].plot(K_list, A_ratio_list, "s-", color="C1", linewidth=2, markersize=8)
    axes[1].set_xlabel("$K$")
    axes[1].set_ylabel("$A_{\\mathrm{ratio}}$")
    axes[1].set_title("Nodal area ratio")
    axes[1].set_xticks(K_list)

    axes[2].plot(K_list, time_ms_list, "^-", color="C2", linewidth=2, markersize=8)
    axes[2].set_xlabel("$K$")
    axes[2].set_ylabel("Time (ms)")
    axes[2].set_title("Compute time per frame")
    axes[2].set_xticks(K_list)

    for ax in axes:
        ax.set_facecolor("white")
    plt.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", OUT_PDF, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
