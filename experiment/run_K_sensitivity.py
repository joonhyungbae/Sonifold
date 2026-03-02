"""
Sensitivity to number of eigenfunctions K (visual resolution / performance).
Fixed mesh (sphere_genus0) + fixed audio (A3), K ∈ {10, 50, 100, 200}: measure β₀, A_ratio, compute time.

Run from project root: python -m experiment.run_K_sensitivity
Requires: data/experiments/eigen/sphere_genus0.npz (if fewer than 50, this script computes N=200)
Output: data/results/results_K_sensitivity.csv
"""
from __future__ import annotations

import csv
import time
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

EXPERIMENTS_DIR = root / "data" / "experiments"
EIGEN_DIR = root / "data" / "experiments" / "eigen"
RESULTS_DIR = root / "data" / "results"
OUT_CSV = RESULTS_DIR / "results_K_sensitivity.csv"
MESH_ID = "sphere_genus0"
AUDIO_ID = "A3"
STRATEGY = "direct"
K_VALUES = [10, 50, 100, 200]
N_EIGEN_MAX = 200


def load_mesh_and_eigen(mesh_id: str, n_eigen: int):
    """Load mesh; load eigen from disk if available with >= n_eigen, else compute up to n_eigen."""
    import trimesh
    obj_path = EXPERIMENTS_DIR / (mesh_id + ".obj")
    if not obj_path.exists():
        raise FileNotFoundError(obj_path)
    m = trimesh.load(obj_path, force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int32)

    npz_path = EIGEN_DIR / (mesh_id + ".npz")
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
        evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
        n_use = min(evecs.shape[0], n_eigen)
        return V, F, evals[:n_use], evecs[:n_use]
    # No precomputed eigen: compute on the fly (requires gpytoolbox)
    try:
        from precompute.eigensolver import compute_eigen
    except Exception as e:
        raise FileNotFoundError(
            "No eigen data at {} and eigensolver unavailable: {}".format(npz_path, e)
        ) from e
    print("Computing", n_eigen, "eigenmodes for", mesh_id, "...", file=sys.stderr)
    evals, evecs = compute_eigen(V, F, N=min(n_eigen, N_EIGEN_MAX))
    return V, F, evals, evecs


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sig, sr = get_audio(AUDIO_ID)
    mag, _ = compute_fft(sig, sr)

    rows = []
    K_max = max(K_VALUES)
    V, F, evals_full, evecs_full = load_mesh_and_eigen(MESH_ID, K_max)

    for K in K_VALUES:
        if K > evecs_full.shape[0]:
            print("Skipping K={} (only {} modes available)".format(K, evecs_full.shape[0]), file=sys.stderr)
            continue
        evals = evals_full[:K]
        evecs = evecs_full[:K]
        coef = map_fft_to_coefficients(mag, K, strategy=STRATEGY, eigenvalues=evals)
        t0 = time.perf_counter()
        f = compute_scalar_field(evecs, coef)
        m = compute_topology_metrics(V, F, f)
        elapsed = time.perf_counter() - t0
        rows.append({
            "K": K,
            "mesh_id": MESH_ID,
            "audio_id": AUDIO_ID,
            "beta0": m.beta0,
            "A_ratio": m.A_ratio,
            "time_sec": round(elapsed, 6),
        })

    fieldnames = ["K", "mesh_id", "audio_id", "beta0", "A_ratio", "time_sec"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Saved", OUT_CSV, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
