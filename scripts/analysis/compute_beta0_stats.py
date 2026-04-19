"""
Compute mean and standard deviation of β₀ over all STFT frames for each
(mesh × audio) combination (direct mapping). Outputs LaTeX table rows for
Table 1 (tab:beta0) and saves data/results/beta0_stats.csv. Also prints summary
statistics for Section 5.5 (temporal persistence) verification.

Usage:
  python scripts/analysis/compute_beta0_stats.py

Prerequisites:
  - Precomputed eigenbases in data/eigen/ (e.g. sphere.npz, torus.npz, ...)
  - Audio files in data/audio/ for A3, A4, A7 (or placeholders from audio_library)

Output:
  - data/results/beta0_stats.csv (columns: mesh, audio, mean, std, n_frames)
  - LaTeX table snippet printed to stdout (copy into paper.tex Table 1)
  - Section 5.5 verification: mean ± std for (sphere,A3), (torus,A3), (sphere,A7), (torus,A7)

Uses existing repository code: audio_library, fft_analysis, spectral_mapping,
scalar_field, nodal_surface (see imports below).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio, list_audio_ids
from audio.fft_analysis import compute_fft_frames
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

EIGEN_DIR = root / "data" / "eigen"
RESULTS_DIR = root / "data" / "results"
OUT_CSV = RESULTS_DIR / "beta0_stats.csv"
N_EIGEN = 50
STRATEGY = "direct"
HOP_LENGTH = 512

# Row order and display names for paper.tex Table 1
MESH_ORDER = [
    "sphere",
    "torus",
    "cube",
    "ellipsoid",
    "double_torus",
    "flat_plate",
    "tetrahedron",
    "octahedron",
    "icosahedron",
]
MESH_DISPLAY = {
    "sphere": "sphere",
    "torus": "torus",
    "cube": "cube",
    "ellipsoid": "ellipsoid",
    "double_torus": "double torus",
    "flat_plate": "flat plate",
    "tetrahedron": "tetrahedron",
    "octahedron": "octahedron",
    "icosahedron": "icosahedron",
}
AUDIO_COLS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]


def load_mesh_eigen(mesh_name: str):
    """Load vertices, faces, eigenvectors (first N_EIGEN), eigenvalues from data/eigen/<mesh>.npz.
    Expects eigenvectors shape (N, V) from precompute; uses first N_EIGEN rows.
    """
    path = EIGEN_DIR / f"{mesh_name}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    V = np.asarray(data["vertices"], dtype=np.float64)
    F = np.asarray(data["faces"], dtype=np.int32)
    evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
    evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel() if "eigenvalues" in data.files else None
    # Precompute stores (N, V); ensure we have (N_EIGEN, n_vertices)
    if evecs.shape[0] == V.shape[0]:
        evecs = evecs[:, :N_EIGEN].T
    else:
        evecs = evecs[:N_EIGEN, :]
    if evecs.shape[0] < N_EIGEN:
        return None
    evals = evals[:N_EIGEN] if evals is not None and len(evals) >= N_EIGEN else None
    return V, F, evecs, evals


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stats = []
    for mesh_name in tqdm(MESH_ORDER, desc="mesh"):
        eigen = load_mesh_eigen(mesh_name)
        if eigen is None:
            print(f"Skip {mesh_name}: no data/eigen/{mesh_name}.npz", file=sys.stderr)
            for audio_id in list_audio_ids():
                stats.append({"mesh": mesh_name, "audio": audio_id, "mean": None, "std": None, "n_frames": 0})
            continue

        V, F, evecs, evals = eigen

        for audio_id in list_audio_ids():
            sig, sr = get_audio(audio_id)
            frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
            if not frames:
                stats.append({"mesh": mesh_name, "audio": audio_id, "mean": None, "std": None, "n_frames": 0})
                continue

            beta0_list = []
            for mag, _ in frames:
                coef = map_fft_to_coefficients(mag, N_EIGEN, strategy=STRATEGY, eigenvalues=evals)
                f = compute_scalar_field(evecs, coef)
                m = compute_topology_metrics(V, F, f)
                beta0_list.append(m.beta0)

            arr = np.array(beta0_list, dtype=np.float64)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr)) if len(arr) > 1 else 0.0
            stats.append({
                "mesh": mesh_name,
                "audio": audio_id,
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "n_frames": len(arr),
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mesh", "audio", "mean", "std", "n_frames"])
        w.writeheader()
        w.writerows(stats)
    print(f"Saved {OUT_CSV}", file=sys.stderr)

    # LaTeX tabular rows for copy-paste into paper.tex (compact: one decimal)
    print("\n% --- LaTeX tabular body (replace Table 1 body with the following) ---\n", file=sys.stderr)
    by_mesh = {}
    for r in stats:
        key = (r["mesh"], r["audio"])
        by_mesh[key] = (r["mean"], r["std"])

    for mesh in MESH_ORDER:
        label = MESH_DISPLAY[mesh]
        cells = []
        for a in AUDIO_COLS:
            mean, std = by_mesh.get((mesh, a), (None, None))
            if mean is None:
                cells.append("--")
            else:
                cells.append(f"${mean:.1f}\\pm{std:.1f}$")
        line = f"{label:12} & " + " & ".join(cells) + " \\\\"
        print(line)
    print("", file=sys.stderr)

    # Section 5.5 (temporal persistence) verification: same stats paper cites
    section55_pairs = [("sphere", "A3"), ("torus", "A3"), ("sphere", "A7"), ("torus", "A7")]
    print("", file=sys.stderr)
    print("% --- Section 5.5 (sec:temporal): replace paper numbers with below ---", file=sys.stderr)
    for mesh_id, audio_id in section55_pairs:
        key = (mesh_id, audio_id)
        if key in by_mesh:
            mean, std = by_mesh[key]
            print(f"  {mesh_id} {audio_id}: mean beta0 = {mean:.1f}, sigma = {std:.1f}", file=sys.stderr)
    print("", file=sys.stderr)
    # Also print to stdout for copy-paste into paper.tex Section 5.5
    print("\n% Section 5.5 verification (paste into sec:temporal if updating numbers):")
    for mesh_id, audio_id in section55_pairs:
        key = (mesh_id, audio_id)
        if key in by_mesh:
            mean, std = by_mesh[key]
            print(f"%   {mesh_id} {audio_id}: mean $\\beta_0$ = {mean:.1f}, $\\sigma(\\beta_0)$ = {std:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
