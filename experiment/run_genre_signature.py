"""
Spectral Signature by audio genre: compute entropy of coefficient vector a for mesh × audio.
H(a) = -Σ p_k log(p_k), p_k = a_k^2 / Σ a_j^2 (energy ratio).

Run from project root: python -m experiment.run_genre_signature
Requires: data/experiments/eigen/*.npz or data/eigen/*.json (sphere, torus, double_torus, etc.)
Output: data/results/results_genre_signature.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio, list_audio_ids
from audio.fft_analysis import compute_fft
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

EIGEN_EXPERIMENTS = root / "data" / "experiments" / "eigen"
EIGEN_MAIN = root / "data" / "eigen"
RESULTS_DIR = root / "data" / "results"
OUT_CSV = RESULTS_DIR / "results_genre_signature.csv"
N_EIGEN = 50
STRATEGY = "direct"


def entropy_of_coefficients(coef: np.ndarray) -> float:
    """Normalise to distribution (energy): p_k = a_k^2 / sum(a^2), then H = -sum(p*log(p))."""
    a = np.asarray(coef, dtype=np.float64).ravel()
    s = (a ** 2).sum()
    if s < 1e-20:
        return 0.0
    p = (a ** 2) / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-20)))


def load_eigen(mesh_id: str):
    for base, ext in [(EIGEN_EXPERIMENTS, ".npz"), (EIGEN_EXPERIMENTS, ".json"), (EIGEN_MAIN, ".json")]:
        path = base / (mesh_id + ext)
        if not path.exists():
            alt_id = mesh_id.replace("_genus0", "").replace("_genus1", "").replace("_genus2", "")
            path = base / (alt_id + ext)
            if not path.exists():
                continue
            mesh_id = alt_id
        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            V = np.asarray(data["vertices"], dtype=np.float64)
            F = np.asarray(data["faces"], dtype=np.int32)
            evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
            evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
        else:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            V = np.array(data["vertices"], dtype=np.float64)
            F = np.array(data["faces"], dtype=np.int32)
            evecs = np.array(data["eigenvectors"], dtype=np.float64)
            evals = np.array(data.get("eigenvalues", []), dtype=np.float64)
        if evecs.shape[0] < N_EIGEN:
            continue
        return V, F, evals[:N_EIGEN], evecs[:N_EIGEN], mesh_id
    return None


def main():
    mesh_ids = set()
    for base in [EIGEN_EXPERIMENTS, EIGEN_MAIN]:
        if base.exists():
            for p in list(base.glob("*.npz")) + list(base.glob("*.json")):
                mesh_ids.add(p.stem)
    mesh_ids = sorted(mesh_ids)
    if not mesh_ids:
        print("No eigen data in", EIGEN_EXPERIMENTS, "or", EIGEN_MAIN, file=sys.stderr)
        sys.exit(1)

    audio_ids = list_audio_ids()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for mesh_id in tqdm(mesh_ids, desc="mesh"):
        eigen = load_eigen(mesh_id)
        if eigen is None:
            continue
        V, F, evals, evecs, mesh_id_use = eigen
        for audio_id in audio_ids:
            sig, sr = get_audio(audio_id)
            mag, _ = compute_fft(sig, sr)
            coef = map_fft_to_coefficients(mag, N_EIGEN, strategy=STRATEGY, eigenvalues=evals)
            ent = entropy_of_coefficients(coef)
            f = compute_scalar_field(evecs, coef)
            m = compute_topology_metrics(V, F, f)
            rows.append({
                "mesh_id": mesh_id_use,
                "audio_id": audio_id,
                "entropy": round(ent, 4),
                "beta0": m.beta0,
                "A_ratio": round(m.A_ratio, 4),
            })

    fieldnames = ["mesh_id", "audio_id", "entropy", "beta0", "A_ratio"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Saved", OUT_CSV, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
