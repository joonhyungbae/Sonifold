"""
Temporal phase analysis: compute frame-wise β₀ over ~10s of audio -> save std etc.
If audio is 1s, tile 10x to form 10s. STFT hop 512, window 2048.

Run from project root: python -m experiment.run_temporal_persistence
Requires: data/experiments/eigen/sphere_genus0.npz (or data/eigen/sphere.json)
Output: data/results/results_temporal.csv
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

from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft_frames
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

EIGEN_EXPERIMENTS = root / "data" / "experiments" / "eigen"
EIGEN_MAIN = root / "data" / "eigen"
RESULTS_DIR = root / "data" / "results"
OUT_CSV = RESULTS_DIR / "results_temporal.csv"
TARGET_DURATION_SEC = 10.0
HOP_LENGTH = 512
N_EIGEN = 50
STRATEGY = "direct"
MESH_IDS = ["sphere_genus0", "torus_genus1"]
AUDIO_IDS = ["A3", "A7"]


def load_eigen(mesh_id: str):
    """Load eigen from experiments/eigen or data/eigen."""
    for base, ext in [(EIGEN_EXPERIMENTS, ".npz"), (EIGEN_EXPERIMENTS, ".json"), (EIGEN_MAIN, ".json")]:
        path = base / (mesh_id + ext)
        if not path.exists():
            # try without _genus0 suffix
            alt_id = mesh_id.replace("_genus0", "").replace("_genus1", "").replace("_genus2", "")
            path_alt = base / (alt_id + ext)
            if path_alt.exists():
                path, mesh_id = path_alt, alt_id
            else:
                continue
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
        return V, F, evals[:N_EIGEN], evecs[:N_EIGEN]
    return None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for audio_id in AUDIO_IDS:
        sig, sr = get_audio(audio_id)
        n_one_sec = len(sig)
        n_target = int(TARGET_DURATION_SEC * sr)
        if len(sig) < n_target:
            # Repeat to get ~10s
            repeats = (n_target + len(sig) - 1) // len(sig)
            sig = np.tile(sig, repeats)[:n_target]
        else:
            sig = sig[:n_target]

        frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
        t_sec = np.array([ start / sr for _, start in frames ])

        for mesh_id in MESH_IDS:
            eigen = load_eigen(mesh_id)
            if eigen is None:
                print("Skip", mesh_id, "(no eigen)", file=sys.stderr)
                continue
            V, F, evals, evecs = eigen
            beta0_list = []
            A_ratio_list = []
            for frame_idx, (mag, start_idx) in enumerate(tqdm(frames, desc=f"{audio_id}_{mesh_id}", leave=False)):
                coef = map_fft_to_coefficients(mag, N_EIGEN, strategy=STRATEGY, eigenvalues=evals)
                f = compute_scalar_field(evecs, coef)
                m = compute_topology_metrics(V, F, f)
                beta0_list.append(m.beta0)
                A_ratio_list.append(m.A_ratio)
                rows.append({
                    "audio_id": audio_id,
                    "mesh_id": mesh_id,
                    "frame": frame_idx,
                    "t_sec": round(start_idx / sr, 4),
                    "beta0": m.beta0,
                    "A_ratio": m.A_ratio,
                })

            b0 = np.array(beta0_list)
            print("{} {}: beta0 mean={:.1f} std={:.1f}".format(audio_id, mesh_id, b0.mean(), b0.std()), file=sys.stderr)

    fieldnames = ["audio_id", "mesh_id", "frame", "t_sec", "beta0", "A_ratio"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Summary stats per (audio_id, mesh_id)
    summary_path = RESULTS_DIR / "results_temporal_summary.csv"
    by_key = {}
    for r in rows:
        k = (r["audio_id"], r["mesh_id"])
        if k not in by_key:
            by_key[k] = []
        by_key[k].append(r["beta0"])
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_id", "mesh_id", "beta0_mean", "beta0_std", "n_frames"])
        for (aid, mid), vals in sorted(by_key.items()):
            v = np.array(vals)
            w.writerow([aid, mid, round(v.mean(), 2), round(v.std(), 2), len(v)])
    print("Saved", OUT_CSV, "and", summary_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
