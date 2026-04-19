"""
Frame-by-frame β₀ for A3 over 10s: sphere (blue) and torus (orange).
Paper Fig. temporal: figures/fig_temporal_beta0.pdf

Run from project root: python figures/fig_temporal_beta0.py
Requires: data/eigen/sphere.npz, data/eigen/torus.npz, audio A3
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft_frames
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

EIGEN_DIR = root / "data" / "eigen"
OUT_PATH = root / "figures" / "fig_temporal_beta0.pdf"
N_EIGEN = 50
HOP_LENGTH = 512
TARGET_DURATION_SEC = 10.0


def load_mesh_eigen(mesh_name: str):
    path = EIGEN_DIR / f"{mesh_name}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    V = np.asarray(data["vertices"], dtype=np.float64)
    F = np.asarray(data["faces"], dtype=np.int32)
    evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
    evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel() if "eigenvalues" in data.files else None
    if evecs.shape[0] == V.shape[0]:
        evecs = evecs[:, :N_EIGEN].T
    else:
        evecs = evecs[:N_EIGEN, :]
    if evecs.shape[0] < N_EIGEN:
        return None
    evals = evals[:N_EIGEN] if evals is not None and len(evals) >= N_EIGEN else None
    return V, F, evecs, evals


def main():
    sphere_data = load_mesh_eigen("sphere")
    torus_data = load_mesh_eigen("torus")
    if sphere_data is None or torus_data is None:
        print("Need data/eigen/sphere.npz and data/eigen/torus.npz", file=sys.stderr)
        sys.exit(1)

    sig, sr = get_audio("A3")
    frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
    if not frames:
        print("No FFT frames for A3", file=sys.stderr)
        sys.exit(1)

    # Restrict to first 10s
    t_sec_list = []
    beta0_sphere = []
    beta0_torus = []
    max_t = TARGET_DURATION_SEC
    V_s, F_s, evecs_s, evals_s = sphere_data
    V_t, F_t, evecs_t, evals_t = torus_data

    for mag, start in frames:
        t_sec = start / float(sr)
        if t_sec > max_t:
            break
        t_sec_list.append(t_sec)
        coef_s = map_fft_to_coefficients(mag, N_EIGEN, strategy="direct", eigenvalues=evals_s)
        f_s = compute_scalar_field(evecs_s, coef_s)
        m_s = compute_topology_metrics(V_s, F_s, f_s)
        beta0_sphere.append(m_s.beta0)
        coef_t = map_fft_to_coefficients(mag, N_EIGEN, strategy="direct", eigenvalues=evals_t)
        f_t = compute_scalar_field(evecs_t, coef_t)
        m_t = compute_topology_metrics(V_t, F_t, f_t)
        beta0_torus.append(m_t.beta0)

    if not t_sec_list:
        print("No frames in 10s", file=sys.stderr)
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(t_sec_list, beta0_sphere, color="C0", label="sphere", linewidth=1.2)
    ax.plot(t_sec_list, beta0_torus, color="C1", label="torus", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\beta_0$")
    ax.legend(loc="upper right")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(max_t, t_sec_list[-1] if t_sec_list else max_t))
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", OUT_PATH, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
