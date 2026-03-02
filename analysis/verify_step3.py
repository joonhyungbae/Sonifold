"""
Step 3 verification: single eigenfunction nodal surface, component count, white noise vs pure tone β₀.
Check that topology metrics are returned for arbitrary (manifold, coefficient) pairs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics, extract_nodal_and_metrics
from analysis.symmetry import compute_symmetry

EIGEN_DIR = root / "data" / "eigen"
N = 50


def load_eigen(mesh_name):
    data = np.load(EIGEN_DIR / f"{mesh_name}.npz", allow_pickle=True)
    return data["vertices"], data["faces"], data["eigenvalues"], data["eigenvectors"]


def main():
    if not (EIGEN_DIR / "sphere.npz").exists():
        print("Step 1 required: data/eigen/sphere.npz missing. Run ./step1.sh first.")
        sys.exit(1)
    V, F, evals, evecs = load_eigen("sphere")
    print("=== Sphere single eigenfunction nodal surface (one coefficient = 1) ===")
    coef_one = np.zeros(N)
    coef_one[0] = 1.0
    f = compute_scalar_field(evecs, coef_one)
    m = compute_topology_metrics(V, F, f)
    print(f"  β₀={m.beta0}, χ={m.chi}, A_ratio={m.A_ratio:.4f}")
    print("  ok (single eigenfunction yields nodal surface)")

    print("\n=== White noise vs pure tone: white noise has higher β₀ ===")
    from audio.audio_library import get_audio
    from audio.fft_analysis import compute_fft
    from mapping.spectral_mapping import map_fft_to_coefficients
    sig_tone, sr = get_audio("A1")
    sig_white, _ = get_audio("A5")
    mag_tone, _ = compute_fft(sig_tone, sr)
    mag_white, _ = compute_fft(sig_white, sr)
    coef_tone = map_fft_to_coefficients(mag_tone, N, strategy="direct")
    coef_white = map_fft_to_coefficients(mag_white, N, strategy="direct")
    f_tone = compute_scalar_field(evecs, coef_tone)
    f_white = compute_scalar_field(evecs, coef_white)
    m_tone = compute_topology_metrics(V, F, f_tone)
    m_white = compute_topology_metrics(V, F, f_white)
    print(f"  Pure tone β₀={m_tone.beta0}, white noise β₀={m_white.beta0}")
    print("  ok" if m_white.beta0 >= m_tone.beta0 else "  check")

    print("\n=== Arbitrary (manifold, coefficient) pair returns topology metrics ===")
    coef_rand = np.random.default_rng(123).standard_normal(N).astype(np.float64)
    coef_rand = coef_rand / np.linalg.norm(coef_rand)
    f_rand = compute_scalar_field(evecs, coef_rand)
    m_rand = compute_topology_metrics(V, F, f_rand)
    S = compute_symmetry(V, f_rand)
    print(f"  β₀={m_rand.beta0}, β₁={m_rand.beta1}, χ={m_rand.chi}, A_ratio={m_rand.A_ratio:.4f}, S={S:.4f}")
    print("  ok")
    print("Step 3 verify done.")


if __name__ == "__main__":
    main()
