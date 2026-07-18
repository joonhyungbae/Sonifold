"""
Reconcile the paper's H-vs-beta0 correlations.

The abstract, conclusion, and Section 6 state two Spearman rank correlations for
spectral entropy H against nodal complexity:
  (i)  rho ~ +0.50 (n=7, genus sequence, random-coefficient beta0)
  (ii) rho ~ +0.4  (n=9, direct mapping of A5)

Neither reproduces from the obvious full-set computation (it comes out negative),
so this script brute-forces the plausible degrees of freedom (mesh set, beta0
source, whether the degeneracy-dominated sphere is included) and reports which
configuration, if any, matches each printed value. Run from project root:

  python scripts/analysis/reconcile_H_beta0_correlation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from analysis.spectral_descriptors import spectral_entropy
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

EIGEN = root / "data" / "eigen"
EIGEN_EXP = root / "data" / "experiments" / "eigen"
K = 50
N_RANDOM = 200
SEED = 42

GENUS_SEQ = [  # matched ~5000-vertex budget, the "genus sequence"
    ("sphere_genus0", EIGEN_EXP), ("torus_genus1", EIGEN_EXP),
    ("double_torus_genus2", EIGEN_EXP), ("triple_torus_genus3", EIGEN_EXP),
    ("quad_torus_genus4", EIGEN_EXP), ("penta_torus_genus5", EIGEN_EXP),
    ("hex_torus_genus6", EIGEN_EXP),
]
EXPLORE_9 = [  # the nine exploration meshes (webapp-resolution bases)
    ("sphere", EIGEN), ("torus", EIGEN), ("cube", EIGEN), ("ellipsoid", EIGEN),
    ("double_torus", EIGEN), ("flat_plate", EIGEN), ("tetrahedron", EIGEN),
    ("octahedron", EIGEN), ("icosahedron", EIGEN),
]


def load(name, d):
    z = np.load(d / f"{name}.npz")
    return (z["vertices"].astype(np.float64), z["faces"].astype(np.int32),
            z["eigenvalues"].astype(np.float64)[:K], z["eigenvectors"].astype(np.float64)[:K])


def beta0_random(V, F, evecs, dist, seed=SEED, n=N_RANDOM):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        if dist == "uniform_sphere":
            a = rng.standard_normal(K); a /= np.linalg.norm(a)
        elif dist == "normal_L2":
            a = rng.standard_normal(K); a /= np.linalg.norm(a)
        elif dist == "dirichlet":
            a = rng.dirichlet(np.ones(K)); a -= a.mean()
        else:
            raise ValueError(dist)
        out.append(compute_topology_metrics(V, F, compute_scalar_field(evecs, a)).beta0)
    return float(np.mean(out))


def beta0_audio(V, F, evecs, evals, audio_id, strategy):
    from audio.audio_library import get_audio
    from audio.fft_analysis import compute_fft
    from mapping.spectral_mapping import map_fft_to_coefficients
    sig, sr = get_audio(audio_id)
    mag, _ = compute_fft(sig, sr)
    coef = map_fft_to_coefficients(mag, K, strategy=strategy, eigenvalues=evals)
    return compute_topology_metrics(V, F, compute_scalar_field(evecs, coef)).beta0


def spearman(H, B):
    rho, p = stats.spearmanr(H, B)
    return rho, p


def report(label, names, H, B, target):
    rho, p = spearman(H, B)
    hit = "  <== MATCH" if abs(rho - target) < 0.12 else ""
    print(f"  {label:42s} n={len(names)} rho={rho:+.3f} p={p:.3f}{hit}")
    return rho


def main():
    print("Loading eigenbases and computing H + beta0 sources...\n")

    # genus sequence
    gs = {}
    for name, d in GENUS_SEQ:
        V, F, ev, evec = load(name, d)
        gs[name] = {
            "H": spectral_entropy(ev),
            "rand_us": beta0_random(V, F, evec, "uniform_sphere"),
            "rand_dir": beta0_random(V, F, evec, "dirichlet"),
            "A5d": beta0_audio(V, F, evec, ev, "A5", "direct"),
            "A3d": beta0_audio(V, F, evec, ev, "A3", "direct"),
        }
    # exploration 9
    ex = {}
    for name, d in EXPLORE_9:
        V, F, ev, evec = load(name, d)
        ex[name] = {
            "H": spectral_entropy(ev),
            "rand_us": beta0_random(V, F, evec, "uniform_sphere"),
            "A5d": beta0_audio(V, F, evec, ev, "A5", "direct"),
        }

    print("=== Target (i): rho ~ +0.50, n=7, genus sequence, random-coefficient beta0 ===")
    names = [n for n, _ in GENUS_SEQ]
    H = [gs[n]["H"] for n in names]
    for src in ["rand_us", "rand_dir", "A5d", "A3d"]:
        B = [gs[n][src] for n in names]
        report(f"genus-7 full, {src}", names, H, B, 0.50)
    # sphere-excluded variants (paper frames sphere as the degeneracy exception)
    ns = names[1:]
    Hs = [gs[n]["H"] for n in ns]
    for src in ["rand_us", "rand_dir"]:
        Bs = [gs[n][src] for n in ns]
        report(f"genus-6 (no sphere), {src}", ns, Hs, Bs, 0.50)

    print("\n=== Target (ii): rho ~ +0.4, n=9, direct mapping of A5 ===")
    en = [n for n, _ in EXPLORE_9]
    He = [ex[n]["H"] for n in en]
    for src in ["A5d", "rand_us"]:
        Be = [ex[n][src] for n in en]
        report(f"explore-9 full, {src}", en, He, Be, 0.40)
    # sphere-excluded
    en8 = [n for n in en if n != "sphere"]
    He8 = [ex[n]["H"] for n in en8]
    for src in ["A5d", "rand_us"]:
        Be8 = [ex[n][src] for n in en8]
        report(f"explore-8 (no sphere), {src}", en8, He8, Be8, 0.40)

    print("\nH values (genus seq):",
          {n: round(gs[n]["H"], 3) for n in names})
    print("random-beta0 (genus seq, uniform):",
          {n: round(gs[n]["rand_us"], 1) for n in names})


if __name__ == "__main__":
    main()
