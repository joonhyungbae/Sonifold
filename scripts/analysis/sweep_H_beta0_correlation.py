"""
Exhaustive sweep for the paper's rho ~ +0.50 (n=7) and +0.4 (n=9) H-beta0 claims.

Tries every descriptor, every beta0 source, every mesh subset, and H over several
truncation levels K, then prints any configuration whose Spearman rho lands in a
positive window around the printed values. If nothing matches, the printed numbers
are not reproducible from the committed eigenbases under any of these degrees of
freedom. Run from project root:

  python scripts/analysis/sweep_H_beta0_correlation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from analysis.spectral_descriptors import (
    spectral_entropy, effective_multiplicity, eigenvalue_density, spectral_gap_ratio)
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

EIGEN = root / "data" / "eigen"
EIGEN_EXP = root / "data" / "experiments" / "eigen"
N_RANDOM = 200
SEED = 42

GENUS_SEQ = ["sphere_genus0", "torus_genus1", "double_torus_genus2",
             "triple_torus_genus3", "quad_torus_genus4", "penta_torus_genus5",
             "hex_torus_genus6"]
EXPLORE_9 = ["sphere", "torus", "cube", "ellipsoid", "double_torus",
             "flat_plate", "tetrahedron", "octahedron", "icosahedron"]


def load_full(name, d):
    z = np.load(d / f"{name}.npz")
    return (z["vertices"].astype(np.float64), z["faces"].astype(np.int32),
            z["eigenvalues"].astype(np.float64), z["eigenvectors"].astype(np.float64))


def H_eigengap(ev, K):
    return spectral_entropy(ev[:K])


def H_eigenvalue(ev, K):
    """Alternative H: Shannon entropy of the normalized eigenvalues themselves."""
    e = np.maximum(ev[:K].astype(np.float64), 1e-20)
    p = e / e.sum()
    return float(-np.sum(p * np.log(p)))


def beta0_random(V, F, evecs, K, seed=SEED, n=N_RANDOM):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = rng.standard_normal(K); a /= np.linalg.norm(a)
        out.append(compute_topology_metrics(V, F, compute_scalar_field(evecs[:K], a)).beta0)
    return float(np.mean(out))


def beta0_audio(V, F, evecs, ev, K, audio_id, strategy):
    from audio.audio_library import get_audio
    from audio.fft_analysis import compute_fft
    from mapping.spectral_mapping import map_fft_to_coefficients
    sig, sr = get_audio(audio_id)
    mag, _ = compute_fft(sig, sr)
    coef = map_fft_to_coefficients(mag, K, strategy=strategy, eigenvalues=ev[:K])
    return compute_topology_metrics(V, F, compute_scalar_field(evecs[:K], coef)).beta0


def descriptors(ev, K):
    cl, msz = effective_multiplicity(ev[:K], eps=1e-3 * float(ev[:K].mean()))
    return {
        "H_gap": H_eigengap(ev, K),
        "H_eval": H_eigenvalue(ev, K),
        "Meff_clusters": cl,
        "Meff_meansize": msz,
        "density": eigenvalue_density(ev[:K]),
        "gap": spectral_gap_ratio(ev[:K]),
    }


def main():
    matches = []
    for setname, meshes, ddir, target, beta_sources in [
        ("genus-7", GENUS_SEQ, EIGEN_EXP, 0.50, ["rand", "A5d", "A3d"]),
        ("explore-9", EXPLORE_9, EIGEN, 0.40, ["rand", "A5d"]),
    ]:
        # load once, compute beta0 sources at K=50 (audio) and descriptors at several K
        data = {}
        for m in meshes:
            V, F, ev, evec = load_full(m, ddir)
            data[m] = {"V": V, "F": F, "ev": ev, "evec": evec}
            data[m]["rand"] = beta0_random(V, F, evec, 50)
            data[m]["A5d"] = beta0_audio(V, F, evec, ev, 50, "A5", "direct")
            data[m]["A3d"] = beta0_audio(V, F, evec, ev, 50, "A3", "direct")

        print(f"\n=== {setname} (target rho ~ +{target:.2f}) ===")
        for K in [10, 50, 100, 200]:
            for m in meshes:
                data[m][f"desc{K}"] = descriptors(data[m]["ev"], K)
            for dname in ["H_gap", "H_eval", "Meff_clusters", "Meff_meansize", "density", "gap"]:
                D = [data[m][f"desc{K}"][dname] for m in meshes]
                if any(np.isnan(D)):
                    continue
                for bsrc in beta_sources:
                    B = [data[m][bsrc] for m in meshes]
                    for subset_name, idx in [("all", list(range(len(meshes)))),
                                             ("no-sphere", list(range(1, len(meshes))))]:
                        Dd = [D[i] for i in idx]; Bb = [B[i] for i in idx]
                        if len(set(Dd)) < 2:
                            continue
                        rho, p = stats.spearmanr(Dd, Bb)
                        if abs(rho - target) < 0.10:
                            tag = f"{setname} K={K} {dname} vs beta0[{bsrc}] {subset_name}"
                            matches.append((tag, rho, p, len(idx)))
                            print(f"  MATCH  {tag:52s} rho={rho:+.3f} p={p:.3f} n={len(idx)}")

    print("\n" + "=" * 60)
    if matches:
        print(f"{len(matches)} configuration(s) reproduce the printed value.")
    else:
        print("NO configuration reproduces the printed +0.50 / +0.4 for ANY")
        print("descriptor, beta0 source, K, or sphere-inclusion choice.")
        print("The printed correlations do not come from the committed eigenbases.")


if __name__ == "__main__":
    main()
