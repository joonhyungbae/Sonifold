"""
Shared spectral descriptors and beta0 stats for scripts/figures and scripts/analysis.
Used by eigenvalue_multiplicity_analysis, spectral_entropy_conjecture_test,
extended_mesh_conjecture_test, eigenvalue_convergence_test*, epsilon_robustness_analysis.
"""
from __future__ import annotations

import numpy as np

# 9 exploration meshes (name, genus) + 4 genus 3–6 meshes. Paths stay in callers.
MESHES_9 = [
    ("sphere", 0),
    ("torus", 1),
    ("cube", 0),
    ("ellipsoid", 0),
    ("double_torus", 2),
    ("flat_plate", 0),
    ("tetrahedron", 0),
    ("octahedron", 0),
    ("icosahedron", 0),
]
MESHES_GENUS_36 = [
    ("triple_torus_genus3", 3),
    ("quad_torus_genus4", 4),
    ("penta_torus_genus5", 5),
    ("hex_torus_genus6", 6),
]


def effective_multiplicity(ev: np.ndarray, eps: float) -> tuple[int, float]:
    """Cluster consecutive eigenvalues with diff < eps. Return (n_clusters, mean_cluster_size)."""
    n = len(ev)
    if n < 2:
        return 1, 1.0
    gaps = np.diff(ev)
    clusters = 1
    current_size = 1
    sizes = []
    for g in gaps:
        if g < eps:
            current_size += 1
        else:
            sizes.append(current_size)
            current_size = 1
            clusters += 1
    sizes.append(current_size)
    return clusters, float(np.mean(sizes)) if sizes else 1.0


def spectral_entropy(ev: np.ndarray) -> float:
    """Shannon entropy of normalized eigenvalue gaps (Δλ_k / Σ Δλ). High = evenly spaced."""
    n = len(ev)
    if n < 2:
        return 0.0
    gaps = np.diff(ev).astype(np.float64)
    gaps = np.maximum(gaps, 1e-20)
    p = gaps / gaps.sum()
    H = -np.sum(p * np.log(p))
    return float(H)


def spectral_gap_ratio(ev: np.ndarray) -> float:
    """γ = λ_2 / λ_1."""
    if len(ev) < 2 or ev[0] <= 0:
        return np.nan
    return float(ev[1] / ev[0])


def eigenvalue_density(ev: np.ndarray) -> float:
    """ρ = (K-1) / (λ_K - λ_1)."""
    if len(ev) < 2:
        return np.nan
    span = ev[-1] - ev[0]
    if span <= 0:
        return np.nan
    return float((len(ev) - 1) / span)


def sample_uniform_sphere(k: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-norm random vector in R^k (uniform on sphere)."""
    a = rng.standard_normal(k)
    n = np.linalg.norm(a)
    if n < 1e-20:
        a = rng.standard_normal(k)
        n = np.linalg.norm(a)
    return (a / n).astype(np.float64)


def compute_beta0_stats(V, F, evecs, k: int, n_trials: int, rng: np.random.Generator):
    """evecs (k, n_verts). Returns (mean_beta0, std_beta0)."""
    from analysis.scalar_field import compute_scalar_field
    from analysis.nodal_surface import compute_topology_metrics

    beta0s = []
    for _ in range(n_trials):
        coef = sample_uniform_sphere(k, rng)
        f = compute_scalar_field(evecs, coef)
        m = compute_topology_metrics(V, F, f)
        beta0s.append(m.beta0)
    arr = np.array(beta0s, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0
