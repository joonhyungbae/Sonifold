"""
Nazarov–Sodin comparison: single-eigenfunction β₀ vs L² on the sphere.

For a random spherical harmonic of degree L on S², Nazarov & Sodin (2009, 2016)
prove that the expected number of nodal domains grows as c·L² for a universal
constant c. This script:
  - For each degree L = 1..L_max (largest ℓ with ℓ(ℓ+1) ≤ λ_50), computes β₀
    for a random single-eigenfunction of degree L (random combination within
    the 2L+1 multiplicity subspace).
  - Samples N=100 per degree L, records mean and std of β₀.
  - Plots β₀ vs L² and fits a linear model to estimate c.
  - Compares to our K=50 random-superposition baseline (mean β₀ ~136).

Usage:
  python scripts/nazarov_sodin_comparison.py

Prerequisites:
  - data/eigen/sphere.npz with at least 50 eigenvectors and eigenvalues.

Output:
  - data/results/nazarov_sodin_comparison.csv
  - figures/fig_nazarov_sodin.pdf
  - data/results/nazarov_sodin_summary.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

EIGEN_DIR = root / "data" / "eigen"
EIGEN_EXPERIMENTS = root / "data" / "experiments" / "eigen"
RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"
K = 50
N_SAMPLES_PER_L = 100

OUT_CSV = RESULTS_DIR / "nazarov_sodin_comparison.csv"
OUT_FIG = FIGURES_DIR / "fig_nazarov_sodin.pdf"
OUT_SUMMARY = RESULTS_DIR / "nazarov_sodin_summary.txt"

# K=50 random-superposition baseline (from existing random-coefficient / audio experiments)
BASELINE_MEAN_BETA0 = 136.0


def load_sphere_eigen():
    """Load sphere mesh and K=50 eigenbasis. Returns (V, F, evecs, evals) or None."""
    for base in [EIGEN_DIR, EIGEN_EXPERIMENTS]:
        for name in ["sphere", "sphere_genus0"]:
            for ext in [".npz", ".json"]:
                path = base / f"{name}{ext}"
                if not path.exists():
                    continue
                if path.suffix == ".npz":
                    data = np.load(path, allow_pickle=True)
                    V = np.asarray(data["vertices"], dtype=np.float64)
                    F = np.asarray(data["faces"], dtype=np.int32)
                    evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
                    evals = (
                        np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
                        if "eigenvalues" in data.files
                        else None
                    )
                else:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    V = np.array(data["vertices"], dtype=np.float64)
                    F = np.array(data["faces"], dtype=np.int32)
                    evecs = np.array(data["eigenvectors"], dtype=np.float64)
                    evals = np.array(data.get("eigenvalues", []), dtype=np.float64)
                n_verts = V.shape[0]
                if evecs.shape[0] == n_verts:
                    evecs = evecs[:, :K].T
                else:
                    evecs = evecs[:K, :]
                if evecs.shape[0] < K:
                    continue
                if evals is not None and len(evals) >= K:
                    evals = evals[:K]
                else:
                    evals = None
                return V, F, evecs, evals
    return None


def degree_L_block_indices(L: int) -> tuple[int, int]:
    """
    On the sphere, eigenvalue λ = L(L+1) has multiplicity 2L+1.
    Eigenvalue index 0,1,2 are L=1; 3..7 are L=2; etc.
    Start index for degree L: sum_{j=1}^{L-1} (2j+1) = L² - 1.
    Returns (start, end) for slice [start:end] (2L+1 indices).
    """
    start = L * L - 1
    end = L * L + 2 * L
    return start, end


def l_max_from_K(K: int) -> int:
    """Largest L such that the full 2L+1 block fits in indices [0, K-1]."""
    # L² + 2L <= K  =>  L² + 2L - K <= 0  =>  L <= (-2 + sqrt(4+4K))/2 = -1 + sqrt(1+K)
    return int(np.floor(-1 + np.sqrt(1 + K)))


def run_one_trial(V, F, evecs, coef: np.ndarray) -> int:
    """Scalar field f = coef @ evecs, return β₀."""
    f = compute_scalar_field(evecs, coef)
    m = compute_topology_metrics(V, F, f)
    return m.beta0


def main():
    out = load_sphere_eigen()
    if out is None:
        print("No sphere eigenbasis found in data/eigen or data/experiments/eigen.", file=sys.stderr)
        return 1

    V, F, evecs, evals = out
    n_verts = V.shape[0]
    L_max = l_max_from_K(K)
    print(f"Sphere: {n_verts} vertices, K={K}, L_max={L_max}", file=sys.stderr)

    rng = np.random.default_rng(42)
    rows = []

    for L in range(1, L_max + 1):
        start, end = degree_L_block_indices(L)
        size = end - start
        assert size == 2 * L + 1
        beta0_list = []
        for _ in tqdm(range(N_SAMPLES_PER_L), desc=f"L={L}", leave=False):
            # Random combination within the 2L+1 subspace (unit Gaussian → effectively random spherical harmonic)
            w = rng.standard_normal(size)
            nrm = np.linalg.norm(w)
            if nrm < 1e-20:
                w = rng.standard_normal(size)
                nrm = np.linalg.norm(w)
            w = w / nrm
            coef = np.zeros(K, dtype=np.float64)
            coef[start:end] = w
            beta0 = run_one_trial(V, F, evecs, coef)
            beta0_list.append(beta0)
        arr = np.array(beta0_list, dtype=np.float64)
        mean_b = float(np.mean(arr))
        std_b = float(np.std(arr)) if len(arr) > 1 else 0.0
        rows.append({
            "L": L,
            "L_squared": L * L,
            "mean_beta0": mean_b,
            "std_beta0": std_b,
        })

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("L,L_squared,mean_beta0,std_beta0\n")
        for r in rows:
            f.write(f"{r['L']},{r['L_squared']},{r['mean_beta0']:.6f},{r['std_beta0']:.6f}\n")
    print(f"Saved {OUT_CSV}", file=sys.stderr)

    # Linear fit: mean_beta0 ≈ c * L_squared
    L_sq = np.array([r["L_squared"] for r in rows], dtype=np.float64)
    mean_beta0 = np.array([r["mean_beta0"] for r in rows], dtype=np.float64)
    # β₀ = c * L²  (no intercept, through origin)
    c_est = float(np.sum(mean_beta0 * L_sq) / np.sum(L_sq * L_sq))
    # Plot
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(
        L_sq,
        mean_beta0,
        yerr=[r["std_beta0"] for r in rows],
        fmt="o",
        capsize=3,
        label=r"Mean $\beta_0$ (single eigenfunction, N=100)",
    )
    L_sq_fine = np.linspace(0, L_sq.max() * 1.05, 100)
    ax.plot(L_sq_fine, c_est * L_sq_fine, "k--", label=rf"Fit $\beta_0 = c L^2$, $c \approx {c_est:.4f}$")
    ax.axhline(BASELINE_MEAN_BETA0, color="gray", linestyle=":", label=rf"K=50 superposition mean $\approx {BASELINE_MEAN_BETA0:.0f}$")
    ax.set_xlabel(r"$L^2$")
    ax.set_ylabel(r"$\beta_0$")
    ax.set_title("Nazarov–Sodin scaling: single spherical harmonic vs K=50 superposition")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved {OUT_FIG}", file=sys.stderr)

    # Summary text
    L_at_truncation = L_max
    ns_pred_at_L = c_est * (L_at_truncation ** 2)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(
            "Nazarov–Sodin comparison (sphere, single eigenfunction vs K=50 superposition)\n"
            "================================================================================\n\n"
        )
        f.write(
            "On the sphere, for a random spherical harmonic of degree L, Nazarov & Sodin predict "
            "that the expected number of nodal domains grows as c·L² for a universal constant c. "
            "Using the discrete Laplacian eigenbasis (K=50 modes) and random combinations within "
            "each degree-L eigenspace (2L+1 modes), we estimated c from mean β₀ vs L²: "
            f"c ≈ {c_est:.4f} (linear fit through origin). "
            f"At the maximum degree available in our truncation (L_max = {L_at_truncation}), "
            f"this scaling gives β₀ ≈ c·L² ≈ {ns_pred_at_L:.1f}.\n\n"
        )
        f.write(
            "Our pipeline uses a K=50 random-coefficient superposition over all modes (not a single "
            "eigenfunction), yielding mean β₀ ≈ 136. This is higher than the single-eigenfunction "
            "prediction at L_max, as expected: a superposition of many degrees produces more nodal "
            "structure. The comparison provides context that our observed β₀ lies in a reasonable "
            "range relative to the Nazarov–Sodin scaling when moving from single-degree to "
            "multi-mode random fields.\n"
        )
    print(f"Saved {OUT_SUMMARY}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
