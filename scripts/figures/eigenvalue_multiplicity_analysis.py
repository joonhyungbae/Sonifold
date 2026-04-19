"""
Eigenvalue multiplicity structure analysis for Conjecture 5.1.

For each of 13 meshes (9 exploration + genus 3–6), loads first K=50 eigenvalues,
computes spectral descriptors (effective multiplicity, spectral entropy, gap, density,
product structure score), outputs CSVs and correlation with β₀, and generates figures.

Run from project root: python scripts/eigenvalue_multiplicity_analysis.py

Outputs:
  data/results/eigenvalue_multiplicity_analysis.csv
  data/results/spectral_descriptor_correlations.csv
  figures/fig_spectral_descriptors.pdf
  figures/fig_eigenvalue_clustering.pdf
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from analysis.spectral_descriptors import (
    MESHES_9,
    MESHES_GENUS_36,
    effective_multiplicity,
    eigenvalue_density,
    spectral_entropy,
    spectral_gap_ratio,
)

# --- Config ---
K = 50
EPS_FRAC = 0.01  # ε = EPS_FRAC * (λ_50 - λ_1) for clustering

EIGEN_DIR = root / "data" / "eigen"
EIGEN_EXPERIMENTS_DIR = root / "data" / "experiments" / "eigen"
RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"
OUT_CSV = RESULTS_DIR / "eigenvalue_multiplicity_analysis.csv"
CORR_CSV = RESULTS_DIR / "spectral_descriptor_correlations.csv"
FIG_SPECTRAL = FIGURES_DIR / "fig_spectral_descriptors.pdf"
FIG_CLUSTERING = FIGURES_DIR / "fig_eigenvalue_clustering.pdf"

# Genus 0–6 mesh list for 7-point sequence (for correlation): one representative per genus
GENUS_TO_MESH = [
    ("sphere", EIGEN_DIR),
    ("torus", EIGEN_DIR),
    ("double_torus", EIGEN_DIR),
    ("triple_torus_genus3", EIGEN_EXPERIMENTS_DIR),
    ("quad_torus_genus4", EIGEN_EXPERIMENTS_DIR),
    ("penta_torus_genus5", EIGEN_EXPERIMENTS_DIR),
    ("hex_torus_genus6", EIGEN_EXPERIMENTS_DIR),
]


def load_eigenvalues(mesh_id: str, eigen_dirs: list[Path]) -> np.ndarray | None:
    """Load first K eigenvalues for a mesh. Tries each eigen dir. Returns None if missing."""
    evals = None
    for d in eigen_dirs:
        p = d / f"{mesh_id}.npz"
        if not p.exists():
            continue
        try:
            data = np.load(p, allow_pickle=True)
            if "eigenvalues" not in data.files:
                continue
            ev = np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
            ev = np.sort(ev)
            if len(ev) >= K:
                return ev[:K].copy()
            # allow shorter if that's all we have
            if len(ev) >= 10:
                return ev.copy()
        except Exception as e:
            warnings.warn(f"Load {p}: {e}", UserWarning)
    return None


def _lattice_pairs_ordered(n_pairs: int) -> list[tuple[int, int]]:
    """First n_pairs (m,n) with m,n >= 0, ordered by m^2+n^2, excluding (0,0)."""
    cand = []
    for m in range(8):
        for n in range(8):
            if m == 0 and n == 0:
                continue
            cand.append((m * m + n * n, m, n))
    cand.sort(key=lambda x: (x[0], x[1], x[2]))
    # dedupe by value (keep order)
    seen = set()
    out = []
    for _, m, n in cand:
        if (m, n) in seen:
            continue
        seen.add((m, n))
        out.append((m, n))
        if len(out) >= n_pairs:
            break
    # ensure we have exactly n_pairs (pad with (0,0) if needed)
    while len(out) < n_pairs:
        out.append((0, 0))
    return out[:n_pairs]


def product_structure_score(ev: np.ndarray) -> float:
    """
    Fit λ ≈ (m/R)² + (n/r)² for torus-like lattice. Report R² of fit.
    Model: λ = A*m² + B*n² (A=1/R², B=1/r²). Linear regression.
    """
    n = len(ev)
    if n < 3:
        return 0.0
    pairs = _lattice_pairs_ordered(n)
    X = np.array([[m * m, n * n] for m, n in pairs], dtype=np.float64)
    y = ev.astype(np.float64)
    # y = X @ [A, B]
    try:
        from numpy.linalg import lstsq
        sol, residuals, rank, s = lstsq(X, y, rcond=None)
        y_pred = X @ sol
        ss_tot = np.sum((y - y.mean()) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        if ss_tot <= 0:
            return 0.0
        r2 = float(1.0 - ss_res / ss_tot)
        return max(0.0, min(1.0, r2))
    except Exception:
        return 0.0


def compute_descriptors(ev: np.ndarray) -> dict | None:
    """Compute all spectral descriptors from eigenvalue array (length >= 2)."""
    if ev is None or len(ev) < 2:
        return None
    ev = np.asarray(ev, dtype=np.float64)
    span = ev[-1] - ev[0]
    eps = EPS_FRAC * span if span > 0 else 0.0
    n_clusters, mean_size = effective_multiplicity(ev, eps)
    H = spectral_entropy(ev)
    gamma = spectral_gap_ratio(ev)
    rho = eigenvalue_density(ev)
    P = product_structure_score(ev)
    return {
        "M_eff_clusters": n_clusters,
        "M_eff_mean_size": mean_size,
        "spectral_entropy_H": H,
        "spectral_gap_gamma": gamma,
        "eigenvalue_density_rho": rho,
        "product_score_P": P,
    }


def run_analysis() -> list[dict]:
    """Load all 13 meshes, compute descriptors. Return list of row dicts (mesh, genus, ...)."""
    rows = []
    for mesh_id, genus in MESHES_9:
        ev = load_eigenvalues(mesh_id, [EIGEN_DIR])
        if ev is None:
            print(f"Warning: no eigen for {mesh_id}, skipping.", file=sys.stderr)
            continue
        d = compute_descriptors(ev)
        if d is None:
            continue
        rows.append({"mesh": mesh_id, "genus": genus, **d})

    for mesh_id, genus in MESHES_GENUS_36:
        ev = load_eigenvalues(mesh_id, [EIGEN_EXPERIMENTS_DIR])
        if ev is None:
            print(f"Warning: no eigen for {mesh_id}, skipping.", file=sys.stderr)
            continue
        d = compute_descriptors(ev)
        if d is None:
            continue
        rows.append({"mesh": mesh_id, "genus": genus, **d})

    return rows


def write_descriptor_csv(rows: list[dict]) -> None:
    """Write eigenvalue_multiplicity_analysis.csv."""
    if not rows:
        return
    fieldnames = [
        "mesh", "genus",
        "M_eff_clusters", "M_eff_mean_size", "spectral_entropy_H",
        "spectral_gap_gamma", "eigenvalue_density_rho", "product_score_P",
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV}", file=sys.stderr)


def load_beta0_genus_7point() -> dict[int, float] | None:
    """Load beta0 by genus from results_genus_extended_7point.csv. Returns {genus: beta0}."""
    path = RESULTS_DIR / "results_genus_extended_7point.csv"
    if not path.exists():
        return None
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                g = int(float(r["genus"]))
                out[g] = float(r["beta0"])
            except (ValueError, KeyError):
                pass
    return out if out else None


def load_beta0_9mesh_A5() -> dict[str, float] | None:
    """Load beta0 for 9 meshes, A5, direct from results.csv. Returns {mesh: beta0}."""
    path = RESULTS_DIR / "results.csv"
    if not path.exists():
        return None
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("audio") == "A5" and r.get("strategy") == "direct":
                try:
                    out[r["mesh"]] = float(r["beta0"])
                except (ValueError, KeyError):
                    pass
    return out if out else None


def compute_correlations(rows: list[dict]) -> list[dict]:
    """Spearman correlation of each descriptor with β₀ (genus sequence and 9-mesh A5)."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return []

    beta0_genus = load_beta0_genus_7point()
    beta0_A5 = load_beta0_9mesh_A5()

    descriptors = [
        "M_eff_clusters", "M_eff_mean_size", "spectral_entropy_H",
        "spectral_gap_gamma", "eigenvalue_density_rho", "product_score_P",
    ]
    corr_rows = []

    # Genus sequence: 7 meshes (one per genus 0–6)
    if beta0_genus is not None and rows:
        df_like = {d: [] for d in descriptors}
        df_like["beta0"] = []
        for g in range(7):
            # find row for this genus (first mesh with that genus)
            r = next((x for x in rows if x["genus"] == g), None)
            if r is None:
                continue
            if g not in beta0_genus:
                continue
            df_like["beta0"].append(beta0_genus[g])
            for d in descriptors:
                df_like[d].append(r[d])
        n = len(df_like["beta0"])
        if n >= 3:
            for d in descriptors:
                x = np.array(df_like[d], dtype=np.float64)
                y = np.array(df_like["beta0"], dtype=np.float64)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 3:
                    rho, p = spearmanr(x[ok], y[ok])
                    corr_rows.append({
                        "beta0_source": "genus_sequence_7point",
                        "descriptor": d,
                        "spearman_rho": float(rho),
                        "spearman_p": float(p),
                        "n": int(ok.sum()),
                    })

    # 9-mesh A5 direct
    if beta0_A5 is not None and rows:
        df_like = {d: [] for d in descriptors}
        df_like["beta0"] = []
        for r in rows:
            mesh = r["mesh"]
            if mesh not in beta0_A5:
                continue
            df_like["beta0"].append(beta0_A5[mesh])
            for d in descriptors:
                df_like[d].append(r[d])
        n = len(df_like["beta0"])
        if n >= 3:
            for d in descriptors:
                x = np.array(df_like[d], dtype=np.float64)
                y = np.array(df_like["beta0"], dtype=np.float64)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 3:
                    rho, p = spearmanr(x[ok], y[ok])
                    corr_rows.append({
                        "beta0_source": "9mesh_A5_direct",
                        "descriptor": d,
                        "spearman_rho": float(rho),
                        "spearman_p": float(p),
                        "n": int(ok.sum()),
                    })

    return corr_rows


def write_correlation_csv(corr_rows: list[dict]) -> None:
    """Write spectral_descriptor_correlations.csv."""
    if not corr_rows:
        return
    fieldnames = ["beta0_source", "descriptor", "spearman_rho", "spearman_p", "n"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORR_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(corr_rows)
    print(f"Wrote {CORR_CSV}", file=sys.stderr)


def plot_spectral_descriptors(rows: list[dict], corr_rows: list[dict]) -> None:
    """2×3 scatter: each descriptor vs β₀, mesh labels, Spearman ρ, color by genus."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    beta0_genus = load_beta0_genus_7point()
    beta0_A5 = load_beta0_9mesh_A5()

    # Use 9-mesh A5 beta0 when available, else genus-sequence beta0 by genus
    beta0_by_mesh = {}
    for r in rows:
        mesh, genus = r["mesh"], r["genus"]
        if beta0_A5 is not None and mesh in beta0_A5:
            beta0_by_mesh[mesh] = beta0_A5[mesh]
        elif beta0_genus is not None and genus in beta0_genus:
            beta0_by_mesh[mesh] = beta0_genus[genus]
    # Prefer A5 for 9 meshes so we have one beta0 per mesh
    for r in rows:
        mesh, genus = r["mesh"], r["genus"]
        if mesh in beta0_by_mesh:
            continue
        if beta0_genus is not None and genus in beta0_genus:
            beta0_by_mesh[mesh] = beta0_genus[genus]

    plot_rows = [r for r in rows if r["mesh"] in beta0_by_mesh]
    if not plot_rows:
        print("No beta0 data for plotting; skipping fig_spectral_descriptors.pdf", file=sys.stderr)
        return

    descriptors = [
        "M_eff_clusters", "M_eff_mean_size", "spectral_entropy_H",
        "spectral_gap_gamma", "eigenvalue_density_rho", "product_score_P",
    ]
    rho_by = {}  # (source, descriptor) -> rho for annotation
    for c in corr_rows:
        key = (c["beta0_source"], c["descriptor"])
        rho_by[key] = c["spearman_rho"]

    # Genus colors: 0=blue, 1=green, 2=red, 3+=purple
    def genus_color(g):
        if g == 0:
            return "C0"  # blue
        if g == 1:
            return "C2"  # green
        if g == 2:
            return "C3"  # red
        return "purple"

    plt.rc("font", family="serif")
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes = axes.ravel()

    for idx, desc in enumerate(descriptors):
        ax = axes[idx]
        x = [r[desc] for r in plot_rows]
        y = [beta0_by_mesh[r["mesh"]] for r in plot_rows]
        colors = [genus_color(r["genus"]) for r in plot_rows]
        labels = [r["mesh"].replace("_", " ") for r in plot_rows]
        ax.scatter(x, y, c=colors, s=60, zorder=2, edgecolors="k", linewidths=0.5)
        for i, lb in enumerate(labels):
            ax.annotate(lb, (x[i], y[i]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6)
        # Annotate Spearman (both sources when available)
        rho_9 = rho_by.get(("9mesh_A5_direct", desc))
        rho_7 = rho_by.get(("genus_sequence_7point", desc))
        parts = []
        if rho_7 is not None:
            parts.append(f"ρ₇={rho_7:.2f}")
        if rho_9 is not None:
            parts.append(f"ρ₉={rho_9:.2f}")
        if parts:
            ax.text(0.05, 0.95, "Spearman " + ", ".join(parts), transform=ax.transAxes, fontsize=7, verticalalignment="top")
        ax.set_xlabel(desc.replace("_", " "))
        ax.set_ylabel(r"$\beta_0$")
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3)

    # Legend for genus colors
    from matplotlib.lines import Line2D
    leg_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", label="genus 0", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C2", label="genus 1", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C3", label="genus 2", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="purple", label="genus 3+", markersize=8),
    ]
    fig.legend(handles=leg_handles, loc="lower right", ncol=4, fontsize=7)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_SPECTRAL, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved {FIG_SPECTRAL}", file=sys.stderr)


def get_cluster_boundaries(ev: np.ndarray, eps: float) -> np.ndarray:
    """Indices where a new cluster starts (0 always, then after each gap >= eps)."""
    if len(ev) < 2:
        return np.array([0])
    gaps = np.diff(ev)
    boundaries = [0]
    for i, g in enumerate(gaps):
        if g >= eps:
            boundaries.append(i + 1)
    return np.array(boundaries)


def plot_eigenvalue_clustering(rows: list[dict]) -> None:
    """Stem plot of eigenvalues with cluster boundaries for 4 representative meshes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Four representative meshes: sphere, torus, double torus, genus-4 (paper spec)
    want = ["sphere", "torus", "double_torus", "quad_torus_genus4"]
    by_mesh = {r["mesh"]: r for r in rows}
    plots = [name for name in want if name in by_mesh]
    if len(plots) < 4:
        for r in rows:
            if r["mesh"] not in plots and r["genus"] >= 3:
                plots.append(r["mesh"])
                if len(plots) >= 4:
                    break
    plots = plots[:4]

    # Load eigenvalues again for these
    eigen_dirs = [EIGEN_DIR, EIGEN_EXPERIMENTS_DIR]
    data = []
    for mesh_id in plots:
        ev = load_eigenvalues(mesh_id, eigen_dirs)
        if ev is None or len(ev) < 2:
            continue
        span = ev[-1] - ev[0]
        eps = EPS_FRAC * span if span > 0 else 0.0
        boundaries = get_cluster_boundaries(ev, eps)
        data.append((mesh_id, ev, boundaries, eps))

    if not data:
        print("No eigen data for clustering figure; skipping.", file=sys.stderr)
        return

    plt.rc("font", family="serif")
    n_plots = len(data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(6, 1.8 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    for ax, (mesh_id, ev, boundaries, eps) in zip(axes, data):
        k = np.arange(1, len(ev) + 1, dtype=float)
        ax.stem(k, ev, linefmt="C0-", markerfmt="C0o", basefmt=" ")
        for b in boundaries:
            if b > 0:
                ax.axvline(x=b, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel(r"$\lambda_k$")
        ax.set_title(mesh_id.replace("_", " "))
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("$k$")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_CLUSTERING, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved {FIG_CLUSTERING}", file=sys.stderr)


def print_summary_table(rows: list[dict], corr_rows: list[dict]) -> None:
    """Print readable summary table to stdout."""
    if not rows:
        return
    print("\n" + "=" * 80)
    print("Eigenvalue multiplicity analysis — descriptor summary")
    print("=" * 80)
    # Header
    cols = ["mesh", "genus", "M_eff_cl", "M_eff_sz", "H", "γ", "ρ", "P"]
    fmt = "{:22} {:>5} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}"
    print(fmt.format(*cols))
    print("-" * 80)
    for r in rows:
        print(fmt.format(
            r["mesh"][:22],
            r["genus"],
            r["M_eff_clusters"],
            round(r["M_eff_mean_size"], 2),
            round(r["spectral_entropy_H"], 3),
            round(r["spectral_gap_gamma"], 4) if np.isfinite(r["spectral_gap_gamma"]) else "—",
            round(r["eigenvalue_density_rho"], 4) if np.isfinite(r["eigenvalue_density_rho"]) else "—",
            round(r["product_score_P"], 4),
        ))
    print("=" * 80)
    if corr_rows:
        print("\nSpearman correlation with β₀")
        print("-" * 40)
        for c in corr_rows:
            print(f"  {c['beta0_source']:25} {c['descriptor']:30} ρ = {c['spearman_rho']:+.3f}  (n={c['n']})")
    print()


def main() -> int:
    rows = run_analysis()
    if not rows:
        print("No mesh data; ensure data/eigen/*.npz and/or data/experiments/eigen/*.npz exist.", file=sys.stderr)
        return 1
    write_descriptor_csv(rows)
    corr_rows = compute_correlations(rows)
    write_correlation_csv(corr_rows)
    plot_spectral_descriptors(rows, corr_rows)
    plot_eigenvalue_clustering(rows)
    print_summary_table(rows, corr_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
