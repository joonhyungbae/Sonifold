"""
Spectral entropy H vs β₀ conjecture test.

Load eigenvalues for all 13 meshes (9 exploration + genus 3–6). Compute H (spectral entropy
of normalized eigenvalue gaps) for K=50, 100, 200. For each K and mesh, compute β₀ under:
  (a) random coefficients (N=200, uniform on sphere) → mean ± std
  (b) A5 white noise (direct mapping) → mean over frames
  (c) A3 piano (direct mapping) → mean over frames
Compute Spearman ρ(H, β₀) for each (K, coefficient_source). Test whether correlation
strengthens or weakens with K.

Outputs:
  - data/results/spectral_entropy_vs_beta0.csv
  - data/results/spectral_entropy_correlations.csv
  - figures/fig_spectral_entropy_main.pdf

Run from project root: python scripts/spectral_entropy_conjecture_test.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from analysis.spectral_descriptors import MESHES_9, MESHES_GENUS_36, spectral_entropy

RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"
EIGEN_DIR = root / "data" / "eigen"
EIGEN_EXPERIMENTS_DIR = root / "data" / "experiments" / "eigen"

K_VALUES = [50, 100, 200]
N_RANDOM = 200
RNG = np.random.default_rng(42)
STRATEGY = "direct"
HOP_LENGTH = 512
TARGET_DURATION_SEC = 10.0

OUT_MAIN_CSV = RESULTS_DIR / "spectral_entropy_vs_beta0.csv"
OUT_CORR_CSV = RESULTS_DIR / "spectral_entropy_correlations.csv"
OUT_FIG = FIGURES_DIR / "fig_spectral_entropy_main.pdf"


def mesh_genus_list():
    """(mesh_id, genus) for all 13 meshes in fixed order."""
    return list(MESHES_9) + list(MESHES_GENUS_36)


def _load_ev_from_path(p: Path, k: int) -> np.ndarray | None:
    """Load up to k eigenvalues from npz or json. Returns None if too few."""
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        if "eigenvalues" not in data.files:
            return None
        ev = np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
    else:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        ev = np.array(data.get("eigenvalues", []), dtype=np.float64)
    ev = np.sort(ev)
    if len(ev) < 2:
        return None
    return ev[: min(k, len(ev))].copy()


def load_eigenvalues_up_to(mesh_id: str, k: int) -> np.ndarray | None:
    """Load first k eigenvalues. Tries EIGEN_DIR for 9 meshes, EIGEN_EXPERIMENTS_DIR for genus 3–6."""
    for mesh_id_cur, _ in MESHES_9:
        if mesh_id_cur == mesh_id:
            for d in [EIGEN_DIR]:
                for ext in (".npz", ".json"):
                    p = d / (mesh_id + ext)
                    if p.exists():
                        return _load_ev_from_path(p, k)
            return None
    for mesh_id_cur, _ in MESHES_GENUS_36:
        if mesh_id_cur == mesh_id:
            for d in [EIGEN_EXPERIMENTS_DIR]:
                for ext in (".npz", ".json"):
                    p = d / (mesh_id + ext)
                    if p.exists():
                        return _load_ev_from_path(p, k)
            return None
    return None


def load_mesh_eigenbasis(mesh_id: str, genus: int, k: int):
    """Load V, F, evals (length k), evecs (k, n_verts). Returns None if insufficient modes."""
    # 9 meshes: data/eigen (short name); genus 3–6: data/experiments/eigen
    is_genus36 = any(m == mesh_id for m, _ in MESHES_GENUS_36)
    if is_genus36:
        bases = [EIGEN_EXPERIMENTS_DIR]
        names = [mesh_id]
    else:
        bases = [EIGEN_DIR]
        names = [mesh_id]
    p = None
    for base, name in zip(bases, names):
        for ext in (".npz", ".json"):
            p = base / (name + ext)
            if p.exists():
                break
        else:
            p = None
            continue
        break
    if p is None or not p.exists():
        return None

    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        V = np.asarray(data["vertices"], dtype=np.float64)
        F = np.asarray(data["faces"], dtype=np.int32)
        evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
        evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel() if "eigenvalues" in data.files else None
    else:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        V = np.array(data["vertices"], dtype=np.float64)
        F = np.array(data["faces"], dtype=np.int32)
        evecs = np.array(data["eigenvectors"], dtype=np.float64)
        evals = np.array(data.get("eigenvalues", []), dtype=np.float64)

    n_verts = V.shape[0]
    evals = np.asarray(evals, dtype=np.float64).ravel() if evals is not None else None
    n_avail = min(evecs.shape[0], len(evals) if evals is not None else evecs.shape[0])
    if n_avail < k:
        return None
    n_use = min(k, n_avail)
    evals = evals[: n_use] if evals is not None else None
    if evecs.shape[0] == n_verts:
        evecs = evecs[:, : n_use].T
    else:
        evecs = evecs[: n_use, :]
    if evecs.shape[0] != n_use:
        return None
    return V, F, evals, evecs


def sample_uniform_sphere(k: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal(k)
    n = np.linalg.norm(a)
    if n < 1e-20:
        a = rng.standard_normal(k)
        n = np.linalg.norm(a)
    return (a / n).astype(np.float64)


def beta0_random(V, F, evecs, k: int, n_trials: int):
    from analysis.scalar_field import compute_scalar_field
    from analysis.nodal_surface import compute_topology_metrics
    beta0s = []
    for _ in range(n_trials):
        coef = sample_uniform_sphere(k, RNG)
        f = compute_scalar_field(evecs, coef)
        m = compute_topology_metrics(V, F, f)
        beta0s.append(m.beta0)
    arr = np.array(beta0s, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0


def beta0_audio_frames(mesh_id: str, genus: int, k: int, audio_id: str):
    """Mean β₀ over STFT frames, direct mapping."""
    from audio.audio_library import get_audio
    from audio.fft_analysis import compute_fft_frames
    from mapping.spectral_mapping import map_fft_to_coefficients
    from analysis.scalar_field import compute_scalar_field
    from analysis.nodal_surface import compute_topology_metrics

    out = load_mesh_eigenbasis(mesh_id, genus, k)
    if out is None:
        return None
    V, F, evals, evecs = out
    sig, sr = get_audio(audio_id)
    n_target = int(TARGET_DURATION_SEC * sr)
    if len(sig) < n_target:
        repeats = (n_target + len(sig) - 1) // len(sig)
        sig = np.tile(sig, repeats)[: n_target]
    else:
        sig = sig[: n_target]
    frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
    if not frames:
        return None
    mags = [mag for mag, _ in frames]
    beta0s = []
    for mag in mags:
        coef = map_fft_to_coefficients(mag, k, strategy=STRATEGY, eigenvalues=evals)
        f = compute_scalar_field(evecs, coef)
        m = compute_topology_metrics(V, F, f)
        beta0s.append(m.beta0)
    return float(np.mean(beta0s))


def run_per_mesh_k(mesh_id: str, genus: int, k: int):
    """One (mesh, K): H, beta0_random_mean, beta0_random_std, beta0_A5, beta0_A3."""
    ev = load_eigenvalues_up_to(mesh_id, k)
    if ev is None or len(ev) < 2:
        return None
    ev = np.sort(ev.ravel())[: k]
    H = spectral_entropy(ev)

    out = load_mesh_eigenbasis(mesh_id, genus, k)
    if out is None:
        return None
    V, F, evals, evecs = out

    b0_mean, b0_std = beta0_random(V, F, evecs, k, N_RANDOM)
    b0_A5 = beta0_audio_frames(mesh_id, genus, k, "A5")
    b0_A3 = beta0_audio_frames(mesh_id, genus, k, "A3")

    return {
        "mesh": mesh_id,
        "genus": genus,
        "K": k,
        "H": round(H, 6),
        "beta0_random_mean": round(b0_mean, 4),
        "beta0_random_std": round(b0_std, 4),
        "beta0_A5": round(b0_A5, 4) if b0_A5 is not None else "",
        "beta0_A3": round(b0_A3, 4) if b0_A3 is not None else "",
    }


def compute_correlations(rows: list[dict]) -> list[dict]:
    from scipy.stats import spearmanr
    corr_rows = []
    for k in K_VALUES:
        sub = [r for r in rows if r["K"] == k and r.get("H") is not None]
        if len(sub) < 3:
            continue
        H = np.array([r["H"] for r in sub], dtype=np.float64)
        for source, key in [("random", "beta0_random_mean"), ("A5", "beta0_A5"), ("A3", "beta0_A3")]:
            y_list = []
            for r in sub:
                v = r.get(key)
                if v == "" or v is None:
                    y_list.append(np.nan)
                else:
                    try:
                        y_list.append(float(v))
                    except (TypeError, ValueError):
                        y_list.append(np.nan)
            y = np.array(y_list, dtype=np.float64)
            ok = np.isfinite(H) & np.isfinite(y)
            n = int(np.sum(ok))
            if n >= 3:
                rho, p = spearmanr(H[ok], y[ok])
                corr_rows.append({
                    "K": k,
                    "source": source,
                    "spearman_rho": round(float(rho), 4),
                    "p_value": round(float(p), 6),
                    "n": n,
                })
    return corr_rows


def plot_main_figure(rows: list[dict], corr_rows: list[dict]):
    """Single scatter: H vs β₀ for K=50, mesh labels, Spearman ρ, color by genus."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = [r for r in rows if r["K"] == 50]
    if not sub:
        print("No K=50 rows for figure; skipping.", file=sys.stderr)
        return
    # Use beta0_random_mean for main plot (structural relationship)
    x = [r["H"] for r in sub]
    y = [r["beta0_random_mean"] for r in sub]
    labels = [r["mesh"].replace("_", " ") for r in sub]
    genera = [r["genus"] for r in sub]

    rho_50_random = next((c["spearman_rho"] for c in corr_rows if c["K"] == 50 and c["source"] == "random"), None)

    def genus_color(g):
        if g == 0:
            return "C0"
        if g == 1:
            return "C2"
        if g == 2:
            return "C3"
        return "purple"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rc("font", family="serif")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [genus_color(g) for g in genera]
    ax.scatter(x, y, c=colors, s=80, zorder=2, edgecolors="k", linewidths=0.6)
    for i, lb in enumerate(labels):
        ax.annotate(lb, (x[i], y[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel("Spectral entropy $H$ (normalized gaps)")
    ax.set_ylabel(r"$\beta_0$ (random coefficients, mean)")
    ax.set_title(r"$H$ vs $\beta_0$ (K=50)")
    if rho_50_random is not None:
        ax.text(0.05, 0.95, f"Spearman $\\rho$ = {rho_50_random:.2f}", transform=ax.transAxes, fontsize=11, va="top")
    plt.tight_layout()
    plt.savefig(OUT_FIG, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_FIG}", file=sys.stderr)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    mesh_genus = mesh_genus_list()
    for mesh_id, genus in mesh_genus:
        for k in K_VALUES:
            row = run_per_mesh_k(mesh_id, genus, k)
            if row is not None:
                all_rows.append(row)
                print(f"[{mesh_id}] K={k} H={row['H']} beta0_r={row['beta0_random_mean']} A5={row.get('beta0_A5')} A3={row.get('beta0_A3')}", file=sys.stderr)
            else:
                print(f"Skip {mesh_id} K={k}", file=sys.stderr)

    if not all_rows:
        print("No rows; exiting.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["mesh", "genus", "K", "H", "beta0_random_mean", "beta0_random_std", "beta0_A5", "beta0_A3"]
    with open(OUT_MAIN_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {OUT_MAIN_CSV}", file=sys.stderr)

    corr_rows = compute_correlations(all_rows)
    with open(OUT_CORR_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["K", "source", "spearman_rho", "p_value", "n"])
        w.writeheader()
        w.writerows(corr_rows)
    print(f"Wrote {OUT_CORR_CSV}", file=sys.stderr)

    plot_main_figure(all_rows, corr_rows)

    # K sensitivity: does ρ(H, β₀) strengthen or weaken with K?
    print("\n--- Correlation H ↔ β₀ by K ---")
    for source in ["random", "A5", "A3"]:
        by_k = [(c["K"], c["spearman_rho"]) for c in corr_rows if c["source"] == source]
        by_k.sort(key=lambda x: x[0])
        if len(by_k) >= 2:
            rhos = [r for _, r in by_k]
            trend = "strengthens" if all(rhos[i] < rhos[i + 1] for i in range(len(rhos) - 1)) else (
                "weakens" if all(rhos[i] > rhos[i + 1] for i in range(len(rhos) - 1)) else "mixed")
            parts = [f"K{k}={r:.3f}" for k, r in by_k]
            print(f"  {source}: " + ", ".join(parts) + f" → correlation {trend} with K")
        elif by_k:
            print(f"  {source}: K{by_k[0][0]}={by_k[0][1]:.3f}")

    # Conjecture sharpening
    if corr_rows:
        min_rho = min(c["spearman_rho"] for c in corr_rows)
        if min_rho > 0.6:
            print("\nConjecture can be sharpened: ρ(H, β₀) > 0.6 across all K and coefficient sources.")
            print('  "E[β₀] increases monotonically with spectral entropy H."')
        else:
            print(f"\nmin Spearman ρ = {min_rho:.2f}; not all > 0.6; do not sharpen conjecture.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
