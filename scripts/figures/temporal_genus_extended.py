"""
Compute frame-by-frame β₀ for A3(piano) on genus 0–4.
Compare temporal variance of β₀ across higher-genus shapes.

Run from project root: python scripts/temporal_genus_extended.py
Requires: data/experiments/eigen/{sphere_genus0,torus_genus1,double_torus_genus2,
         triple_torus_genus3,quad_torus_genus4}.json or .npz (genus 3,4: precompute_experiment_eigen)
Output: data/results/temporal_genus_extended.csv, temporal_genus_summary.csv
        figures/fig_temporal_genus.pdf, figures/fig_temporal_variance_vs_genus.pdf
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft_frames, N_FFT
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

# Genus sequence (Section 4.4): ~5000 vertices, precomputed eigen
GENUS_MESHES = [
    ("sphere_genus0", 0),
    ("torus_genus1", 1),
    ("double_torus_genus2", 2),
    ("triple_torus_genus3", 3),
    ("quad_torus_genus4", 4),
]
AUDIO_ID = "A3"
TARGET_DURATION_SEC = 10.0
HOP_LENGTH = 512
N_EIGEN = 50
STRATEGY = "direct"

EIGEN_EXPERIMENTS = root / "data" / "experiments" / "eigen"
EIGEN_MAIN = root / "data" / "eigen"
RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"
OUT_CSV = RESULTS_DIR / "temporal_genus_extended.csv"
SUMMARY_CSV = RESULTS_DIR / "temporal_genus_summary.csv"
FIG_TEMPORAL_GENUS = FIGURES_DIR / "fig_temporal_genus.pdf"
FIG_VARIANCE_GENUS = FIGURES_DIR / "fig_temporal_variance_vs_genus.pdf"


def load_eigen(mesh_id: str):
    """Load eigen from experiments/eigen or data/eigen (same logic as run_temporal_persistence)."""
    for base, ext in [(EIGEN_EXPERIMENTS, ".npz"), (EIGEN_EXPERIMENTS, ".json"), (EIGEN_MAIN, ".json")]:
        path = base / (mesh_id + ext)
        if not path.exists():
            alt_id = mesh_id.replace("_genus0", "").replace("_genus1", "").replace("_genus2", "").replace("_genus3", "").replace("_genus4", "")
            path_alt = base / (alt_id + ext)
            if path_alt.exists():
                path = path_alt
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


def compute_rms_per_frame(signal: np.ndarray, frames: list, n_fft: int) -> np.ndarray:
    """RMS per STFT frame (same as temporal_with_audio)."""
    rms = []
    for _mag, start in frames:
        win = signal[start : start + n_fft]
        rms.append(np.sqrt(np.mean(win.astype(np.float64) ** 2)))
    return np.array(rms)


def compute_spectral_flux(frames: list) -> np.ndarray:
    """Spectral flux: frame-to-frame L1 change in magnitude spectrum. First frame flux = 0."""
    mags = np.array([m for m, _ in frames], dtype=np.float64)
    flux = np.zeros(len(mags))
    for i in range(1, len(mags)):
        flux[i] = np.sum(np.abs(mags[i] - mags[i - 1]))
    return flux


def build_audio_10s(audio_id: str):
    """Load audio and tile/pad to TARGET_DURATION_SEC (same as run_temporal_persistence)."""
    sig, sr = get_audio(audio_id)
    n_target = int(TARGET_DURATION_SEC * sr)
    if len(sig) < n_target:
        repeats = (n_target + len(sig) - 1) // len(sig)
        sig = np.tile(sig, repeats)[:n_target]
    else:
        sig = sig[:n_target]
    return sig, sr


def run_pipeline():
    """Compute frame-by-frame beta0 for A3 on all genus 0–4 meshes; return rows and per-mesh series."""
    sig, sr = build_audio_10s(AUDIO_ID)
    frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
    n_frames = len(frames)
    rms = compute_rms_per_frame(sig, frames, N_FFT)
    flux = compute_spectral_flux(frames)

    rows = []
    series_by_mesh = {}  # mesh_name -> (t_sec, beta0_arr)

    for mesh_name, genus in tqdm(GENUS_MESHES, desc="meshes", leave=True):
        eigen = load_eigen(mesh_name)
        if eigen is None:
            print("Skip", mesh_name, "(no eigen)", file=sys.stderr)
            continue
        V, F, evals, evecs = eigen
        beta0_list = []
        for frame_idx, (mag, start_idx) in enumerate(tqdm(frames, desc=mesh_name, leave=False)):
            coef = map_fft_to_coefficients(mag, N_EIGEN, strategy=STRATEGY, eigenvalues=evals)
            f = compute_scalar_field(evecs, coef)
            m = compute_topology_metrics(V, F, f)
            beta0_list.append(m.beta0)
            rows.append({
                "mesh_name": mesh_name,
                "genus": genus,
                "frame_index": frame_idx,
                "beta0": m.beta0,
                "rms": rms[frame_idx],
                "spectral_flux": flux[frame_idx],
            })
        t_sec = np.array([start / sr for _, start in frames])
        series_by_mesh[mesh_name] = (t_sec, np.array(beta0_list))

    return rows, series_by_mesh, (np.array([start / sr for _, start in frames]), rms)


def compute_summary(rows):
    """Aggregate per (mesh_name, genus): mean_beta0, std_beta0, cv_beta0, mean_delta_beta0, max_delta_beta0."""
    from collections import defaultdict
    by_mesh = defaultdict(list)
    for r in rows:
        by_mesh[(r["mesh_name"], r["genus"])].append(r["beta0"])

    summary = []
    for (mesh_name, genus), beta0_vals in sorted(by_mesh.items(), key=lambda x: x[0][1]):
        b = np.array(beta0_vals, dtype=np.float64)
        mean_b = float(np.mean(b))
        std_b = float(np.std(b))
        cv_b = std_b / (mean_b + 1e-20)
        delta = np.abs(np.diff(b))
        mean_delta = float(np.mean(delta)) if len(delta) else 0.0
        max_delta = float(np.max(delta)) if len(delta) else 0.0
        summary.append({
            "mesh_name": mesh_name,
            "genus": genus,
            "mean_beta0": mean_b,
            "std_beta0": std_b,
            "cv_beta0": cv_b,
            "mean_delta_beta0": mean_delta,
            "max_delta_beta0": max_delta,
        })
    return summary


def save_csv(rows, summary):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mesh_name", "genus", "frame_index", "beta0", "rms", "spectral_flux"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Saved", OUT_CSV, file=sys.stderr)

    summary_fieldnames = ["mesh_name", "genus", "mean_beta0", "std_beta0", "cv_beta0", "mean_delta_beta0", "max_delta_beta0"]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fieldnames)
        w.writeheader()
        w.writerows(summary)
    print("Saved", SUMMARY_CSV, file=sys.stderr)


def plot_temporal_genus(series_by_mesh, t_rms, rms):
    """Overlay β₀ time series for 5 meshes + RMS envelope subplot (shared time axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    mesh_to_genus = {name: g for name, g in GENUS_MESHES}
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 8.5), sharex=True, height_ratios=[1, 1.35]
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(GENUS_MESHES)))

    for mesh_name, (t_sec, beta0_arr) in sorted(series_by_mesh.items(), key=lambda x: mesh_to_genus.get(x[0], 0)):
        genus = mesh_to_genus.get(mesh_name, 0)
        ax_bot.plot(t_sec, beta0_arr, color=colors[genus], linewidth=1.0, label=f"g={genus} ({mesh_name})", alpha=0.9)

    ax_top.plot(t_rms, rms, color="C0", linewidth=1.2, label="RMS envelope")
    ax_top.set_ylabel("RMS")
    ax_top.legend(loc="upper right", fontsize=7)
    ax_top.set_facecolor("white")
    ax_top.grid(True, alpha=0.3)
    if len(t_rms):
        ax_top.set_xlim(0, t_rms[-1])

    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel(r"$\beta_0$")
    ax_bot.legend(loc="upper right", fontsize=7)
    ax_bot.set_facecolor("white")
    ax_bot.grid(True, alpha=0.3)
    fig.suptitle("A3 (piano) — " + r"$\beta_0$ time series by genus")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_TEMPORAL_GENUS, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", FIG_TEMPORAL_GENUS, file=sys.stderr)


def plot_variance_vs_genus(summary):
    """x = genus (0–4), y = σ(β₀) or CV(β₀); scatter with trend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    genus = [s["genus"] for s in summary]
    std_beta0 = [s["std_beta0"] for s in summary]
    cv_beta0 = [s["cv_beta0"] for s in summary]

    fig, (ax_std, ax_cv) = plt.subplots(1, 2, figsize=(8, 4.2))
    trend_handle = None
    ax_std.scatter(genus, std_beta0, s=80, color="C0", zorder=3)
    if len(genus) >= 2:
        z = np.polyfit(genus, std_beta0, 1)
        xl = np.array([min(genus), max(genus)])
        (ln,) = ax_std.plot(xl, np.polyval(z, xl), "--", color="gray", alpha=0.8, label="trend")
        trend_handle = ln
    ax_std.set_xlabel("Genus")
    ax_std.set_ylabel(r"$\sigma(\beta_0)$")
    ax_std.set_xticks(genus)
    if trend_handle is not None:
        ax_std.legend(handles=[trend_handle], loc="upper left", fontsize=8, frameon=True, framealpha=0.95)
    ax_std.set_facecolor("white")
    ax_std.grid(True, alpha=0.3)

    ax_cv.scatter(genus, cv_beta0, s=80, color="C1", zorder=3)
    if len(genus) >= 2:
        z = np.polyfit(genus, cv_beta0, 1)
        ax_cv.plot(xl, np.polyval(z, xl), "--", color="gray", alpha=0.8)
    ax_cv.set_xlabel("Genus")
    ax_cv.set_ylabel(r"CV($\beta_0$) = $\sigma/\mu$")
    ax_cv.set_xticks(genus)
    ax_cv.set_facecolor("white")
    ax_cv.grid(True, alpha=0.3)

    fig.suptitle("A3 (piano) — temporal variance of " + r"$\beta_0$" + " vs genus")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_VARIANCE_GENUS, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", FIG_VARIANCE_GENUS, file=sys.stderr)


def load_from_csv_and_plot():
    """Load existing CSVs and regenerate figures only (no recompute)."""
    if not OUT_CSV.exists() or not SUMMARY_CSV.exists():
        print("Run without --plot-only first to create", OUT_CSV, file=sys.stderr)
        return 1
    rows = []
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({**r, "beta0": float(r["beta0"]), "genus": int(r["genus"]), "frame_index": int(r["frame_index"]), "rms": float(r["rms"]), "spectral_flux": float(r["spectral_flux"])})
    summary = []
    with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            summary.append({**r, "genus": int(r["genus"]), "mean_beta0": float(r["mean_beta0"]), "std_beta0": float(r["std_beta0"]), "cv_beta0": float(r["cv_beta0"]), "mean_delta_beta0": float(r["mean_delta_beta0"]), "max_delta_beta0": float(r["max_delta_beta0"])})
    from collections import defaultdict
    by_mesh = defaultdict(list)
    for r in rows:
        by_mesh[r["mesh_name"]].append((r["frame_index"], r["beta0"], r["rms"]))
    series_by_mesh = {}
    t_rms = None
    rms = None
    for mesh_name, idx_beta_rms in by_mesh.items():
        idx_beta_rms.sort(key=lambda x: x[0])
        frame_indices = [x[0] for x in idx_beta_rms]
        beta0_arr = np.array([x[1] for x in idx_beta_rms])
        rms_arr = np.array([x[2] for x in idx_beta_rms])
        t_sec = np.array(frame_indices, dtype=np.float64) * HOP_LENGTH / 44100.0
        series_by_mesh[mesh_name] = (t_sec, beta0_arr)
        if t_rms is None:
            t_rms = t_sec
            rms = rms_arr
    plot_temporal_genus(series_by_mesh, t_rms, rms)
    plot_variance_vs_genus(summary)
    return 0


def main():
    import argparse
    p = argparse.ArgumentParser(description="Temporal β₀ by genus (A3)")
    p.add_argument("--plot-only", action="store_true", help="Regenerate figures from existing CSVs")
    args = p.parse_args()
    if args.plot_only:
        return load_from_csv_and_plot()
    rows, series_by_mesh, (t_rms, rms) = run_pipeline()
    if not rows:
        print("No data (eigen missing for all meshes?).", file=sys.stderr)
        return 1
    summary = compute_summary(rows)
    save_csv(rows, summary)
    plot_temporal_genus(series_by_mesh, t_rms, rms)
    plot_variance_vs_genus(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
