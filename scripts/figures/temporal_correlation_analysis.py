"""
Systematic analysis of correlation between β₀ time series and audio descriptors.

- Reuse temporal_with_audio pipeline: A3(piano), A7(orchestral), sphere/torus frame-by-frame β₀
- Audio descriptors: RMS, spectral flux (L2), spectral centroid, active frequency bins
- Correlation: Pearson, Spearman, cross-correlation lag ∈ [-5, +5]
- Output: data/results/temporal_correlations.csv, figures/fig_temporal_correlations.pdf, figures/fig_cross_correlation.pdf

Run from project root:
  python scripts/temporal_correlation_analysis.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft_frames, N_FFT, N_BINS

# Reuse pipeline from temporal_with_audio
from scripts.temporal_with_audio import (
    load_temporal_beta0,
    build_audio_10s,
    HOP_LENGTH,
    RESULTS_CSV as TEMPORAL_CSV,
)

TARGET_DURATION_SEC = 10.0
LAG_MIN, LAG_MAX = -5, 5
OUT_CSV = root / "data" / "results" / "temporal_correlations.csv"
FIG_CORR = root / "figures" / "fig_temporal_correlations.pdf"
FIG_XCORR = root / "figures" / "fig_cross_correlation.pdf"

AUDIO_IDS = ["A3", "A7"]
MESH_IDS = ["sphere", "torus"]
DESCRIPTOR_NAMES = ["rms", "spectral_flux", "spectral_centroid", "n_active_bins"]


def compute_rms_per_frame(signal: np.ndarray, frames: list, n_fft: int) -> np.ndarray:
    """RMS per STFT frame (same window boundaries as STFT)."""
    rms = []
    for mag, start in frames:
        win = signal[start : start + n_fft]
        rms.append(np.sqrt(np.mean(win.astype(np.float64) ** 2)))
    return np.array(rms)


def compute_spectral_flux_l2(frames: list) -> np.ndarray:
    """Spectral flux: frame-to-frame magnitude spectrum change, L2 norm. First frame = 0."""
    mags = np.array([m for m, _ in frames], dtype=np.float64)
    flux = np.zeros(len(mags))
    for i in range(1, len(mags)):
        flux[i] = np.linalg.norm(mags[i] - mags[i - 1])
    return flux


def compute_spectral_centroid_per_frame(frames: list, sample_rate: int = 44100) -> np.ndarray:
    """Spectral centroid per frame (Hz). Uses power = magnitude^2."""
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)[:N_BINS]
    mags = np.array([m for m, _ in frames], dtype=np.float64)
    centroids = np.zeros(len(mags))
    for i in range(len(mags)):
        power = mags[i] ** 2
        total = power.sum()
        if total < 1e-20:
            centroids[i] = 0.0
        else:
            centroids[i] = np.dot(freqs, power) / total
    return centroids


def compute_n_active_bins_per_frame(frames: list) -> np.ndarray:
    """Number of bins with magnitude > mean magnitude (threshold = mean per frame)."""
    mags = np.array([m for m, _ in frames], dtype=np.float64)
    out = np.zeros(len(mags), dtype=np.int64)
    for i in range(len(mags)):
        thresh = np.mean(mags[i])
        out[i] = np.sum(mags[i] > thresh)
    return out


def compute_all_descriptors(signal: np.ndarray, frames: list, sr: int) -> dict[str, np.ndarray]:
    """Returns dict of descriptor_name -> array of length len(frames)."""
    return {
        "rms": compute_rms_per_frame(signal, frames, N_FFT),
        "spectral_flux": compute_spectral_flux_l2(frames),
        "spectral_centroid": compute_spectral_centroid_per_frame(frames, sr),
        "n_active_bins": compute_n_active_bins_per_frame(frames),
    }


def align_beta0_to_frames(csv_path: Path, audio_id: str):
    """
    Load temporal β₀ from CSV and return (t_sec, beta0_sphere, beta0_torus) aligned by frame.
    Mesh IDs in CSV are 'sphere_genus0' and 'torus_genus1'.
    """
    data = load_temporal_beta0(csv_path, audio_id)
    if data is None:
        return None
    return data


def cross_correlation_at_lags(beta0: np.ndarray, audio_feat: np.ndarray, lag_min: int, lag_max: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlation for lags in [lag_min, lag_max].
    Returns (lags, values). value at lag k = correlation of beta0[t] with audio_feat[t+k].
    Valid range: we need both arrays to have overlapping valid indices.
    """
    n = min(len(beta0), len(audio_feat))
    lags = np.arange(lag_min, lag_max + 1, dtype=np.int64)
    vals = np.zeros(len(lags))
    b = (beta0 - np.mean(beta0)) / (np.std(beta0) + 1e-12)
    a = (audio_feat - np.mean(audio_feat)) / (np.std(audio_feat) + 1e-12)
    for i, lag in enumerate(lags):
        if lag <= 0:
            # beta0[0:n+lag] vs audio_feat[-lag:n]
            start_b, start_a = 0, -lag
            L = n + lag
        else:
            # beta0[lag:n] vs audio_feat[0:n-lag]
            start_b, start_a = lag, 0
            L = n - lag
        if L <= 0:
            vals[i] = np.nan
            continue
        vals[i] = np.corrcoef(b[start_b : start_b + L], a[start_a : start_a + L])[0, 1]
    return lags, np.array(vals)


def run_correlations_for_condition(
    beta0: np.ndarray,
    descriptors: dict[str, np.ndarray],
    lag_min: int,
    lag_max: int,
) -> list[dict]:
    """For one (stimulus, mesh) condition: compute all correlations. Returns list of row dicts for CSV."""
    n = min(len(beta0), min(len(v) for v in descriptors.values()))
    beta0 = beta0[:n]
    rows = []
    for name, feat in descriptors.items():
        feat = feat[:n]
        pearson_r, pearson_p = stats.pearsonr(beta0, feat)
        spearman_rho, spearman_p = stats.spearmanr(beta0, feat)
        lags, xcorr_vals = cross_correlation_at_lags(beta0, feat, lag_min, lag_max)
        valid = ~np.isnan(xcorr_vals)
        if np.any(valid):
            best_idx = np.nanargmax(np.abs(xcorr_vals))
            best_lag = int(lags[best_idx])
            cross_corr_best = float(xcorr_vals[best_idx])
        else:
            best_lag = 0
            cross_corr_best = np.nan
        rows.append({
            "audio_descriptor": name,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "best_lag": best_lag,
            "cross_corr_at_best_lag": cross_corr_best,
            "xcorr_lags": lags,
            "xcorr_vals": xcorr_vals,
        })
    return rows


def collect_all_data():
    """
    Load or compute β₀ and audio descriptors for A3 and A7.
    Returns: dict (audio_id, mesh_key) -> { "t_sec", "beta0", "descriptors", "descriptor_names" }
    mesh_key is "sphere" or "torus".
    """
    if not TEMPORAL_CSV.exists():
        raise FileNotFoundError("Run experiment.run_temporal_persistence first to create " + str(TEMPORAL_CSV))

    out = {}
    for audio_id in AUDIO_IDS:
        data = align_beta0_to_frames(TEMPORAL_CSV, audio_id)
        if data is None:
            continue
        t_sec, beta0_sphere, beta0_torus = data

        sig, sr = build_audio_10s(audio_id)
        frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
        n_align = min(len(t_sec), len(frames))
        t_sec = t_sec[:n_align]
        beta0_sphere = beta0_sphere[:n_align]
        beta0_torus = beta0_torus[:n_align]
        frames = frames[:n_align]
        t_sec = np.array([start / sr for _, start in frames])

        descriptors = compute_all_descriptors(sig, frames, sr)
        for key, arr in descriptors.items():
            descriptors[key] = arr[:n_align]

        out[(audio_id, "sphere")] = {
            "t_sec": t_sec,
            "beta0": beta0_sphere,
            "descriptors": descriptors,
        }
        out[(audio_id, "torus")] = {
            "t_sec": t_sec,
            "beta0": beta0_torus,
            "descriptors": descriptors,
        }
    return out


def main():
    all_data = collect_all_data()

    # 1) Build CSV rows
    csv_rows = []
    xcorr_store = {}  # (stimulus, mesh) -> list of (descriptor, lags, vals) for best descriptor per condition

    for (audio_id, mesh) in [(a, m) for a in AUDIO_IDS for m in MESH_IDS]:
        key = (audio_id, mesh)
        if key not in all_data:
            continue
        d = all_data[key]
        rows = run_correlations_for_condition(
            d["beta0"], d["descriptors"], LAG_MIN, LAG_MAX
        )
        for r in rows:
            csv_rows.append({
                "stimulus": audio_id,
                "mesh": mesh,
                "audio_descriptor": r["audio_descriptor"],
                "pearson_r": r["pearson_r"],
                "pearson_p": r["pearson_p"],
                "spearman_rho": r["spearman_rho"],
                "spearman_p": r["spearman_p"],
                "best_lag": r["best_lag"],
                "cross_corr_at_best_lag": r["cross_corr_at_best_lag"],
            })
        # Best descriptor by absolute Spearman for this condition
        best_row = max(rows, key=lambda x: abs(x["spearman_rho"]))
        xcorr_store[key] = {
            "best_descriptor": best_row["audio_descriptor"],
            "spearman_rho": best_row["spearman_rho"],
            "lags": best_row["xcorr_lags"],
            "vals": best_row["xcorr_vals"],
            "beta0": d["beta0"],
            "descriptor_vals": d["descriptors"][best_row["audio_descriptor"]],
            "t_sec": d["t_sec"],
        }

    # 2) Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "stimulus", "mesh", "audio_descriptor", "pearson_r", "pearson_p",
            "spearman_rho", "spearman_p", "best_lag", "cross_corr_at_best_lag",
        ])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print("Saved", OUT_CSV, file=sys.stderr)

    # 3) Figure: 2×2 temporal overlay
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    # rows: A3, A7; cols: sphere, torus
    for row, audio_id in enumerate(AUDIO_IDS):
        for col, mesh in enumerate(MESH_IDS):
            ax = axes[row, col]
            key = (audio_id, mesh)
            if key not in xcorr_store:
                ax.set_visible(False)
                continue
            s = xcorr_store[key]
            t = s["t_sec"]
            beta0 = s["beta0"]
            feat = s["descriptor_vals"]
            rho = s["spearman_rho"]
            name = s["best_descriptor"]

            ax.plot(t, beta0, color="C0", linewidth=1.0, label=r"$\beta_0$", alpha=0.9)
            ax_twin = ax.twinx()
            # Normalize feature to similar scale for overlay (optional: scale to beta0 range for visibility)
            feat_n = (feat - np.mean(feat)) / (np.std(feat) + 1e-12)
            ax_twin.plot(t, feat_n, color="C1", linewidth=0.8, alpha=0.8, label=name)
            ax_twin.set_ylabel(name, color="C1", fontsize=8)
            ax_twin.tick_params(axis="y", labelcolor="C1", labelsize=7)
            ax.set_ylabel(r"$\beta_0$", color="C0", fontsize=9)
            ax.tick_params(axis="y", labelcolor="C0")
            ax.set_title("{} — {} (Spearman $\\rho$ = {:.3f})".format(audio_id, mesh, rho), fontsize=9)
            ax.set_facecolor("white")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, t[-1] if len(t) else 10)

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    FIG_CORR.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_CORR, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", FIG_CORR, file=sys.stderr)

    # 4) Figure: cross-correlation curves
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    for (audio_id, mesh), s in xcorr_store.items():
        label = "{} — {}".format(audio_id, mesh)
        ax2.plot(s["lags"], s["vals"], "o-", label=label, markersize=4)
    ax2.set_xlabel("Lag (frames)")
    ax2.set_ylabel("Cross-correlation")
    ax2.set_title("β₀ vs best audio descriptor (by |Spearman ρ|)")
    ax2.legend(loc="best", fontsize=8)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("white")
    plt.tight_layout()
    fig2.savefig(FIG_XCORR, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", FIG_XCORR, file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
