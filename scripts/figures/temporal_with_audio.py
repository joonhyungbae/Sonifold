"""
Visualize β₀ time series and audio events (RMS envelope, spectral flux) on the same time axis.
Reviewer response: shows which audio events correspond to β₀ variation.

Run from project root:
  python scripts/temporal_with_audio.py
Requires: data/results/results_temporal.csv (after experiment.run_temporal_persistence)
Output: figures/fig_temporal_with_audio.pdf (A3), figures/fig_temporal_with_audio_A7.pdf (A7)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

# STFT parameters match run_temporal_persistence / fft_analysis
from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft_frames, N_FFT

HOP_LENGTH = 512
TARGET_DURATION_SEC = 10.0
RESULTS_CSV = root / "data" / "results" / "results_temporal.csv"


def load_temporal_beta0(csv_path: Path, audio_id: str):
    """Load (t_sec, beta0_sphere, beta0_torus) from results_temporal.csv for given audio_id."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sphere = [(float(r["t_sec"]), int(r["beta0"])) for r in rows if r["audio_id"] == audio_id and r["mesh_id"] == "sphere_genus0"]
    torus = [(float(r["t_sec"]), int(r["beta0"])) for r in rows if r["audio_id"] == audio_id and r["mesh_id"] == "torus_genus1"]
    sphere.sort(key=lambda x: x[0])
    torus.sort(key=lambda x: x[0])
    if not sphere or not torus:
        return None
    # align by frame index (same length as in experiment)
    t_sec = [p[0] for p in sphere]
    beta0_s = [p[1] for p in sphere]
    beta0_t = [p[1] for p in torus]
    return np.array(t_sec), np.array(beta0_s), np.array(beta0_t)


def compute_rms_per_frame(signal: np.ndarray, frames: list, n_fft: int) -> np.ndarray:
    """RMS per STFT frame (same window boundaries as STFT)."""
    rms = []
    for mag, start in frames:
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
    """Load audio and tile/pad to TARGET_DURATION_SEC, same as run_temporal_persistence."""
    sig, sr = get_audio(audio_id)
    n_target = int(TARGET_DURATION_SEC * sr)
    if len(sig) < n_target:
        repeats = (n_target + len(sig) - 1) // len(sig)
        sig = np.tile(sig, repeats)[:n_target]
    else:
        sig = sig[:n_target]
    return sig, sr


def make_figure(audio_id: str, out_path: Path, title_label: str):
    if not RESULTS_CSV.exists():
        print("Run experiment.run_temporal_persistence first to create", RESULTS_CSV, file=sys.stderr)
        return 1
    data = load_temporal_beta0(RESULTS_CSV, audio_id)
    if data is None:
        print("No temporal data for", audio_id, file=sys.stderr)
        return 1
    t_sec, beta0_sphere, beta0_torus = data

    sig, sr = build_audio_10s(audio_id)
    frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
    if len(frames) != len(t_sec):
        print("Frame count mismatch: audio frames {} vs CSV {}".format(len(frames), len(t_sec)), file=sys.stderr)
        # Use minimum length to align
        n = min(len(frames), len(t_sec))
        frames = frames[:n]
        t_sec = t_sec[:n]
        beta0_sphere = beta0_sphere[:n]
        beta0_torus = beta0_torus[:n]

    t_sec = np.array([start / sr for _, start in frames])  # use same t_sec from frames
    rms = compute_rms_per_frame(sig, frames, N_FFT)
    flux = compute_spectral_flux(frames)

    # Correlation: sphere β₀ vs RMS envelope
    from scipy.stats import pearsonr, spearmanr
    r_pearson, p_pearson = pearsonr(beta0_sphere, rms)
    r_spearman, p_spearman = spearmanr(beta0_sphere, rms)
    corr_text = r"$\beta_0$ vs RMS: Pearson $r$={:.3f}, Spearman $\rho$={:.3f}".format(r_pearson, r_spearman)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)
    # Top: RMS envelope + spectral flux (secondary y-axis)
    ax_top.plot(t_sec, rms, color="C0", linewidth=1.2, label="RMS envelope")
    ax_top.set_ylabel("RMS", color="C0")
    ax_top.tick_params(axis="y", labelcolor="C0")
    ax_top.set_facecolor("white")
    ax_top.grid(True, alpha=0.3)

    ax2 = ax_top.twinx()
    ax2.plot(t_sec, flux, color="gray", linewidth=0.9, alpha=0.8, label="Spectral flux")
    ax2.set_ylabel("Spectral flux", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax_top.set_xlim(0, t_sec[-1] if len(t_sec) else 10)

    # Bottom: β₀ sphere (blue) and torus (orange)
    ax_bot.plot(t_sec, beta0_sphere, color="C0", linewidth=1.2, label="sphere")
    ax_bot.plot(t_sec, beta0_torus, color="C1", linewidth=1.2, label="torus")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel(r"$\beta_0$")
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.set_facecolor("white")
    ax_bot.grid(True, alpha=0.3)

    fig.suptitle("{} — {}".format(title_label, corr_text), fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", out_path, file=sys.stderr)
    return 0


def main():
    figures_dir = root / "figures"
    err = make_figure("A3", figures_dir / "fig_temporal_with_audio.pdf", "A3 (piano)")
    if err:
        return err
    err = make_figure("A7", figures_dir / "fig_temporal_with_audio_A7.pdf", "A7 (orchestral)")
    return err


if __name__ == "__main__":
    sys.exit(main() or 0)
