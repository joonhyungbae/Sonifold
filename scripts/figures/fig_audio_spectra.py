"""
Generate audio stimuli spectrograms (A1--A7) for Section 4.2 reference.
Run from project root: python figures/fig_audio_spectra.py
Output: figures/fig_audio_spectra.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio, list_audio_ids

AUDIO_LABELS = {
    "A1": "A1: pure tone 440 Hz",
    "A2": "A2: C major triad",
    "A3": "A3: piano",
    "A4": "A4: voice",
    "A5": "A5: white noise",
    "A6": "A6: pink noise",
    "A7": "A7: orchestra",
}


def stft_spectrogram(signal: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256):
    """Magnitude spectrogram (time × freq)."""
    from scipy import signal as scipy_signal
    f, t, Sxx = scipy_signal.spectrogram(
        signal, sr, nperseg=n_fft, noverlap=n_fft - hop
    )
    return f, t, np.abs(Sxx)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    audio_ids = list_audio_ids()
    n_fft, hop = 1024, 256

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    axes = axes.flatten()
    for i, aid in enumerate(audio_ids):
        sig, sr = get_audio(aid)
        f, t, Sxx = stft_spectrogram(sig, sr, n_fft=n_fft, hop=hop)
        ax = axes[i]
        ax.pcolormesh(t, f, Sxx, shading="auto", cmap="viridis", rasterized=True)
        ax.set_ylabel("Hz")
        ax.set_title(AUDIO_LABELS.get(aid, aid))
        ax.set_ylim(0, min(8000, sr // 2))
    axes[7].axis("off")
    for ax in axes[:7]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Audio stimuli: magnitude spectrograms", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "fig_audio_spectra.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
