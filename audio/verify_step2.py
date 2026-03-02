"""
Step 2 verification: 440Hz FFT, Direct mapping gives single eigenfunction; white noise uniform activation.
Check that 7 x 3 = 21 coefficient vectors are produced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio, list_audio_ids
from audio.fft_analysis import compute_fft, N_BINS
from mapping.spectral_mapping import map_fft_to_coefficients

N = 50


def main():
    print("=== 440Hz pure tone: FFT peak near 440Hz bin ===")
    sig, sr = get_audio("A1")
    mag, feats = compute_fft(sig, sr)
    bin_hz = sr / 2.0 / (N_BINS - 1) * np.arange(N_BINS)
    peak_bin = int(np.argmax(mag))
    print(f"  peak bin {peak_bin} -> ~{bin_hz[peak_bin]:.0f} Hz (expect ~440)")
    print("  ok" if 400 < bin_hz[peak_bin] < 500 else "  check")

    print("\n=== Direct mapping: pure tone activates one or two eigenfunctions strongly ===")
    coef = map_fft_to_coefficients(mag, N, strategy="direct")
    top = np.argsort(np.abs(coef))[-3:][::-1]
    print(f"  top 3 indices: {top}, values: {coef[top]}")
    # If 440Hz falls on bin boundary, two adjacent groups get energy -> one or two dominant is ok
    one_dominant = np.abs(coef[top[0]]) > 0.9
    two_dominant = np.abs(coef[top[0]]) + np.abs(coef[top[1]]) > 0.9 and np.abs(coef[top[2]]) < 0.1
    print("  ok" if (one_dominant or two_dominant) else "  check")

    print("\n=== White noise: roughly uniform activation of all eigenfunctions ===")
    sig, sr = get_audio("A5")
    mag, _ = compute_fft(sig, sr)
    coef = map_fft_to_coefficients(mag, N, strategy="direct")
    std = np.std(coef)
    print(f"  coefficient std {std:.4f} (expect small, relatively uniform)")
    print("  ok" if std < 0.2 else "  check")

    print("\n=== 7 audio x 3 mapping = 21 coefficient vectors ===")
    strategies = ["direct", "mel", "energy"]
    count = 0
    for aid in list_audio_ids():
        sig, sr = get_audio(aid)
        mag, _ = compute_fft(sig, sr)
        for s in strategies:
            c = map_fft_to_coefficients(mag, N, strategy=s)
            assert len(c) == N and np.abs(np.linalg.norm(c) - 1.0) < 0.01
            count += 1
    print(f"  {count} vectors ok (7 x 3 = 21)")
    print("Step 2 verify done.")


if __name__ == "__main__":
    main()
