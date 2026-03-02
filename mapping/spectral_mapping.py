"""
FFT spectrum 1024 bins -> N eigenfunction coefficients. L2 normalized.
Strategies: direct | mel | energy
"""
import numpy as np

N_FFT_BINS = 1024


def map_fft_to_coefficients(fft_magnitudes, N, strategy="direct", eigenvalues=None):
    """
    fft_magnitudes: shape (1024,) or (1024,). N: number of eigenfunctions.
    strategy: direct | mel | energy.
    eigenvalues: unused (kept for API compatibility). Energy strategy uses mel-band sum of squared magnitudes only.
    Returns: coefficients[N], L2 normalized.
    """
    m = np.asarray(fft_magnitudes, dtype=np.float64).ravel()
    if len(m) != N_FFT_BINS:
        m = np.pad(m, (0, max(0, N_FFT_BINS - len(m))))[:N_FFT_BINS]
    if strategy == "direct":
        groups = np.array_split(m, N)
        coef = np.array([g.mean() for g in groups], dtype=np.float64)
    elif strategy == "mel":
        # Mel scale: N band edges (0..1024 to mel then back to bin indices)
        mel_edges = _mel_band_edges(N)
        coef = np.zeros(N, dtype=np.float64)
        for i in range(N):
            lo, hi = int(mel_edges[i]), int(mel_edges[i + 1])
            hi = min(hi, N_FFT_BINS)
            if hi > lo:
                coef[i] = m[lo:hi].mean()
            else:
                coef[i] = m[lo] if lo < N_FFT_BINS else 0.0
    elif strategy == "energy":
        # Paper: "Each coefficient is the total energy (sum of squared magnitudes) in the corresponding mel band."
        mel_edges = _mel_band_edges(N)
        coef = np.zeros(N, dtype=np.float64)
        for i in range(N):
            lo, hi = int(mel_edges[i]), int(mel_edges[i + 1])
            hi = min(hi, N_FFT_BINS)
            if hi > lo:
                coef[i] = np.sum(m[lo:hi] ** 2)
            else:
                coef[i] = (m[lo] ** 2) if lo < N_FFT_BINS else 0.0
        coef = coef / (np.linalg.norm(coef) + 1e-20)
        return coef
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    nrm = np.linalg.norm(coef)
    if nrm > 1e-20:
        coef = coef / nrm
    return coef


def _mel_band_edges(N):
    """N+1 mel-scale band edges over 0..N_FFT_BINS."""
    low_mel = 0.0
    high_mel = 2595.0 * np.log10(1.0 + (N_FFT_BINS - 1) / 700.0)
    mel_pts = np.linspace(low_mel, high_mel, N + 1)
    freq_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bin_edges = np.clip(freq_pts / (N_FFT_BINS - 1) * (N_FFT_BINS - 1), 0, N_FFT_BINS - 1)
    bin_edges[-1] = N_FFT_BINS
    return bin_edges
