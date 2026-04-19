"""
FFT spectrum 1024 bins -> N eigenfunction coefficients. L2 normalized.
Strategies: direct | mel | energy | freqmatch (revision round 1, see Section 5 of paper)
"""
import numpy as np

N_FFT_BINS = 1024
SAMPLE_RATE_DEFAULT = 44100
N_FFT_DEFAULT = 2048


def map_fft_to_coefficients(
    fft_magnitudes,
    N,
    strategy="direct",
    eigenvalues=None,
    sample_rate=SAMPLE_RATE_DEFAULT,
    n_fft=N_FFT_DEFAULT,
):
    """
    fft_magnitudes: shape (1024,) or (1024,). N: number of eigenfunctions.
    strategy: direct | mel | energy | freqmatch.
    eigenvalues: required for strategy='freqmatch'; otherwise ignored. Must be
        length N (in the eigenvalue units returned by compute_eigen).
    sample_rate, n_fft: used only by 'freqmatch' to convert FFT bin index to Hz.
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
    elif strategy == "freqmatch":
        # Revision round 1 ablation (paper §5, ordinal vs dimensional mapping).
        # For each eigenmode k with eigenvalue λ_k, assign the FFT magnitude at
        # the bin whose center frequency is closest (in log scale) to an
        # ``audio analogue'' of sqrt(λ_k). Since λ_k lives in mesh-intrinsic
        # units unrelated to Hz, we first affine-map [sqrt(λ_1), sqrt(λ_N)]
        # onto [f_min, f_max] = [50 Hz, sample_rate/2] in log-frequency,
        # then pick the nearest FFT bin.
        if eigenvalues is None:
            raise ValueError("strategy='freqmatch' requires eigenvalues")
        lam = np.asarray(eigenvalues, dtype=np.float64).ravel()
        if lam.shape[0] != N:
            raise ValueError(
                f"eigenvalues length {lam.shape[0]} != N={N}"
            )
        sqrt_lam = np.sqrt(np.maximum(lam, 1e-20))
        lo, hi = float(sqrt_lam.min()), float(sqrt_lam.max())
        if hi <= lo:
            # Degenerate spectrum — fall back to direct.
            groups = np.array_split(m, N)
            coef = np.array([g.mean() for g in groups], dtype=np.float64)
        else:
            f_min, f_max = 50.0, sample_rate / 2.0
            log_lo, log_hi = np.log(lo), np.log(hi)
            log_fmin, log_fmax = np.log(f_min), np.log(f_max)
            log_sqrt = np.log(sqrt_lam)
            # Affine in log coordinates
            target_hz = np.exp(
                log_fmin
                + (log_sqrt - log_lo) * (log_fmax - log_fmin) / (log_hi - log_lo)
            )
            # FFT bin centers (assuming rfft over n_fft samples)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)[:N_FFT_BINS]
            # For each target_hz, find nearest bin index and pull its magnitude
            bin_idx = np.searchsorted(freqs, target_hz)
            bin_idx = np.clip(bin_idx, 1, N_FFT_BINS - 1)
            # pick nearer of bin_idx-1 and bin_idx
            left_dist = np.abs(np.log(freqs[bin_idx - 1] + 1e-20) - np.log(target_hz))
            right_dist = np.abs(np.log(freqs[bin_idx] + 1e-20) - np.log(target_hz))
            use_left = left_dist < right_dist
            bin_idx = np.where(use_left, bin_idx - 1, bin_idx)
            coef = m[bin_idx]
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
