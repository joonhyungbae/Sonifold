# FFT analysis
import numpy as np
N_FFT = 2048
N_BINS = 1024


def compute_fft_frames(signal, sample_rate=44100, hop_length=512, n_fft=None):
    """
    Overlapping STFT frames for temporal analysis.
    signal: 1D float array.
    Returns: list of (magnitudes, start_sample_idx), each magnitudes shape (N_BINS,).
    """
    n_fft = n_fft or N_FFT
    x = np.asarray(signal, dtype=np.float64).ravel()
    n = len(x)
    frames = []
    start = 0
    while start + n_fft <= n:
        frame = x[start : start + n_fft] * np.hanning(n_fft)
        spec = np.fft.rfft(frame, n=n_fft)
        mag = np.abs(spec)[:N_BINS]
        frames.append((mag.astype(np.float64), start))
        start += hop_length
    return frames


def compute_fft(signal, sample_rate=44100):
    x = np.asarray(signal, dtype=np.float64).ravel()
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    frame = x[:N_FFT] * np.hanning(N_FFT)
    spec = np.fft.rfft(frame, n=N_FFT)
    magnitudes = np.abs(spec)[:N_BINS]
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)[:N_BINS]
    power = magnitudes * magnitudes
    total = power.sum()
    if total < 1e-20:
        features = {"spectral_centroid": 0.0, "bandwidth": 0.0, "spectral_flatness": 0.0, "rolloff": 0.0}
    else:
        sc = np.dot(freqs, power) / total
        bw = np.sqrt(np.dot((freqs - sc) ** 2, power) / total)
        n = len(power)
        gm = np.exp(np.mean(np.log(power + 1e-20)))
        am = total / n
        flat = gm / (am + 1e-20)
        cum = np.cumsum(power)
        idx = np.searchsorted(cum, 0.85 * total)
        rolloff = freqs[min(idx, n - 1)]
        features = {"spectral_centroid": float(sc), "bandwidth": float(bw), "spectral_flatness": float(flat), "rolloff": float(rolloff)}
    return magnitudes.astype(np.float64), features
