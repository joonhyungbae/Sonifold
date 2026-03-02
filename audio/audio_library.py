"""
Generate or load 7 test audio clips.
Each 1s, 44100Hz, mono. If .wav missing in data/audio/, generate when possible else placeholder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

_SR = 44100
_DURATION = 1.0
_ROOT = Path(__file__).resolve().parent.parent
_AUDIO_DIR = _ROOT / "data" / "audio"


def _t() -> np.ndarray:
    return np.linspace(0, _DURATION, int(_SR * _DURATION), endpoint=False)


def get_audio(audio_id: str) -> Tuple[np.ndarray, int]:
    """
    audio_id: A1 | A2 | A3 | A4 | A5 | A6 | A7
    Returns: (signal[44100], sample_rate=44100), mono, float [-1,1] recommended.
    """
    audio_id = audio_id.strip().upper()
    n = int(_SR * _DURATION)
    t = _t()

    if audio_id == "A1":
        # Pure tone 440Hz
        return (np.sin(2 * np.pi * 440.0 * t).astype(np.float64), _SR)
    if audio_id == "A2":
        # C major triad: C4=261.6, E4=329.6, G4=392.0
        s = np.sin(2 * np.pi * 261.6 * t) + np.sin(2 * np.pi * 329.6 * t) + np.sin(2 * np.pi * 392.0 * t)
        s = s / 3.0
        return (s.astype(np.float64), _SR)
    if audio_id == "A3":
        # Piano chord .wav
        p = _AUDIO_DIR / "piano_cmajor.wav"
        return _load_wav_or_placeholder(p, "A3 piano", fallback_hz=261.6)
    if audio_id == "A4":
        # Voice vowel
        p = _AUDIO_DIR / "voice_ah.wav"
        return _load_wav_or_placeholder(p, "A4 voice", fallback_hz=440.0)
    if audio_id == "A5":
        # White noise
        rng = np.random.default_rng(42)
        s = rng.standard_normal(n)
        s = s / (np.max(np.abs(s)) + 1e-12)
        return (s.astype(np.float64), _SR)
    if audio_id == "A6":
        # Pink noise: 1/f filter
        rng = np.random.default_rng(43)
        white = rng.standard_normal(n)
        n_fft = 2 ** int(np.ceil(np.log2(n)))
        W = np.fft.rfft(white, n=n_fft)
        freqs = np.arange(len(W), dtype=float) + 1
        W = W / np.sqrt(freqs)
        pink = np.fft.irfft(W, n=n_fft)[:n]
        pink = pink / (np.max(np.abs(pink)) + 1e-12)
        return (pink.astype(np.float64), _SR)
    if audio_id == "A7":
        # Orchestra .wav
        p = _AUDIO_DIR / "orchestra_excerpt.wav"
        return _load_wav_or_placeholder(p, "A7 orchestra", fallback_hz=440.0)

    raise ValueError(f"Unknown audio_id: {audio_id}")


def _load_wav_or_placeholder(
    path: Path, label: str, fallback_hz: float = 440.0
) -> Tuple[np.ndarray, int]:
    if path.exists():
        try:
            import scipy.io.wavfile as wav
            sr, data = wav.read(path)
            if data.dtype.kind in "iu":
                data = data.astype(np.float64) / np.iinfo(data.dtype).max
            if data.ndim > 1:
                data = data.mean(axis=1)
            n_target = int(_SR * _DURATION)
            if len(data) != n_target or sr != _SR:
                from scipy.signal import resample
                data = resample(data, n_target)
            return (data.astype(np.float64), _SR)
        except Exception as e:
            import warnings
            warnings.warn(f"{label} load failed ({path}): {e}; using placeholder.")
    n = int(_SR * _DURATION)
    t = _t()
    placeholder = np.sin(2 * np.pi * fallback_hz * t).astype(np.float64) * 0.5
    return (placeholder, _SR)


def list_audio_ids() -> list[str]:
    return ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
