"""
Export experiment results and hypothesis test summary for the web demo.
Writes webapp/public/data/experiment.json and webapp/public/data/coefficients/{mesh}_{audio}.json
so the demo can show batch results and preview each experiment audio's effect (same coefficients as in the experiment).

Run from project root after step4:
  python -m experiment.export_for_web
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

RESULTS_CSV = root / "data" / "results" / "results.csv"
EIGEN_DIR = root / "data" / "eigen"
OUT_DIR = root / "webapp" / "public" / "data"
OUT_PATH = OUT_DIR / "experiment.json"
COEFF_DIR = OUT_DIR / "coefficients"
AUDIO_OUT_DIR = root / "webapp" / "public" / "audio"
N_EIGEN = 50
STRATEGIES = ["direct", "mel", "energy"]


def list_available_meshes():
    if not EIGEN_DIR.exists():
        return []
    return sorted([p.stem for p in EIGEN_DIR.glob("*.npz")])


def write_coefficient_files():
    """Export (mesh, audio) -> { direct, mel, energy } coefficient vectors for web demo."""
    from audio.audio_library import get_audio, list_audio_ids
    from audio.fft_analysis import compute_fft
    from mapping.spectral_mapping import map_fft_to_coefficients

    COEFF_DIR.mkdir(parents=True, exist_ok=True)
    meshes = list_available_meshes()
    audio_ids = list_audio_ids()
    for mesh_name in meshes:
        data = np.load(EIGEN_DIR / f"{mesh_name}.npz", allow_pickle=True)
        evals = data["eigenvalues"] if "eigenvalues" in data.files else None
        for audio_id in audio_ids:
            sig, sr = get_audio(audio_id)
            mag, _ = compute_fft(sig, sr)
            out = {}
            for strategy in STRATEGIES:
                coef = map_fft_to_coefficients(mag, N_EIGEN, strategy=strategy, eigenvalues=evals)
                out[strategy] = [float(x) for x in coef]
            path = COEFF_DIR / f"{mesh_name}_{audio_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f)
    print("Wrote coefficient files to", COEFF_DIR, file=sys.stderr)


def write_audio_wav_files():
    """Export A1–A7 as WAV for the web demo player."""
    from audio.audio_library import get_audio, list_audio_ids

    try:
        from scipy.io import wavfile
    except ImportError:
        print("scipy not available, skipping WAV export", file=sys.stderr)
        return

    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for audio_id in list_audio_ids():
        sig, sr = get_audio(audio_id)
        sig = np.clip(sig, -1.0, 1.0)
        samples = (sig * 32767).astype(np.int16)
        path = AUDIO_OUT_DIR / f"{audio_id}.wav"
        wavfile.write(path, sr, samples)
    print("Wrote WAV files to", AUDIO_OUT_DIR, file=sys.stderr)


def main():
    if not RESULTS_CSV.exists():
        print("results.csv not found. Run ./step4.sh first.", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    from experiment.hypothesis_test import h1_test, h2_test, h3_test

    df = pd.read_csv(RESULTS_CSV)
    h1 = h1_test(df)
    h2 = h2_test(df)
    h3 = h3_test(df)

    summary = {
        "h1": {"rho": h1["rho"], "p": h1["p"]},
        "h2": {"p": h2["p"], "high_mean": h2["high_mean"], "low_mean": h2["low_mean"]},
        "h3": {"r2": h3["r2"], "p": h3["p"]},
    }

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "mesh": r["mesh"],
            "audio": r["audio"],
            "strategy": r["strategy"],
            "beta0": int(r["beta0"]),
            "beta1": int(r["beta1"]),
            "chi": int(r["chi"]),
            "A_ratio": round(float(r["A_ratio"]), 4),
            "S": round(float(r["S"]), 4),
        })

    payload = {"summary": summary, "rows": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", OUT_PATH, file=sys.stderr)

    write_coefficient_files()
    write_audio_wav_files()


if __name__ == "__main__":
    main()
