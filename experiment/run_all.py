"""
Full experiment: for each (manifold × audio × mapping strategy) measure topology metrics -> results.csv.
Uses only meshes in data/eigen/*.npz. 7 audio, 3 strategies.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio, list_audio_ids
from audio.fft_analysis import compute_fft
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics
from analysis.symmetry import compute_symmetry

EIGEN_DIR = root / "data" / "eigen"
RESULTS_DIR = root / "data" / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
N_EIGEN = 50
STRATEGIES = ["direct", "mel", "energy"]

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
log = logging.getLogger(__name__)


def list_available_meshes():
    if not EIGEN_DIR.exists():
        return []
    return sorted([p.stem for p in EIGEN_DIR.glob("*.npz")])


def run_one(mesh_name, audio_id, strategy):
    data = np.load(EIGEN_DIR / f"{mesh_name}.npz", allow_pickle=True)
    V, F = data["vertices"], data["faces"]
    evecs = data["eigenvectors"]
    evals = data["eigenvalues"] if "eigenvalues" in data.files else None
    sig, sr = get_audio(audio_id)
    mag, _ = compute_fft(sig, sr)
    coef = map_fft_to_coefficients(mag, N_EIGEN, strategy=strategy, eigenvalues=evals)
    f = compute_scalar_field(evecs, coef)
    m = compute_topology_metrics(V, F, f)
    S = compute_symmetry(V, f)
    return {"beta0": m.beta0, "beta1": m.beta1, "chi": m.chi, "A_ratio": m.A_ratio, "S": S}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meshes = list_available_meshes()
    if not meshes:
        log.error("data/eigen/*.npz missing. Run step1.sh first.")
        sys.exit(1)
    audio_ids = list_audio_ids()
    rows = []
    total = len(meshes) * len(audio_ids) * len(STRATEGIES)
    with tqdm(total=total, desc="experiment", unit="run") as pbar:
        for mesh_name in meshes:
            for audio_id in audio_ids:
                for strategy in STRATEGIES:
                    try:
                        r = run_one(mesh_name, audio_id, strategy)
                        rows.append({
                            "mesh": mesh_name, "audio": audio_id, "strategy": strategy,
                            **r,
                        })
                    except Exception as e:
                        log.exception("%s %s %s: %s", mesh_name, audio_id, strategy, e)
                        raise
                    pbar.update(1)
    fieldnames = ["mesh", "audio", "strategy", "beta0", "beta1", "chi", "A_ratio", "S"]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log.info("Saved results.csv: %d rows -> %s", len(rows), RESULTS_CSV)
    print("Step 4 run_all done.", file=sys.stderr)


if __name__ == "__main__":
    main()
