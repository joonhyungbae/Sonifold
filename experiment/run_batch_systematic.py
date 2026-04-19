"""
Batch analysis: apply same audio (A3 piano) + Direct Mapping to data/experiments/ meshes,
extract beta0, S -> save to results_systematic.csv.
Log whether beta0 increases with genus and S decreases with symmetry breaking.

Run from project root: python -m experiment.run_batch_systematic
Requires: run experiment.generate_experiment_meshes first, then experiment/precompute_experiment_eigen.py for eigen.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics
from analysis.symmetry import compute_symmetry

EXPERIMENTS_DIR = root / "data" / "experiments"
EIGEN_EXPERIMENTS_DIR = root / "data" / "experiments" / "eigen"
RESULTS_DIR = root / "data" / "results"
RESULTS_CSV = RESULTS_DIR / "results_systematic.csv"
N_EIGEN = 50
AUDIO_ID = "A3"  # piano
STRATEGY = "direct"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def load_mesh_obj(path: Path):
    """Load vertices and faces from .obj file."""
    import trimesh
    m = trimesh.load(path, force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)


def load_eigen_for_mesh(mesh_id: str):
    """Load precomputed eigenbasis from data/experiments/eigen/{mesh_id}.npz or .json."""
    npz_path = EIGEN_EXPERIMENTS_DIR / (mesh_id + ".npz")
    json_path = EIGEN_EXPERIMENTS_DIR / (mesh_id + ".json")
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        V = np.asarray(data["vertices"], dtype=np.float64)
        F = np.asarray(data["faces"], dtype=np.int32)
        evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
        evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
        return V, F, evecs, evals
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        V = np.array(data["vertices"], dtype=np.float64)
        F = np.array(data["faces"], dtype=np.int32)
        evecs = np.array(data["eigenvectors"], dtype=np.float64)
        evals = np.array(data.get("eigenvalues", []), dtype=np.float64)
        return V, F, evecs, evals
    raise FileNotFoundError("No eigen data for {}".format(mesh_id))


def run_one(mesh_id: str, genus=None, asymmetry_level=None):
    """Compute beta0 and S for one mesh with A3 + direct mapping."""
    V, F, evecs, evals = load_eigen_for_mesh(mesh_id)
    sig, sr = get_audio(AUDIO_ID)
    mag, _ = compute_fft(sig, sr)
    coef = map_fft_to_coefficients(mag, N_EIGEN, strategy=STRATEGY, eigenvalues=evals)
    f = compute_scalar_field(evecs, coef)
    m = compute_topology_metrics(V, F, f)
    S = compute_symmetry(V, f)
    return {
        "mesh_id": mesh_id,
        "genus": genus,
        "asymmetry_level": asymmetry_level,
        "beta0": m.beta0,
        "A_ratio": m.A_ratio,
        "S": S,
    }


def main():
    if not EIGEN_EXPERIMENTS_DIR.exists():
        log.error(
            "data/experiments/eigen/ not found. Run experiment/precompute_experiment_eigen.py first "
            "(after generate_experiment_meshes)."
        )
        sys.exit(1)

    # Discover mesh IDs from eigen folder (stem of .npz or .json)
    mesh_ids = set()
    for p in EIGEN_EXPERIMENTS_DIR.glob("*.npz"):
        mesh_ids.add(p.stem)
    for p in EIGEN_EXPERIMENTS_DIR.glob("*.json"):
        mesh_ids.add(p.stem)
    mesh_ids = sorted(mesh_ids)
    if not mesh_ids:
        log.error("No .npz or .json in data/experiments/eigen/")
        sys.exit(1)

    # Assign genus / asymmetry for logging
    def meta(mid):
        if "genus0" in mid:
            return 0, None
        if "genus1" in mid:
            return 1, None
        if "genus2" in mid:
            return 2, None
        if "genus3" in mid:
            return 3, None
        if "genus4" in mid:
            return 4, None
        if "genus5" in mid:
            return 5, None
        if "genus6" in mid:
            return 6, None
        if "octahedron_sym" in mid:
            try:
                stretch = float(mid.replace("octahedron_sym_", ""))
                return None, stretch
            except ValueError:
                pass
        return None, None

    rows = []
    genus_beta0 = []
    asym_s = []

    for mesh_id in tqdm(mesh_ids, desc="batch", unit="mesh"):
        genus, asym = meta(mesh_id)
        try:
            r = run_one(mesh_id, genus=genus, asymmetry_level=asym)
            rows.append(r)
            if genus is not None:
                genus_beta0.append((genus, r["beta0"]))
            if asym is not None:
                asym_s.append((asym, r["S"]))
        except Exception as e:
            log.exception("%s: %s", mesh_id, e)
            raise

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mesh_id", "genus", "asymmetry_level", "beta0", "A_ratio", "S"]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    log.info("Saved %d rows -> %s", len(rows), RESULTS_CSV)

    # Log trends for genus vs beta0 and asymmetry vs S
    if genus_beta0:
        genus_beta0.sort(key=lambda x: x[0])
        log.info("[Genus vs beta0] genus | beta0: %s", genus_beta0)
        if len(genus_beta0) >= 2:
            beta0_vals = [x[1] for x in genus_beta0]
            if beta0_vals == sorted(beta0_vals):
                log.info("[Genus] beta0 increases (or stable) as genus increases (consistent with shape-as-filter).")
            else:
                log.info("[Genus] beta0 trend (check): %s", beta0_vals)
    if asym_s:
        asym_s.sort(key=lambda x: x[0])
        log.info("[Asymmetry vs S] stretch | S: %s", asym_s)
        S_vals = [x[1] for x in asym_s]
        if len(S_vals) >= 2 and S_vals[0] >= S_vals[-1]:
            log.info("[Symmetry breaking] S decreases as stretch increases (symmetric -> lower S).")
        else:
            log.info("[Symmetry breaking] S trend (check): %s", S_vals)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
