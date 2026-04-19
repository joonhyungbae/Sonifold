"""Precompute eigen for genus 3 and 4 only; append beta0 to results_genus_extended.csv."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics
from analysis.symmetry import compute_symmetry
from precompute.eigensolver import compute_eigen
from precompute.export_json import export_json

EXPERIMENTS_DIR = root / "data" / "experiments"
EIGEN_DIR = root / "data" / "experiments" / "eigen"
RESULTS_EXTENDED = root / "data" / "results" / "results_genus_extended.csv"
N_EIGEN = 50
AUDIO_ID = "A3"
STRATEGY = "direct"

GENUS34 = [("triple_torus_genus3", 3), ("quad_torus_genus4", 4)]


def load_obj(path):
    import trimesh
    m = trimesh.load(path, force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)


def main():
    EIGEN_DIR.mkdir(parents=True, exist_ok=True)
    new_rows = []
    for mesh_id, genus in GENUS34:
        npz_path = EIGEN_DIR / (mesh_id + ".npz")
        if not npz_path.exists():
            obj_path = EXPERIMENTS_DIR / (mesh_id + ".obj")
            if not obj_path.exists():
                print("Skip", mesh_id, file=sys.stderr)
                continue
            print("Precomputing eigen for", mesh_id, file=sys.stderr)
            V, F = load_obj(obj_path)
            evals, evecs = compute_eigen(V, F, N=N_EIGEN)
            np.savez_compressed(npz_path, vertices=V, faces=F, eigenvalues=evals, eigenvectors=evecs)
            export_json(V, F, evals, evecs, EIGEN_DIR / (mesh_id + ".json"))
        data = np.load(npz_path, allow_pickle=True)
        V = np.asarray(data["vertices"], dtype=np.float64)
        F = np.asarray(data["faces"], dtype=np.int32)
        evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
        evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
        sig, sr = get_audio(AUDIO_ID)
        mag, _ = compute_fft(sig, sr)
        coef = map_fft_to_coefficients(mag, N_EIGEN, strategy=STRATEGY, eigenvalues=evals)
        f = compute_scalar_field(evecs, coef)
        m = compute_topology_metrics(V, F, f)
        S = compute_symmetry(V, f)
        beta0_val = int(round(m.beta0))
        new_rows.append({"mesh_id": mesh_id, "genus": genus, "beta0": beta0_val, "A_ratio": m.A_ratio, "S": S})
        print(mesh_id, "beta0=", beta0_val, file=sys.stderr)
    if not new_rows:
        return 1
    existing = []
    if RESULTS_EXTENDED.exists():
        with open(RESULTS_EXTENDED, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            fieldnames = r.fieldnames
            existing = list(r)
    else:
        fieldnames = ["mesh_id", "genus", "beta0", "A_ratio", "S"]
    genus_seen = {int(float(r["genus"])) for r in existing}
    for row in new_rows:
        if row["genus"] not in genus_seen:
            existing.append(row)
            genus_seen.add(row["genus"])
    existing.sort(key=lambda r: int(float(r["genus"])))
    RESULTS_EXTENDED.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_EXTENDED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(existing)
    print("Saved", RESULTS_EXTENDED, len(existing), "rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
