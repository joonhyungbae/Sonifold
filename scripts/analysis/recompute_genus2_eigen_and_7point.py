"""
Recompute eigen for double_torus_genus2 (new mesh) and update 7-point / genus_extended CSVs.
Run from project root: python scripts/analysis/recompute_genus2_eigen_and_7point.py
If CuPy is not available, set USE_GPU=0 or the script will use CPU.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

# Use CPU when CuPy is not available (eigensolver checks USE_GPU)
if os.environ.get("USE_GPU", "") != "1":
    os.environ["USE_GPU"] = "0"
else:
    try:
        import cupy  # noqa: F401
    except ImportError:
        os.environ["USE_GPU"] = "0"

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

EXPERIMENTS_DIR = root / "data" / "experiments"
EIGEN_DIR = root / "data" / "experiments" / "eigen"
RESULTS_DIR = root / "data" / "results"
MESH_ID = "double_torus_genus2"
N_EIGEN = 50
AUDIO_ID = "A3"
STRATEGY = "direct"


def load_obj(path: Path):
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)


def compute_and_save_eigen_force():
    """Recompute eigen for double_torus_genus2 and overwrite npz/json."""
    from precompute.eigensolver import compute_eigen
    from precompute.export_json import export_json

    obj_path = EXPERIMENTS_DIR / (MESH_ID + ".obj")
    npz_path = EIGEN_DIR / (MESH_ID + ".npz")
    json_path = EIGEN_DIR / (MESH_ID + ".json")
    if not obj_path.exists():
        raise FileNotFoundError(f"Mesh not found: {obj_path}")

    V, F = load_obj(obj_path)
    print(f"[eigen] Computing K={N_EIGEN} for {MESH_ID} (V={V.shape[0]})...", file=sys.stderr)
    evals, evecs = compute_eigen(V, F, N=N_EIGEN)
    EIGEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, vertices=V, faces=F, eigenvalues=evals, eigenvectors=evecs)
    export_json(V, F, evals, evecs, json_path)
    print(f"[eigen] Saved {npz_path}", file=sys.stderr)


def run_beta0_pipeline():
    """A3 + direct → beta0, A_ratio, S."""
    from scripts.generate_genus_extended import load_eigen
    from audio.audio_library import get_audio
    from audio.fft_analysis import compute_fft
    from mapping.spectral_mapping import map_fft_to_coefficients
    from analysis.scalar_field import compute_scalar_field
    from analysis.nodal_surface import compute_topology_metrics
    from analysis.symmetry import compute_symmetry

    out = load_eigen(MESH_ID)
    if out is None:
        return None
    V, F, evecs, evals = out
    sig, sr = get_audio(AUDIO_ID)
    mag, _ = compute_fft(sig, sr)
    coef = map_fft_to_coefficients(mag, N_EIGEN, strategy=STRATEGY, eigenvalues=evals)
    f = compute_scalar_field(evecs, coef)
    m = compute_topology_metrics(V, F, f)
    S = compute_symmetry(V, f)
    return {"beta0": int(round(m.beta0)), "A_ratio": m.A_ratio, "S": S}


def update_csv(csv_path: Path, new_row: dict, genus_col="genus", genus_val=2):
    """Replace row with genus=2 by new_row; keep others."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        for row in r:
            try:
                g = int(float(row.get(genus_col, -1)))
                if g == genus_val:
                    rows.append(new_row)
                else:
                    rows.append(row)
            except (ValueError, TypeError):
                rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] Updated {csv_path}", file=sys.stderr)


def main():
    compute_and_save_eigen_force()
    res = run_beta0_pipeline()
    if res is None:
        print("run_beta0_pipeline failed", file=sys.stderr)
        return 1

    new_row = {
        "mesh_id": MESH_ID,
        "genus": 2,
        "beta0": res["beta0"],
        "A_ratio": res["A_ratio"],
        "S": res["S"],
    }
    print(f"[7point] double_torus_genus2: beta0={res['beta0']} A_ratio={res['A_ratio']:.6f} S={res['S']:.6f}", file=sys.stderr)

    # Update 7-point CSV (replace genus-2 row)
    path_7 = RESULTS_DIR / "results_genus_extended_7point.csv"
    if path_7.exists():
        update_csv(path_7, new_row)
    else:
        # Create from scratch with only genus 2 (unusual)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(path_7, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["mesh_id", "genus", "beta0", "A_ratio", "S"])
            w.writeheader()
            w.writerow(new_row)
        print(f"[csv] Created {path_7}", file=sys.stderr)

    # Update results_genus_extended.csv (genus-2 row)
    path_ext = RESULTS_DIR / "results_genus_extended.csv"
    if path_ext.exists():
        update_csv(path_ext, new_row)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
