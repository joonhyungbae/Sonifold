"""
Compute 50 LB eigenpairs for data/experiments/*.obj meshes -> save to data/experiments/eigen/*.npz, *.json.
Required before run_batch_systematic.py.

Run from project root: python -m experiment.precompute_experiment_eigen
Speed: --genus-only to compute only genus 3,4,5,6 meshes (4 only). USE_GPU=1 for CuPy GPU. EIGEN_TOL=1e-3 for faster convergence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from precompute.eigensolver import compute_eigen
from precompute.export_json import export_json

EXPERIMENTS_DIR = root / "data" / "experiments"
EIGEN_DIR = root / "data" / "experiments" / "eigen"
N_EIGEN = 50

GENUS_ONLY_STEMS = ("triple_torus_genus3", "quad_torus_genus4", "penta_torus_genus5", "hex_torus_genus6")


def load_obj(path):
    import trimesh
    m = trimesh.load(path, force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(description="Precompute LB eigen for experiment meshes.")
    ap.add_argument("--genus-only", action="store_true", help="Only compute genus 3,4,5,6 (4 meshes); much faster.")
    args = ap.parse_args()

    EIGEN_DIR.mkdir(parents=True, exist_ok=True)
    obj_files = sorted(EXPERIMENTS_DIR.glob("*.obj"))
    if not obj_files:
        print("No .obj in data/experiments/. Run experiment.generate_experiment_meshes first.", file=sys.stderr)
        sys.exit(1)

    if args.genus_only:
        obj_files = [p for p in obj_files if p.stem in GENUS_ONLY_STEMS]
        if not obj_files:
            print("No genus 3–6 .obj found.", file=sys.stderr)
            sys.exit(1)
        print("Genus-only mode: {} meshes".format(len(obj_files)), file=sys.stderr)

    for path in tqdm(obj_files, desc="eigen", unit="mesh"):
        mesh_id = path.stem
        try:
            V, F = load_obj(path)
            evals, evecs = compute_eigen(V, F, N=N_EIGEN)
            np.savez_compressed(
                EIGEN_DIR / (mesh_id + ".npz"),
                vertices=V,
                faces=F,
                eigenvalues=evals,
                eigenvectors=evecs,
            )
            export_json(V, F, evals, evecs, EIGEN_DIR / (mesh_id + ".json"))
        except Exception as e:
            print("{} failed: {}".format(mesh_id, e), file=sys.stderr)
            raise

    print("Done. Eigen data in", EIGEN_DIR, file=sys.stderr)


if __name__ == "__main__":
    main()
