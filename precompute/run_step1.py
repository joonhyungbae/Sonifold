"""
Step 1 pipeline: compute 50 LB eigenpairs for 8 meshes -> save .npz + export JSON.
Run from project root: python -m precompute.run_step1
      python -m precompute.run_step1 --sequential  (no parallel, sequential)
"""
from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from precompute.eigensolver import compute_eigen
from precompute.export_json import export_json
from precompute.mesh_library import get_mesh, list_mesh_names

N_EIGEN = 50
DATA_DIR = _root / "data"
EIGEN_DIR = DATA_DIR / "eigen"
MESHES_DIR = DATA_DIR / "meshes"
# Per-mesh parallel: at most (cores - 4), not more than task count (avoid too many workers)
def _max_workers(n_tasks: int) -> int:
    n_core = os.cpu_count() or 4
    return min(max(1, n_core - 4), max(1, n_tasks))

# Log to stderr (so it does not clash with tqdm progress bar)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger(__name__)


def _process_one(name: str, root_str: str) -> tuple[str, bool, str]:
    """Compute and save eigenvalues for one mesh. Runs in separate process so path/import are reset."""
    root = Path(root_str)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from precompute.eigensolver import compute_eigen
    from precompute.export_json import export_json
    from precompute.mesh_library import get_mesh

    eigen_dir = root / "data" / "eigen"
    try:
        V, F = get_mesh(name)
        evals, evecs = compute_eigen(V, F, N=N_EIGEN)
        npz_path = eigen_dir / f"{name}.npz"
        np.savez_compressed(
            npz_path,
            vertices=V,
            faces=F,
            eigenvalues=evals,
            eigenvectors=evecs,
        )
        json_path = eigen_dir / f"{name}.json"
        export_json(V, F, evals, evecs, json_path)
        return (name, True, f"{npz_path.name}, {json_path.name}")
    except Exception as e:
        return (name, False, str(e))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Step 1: LB eigen for all meshes")
    ap.add_argument("--sequential", action="store_true", help="Run one mesh at a time (no parallel)")
    args = ap.parse_args()

    EIGEN_DIR.mkdir(parents=True, exist_ok=True)
    MESHES_DIR.mkdir(parents=True, exist_ok=True)

    names = list_mesh_names()
    obj_only = ("bunny", "spot", "dragon", "armadillo")
    to_run = []
    for name in names:
        if name in obj_only and not (MESHES_DIR / f"{name}.obj").exists():
            continue
        to_run.append(name)

    log.info("Step 1: LB eigen pipeline (N=%d meshes)", len(to_run))
    if len(to_run) < len(names):
        missing = [n for n in obj_only if not (MESHES_DIR / f"{n}.obj").exists()]
        if missing:
            log.info("Optional meshes (add data/meshes/*.obj to include): %s", missing)
    if not to_run:
        log.error("No meshes to run. Add data/meshes/*.obj for bunny/spot/dragon/armadillo or ensure procedural meshes are available.")
        sys.exit(1)

    root_str = str(_root)
    if args.sequential or len(to_run) <= 1:
        log.info("Sequential mode: %d mesh(es)", len(to_run))
        for name in tqdm(to_run, desc="mesh", unit="mesh"):
            name2, ok, msg = _process_one(name, root_str)
            if ok:
                log.info("%s ok -> %s", name2, msg)
            else:
                log.error("%s failed: %s", name2, msg)
                raise RuntimeError(f"{name2}: {msg}")
    else:
        n_workers = _max_workers(len(to_run))
        log.info("Parallel mode: %d meshes, max_workers=%d", len(to_run), n_workers)
        log.info("Each mesh may take a few minutes; progress updates as each completes.")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_process_one, name, root_str): name for name in to_run}
            with tqdm(total=len(futures), desc="mesh", unit="mesh") as pbar:
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        name2, ok, msg = fut.result()
                        if ok:
                            log.info("%s ok -> %s", name2, msg)
                        else:
                            log.error("%s failed: %s", name2, msg)
                            raise RuntimeError(f"{name2}: {msg}")
                    except Exception as e:
                        log.exception("%s: %s", name, e)
                        raise
                    pbar.update(1)

    log.info("Step 1 done.")


if __name__ == "__main__":
    main()
