"""
Genus-2 mesh check and fix.
1) Report V, E, F, χ for main exploration mesh and for experiments double_torus_genus2.obj.
2) If needed, create valid genus-2 mesh (χ = -2) via boolean union of two tori, save to
   data/meshes/double_torus.obj and data/experiments/double_torus_genus2.obj (~5k).
Run from project root: python scripts/analysis/genus2_mesh_check_and_fix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))


def count_edges(F: np.ndarray) -> int:
    e0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    e1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    edges = np.sort(np.column_stack([e0, e1]), axis=1)
    keys = edges[:, 0].astype(np.int64) * (int(F.max()) + 1) + edges[:, 1]
    return int(len(np.unique(keys)))


def euler_char(V: np.ndarray, F: np.ndarray) -> int:
    nv = V.shape[0]
    nf = F.shape[0]
    ne = count_edges(F)
    return int(nv - ne + nf)


def load_experiments_genus2():
    """Load data/experiments/double_torus_genus2.obj."""
    import trimesh
    path = root / "data" / "experiments" / "double_torus_genus2.obj"
    if not path.exists():
        return None, None
    m = trimesh.load(str(path), force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int32)
    if F.shape[1] != 3:
        return None, None
    return V, F


def main_exploration_mesh():
    """Mesh used in Sections 4.1–4.3, fig2, fig3: mesh_library.get_mesh('double_torus')."""
    from precompute import mesh_library
    V, F = mesh_library.get_mesh("double_torus")
    return V, F


def main():
    results = []

    # ---- 1) Main exploration mesh (used in fig2, fig3, etc.) ----
    print("=== 1) Main exploration mesh (mesh_library.get_mesh('double_torus')) ===")
    try:
        V, F = main_exploration_mesh()
        nv, nf = V.shape[0], F.shape[0]
        ne = count_edges(F)
        chi = euler_char(V, F)
        expected = 2 - 2 * 2  # -2 for genus 2
        main_valid = chi == expected
        print(f"  V = {nv}, E = {ne}, F = {nf}, χ = V−E+F = {chi} (expected {expected})")
        print(f"  Main exploration mesh valid (χ = -2): {'yes' if main_valid else 'no'}")
        results.append(("main_exploration", nv, ne, nf, chi, main_valid))
    except Exception as e:
        print(f"  Error: {e}")
        results.append(("main_exploration", None, None, None, None, False))

    # Check source: does data/meshes/double_torus.obj exist?
    meshes_dir = root / "data" / "meshes"
    has_double_torus_obj = (meshes_dir / "double_torus.obj").exists()
    print(f"  data/meshes/double_torus.obj exists: {has_double_torus_obj}")
    if not has_double_torus_obj:
        print("  → Main exploration currently uses FALLBACK (single torus → χ=0).")

    # ---- 2) Genus-sequence mesh (eigenvalue_convergence_test_all_genus, etc.) ----
    print("\n=== 2) Genus-sequence mesh (data/experiments/double_torus_genus2.obj) ===")
    V2, F2 = load_experiments_genus2()
    if V2 is None:
        print("  File not found or invalid.")
        results.append(("experiments_genus2", None, None, None, None, False))
    else:
        nv, nf = V2.shape[0], F2.shape[0]
        ne = count_edges(F2)
        chi = euler_char(V2, F2)
        expected = -2
        seq_valid = chi == expected
        print(f"  V = {nv}, E = {ne}, F = {nf}, χ = {chi} (expected {expected})")
        print(f"  Genus-sequence mesh valid (χ = -2): {'yes' if seq_valid else 'no'}")
        results.append(("experiments_genus2", nv, ne, nf, chi, seq_valid))

    return results


def create_and_save_genus2():
    """
    Create valid genus-2 mesh via boolean union of two tori.
    Save to data/meshes/double_torus.obj (~10k) and data/experiments/double_torus_genus2.obj (~5k).
    Returns (success, info_dict).
    """
    from experiment.generate_experiment_meshes import (
        _high_genus_by_tori,
        _remesh_to_target,
        save_obj,
    )
    TARGET_5K = 5000
    TARGET_10K = 10000
    meshes_dir = root / "data" / "meshes"
    experiments_dir = root / "data" / "experiments"

    print("\n=== Creating valid genus-2 mesh (boolean union of two tori) ===")
    try:
        V, F = _high_genus_by_tori(2, R=2.0, r=0.55, n_ring=50)
    except Exception as e:
        print(f"  Boolean union failed: {e}")
        return False, {}

    chi = euler_char(V, F)
    nv0, ne0, nf0 = V.shape[0], count_edges(F), F.shape[0]
    print(f"  Raw mesh: V={nv0}, E={ne0}, F={nf0}, χ={chi}")
    if chi != -2:
        print(f"  ERROR: expected χ=-2, got {chi}. Aborting.")
        return False, {}

    # ~5k for experiments (genus-sequence)
    V5, F5 = _remesh_to_target(V, F, TARGET_5K)
    chi5 = euler_char(V5, F5)
    if chi5 != -2:
        print(f"  WARNING: after 5k remesh χ={chi5} (expected -2).")
    path_5k = experiments_dir / "double_torus_genus2.obj"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    save_obj(V5, F5, path_5k)
    print(f"  Saved ~5k: {path_5k}  V={V5.shape[0]}, χ={chi5}")

    # ~10k for main exploration (data/meshes/double_torus.obj)
    V10, F10 = _remesh_to_target(V, F, TARGET_10K)
    chi10 = euler_char(V10, F10)
    if chi10 != -2:
        print(f"  WARNING: after 10k remesh χ={chi10} (expected -2).")
    meshes_dir.mkdir(parents=True, exist_ok=True)
    path_10k = meshes_dir / "double_torus.obj"
    save_obj(V10, F10, path_10k)
    print(f"  Saved ~10k: {path_10k}  V={V10.shape[0]}, χ={chi10}")

    def mean_aspect_ratio(V, F):
        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]
        a = np.linalg.norm(v1 - v0, axis=1)
        b = np.linalg.norm(v2 - v1, axis=1)
        c = np.linalg.norm(v0 - v2, axis=1)
        sides = np.stack([a, b, c], axis=1)
        longest = np.max(sides, axis=1)
        shortest = np.min(sides, axis=1)
        ar = np.where(shortest > 1e-12, longest / shortest, np.nan)
        return float(np.nanmean(ar))

    info = {
        "genus2_5k": (V5.shape[0], count_edges(F5), F5.shape[0], chi5, mean_aspect_ratio(V5, F5)),
        "genus2_10k": (V10.shape[0], count_edges(F10), F10.shape[0], chi10, mean_aspect_ratio(V10, F10)),
    }
    return True, info


if __name__ == "__main__":
    main()
    # If genus-sequence mesh was invalid, create and save valid genus-2
    V2, F2 = load_experiments_genus2()
    need_fix = V2 is None or euler_char(V2, F2) != -2
    if need_fix:
        ok, info = create_and_save_genus2()
        if ok:
            print("\nValid genus-2 meshes created. Re-run this script to verify.")
        else:
            print("\nCreation failed. Install: pip install gpytoolbox trimesh manifold3d")
            sys.exit(1)
