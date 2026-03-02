"""
Genus 5, 6 mesh generation → eigenbasis computation → genus 0–6 β₀ analysis → Random coefficient test → 7-point visualization.

1. Genus 5, 6 mesh: same as experiment.generate_experiment_meshes (overlapping tori boolean union), ~5k vertices.
2. Cotangent-weight LB → K=50 eigenvalues/eigenvectors, save npz/json to data/experiments/eigen.
3. Compute β₀ for genus 0–6 → save to data/results/results_genus_extended_7point.csv.
4. Run random coefficient test for genus 5, 6 and append to data/results/random_coefficient_beta0.csv.
5. Generate figures/fig_genus_beta0_7point.pdf (same style as fig_genus_beta0.pdf, 7 points).

Run from project root: python scripts/generate_genus_extended.py

Speed up:
  - GPU: USE_GPU=1 python scripts/generate_genus_extended.py (CuPy + CUDA required)
  - Relaxed tolerance (faster convergence): EIGEN_TOL=1e-3 python scripts/generate_genus_extended.py
  - If genus 3,4 eigen missing, run first: python -m experiment.precompute_experiment_eigen --genus-only

Required: gpytoolbox, trimesh. For boolean union: manifold3d or Blender.
  pip install gpytoolbox trimesh
  pip install manifold3d   # or install Blender
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

EXPERIMENTS_DIR = root / "data" / "experiments"
EIGEN_DIR = root / "data" / "experiments" / "eigen"
RESULTS_DIR = root / "data" / "results"
RESULTS_7POINT = RESULTS_DIR / "results_genus_extended_7point.csv"
RESULTS_EXTENDED = RESULTS_DIR / "results_genus_extended.csv"
RANDOM_BETA0_CSV = RESULTS_DIR / "random_coefficient_beta0.csv"
N_EIGEN = 50
AUDIO_ID = "A3"
STRATEGY = "direct"

# Genus 0–6 mesh_id order (existing 0–4 + new 5, 6)
GENUS_MESH_IDS = [
    ("sphere_genus0", 0),
    ("torus_genus1", 1),
    ("double_torus_genus2", 2),
    ("triple_torus_genus3", 3),
    ("quad_torus_genus4", 4),
    ("penta_torus_genus5", 5),
    ("hex_torus_genus6", 6),
]
GENUS56 = [("penta_torus_genus5", 5), ("hex_torus_genus6", 6)]


def ensure_genus56_meshes():
    """Create genus 5, 6 .obj if missing, using experiment.generate_experiment_meshes approach."""
    from experiment.generate_experiment_meshes import _genus5_5k, _genus6_5k, save_obj

    for name, genus in GENUS56:
        path = EXPERIMENTS_DIR / (name + ".obj")
        if path.exists():
            print("[genus] {} already exists: {}".format(name, path), file=sys.stderr)
            continue
        try:
            V, F = (_genus5_5k() if genus == 5 else _genus6_5k())
            save_obj(V, F, path)
            print("[genus] {} genus={} n_verts={} -> {}".format(name, genus, V.shape[0], path), file=sys.stderr)
        except Exception as e:
            print(
                "[genus] {} failed: {}. Install: pip install gpytoolbox trimesh manifold3d".format(name, e),
                file=sys.stderr,
            )
            raise


def load_obj(path: Path):
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)


def compute_and_save_eigen(mesh_id: str):
    """Load mesh → compute cotangent LB K=50 eigenpairs → save npz, json."""
    from precompute.eigensolver import compute_eigen
    from precompute.export_json import export_json

    obj_path = EXPERIMENTS_DIR / (mesh_id + ".obj")
    npz_path = EIGEN_DIR / (mesh_id + ".npz")
    if not obj_path.exists():
        raise FileNotFoundError("Mesh not found: {}".format(obj_path))
    if npz_path.exists():
        print("[eigen] {} already computed: {}".format(mesh_id, npz_path), file=sys.stderr)
        return
    V, F = load_obj(obj_path)
    evals, evecs = compute_eigen(V, F, N=N_EIGEN)
    EIGEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, vertices=V, faces=F, eigenvalues=evals, eigenvectors=evecs)
    export_json(V, F, evals, evecs, EIGEN_DIR / (mesh_id + ".json"))
    print("[eigen] {} -> {}".format(mesh_id, npz_path), file=sys.stderr)


def load_eigen(mesh_id: str):
    npz_path = EIGEN_DIR / (mesh_id + ".npz")
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    V = np.asarray(data["vertices"], dtype=np.float64)
    F = np.asarray(data["faces"], dtype=np.int32)
    evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
    evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
    return V, F, evecs, evals


def run_beta0_pipeline(mesh_id: str):
    """A3 + direct mapping → returns β₀, A_ratio, S."""
    from audio.audio_library import get_audio
    from audio.fft_analysis import compute_fft
    from mapping.spectral_mapping import map_fft_to_coefficients
    from analysis.scalar_field import compute_scalar_field
    from analysis.nodal_surface import compute_topology_metrics
    from analysis.symmetry import compute_symmetry

    out = load_eigen(mesh_id)
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


def build_results_7point():
    """Collect genus 0–6 β₀ and save to results_genus_extended_7point.csv."""
    rows_by_genus = {}

    # Load 0–4 from existing results_genus_extended.csv
    if RESULTS_EXTENDED.exists():
        with open(RESULTS_EXTENDED, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    g = int(float(r["genus"]))
                    if 0 <= g <= 6:
                        rows_by_genus[g] = {
                            "mesh_id": r["mesh_id"],
                            "genus": g,
                            "beta0": int(round(float(r["beta0"]))),
                            "A_ratio": float(r["A_ratio"]),
                            "S": float(r["S"]),
                        }
                except (ValueError, KeyError):
                    pass

    # Compute missing genus from eigen
    for mesh_id, genus in GENUS_MESH_IDS:
        if genus in rows_by_genus:
            continue
        res = run_beta0_pipeline(mesh_id)
        if res is None:
            print("[7point] Skip {} (no eigen)".format(mesh_id), file=sys.stderr)
            continue
        rows_by_genus[genus] = {
            "mesh_id": mesh_id,
            "genus": genus,
            "beta0": res["beta0"],
            "A_ratio": res["A_ratio"],
            "S": res["S"],
        }
        print("[7point] {} beta0={}".format(mesh_id, res["beta0"]), file=sys.stderr)

    rows = [rows_by_genus[g] for g in range(7) if g in rows_by_genus]
    rows.sort(key=lambda x: x["genus"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mesh_id", "genus", "beta0", "A_ratio", "S"]
    with open(RESULTS_7POINT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("[7point] Saved {} rows -> {}".format(len(rows), RESULTS_7POINT), file=sys.stderr)
    return rows


def run_random_coefficient_test():
    """Run random coefficient → β₀ for genus 5, 6 and append to random_coefficient_beta0.csv."""
    from analysis.scalar_field import compute_scalar_field
    from analysis.nodal_surface import compute_topology_metrics

    rng = np.random.default_rng(42)
    new_rows = []
    for mesh_id, genus in GENUS56:
        out = load_eigen(mesh_id)
        if out is None:
            print("[random] Skip {} (no eigen)".format(mesh_id), file=sys.stderr)
            continue
        V, F, evecs, evals = out
        coef = rng.standard_normal(N_EIGEN).astype(np.float64)
        coef = coef / np.linalg.norm(coef)
        f = compute_scalar_field(evecs, coef)
        m = compute_topology_metrics(V, F, f)
        beta0 = int(round(m.beta0))
        new_rows.append({"mesh_id": mesh_id, "genus": genus, "beta0": beta0})
        print("[random] {} genus={} beta0={}".format(mesh_id, genus, beta0), file=sys.stderr)

    fieldnames = ["mesh_id", "genus", "beta0"]
    existing = []
    if RANDOM_BETA0_CSV.exists():
        with open(RANDOM_BETA0_CSV, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            fn = list(r.fieldnames) if r.fieldnames else fieldnames
            existing = list(r)
    else:
        fn = fieldnames

    genus_seen = {int(float(r["genus"])) for r in existing}
    for row in new_rows:
        if row["genus"] not in genus_seen:
            existing.append(row)
            genus_seen.add(row["genus"])
    existing.sort(key=lambda r: int(float(r["genus"])))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RANDOM_BETA0_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        w.writerows(existing)
    print("[random] Saved {} rows -> {}".format(len(existing), RANDOM_BETA0_CSV), file=sys.stderr)


def run_figure_7point():
    """Run scripts/figures/fig_genus_beta0_7point.py → generate fig_genus_beta0_7point.pdf."""
    fig_script = Path(__file__).resolve().parent / "fig_genus_beta0_7point.py"
    if not fig_script.exists():
        print("[fig] Script not found: {}".format(fig_script), file=sys.stderr)
        return
    import subprocess
    out = subprocess.run([sys.executable, str(fig_script)], cwd=str(root), capture_output=True, text=True)
    if out.returncode != 0:
        print("[fig] stderr: {}".format(out.stderr), file=sys.stderr)
    else:
        print("[fig] Saved figures/fig_genus_beta0_7point.pdf", file=sys.stderr)


def main():
    print("=== 1. Genus 5, 6 mesh generation ===", file=sys.stderr)
    ensure_genus56_meshes()

    print("=== 2. Eigenbasis (K=50) for genus 5, 6 ===", file=sys.stderr)
    for mesh_id, _ in GENUS56:
        compute_and_save_eigen(mesh_id)

    print("=== 3. Genus 0–6 β₀ → results_genus_extended_7point.csv ===", file=sys.stderr)
    build_results_7point()

    print("=== 4. Random coefficient test (genus 5, 6) → random_coefficient_beta0.csv ===", file=sys.stderr)
    run_random_coefficient_test()

    print("=== 5. Figure: fig_genus_beta0_7point.pdf ===", file=sys.stderr)
    run_figure_7point()

    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
