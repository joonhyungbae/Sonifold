"""
Run genus-sequence pipeline on remeshed genus 3–6 meshes.

1. Compute cotangent-weight LB eigenbasis (K=200) for each remeshed mesh.
2. Run β₀ over STFT frames for each mesh × K ∈ {50, 100, 200} × stimulus ∈ {A1, A2, A3, A5}.
3. Write results to data/results/remeshed_genus/ (CSVs and eigen npz).

Uses same pipeline as run_genus_K100_K200.py (precompute.eigensolver, mapping, analysis).
Mesh source: data/results/remeshed_genus/{mesh_id}_remeshed.obj.
Eigen output: data/results/remeshed_genus/eigen/{mesh_id}_remeshed.npz.

Usage (from project root):
  python scripts/analysis/run_remeshed_genus_pipeline.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

REMESHED_DIR = root / "data" / "results" / "remeshed_genus"
EIGEN_DIR = REMESHED_DIR / "eigen"
RESULTS_DIR_REMESHED = REMESHED_DIR
N_EIGEN_TARGET = 200
K_VALUES = [50, 100, 200]
STIMULI = ["A1", "A2", "A3", "A5"]
STRATEGY = "direct"
HOP_LENGTH = 512
TARGET_DURATION_SEC = 10.0
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "1"))
_CPU_COUNT = getattr(os, "cpu_count", lambda: None)() or 4
EIGEN_JOBS = int(os.environ.get("EIGEN_JOBS", str(min(4, _CPU_COUNT))))
BETA0_JOBS = int(os.environ.get("BETA0_JOBS", str(min(4, _CPU_COUNT))))
FRAME_JOBS = int(os.environ.get("FRAME_JOBS", "8"))

# Remeshed mesh IDs (genus 3–6 only)
REMESHED_MESH_IDS = [
    ("triple_torus_genus3_remeshed", 3),
    ("quad_torus_genus4_remeshed", 4),
    ("penta_torus_genus5_remeshed", 5),
    ("hex_torus_genus6_remeshed", 6),
]

OUT_K50 = RESULTS_DIR_REMESHED / "results_genus_K50.csv"
OUT_K100 = RESULTS_DIR_REMESHED / "results_genus_K100.csv"
OUT_K200 = RESULTS_DIR_REMESHED / "results_genus_K200.csv"
OUT_COMPARISON = RESULTS_DIR_REMESHED / "K_sensitivity_genus_comparison.csv"


def _eigen_worker(item):
    mesh_id, _ = item
    compute_and_save_eigen(mesh_id, N_EIGEN_TARGET)
    return mesh_id


def load_obj(mesh_id: str):
    """Load remeshed mesh: REMESHED_DIR / {mesh_id}.obj."""
    import trimesh
    path = REMESHED_DIR / (mesh_id + ".obj")
    if not path.exists():
        return None
    m = trimesh.load(str(path), force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)


def load_eigen(mesh_id: str, n_eigen: int):
    """Load V, F, evals, evecs from EIGEN_DIR; truncate to n_eigen."""
    npz_path = EIGEN_DIR / (mesh_id + ".npz")
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    V = np.asarray(data["vertices"], dtype=np.float64)
    F = np.asarray(data["faces"], dtype=np.int32)
    evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
    evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
    n_use = min(evecs.shape[0], len(evals), n_eigen)
    if n_use < n_eigen:
        return None
    evecs = evecs[:n_use]
    evals = evals[:n_use]
    return V, F, evals, evecs


def compute_and_save_eigen(mesh_id: str, n_eigen: int):
    """Compute n_eigen eigenpairs for remeshed mesh and save to EIGEN_DIR."""
    existing = load_eigen(mesh_id, n_eigen)
    if existing is not None:
        print("[eigen] {} already has >= {} modes".format(mesh_id, n_eigen), file=sys.stderr)
        return
    obj_path = REMESHED_DIR / (mesh_id + ".obj")
    if not obj_path.exists():
        print("[eigen] Skip {}: mesh not found".format(mesh_id), file=sys.stderr)
        return
    from precompute.eigensolver import compute_eigen

    V, F = load_obj(mesh_id)
    if V is None:
        return
    evals, evecs = compute_eigen(V, F, N=n_eigen)
    EIGEN_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = EIGEN_DIR / (mesh_id + ".npz")
    np.savez_compressed(npz_path, vertices=V, faces=F, eigenvalues=evals, eigenvectors=evecs)
    print("[eigen] {} -> {} (K={})".format(mesh_id, npz_path, n_eigen), file=sys.stderr)


def build_audio_padded(audio_id: str):
    from audio.audio_library import get_audio
    sig, sr = get_audio(audio_id)
    n_target = int(TARGET_DURATION_SEC * sr)
    if len(sig) < n_target:
        repeats = (n_target + len(sig) - 1) // len(sig)
        sig = np.tile(sig, repeats)[:n_target]
    else:
        sig = sig[:n_target]
    return sig, sr


def _single_frame_beta0(V, F, evals, evecs, mag):
    from mapping.spectral_mapping import map_fft_to_coefficients
    from analysis.scalar_field import compute_scalar_field
    from analysis.nodal_surface import compute_topology_metrics
    K = evecs.shape[0]
    coef = map_fft_to_coefficients(mag, K, strategy=STRATEGY, eigenvalues=evals)
    f = compute_scalar_field(evecs, coef)
    m = compute_topology_metrics(V, F, f)
    return m.beta0


def run_beta0_frames_core(V, F, evals, evecs, stimulus: str, frame_stride: int = 1, frame_jobs: int = 1):
    from audio.fft_analysis import compute_fft_frames
    from concurrent.futures import ThreadPoolExecutor

    sig, sr = build_audio_padded(stimulus)
    frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
    if not frames:
        return None
    subset = frames[::frame_stride]
    if not subset:
        return None
    mags = [mag for mag, _ in subset]
    if frame_jobs > 1:
        with ThreadPoolExecutor(max_workers=frame_jobs) as ex:
            beta0_list = list(ex.map(lambda mag: _single_frame_beta0(V, F, evals, evecs, mag), mags))
    else:
        beta0_list = [_single_frame_beta0(V, F, evals, evecs, mag) for mag in mags]
    arr = np.array(beta0_list, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0


def _beta0_worker(item):
    mesh_id, genus = item
    out = load_eigen(mesh_id, N_EIGEN_TARGET)
    if out is None:
        return (mesh_id, [])
    V, F, evals_full, evecs_full = out
    rows = []
    for K in K_VALUES:
        evals = evals_full[:K]
        evecs = evecs_full[:K]
        for stimulus in STIMULI:
            res = run_beta0_frames_core(V, F, evals, evecs, stimulus, frame_stride=FRAME_STRIDE, frame_jobs=FRAME_JOBS)
            if res is None:
                continue
            mean_b, std_b = res
            rows.append({
                "mesh": mesh_id,
                "genus": genus,
                "K": K,
                "stimulus": stimulus,
                "beta0_mean": round(mean_b, 4),
                "beta0_std": round(std_b, 4),
            })
    return (mesh_id, rows)


def _progress_iter(iterable, total, desc, unit="meshes"):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, unit=unit, file=sys.stderr, dynamic_ncols=True)
    except ImportError:
        return iterable


def main():
    import multiprocessing

    REMESHED_DIR.mkdir(parents=True, exist_ok=True)
    n_meshes = len(REMESHED_MESH_IDS)
    print("Remeshed genus pipeline: {} meshes, eigen K=200, beta0 K=50,100,200 × A1,A2,A3,A5".format(n_meshes), file=sys.stderr)

    # 1. Eigenbasis K=200 for all remeshed meshes
    print("=== 1. Eigenbasis K=200 for remeshed meshes (EIGEN_JOBS={}) ===".format(EIGEN_JOBS), file=sys.stderr)
    if EIGEN_JOBS > 1:
        with multiprocessing.Pool(processes=EIGEN_JOBS) as pool:
            for done, mesh_id in enumerate(_progress_iter(pool.imap_unordered(_eigen_worker, REMESHED_MESH_IDS), n_meshes, "Eigen"), 1):
                pass
    else:
        for mesh_id, _ in REMESHED_MESH_IDS:
            compute_and_save_eigen(mesh_id, N_EIGEN_TARGET)

    # 2. β₀ for each mesh
    print("=== 2. β₀ pipeline ===", file=sys.stderr)
    all_rows = []
    if BETA0_JOBS > 1:
        with multiprocessing.Pool(processes=BETA0_JOBS) as pool:
            beta0_iter = pool.imap_unordered(_beta0_worker, REMESHED_MESH_IDS)
            for done, (mesh_id, rows) in enumerate(_progress_iter(beta0_iter, n_meshes, "Beta0"), 1):
                all_rows.extend(rows)
    else:
        for item in REMESHED_MESH_IDS:
            mesh_id, rows = _beta0_worker(item)
            all_rows.extend(rows)
    all_rows.sort(key=lambda x: (x["genus"], x["K"], x["stimulus"]))

    # 3. Write CSVs
    fieldnames = ["mesh", "genus", "K", "stimulus", "beta0_mean", "beta0_std"]
    for K, out_path in [(50, OUT_K50), (100, OUT_K100), (200, OUT_K200)]:
        rows_k = [r for r in all_rows if r["K"] == K]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_k)
        print("Saved {} -> {} rows".format(out_path, len(rows_k)), file=sys.stderr)

    by_mesh = {}
    for r in all_rows:
        if r["stimulus"] != "A5":
            continue
        key = (r["mesh"], r["genus"])
        if key not in by_mesh:
            by_mesh[key] = {}
        by_mesh[key][r["K"]] = r["beta0_mean"]
    comp_rows = []
    for (mesh, genus), vals in sorted(by_mesh.items(), key=lambda x: x[0][1]):
        comp_rows.append({
            "mesh": mesh,
            "genus": genus,
            "beta0_K50": vals.get(50),
            "beta0_K100": vals.get(100),
            "beta0_K200": vals.get(200),
        })
    with open(OUT_COMPARISON, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mesh", "genus", "beta0_K50", "beta0_K100", "beta0_K200"])
        w.writeheader()
        w.writerows(comp_rows)
    print("Saved {}".format(OUT_COMPARISON), file=sys.stderr)

    # Non-monotonicity check (A5)
    print("\n--- Non-monotonic genus–β₀ (A5, remeshed) ---", file=sys.stderr)
    for K in K_VALUES:
        sub = [r for r in all_rows if r["K"] == K and r["stimulus"] == "A5"]
        sub = sorted(sub, key=lambda r: r["genus"])
        beta0s = [r["beta0_mean"] for r in sub]
        monotonic = all(beta0s[i] <= beta0s[i + 1] for i in range(len(beta0s) - 1))
        print("  K={}: β₀ = {} -> monotonic? {}".format(K, beta0s, monotonic), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
