"""
Genus sequence β₀ at K=100 and K=200.

1. Recompute eigenbases for all 7 genus meshes at K=200 (one file per mesh; use first K for 50/100/200).
2. Rerun β₀ over STFT frames for each mesh × K ∈ {50, 100, 200} × stimulus ∈ {A1, A2, A3, A5}.
3. Save results_genus_K100.csv, results_genus_K200.csv (columns: mesh, genus, K, stimulus, beta0_mean, beta0_std).
4. Save K_sensitivity_genus_comparison.csv (mesh, genus, beta0_K50, beta0_K100, beta0_K200) for A5.
5. Print rank-order comparison (A1/A2 torus anomaly, non-monotonicity across K).

Run from project root: python scripts/analysis/run_genus_K100_K200.py

Mathematical verification rigor (preserved):
  - EIGEN_TOL=1e-4 (solver default), FRAME_STRIDE=1 (all frames). Relaxing is not for verification.
  - Speed only via parallelization/GPU: results identical. EIGEN_JOBS/BETA0_JOBS (mesh parallel), FRAME_JOBS (frame threads). GPU used by default if CuPy (USE_GPU=0 to force CPU).

Quick verification: EIGEN_JOBS=7 BETA0_JOBS=7 FRAME_JOBS=8  (GPU default, rigor preserved)

STFT: Hann 2048, hop 512, FFT 2048, 44.1 kHz (same as existing pipeline).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

EXPERIMENTS_DIR = root / "data" / "experiments"
EIGEN_DIR = root / "data" / "experiments" / "eigen"
RESULTS_DIR = root / "data" / "results"
N_EIGEN_TARGET = 200
K_VALUES = [50, 100, 200]
STIMULI = ["A1", "A2", "A3", "A5"]
STRATEGY = "direct"
HOP_LENGTH = 512
TARGET_DURATION_SEC = 10.0
# Verification rigor: use all frames (1). Subsampling (>1) is for exploration only.
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "1"))
_CPU_COUNT = getattr(os, "cpu_count", lambda: None)() or 4
EIGEN_JOBS = int(os.environ.get("EIGEN_JOBS", str(min(7, _CPU_COUNT))))  # mesh parallel, same result
BETA0_JOBS = int(os.environ.get("BETA0_JOBS", str(min(7, _CPU_COUNT))))
FRAME_JOBS = int(os.environ.get("FRAME_JOBS", "8"))  # frame threads within (mesh,K,stimulus), same result

GENUS_MESH_IDS = [
    ("sphere_genus0", 0),
    ("torus_genus1", 1),
    ("double_torus_genus2", 2),
    ("triple_torus_genus3", 3),
    ("quad_torus_genus4", 4),
    ("penta_torus_genus5", 5),
    ("hex_torus_genus6", 6),
]

OUT_K100 = RESULTS_DIR / "results_genus_K100.csv"
OUT_K200 = RESULTS_DIR / "results_genus_K200.csv"
OUT_COMPARISON = RESULTS_DIR / "K_sensitivity_genus_comparison.csv"


def load_obj(path: Path):
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)


def load_eigen(mesh_id: str, n_eigen: int):
    """Load V, F, evals, evecs from EIGEN_DIR; evecs/evals truncated to n_eigen. Returns (V, F, evals, evecs)."""
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
    """Compute n_eigen eigenpairs and save to EIGEN_DIR/(mesh_id).npz. Skip if already >= n_eigen."""
    existing = load_eigen(mesh_id, n_eigen)
    if existing is not None:
        print("[eigen] {} already has >= {} modes".format(mesh_id, n_eigen), file=sys.stderr)
        return
    obj_path = EXPERIMENTS_DIR / (mesh_id + ".obj")
    if not obj_path.exists():
        raise FileNotFoundError("Mesh not found: {}".format(obj_path))
    from precompute.eigensolver import compute_eigen
    from precompute.export_json import export_json

    V, F = load_obj(obj_path)
    evals, evecs = compute_eigen(V, F, N=n_eigen)
    EIGEN_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = EIGEN_DIR / (mesh_id + ".npz")
    np.savez_compressed(npz_path, vertices=V, faces=F, eigenvalues=evals, eigenvectors=evecs)
    export_json(V, F, evals, evecs, EIGEN_DIR / (mesh_id + ".json"))
    print("[eigen] {} -> {} (K={})".format(mesh_id, npz_path, n_eigen), file=sys.stderr)


def build_audio_padded(audio_id: str):
    """Load audio and pad/tile to TARGET_DURATION_SEC for frame extraction."""
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
    """One STFT frame → β₀. Same inputs → same output (thread-safe, deterministic)."""
    from mapping.spectral_mapping import map_fft_to_coefficients
    from analysis.scalar_field import compute_scalar_field
    from analysis.nodal_surface import compute_topology_metrics
    K = evecs.shape[0]
    coef = map_fft_to_coefficients(mag, K, strategy=STRATEGY, eigenvalues=evals)
    f = compute_scalar_field(evecs, coef)
    m = compute_topology_metrics(V, F, f)
    return m.beta0


def run_beta0_frames_core(V, F, evals, evecs, stimulus: str, frame_stride: int = 1, frame_jobs: int = 1):
    """β₀ over STFT frames. frame_jobs>1: parallelize frames with threads (same result)."""
    from audio.fft_analysis import compute_fft_frames
    from concurrent.futures import ThreadPoolExecutor

    K = evecs.shape[0]
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


def run_beta0_frames(mesh_id: str, genus: int, K: int, stimulus: str):
    """Run β₀ over STFT frames for (mesh_id, K, stimulus). Return (beta0_mean, beta0_std)."""
    out = load_eigen(mesh_id, K)
    if out is None:
        return None
    V, F, evals, evecs = out
    return run_beta0_frames_core(V, F, evals, evecs, stimulus, frame_stride=FRAME_STRIDE, frame_jobs=FRAME_JOBS)


def _eigen_worker(item):
    """One mesh: compute and save 200 modes. For multiprocessing. Returns mesh_id for progress."""
    mesh_id, _ = item
    compute_and_save_eigen(mesh_id, N_EIGEN_TARGET)
    return mesh_id


def _beta0_worker(item):
    """One mesh: load eigen once, run all K × stimulus. Returns (mesh_id, rows) for progress."""
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
    """Wrap iterable with tqdm if available, else simple count print."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, unit=unit, file=sys.stderr, dynamic_ncols=True)
    except ImportError:
        return iterable


def main():
    import multiprocessing

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_meshes = len(GENUS_MESH_IDS)
    print("Total: 2 steps — (1) Eigen K=200 for {} meshes, (2) Beta0 for {} meshes × 12 combos each".format(
        n_meshes, n_meshes), file=sys.stderr)

    # 1. Ensure eigenbasis with at least 200 modes for all 7 meshes (parallel)
    print("=== 1. Eigenbasis at K=200 for {} meshes (EIGEN_JOBS={}) ===".format(n_meshes, EIGEN_JOBS), file=sys.stderr)
    if EIGEN_JOBS > 1:
        with multiprocessing.Pool(processes=EIGEN_JOBS) as pool:
            eigen_iter = pool.imap_unordered(_eigen_worker, GENUS_MESH_IDS)
            for done, mesh_id in enumerate(_progress_iter(eigen_iter, n_meshes, "Eigen"), 1):
                print("[progress] Eigen {}/{}: {} done".format(done, n_meshes, mesh_id), flush=True, file=sys.stderr)
    else:
        for done, item in enumerate(GENUS_MESH_IDS, 1):
            _eigen_worker(item)
            print("[progress] Eigen {}/{}: {} done".format(done, n_meshes, item[0]), flush=True, file=sys.stderr)

    # 2. Run β₀ for each mesh (parallel over meshes; each mesh does all K × stimulus)
    print("=== 2. β₀ pipeline (K=50,100,200 × A1,A2,A3,A5), FRAME_STRIDE={}, FRAME_JOBS={}, BETA0_JOBS={} ===".format(
        FRAME_STRIDE, FRAME_JOBS, BETA0_JOBS), file=sys.stderr)
    all_rows = []
    if BETA0_JOBS > 1:
        with multiprocessing.Pool(processes=BETA0_JOBS) as pool:
            beta0_iter = pool.imap_unordered(_beta0_worker, GENUS_MESH_IDS)
            for done, (mesh_id, rows) in enumerate(_progress_iter(beta0_iter, n_meshes, "Beta0"), 1):
                all_rows.extend(rows)
                print("[progress] Beta0 {}/{}: {} done ({} combos)".format(
                    done, n_meshes, mesh_id, len(rows)), flush=True, file=sys.stderr)
        for r in sorted(all_rows, key=lambda x: (x["genus"], x["K"], x["stimulus"])):
            print("[beta0] {} genus={} K={} {} -> mean={:.2f} std={:.2f}".format(
                r["mesh"], r["genus"], r["K"], r["stimulus"], r["beta0_mean"], r["beta0_std"]), file=sys.stderr)
    else:
        for done, item in enumerate(GENUS_MESH_IDS, 1):
            mesh_id, rows = _beta0_worker(item)
            all_rows.extend(rows)
            print("[progress] Beta0 {}/{}: {} done ({} combos)".format(
                done, n_meshes, mesh_id, len(rows)), flush=True, file=sys.stderr)
            for r in rows:
                print("[beta0] {} genus={} K={} {} -> mean={:.2f} std={:.2f}".format(
                    r["mesh"], r["genus"], r["K"], r["stimulus"], r["beta0_mean"], r["beta0_std"]), file=sys.stderr)
    all_rows.sort(key=lambda x: (x["genus"], x["K"], x["stimulus"]))

    print("=== 3. Writing CSVs ===", file=sys.stderr)
    # Write results_genus_K100.csv, results_genus_K200.csv
    fieldnames = ["mesh", "genus", "K", "stimulus", "beta0_mean", "beta0_std"]
    rows_100 = [r for r in all_rows if r["K"] == 100]
    rows_200 = [r for r in all_rows if r["K"] == 200]
    with open(OUT_K100, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_100)
    with open(OUT_K200, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_200)
    print("Saved {} -> {} rows".format(OUT_K100, len(rows_100)), file=sys.stderr)
    print("Saved {} -> {} rows".format(OUT_K200, len(rows_200)), file=sys.stderr)

    # 4. K_sensitivity_genus_comparison.csv (A5 only)
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
    print("Saved {} -> {} rows".format(OUT_COMPARISON, len(comp_rows)), file=sys.stderr)

    # 5. Rank-order summary to stdout (for Conjecture 4.1 defense)
    print("\n" + "=" * 60)
    print("Rank-order comparison (by beta0_mean, genus sequence)")
    print("=" * 60)

    mesh_order = [m for m, _ in GENUS_MESH_IDS]
    for stimulus in ["A1", "A2", "A3", "A5"]:
        print("\n--- Stimulus {} ---".format(stimulus))
        for K in K_VALUES:
            sub = [r for r in all_rows if r["K"] == K and r["stimulus"] == stimulus]
            by_mesh_k = {r["mesh"]: r["beta0_mean"] for r in sub}
            order = sorted(mesh_order, key=lambda m: (by_mesh_k.get(m, -1), m), reverse=True)
            rank_str = "  K={}: ".format(K) + " > ".join(
                "{} (β₀={:.1f})".format(m, by_mesh_k.get(m, float("nan"))) for m in order
            )
            print(rank_str)
        # Torus anomaly for A1/A2
        if stimulus in ("A1", "A2"):
            sub50 = [r for r in all_rows if r["K"] == 50 and r["stimulus"] == stimulus]
            sub100 = [r for r in all_rows if r["K"] == 100 and r["stimulus"] == stimulus]
            sub200 = [r for r in all_rows if r["K"] == 200 and r["stimulus"] == stimulus]
            t50 = next((r["beta0_mean"] for r in sub50 if r["mesh"] == "torus_genus1"), None)
            t100 = next((r["beta0_mean"] for r in sub100 if r["mesh"] == "torus_genus1"), None)
            t200 = next((r["beta0_mean"] for r in sub200 if r["mesh"] == "torus_genus1"), None)
            print("  Torus (genus-1) β₀: K50={}, K100={}, K200={} (anomalously low? {})".format(
                t50, t100, t200, "yes" if (t50 is not None and t50 < 2 and t100 is not None and t100 < 2 and t200 is not None and t200 < 2) else "check"
            ))

    # Non-monotonicity: genus vs beta0 should not be strictly increasing
    print("\n--- Non-monotonic genus–β₀ (A5) ---")
    for K in K_VALUES:
        sub = [r for r in all_rows if r["K"] == K and r["stimulus"] == "A5"]
        sub = sorted(sub, key=lambda r: r["genus"])
        beta0s = [r["beta0_mean"] for r in sub]
        monotonic = all(beta0s[i] <= beta0s[i + 1] for i in range(len(beta0s) - 1))
        print("  K={}: genus order β₀ = {} -> monotonic? {}".format(K, beta0s, monotonic))

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
