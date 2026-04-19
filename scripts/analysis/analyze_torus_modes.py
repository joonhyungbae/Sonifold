"""
Analyze torus Laplace–Beltrami modes for A3 (piano) with direct mapping.
Outputs dominant modes, eigenvalues, (m,n) quantum numbers, multiplicity,
and β₀ for top-5 vs full K modes. For paper: "dominant modes (m,n)=(1,0) and (0,1),
superposition on flat torus produces a single connected nodal curve."

Usage:
  python scripts/analysis/analyze_torus_modes.py                  # data/eigen/torus (~10k verts)
  python scripts/analysis/analyze_torus_modes.py --mesh torus_genus1   # genus sequence (~5k)
  python scripts/analysis/analyze_torus_modes.py --mesh torus_genus1 --audio A1   # spectrally simple → β₀=0, dominant (m,n)=(1,0)
  python scripts/analysis/analyze_torus_modes.py --mesh torus_genus1 --single-frame --scan-frames 100   # beta0 per frame

Prerequisites:
  - data/eigen/torus.npz or data/experiments/eigen/torus_genus1.json (per --mesh)
  - Audio (A3: data/audio/piano_cmajor.wav or placeholder; A1/A2 synthetic)

Output:
  - Printed report to stdout
  - data/results/<mesh_id>_<audio>_mode_analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft, compute_fft_frames
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

N_EIGEN = 50
HOP_LENGTH = 512
THRESHOLD_RATIO = 0.01
TOP_N = 5
EIGEN_DIR = root / "data" / "eigen"
EIGENBASES_DIR = root / "data" / "eigenbases"
EIGEN_EXPERIMENTS_DIR = root / "data" / "experiments" / "eigen"
RESULTS_DIR = root / "data" / "results"

# Torus geometry from precompute/mesh_library: R=3, r=1 (major/minor radius).
# Flat torus λ = (2πm/L₁)² + (2πn/L₂)² with L₁=2πR, L₂=2πr → λ = (m/R)² + (n/r)².
TORUS_R = 3.0
TORUS_r = 1.0
# (m,n) search range for matching λ_k
MN_MAX = 12
REL_TOL_EVAL = 1e-5  # relative tolerance for eigenvalue equality (multiplicity)
REL_TOL_MN = 0.15    # relative tolerance for (m,n) match


def load_torus_eigen(mesh_id: str):
    """
    Load vertices, faces, eigenvectors (N_EIGEN, V), eigenvalues.
    mesh_id: "torus" -> data/eigen or data/eigenbases; "torus_genus1" -> data/experiments/eigen.
    """
    if mesh_id == "torus_genus1" and EIGEN_EXPERIMENTS_DIR.exists():
        bases = [EIGEN_EXPERIMENTS_DIR]
    else:
        bases = [EIGEN_DIR, EIGENBASES_DIR]
    for base in bases:
        if not base.exists():
            continue
        for ext in [".npz", ".json"]:
            path = base / f"{mesh_id}{ext}"
            if not path.exists():
                continue
            try:
                if path.suffix == ".npz":
                    data = np.load(path, allow_pickle=True)
                    V = np.asarray(data["vertices"], dtype=np.float64)
                    F = np.asarray(data["faces"], dtype=np.int32)
                    evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
                    evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
                else:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    V = np.array(data["vertices"], dtype=np.float64)
                    F = np.array(data["faces"], dtype=np.int32)
                    evecs = np.array(data["eigenvectors"], dtype=np.float64)
                    evals = np.array(data.get("eigenvalues", []), dtype=np.float64)
                if evecs.shape[0] == V.shape[0]:
                    evecs = evecs[:, :N_EIGEN].T
                else:
                    evecs = evecs[:N_EIGEN, :]
                if evecs.shape[0] < N_EIGEN:
                    return None
                evals = evals[:N_EIGEN]
                return V, F, evecs, evals
            except Exception as e:
                print(f"Warning: failed to load {path}: {e}", file=sys.stderr)
                continue
    return None


def eigenvalue_multiplicity(evals, rel_tol=REL_TOL_EVAL):
    """Return list of (λ_value, multiplicity) for unique eigenvalues (sorted)."""
    ev = np.asarray(evals, dtype=np.float64).ravel()
    ev = np.sort(ev)
    mults = []
    i = 0
    while i < len(ev):
        lam = ev[i]
        j = i + 1
        while j < len(ev) and ev[j] <= lam * (1 + rel_tol) and ev[j] >= lam * (1 - rel_tol):
            j += 1
        mults.append((float(lam), j - i))
        i = j
    return mults


def multiplicity_of(evals, lam_k, rel_tol=REL_TOL_EVAL):
    """Number of eigenvalues equal to lam_k within relative tolerance."""
    ev = np.asarray(evals, dtype=np.float64).ravel()
    return int(np.sum(np.abs(ev - lam_k) <= lam_k * rel_tol + 1e-20))


def lambda_to_mn(lam, R=TORUS_R, r=TORUS_r):
    """
    For flat torus λ = (m/R)² + (n/r)², find best (m, n) in [0, MN_MAX].
    Returns (m, n) with smallest |λ - (m/R)² - (n/r)²|, or (None, None) if no match within REL_TOL_MN.
    """
    best_mn = (None, None)
    best_err = float("inf")
    for m in range(MN_MAX + 1):
        for n in range(MN_MAX + 1):
            if m == 0 and n == 0:
                continue
            theory = (m / R) ** 2 + (n / r) ** 2
            err = abs(lam - theory)
            if err < best_err:
                best_err = err
                best_mn = (m, n)
    if lam > 1e-20 and best_err / lam <= REL_TOL_MN:
        return best_mn
    if lam <= 1e-20 and best_err < 1e-10:
        return best_mn
    return best_mn  # always return best match for display


def main():
    ap = argparse.ArgumentParser(description="Torus mode analysis (A3, direct mapping).")
    ap.add_argument(
        "--mesh",
        default="torus",
        choices=["torus", "torus_genus1"],
        help="Mesh: torus (data/eigen, ~10k) or torus_genus1 (data/experiments/eigen, genus sequence)",
    )
    ap.add_argument(
        "--single-frame",
        action="store_true",
        help="Use single FFT frame, like genus experiment; default is time-averaged over all frames.",
    )
    ap.add_argument(
        "--frame-index",
        type=int,
        default=None,
        metavar="N",
        help="With --single-frame: use N-th frame from STFT (0-based). Default: use compute_fft (first window).",
    )
    ap.add_argument(
        "--threshold-ratio",
        type=float,
        default=THRESHOLD_RATIO,
        metavar="R",
        help="Nodal set epsilon = R * max|f| (default %.3f)." % THRESHOLD_RATIO,
    )
    ap.add_argument(
        "--scan-frames",
        type=int,
        default=0,
        metavar="M",
        help="If >0: run single-frame for frame indices 0..M-1 and print beta0 per frame; then exit (no JSON).",
    )
    ap.add_argument(
        "--audio",
        default="A3",
        choices=["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
        help="Audio stimulus (default A3). A1/A2 often give low beta0 on torus.",
    )
    args = ap.parse_args()
    mesh_id = args.mesh
    single_frame = args.single_frame
    frame_index = args.frame_index
    threshold_ratio = args.threshold_ratio
    scan_frames = args.scan_frames
    audio_id = args.audio

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"{mesh_id}_{audio_id}_mode_analysis.json"

    data = load_torus_eigen(mesh_id)
    if data is None:
        print(
            f"Error: eigenbasis for {mesh_id} not found (data/eigen, data/eigenbases, or data/experiments/eigen).",
            file=sys.stderr,
        )
        return 1
    V, F, evecs, evals = data
    K = evecs.shape[0]

    sig, sr = get_audio(audio_id)
    frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
    if not frames:
        print("Error: no FFT frames for {}.".format(audio_id), file=sys.stderr)
        return 1

    if scan_frames > 0:
        # Scan mode: report beta0 for frame 0..scan_frames-1
        print("Mesh: {} | audio: {} | threshold_ratio={} | scanning frames 0..{}".format(mesh_id, audio_id, threshold_ratio, scan_frames - 1))
        for fi in range(min(scan_frames, len(frames))):
            mag, _ = frames[fi]
            coef = map_fft_to_coefficients(mag, K, strategy="direct", eigenvalues=evals)
            f = compute_scalar_field(evecs, coef)
            m = compute_topology_metrics(V, F, f, threshold_ratio=threshold_ratio)
            print("frame_index={}  beta0={}".format(fi, m.beta0))
        return 0

    if single_frame:
        if frame_index is not None:
            fi = min(max(0, frame_index), len(frames) - 1)
            mag, _ = frames[fi]
        else:
            mag, _ = compute_fft(sig, sr)
        coef = map_fft_to_coefficients(mag, K, strategy="direct", eigenvalues=evals)
    else:
        mag_mean = np.mean(np.array([mag for mag, _ in frames], dtype=np.float64), axis=0)
        coef = map_fft_to_coefficients(mag_mean, K, strategy="direct", eigenvalues=evals)

    # Dominant modes: indices k (1-based in output) by |a_k| descending
    abs_coef = np.abs(coef)
    dominant_idx = np.argsort(-abs_coef)[:TOP_N]

    # β₀ for full K modes
    f_full = compute_scalar_field(evecs, coef)
    m_full = compute_topology_metrics(V, F, f_full, threshold_ratio=threshold_ratio)
    beta0_full = m_full.beta0

    # β₀ for top-5 modes only
    coef_top5 = np.zeros(K, dtype=np.float64)
    for idx in dominant_idx:
        coef_top5[idx] = coef[idx]
    nrm = np.linalg.norm(coef_top5)
    if nrm > 1e-20:
        coef_top5 = coef_top5 / nrm
    f_top5 = compute_scalar_field(evecs, coef_top5)
    m_top5 = compute_topology_metrics(V, F, f_top5, threshold_ratio=threshold_ratio)
    beta0_top5 = m_top5.beta0

    # Multiplicity pattern
    mult_list = eigenvalue_multiplicity(evals)

    # Build report
    report_lines = [
        "=== Torus Mode Analysis ({}, direct mapping) ===".format(audio_id),
        "Mesh: {} | audio: {} | {} | threshold_ratio={}".format(
            mesh_id,
            audio_id,
            "single FFT frame" + (" (frame {})".format(frame_index) if frame_index is not None else "") if single_frame else "time-averaged magnitude",
            threshold_ratio,
        ),
        "",
        "Top {} dominant modes:".format(TOP_N),
    ]
    dominant_records = []
    for rank, idx in enumerate(dominant_idx, start=1):
        k = int(idx) + 1
        ak = float(coef[idx])
        lam_k = float(evals[idx])
        m, n = lambda_to_mn(lam_k)
        mn_str = "({},{})".format(m, n) if m is not None else "(-,-)"
        mult = multiplicity_of(evals, lam_k)
        report_lines.append("k={}, |a_k|={:.3f}, λ_k={:.4f}, mult={}, (m,n)={}".format(k, abs(ak), lam_k, mult, mn_str))
        dominant_records.append({
            "k": k,
            "|a_k|": round(abs(ak), 4),
            "a_k": round(ak, 6),
            "λ_k": round(lam_k, 6),
            "(m,n)": [m, n] if m is not None else None,
            "multiplicity": int(mult),
        })

    report_lines.extend([
        "",
        "β₀ (top {} modes only): {}".format(TOP_N, beta0_top5),
        "β₀ (all K={} modes): {}".format(K, beta0_full),
        "",
        "Eigenvalue multiplicity pattern: " + ", ".join(
            "λ={:.4f} has mult={}".format(lam, cnt) for lam, cnt in mult_list[:15]
        ) + (" ..." if len(mult_list) > 15 else ""),
    ])

    print("\n".join(report_lines))

    out = {
        "mesh_id": mesh_id,
        "single_frame": single_frame,
        "frame_index": frame_index,
        "threshold_ratio": threshold_ratio,
        "audio": audio_id,
        "mapping": "direct",
        "K": K,
        "n_vertices": int(V.shape[0]),
        "top_n_dominant": dominant_records,
        "beta0_top5_modes": int(beta0_top5),
        "beta0_all_modes": int(beta0_full),
        "eigenvalue_multiplicity": [{"λ": round(lam, 6), "mult": cnt} for lam, cnt in mult_list],
        "torus_geometry": {"R": TORUS_R, "r": TORUS_r},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
