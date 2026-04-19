"""
Generate shape-sequencing figure: same audio (A3 piano) on three shapes (sphere, torus,
double torus) for three representative frames (sparse, climax, quiet). Section 6 visual.

Usage (from project root):
  python scripts/generate_shape_sequencing_figure.py

Outputs:
  - figures/fig_shape_sequencing.pdf, figures/fig_shape_sequencing.png
  - data/results/shape_sequencing_data.csv

Reuses: audio_library, fft_analysis, spectral_mapping, scalar_field, nodal_surface.
STFT: Hann 2048, hop 512, n_fft 2048, 44100 Hz. K=50 (skip eigenmode 0). ε = 0.05 * max|f|.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft_frames, N_FFT
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import extract_nodal_and_metrics

# Parameters (paper-aligned)
N_EIGEN = 50
HOP_LENGTH = 512
THRESHOLD_RATIO = 0.05  # ε = 0.05 * max|f(v)| (match web/paper)
CONTOUR_LEVELS = 16
CONTOUR_BAND = 0.006
FIG_DPI = 600  # high resolution to match web sharpness
TARGET_DURATION_SEC = 10.0  # tile A3 to get enough frames for climax/quiet
SPARSE_FRAME_INDEX = 50  # frame near start (sparse)

EIGEN_DIR = root / "data" / "eigen"
EIGENBASES_DIR = root / "data" / "eigenbases"
EXPERIMENTS_EIGEN = root / "data" / "experiments" / "eigen"
SHAPES = ["sphere", "torus", "double_torus"]
RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"
OUT_CSV = RESULTS_DIR / "shape_sequencing_data.csv"
OUT_PDF = FIGURES_DIR / "fig_shape_sequencing.pdf"
OUT_PNG = FIGURES_DIR / "fig_shape_sequencing.png"


def load_mesh_eigen(mesh_name: str):
    """
    Load vertices, faces, eigenvectors (modes 1..N_EIGEN, skip constant mode 0),
    eigenvalues. Tries data/eigen, data/eigenbases, data/experiments/eigen.
    Returns (V, F, evecs, evals) with evecs shape (N_EIGEN, n_vertices), or None.
    """
    # Name variants: sphere, sphere_eigen, sphere_genus0
    candidates = []
    for base in [EIGEN_DIR, EIGENBASES_DIR, EXPERIMENTS_EIGEN]:
        if not base.exists():
            continue
        candidates.append((base, f"{mesh_name}.npz"))
        candidates.append((base, f"{mesh_name}.json"))
        if base == EIGENBASES_DIR:
            candidates.append((base, f"{mesh_name}_eigen.npz"))
        if base == EXPERIMENTS_EIGEN:
            genus_suffix = {"sphere": "genus0", "torus": "genus1", "double_torus": "genus2"}.get(mesh_name)
            if genus_suffix:
                candidates.append((base, f"{mesh_name}_{genus_suffix}.npz"))
                candidates.append((base, f"{mesh_name}_{genus_suffix}.json"))

    for base, name in candidates:
        path = base / name
        if not path.exists():
            continue
        try:
            if path.suffix == ".npz":
                data = np.load(path, allow_pickle=True)
                files = set(data.files)
                if "vertices" in files and "faces" in files:
                    V = np.asarray(data["vertices"], dtype=np.float64)
                    F = np.asarray(data["faces"], dtype=np.int32)
                else:
                    continue
                if "eigenvectors" in files:
                    evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
                    evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel() if "eigenvalues" in files else None
                elif "phis" in files:
                    evecs = np.asarray(data["phis"], dtype=np.float64)
                    evals = np.asarray(data["lambdas"], dtype=np.float64).ravel() if "lambdas" in files else None
                else:
                    continue
            else:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                V = np.array(data["vertices"], dtype=np.float64)
                F = np.array(data["faces"], dtype=np.int32)
                evecs = np.array(data["eigenvectors"], dtype=np.float64)
                evals = np.array(data.get("eigenvalues", []), dtype=np.float64)
                if evals.size == 0:
                    evals = None

            n_verts = V.shape[0]
            if evecs.shape[0] == n_verts:
                evecs = evecs.T  # (n_verts, K) -> (K, n_verts)
            if evecs.shape[1] != n_verts:
                evecs = evecs.T
            # Need at least N_EIGEN+1 modes to skip mode 0
            n_modes = evecs.shape[0]
            if n_modes > N_EIGEN:
                # Skip constant mode (index 0); use modes 1..N_EIGEN
                evecs = evecs[1 : N_EIGEN + 1, :]
                evals = evals[1 : N_EIGEN + 1] if evals is not None else None
            else:
                evecs = evecs[:N_EIGEN, :]
                evals = evals[:N_EIGEN] if evals is not None else None
            if evecs.shape[0] != N_EIGEN:
                continue
            return V, F, evecs, evals
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}", file=sys.stderr)
            continue
    return None


def build_audio_a3():
    """Load A3 (piano) and tile/pad to TARGET_DURATION_SEC for frame selection."""
    sig, sr = get_audio("A3")
    n_target = int(TARGET_DURATION_SEC * sr)
    if len(sig) < n_target:
        repeats = (n_target + len(sig) - 1) // len(sig)
        sig = np.tile(sig, repeats)[:n_target]
    else:
        sig = sig[:n_target]
    return sig, sr


def compute_rms_per_frame(signal: np.ndarray, frames: list) -> np.ndarray:
    """RMS per STFT frame (same window boundaries as STFT)."""
    rms = []
    for mag, start in frames:
        win = signal[start : start + N_FFT]
        rms.append(np.sqrt(np.mean(win.astype(np.float64) ** 2)))
    return np.array(rms)


def select_three_frames(rms: np.ndarray, n_frames: int):
    """
    Return (idx_sparse, idx_climax, idx_quiet).
    Sparse: frame near start (SPARSE_FRAME_INDEX, or min(SPARSE_FRAME_INDEX, n_frames-1)).
    Climax: argmax RMS.
    Quiet: local minimum of RMS after climax (or last frame if none).
    """
    idx_sparse = min(SPARSE_FRAME_INDEX, n_frames - 1) if n_frames else 0
    idx_climax = int(np.argmax(rms))
    # Local min after climax
    idx_quiet = idx_climax
    for i in range(idx_climax + 1, n_frames - 1):
        if rms[i] <= rms[i - 1] and rms[i] <= rms[i + 1]:
            idx_quiet = i
            break
    else:
        idx_quiet = min(idx_climax + (n_frames // 4), n_frames - 1)
    return idx_sparse, idx_climax, idx_quiet


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load A3 and get frames
    sig, sr = build_audio_a3()
    frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
    if not frames:
        print("No STFT frames from A3.", file=sys.stderr)
        return 1
    rms = compute_rms_per_frame(sig, frames)
    n_frames = len(frames)
    idx_sparse, idx_climax, idx_quiet = select_three_frames(rms, n_frames)
    frame_indices = [idx_sparse, idx_climax, idx_quiet]
    row_labels = [
        f"Sparse (frame {idx_sparse})",
        f"Climax (frame {idx_climax})",
        f"Quiet (frame {idx_quiet})",
    ]
    rms_values = [float(rms[i]) for i in frame_indices]
    print("Selected frames (A3 piano):")
    for name, idx, val in zip(row_labels, frame_indices, rms_values):
        print(f"  {name}: RMS = {val:.6f}")
    print()

    # Load eigen for each shape
    mesh_data = {}
    for shape in SHAPES:
        out = load_mesh_eigen(shape)
        if out is None:
            print(f"Skip shape {shape}: no eigen data.", file=sys.stderr)
            continue
        V, F, evecs, evals = out
        mesh_data[shape] = {"V": V, "F": F, "evecs": evecs, "evals": evals}
    if len(mesh_data) != 3:
        print("Need eigen data for sphere, torus, double_torus.", file=sys.stderr)
        return 1

    # Compute per (frame_index, shape): coef, scalar field, beta0, A_ratio
    # grid[row][col] = (V, F, f, beta0, A_ratio) with row=0,1,2 = sparse, climax, quiet; col=0,1,2 = sphere, torus, double_torus
    grid = [[None] * 3 for _ in range(3)]
    csv_rows = []
    all_f_values = []

    for row, frame_idx in enumerate(frame_indices):
        mag, _ = frames[frame_idx]
        rms_val = rms_values[row]
        for col, shape in enumerate(SHAPES):
            if shape not in mesh_data:
                continue
            V = mesh_data[shape]["V"]
            F = mesh_data[shape]["F"]
            evecs = mesh_data[shape]["evecs"]
            evals = mesh_data[shape]["evals"]
            coef = map_fft_to_coefficients(mag, N_EIGEN, strategy="direct", eigenvalues=evals)
            f = compute_scalar_field(evecs, coef)
            nodal_set, metrics = extract_nodal_and_metrics(
                V, F, f, threshold_ratio=THRESHOLD_RATIO
            )
            grid[row][col] = (V, F, f, metrics.beta0, metrics.A_ratio)
            all_f_values.append(f)
            csv_rows.append({
                "frame_index": frame_idx,
                "rms": round(rms_val, 8),
                "shape": shape,
                "beta0": metrics.beta0,
                "a_ratio": round(metrics.A_ratio, 6),
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index", "rms", "shape", "beta0", "a_ratio"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Saved {OUT_CSV}")

    # Global color scale
    all_f = np.concatenate([x.ravel() for x in all_f_values])
    f_max = np.max(np.abs(all_f))
    vmin, vmax = -f_max * 0.95, f_max * 0.95
    if vmax - vmin < 1e-12:
        vmin, vmax = -1.0, 1.0

    # Render 3×3
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("matplotlib not available; skipping figure.", file=sys.stderr)
        return 0

    try:
        cmap = cm.colormaps["RdBu_r"]
    except (AttributeError, KeyError):
        cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Camera angles per column (shape) for consistency
    view_per_col = [
        (20, 45),   # sphere
        (20, 45),   # torus
        (20, 45),   # double torus
    ]

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(3, 4, width_ratios=[0.5, 1, 1, 1], hspace=0.25, wspace=0.15)
    for row in range(3):
        # Row label (left column)
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.axis("off")
        idx = frame_indices[row]
        rv = rms_values[row]
        ax_label.text(1, 0.5, f"{row_labels[row]}\nRMS={rv:.4f}", fontsize=9, va="center", ha="right")
        for col in range(3):
            ax = fig.add_subplot(gs[row, col + 1], projection="3d")
            cell = grid[row][col]
            if cell is None:
                ax.text(0.5, 0.5, 0.5, "No data", ha="center")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                ax.axis("off")
                continue
            V, F, f, beta0, A_ratio = cell
            elev, azim = view_per_col[col]
            minS, maxS = float(f.min()), float(f.max())
            rangeS = maxS - minS if maxS > minS else 1.0
            norm_local = plt.Normalize(vmin=minS, vmax=maxS)
            nodal_eps = THRESHOLD_RATIO * max(np.max(np.abs(f)), 1e-10)
            verts = [V[F[k]] for k in range(len(F))]
            face_colors = np.zeros((len(F), 4))
            for k in range(len(F)):
                s = float(f[F[k]].mean())
                if np.abs(s) <= nodal_eps:
                    face_colors[k] = (0, 0, 0, 1)
                else:
                    v_norm = (s - minS) / rangeS * 2.0 - 1.0
                    is_contour = False
                    for i in range(CONTOUR_LEVELS):
                        level = -1.0 + (2.0 * i + 1.0) / CONTOUR_LEVELS
                        if abs(v_norm - level) < CONTOUR_BAND:
                            is_contour = True
                            break
                    if is_contour:
                        face_colors[k] = (0.95, 0.9, 0.6, 1.0)
                    else:
                        face_colors[k] = (*cmap(norm_local(s))[:3], 1)
            poly = Poly3DCollection(verts, facecolors=face_colors, edgecolors="none", shade=False)
            ax.add_collection3d(poly)
            ax.set_xlim(V[:, 0].min(), V[:, 0].max())
            ax.set_ylim(V[:, 1].min(), V[:, 1].max())
            ax.set_zlim(V[:, 2].min(), V[:, 2].max())
            ax.set_aspect("equal")
            ax.view_init(elev=elev, azim=azim)
            ax.axis("off")
            shape_name = ["Sphere", "Torus", "Double Torus"][col]
            if row == 0:
                ax.set_title(shape_name, fontsize=10)
            ax.text2D(0.98, 0.02, f"$\\beta_0$={beta0}", transform=ax.transAxes, fontsize=9, ha="right", va="bottom")

    plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.02, wspace=0.12, hspace=0.2)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=FIG_DPI)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=FIG_DPI)
    plt.close()
    print(f"Saved {OUT_PDF} and {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
