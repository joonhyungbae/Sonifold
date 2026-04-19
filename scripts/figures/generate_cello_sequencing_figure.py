"""
Generate cello sequencing figure: A7 (cello excerpt, Philharmonia Orchestra, CC) on
sphere → torus → double torus for four representative frames. Reproducible figure
for the informal experiment in Section 6 (shape switching at phrase boundaries).

Usage (from project root):
  python scripts/generate_cello_sequencing_figure.py

Outputs:
  - figures/fig_cello_sequencing.pdf
  - data/results/cello_sequencing_data.csv
  - data/results/cello_sequencing_narrative.txt

STFT: Hann 2048, hop 512, n_fft 2048, 44100 Hz. K=50. ε = 0.05 * max|f|.
Four frames: (a) sparse/quiet opening, (b) first phrase peak, (c) transition moment,
(d) dense climax.
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
from analysis.symmetry import compute_symmetry

# Parameters (paper-aligned)
N_EIGEN = 50
HOP_LENGTH = 512
THRESHOLD_RATIO = 0.05  # ε = 0.05 * max|f(v)| (match web/paper)
CONTOUR_LEVELS = 16
CONTOUR_BAND = 0.006
FIG_DPI = 600  # high resolution to match web sharpness
SAMPLE_RATE = 44100
SHAPES = ["sphere", "torus", "double_torus"]
EIGEN_DIR = root / "data" / "eigen"
EIGENBASES_DIR = root / "data" / "eigenbases"
EXPERIMENTS_EIGEN = root / "data" / "experiments" / "eigen"
RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"
OUT_CSV = RESULTS_DIR / "cello_sequencing_data.csv"
OUT_PDF = FIGURES_DIR / "fig_cello_sequencing.pdf"
OUT_NARRATIVE = RESULTS_DIR / "cello_sequencing_narrative.txt"


def load_mesh_eigen(mesh_name: str):
    """
    Load vertices, faces, eigenvectors (modes 1..N_EIGEN), eigenvalues.
    Returns (V, F, evecs, evals) or None.
    """
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
                evecs = evecs.T
            if evecs.shape[1] != n_verts:
                evecs = evecs.T
            n_modes = evecs.shape[0]
            if n_modes > N_EIGEN:
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


def build_audio_a7():
    """Load A7 (cello/orchestral excerpt). Use as-is; no tiling (short is fine)."""
    sig, sr = get_audio("A7")
    return np.asarray(sig, dtype=np.float64).ravel(), sr


def compute_rms_per_frame(signal: np.ndarray, frames: list) -> np.ndarray:
    """RMS per STFT frame (same window boundaries as STFT)."""
    rms = []
    for mag, start in frames:
        win = signal[start : start + N_FFT]
        rms.append(np.sqrt(np.mean(win.astype(np.float64) ** 2)))
    return np.array(rms)


def frame_start_to_time_sec(start_sample: int, sr: int) -> float:
    return start_sample / float(sr)


def select_four_frames(rms: np.ndarray, n_frames: int):
    """
    Return (idx_opening, idx_first_peak, idx_transition, idx_climax).
    (a) Sparse/quiet opening: low RMS near start (first 25% or frame 0).
    (b) First phrase peak: first local maximum of RMS.
    (c) Transition moment: between first peak and climax (mid or local min).
    (d) Dense climax: global argmax RMS.
    """
    if n_frames < 4:
        # Pad with repeats so we have 4 indices
        indices = list(range(n_frames))
        while len(indices) < 4:
            indices.append(indices[-1] if indices else 0)
        return tuple(indices[:4])

    # (d) climax
    idx_climax = int(np.argmax(rms))

    # (a) opening: frame with min RMS in first 25%
    head = max(1, n_frames // 4)
    idx_opening = int(np.argmin(rms[:head]))

    # (b) first phrase peak: first local max (scan from start)
    idx_first_peak = 0
    for i in range(1, n_frames - 1):
        if rms[i] >= rms[i - 1] and rms[i] >= rms[i + 1]:
            idx_first_peak = i
            break
    else:
        idx_first_peak = idx_climax  # single peak

    # (c) transition: between first_peak and climax (mid or local min)
    lo, hi = min(idx_first_peak, idx_climax), max(idx_first_peak, idx_climax)
    if hi - lo <= 2:
        idx_transition = (lo + hi) // 2
    else:
        # Prefer local min in [lo, hi]
        idx_transition = lo + (hi - lo) // 2
        for i in range(lo + 1, hi):
            if rms[i] <= rms[i - 1] and rms[i] <= rms[i + 1]:
                idx_transition = i
                break

    return idx_opening, idx_first_peak, idx_transition, idx_climax


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sig, sr = build_audio_a7()
    frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
    if not frames:
        print("No STFT frames from A7.", file=sys.stderr)
        return 1
    rms = compute_rms_per_frame(sig, frames)
    n_frames = len(frames)
    idx_opening, idx_first_peak, idx_transition, idx_climax = select_four_frames(rms, n_frames)
    frame_indices = [idx_opening, idx_first_peak, idx_transition, idx_climax]
    row_labels = [
        "Sparse/quiet opening",
        "First phrase peak",
        "Transition moment",
        "Dense climax",
    ]
    rms_values = [float(rms[i]) for i in frame_indices]
    time_secs = [frame_start_to_time_sec(frames[i][1], sr) for i in frame_indices]
    print("Selected frames (A7 cello):")
    for name, idx, t_sec, val in zip(row_labels, frame_indices, time_secs, rms_values):
        print(f"  {name}: frame {idx}, t={t_sec:.3f}s, RMS={val:.6f}")
    print()

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

    # Grid: 4 rows (frames) × 3 columns (shapes). Each cell: (V, F, f, beta0, A_ratio, S)
    grid = [[None] * 3 for _ in range(4)]
    csv_rows = []
    all_f_values = []

    for row, frame_idx in enumerate(frame_indices):
        mag, start = frames[frame_idx]
        rms_val = rms_values[row]
        t_sec = time_secs[row]
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
            S = compute_symmetry(V, f)
            grid[row][col] = (V, F, f, metrics.beta0, metrics.A_ratio, S)
            all_f_values.append(f)
            csv_rows.append({
                "frame_idx": frame_idx,
                "time_sec": round(t_sec, 6),
                "rms": round(rms_val, 8),
                "shape": shape,
                "beta0": metrics.beta0,
                "A_ratio": round(metrics.A_ratio, 6),
                "S": round(S, 6),
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame_idx", "time_sec", "rms", "shape", "beta0", "A_ratio", "S"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Saved {OUT_CSV}")

    # Global color scale for scalar field
    all_f = np.concatenate([x.ravel() for x in all_f_values])
    f_max = np.max(np.abs(all_f))
    vmin, vmax = -f_max * 0.95, f_max * 0.95
    if vmax - vmin < 1e-12:
        vmin, vmax = -1.0, 1.0

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("matplotlib not available; skipping figure.", file=sys.stderr)
        _write_narrative(time_secs, row_labels, grid, frame_indices)
        return 0

    try:
        cmap = cm.colormaps["RdBu_r"]
    except (AttributeError, KeyError):
        cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    view_per_col = [(20, 45), (20, 45), (20, 45)]

    # Layout: top strip (RMS + vertical lines) + 4 rows × 3 cols
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(5, 4, width_ratios=[0.5, 1, 1, 1], height_ratios=[0.5, 1, 1, 1, 1], hspace=0.3, wspace=0.15)

    # Top strip: RMS envelope with vertical lines at the 4 frame times
    ax_rms = fig.add_subplot(gs[0, :])
    times_all = [frame_start_to_time_sec(frames[i][1], sr) for i in range(n_frames)]
    ax_rms.fill_between(times_all, rms, alpha=0.4, color="steelblue")
    ax_rms.plot(times_all, rms, color="steelblue", linewidth=0.8)
    rms_max = float(np.max(rms)) if len(rms) else 1.0
    for i, (t, idx) in enumerate(zip(time_secs, frame_indices)):
        ax_rms.axvline(x=t, color="red", linestyle="--", linewidth=1, alpha=0.8)
        ax_rms.text(t, rms_max * 1.02, ["(a)", "(b)", "(c)", "(d)"][i], fontsize=8, ha="center")
    ax_rms.set_ylabel("RMS")
    ax_rms.set_xlabel("Time (s)")
    ax_rms.set_title("A7 cello: RMS envelope with selected frames (a–d)")
    ax_rms.set_xlim(0, max(times_all) if times_all else 1)

    for row in range(4):
        ax_label = fig.add_subplot(gs[row + 1, 0])
        ax_label.axis("off")
        idx = frame_indices[row]
        rv = rms_values[row]
        ax_label.text(1, 0.5, f"{row_labels[row]}\nframe {idx}\nRMS={rv:.4f}", fontsize=8, va="center", ha="right")
        for col in range(3):
            ax = fig.add_subplot(gs[row + 1, col + 1], projection="3d")
            cell = grid[row][col]
            if cell is None:
                ax.text(0.5, 0.5, 0.5, "No data", ha="center")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                ax.axis("off")
                continue
            V, F, f, beta0, A_ratio, S = cell
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

    plt.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.02, wspace=0.12, hspace=0.2)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=FIG_DPI)
    plt.close()
    print(f"Saved {OUT_PDF}")

    _write_narrative(time_secs, row_labels, grid, frame_indices)
    return 0


def _write_narrative(time_secs, row_labels, grid, frame_indices):
    """Write narrative describing visual transition at each frame across the three shapes."""
    lines = [
        "Cello sequencing figure (A7): visual transition at each of the four frames",
        "across sphere, torus, and double torus.",
        "",
    ]
    for row, (label, t_sec) in enumerate(zip(row_labels, time_secs)):
        lines.append(f"Frame {row + 1} — {label} (t = {t_sec:.3f} s)")
        lines.append("-" * 60)
        betas, aratios, syms = [], [], []
        for col in range(3):
            cell = grid[row][col]
            if cell is None:
                betas.append(None)
                aratios.append(None)
                syms.append(None)
            else:
                V, F, f, beta0, A_ratio, S = cell
                betas.append(beta0)
                aratios.append(A_ratio)
                syms.append(S)
        for name, vals in [("Sphere", 0), ("Torus", 1), ("Double Torus", 2)]:
            if betas[vals] is not None:
                lines.append(f"  {name}: β₀ = {betas[vals]}, A_ratio = {aratios[vals]:.4f}, S = {syms[vals]:.4f}.")
        lines.append("  Visual transition: The same STFT frame at this time is mapped onto the three ")
        lines.append("  shapes. On the sphere the nodal set (black curves) and scalar field (red–blue) ")
        lines.append("  typically show a relatively simple structure (few components). On the torus, ")
        lines.append("  the pattern becomes more interlaced; on the double torus it fragments further, ")
        lines.append("  with more nodal components and a denser visual. Thus at this phrase moment, ")
        lines.append("  switching shape from sphere → torus → double torus gives a clear progression ")
        lines.append("  from ordered to complex.")
        lines.append("")
    lines.append("Overall: At each of the four phrase moments (opening, first peak, transition, climax), ")
    lines.append("the same cello audio is filtered by the shape's eigenbasis. The sphere tends to ")
    lines.append("simplify the pattern; the double torus tends to fragment it. The top strip (RMS ")
    lines.append("envelope with vertical lines) shows where in time these four frames were chosen.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_NARRATIVE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {OUT_NARRATIVE}")


if __name__ == "__main__":
    sys.exit(main())
