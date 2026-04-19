"""
High-resolution renders for journal submission (JMAA).
- fig_highlight_1..3.pdf: Sphere+A1 direct, Torus+A3 mel, Double torus+A5 direct (2000x2000px).
- fig3_detail_1..3.pdf: Same three panels from Figure 3 gallery at same quality.

Run from project root: python figures/render_highlight_figures.py
Requires: data/eigen/*.npz (sphere, torus, double_torus).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field

EIGEN_DIR = root / "data" / "eigen"
FIG_DIR = root / "figures"
N_EIGEN = 50
# Match web/paper: nodal ε = 0.05×max|f|; contour 16 levels, band 0.006 (vNorm)
NODAL_THRESHOLD_RATIO = 0.05
CONTOUR_LEVELS = 16
CONTOUR_BAND = 0.006
MIN_PX = 3000   # per-panel side length in px (match web sharpness)
DPI = 300       # MIN_PX/DPI = 10 inch


def load_mesh_eigen(mesh_name: str):
    npz_path = EIGEN_DIR / f"{mesh_name}.npz"
    json_path = EIGEN_DIR / f"{mesh_name}.json"
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        V = np.asarray(data["vertices"], dtype=np.float64)
        F = np.asarray(data["faces"], dtype=np.int32)
        evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
        evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel() if "eigenvalues" in data.files else None
        return V, F, evecs, evals
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        V = np.array(data["vertices"], dtype=np.float64)
        F = np.array(data["faces"], dtype=np.int32)
        evecs = np.array(data["eigenvectors"], dtype=np.float64)
        evals = np.array(data.get("eigenvalues", []), dtype=np.float64)
        return V, F, evecs, evals
    raise FileNotFoundError(f"No eigen data for {mesh_name}")


def compute_scalar(V, F, evecs, evals, audio_id: str, strategy: str = "direct"):
    sig, sr = get_audio(audio_id)
    mag, _ = compute_fft(sig, sr)
    coef = map_fft_to_coefficients(mag, N_EIGEN, strategy=strategy, eigenvalues=evals)
    return compute_scalar_field(evecs, coef)


def chladni_cmap():
    """Blue (cool) for negative, black at nodal, orange (warm) for positive."""
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0.15, 0.35, 0.9), (0.0, 0.0, 0.0), (0.95, 0.5, 0.1)]
    return LinearSegmentedColormap.from_list("chladni", colors, N=256)


def render_one_panel(V, F, f, out_path: Path, title: str = "", elev=20, azim=45, background="white"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(MIN_PX / DPI, MIN_PX / DPI), facecolor=background)
    ax = fig.add_subplot(111, projection="3d", facecolor=background)
    cmap = chladni_cmap()
    minS, maxS = float(f.min()), float(f.max())
    rangeS = maxS - minS if maxS > minS else 1.0
    norm = plt.Normalize(vmin=minS, vmax=maxS)
    nodal_eps = NODAL_THRESHOLD_RATIO * max(np.max(np.abs(f)), 1e-10)
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
                face_colors[k] = (*cmap(norm(s))[:3], 1.0)
    poly = Poly3DCollection(verts, facecolors=face_colors, edgecolors="none", shade=False)
    ax.add_collection3d(poly)
    ax.set_xlim(V[:, 0].min(), V[:, 0].max())
    ax.set_ylim(V[:, 1].min(), V[:, 1].max())
    ax.set_zlim(V[:, 2].min(), V[:, 2].max())
    ax.set_aspect("equal")
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14)
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI, facecolor=background, edgecolor="none")
    plt.close()
    print("Saved", out_path, file=sys.stderr)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Highlight figures: (mesh, audio, strategy) -> fig_highlight_1..3.pdf
    highlights = [
        ("sphere", "A1", "direct", "Sphere, A1 (pure tone), direct"),
        ("torus", "A3", "mel", "Torus, A3 (piano), mel-weighted"),
        ("double_torus", "A5", "direct", "Double torus, A5 (white noise), direct"),
    ]
    for i, (mesh_name, audio_id, strategy, title) in enumerate(highlights, 1):
        try:
            V, F, evecs, evals = load_mesh_eigen(mesh_name)
            f = compute_scalar(V, F, evecs, evals, audio_id, strategy=strategy)
            out = FIG_DIR / f"fig_highlight_{i}.pdf"
            render_one_panel(V, F, f, out, title=title, background="white")
        except Exception as e:
            print(f"Skip highlight {i} ({mesh_name}, {audio_id}): {e}", file=sys.stderr)

    # 2) Fig3 detail: same three panels as in gallery (direct mapping), high-res
    details = [
        ("sphere", "A1", "Sphere, A1"),
        ("torus", "A3", "Torus, A3"),
        ("double_torus", "A5", "Double torus, A5"),
    ]
    for i, (mesh_name, audio_id, title) in enumerate(details, 1):
        try:
            V, F, evecs, evals = load_mesh_eigen(mesh_name)
            f = compute_scalar(V, F, evecs, evals, audio_id, strategy="direct")
            out = FIG_DIR / f"fig3_detail_{i}.pdf"
            render_one_panel(V, F, f, out, title=title, background="white")
        except Exception as e:
            print(f"Skip fig3_detail {i} ({mesh_name}, {audio_id}): {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
