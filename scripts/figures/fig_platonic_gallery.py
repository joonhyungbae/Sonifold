"""
Generate Platonic gallery: tetrahedron, octahedron, icosahedron × A1, A5.
Paper caption: Nodal patterns on the three Platonic solids for stimuli A1 and A5.
Run from project root: python figures/fig_platonic_gallery.py
Output: figures/fig_platonic_gallery.pdf
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
N_EIGEN = 50
MESHES = ["tetrahedron", "octahedron", "icosahedron"]
AUDIOS = ["A1", "A5"]
# Match web/paper: nodal ε = 0.05×max|f|; contour 16 levels, band 0.006 (vNorm)
NODAL_THRESHOLD_RATIO = 0.05
CONTOUR_LEVELS = 16
CONTOUR_BAND = 0.006
FIG_DPI = 600  # high resolution to match web sharpness


def load_mesh_eigen(mesh_name: str):
    npz_path = EIGEN_DIR / f"{mesh_name}.npz"
    json_path = EIGEN_DIR / f"{mesh_name}.json"
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        V = np.asarray(data["vertices"], dtype=np.float64)
        F = np.asarray(data["faces"], dtype=np.int32)
        evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
        evals = data["eigenvalues"] if "eigenvalues" in data.files else None
        if evals is not None:
            evals = np.asarray(evals, dtype=np.float64).ravel()
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


def compute_scalar_for(V, F, evecs, evals, audio_id: str):
    sig, sr = get_audio(audio_id)
    mag, _ = compute_fft(sig, sr)
    coef = map_fft_to_coefficients(mag, N_EIGEN, strategy="direct", eigenvalues=evals)
    f = compute_scalar_field(evecs, coef)
    return f


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_path = root / "figures" / "fig_platonic_gallery.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_grid = []
    for mesh_name in MESHES:
        try:
            V, F, evecs, evals = load_mesh_eigen(mesh_name)
        except Exception as e:
            print(f"Skip mesh {mesh_name}: {e}", file=sys.stderr)
            data_grid.append([None] * len(AUDIOS))
            continue
        row = []
        for audio_id in AUDIOS:
            try:
                f = compute_scalar_for(V, F, evecs, evals, audio_id)
                row.append((V, F, f))
            except Exception as e:
                print(f"Skip {mesh_name} x {audio_id}: {e}", file=sys.stderr)
                row.append(None)
        data_grid.append(row)

    try:
        cmap = cm.colormaps["RdBu_r"]
    except (AttributeError, KeyError):
        cmap = plt.cm.RdBu_r
    n_rows, n_cols = len(MESHES), len(AUDIOS)
    fig = plt.figure(figsize=(6, 3.2 * n_rows), dpi=FIG_DPI)
    for i, mesh_name in enumerate(MESHES):
        for j, audio_id in enumerate(AUDIOS):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection="3d")
            cell = data_grid[i][j]
            if cell is None:
                ax.text(0.5, 0.5, 0.5, f"{mesh_name}\n{audio_id}\n(no data)", ha="center")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                continue
            V, F, f = cell
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
                    for ci in range(CONTOUR_LEVELS):
                        level = -1.0 + (2.0 * ci + 1.0) / CONTOUR_LEVELS
                        if abs(v_norm - level) < CONTOUR_BAND:
                            is_contour = True
                            break
                    if is_contour:
                        face_colors[k] = (0.95, 0.9, 0.6, 1.0)
                    else:
                        face_colors[k] = cmap(norm(s))
            poly = Poly3DCollection(verts, facecolors=face_colors, edgecolors="none", shade=False)
            ax.add_collection3d(poly)
            ax.set_xlim(V[:, 0].min(), V[:, 0].max())
            ax.set_ylim(V[:, 1].min(), V[:, 1].max())
            ax.set_zlim(V[:, 2].min(), V[:, 2].max())
            ax.set_aspect("equal")
            ax.set_title(f"{mesh_name.replace('_', ' ')}, {audio_id}", fontsize=9)
            ax.view_init(elev=20, azim=45)
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close()
    print("Saved", out_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
