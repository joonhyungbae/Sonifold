"""
Render 1×3 comparison of direct / mel-weighted / energy-weighted nodal patterns
for a single mesh × audio pair. Matches fig3 style (matplotlib 3D, RdBu_r,
nodal in black, same camera). Used for paper: visual comparison for Table 2.

Run from project root:
  python scripts/render_mapping_comparison.py

Outputs:
  figures/fig_mapping_comparison.pdf     — torus × A3
  figures/fig_mapping_comparison_dt.pdf  — double_torus × A5
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
from analysis.nodal_surface import compute_topology_metrics
from analysis.symmetry import compute_symmetry

EIGEN_DIR = root / "data" / "eigen"
N_EIGEN = 50
# Match web/paper: nodal ε = 0.05×max|f|; contour 16 levels, band 0.006 (vNorm)
NODAL_THRESHOLD_RATIO = 0.05
CONTOUR_LEVELS = 16
CONTOUR_BAND = 0.006
THRESHOLD_RATIO = 0.01  # for β₀, A_ratio (nodal_surface default)

STRATEGIES = [
    ("direct", "Direct"),
    ("mel", "Mel-weighted"),
    ("energy", "Energy-weighted"),
]

# Output resolution: match web sharpness (high DPI for 3D mesh)
FIG_DPI = 600

# (mesh_name, audio_id) -> output filename (under figures/)
CONFIGURATIONS = [
    ("torus", "A3", "fig_mapping_comparison.pdf"),
    ("double_torus", "A5", "fig_mapping_comparison_dt.pdf"),
]


def load_mesh_eigen(mesh_name: str):
    """Load vertices, faces, eigenvectors, eigenvalues from .npz or .json."""
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


def compute_fields_and_metrics(V, F, evecs, evals, audio_id: str):
    """
    For one mesh × audio: compute 3 scalar fields and (β₀, A_ratio, S) per strategy.
    Returns: list of (field_array, metrics_dict) with keys beta0, A_ratio, S.
    """
    sig, sr = get_audio(audio_id)
    mag, _ = compute_fft(sig, sr)
    out = []
    for strategy_key, _ in STRATEGIES:
        coef = map_fft_to_coefficients(
            mag, N_EIGEN, strategy=strategy_key, eigenvalues=evals
        )
        f = compute_scalar_field(evecs, coef)
        m = compute_topology_metrics(V, F, f, threshold_ratio=THRESHOLD_RATIO)
        S = compute_symmetry(V, f)
        out.append((f, {"beta0": m.beta0, "A_ratio": m.A_ratio, "S": S}))
    return out


def render_comparison(V, F, fields_and_metrics, out_path: Path):
    """
    Draw 1×3 subplots with shared color scale and same camera.
    Each panel: nodal pattern + annotation (mapping name, β₀, A_ratio, S).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Global color range across all three fields (min/max for vNorm, match web)
    all_f = [x[0] for x in fields_and_metrics]
    try:
        cmap = cm.colormaps["RdBu_r"]
    except (AttributeError, KeyError):
        cmap = plt.cm.RdBu_r

    fig = plt.figure(figsize=(9, 3.2), dpi=FIG_DPI)
    for idx, (f, metrics) in enumerate(fields_and_metrics):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
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
                    face_colors[k] = cmap(norm(s))
        poly = Poly3DCollection(verts, facecolors=face_colors, edgecolors="none", shade=False)
        ax.add_collection3d(poly)
        ax.set_xlim(V[:, 0].min(), V[:, 0].max())
        ax.set_ylim(V[:, 1].min(), V[:, 1].max())
        ax.set_zlim(V[:, 2].min(), V[:, 2].max())
        ax.set_aspect("equal")
        ax.view_init(elev=20, azim=45)
        ax.axis("off")
        title = STRATEGIES[idx][1]
        ax.set_title(title, fontsize=10)
        # Annotation: β₀, A_ratio, S (paper notation)
        b0, ar, s = metrics["beta0"], metrics["A_ratio"], metrics["S"]
        ax.text2D(0.02, 0.02, f"$\\beta_0$={b0}\n$A_{{ratio}}$={ar:.3f}\n$S$={s:.2f}",
                 transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
                 family="serif")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close()


def main():
    figures_dir = root / "figures"
    for mesh_name, audio_id, filename in CONFIGURATIONS:
        try:
            V, F, evecs, evals = load_mesh_eigen(mesh_name)
        except Exception as e:
            print(f"Skip {mesh_name}×{audio_id}: {e}", file=sys.stderr)
            continue
        try:
            data = compute_fields_and_metrics(V, F, evecs, evals, audio_id)
        except Exception as e:
            print(f"Skip {mesh_name}×{audio_id} (compute): {e}", file=sys.stderr)
            continue
        out_path = figures_dir / filename
        render_comparison(V, F, data, out_path)
        print("Saved", out_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
