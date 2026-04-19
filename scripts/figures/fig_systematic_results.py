"""
Paper-style visualization from results_systematic.csv: Genus vs beta0, Asymmetry vs S dual charts + stepwise nodal snapshots.
Seaborn paper style, journal quality.

Run from project root: python figures/fig_systematic_results.py
Requires: data/results/results_systematic.csv (after experiment.run_batch_systematic)
          data/experiments/eigen/*.npz (after experiment.precompute_experiment_eigen)
Output: figures/fig_systematic_results.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

RESULTS_CSV = root / "data" / "results" / "results_systematic.csv"
EIGEN_DIR = root / "data" / "experiments" / "eigen"
OUT_PATH = root / "figures" / "fig_systematic_results.pdf"
N_EIGEN = 50
# Match web/paper: nodal ε = 0.05×max|f|; contour 16 levels, band 0.006 (vNorm)
NODAL_THRESHOLD_RATIO = 0.05
CONTOUR_LEVELS = 16
CONTOUR_BAND = 0.006
FIG_DPI = 600  # high resolution to match web sharpness


def load_eigen(mesh_id):
    npz_path = EIGEN_DIR / (mesh_id + ".npz")
    json_path = EIGEN_DIR / (mesh_id + ".json")
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        V = np.asarray(data["vertices"], dtype=np.float64)
        F = np.asarray(data["faces"], dtype=np.int32)
        evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
        evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
        return V, F, evecs, evals
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        V = np.array(data["vertices"], dtype=np.float64)
        F = np.array(data["faces"], dtype=np.int32)
        evecs = np.array(data["eigenvectors"], dtype=np.float64)
        evals = np.array(data.get("eigenvalues", []), dtype=np.float64)
        return V, F, evecs, evals
    raise FileNotFoundError(mesh_id)


def compute_scalar_for_mesh(mesh_id):
    from audio.audio_library import get_audio
    from audio.fft_analysis import compute_fft
    from mapping.spectral_mapping import map_fft_to_coefficients
    from analysis.scalar_field import compute_scalar_field
    V, F, evecs, evals = load_eigen(mesh_id)
    sig, sr = get_audio("A3")
    mag, _ = compute_fft(sig, sr)
    coef = map_fft_to_coefficients(mag, N_EIGEN, strategy="direct", eigenvalues=evals)
    f = compute_scalar_field(evecs, coef)
    return V, F, f


def main():
    if not RESULTS_CSV.exists():
        print("Run experiment.run_batch_systematic first to create", RESULTS_CSV, file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(RESULTS_CSV)
    genus_df = df[df["genus"].notna()].sort_values("genus")
    asym_df = df[df["asymmetry_level"].notna()].sort_values("asymmetry_level")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", font_scale=1.1)
        sns.set_context("paper", rc={"figure.dpi": 150})
    except ImportError:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("ggplot")

    from matplotlib.gridspec import GridSpec
    n_genus = len(genus_df)
    n_asym = len(asym_df)
    n_cols = max(2, n_genus, n_asym)
    fig = plt.figure(figsize=(2 * 2.5 + (n_cols - 2) * 1.2, 9))
    gs = GridSpec(3, n_cols, figure=fig, hspace=0.35, wspace=0.25)

    # ---- Row 0: two dual charts ----
    ax1 = fig.add_subplot(gs[0, 0])
    if n_genus > 0:
        ax1.plot(genus_df["genus"], genus_df["beta0"], "o-", color="C0", linewidth=2, markersize=8)
        ax1.set_xlabel("Genus")
        ax1.set_ylabel(r"$\beta_0$", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        ax1.set_xticks(genus_df["genus"].astype(int))
    ax1.set_title(r"Genus vs $\beta_0$ (A3, direct)")
    ax1.set_facecolor("white")

    ax2 = fig.add_subplot(gs[0, 1])
    if n_asym > 0:
        ax2.plot(asym_df["asymmetry_level"], asym_df["S"], "s-", color="C1", linewidth=2, markersize=8)
        ax2.set_xlabel("Asymmetry (z-stretch)")
        ax2.set_ylabel("$S$", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
    ax2.set_title("Symmetry breaking vs $S$")
    ax2.set_facecolor("white")

    # ---- Row 1: genus snapshots ----
    genus_ids = genus_df["mesh_id"].tolist()
    for i, mid in enumerate(genus_ids):
        ax = fig.add_subplot(gs[1, i], projection="3d")
        if EIGEN_DIR.exists():
            try:
                V, F, f = compute_scalar_for_mesh(mid)
                g = int(genus_df[genus_df["mesh_id"] == mid]["genus"].iloc[0])
                _draw_nodal_3d(ax, V, F, f, title="g={}".format(g))
            except Exception as e:
                ax.text(0.5, 0.5, 0.5, str(e)[:20], ha="center", fontsize=6)
        else:
            ax.text(0.5, 0.5, 0.5, "No eigen", ha="center")
        ax.set_facecolor("white")

    # ---- Row 2: asymmetry snapshots ----
    asym_ids = asym_df["mesh_id"].tolist()
    for i, mid in enumerate(asym_ids):
        ax = fig.add_subplot(gs[2, i], projection="3d")
        if EIGEN_DIR.exists():
            try:
                V, F, f = compute_scalar_for_mesh(mid)
                a = asym_df[asym_df["mesh_id"] == mid]["asymmetry_level"].iloc[0]
                _draw_nodal_3d(ax, V, F, f, title="{:.1f}".format(a))
            except Exception as e:
                ax.text(0.5, 0.5, 0.5, str(e)[:20], ha="center", fontsize=6)
        else:
            ax.text(0.5, 0.5, 0.5, "No eigen", ha="center")
        ax.set_facecolor("white")

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=FIG_DPI)
    plt.close()
    print("Saved", OUT_PATH, file=sys.stderr)
    return 0


def _draw_nodal_3d(ax, V, F, f, title=""):
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    minS, maxS = float(f.min()), float(f.max())
    rangeS = maxS - minS if maxS > minS else 1.0
    norm = plt.Normalize(vmin=minS, vmax=maxS)
    nodal_eps = NODAL_THRESHOLD_RATIO * max(np.max(np.abs(f)), 1e-10)
    try:
        cmap = cm.colormaps["RdBu_r"]
    except (AttributeError, KeyError):
        cmap = plt.cm.RdBu_r
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
    if title:
        ax.set_title(title, fontsize=9)


# Need plt in module scope for _draw_nodal_3d
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sys.exit(main() or 0)
