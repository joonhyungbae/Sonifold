"""
Generate Fig.8: Spherical harmonics Y_l^m on the unit sphere (l=1,2,3; representative m).
Nodal lines (zero level) in black. Textbook-style, serif font.
Run from project root: python figures/fig8_spherical_harmonics.py
Output: figures/fig8_spherical_harmonics.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import sph_harm

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

# (l, m) pairs: l=1,2,3 with representative m (use real combination for real-valued viz)
# Y_l^m: l=1 -> m=-1,0,1; l=2 -> m=-2..2; l=3 -> m=-3..3. We show one per l for clarity.
L_M_PAIRS = [(1, 0), (2, 0), (3, 0)]  # m=0 gives zonal harmonics (nodal lines are latitude circles)


def real_sph_harm(m, l, theta, phi):
    """Real-valued combination for visualization: Re(Y_l^m) for m>=0, Im(Y_l^m) for m<0 would be sin; we use Re."""
    y = sph_harm(m, l, theta, phi)
    return y.real


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    plt.rc("font", family="serif")
    plt.rc("mathtext", fontset="dejavuserif")

    out_path = root / "figures" / "fig8_spherical_harmonics.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_theta, n_phi = 120, 160
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi)
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)

    try:
        cmap = cm.colormaps["RdBu_r"]
    except (AttributeError, KeyError):
        cmap = plt.cm.RdBu_r

    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(10, 4))
    for ax, (l, m) in zip(axes, L_M_PAIRS):
        Yval = real_sph_harm(m, l, Theta, Phi)
        vmax = np.max(np.abs(Yval)) * 0.95
        vmin = -vmax
        if vmax < 1e-12:
            vmin, vmax = -1, 1
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        facecolors = cmap(norm(Yval))
        ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, shade=False, antialiased=False)
        # Nodal line: get contour at 0 in (phi, theta) from a dummy 2D plot, then map to 3D
        fig_tmp, ax_tmp = plt.subplots()
        cs = ax_tmp.contour(Phi, Theta, Yval, levels=[0])
        plt.close(fig_tmp)
        segs = cs.allsegs[0] if hasattr(cs, "allsegs") and len(cs.allsegs) > 0 else []
        if len(segs) >= 1:
            for seg in segs:
                pts = np.asarray(seg)
                if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] >= 2:
                    ph, th = pts[:, 0], pts[:, 1]
                    x = np.sin(th) * np.cos(ph)
                    y = np.sin(th) * np.sin(ph)
                    z = np.cos(th)
                    ax.plot(x, y, z, "k-", linewidth=1.5)
        ax.set_title(f"$\\ell = {l}$, $m = {m}$", fontsize=12)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_aspect("equal")
        ax.view_init(elev=25, azim=45)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", out_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
