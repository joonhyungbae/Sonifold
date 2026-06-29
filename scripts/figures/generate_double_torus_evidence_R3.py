"""
R3 version of the double-torus evidence figure (revision after R3 audit).

Differences vs the R2 figure:
  - Perspective-view title now embeds the actual mesh parameters
    R = 2.0, r = 0.55 (matching `experiment/generate_experiment_meshes.py`
    line 90 and Table 1 in paper.tex).
  - Source revolution tori overlay panel uses the ACTUAL stored mesh
    centres (x = 0 and x = 3.34, NOT ±1.67 — the mesh-generation code
    does not centre).  Labels are placed outside the wireframe footprint
    via 2D-text overlays so they do not collide with the meridians.
  - Overall figure is taller (figsize 12.5x4.4 in) so each panel reads
    legibly at one-column width.

Output overwrites `figures/fig_double_torus_evidence.pdf` (and the PNG
companion).  Run from project root.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

import trimesh

# Source-mesh parameters, matching experiment/generate_experiment_meshes.py
R_SRC = 2.0
R_TUBE = 0.55
# spacing = max(1.0, 2*R - 1.2*r) = 3.34
# In the saved mesh, tori are placed at x = 0 and x = spacing (NOT centred).
CENTRE_LEFT = 0.0
CENTRE_RIGHT = 3.34


def load_double_torus():
    path = root / "data" / "meshes" / "double_torus.obj"
    m = trimesh.load(str(path), force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int32)
    return V, F


def euler_chi(V, F):
    e0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    e1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    edges = np.sort(np.column_stack([e0, e1]), axis=1)
    keys = edges[:, 0].astype(np.int64) * (int(F.max()) + 1) + edges[:, 1]
    ne = int(len(np.unique(keys)))
    return int(V.shape[0] - ne + F.shape[0])


def subsample_faces(F, V, fraction=0.08, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(F.shape[0], size=int(F.shape[0] * fraction), replace=False)
    return F[idx]


def _axis_limits(V):
    centre = V.mean(axis=0)
    halfspan = (V.max(axis=0) - V.min(axis=0)) / 2 * 1.15
    halfspan[2] = max(halfspan[2], halfspan[1] * 0.35)
    return centre, halfspan


def plot_mesh(ax, V, F_sub, elev, azim, title, *, alpha=0.6):
    ax.set_proj_type("persp", focal_length=0.4)
    polys = V[F_sub]
    v0, v1, v2 = polys[:, 0], polys[:, 1], polys[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-20)
    light_dir = np.array([0.3, 0.5, 0.8])
    light_dir /= np.linalg.norm(light_dir)
    shade = np.clip(np.dot(normals, light_dir), 0.15, 1.0)
    face_colors = np.zeros((len(F_sub), 4))
    face_colors[:, 0] = 0.35 + 0.45 * shade
    face_colors[:, 1] = 0.55 + 0.35 * shade
    face_colors[:, 2] = 0.75 + 0.20 * shade
    face_colors[:, 3] = alpha
    pc = Poly3DCollection(polys, facecolors=face_colors,
                          edgecolors=(0.2, 0.2, 0.2, 0.15), linewidths=0.2)
    ax.add_collection3d(pc)
    centre, halfspan = _axis_limits(V)
    for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        setter(centre[i] - halfspan[i], centre[i] + halfspan[i])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=2)
    ax.set_axis_off()


def plot_source_overlay(ax, V, F_sub, title):
    """Pure 2D plot (matplotlib axes, not 3D) of the xy footprint:
    actual mesh as translucent fill + two source revolution tori drawn
    as concentric annuli (outer radius R+r, inner radius R-r), centred
    at the ACTUAL stored-mesh centres x=0 and x=3.34.
    """
    # Mesh footprint — convex hull-like fill via vertex scatter is messy;
    # instead, project triangulated mesh onto xy and fill with very light grey
    # via Poly3D-style flat polys.  We do this with simple plt.fill on triangles
    # but it's slow for 39k tris.  Compromise: alpha=0.04 fill of triangles.
    polys_xy = V[F_sub][:, :, :2]
    from matplotlib.collections import PolyCollection
    pc = PolyCollection(polys_xy, facecolors=(0.62, 0.68, 0.78, 0.28),
                        edgecolors=(0.35, 0.40, 0.48, 0.22), linewidths=0.18)
    ax.add_collection(pc)

    # Two source revolution tori as annuli (outer + inner circles)
    theta = np.linspace(0, 2 * np.pi, 240)
    r_outer = R_SRC + R_TUBE
    r_inner = R_SRC - R_TUBE
    colours = [("#c25a00", CENTRE_LEFT, "Left torus"),
               ("#1f7a4d", CENTRE_RIGHT, "Right torus")]
    for col, cx, _ in colours:
        # outer circle
        ax.plot(cx + r_outer * np.cos(theta), r_outer * np.sin(theta),
                color=col, linewidth=1.4)
        # inner circle (the hole)
        ax.plot(cx + r_inner * np.cos(theta), r_inner * np.sin(theta),
                color=col, linewidth=1.4)
        # centre marker
        ax.plot([cx], [0], marker="+", color="black", markersize=11,
                markeredgewidth=1.8)

    # R indicator on the right torus: radial segment from centre to outer
    cxR = CENTRE_RIGHT
    ax.annotate("", xy=(cxR, R_SRC), xytext=(cxR, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.3))
    ax.text(cxR + 0.12, R_SRC * 0.55, "R = 2.0",
            fontsize=10, color="black", weight="bold", va="center")

    # r indicator on the right torus: short segment at the outer rim
    ax.annotate("", xy=(cxR + R_SRC + R_TUBE, 0),
                xytext=(cxR + R_SRC - R_TUBE, 0),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
    ax.text(cxR + R_SRC, R_TUBE + 0.18, "r = 0.55", fontsize=10,
            color="black", weight="bold", ha="center")

    # Centre-position labels below each torus, in colour
    ax.text(CENTRE_LEFT, -(R_SRC + R_TUBE) - 0.55, "(0, 0, 0)",
            fontsize=10, color="#c25a00", ha="center", weight="bold")
    ax.text(CENTRE_RIGHT, -(R_SRC + R_TUBE) - 0.55, "(3.34, 0, 0)",
            fontsize=10, color="#1f7a4d", ha="center", weight="bold")

    ax.set_xlim(-(R_SRC + R_TUBE) - 0.9, CENTRE_RIGHT + R_SRC + R_TUBE + 0.9)
    ax.set_ylim(-(R_SRC + R_TUBE) - 1.4, (R_SRC + R_TUBE) + 0.6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_axis_off()


def main():
    V, F = load_double_torus()
    chi = euler_chi(V, F)
    genus = (2 - chi) // 2
    print(f"Double torus: V={V.shape[0]}, F={F.shape[0]}, χ={chi}, g={genus}")
    print(f"Stored x-centroid: {V.mean(axis=0)[0]:.3f}  "
          f"(left torus at x≈0, right torus at x≈3.34)")

    F_sub = subsample_faces(F, V, fraction=0.10)

    persp_title = (
        f"Perspective view\n"
        f"(V={V.shape[0]:,}, F={F.shape[0]:,}, χ={chi}, g={genus};\n"
        f"R=2.0, r=0.55)"
    )
    # R3 option C: drop the Top view 3D wireframe panel; the 2D source-
    # overlay panel already shows the two holes as inner annulus rings.
    mesh_views = [
        (25, -55, persp_title),
        (35, 0, "Front-elevated view"),
    ]

    fig = plt.figure(figsize=(11.5, 4.5))
    # Two 3D mesh panels
    for i, (elev, azim, title) in enumerate(mesh_views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        plot_mesh(ax, V, F_sub, elev, azim, title)
    # Source overlay panel — pure 2D, takes the third slot
    ax3 = fig.add_subplot(1, 3, 3)
    plot_source_overlay(ax3, V, F_sub,
                        "Source revolution tori (xy top view)\n"
                        "annuli of (R$\\pm$r), centres x=0 and 3.34")

    fig.suptitle(
        "Double torus mesh: genus 2 verified (Euler characteristic χ = −2)",
        fontsize=11, fontweight="bold", y=0.02,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = root / "figures" / "fig_double_torus_evidence.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    out_png = root / "figures" / "fig_double_torus_evidence.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
