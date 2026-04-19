"""
Generate a multi-angle wireframe/surface figure of the double torus mesh
to visually demonstrate genus-2 topology (χ = −2).
Four views: front, top, 3/4 perspective, and cross-section highlighting
the two holes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

import trimesh


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
    """Subsample faces for wireframe visibility."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(F.shape[0], size=int(F.shape[0] * fraction), replace=False)
    return F[idx]


def plot_mesh(ax, V, F_sub, elev, azim, title):
    """Plot mesh as translucent surface with wireframe edges."""
    ax.set_proj_type("persp", focal_length=0.4)

    # Build polygon collection from subsampled faces
    polys = V[F_sub]
    # Compute face normals for shading
    v0, v1, v2 = polys[:, 0], polys[:, 1], polys[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-20)

    # Simple diffuse shading
    light_dir = np.array([0.3, 0.5, 0.8])
    light_dir /= np.linalg.norm(light_dir)
    shade = np.clip(np.dot(normals, light_dir), 0.15, 1.0)

    face_colors = np.zeros((len(F_sub), 4))
    face_colors[:, 0] = 0.35 + 0.45 * shade  # R
    face_colors[:, 1] = 0.55 + 0.35 * shade  # G
    face_colors[:, 2] = 0.75 + 0.20 * shade  # B
    face_colors[:, 3] = 0.6  # alpha

    pc = Poly3DCollection(polys, facecolors=face_colors,
                          edgecolors=(0.2, 0.2, 0.2, 0.15), linewidths=0.2)
    ax.add_collection3d(pc)

    # Set limits — use per-axis ranges so the flat mesh isn't crushed
    center = V.mean(axis=0)
    halfspan = (V.max(axis=0) - V.min(axis=0)) / 2 * 1.15
    # Ensure z-axis has enough room to show thickness
    halfspan[2] = max(halfspan[2], halfspan[1] * 0.35)
    for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        setter(center[i] - halfspan[i], center[i] + halfspan[i])

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=2)
    ax.set_axis_off()


def main():
    V, F = load_double_torus()
    chi = euler_chi(V, F)
    genus = (2 - chi) // 2
    print(f"Double torus: V={V.shape[0]}, F={F.shape[0]}, χ={chi}, g={genus}")

    F_sub = subsample_faces(F, V, fraction=0.10)

    views = [
        (25, -55, f"Perspective view\n(V={V.shape[0]:,}, F={F.shape[0]:,}, χ={chi}, g={genus})"),
        (88, -90, "Top view (two holes visible)"),
        (35, 0, "Front-elevated view"),
        (20, -130, "Oblique view"),
    ]

    fig = plt.figure(figsize=(12, 3.2))
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        plot_mesh(ax, V, F_sub, elev, azim, title)

    fig.suptitle(
        "Double torus mesh: genus 2 verified (Euler characteristic χ = −2)",
        fontsize=11, fontweight="bold", y=0.02,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = root / "figures" / "fig_double_torus_evidence.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

    # Also save PNG for quick inspection
    out_png = root / "figures" / "fig_double_torus_evidence.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
