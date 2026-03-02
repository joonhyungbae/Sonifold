"""
Mesh creation/load and remesh to ~10,000 vertices.
Input: mesh name (sphere, ellipsoid, cube, torus, double_torus, flat_plate,
       tetrahedron, octahedron, icosahedron, bunny, spot, dragon, armadillo)
Output: (vertices, faces) — numpy arrays, vertices (V,3), faces (F,3)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

# Procedural meshes: trimesh + gpytoolbox
try:
    import trimesh
except ImportError:
    trimesh = None
try:
    import gpytoolbox as gpy
except ImportError:
    gpy = None

# Target vertex count (experiment control)
TARGET_VERTICES = 10_000
# Allowed range
VERTEX_TOLERANCE = 0.2  # ±20%

# data path relative to project root
def _data_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "meshes"


def get_mesh(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (vertices, faces) by mesh name. All meshes are fit to ~TARGET_VERTICES vertices.
    name: sphere | ellipsoid | cube | torus | double_torus | flat_plate |
          tetrahedron | octahedron | icosahedron | bunny | spot | dragon | armadillo
    """
    name = name.strip().lower()
    if name == "sphere":
        V, F = _sphere()
    elif name == "ellipsoid":
        V, F = _ellipsoid()
    elif name == "cube":
        V, F = _cube()
    elif name == "torus":
        V, F = _torus()
    elif name == "double_torus":
        V, F = _double_torus()
    elif name == "bunny":
        V, F = _load_obj("bunny")
    elif name == "spot":
        V, F = _load_obj("spot")
    elif name == "flat_plate":
        V, F = _flat_plate()
    elif name == "tetrahedron":
        V, F = _tetrahedron()
    elif name == "octahedron":
        V, F = _octahedron()
    elif name == "icosahedron":
        V, F = _icosahedron()
    elif name == "dragon":
        V, F = _load_obj("dragon")
    elif name == "armadillo":
        V, F = _load_obj("armadillo")
    else:
        raise ValueError(f"Unknown mesh name: {name}")

    V, F = _remesh_to_target(V, F)
    return V.astype(np.float64), F.astype(np.int32)


def _sphere() -> Tuple[np.ndarray, np.ndarray]:
    """Unit sphere. trimesh icosphere subdivisions=5 -> ~10k vertices."""
    if trimesh is None:
        raise ImportError("trimesh required for sphere. pip install trimesh")
    # subdivisions=5 → 10242 vertices
    m = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    return m.vertices, m.faces


def _ellipsoid() -> Tuple[np.ndarray, np.ndarray]:
    """Ellipsoid axis ratio 1:2:3 (sonifold spec)."""
    V, F = _sphere()
    V[:, 0] *= 1.0
    V[:, 1] *= 2.0
    V[:, 2] *= 3.0
    return V, F


def _cube() -> Tuple[np.ndarray, np.ndarray]:
    """Cube. Box then subdivide to ~10k."""
    if trimesh is None:
        raise ImportError("trimesh required for cube. pip install trimesh")
    # 1x1x1 box -> subdivide
    m = trimesh.creation.box(extents=(1, 1, 1))
    # Iterative subdivision to reach ~10k
    for _ in range(3):
        m = m.subdivide()
        if m.vertices.shape[0] >= TARGET_VERTICES * (1 - VERTEX_TOLERANCE):
            break
    return m.vertices, m.faces


def _torus() -> Tuple[np.ndarray, np.ndarray]:
    """Torus R:r = 3:1 (sonifold M4)."""
    if gpy is None:
        raise ImportError("gpytoolbox required for torus. pip install gpytoolbox")
    # R=3, r=1 → nR*nr ≈ 10000. nR=100, nr=100 → 10000 vertices
    nR, nr = 100, 100
    V, F = gpy.torus(nR, nr, R=3.0, r=1.0)
    return V, F


def _double_torus() -> Tuple[np.ndarray, np.ndarray]:
    """Double torus (genus 2). Load data/meshes/double_torus.obj if present, else single torus."""
    path = _data_dir() / "double_torus.obj"
    if path.exists():
        return _load_obj("double_torus")
    if gpy is None:
        raise ImportError("gpytoolbox required. pip install gpytoolbox")
    # Fallback: single torus (provide double_torus.obj for experiments)
    V, F = gpy.torus(80, 80, R=2.0, r=0.6)
    return V, F


def _tetrahedron() -> Tuple[np.ndarray, np.ndarray]:
    """Tetrahedron (Platonic). Scaled to inscribe in unit sphere."""
    if trimesh is None:
        raise ImportError("trimesh required for tetrahedron. pip install trimesh")
    # Tetrahedron vertices (on unit sphere)
    t = 1.0 / np.sqrt(3)
    V = np.array([[t, t, t], [t, -t, -t], [-t, t, -t], [-t, -t, t]], dtype=np.float64)
    F = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)
    m = trimesh.Trimesh(vertices=V, faces=F)
    while m.vertices.shape[0] < TARGET_VERTICES * (1 - VERTEX_TOLERANCE):
        m = m.subdivide()
        if m.vertices.shape[0] >= TARGET_VERTICES * (1 + VERTEX_TOLERANCE):
            break
    return m.vertices, m.faces


def _octahedron() -> Tuple[np.ndarray, np.ndarray]:
    """Octahedron (Platonic). Inscribed in unit sphere."""
    if trimesh is None:
        raise ImportError("trimesh required for octahedron. pip install trimesh")
    V = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    V = V / np.linalg.norm(V[0])
    F = np.array(
        [
            [4, 2, 0], [4, 0, 3], [4, 3, 1], [4, 1, 2],
            [5, 0, 2], [5, 3, 0], [5, 1, 3], [5, 2, 1],
        ],
        dtype=np.int32,
    )
    m = trimesh.Trimesh(vertices=V, faces=F)
    while m.vertices.shape[0] < TARGET_VERTICES * (1 - VERTEX_TOLERANCE):
        m = m.subdivide()
        if m.vertices.shape[0] >= TARGET_VERTICES * (1 + VERTEX_TOLERANCE):
            break
    return m.vertices, m.faces


def _icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    """Icosahedron (Platonic). trimesh icosphere subdivisions=0 -> 12 vertices, 20 faces."""
    if trimesh is None:
        raise ImportError("trimesh required for icosahedron. pip install trimesh")
    m = trimesh.creation.icosphere(subdivisions=0, radius=1.0)
    while m.vertices.shape[0] < TARGET_VERTICES * (1 - VERTEX_TOLERANCE):
        m = m.subdivide()
        if m.vertices.shape[0] >= TARGET_VERTICES * (1 + VERTEX_TOLERANCE):
            break
    return m.vertices, m.faces


def _flat_plate() -> Tuple[np.ndarray, np.ndarray]:
    """Flat plate (2D reference). Square triangle grid, z=0."""
    n = 100
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y)
    V = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n * n)])
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b, c, d = a + 1, a + n, a + n + 1
            F.append([a, c, b])
            F.append([b, c, d])
    F = np.array(F, dtype=np.int32)
    return V, F


def _load_obj(short_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load data/meshes/{short_name}.obj."""
    if trimesh is None:
        raise ImportError("trimesh required for OBJ load. pip install trimesh")
    path = _data_dir() / f"{short_name}.obj"
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    m = trimesh.load(path, force="mesh")
    if not hasattr(m, "vertices"):
        raise ValueError(f"Loaded {path} is not a single mesh.")
    return m.vertices, m.faces


def _remesh_to_target(V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remesh to near TARGET_VERTICES. Decimate or subdivide."""
    n = V.shape[0]
    if n >= TARGET_VERTICES * (1 - VERTEX_TOLERANCE) and n <= TARGET_VERTICES * (1 + VERTEX_TOLERANCE):
        return V, F
    if trimesh is None:
        return V, F
    m = trimesh.Trimesh(vertices=V, faces=F)
    if n > TARGET_VERTICES * (1 + VERTEX_TOLERANCE):
        # Reduce: quadric decimation
        try:
            target_faces = int(TARGET_VERTICES * 2)  # roughly face ≈ 2*vertices
            m = m.simplify_quadric_decimation(target_faces)
        except Exception:
            pass
    else:
        # Increase: subdivide
        while m.vertices.shape[0] < TARGET_VERTICES * (1 - VERTEX_TOLERANCE):
            m = m.subdivide()
            if m.vertices.shape[0] >= 2 * TARGET_VERTICES:
                break
    return m.vertices, m.faces


def list_mesh_names() -> list[str]:
    """List of supported mesh names. OBJ required: bunny, spot, dragon, armadillo."""
    return [
        "sphere",
        "ellipsoid",
        "cube",
        "torus",
        "double_torus",
        "flat_plate",
        "tetrahedron",
        "octahedron",
        "icosahedron",
        "bunny",
        "spot",
        "dragon",
        "armadillo",
    ]
