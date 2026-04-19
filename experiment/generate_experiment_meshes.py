"""
Generate meshes for Progressive Genus and Symmetry Breaking experiments.
- Genus 0 (Sphere), 1 (Torus), 2 (Double Torus), 3, 4 remeshed to ~5,000 vertices.
- Genus 3/4: load data/meshes/genus3.obj, genus4.obj if present; else overlapping tori boolean union.
- 6 meshes: regular Octahedron with z-axis stretched by 1.0, 1.1, 1.2, 1.3, 1.4, 1.5.

Run from project root: python -m experiment.generate_experiment_meshes
Output: data/experiments/*.obj
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

# Target vertex count (experiment control)
TARGET_VERTICES = 5000
VERTEX_TOLERANCE = 0.2
EXPERIMENTS_DIR = root / "data" / "experiments"


def _remesh_to_target(V, F, target):
    """Remesh to approximately target vertex count."""
    try:
        import trimesh
    except ImportError:
        return V, F
    n = V.shape[0]
    if target * (1 - VERTEX_TOLERANCE) <= n <= target * (1 + VERTEX_TOLERANCE):
        return V, F
    m = trimesh.Trimesh(vertices=V, faces=F)
    if n > target * (1 + VERTEX_TOLERANCE):
        try:
            target_faces = int(target * 2)
            m = m.simplify_quadric_decimation(target_faces)
        except Exception:
            pass
    else:
        while m.vertices.shape[0] < target * (1 - VERTEX_TOLERANCE):
            m = m.subdivide()
            if m.vertices.shape[0] >= 2 * target:
                break
    return m.vertices.astype(np.float64), m.faces.astype(np.int32)


def _sphere_5k():
    """Sphere, ~5k vertices."""
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    return _remesh_to_target(m.vertices, m.faces, TARGET_VERTICES)


def _torus_5k():
    """Torus (Genus 1), ~5k vertices."""
    try:
        import gpytoolbox as gpy
    except ImportError:
        raise ImportError("gpytoolbox required: pip install gpytoolbox")
    nR, nr = 71, 71
    V, F = gpy.torus(nR, nr, R=3.0, r=1.0)
    return _remesh_to_target(V, F, TARGET_VERTICES)


def _double_torus_5k():
    """Double torus (Genus 2), ~5k vertices. Load data/meshes/double_torus.obj if present."""
    meshes_dir = root / "data" / "meshes"
    obj_path = meshes_dir / "double_torus.obj"
    if obj_path.exists():
        import trimesh
        m = trimesh.load(obj_path, force="mesh")
        V, F = m.vertices, m.faces
    else:
        try:
            import gpytoolbox as gpy
            import logging
            logging.warning(
                "data/meshes/double_torus.obj not found; using single torus as proxy for genus-2 experiment."
            )
            V, F = gpy.torus(71, 71, R=2.0, r=0.6)
        except ImportError:
            raise FileNotFoundError(
                "Double torus requires either data/meshes/double_torus.obj or gpytoolbox."
            )
    return _remesh_to_target(V, F, TARGET_VERTICES)


def _high_genus_by_tori(num_tori, R=2.0, r=0.55, n_ring=50):
    """Create a closed surface of genus num_tori by boolean union of overlapping tori along x-axis."""
    try:
        import gpytoolbox as gpy
        import trimesh
    except ImportError as e:
        raise ImportError("gpytoolbox and trimesh required for high-genus mesh") from e
    # Place tori so they overlap (centers spaced by < 2*R)
    spacing = max(1.0, 2 * R - 1.2 * r)
    meshes = []
    for i in range(num_tori):
        V, F = gpy.torus(n_ring, n_ring, R=R, r=r)
        V = V + np.array([i * spacing, 0.0, 0.0], dtype=np.float64)
        meshes.append(trimesh.Trimesh(vertices=V, faces=F))
    try:
        merged = trimesh.boolean.union(meshes, engine="blender")
    except Exception:
        try:
            merged = trimesh.boolean.union(meshes, engine="manifold")
        except Exception as e:
            raise RuntimeError(
                "Boolean union of tori failed. Install manifold3d (pip install manifold3d) "
                "or provide data/meshes/genus3.obj and genus4.obj."
            ) from e
    if merged is None:
        raise RuntimeError("Boolean union returned None")
    if hasattr(merged, "process"):
        merged.process(validate=True)
    V = np.asarray(merged.vertices, dtype=np.float64)
    F = np.asarray(merged.faces, dtype=np.int32)
    return V, F


def _genus3_5k():
    """Genus 3 closed surface, ~5k vertices. Prefer data/meshes/genus3.obj."""
    meshes_dir = root / "data" / "meshes"
    obj_path = meshes_dir / "genus3.obj"
    if obj_path.exists():
        import trimesh
        m = trimesh.load(obj_path, force="mesh")
        return _remesh_to_target(m.vertices, m.faces, TARGET_VERTICES)
    V, F = _high_genus_by_tori(3, R=2.0, r=0.55, n_ring=50)
    return _remesh_to_target(V, F, TARGET_VERTICES)


def _genus4_5k():
    """Genus 4 closed surface, ~5k vertices. Prefer data/meshes/genus4.obj."""
    meshes_dir = root / "data" / "meshes"
    obj_path = meshes_dir / "genus4.obj"
    if obj_path.exists():
        import trimesh
        m = trimesh.load(obj_path, force="mesh")
        return _remesh_to_target(m.vertices, m.faces, TARGET_VERTICES)
    V, F = _high_genus_by_tori(4, R=2.0, r=0.5, n_ring=45)
    return _remesh_to_target(V, F, TARGET_VERTICES)


def _genus5_5k():
    """Genus 5 closed surface, ~5k vertices. Prefer data/meshes/genus5.obj."""
    meshes_dir = root / "data" / "meshes"
    obj_path = meshes_dir / "genus5.obj"
    if obj_path.exists():
        import trimesh
        m = trimesh.load(obj_path, force="mesh")
        return _remesh_to_target(m.vertices, m.faces, TARGET_VERTICES)
    V, F = _high_genus_by_tori(5, R=2.0, r=0.48, n_ring=42)
    return _remesh_to_target(V, F, TARGET_VERTICES)


def _genus6_5k():
    """Genus 6 closed surface, ~5k vertices. Prefer data/meshes/genus6.obj."""
    meshes_dir = root / "data" / "meshes"
    obj_path = meshes_dir / "genus6.obj"
    if obj_path.exists():
        import trimesh
        m = trimesh.load(obj_path, force="mesh")
        return _remesh_to_target(m.vertices, m.faces, TARGET_VERTICES)
    V, F = _high_genus_by_tori(6, R=2.0, r=0.45, n_ring=40)
    return _remesh_to_target(V, F, TARGET_VERTICES)


def _octahedron_base_5k():
    """Regular Octahedron, ~5k vertices."""
    import trimesh
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
    return _remesh_to_target(m.vertices, m.faces, TARGET_VERTICES)


def _stretched_octahedron(V, F, stretch_z):
    """Scale vertices along z-axis (symmetry breaking)."""
    V2 = V.copy()
    V2[:, 2] *= stretch_z
    return V2, F.copy()


def save_obj(V, F, path):
    """Save mesh as OBJ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import trimesh
        m = trimesh.Trimesh(vertices=V, faces=F)
        m.export(str(path))
    except Exception:
        with open(path, "w", encoding="utf-8") as f:
            for i in range(V.shape[0]):
                f.write("v {} {} {}\n".format(V[i, 0], V[i, 1], V[i, 2]))
            for i in range(F.shape[0]):
                f.write("f {} {} {}\n".format(F[i, 0] + 1, F[i, 1] + 1, F[i, 2] + 1))


def main():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    genus_meshes = [
        ("sphere_genus0", 0, _sphere_5k),
        ("torus_genus1", 1, _torus_5k),
        ("double_torus_genus2", 2, _double_torus_5k),
        ("triple_torus_genus3", 3, _genus3_5k),
        ("quad_torus_genus4", 4, _genus4_5k),
        ("penta_torus_genus5", 5, _genus5_5k),
        ("hex_torus_genus6", 6, _genus6_5k),
    ]
    for name, genus, fn in genus_meshes:
        try:
            V, F = fn()
            path = EXPERIMENTS_DIR / (name + ".obj")
            save_obj(V, F, path)
            print("[genus] {} genus={} n_verts={} -> {}".format(name, genus, V.shape[0], path))
        except Exception as e:
            print("[genus] {} failed: {}".format(name, e), file=sys.stderr)

    try:
        V_oct, F_oct = _octahedron_base_5k()
    except Exception as e:
        print("[symmetry] Octahedron base failed: {}".format(e), file=sys.stderr)
        return
    for stretch in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        V_s, F_s = _stretched_octahedron(V_oct, F_oct, stretch)
        name = "octahedron_sym_{:.1f}".format(stretch)
        path = EXPERIMENTS_DIR / (name + ".obj")
        save_obj(V_s, F_s, path)
        print("[symmetry] {} stretch_z={} n_verts={} -> {}".format(name, stretch, V_s.shape[0], path))

    print("Done. Meshes in", EXPERIMENTS_DIR, file=sys.stderr)


if __name__ == "__main__":
    main()
