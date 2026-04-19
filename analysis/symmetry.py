"""
Symmetry index S for the nodal pattern (scalar field on the mesh).

S does *not* use the mesh's symmetry group explicitly. It measures how much
the scalar field f (nodal pattern) is invariant under π-rotation about the
x, y, z axes: we rotate the mesh vertices by π about each axis, interpolate
f at the rotated positions, and take the correlation with the original f.
The average of (1 + correlation)/2 over the three axes is S in [0, 1].
So S is a field-based measure of "three-axis π-symmetry" of the pattern;
shapes with high intrinsic symmetry (e.g. sphere) tend to produce patterns
with higher S when the audio is simple, but S is defined on the field, not
on the shape.
"""
import numpy as np


def compute_symmetry(vertices, scalar_values, n_axes=3):
    """
    Symmetry index S: average over x,y,z of (1 + corr(f, f_π-rot))/2.

    vertices (V,3), scalar_values (V,). For each coordinate axis we apply
    a π-rotation to the vertices, interpolate the field at rotated positions,
    and compute the correlation with the original field. S in [0, 1]; 1 means
    the pattern is invariant under those π-rotations.
    """
    V = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(scalar_values, dtype=np.float64).ravel()
    if V.shape[0] != len(f):
        raise ValueError("vertices and scalar_values length mismatch")
    if np.std(f) < 1e-20:
        return 1.0
    corrs = []
    for axis in range(min(3, n_axes)):
        R = _rotation_matrix(np.pi, axis)
        V_rot = (R @ V.T).T
        f_rot = _interp_at_vertices(V, f, V_rot)
        if np.std(f_rot) < 1e-20:
            corrs.append(1.0)
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(f, f_rot)[0, 1]
        if np.isnan(c):
            c = 0.0
        corrs.append((c + 1) / 2.0)
    return float(np.mean(corrs))


def _rotation_matrix(angle, axis):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _interp_at_vertices(V, f, Q):
    """Interpolate f on V at each point of Q by nearest-neighbor."""
    from scipy.spatial import cKDTree
    tree = cKDTree(V)
    _, idx = tree.query(Q, k=1)
    return f[idx]
