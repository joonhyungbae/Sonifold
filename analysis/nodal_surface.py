"""
Nodal surface extraction and topology metrics: β₀, β₁, χ, A_ratio.
Nodal: vertices with |f(v)| < ε. ε = 0.01 * max|f| (control).
"""
from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np


class TopologyMetrics(NamedTuple):
    beta0: int   # number of connected components
    beta1: int   # first Betti number (approx: graph-based)
    chi: int     # Euler char V-E+F (V-E for graph)
    A_ratio: float  # nodal surface area ratio


def extract_nodal_and_metrics(vertices, faces, scalar_values, threshold_ratio=0.01):
    """
    vertices (V,3), faces (F,3), scalar_values (V,).
    threshold_ratio: ε = threshold_ratio * max|f|.
    Returns: (nodal_vertex_set, TopologyMetrics).
    """
    f = np.asarray(scalar_values, dtype=np.float64).ravel()
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int32)
    n_verts = V.shape[0]
    if f.shape[0] != n_verts:
        raise ValueError("scalar_values length != vertices count")
    max_abs = np.max(np.abs(f)) + 1e-20
    eps = threshold_ratio * max_abs
    nodal = set(np.where(np.abs(f) < eps)[0])
    if not nodal:
        return nodal, TopologyMetrics(beta0=0, beta1=0, chi=0, A_ratio=0.0)
    edge_set = _mesh_edges(F)
    nodal_edges = [(i, j) for i, j in edge_set if i in nodal and j in nodal]
    adj = _adjacency_from_edges(nodal_edges)
    beta0 = _count_components(adj, nodal)
    V_nodal, E_nodal = len(nodal), len(nodal_edges)
    chi = V_nodal - E_nodal
    beta1 = max(0, E_nodal - V_nodal + beta0)
    total_area = _mesh_total_area(V, F)
    nodal_area = _nodal_face_area(V, F, nodal)
    A_ratio = (nodal_area / total_area) if total_area > 1e-20 else 0.0
    return nodal, TopologyMetrics(beta0=beta0, beta1=beta1, chi=chi, A_ratio=A_ratio)


def compute_topology_metrics(vertices, faces, scalar_values, threshold_ratio=0.01):
    """Return only nodal topology metrics."""
    _, m = extract_nodal_and_metrics(vertices, faces, scalar_values, threshold_ratio)
    return m


def _mesh_edges(F):
    out = set()
    for a, b, c in F:
        out.add((min(a, b), max(a, b)))
        out.add((min(b, c), max(b, c)))
        out.add((min(c, a), max(c, a)))
    return out


def _adjacency_from_edges(edge_list):
    adj = {}
    for i, j in edge_list:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    return adj


def _count_components(adj, vertices):
    visited = set()
    count = 0
    for v in vertices:
        if v in visited:
            continue
        count += 1
        q = deque([v])
        visited.add(v)
        while q:
            u = q.popleft()
            for w in adj.get(u, []):
                if w not in visited:
                    visited.add(w)
                    q.append(w)
    return count


def _mesh_total_area(V, F):
    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]
    cross = np.cross(b - a, c - a)
    return 0.5 * np.sum(np.linalg.norm(cross, axis=1))


def _nodal_face_area(V, F, nodal):
    total = 0.0
    for i, j, k in F:
        if i in nodal or j in nodal or k in nodal:
            a, b, c = V[i], V[j], V[k]
            total += 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    return total
