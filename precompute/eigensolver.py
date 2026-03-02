"""
LB (Laplace–Beltrami) eigenvalue/eigenfunction computation.
Input: (vertices, faces), number of eigenfunctions N
Output: eigenvalues (N,), eigenvectors (N, V) — each row is one eigenfunction

Standard form A = M^{-1/2} L M^{-1/2} then eigsh(which='SA').
Default: GPU if CuPy available, else CPU. USE_GPU=0 forces CPU. USE_GPU=1 requires CuPy (RuntimeError if missing).
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh as scipy_eigsh

try:
    import gpytoolbox as gpy
except ImportError:
    gpy = None

_CUPY_AVAILABLE = False
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    import cupyx.scipy.sparse.linalg as cusplinalg
    _CUPY_AVAILABLE = True
except ImportError:
    cp = cusp = cusplinalg = None

def _eigsh_tol():
    """Tolerance for eigsh; looser = faster, less accurate. Env EIGEN_TOL overrides (e.g. 1e-3)."""
    v = os.environ.get("EIGEN_TOL")
    if v is not None:
        try:
            return float(v)
        except ValueError:
            pass
    return 1e-4


def _solve_cpu(A: csr_matrix, k: int):
    tol = _eigsh_tol()
    # maxiter: avoid runaway; larger matrices may need more (default in ARPACK is n*20)
    maxiter = min(A.shape[0] * 10, 100000)
    w, v = scipy_eigsh(A, k=k, which="SA", tol=tol, maxiter=maxiter, return_eigenvectors=True)
    return np.asarray(w, dtype=np.float64), np.asarray(v, dtype=np.float64)


def _solve_gpu(A: csr_matrix, k: int):
    tol = _eigsh_tol()
    A_gpu = cusp.csr_matrix(A)
    w_gpu, v_gpu = cusplinalg.eigsh(A_gpu, k=k, which="SA", tol=tol, return_eigenvectors=True)
    return np.asarray(cp.asnumpy(w_gpu), dtype=np.float64), np.asarray(cp.asnumpy(v_gpu), dtype=np.float64)


def compute_eigen(V: np.ndarray, F: np.ndarray, N: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve generalized eigenvalue problem L·φ = λ·M·φ with cotangent Laplacian L and mass matrix M.
    Return N eigenpairs excluding constant (λ=0). Uses GPU if CuPy available, else CPU (scipy).
    """
    if gpy is None:
        raise ImportError("gpytoolbox required. pip install gpytoolbox")

    n_verts = V.shape[0]
    F = np.asarray(F, dtype=np.int32)
    if F.max() >= n_verts or F.min() < 0:
        raise ValueError("Face indices out of range for vertex count.")

    l_sq = gpy.halfedge_lengths_squared(V, F)
    L = gpy.cotangent_laplacian_intrinsic(l_sq, F, n=n_verts)
    M = gpy.massmatrix_intrinsic(l_sq, F, n=n_verts, type="voronoi")

    if not isinstance(L, csr_matrix):
        L = csr_matrix(L)
    if not isinstance(M, csr_matrix):
        M = csr_matrix(M)

    k = min(N + 1, n_verts - 1)
    d = np.array(M.diagonal()).flatten()
    if not (np.all(d > 1e-20) and M.nnz <= n_verts * 2):
        raise NotImplementedError("Eigensolver requires diagonal mass matrix (Voronoi).")

    dinvsqrt = 1.0 / np.sqrt(np.maximum(d, 1e-20))
    Dinv = diags(dinvsqrt, format="csr")
    A = Dinv @ L @ Dinv

    use_gpu_env = os.environ.get("USE_GPU", "1").strip().lower()
    if use_gpu_env == "1" and not _CUPY_AVAILABLE:
        raise RuntimeError(
            "USE_GPU=1 but CuPy is not available. "
            "Install cupy-cuda12x (CUDA 12) or cupy-cuda11x (CUDA 11) and ensure nvidia-smi shows a GPU. "
            "To use CPU only, set USE_GPU=0."
        )
    use_gpu = use_gpu_env != "0" and _CUPY_AVAILABLE
    tol = _eigsh_tol()
    if use_gpu:
        eigenvalues, w = _solve_gpu(A, k)
        print("[eigensolver] GPU (CuPy), tol={}".format(tol), file=sys.stderr)
    else:
        eigenvalues, w = _solve_cpu(A, k)
        print("[eigensolver] CPU (scipy), tol={}".format(tol), file=sys.stderr)

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    w = w[:, order]
    if eigenvalues.shape[0] > N:
        eigenvalues = eigenvalues[1 : N + 1]
        w = w[:, 1 : N + 1]
    eigenvectors = (dinvsqrt[:, np.newaxis] * w).T  # (N, V)

    for i in range(eigenvectors.shape[0]):
        u = eigenvectors[i, :].copy()
        mass = np.dot(u, d * u)
        if mass > 1e-20:
            eigenvectors[i, :] = u / np.sqrt(mass)

    return eigenvalues, eigenvectors.astype(np.float64)
