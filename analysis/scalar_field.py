"""
Scalar field synthesis: f(v) = Σ a_k · φ_k(v) = coefficients @ eigenvectors.
eigenvectors (N, V), coefficients (N,) -> scalar_values (V,).
"""
import numpy as np


def compute_scalar_field(eigenvectors, coefficients):
    """
    eigenvectors: (N, V), coefficients: (N,).
    Returns: (V,) float.
    """
    e = np.asarray(eigenvectors, dtype=np.float64)
    c = np.asarray(coefficients, dtype=np.float64).ravel()
    if e.shape[0] != len(c):
        raise ValueError("eigenvectors.shape[0] != len(coefficients)")
    return (c @ e).astype(np.float64)
