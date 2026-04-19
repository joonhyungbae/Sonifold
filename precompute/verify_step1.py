"""
Step 1 verification: sphere eigenvalues l(l+1), torus eigenvalue pattern.
Run: python -m precompute.verify_step1   (full meshes, slow)
     python -m precompute.verify_step1 --quick  (small meshes, fast)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from precompute.eigensolver import compute_eigen
from precompute.mesh_library import get_mesh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Use small meshes for fast check")
    args = ap.parse_args()
    if args.quick:
        import trimesh
        import gpytoolbox as gpy
        print("(quick mode: small meshes)")
        # Small sphere
        m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        V_s, F_s = m.vertices, m.faces
        V_s, F_s = V_s.astype(np.float64), F_s.astype(np.int32)
    else:
        V_s, F_s = get_mesh("sphere")
    print("=== Sphere: eigenvalues ~ l(l+1) (λ₁≈2, λ₂≈2, λ₃≈2, λ₄≈6, λ₅≈6, ...) ===")
    evals, _ = compute_eigen(V_s, F_s, N=20)
    for i in range(min(10, len(evals))):
        l = i + 1
        theory = l * (l + 1)
        print(f"  k={i+1}  λ={evals[i]:.4f}  l(l+1)={theory}  (l={l})")
    # Sphere is degenerate: λ₁=λ₂=λ₃=2 (l=1), λ₄=λ₅=λ₆=6 (l=2) ...
    ok = abs(evals[0] - 2) < 0.5 and abs(evals[3] - 6) < 1.0
    print("  → PASS" if ok else "  → CHECK (mesh or solver)")

    print("\n=== Torus (R:r=3:1): eigenvalues ~ (2πm/R)²+(2πn/r)² ===")
    if args.quick:
        V_t, F_t = gpy.torus(15, 15, R=3.0, r=1.0)
    else:
        V_t, F_t = get_mesh("torus")
    evals, _ = compute_eigen(V_t, F_t, N=15)
    for i in range(min(8, len(evals))):
        print(f"  k={i+1}  λ={evals[i]:.4f}")
    print("  → First eigenvalues should increase; exact formula depends on (m,n).")


if __name__ == "__main__":
    main()
