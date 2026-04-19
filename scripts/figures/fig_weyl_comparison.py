"""
Generate Weyl asymptotics comparison: N(λ) vs theoretical (area/4π)*λ.
Educational figure for Section 3.3 (Weyl's law).
Run from project root: python figures/fig_weyl_comparison.py
Output: figures/fig_weyl_comparison.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

EIGEN_DIR = root / "data" / "eigen"
MESHES = ["sphere", "torus", "cube", "ellipsoid", "double_torus", "flat_plate"]


def load_eigen_and_area(mesh_name: str):
    """Return (eigenvalues array, surface area) or (None, None)."""
    npz_path = EIGEN_DIR / f"{mesh_name}.npz"
    json_path = EIGEN_DIR / f"{mesh_name}.json"
    if npz_path.exists():
        d = np.load(npz_path, allow_pickle=True)
        V = np.asarray(d["vertices"])
        F = np.asarray(d["faces"])
        ev = np.sort(np.asarray(d["eigenvalues"]).ravel())
        area = _mesh_area(V, F)
        return ev, area
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        V = np.array(data["vertices"])
        F = np.array(data["faces"])
        ev = np.sort(np.array(data.get("eigenvalues", [])))
        area = _mesh_area(V, F)
        return ev, area
    return None, None


def _mesh_area(V: np.ndarray, F: np.ndarray) -> float:
    """Total surface area of mesh."""
    out = 0.0
    for (i, j, k) in F:
        a, b, c = V[i], V[j], V[k]
        cross = np.cross(b - a, c - a)
        out += 0.5 * np.sqrt(np.dot(cross, cross))
    return float(out)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    out_path = root / "figures" / "fig_weyl_comparison.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ref_lam = None
    ref_area = None
    for mesh_name in MESHES:
        ev, area = load_eigen_and_area(mesh_name)
        if ev is None or len(ev) < 5 or area is None or area <= 0:
            continue
        lam = np.asarray(ev).ravel()
        k_range = np.arange(1, len(lam) + 1, dtype=float)
        ax.plot(lam, k_range, label=mesh_name.replace("_", " "), alpha=0.8)
        if ref_lam is None:
            ref_lam, ref_area = lam, area
    if ref_lam is not None and ref_area is not None:
        weyl_slope = ref_area / (4 * np.pi)
        lam_line = np.linspace(ref_lam[0], ref_lam[-1], 50)
        ax.plot(lam_line, weyl_slope * lam_line, "k--", alpha=0.7, linewidth=1.5, label="Weyl $|M|/(4\\pi)\\cdot\\lambda$")
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("$N(\\lambda)$ (counting function)")
    ax.set_title("Weyl asymptotics: $N(\\lambda) \\sim \\frac{|M|}{4\\pi}\\lambda$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", out_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
