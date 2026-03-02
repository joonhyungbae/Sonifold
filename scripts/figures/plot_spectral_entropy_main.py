"""
Plot fig_spectral_entropy_main.pdf from existing extended_mesh_spectral_entropy.csv
(filter to 13 meshes used in paper). Use when full spectral_entropy_conjecture_test
has not been run.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = root / "figures"
RESULTS_DIR = root / "data" / "results"
CSV_PATH = RESULTS_DIR / "extended_mesh_spectral_entropy.csv"
OUT_FIG = FIGURES_DIR / "fig_spectral_entropy_main.pdf"

# 13 meshes: MESHES_9 + MESHES_GENUS_36 (same order as spectral_entropy_conjecture_test)
MESHES_13 = [
    ("sphere", 0), ("torus", 1), ("cube", 0), ("ellipsoid", 0), ("double_torus", 2),
    ("flat_plate", 0), ("tetrahedron", 0), ("octahedron", 0), ("icosahedron", 0),
    ("triple_torus_genus3", 3), ("quad_torus_genus4", 4),
    ("penta_torus_genus5", 5), ("hex_torus_genus6", 6),
]


def main():
    if not CSV_PATH.exists():
        print(f"Missing {CSV_PATH}; run spectral_entropy_conjecture_test.py or extended mesh script.", file=sys.stderr)
        return 1
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("mesh_name", "")
            if any(name == m[0] for m in MESHES_13):
                try:
                    h = float(row.get("spectral_entropy_H", row.get("H", "")))
                    b = float(row.get("mean_beta0_random", ""))
                    g = next(gen for (m, gen) in MESHES_13 if m == name)
                    rows.append({"mesh": name, "genus": g, "H": h, "beta0_random_mean": b})
                except (ValueError, TypeError):
                    pass
    # Keep order of MESHES_13
    order = {m[0]: i for i, m in enumerate(MESHES_13)}
    rows.sort(key=lambda r: order.get(r["mesh"], 999))
    if len(rows) < 3:
        print("Too few rows for figure.", file=sys.stderr)
        return 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    x = [r["H"] for r in rows]
    y = [r["beta0_random_mean"] for r in rows]
    labels = [r["mesh"].replace("_", " ") for r in rows]
    genera = [r["genus"] for r in rows]
    rho, _ = spearmanr(x, y)
    def genus_color(g):
        if g == 0: return "C0"
        if g == 1: return "C2"
        if g == 2: return "C3"
        return "purple"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rc("font", family="serif")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [genus_color(g) for g in genera]
    ax.scatter(x, y, c=colors, s=80, zorder=2, edgecolors="k", linewidths=0.6)
    for i, lb in enumerate(labels):
        ax.annotate(lb, (x[i], y[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel("Spectral entropy $H$ (normalized gaps)")
    ax.set_ylabel(r"$\beta_0$ (random coefficients, mean)")
    ax.set_title(r"$H$ vs $\beta_0$ (K=50)")
    ax.text(0.05, 0.95, f"Spearman $\\rho$ = {rho:.2f}", transform=ax.transAxes, fontsize=11, va="top")
    plt.tight_layout()
    plt.savefig(OUT_FIG, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_FIG}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
