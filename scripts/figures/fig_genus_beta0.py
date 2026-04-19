"""
Genus vs β₀ visualization (paper Fig. genus).
Data priority: data/results/results_genus_extended.csv → data/results/beta0_stats.csv (A3, sphere/torus/double_torus).

Run from project root: python figures/fig_genus_beta0.py
Output: figures/fig_genus_beta0.pdf
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

CSV_EXTENDED = root / "data" / "results" / "results_genus_extended.csv"
CSV_FALLBACK = root / "data" / "results" / "beta0_stats.csv"
OUT_PATH = root / "figures" / "fig_genus_beta0.pdf"

# Five-point genus sequence: mesh name -> genus (data/results/results_genus_extended.csv or experiments)
GENUS_MESHES = [
    ("sphere", 0),
    ("torus", 1),
    ("double_torus", 2),
]
GENUS_LABELS = {0: "sphere", 1: "torus", 2: "double torus", 3: "genus 3", 4: "genus 4"}


def _load_from_extended():
    if not CSV_EXTENDED.exists():
        return None
    with open(CSV_EXTENDED, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "genus" not in rows[0] or "beta0" not in rows[0]:
        return None
    genus = [int(float(r["genus"])) for r in rows]
    beta0 = [float(r["beta0"]) for r in rows]
    return sorted(zip(genus, beta0))


def _load_from_beta0_stats():
    if not CSV_FALLBACK.exists():
        return None
    with open(CSV_FALLBACK, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    points = []
    for mesh, g in GENUS_MESHES:
        for r in rows:
            if r.get("mesh") == mesh and r.get("audio") == "A3":
                mean = r.get("mean")
                if mean is not None and mean != "":
                    points.append((g, float(mean)))
                break
    return sorted(points) if points else None


def main():
    pairs = _load_from_extended() or _load_from_beta0_stats()
    if not pairs:
        print(
            "No data: create data/results/results_genus_extended.csv or run scripts/analysis/compute_beta0_stats.py -> data/results/beta0_stats.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    genus = [p[0] for p in pairs]
    beta0 = [p[1] for p in pairs]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.scatter(genus, beta0, s=80, color="C0", zorder=3)
    ax.plot(genus, beta0, "o-", color="C0", linewidth=1.5, markersize=0)
    for g, b in zip(genus, beta0):
        label = GENUS_LABELS.get(g, f"g={g}")
        ax.annotate(label, (g, b), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Genus")
    ax.set_ylabel(r"$\beta_0$ (mean, A3 direct mapping)")
    ax.set_xticks(genus)
    if len(genus) >= 2:
        z = np.polyfit(genus, beta0, 1)
        xl = np.array([min(genus), max(genus)])
        ax.plot(xl, np.polyval(z, xl), "--", color="gray", alpha=0.8, label="linear fit")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", OUT_PATH, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
