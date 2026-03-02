"""
Genus vs β₀ visualization (7-point: genus 0–6).
Data: data/results/results_genus_extended_7point.csv.

Run from project root: python figures/fig_genus_beta0_7point.py
Output: figures/fig_genus_beta0_7point.pdf
Style: same as fig_genus_beta0.pdf.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

CSV_7POINT = root / "data" / "results" / "results_genus_extended_7point.csv"
OUT_PATH = root / "figures" / "fig_genus_beta0_7point.pdf"

GENUS_LABELS = {
    0: "sphere",
    1: "torus",
    2: "double torus",
    3: "genus 3",
    4: "genus 4",
    5: "genus 5",
    6: "genus 6",
}


def _load_7point():
    if not CSV_7POINT.exists():
        return None
    with open(CSV_7POINT, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "genus" not in rows[0] or "beta0" not in rows[0]:
        return None
    genus = [int(float(r["genus"])) for r in rows]
    beta0 = [float(r["beta0"]) for r in rows]
    return sorted(zip(genus, beta0))


def main():
    pairs = _load_7point()
    if not pairs:
        print(
            "No data: run scripts/generate_genus_extended.py to create "
            "data/results/results_genus_extended_7point.csv",
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
