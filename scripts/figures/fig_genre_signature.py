"""
Genre-wise Spectral Signature heatmap: mesh × audio_id → entropy (and β₀).
Run from project root: python figures/fig_genre_signature.py
Requires: data/results/results_genre_signature.csv (after experiment.run_genre_signature)
Output: figures/fig_genre_signature.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

CSV_PATH = root / "data" / "results" / "results_genre_signature.csv"
OUT_PATH = root / "figures" / "fig_genre_signature.pdf"


def main():
    if not CSV_PATH.exists():
        print("Run experiment.run_genre_signature first to create", CSV_PATH, file=sys.stderr)
        sys.exit(1)

    import csv
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(1)

    mesh_ids = sorted(set(r["mesh_id"] for r in rows))
    audio_ids = sorted(set(r["audio_id"] for r in rows))
    by_key = {(r["mesh_id"], r["audio_id"]): r for r in rows}
    pivot_ent = np.zeros((len(mesh_ids), len(audio_ids)))
    pivot_b0 = np.zeros((len(mesh_ids), len(audio_ids)), dtype=int)
    for i, mid in enumerate(mesh_ids):
        for j, aid in enumerate(audio_ids):
            r = by_key.get((mid, aid), {})
            pivot_ent[i, j] = float(r.get("entropy", 0))
            pivot_b0[i, j] = int(r.get("beta0", 0))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(pivot_ent, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(audio_ids)))
    axes[0].set_xticklabels(audio_ids)
    axes[0].set_yticks(range(len(mesh_ids)))
    axes[0].set_yticklabels(mesh_ids)
    axes[0].set_title("Coefficient entropy (genre–mesh)")
    for i in range(len(mesh_ids)):
        for j in range(len(audio_ids)):
            axes[0].text(j, i, "{:.2f}".format(pivot_ent[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im0, ax=axes[0], label="Entropy")
    im1 = axes[1].imshow(pivot_b0, cmap="plasma", aspect="auto")
    axes[1].set_xticks(range(len(audio_ids)))
    axes[1].set_xticklabels(audio_ids)
    axes[1].set_yticks(range(len(mesh_ids)))
    axes[1].set_yticklabels(mesh_ids)
    axes[1].set_title(r"$\beta_0$ (genre–mesh)")
    for i in range(len(mesh_ids)):
        for j in range(len(audio_ids)):
            axes[1].text(j, i, str(pivot_b0[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im1, ax=axes[1], label=r"$\beta_0$")
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", OUT_PATH, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
