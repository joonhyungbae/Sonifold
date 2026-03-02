"""
Temporal β₀ visualization: frame-wise β₀ and σ(β₀) summary over ~10s.
Run from project root: python figures/fig_temporal_persistence.py
Requires: data/results/results_temporal.csv (after experiment.run_temporal_persistence)
Output: figures/fig_temporal_persistence.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

CSV_PATH = root / "data" / "results" / "results_temporal.csv"
OUT_PATH = root / "figures" / "fig_temporal_persistence.pdf"


def main():
    if not CSV_PATH.exists():
        print("Run experiment.run_temporal_persistence first to create", CSV_PATH, file=sys.stderr)
        sys.exit(1)

    import csv
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(1)

    # Group by (audio_id, mesh_id)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        k = (r["audio_id"], r["mesh_id"])
        groups[k].append((float(r["t_sec"]), int(r["beta0"])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    n_plots = len(groups)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.0 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    mean_handle = None
    legend_ax = None
    for idx, ((audio_id, mesh_id), points) in enumerate(sorted(groups.items())):
        ax = axes[idx]
        t = [p[0] for p in points]
        b0 = [p[1] for p in points]
        b0_arr = np.array(b0)
        ax.plot(t, b0, "-", color="C0", linewidth=1)
        is_torus_genus1 = audio_id == "A3" and ("torus_genus1" in mesh_id or mesh_id == "torus")
        if is_torus_genus1:
            mean_handle = ax.axhline(
                b0_arr.mean(), color="gray", linestyle="--", alpha=0.8, label=r"mean $\beta_0$"
            )
            legend_ax = ax
        else:
            ax.axhline(b0_arr.mean(), color="gray", linestyle="--", alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\beta_0$")
        ax.set_title("{} / {}  ($\\sigma(\\beta_0)$ = {:.1f})".format(audio_id, mesh_id, b0_arr.std()))
        ax.set_facecolor("white")
    for j in range(len(groups), len(axes)):
        axes[j].set_visible(False)
    if mean_handle is not None and legend_ax is not None:
        legend_ax.legend(handles=[mean_handle], loc="upper right", fontsize=10, frameon=True, framealpha=0.95)
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", OUT_PATH, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
