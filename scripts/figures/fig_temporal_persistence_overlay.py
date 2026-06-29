"""
Temporal β₀ visualization — R3 overlay version.

Two panels (A3, A7); each panel overlays sphere (blue) and torus (orange)
with their per-frame β₀ time-series and dashed mean lines.

Run from project root: python scripts/figures/fig_temporal_persistence_overlay.py
Requires: data/results/results_temporal.csv
Output  : figures/fig_temporal_persistence.pdf (overwrites existing)
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent.parent
CSV_PATH = root / "data" / "results" / "results_temporal.csv"
OUT_PATH = root / "figures" / "fig_temporal_persistence.pdf"

SPHERE_COLOR = "tab:blue"
TORUS_COLOR = "tab:orange"
MEAN_COLOR = "gray"

STIMULI = ["A3", "A7"]
MESHES = [
    ("sphere_genus0", SPHERE_COLOR, "sphere"),
    ("torus_genus1", TORUS_COLOR, "torus"),
]


def main() -> int:
    if not CSV_PATH.exists():
        print(f"Missing data file: {CSV_PATH}", file=sys.stderr)
        return 1

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV is empty", file=sys.stderr)
        return 1

    groups: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        key = (r["audio_id"], r["mesh_id"])
        groups[key].append((float(r["t_sec"]), int(r["beta0"])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    sigma_summary: dict[tuple[str, str], float] = {}

    for ax, audio_id in zip(axes, STIMULI):
        legend_handles = []
        for mesh_id, color, label in MESHES:
            series = sorted(groups.get((audio_id, mesh_id), []))
            if not series:
                continue
            t = np.array([p[0] for p in series])
            b0 = np.array([p[1] for p in series], dtype=float)
            sigma = float(b0.std())
            sigma_summary[(audio_id, mesh_id)] = sigma
            mean_b0 = float(b0.mean())
            line, = ax.plot(t, b0, "-", color=color, linewidth=1.15,
                            label=f"{label} ($\\bar\\beta_0={mean_b0:.1f}$, $\\sigma={sigma:.1f}$)")
            ax.axhline(mean_b0, color=color, linestyle="--",
                       alpha=0.65, linewidth=1.0)
            legend_handles.append(line)
            if mesh_id == MESHES[-1][0]:
                # Add a dummy entry annotating the dashed convention
                from matplotlib.lines import Line2D
                dashed_proxy = Line2D([0], [0], color="black",
                                      linestyle="--", alpha=0.7,
                                      label=r"per-mesh mean $\bar\beta_0$")
                legend_handles.append(dashed_proxy)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\beta_0$")
        ax.set_title(f"Stimulus {audio_id}", fontsize=12, fontweight="bold")
        ax.set_facecolor("white")
        ax.legend(handles=legend_handles, loc="best", fontsize=9,
                  frameon=True, framealpha=0.93)

    fig.suptitle(
        r"Frame-wise $\beta_0$ on sphere (blue) and torus (orange); "
        r"per-mesh mean shown as dashed line",
        y=1.02, fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")
    for k, v in sorted(sigma_summary.items()):
        print(f"  {k[0]}/{k[1]}: sigma(beta_0)={v:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
