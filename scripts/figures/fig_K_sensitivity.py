"""
K (number of eigenfunctions) sensitivity visualization: K vs β₀, A_ratio, compute time.
Run from project root: python figures/fig_K_sensitivity.py
Requires: data/results/results_K_sensitivity.csv (after experiment.run_K_sensitivity)
Output: figures/fig_K_sensitivity.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

CSV_PATH = root / "data" / "results" / "results_K_sensitivity.csv"
OUT_PATH = root / "figures" / "fig_K_sensitivity.pdf"


def main():
    if not CSV_PATH.exists():
        print("Run experiment.run_K_sensitivity first to create", CSV_PATH, file=sys.stderr)
        sys.exit(1)

    import csv
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(1)
    df = {k: [r[k] for r in rows] for k in rows[0]}
    K = [int(x) for x in df["K"]]
    beta0 = [int(x) for x in df["beta0"]]
    A_ratio = [float(x) for x in df["A_ratio"]]
    time_sec = [float(x) for x in df["time_sec"]]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].plot(K, beta0, "o-", color="C0", linewidth=2, markersize=8)
    axes[0].set_xlabel("$K$ (number of eigenmodes)")
    axes[0].set_ylabel(r"$\beta_0$")
    axes[0].set_title("Visual complexity")
    axes[0].set_xticks(K)

    axes[1].plot(K, A_ratio, "s-", color="C1", linewidth=2, markersize=8)
    axes[1].set_xlabel("$K$")
    axes[1].set_ylabel("$A_{\\mathrm{ratio}}$")
    axes[1].set_title("Nodal area ratio")
    axes[1].set_xticks(K)

    axes[2].plot(K, [t * 1000 for t in time_sec], "^-", color="C2", linewidth=2, markersize=8)
    axes[2].set_xlabel("$K$")
    axes[2].set_ylabel("Time (ms)")
    axes[2].set_title("Compute time per frame")
    axes[2].set_xticks(K)

    for ax in axes:
        ax.set_facecolor("white")
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", OUT_PATH, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
