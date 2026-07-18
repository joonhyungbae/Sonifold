"""
E3 histogram: per-frame beta0 distribution for the sphere under A3, direct vs
freqmatch mapping. The paper's E3 claim turns on the *shape* of this distribution,
not its mean. Under freqmatch the sphere is bimodal across STFT frames (frames where
the dominant piano harmonic lands near an l-shell centre give large coherent shell
sums; frames between shells give sparse fields), which is why the per-frame standard
deviation exceeds the mean. A histogram is the only way to show that.

Reads the per-frame beta0 values retained in E3_freqmatch.json (beta0_frames).
Run from project root: python scripts/figures/fig_E3_histogram.py
Output: figures/fig_E3_histogram.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent.parent.parent
E3 = root / "data" / "experiments" / "revision" / "E3_freqmatch.json"
OUT = root / "figures" / "fig_E3_histogram.pdf"

SHAPE = "sphere"
AUDIO = "A3"


def main():
    records = json.loads(E3.read_text())["records"]
    by = {(r["shape"], r["audio"], r["strategy"]): r for r in records}
    direct = by.get((SHAPE, AUDIO, "direct"))
    freqmatch = by.get((SHAPE, AUDIO, "freqmatch"))
    if not direct or "beta0_frames" not in direct:
        raise SystemExit("E3_freqmatch.json lacks per-frame data; re-run "
                         "scripts/revision_experiments/run_revision_experiments.py E3")

    d = np.array(direct["beta0_frames"], dtype=float)
    f = np.array(freqmatch["beta0_frames"], dtype=float)
    hi = max(d.max(), f.max())
    bins = np.linspace(0, hi, 40)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(d, bins=bins, alpha=0.65, label=f"Direct (mean {d.mean():.1f}, sd {d.std():.1f})",
            color="#3B6FB6")
    ax.hist(f, bins=bins, alpha=0.65, label=f"Freqmatch (mean {f.mean():.1f}, sd {f.std():.1f})",
            color="#C6534B")
    ax.set_xlabel(r"$\beta_0$ per STFT frame")
    ax.set_ylabel("Frame count")
    ax.set_title(f"{SHAPE.capitalize()} $\\times$ {AUDIO}: per-frame $\\beta_0$ distribution")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"Saved {OUT}  (direct n={len(d)}, freqmatch n={len(f)})")


if __name__ == "__main__":
    main()
