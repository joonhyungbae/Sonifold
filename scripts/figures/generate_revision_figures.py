"""
Generate revision-round figures from data/experiments/revision/E{1..4}*.json.

Outputs:
  figures/fig_rev_E1_rescale.pdf      mesh-rescale invariance
  figures/fig_rev_E2_transpose.pdf    pitch-transposition drift
  figures/fig_rev_E3_freqmatch.pdf    direct vs freqmatch mapping
  figures/fig_rev_E4_sameg.pdf        same-genus different metric
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = _ROOT / "data" / "experiments" / "revision"
OUT = _ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _load(name):
    p = DATA / name
    if not p.exists():
        print(f"[skip] {name} not found")
        return None
    return json.loads(p.read_text())


def fig_E1():
    data = _load("E1_rescale.json")
    if data is None:
        return
    alphas = data["alphas"]
    stimuli = data["stimuli"]
    shapes = data["shapes"]
    fig, axes = plt.subplots(1, len(shapes), figsize=(5 * len(shapes), 4), sharey=True)
    if len(shapes) == 1:
        axes = [axes]
    for ax, shape in zip(axes, shapes):
        for aid in stimuli:
            ys = [
                r["beta0_mean"]
                for r in data["records"]
                if r["shape"] == shape and r["audio"] == aid
            ]
            ax.plot(alphas, ys, "-o", label=aid)
        ax.set_xscale("log")
        ax.set_xlabel(r"mesh scale $\alpha$")
        ax.set_title(shape)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"mean $\beta_0$")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(r"E1: mesh-rescale invariance of $\beta_0$ under ordinal mapping")
    fig.tight_layout()
    out = OUT / "fig_rev_E1_rescale.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_E2():
    data = _load("E2_transpose.json")
    if data is None:
        return
    semis = data["semitones"]
    shapes = data["shapes"]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"sphere": "C0", "torus": "C1", "double_torus": "C2"}
    for shape in shapes:
        ys = [
            r["beta0_mean"]
            for r in data["records"]
            if r["shape"] == shape
        ]
        ax.plot(semis, ys, "-o", label=shape, color=colors.get(shape, "k"))
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("pitch shift (semitones)")
    ax.set_ylabel(r"mean $\beta_0$ (A3 piano, direct mapping)")
    ax.set_title("E2: pitch-transposition response")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / "fig_rev_E2_transpose.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_E3():
    data = _load("E3_freqmatch.json")
    if data is None:
        return
    shapes = data["shapes"]
    stimuli = data["stimuli"]
    strategies = data["strategies"]
    fig, axes = plt.subplots(1, len(shapes), figsize=(5 * len(shapes), 4), sharey=False)
    if len(shapes) == 1:
        axes = [axes]
    width = 0.35
    x = np.arange(len(stimuli))
    for ax, shape in zip(axes, shapes):
        for i, strat in enumerate(strategies):
            ys = [
                next(
                    r["beta0_mean"]
                    for r in data["records"]
                    if r["shape"] == shape and r["audio"] == aid and r["strategy"] == strat
                )
                for aid in stimuli
            ]
            ax.bar(x + (i - 0.5) * width, ys, width, label=strat)
        ax.set_xticks(x)
        ax.set_xticklabels(stimuli)
        ax.set_title(shape)
        ax.set_xlabel("stimulus")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel(r"mean $\beta_0$")
    axes[-1].legend(fontsize=8)
    fig.suptitle("E3: direct (ordinal) vs frequency-matched mapping")
    fig.tight_layout()
    out = OUT / "fig_rev_E3_freqmatch.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _pretty_torus_label(spec: str) -> str:
    """Convert internal mesh-code 'torus_<R>_<r>' to a math-mode label '$R{=}3,\\ r{=}1$'."""
    parts = spec.split("_")
    if len(parts) == 3 and parts[0] == "torus":
        return f"$R{{=}}{parts[1]},\\ r{{=}}{parts[2]}$"
    return spec


def fig_E4():
    data = _load("E4_sameg.json")
    if data is None:
        return
    specs = data["specs"]
    stimuli = data["stimuli"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = np.arange(len(specs))
    width = 0.25
    for i, aid in enumerate(stimuli):
        ys = [
            next(r["beta0_mean"] for r in data["records"] if r["label"] == lab and r["audio"] == aid)
            for lab in specs
        ]
        ax.bar(x + (i - 1) * width, ys, width, label=aid)
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_torus_label(s) for s in specs])
    ax.set_xlabel("Genus-1 torus embedding")
    ax.set_ylabel(r"mean $\beta_0$")
    ax.set_title("E4: genus 1, three embedded metrics")
    ax.legend(title="stimulus")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / "fig_rev_E4_sameg.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else ["E1", "E2", "E3", "E4"]
    dispatch = {"E1": fig_E1, "E2": fig_E2, "E3": fig_E3, "E4": fig_E4}
    for n in which:
        dispatch[n]()


if __name__ == "__main__":
    main()
