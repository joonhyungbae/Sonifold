"""
Random baseline vs. audio-driven β₀ comparison: for each mesh, compare mean β₀ under
random coefficients to mean β₀ for each stimulus (A1–A7), compute Δ and z-scores,
and produce a table + figure for reviewer (spectral geometry: does audio deviate from random?).

Usage:
  python scripts/random_vs_audio_comparison.py

Prerequisites:
  - data/results/random_coefficient_summary.csv (from scripts/random_coefficient_test.py)
  - Precomputed eigenbases for genus sequence meshes (data/experiments/eigen or data/eigen)
  - Optional: data/results/audio_driven_beta0_direct.csv (columns: mesh, beta0_A1, ..., beta0_A7)
    to skip recomputing audio-driven β₀; if missing, computes mean over STFT frames (10s audio).

Output:
  - data/results/random_vs_audio_comparison.csv
  - figures/fig_random_vs_audio.pdf
  - stdout: per-mesh summary of stimuli >2σ above or below random baseline
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from audio.audio_library import get_audio
from audio.fft_analysis import compute_fft_frames, N_FFT
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics

# Same mesh set as random_coefficient_test.py (genus 0–4)
GENUS_MESHES = [
    ("sphere_genus0", 0),
    ("torus_genus1", 1),
    ("double_torus_genus2", 2),
    ("triple_torus_genus3", 3),
    ("quad_torus_genus4", 4),
]
STIMULI = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
RANDOM_DISTRIBUTION = "uniform_sphere"  # one baseline per mesh

TARGET_DURATION_SEC = 10.0
HOP_LENGTH = 512
N_EIGEN = 50
STRATEGY = "direct"

RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"
RANDOM_SUMMARY_CSV = RESULTS_DIR / "random_coefficient_summary.csv"
OUT_COMPARISON_CSV = RESULTS_DIR / "random_vs_audio_comparison.csv"
OUT_FIG = FIGURES_DIR / "fig_random_vs_audio.pdf"
# Optional: if present, load audio-driven mean β₀ from this CSV (columns: mesh, A1, A2, ..., A7) to skip long computation
AUDIO_DRIVEN_CSV = RESULTS_DIR / "audio_driven_beta0_direct.csv"

EIGEN_EXPERIMENTS = root / "data" / "experiments" / "eigen"
EIGEN_MAIN = root / "data" / "eigen"


def load_eigen(mesh_id: str):
    """Load V, F, evals, evecs (N_EIGEN, n_vertices) from experiments/eigen or data/eigen. Same convention as random_coefficient_test."""
    for base, ext in [(EIGEN_EXPERIMENTS, ".npz"), (EIGEN_EXPERIMENTS, ".json"), (EIGEN_MAIN, ".json")]:
        path = base / (mesh_id + ext)
        if not path.exists():
            alt_id = (
                mesh_id.replace("_genus0", "")
                .replace("_genus1", "")
                .replace("_genus2", "")
                .replace("_genus3", "")
                .replace("_genus4", "")
            )
            path_alt = base / (alt_id + ext)
            if path_alt.exists():
                path = path_alt
            else:
                continue
        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            V = np.asarray(data["vertices"], dtype=np.float64)
            F = np.asarray(data["faces"], dtype=np.int32)
            evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
            evals = (
                np.asarray(data["eigenvalues"], dtype=np.float64).ravel()
                if "eigenvalues" in data.files
                else None
            )
        else:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            V = np.array(data["vertices"], dtype=np.float64)
            F = np.array(data["faces"], dtype=np.int32)
            evecs = np.array(data["eigenvectors"], dtype=np.float64)
            evals = np.array(data.get("eigenvalues", []), dtype=np.float64)
        n_verts = V.shape[0]
        if evecs.shape[0] == n_verts:
            evecs = evecs[:, :N_EIGEN].T
        else:
            evecs = evecs[:N_EIGEN, :]
        if evecs.shape[0] < N_EIGEN:
            continue
        evals = evals[:N_EIGEN] if evals is not None and len(evals) >= N_EIGEN else None
        return V, F, evals, evecs
    return None


def build_audio_10s(audio_id: str):
    """Load audio and tile/pad to TARGET_DURATION_SEC."""
    sig, sr = get_audio(audio_id)
    n_target = int(TARGET_DURATION_SEC * sr)
    if len(sig) < n_target:
        repeats = (n_target + len(sig) - 1) // len(sig)
        sig = np.tile(sig, repeats)[:n_target]
    else:
        sig = sig[:n_target]
    return sig, sr


def load_random_baseline():
    """Load random coefficient β₀ summary: mesh -> (mean, std) for RANDOM_DISTRIBUTION."""
    if not RANDOM_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Random summary not found: {RANDOM_SUMMARY_CSV}. Run scripts/random_coefficient_test.py first."
        )
    with open(RANDOM_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if r.get("distribution") != RANDOM_DISTRIBUTION:
            continue
        mesh = r["mesh_name"]
        out[mesh] = (float(r["mean_beta0"]), float(r["std_beta0"]))
    return out


def load_audio_driven_from_csv():
    """If AUDIO_DRIVEN_CSV exists with mesh + A1..A7 columns, return dict mesh_name -> {A1: float, ...}; else None."""
    if not AUDIO_DRIVEN_CSV.exists():
        return None
    with open(AUDIO_DRIVEN_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "mesh" not in rows[0]:
        return None
    required = ["mesh"] + [f"beta0_{a}" for a in STIMULI]
    if not all(c in rows[0] for c in required):
        # Try mesh, A1, A2, ...
        if not all(a in rows[0] for a in STIMULI):
            return None
        key_fmt = lambda a: a
    else:
        key_fmt = lambda a: f"beta0_{a}"
    out = {}
    for r in rows:
        mesh = r.get("mesh", r.get("mesh_id", "")).strip()
        if not mesh:
            continue
        out[mesh] = {}
        for a in STIMULI:
            try:
                out[mesh][a] = float(r[key_fmt(a)])
            except (KeyError, ValueError):
                return None
    return out if out else None


def compute_audio_driven_beta0_mean_per_mesh():
    """For each mesh × stimulus, compute mean β₀ over STFT frames (direct mapping). Returns dict (mesh_name -> {A1: mean, ...})."""
    audio_means = load_audio_driven_from_csv()
    if audio_means is not None:
        print(f"Loaded audio-driven β₀ from {AUDIO_DRIVEN_CSV}", file=sys.stderr)
        return audio_means
    audio_means = {}  # mesh_name -> {A1: float, ...}
    for mesh_name, genus in tqdm(GENUS_MESHES, desc="mesh"):
        eigen = load_eigen(mesh_name)
        if eigen is None:
            print(f"Skip {mesh_name}: no eigenbasis", file=sys.stderr)
            continue
        V, F, evals, evecs = eigen
        audio_means[mesh_name] = {}
        for audio_id in tqdm(STIMULI, desc=mesh_name, leave=False):
            sig, sr = build_audio_10s(audio_id)
            frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
            beta0_list = []
            for mag, _ in frames:
                coef = map_fft_to_coefficients(mag, N_EIGEN, strategy=STRATEGY, eigenvalues=evals)
                f = compute_scalar_field(evecs, coef)
                m = compute_topology_metrics(V, F, f)
                beta0_list.append(m.beta0)
            audio_means[mesh_name][audio_id] = float(np.mean(beta0_list))
    return audio_means


def build_comparison_rows(random_baseline, audio_means):
    """Build list of row dicts for CSV: mesh, beta0_random_mean, beta0_random_std, beta0_A1..A7, delta_A1..A7, z_A1..z_A7."""
    rows = []
    for mesh_name, genus in GENUS_MESHES:
        if mesh_name not in random_baseline or mesh_name not in audio_means:
            continue
        r_mean, r_std = random_baseline[mesh_name]
        r_std = max(r_std, 1e-10)
        row = {
            "mesh": mesh_name,
            "beta0_random_mean": round(r_mean, 4),
            "beta0_random_std": round(r_std, 4),
        }
        for a in STIMULI:
            b_audio = audio_means[mesh_name][a]
            delta = b_audio - r_mean
            z = delta / r_std
            row[f"beta0_{a}"] = round(b_audio, 4)
            row[f"delta_{a}"] = round(delta, 4)
            row[f"z_{a}"] = round(z, 4)
        rows.append(row)
    return rows


def write_comparison_csv(rows):
    """Write random_vs_audio_comparison.csv."""
    fieldnames = (
        ["mesh", "beta0_random_mean", "beta0_random_std"]
        + [f"beta0_{a}" for a in STIMULI]
        + [f"delta_{a}" for a in STIMULI]
        + [f"z_{a}" for a in STIMULI]
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {OUT_COMPARISON_CSV}", file=sys.stderr)


def plot_figure(rows):
    """Grouped dot plot: x = meshes, gray band = random mean ± 1σ, colored dots = β₀ per stimulus."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meshes = [r["mesh"] for r in rows]
    n_meshes = len(meshes)
    x = np.arange(n_meshes)
    width = 0.8

    plt.rc("font", family="serif")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Gray band: random mean ± 1σ
    r_mean = np.array([r["beta0_random_mean"] for r in rows])
    r_std = np.array([r["beta0_random_std"] for r in rows])
    ax.fill_between(x, r_mean - r_std, r_mean + r_std, color="gray", alpha=0.35, label="Random baseline ±1σ")
    ax.plot(x, r_mean, color="gray", linewidth=1.5, linestyle="-", label="Random mean")

    # Colored dots: one color per stimulus (A1–A7)
    colors = plt.cm.tab10(np.linspace(0, 1, len(STIMULI)))
    for j, a in enumerate(STIMULI):
        beta0_vals = [r[f"beta0_{a}"] for r in rows]
        # Jitter x slightly so dots don't overlap
        jitter = (j - len(STIMULI) / 2 + 0.5) * (width / (len(STIMULI) + 1))
        ax.scatter(x + jitter, beta0_vals, color=colors[j], s=28, label=a, zorder=3, edgecolors="white", linewidths=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_genus0", "").replace("_genus1", "").replace("_genus2", "").replace("_genus3", "").replace("_genus4", "") for m in meshes], fontsize=8)
    ax.set_xlabel("Mesh")
    ax.set_ylabel(r"$\beta_0$")
    ax.set_title(r"Audio-driven $\beta_0$ vs. random baseline (direct mapping, mean over STFT frames)")
    ax.legend(ncol=2, loc="upper right", fontsize=7)
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved {OUT_FIG}", file=sys.stderr)


def print_summary(rows):
    """Print for each mesh which stimuli are >2σ above or below random baseline."""
    print("\n=== Stimuli >2σ from random baseline (per mesh) ===\n")
    for r in rows:
        mesh = r["mesh"]
        above = [a for a in STIMULI if r[f"z_{a}"] > 2]
        below = [a for a in STIMULI if r[f"z_{a}"] < -2]
        if above or below:
            parts = []
            if above:
                parts.append(f">2σ above: {', '.join(above)}")
            if below:
                parts.append(f">2σ below: {', '.join(below)}")
            print(f"  {mesh}: {'; '.join(parts)}")
        else:
            print(f"  {mesh}: (none)")


def main():
    random_baseline = load_random_baseline()
    audio_means = compute_audio_driven_beta0_mean_per_mesh()
    if not audio_means:
        print("No audio-driven β₀ computed (eigen missing for all meshes?).", file=sys.stderr)
        return 1
    rows = build_comparison_rows(random_baseline, audio_means)
    if not rows:
        print("No rows: no mesh has both random baseline and audio-driven data.", file=sys.stderr)
        return 1
    write_comparison_csv(rows)
    plot_figure(rows)
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
