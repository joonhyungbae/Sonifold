"""
Compare three mapping strategies (direct, mel-weighted, energy-weighted) on
representative mesh–audio pairs. Computes time-averaged β₀, A_ratio, S per
condition and outputs CSV + LaTeX table for the paper.

Usage:
  python scripts/compare_mapping_strategies.py

Prerequisites:
  - Precomputed eigenbases in data/eigen/ or data/eigenbases/ (*.npz or *.json)
  - Audio: A1, A3, A5, A7 (from audio_library / data/audio)

Output:
  - data/results/mapping_comparison.csv
  - figures/tab_mapping_comparison.tex

Reuses: audio_library, fft_analysis, spectral_mapping, scalar_field,
        nodal_surface, symmetry (same K=50, hop=512, ε threshold as existing).
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
from audio.fft_analysis import compute_fft_frames
from mapping.spectral_mapping import map_fft_to_coefficients
from analysis.scalar_field import compute_scalar_field
from analysis.nodal_surface import compute_topology_metrics
from analysis.symmetry import compute_symmetry

# Same parameters as compute_beta0_stats / run_all
N_EIGEN = 50
HOP_LENGTH = 512
THRESHOLD_RATIO = 0.01
STRATEGIES = ["direct", "mel", "energy"]

EIGEN_DIR = root / "data" / "eigen"
EIGENBASES_DIR = root / "data" / "eigenbases"
RESULTS_DIR = root / "data" / "results"
FIGURES_DIR = root / "figures"
OUT_CSV = RESULTS_DIR / "mapping_comparison.csv"
OUT_TEX = FIGURES_DIR / "tab_mapping_comparison.tex"

# Representative mesh × audio pairs for the paper (Table: mapping comparison)
CONDITIONS = [
    ("sphere", "A1"),       # 440 Hz sine
    ("sphere", "A5"),       # white noise
    ("torus", "A3"),        # piano passage
    ("cube", "A7"),         # orchestral excerpt
    ("double_torus", "A5"), # white noise
    ("double_torus", "A3"), # piano passage
]
# Display names for LaTeX table (paper style)
MESH_DISPLAY = {"sphere": "sphere", "torus": "torus", "cube": "cube", "double_torus": "double torus"}


def load_mesh_eigen(mesh_name: str):
    """
    Load vertices, faces, eigenvectors (first N_EIGEN), eigenvalues from
    data/eigen/ or data/eigenbases/ (<mesh>.npz or <mesh>.json).
    Returns (V, F, evecs, evals) with evecs shape (N_EIGEN, n_vertices), or None.
    """
    for base in [EIGEN_DIR, EIGENBASES_DIR]:
        if not base.exists():
            continue
        for ext in [".npz", ".json"]:
            path = base / f"{mesh_name}{ext}"
            if not path.exists():
                continue
            try:
                if path.suffix == ".npz":
                    data = np.load(path, allow_pickle=True)
                    V = np.asarray(data["vertices"], dtype=np.float64)
                    F = np.asarray(data["faces"], dtype=np.int32)
                    evecs = np.asarray(data["eigenvectors"], dtype=np.float64)
                    evals = np.asarray(data["eigenvalues"], dtype=np.float64).ravel() if "eigenvalues" in data.files else None
                else:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    V = np.array(data["vertices"], dtype=np.float64)
                    F = np.array(data["faces"], dtype=np.int32)
                    evecs = np.array(data["eigenvectors"], dtype=np.float64)
                    evals = np.array(data.get("eigenvalues", []), dtype=np.float64)
                    if evals is not None and evals.size == 0:
                        evals = None
                # Normalize to (N_EIGEN, n_vertices) for compute_scalar_field(c @ evecs)
                if evecs.shape[0] == V.shape[0]:
                    evecs = evecs[:, :N_EIGEN].T
                else:
                    evecs = evecs[:N_EIGEN, :]
                if evecs.shape[0] < N_EIGEN:
                    return None
                evals = evals[:N_EIGEN] if evals is not None and len(evals) >= N_EIGEN else None
                return V, F, evecs, evals
            except Exception as e:
                print(f"Warning: failed to load {path}: {e}", file=sys.stderr)
                continue
    return None


def run_condition(mesh_name: str, audio_id: str, strategy: str, V, F, evecs, evals):
    """
    Run pipeline for one (mesh, audio, strategy): STFT frames -> coefficients ->
    scalar field -> topology + symmetry. Returns lists of beta0, A_ratio, S per frame.
    """
    sig, sr = get_audio(audio_id)
    frames = compute_fft_frames(sig, sample_rate=sr, hop_length=HOP_LENGTH)
    if not frames:
        return None
    beta0_list, A_ratio_list, S_list = [], [], []
    for mag, _ in frames:
        coef = map_fft_to_coefficients(mag, N_EIGEN, strategy=strategy, eigenvalues=evals)
        f = compute_scalar_field(evecs, coef)
        m = compute_topology_metrics(V, F, f, threshold_ratio=THRESHOLD_RATIO)
        S = compute_symmetry(V, f)
        beta0_list.append(m.beta0)
        A_ratio_list.append(m.A_ratio)
        S_list.append(S)
    return {
        "beta0": np.array(beta0_list, dtype=np.float64),
        "A_ratio": np.array(A_ratio_list, dtype=np.float64),
        "S": np.array(S_list, dtype=np.float64),
    }


def main():
    # TODO (paper alignment): In the paper, energy-weighted is described as using
    # "sum of squared magnitudes per mel band"; the current implementation uses
    # linear bands and eigenvalue weighting. If you need strict paper alignment,
    # implement mel-band energy weighting in mapping/spectral_mapping.py.
    print(
        "Note: energy mapping uses linear bands + eigenvalue weighting; "
        "paper describes mel-band sum-of-squared-magnitudes (see TODO in script).",
        file=sys.stderr,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for mesh_name, audio_id in tqdm(CONDITIONS, desc="mesh×audio"):
        eigen = load_mesh_eigen(mesh_name)
        if eigen is None:
            print(f"Skip {mesh_name}: no eigen data in data/eigen/ or data/eigenbases/", file=sys.stderr)
            for strategy in STRATEGIES:
                rows.append({
                    "mesh": mesh_name, "audio": audio_id, "mapping": strategy,
                    "beta0_mean": None, "beta0_std": None,
                    "A_ratio_mean": None, "A_ratio_std": None,
                    "S_mean": None, "S_std": None,
                })
            continue
        V, F, evecs, evals = eigen
        for strategy in STRATEGIES:
            try:
                out = run_condition(mesh_name, audio_id, strategy, V, F, evecs, evals)
            except Exception as e:
                print(f"Warning: {mesh_name} {audio_id} {strategy}: {e}", file=sys.stderr)
                rows.append({
                    "mesh": mesh_name, "audio": audio_id, "mapping": strategy,
                    "beta0_mean": None, "beta0_std": None,
                    "A_ratio_mean": None, "A_ratio_std": None,
                    "S_mean": None, "S_std": None,
                })
                continue
            if out is None:
                rows.append({
                    "mesh": mesh_name, "audio": audio_id, "mapping": strategy,
                    "beta0_mean": None, "beta0_std": None,
                    "A_ratio_mean": None, "A_ratio_std": None,
                    "S_mean": None, "S_std": None,
                })
                continue
            n = len(out["beta0"])
            rows.append({
                "mesh": mesh_name,
                "audio": audio_id,
                "mapping": strategy,
                "beta0_mean": round(float(np.mean(out["beta0"])), 2),
                "beta0_std": round(float(np.std(out["beta0"])) if n > 1 else 0.0, 2),
                "A_ratio_mean": round(float(np.mean(out["A_ratio"])), 4),
                "A_ratio_std": round(float(np.std(out["A_ratio"])) if n > 1 else 0.0, 4),
                "S_mean": round(float(np.mean(out["S"])), 4),
                "S_std": round(float(np.std(out["S"])) if n > 1 else 0.0, 4),
            })

    # CSV
    fieldnames = ["mesh", "audio", "mapping", "beta0_mean", "beta0_std", "A_ratio_mean", "A_ratio_std", "S_mean", "S_std"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {OUT_CSV}", file=sys.stderr)

    # LaTeX table: grouped by mesh–audio, three sub-rows (direct / mel / energy)
    lines = [
        "% Mapping strategy comparison (Table for paper).",
        "% Rows grouped by mesh--audio; sub-rows: direct / mel / energy.",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Mesh--audio & Mapping & $\\beta_0 \\pm \\sigma$ & $A_{\\mathrm{ratio}} \\pm \\sigma$ & $S \\pm \\sigma$ \\\\",
        "\\midrule",
    ]
    for (mesh_name, audio_id) in CONDITIONS:
        group = [r for r in rows if r["mesh"] == mesh_name and r["audio"] == audio_id]
        label = MESH_DISPLAY.get(mesh_name, mesh_name)
        pair_label = f"{label} $\\times$ {audio_id}"
        for i, r in enumerate(group):
            if i == 0:
                col1 = pair_label
            else:
                col1 = ""
            col2 = r["mapping"]
            if r["beta0_mean"] is None:
                cells = " & -- & -- & --"
            else:
                b0 = f"${r['beta0_mean']:.1f} \\pm {r['beta0_std']:.1f}$"
                ar = f"${r['A_ratio_mean']:.3f} \\pm {r['A_ratio_std']:.3f}$"
                ss = f"${r['S_mean']:.2f} \\pm {r['S_std']:.2f}$"
                cells = f" & {b0} & {ar} & {ss}"
            lines.append(f"{col1} & {col2}" + cells + " \\\\")
        lines.append("\\addlinespace")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {OUT_TEX}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
