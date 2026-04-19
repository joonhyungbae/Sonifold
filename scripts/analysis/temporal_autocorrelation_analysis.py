#!/usr/bin/env python3
"""
Temporal autocorrelation analysis of β₀ time series.

Quantifies autocorrelation due to overlapping STFT windows (hop=512 < window=2048),
computes effective sample size and corrected standard errors, and produces
a ready-to-paste caveat paragraph for the paper.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Optional: statsmodels for ACF (fallback to manual if not installed)
try:
    from statsmodels.tsa.stattools import acf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "results"
FIG_DIR = PROJECT_ROOT / "figures"
MAX_LAG = 20
SHAPIRO_SUBSAMPLE = 500
CONF_LEVEL = 0.95  # for ACF band ±1.96/sqrt(n)

# Desired conditions: (mesh_display_name, mesh_id, audio)
CONDITIONS = [
    ("Sphere", "sphere_genus0", "A3"),
    ("Torus", "torus_genus1", "A3"),
    ("Double torus", "double_torus_genus2", "A3"),
    ("Sphere", "sphere_genus0", "A7"),
    ("Torus", "torus_genus1", "A7"),
]


def load_beta0_series():
    """
    Load frame-level β₀ time series for each condition.
    Uses results_temporal.csv (audio_id, mesh_id, frame, beta0) and
    temporal_genus_extended.csv (mesh_name, frame_index, beta0) for double_torus.
    Returns dict: (mesh_id, audio) -> 1D array of beta0 (sorted by frame).
    """
    series = {}
    temporal_path = DATA_DIR / "results_temporal.csv"
    extended_path = DATA_DIR / "temporal_genus_extended.csv"

    if not temporal_path.exists():
        print(f"Error: {temporal_path} not found.")
        print("Available files in data/results/:")
        for f in sorted(DATA_DIR.glob("*.csv")):
            try:
                df = pd.read_csv(f, nrows=2)
                print(f"  {f.name}: columns = {list(df.columns)}")
            except Exception as e:
                print(f"  {f.name}: (read error: {e})")
        return series

    df_t = pd.read_csv(temporal_path)
    required = {"audio_id", "mesh_id", "frame", "beta0"}
    if not required.issubset(df_t.columns):
        print(f"Error: results_temporal.csv must have columns {required}; found {list(df_t.columns)}")
        return series

    for _, row in df_t.groupby(["audio_id", "mesh_id"]):
        audio_id = row["audio_id"].iloc[0]
        mesh_id = row["mesh_id"].iloc[0]
        sub = row.sort_values("frame")
        beta0 = sub["beta0"].values.astype(float)
        series[(mesh_id, audio_id)] = beta0

    # Double torus: from temporal_genus_extended (no audio column; assume A3)
    if extended_path.exists():
        df_e = pd.read_csv(extended_path)
        if "mesh_name" in df_e.columns and "frame_index" in df_e.columns and "beta0" in df_e.columns:
            for mesh_name in df_e["mesh_name"].unique():
                sub = df_e[df_e["mesh_name"] == mesh_name].sort_values("frame_index")
                beta0 = sub["beta0"].values.astype(float)
                if (mesh_name, "A3") not in series:  # avoid overwriting if already in results_temporal
                    series[(mesh_name, "A3")] = beta0
                # If we only have extended and no A7 for this mesh, we still only have A3
        else:
            print(f"Warning: temporal_genus_extended.csv missing mesh_name/frame_index/beta0: {list(df_e.columns)}")

    return series


def acf_manual(x, nlags):
    """Compute ACF for lags 0..nlags (inclusive). Returns array of length nlags+1."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    acf_vals = np.zeros(nlags + 1)
    c0 = np.dot(x, x) / n
    if c0 == 0:
        return acf_vals
    acf_vals[0] = 1.0
    for k in range(1, nlags + 1):
        acf_vals[k] = np.dot(x[:-k], x[k:]) / n / c0
    return acf_vals


def compute_acf(beta0, nlags=MAX_LAG):
    if HAS_STATSMODELS:
        acf_vals = acf(beta0, nlags=nlags, fft=True)
        # statsmodels returns lag 0..nlags; ensure length nlags+1
        if len(acf_vals) < nlags + 1:
            acf_vals = np.resize(acf_vals, nlags + 1)
        return np.asarray(acf_vals, dtype=float)
    return acf_manual(beta0, nlags)


def effective_sample_size(n, r1):
    """n_eff = n * (1 - r1) / (1 + r1)."""
    if r1 >= 1:
        return 1.0
    return n * (1.0 - r1) / (1.0 + r1)


def run_shapiro(beta0):
    """Shapiro-Wilk on full series or subsample of 500 if n > 5000."""
    x = np.asarray(beta0, dtype=float)
    if len(x) > SHAPIRO_SUBSAMPLE:
        rng = np.random.default_rng(42)
        x = rng.choice(x, size=SHAPIRO_SUBSAMPLE, replace=False)
    if len(x) < 3:
        return np.nan, np.nan
    w, p = stats.shapiro(x)
    return w, p


def analyze_condition(name, mesh_id, audio, beta0):
    """Compute all statistics for one condition."""
    beta0 = np.asarray(beta0, dtype=float)
    n = len(beta0)
    mean_b = float(np.mean(beta0))
    std_b = float(np.std(beta0, ddof=1)) if n > 1 else 0.0

    acf_vals = compute_acf(beta0, nlags=MAX_LAG)
    r1 = float(acf_vals[1]) if len(acf_vals) > 1 else 0.0

    n_eff = effective_sample_size(n, r1)
    naive_se = std_b / np.sqrt(n) if n > 0 else np.nan
    corrected_se = std_b / np.sqrt(n_eff) if n_eff > 0 else np.nan

    shapiro_w, shapiro_p = run_shapiro(beta0)

    return {
        "mesh": name,
        "audio": audio,
        "n_frames": n,
        "mean_beta0": mean_b,
        "std_beta0": std_b,
        "lag1_autocorrelation": r1,
        "n_effective": n_eff,
        "naive_SE": naive_se,
        "corrected_SE": corrected_se,
        "shapiro_W": shapiro_w,
        "shapiro_p": shapiro_p,
        "acf": acf_vals,
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    series = load_beta0_series()
    if not series:
        print("No β₀ time series loaded. Exiting.")
        return

    # Build condition key -> (display name, mesh_id, audio)
    key_to_display = {}
    for (display_name, mesh_id, audio) in CONDITIONS:
        key = (mesh_id, audio)
        if key not in key_to_display:
            key_to_display[key] = display_name

    results = []
    acf_data = []  # for figure: (display_label, acf_vals, n, color, linestyle)

    # Colors by mesh, linestyle by audio
    mesh_colors = {"Sphere": "C0", "Torus": "C1", "Double torus": "C2"}
    audio_linestyle = {"A3": "-", "A7": "--"}

    for (display_name, mesh_id, audio) in CONDITIONS:
        key = (mesh_id, audio)
        if key not in series:
            print(f"  Skip (no data): {display_name} × {audio}")
            continue
        beta0 = series[key]
        out = analyze_condition(display_name, mesh_id, audio, beta0)
        results.append(out)
        acf_data.append({
            "label": f"{display_name} × {audio}",
            "acf": out["acf"],
            "n": out["n_frames"],
            "color": mesh_colors.get(display_name, "gray"),
            "linestyle": audio_linestyle.get(audio, "-"),
        })

    if not results:
        print("No conditions had data. Exiting.")
        return

    # CSV
    rows = []
    for r in results:
        rows.append({
            "mesh": r["mesh"],
            "audio": r["audio"],
            "n_frames": int(r["n_frames"]),
            "mean_beta0": r["mean_beta0"],
            "std_beta0": r["std_beta0"],
            "lag1_autocorrelation": r["lag1_autocorrelation"],
            "n_effective": r["n_effective"],
            "naive_SE": r["naive_SE"],
            "corrected_SE": r["corrected_SE"],
            "shapiro_W": r["shapiro_W"],
            "shapiro_p": r["shapiro_p"],
        })
    out_df = pd.DataFrame(rows)
    csv_path = DATA_DIR / "temporal_autocorrelation.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Summary markdown
    md_lines = [
        "# Temporal autocorrelation of β₀ time series",
        "",
        "Overlapping STFT windows (hop=512, window=2048) induce temporal autocorrelation.",
        "Effective sample size: n_eff = n × (1 − r₁) / (1 + r₁).",
        "",
        "## Per-condition results",
        "",
    ]
    for r in results:
        md_lines.append(f"### {r['mesh']} × {r['audio']}")
        md_lines.append(f"- n_frames = {r['n_frames']}, n_effective = {r['n_effective']:.1f}")
        md_lines.append(f"- Lag-1 autocorrelation r₁ = {r['lag1_autocorrelation']:.4f}")
        md_lines.append(f"- mean(β₀) = {r['mean_beta0']:.4f}, std(β₀) = {r['std_beta0']:.4f}")
        md_lines.append(f"- Naive SE = {r['naive_SE']:.6f}, Corrected SE = {r['corrected_SE']:.6f}")
        md_lines.append(f"- Shapiro-Wilk W = {r['shapiro_W']:.4f}, p = {r['shapiro_p']:.4e}")
        md_lines.append("")
    summary_path = DATA_DIR / "temporal_autocorrelation_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {summary_path}")

    # Figure: 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: ACF (95% band ±1.96/sqrt(n) with reference n from first condition)
    lags = np.arange(MAX_LAG + 1)
    n_ref = acf_data[0]["n"] if acf_data else 858
    band_ref = 1.96 / np.sqrt(n_ref)
    ax1.axhspan(-band_ref, band_ref, alpha=0.2, color="gray")
    ax1.axhline(band_ref, color="gray", linestyle="--", linewidth=0.8)
    ax1.axhline(-band_ref, color="gray", linestyle="--", linewidth=0.8)
    for d in acf_data:
        ax1.plot(lags, d["acf"], color=d["color"], linestyle=d["linestyle"], label=d["label"])
    ax1.set_xlabel("Lag")
    ax1.set_ylabel("ACF")
    ax1.set_title("Autocorrelation function (lags 0–20)")
    ax1.legend(loc="best", fontsize=8)
    ax1.set_ylim(-0.2, 1.05)
    ax1.grid(True, alpha=0.3)

    # Right: Bar chart of r₁
    labels = [f"{r['mesh']} × {r['audio']}" for r in results]
    r1_vals = [r["lag1_autocorrelation"] for r in results]
    colors = [mesh_colors.get(r["mesh"], "gray") for r in results]
    x_pos = np.arange(len(labels))
    ax2.bar(x_pos, r1_vals, color=colors)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Lag-1 autocorrelation r₁")
    ax2.set_title("Lag-1 autocorrelation by condition")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = FIG_DIR / "fig_autocorrelation.pdf"
    fig.savefig(fig_path)
    plt.close()
    print(f"Wrote {fig_path}")

    # Ready-to-paste paragraph
    r1_vals = [r["lag1_autocorrelation"] for r in results]
    r1_min = min(r1_vals)
    r1_max = max(r1_vals)
    idx_min = int(np.argmin(r1_vals))
    idx_max = int(np.argmax(r1_vals))
    min_cond = f"{results[idx_min]['mesh']} × {results[idx_min]['audio']}"
    max_cond = f"{results[idx_max]['mesh']} × {results[idx_max]['audio']}"
    n_nom = results[0]["n_frames"]
    n_eff_min = min(r["n_effective"] for r in results)
    n_eff_max = max(r["n_effective"] for r in results)
    se_ratios = [r["corrected_SE"] / r["naive_SE"] if r["naive_SE"] and r["naive_SE"] > 0 else np.nan for r in results]
    z_se_ratio = float(np.nanmean(se_ratios)) if se_ratios else np.nan
    normal_conditions = [f"{r['mesh']} × {r['audio']}" for r in results if r["shapiro_p"] > 0.05]
    non_normal = [f"{r['mesh']} × {r['audio']}" for r in results if r["shapiro_p"] <= 0.05]
    if non_normal and normal_conditions:
        norm_text = f"non-normality for {', '.join(non_normal)} and normality for {', '.join(normal_conditions)}"
    elif non_normal:
        norm_text = "non-normality for all conditions"
    else:
        norm_text = "normality for all conditions"

    paragraph = (
        "The β₀ time series exhibit substantial temporal autocorrelation due to the overlapping "
        "STFT windows (hop 512 < window 2048). Lag-1 autocorrelation ranges from r₁ = {r1_min:.3f} "
        "({min_cond}) to r₁ = {r1_max:.3f} ({max_cond}), yielding effective sample sizes of "
        "N_eff ≈ {n_eff_min:.0f}–{n_eff_max:.0f} compared to nominal frame counts of N = {n_nom}. "
        "The corrected standard errors are approximately {z_se_ratio:.2f} times larger than the naive "
        "estimates. Shapiro–Wilk tests indicate {norm_text}. The z-scores reported in Section 5.3 "
        "should therefore be interpreted as descriptive indicators of effect direction and approximate "
        "magnitude, not as formal test statistics."
    ).format(
        r1_min=r1_min,
        r1_max=r1_max,
        min_cond=min_cond,
        max_cond=max_cond,
        n_eff_min=n_eff_min,
        n_eff_max=n_eff_max,
        n_nom=n_nom,
        z_se_ratio=z_se_ratio,
        norm_text=norm_text,
    )

    print("\n" + "=" * 72)
    print("READY-TO-PASTE PARAGRAPH FOR THE PAPER")
    print("=" * 72)
    print(paragraph)
    print("=" * 72)


if __name__ == "__main__":
    main()
