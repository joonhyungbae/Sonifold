"""H1, H2, H3 hypothesis tests. results.csv + data/eigen/*.npz."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
RESULTS_CSV = root / "data" / "results" / "results.csv"
EIGEN_DIR = root / "data" / "eigen"
AUDIO_ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]


def spectral_gap(mesh_name):
    d = np.load(EIGEN_DIR / (mesh_name + ".npz"), allow_pickle=True)
    ev = d["eigenvalues"]
    return float(ev[1] / (ev[0] + 1e-20)) if len(ev) >= 2 else np.nan


def weyl_density(mesh_name, k=10):
    d = np.load(EIGEN_DIR / (mesh_name + ".npz"), allow_pickle=True)
    ev = np.sort(d["eigenvalues"])
    if len(ev) < 3:
        return np.nan
    return float(1.0 / (np.mean(np.diff(ev[:k])) + 1e-20))


def h1_test(df):
    from scipy.stats import spearmanr
    meshes = df["mesh"].unique()
    g = [spectral_gap(m) for m in meshes]
    b = []
    for m in meshes:
        sub = df[(df["mesh"] == m) & (df["strategy"] == "direct")]
        idx = len(AUDIO_ORDER)
        for i, a in enumerate(AUDIO_ORDER):
            r = sub[sub["audio"] == a]
            if not r.empty and r["beta0"].iloc[0] > 1:
                idx = i
                break
        b.append(idx)
    g, b = np.array(g), np.array(b)
    ok = np.isfinite(g)
    if ok.sum() < 3:
        return {"rho": np.nan, "p": np.nan}
    rho, p = spearmanr(g[ok], b[ok])
    return {"rho": float(rho), "p": float(p)}


def h2_test(df):
    from scipy.stats import mannwhitneyu
    sub = df[df["audio"].isin(["A1", "A2"])]
    high = sub[sub["mesh"].isin(["sphere", "torus"])]["S"].dropna()
    low = sub[sub["mesh"].isin(["cube", "ellipsoid", "bunny", "spot"])]["S"].dropna()
    if len(high) < 2 or len(low) < 2:
        return {"p": np.nan, "high_mean": np.nan, "low_mean": np.nan}
    _, p = mannwhitneyu(high, low, alternative="two-sided")
    return {"p": float(p), "high_mean": float(high.mean()), "low_mean": float(low.mean())}


def h3_test(df):
    from scipy import stats
    rows = []
    for mesh in df["mesh"].unique():
        w = weyl_density(mesh)
        for _, r in df[df["mesh"] == mesh].iterrows():
            rows.append({"w": w, "beta0": r["beta0"]})
    h3 = pd.DataFrame(rows).dropna()
    if len(h3) < 5:
        return {"r2": np.nan, "p": np.nan}
    _, _, r, p, _ = stats.linregress(h3["w"], h3["beta0"])
    return {"r2": float(r * r), "p": float(p)}


def main():
    df = pd.read_csv(RESULTS_CSV)
    print("=== H1: spectral gap vs min bandwidth (beta0>1) ===")
    h1 = h1_test(df)
    print("  Spearman rho=%.4f  p=%.4f" % (h1["rho"], h1["p"]))
    print("=== H2: symmetric vs asymmetric mesh (S, A1/A2) ===")
    h2 = h2_test(df)
    print("  Mann-Whitney p=%.4f  high_S=%.4f  low_S=%.4f" % (h2["p"], h2["high_mean"], h2["low_mean"]))
    print("=== H3: Weyl density vs beta0 ===")
    h3 = h3_test(df)
    print("  R2=%.4f  p=%.4f" % (h3["r2"], h3["p"]))


if __name__ == "__main__":
    main()
