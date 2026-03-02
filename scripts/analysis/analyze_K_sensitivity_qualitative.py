"""
Check whether K=50 vs K=100 vs K=200 results are qualitatively the same.

Input:
  - data/results/K_sensitivity_genus_comparison.csv (A5, beta0_K50, beta0_K100, beta0_K200)
  - data/results/results_genus_K100.csv, results_genus_K200.csv (optional: rank comparison per stimulus)

Criteria (Conjecture 4.1 defense):
  1. Rank order: Is mesh β₀ rank order identical at K=50, 100, 200?
  2. Torus anomaly: Is genus-1(torus) still low at A1/A2? (K100/K200 CSV needed)
  3. Non-monotonicity: Does β₀ not increase monotonically with genus? (at all K)

Run: python scripts/analysis/analyze_K_sensitivity_qualitative.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = root / "data" / "results"
COMPARISON_CSV = RESULTS_DIR / "K_sensitivity_genus_comparison.csv"
K100_CSV = RESULTS_DIR / "results_genus_K100.csv"
K200_CSV = RESULTS_DIR / "results_genus_K200.csv"


def load_comparison():
    if not COMPARISON_CSV.exists():
        return None
    with open(COMPARISON_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rank_order(rows, key):
    """Return mesh names in beta0 descending rank order."""
    valid = [(r["mesh"], float(r[key])) for r in rows if r.get(key) and r[key].strip()]
    valid.sort(key=lambda x: -x[1])
    return [m for m, _ in valid]


def is_monotonic(rows, key):
    """Return True if beta0 is monotonically increasing when sorted by genus."""
    valid = [(int(r["genus"]), float(r[key])) for r in rows if r.get(key) and r[key].strip()]
    valid.sort(key=lambda x: x[0])
    beta0s = [b for _, b in valid]
    return all(beta0s[i] <= beta0s[i + 1] for i in range(len(beta0s) - 1))


def main():
    rows = load_comparison()
    if not rows:
        print("K_sensitivity_genus_comparison.csv not found. Run scripts/analysis/run_genus_K100_K200.py first.", file=sys.stderr)
        return 1

    print("=== Qualitative comparison: K=50 vs K=100 vs K=200 (A5) ===\n")

    # 1. Rank order
    r50 = rank_order(rows, "beta0_K50")
    r100 = rank_order(rows, "beta0_K100")
    r200 = rank_order(rows, "beta0_K200")
    rank_same_50_100 = r50 == r100
    rank_same_50_200 = r50 == r200
    rank_same_100_200 = r100 == r200
    rank_unchanged = rank_same_50_100 and rank_same_50_200

    print("1. Rank order (by β₀, A5)")
    print("   K=50:  ", " > ".join(r50))
    print("   K=100: ", " > ".join(r100))
    print("   K=200: ", " > ".join(r200))
    print("   Rank order unchanged across K? {} (50=100: {}, 50=200: {}, 100=200: {})".format(
        rank_unchanged, rank_same_50_100, rank_same_50_200, rank_same_100_200))
    print()

    # 2. Non-monotonicity (genus–β₀)
    mono50 = is_monotonic(rows, "beta0_K50")
    mono100 = is_monotonic(rows, "beta0_K100")
    mono200 = is_monotonic(rows, "beta0_K200")
    print("2. Genus–β₀ monotonic? (A5)")
    print("   K=50:  monotonic={}".format(mono50))
    print("   K=100: monotonic={}".format(mono100))
    print("   K=200: monotonic={}".format(mono200))
    print("   Non-monotonic at all K? {} (i.e. qualitatively same pattern)".format(not mono50 and not mono100 and not mono200))
    print()

    # 3. Torus (genus-1) position
    def get_beta0(rows, key):
        for r in rows:
            if r.get("mesh") == "torus_genus1" and r.get(key):
                return float(r[key])
        return None
    t50 = get_beta0(rows, "beta0_K50")
    t100 = get_beta0(rows, "beta0_K100")
    t200 = get_beta0(rows, "beta0_K200")
    print("3. Torus (genus-1) β₀ (A5): K50={}, K100={}, K200={}".format(t50, t100, t200))
    torus_low = (t50 is not None and t100 is not None and t200 is not None and
                 t50 < 5 and t100 < 5 and t200 < 5)
    print("   Anomalously low (<5) at all K? {}".format(torus_low))
    print()

    # Summary
    same = rank_unchanged and (not mono50 and not mono100 and not mono200)
    print("=== Qualitatively same as K=50? ===")
    print("  Rank order unchanged: {}".format(rank_unchanged))
    print("  Non-monotonic (genus–β₀) at all K: {}".format(not mono50 and not mono100 and not mono200))
    print("  Overall: {}".format("Yes" if same else "No (or partially)"))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
