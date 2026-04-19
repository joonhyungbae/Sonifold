"""
Compare old (original meshes) vs new (remeshed) β₀ values.

Loads:
  - Old: data/results/results_genus_K50.csv (if present), results_genus_K100.csv,
         results_genus_K200.csv; for K50 also K_sensitivity_genus_comparison.csv (A5 only).
  - New: data/results/remeshed_genus/results_genus_K50.csv, K100, K200 (genus 3–6 only).

Outputs:
  - data/results/remeshed_genus/beta0_old_vs_new.csv with columns
    genus, K, stimulus, beta0_old, beta0_new, delta.
  - Non-monotonicity summary: if it persists after remeshing (A5) or which genus changed most.

Mapping: old mesh triple_torus_genus3 -> genus 3; new mesh triple_torus_genus3_remeshed -> genus 3.
We compare only genus 3–6 (old and new both have these).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = root / "data" / "results"
REMESHED_DIR = root / "data" / "results" / "remeshed_genus"

# Old CSVs (full 7-mesh genus sequence; we use genus 3–6 rows)
OLD_K50 = RESULTS_DIR / "results_genus_K50.csv"
OLD_K100 = RESULTS_DIR / "results_genus_K100.csv"
OLD_K200 = RESULTS_DIR / "results_genus_K200.csv"
OLD_K_SENSITIVITY = RESULTS_DIR / "K_sensitivity_genus_comparison.csv"

# New (remeshed) CSVs
NEW_K50 = REMESHED_DIR / "results_genus_K50.csv"
NEW_K100 = REMESHED_DIR / "results_genus_K100.csv"
NEW_K200 = REMESHED_DIR / "results_genus_K200.csv"

OUT_CSV = REMESHED_DIR / "beta0_old_vs_new.csv"
OUT_SUMMARY = REMESHED_DIR / "non_monotonicity_summary.md"

# Old mesh_id -> genus (for genus 3–6)
OLD_MESH_TO_GENUS = {
    "triple_torus_genus3": 3,
    "quad_torus_genus4": 4,
    "penta_torus_genus5": 5,
    "hex_torus_genus6": 6,
}

K_VALUES = [50, 100, 200]
STIMULI = ["A1", "A2", "A3", "A5"]


def load_old_by_genus():
    """Returns dict: (genus, K, stimulus) -> beta0_mean. Genus 3–6 only."""
    out = {}
    # K100 and K200: full table
    for path, K in [(OLD_K100, 100), (OLD_K200, 200)]:
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                mesh = row.get("mesh", "")
                g = OLD_MESH_TO_GENUS.get(mesh)
                if g is None:
                    continue
                stim = row.get("stimulus", "")
                try:
                    out[(g, K, stim)] = float(row.get("beta0_mean", float("nan")))
                except (ValueError, TypeError):
                    out[(g, K, stim)] = float("nan")
    # K50: not in a dedicated file; use K_sensitivity for A5 only
    if OLD_K_SENSITIVITY.exists():
        with open(OLD_K_SENSITIVITY, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                mesh = row.get("mesh", "")
                g = OLD_MESH_TO_GENUS.get(mesh)
                if g is None:
                    continue
                try:
                    out[(g, 50, "A5")] = float(row.get("beta0_K50", float("nan")))
                except (ValueError, TypeError):
                    out[(g, 50, "A5")] = float("nan")
    # If results_genus_K50.csv exists (e.g. after a full run that writes it), use it
    if OLD_K50.exists():
        with open(OLD_K50, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                mesh = row.get("mesh", "")
                g = OLD_MESH_TO_GENUS.get(mesh)
                if g is None:
                    continue
                stim = row.get("stimulus", "")
                try:
                    out[(g, 50, stim)] = float(row.get("beta0_mean", float("nan")))
                except (ValueError, TypeError):
                    out[(g, 50, stim)] = float("nan")
    return out


def load_new_by_genus():
    """Returns dict: (genus, K, stimulus) -> beta0_mean."""
    out = {}
    for path, K in [(NEW_K50, 50), (NEW_K100, 100), (NEW_K200, 200)]:
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    g = int(row.get("genus", -1))
                except (ValueError, TypeError):
                    continue
                stim = row.get("stimulus", "")
                try:
                    out[(g, K, stim)] = float(row.get("beta0_mean", float("nan")))
                except (ValueError, TypeError):
                    out[(g, K, stim)] = float("nan")
    return out


def main() -> int:
    old = load_old_by_genus()
    new = load_new_by_genus()
    genera = [3, 4, 5, 6]

    rows = []
    for g in genera:
        for K in K_VALUES:
            for stim in STIMULI:
                beta_old = old.get((g, K, stim), float("nan"))
                beta_new = new.get((g, K, stim), float("nan"))
                delta = float("nan")
                if not (isinstance(beta_old, str) or isinstance(beta_new, str)):
                    try:
                        delta = beta_new - beta_old if (beta_old == beta_old and beta_new == beta_new) else float("nan")
                    except TypeError:
                        pass
                rows.append({
                    "genus": g,
                    "K": K,
                    "stimulus": stim,
                    "beta0_old": beta_old,
                    "beta0_new": beta_new,
                    "delta": delta,
                })

    REMESHED_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["genus", "K", "stimulus", "beta0_old", "beta0_new", "delta"])
        w.writeheader()
        w.writerows(rows)
    print("Saved {}".format(OUT_CSV), file=sys.stderr)

    # Non-monotonicity: A5, by K. Old and new genus-order β₀.
    lines = ["# Non-monotonicity: old vs remeshed (A5)\n"]
    for K in K_VALUES:
        old_vals = [old.get((g, K, "A5"), float("nan")) for g in genera]
        new_vals = [new.get((g, K, "A5"), float("nan")) for g in genera]
        old_mono = all(old_vals[i] <= old_vals[i + 1] for i in range(3) if old_vals[i] == old_vals[i] and old_vals[i + 1] == old_vals[i + 1])
        new_mono = all(new_vals[i] <= new_vals[i + 1] for i in range(3) if new_vals[i] == new_vals[i] and new_vals[i + 1] == new_vals[i + 1])
        # Check with valid numbers only
        old_valid = [v for v in old_vals if v == v]
        new_valid = [v for v in new_vals if v == v]
        old_mono = len(old_valid) == 4 and all(old_valid[i] <= old_valid[i + 1] for i in range(3))
        new_mono = len(new_valid) == 4 and all(new_valid[i] <= new_valid[i + 1] for i in range(3))
        lines.append("## K={}\n".format(K))
        lines.append("- Old (genus 3,4,5,6) β₀: {}\n".format(old_vals))
        lines.append("- Old monotonic? {}\n".format(old_mono))
        lines.append("- New (remeshed) β₀: {}\n".format(new_vals))
        lines.append("- New monotonic? {}\n".format(new_mono))
        if new_mono and not old_mono:
            lines.append("- **Non-monotonicity disappeared after remeshing.**\n")
        elif not new_mono and not old_mono:
            lines.append("- **Non-monotonicity persists:** consistent with genuine spectral-geometric effect.\n")
        elif not new_mono and old_mono:
            lines.append("- **Non-monotonicity appeared after remeshing.**\n")
        lines.append("\n")

    # Which genus changed most (by mean |delta| over K and stimuli)
    by_genus_delta = {}
    for r in rows:
        g = r["genus"]
        d = r["delta"]
        if d != d:
            continue
        by_genus_delta.setdefault(g, []).append(abs(d))
    if by_genus_delta:
        mean_abs_delta = {g: sum(v) / len(v) for g, v in by_genus_delta.items()}
        most_changed = max(mean_abs_delta.items(), key=lambda x: x[1])
        lines.append("## Genus that changed most (mean |Δβ₀|)\n")
        lines.append("Genus {} (mean |delta| = {:.2f}).\n".format(most_changed[0], most_changed[1]))

    OUT_SUMMARY.write_text("".join(lines), encoding="utf-8")
    print("Saved {}".format(OUT_SUMMARY), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
