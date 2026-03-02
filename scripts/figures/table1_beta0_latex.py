"""
Generate LaTeX table rows for Table 1 (beta0, direct strategy) from results.csv.
Run from project root: python figures/table1_beta0_latex.py
Output: tabular body rows to stdout; optionally writes figures/tab1_beta0_direct.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

# Row order and display names for paper.tex
MESH_ORDER = [
    "sphere",
    "torus",
    "cube",
    "ellipsoid",
    "double_torus",
    "flat_plate",
    "tetrahedron",
    "octahedron",
    "icosahedron",
]
MESH_DISPLAY = {
    "sphere": "sphere",
    "torus": "torus",
    "cube": "cube",
    "ellipsoid": "ellipsoid",
    "double_torus": "double torus",
    "flat_plate": "flat plate",
    "tetrahedron": "tetrahedron",
    "octahedron": "octahedron",
    "icosahedron": "icosahedron",
}
AUDIO_COLS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]


def main():
    import pandas as pd

    csv_path = root / "data" / "results" / "results.csv"
    if not csv_path.exists():
        print("Error: data/results/results.csv not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    sub = df[df["strategy"] == "direct"].pivot_table(
        index="mesh", columns="audio", values="beta0", aggfunc="first"
    )

    # Ensure all audio columns exist
    for c in AUDIO_COLS:
        if c not in sub.columns:
            sub[c] = None

    # Optional: save CSV
    out_csv = root / "figures" / "tab1_beta0_direct.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.reindex(index=MESH_ORDER, columns=AUDIO_COLS).to_csv(out_csv)
    print("Saved", out_csv, file=sys.stderr)

    # LaTeX rows
    lines = []
    for mesh in MESH_ORDER:
        label = MESH_DISPLAY[mesh]
        row_vals = []
        if mesh not in sub.index:
            row_vals = ["--"] * 7
        else:
            for a in AUDIO_COLS:
                val = sub.loc[mesh, a] if a in sub.columns else None
                if pd.isna(val):
                    row_vals.append("--")
                else:
                    row_vals.append(str(int(val)))
        line = f"{label:12} & " + " & ".join(row_vals) + " \\\\"
        lines.append(line)

    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
