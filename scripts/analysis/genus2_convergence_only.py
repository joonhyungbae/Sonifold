"""Run eigenvalue convergence test for genus-2 only (quick H check)."""
from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

# Reuse convergence test logic for one mesh
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "conv", root / "scripts" / "figures" / "eigenvalue_convergence_test_all_genus.py"
)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

def main():
    mesh_id, genus = "double_torus_genus2", 2
    resolutions = conv.build_resolutions(mesh_id, genus)
    if not resolutions:
        print("No resolutions (mesh skip or load failed).", file=sys.stderr)
        return 1
    rows = []
    for res_label, V, F in resolutions:
        row = conv.run_one_resolution(mesh_id, genus, res_label, V, F)
        if row:
            rows.append(row)
            print(f"  {mesh_id} {res_label}: n={row['num_vertices']} H={row['spectral_entropy_H']} mean_β0={row['mean_beta0']}")
    if len(rows) < 2:
        return 1
    # H change %
    H_vals = [r["spectral_entropy_H"] for r in rows]
    H_min, H_max = min(H_vals), max(H_vals)
    ref = H_vals[0]
    change_pct = 100.0 * (H_max - H_min) / ref if ref else 0
    print(f"\nGenus 2: H in [{H_min:.6f}, {H_max:.6f}], max change vs first = {change_pct:.2f}%")
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
