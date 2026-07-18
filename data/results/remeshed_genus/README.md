# Remeshed genus 3–6 results

This directory holds outputs for the remeshed genus-sequence pipeline (Section 5.3 genus–β₀ quality check).

## Inputs

- **Source meshes**: `data/experiments/triple_torus_genus3.obj` … `hex_torus_genus6.obj`
- **Target edge length**: mean edge length of `torus_genus1.obj` (isotropic explicit remeshing)

## Generated files

| File | Description |
|------|-------------|
| `*_remeshed.obj` | Remeshed genus 3–6 meshes (PyMeshLab isotropic remeshing) |
| `eigen/*.npz` | Cotangent LB eigenbasis (K=200) for each remeshed mesh |
| `mesh_quality_report.csv` | Quality metrics for remeshed meshes only |
| `results_genus_K50.csv`, `K100`, `K200` | β₀ (mean, std) per mesh × K × stimulus (A1, A2, A3, A5) |
| `K_sensitivity_genus_comparison.csv` | A5 only: β₀ at K=50, 100, 200 per mesh |
| `beta0_old_vs_new.csv` | Comparison: genus, K, stimulus, beta0_old, beta0_new, delta |
| `non_monotonicity_summary.md` | Whether non-monotonicity (A5) persists after remeshing; genus that changed most |

## Scripts (run from project root)

1. `python scripts/remesh_genus_meshes.py` — remesh genus 3–6, verify aspect ratio &lt; 5, area CV &lt; 0.3, Euler = 2−2g  
2. `python scripts/mesh_quality_remeshed.py` — quality report for remeshed meshes (run from `scripts/` or ensure `scripts` on PATH)  
3. `python scripts/run_remeshed_genus_pipeline.py` — eigen K=200, then β₀ for K=50,100,200 × A1,A2,A3,A5  
4. `python scripts/compare_beta0_remeshed.py` — old vs new β₀ CSV and non-monotonicity summary  

Or run all: `bash scripts/run_remeshed_genus_full.sh`

## Interpretation

- If **non-monotonicity persists** after remeshing → consistent with a genuine spectral-geometric effect.
- If it **disappears** → flag in `non_monotonicity_summary.md`; the genus that changed most is reported.
