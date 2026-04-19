#!/usr/bin/env bash
# Full remeshed genus pipeline: remesh -> quality report -> eigen+beta0 -> compare old vs new.
# Run from project root: bash scripts/analysis/run_remeshed_genus_full.sh

set -e
cd "$(dirname "$0")/../.."

echo "=== 1. Remesh genus 3-6 (target edge = torus mean) ==="
python scripts/analysis/remesh_genus_meshes.py

echo "=== 2. Mesh quality analysis (remeshed only) ==="
(cd scripts/analysis && python mesh_quality_remeshed.py)

echo "=== 3. Eigen K=200 + beta0 (K=50,100,200 x A1,A2,A3,A5) ==="
python scripts/analysis/run_remeshed_genus_pipeline.py

echo "=== 4. Compare old vs new beta0, non-monotonicity summary ==="
python scripts/analysis/compare_beta0_remeshed.py

echo "Done. Results in data/results/remeshed_genus/"
