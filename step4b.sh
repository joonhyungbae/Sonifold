#!/usr/bin/env bash
# Step 4b: Extended experiments (paper Section 5.4--5.7)
# - Genus 5-point, K sensitivity, temporal phase, genre signature → CSV + figure PDF
#
# Usage: ./step4b.sh
# Requires: experiment eigen and systematic results. If missing, run first:
#   python -m experiment.generate_experiment_meshes
#   USE_GPU=1 EIGEN_TOL=1e-3 python -m experiment.precompute_experiment_eigen --genus-only
#   python -m experiment.run_batch_systematic

set -e
cd "$(dirname "$0")"

python -c "import numpy, scipy" || { echo "pip install -r requirements.txt"; exit 1; }

# Need experiment eigen or systematic results
if [ ! -d "data/experiments/eigen" ] || [ -z "$(ls -A data/experiments/eigen 2>/dev/null)" ]; then
  if [ ! -f "data/results/results_systematic.csv" ]; then
    echo "Extended experiment prerequisites required."
    echo "  python -m experiment.generate_experiment_meshes"
    echo "  USE_GPU=1 EIGEN_TOL=1e-3 python -m experiment.precompute_experiment_eigen --genus-only"
    echo "  python -m experiment.run_batch_systematic"
    exit 1
  fi
fi

echo "[4b] Genus extension (β₀ vs genus)..."
python -m experiment.run_genus_extended
python scripts/figures/fig_genus_beta0.py

echo "[4b] K sensitivity..."
python -m experiment.run_K_sensitivity
python scripts/figures/fig_K_sensitivity.py

echo "[4b] Temporal phase..."
python -m experiment.run_temporal_persistence
python scripts/figures/fig_temporal_persistence.py

echo "[4b] Genre signature..."
python -m experiment.run_genre_signature
python scripts/figures/fig_genre_signature.py

echo "Step 4b done (extended experiments + figures)."
