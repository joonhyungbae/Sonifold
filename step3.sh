#!/usr/bin/env bash
# Step 3: Nodal surface extraction + topology analysis verification
# - Requires Step 1 output (data/eigen/*.npz)
# - Sphere single eigenfunction, white noise vs pure tone β₀, topology API check
#
# Usage: ./step3.sh
# Requires: ./step1.sh done, pip install -r requirements.txt

set -e
cd "$(dirname "$0")"

if ! python -c "import numpy, scipy" 2>/dev/null; then
  echo "Dependency check failed. pip install -r requirements.txt"
  exit 1
fi

if [ ! -f "data/eigen/sphere.npz" ]; then
  echo "Step 1 required: data/eigen/sphere.npz missing. Run ./step1.sh first."
  exit 1
fi

python -m analysis.verify_step3
