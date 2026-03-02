#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python -c "import numpy, pandas" || { echo "pip install pandas"; exit 1; }
[ -f data/eigen/sphere.npz ] || { echo "Run ./step1.sh first"; exit 1; }
python -m experiment.run_all
python -m experiment.hypothesis_test
python -m experiment.export_for_web 2>/dev/null || true
echo "Step 4 done."