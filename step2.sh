#!/usr/bin/env bash
# Step 2: Audio analysis + mapping verification
# - 7 audio × 3 mapping strategies -> 21 coefficient vectors
# - 440Hz FFT, Direct single activation, white noise uniform activation check
#
# Usage: ./step2.sh
# Requires: run from project root, pip install -r requirements.txt (numpy, scipy, etc.)

set -e
cd "$(dirname "$0")"

if ! python -c "import numpy, scipy" 2>/dev/null; then
  echo "Dependency check failed. pip install -r requirements.txt"
  exit 1
fi

python -m audio.verify_step2
