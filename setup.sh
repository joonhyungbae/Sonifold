#!/usr/bin/env bash
# Create Sonifold conda environment and install dependencies
# Usage: ./setup.sh
# Then: conda activate sonifold && ./step1.sh

set -e
cd "$(dirname "$0")"

ENV_NAME="sonifold"
PYTHON_VERSION="3.10"

if ! command -v conda &>/dev/null; then
  echo "conda not found. Install Anaconda or Miniconda and run again."
  exit 1
fi

echo "[setup] Creating conda env '$ENV_NAME' (python=${PYTHON_VERSION}) ..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "[setup] Installing dependencies (pip install -r requirements.txt) ..."
conda run -n "$ENV_NAME" pip install -r requirements.txt

echo ""
echo "Setup done. To run Step 1:"
echo "  conda activate $ENV_NAME"
echo "  ./step1.sh"
echo ""
echo "Or in one line: conda activate $ENV_NAME && ./step1.sh"
