#!/usr/bin/env bash
# Step 1: Python eigenfunction pipeline
# - Compute 50 LB eigenpairs for 8 meshes -> data/eigen/*.npz, data/eigen/*.json
# - Default: parallel per mesh. Use --sequential for sequential
#
# Usage: ./step1.sh [--sequential]
#   With GPU (default): ./step1.sh   (requires CuPy + CUDA)
#   CPU only:           USE_GPU=0 python -m precompute.run_step1 [--sequential]
#
# Requires: run from project root, pip install -r requirements.txt
# GPU: cupy-cuda12x (CUDA 12) or cupy-cuda11x (CUDA 11). Without GPU, omit cupy and run with USE_GPU=0.

set -e
cd "$(dirname "$0")"

# Base deps (always required)
_missing=
for pkg in numpy scipy trimesh gpytoolbox tqdm; do
  if ! python -c "import $pkg" 2>/dev/null; then
    _missing="$_missing $pkg"
  fi
done
if [ -n "$_missing" ]; then
  echo "Dependency check failed (missing:$_missing )."
  echo "  pip install -r requirements.txt"
  exit 1
fi

# GPU path: require CuPy and a visible GPU unless USE_GPU=0
if [ "${USE_GPU:-1}" = "1" ]; then
  for pkg in cupy; do
    if ! python -c "import $pkg" 2>/dev/null; then
      echo "GPU mode (USE_GPU=1) requires CuPy. Install cupy-cuda12x or cupy-cuda11x, or run with: USE_GPU=0 python -m precompute.run_step1 $*"
      exit 1
    fi
  done
  if ! python -c "
import sys
try:
  import cupy as cp
  import cupyx.scipy.sparse
  import cupyx.scipy.sparse.linalg
  if hasattr(cp, 'cuda') and cp.cuda.runtime.getDeviceCount() < 1:
    raise RuntimeError('No GPU found. Check with nvidia-smi.')
except ImportError as e:
  print('CuPy import failed:', e, file=sys.stderr)
  print('  CUDA 12: pip install cupy-cuda12x', file=sys.stderr)
  print('  CUDA 11: pip install cupy-cuda11x', file=sys.stderr)
  print('  Or run without GPU: USE_GPU=0 python -m precompute.run_step1', file=sys.stderr)
  sys.exit(1)
except Exception as e:
  print('CuPy/GPU error:', e, file=sys.stderr)
  print('  Check GPU with nvidia-smi. Or run with USE_GPU=0 for CPU.', file=sys.stderr)
  sys.exit(1)
" 2>&1; then
    exit 1
  fi
  export USE_GPU=1
else
  export USE_GPU=0
  echo "Step 1 running in CPU mode (USE_GPU=0). This may be slow for many meshes."
fi

python -m precompute.run_step1 "$@"
