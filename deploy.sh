#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Generate audio and data for web demo (otherwise "Audio file not found" after deploy)
echo "Exporting data and audio for web..."
if ! python -m experiment.export_for_web; then
  echo "Error: export_for_web failed. Run ./step4.sh first, then try again." >&2
  exit 1
fi

cd "$ROOT/webapp"
npm run deploy
