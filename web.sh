#!/usr/bin/env bash
# Start the Sonifold webapp dev server (Vite).
# Usage: ./web.sh
# Requires: run from project root, node/npm. Install once: cd webapp && npm install

set -e
cd "$(dirname "$0")"

if [[ ! -d webapp/node_modules ]]; then
  echo "Installing webapp dependencies (first run)..."
  (cd webapp && npm install)
fi

echo "Starting webapp at http://localhost:5173"
exec npm run dev --prefix webapp
