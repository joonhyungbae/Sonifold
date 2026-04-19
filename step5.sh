#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p webapp/public/data
cp -r data/eigen webapp/public/data/
cd webapp
if [[ ! -d node_modules ]]; then
  npm install
fi
npm run dev
