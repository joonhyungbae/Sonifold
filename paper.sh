#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$ROOT_DIR/sonifold_paper"

if [ ! -d "$PAPER_DIR" ]; then
  echo "Error: sonifold_paper directory not found."
  exit 1
fi

if ! git -C "$PAPER_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: sonifold_paper is not a git repository."
  exit 1
fi

if [ -z "$(git -C "$PAPER_DIR" status --porcelain)" ]; then
  echo "No changes in sonifold_paper to commit."
  exit 0
fi

COMMIT_MSG="${*:-chore(paper): update sonifold_paper $(date '+%Y-%m-%d %H:%M:%S')}"
BRANCH_NAME="$(git -C "$PAPER_DIR" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
if [ -z "$BRANCH_NAME" ]; then
  BRANCH_NAME="$(git -C "$PAPER_DIR" branch --show-current 2>/dev/null || true)"
fi

if [ -z "$BRANCH_NAME" ]; then
  echo "Error: could not determine current branch in sonifold_paper."
  echo "Fix: checkout or create a branch first (e.g., git -C ./sonifold_paper switch -c main)."
  exit 1
fi

git -C "$PAPER_DIR" add -A
GIT_COMMIT_BIN="git"
"$GIT_COMMIT_BIN" -C "$PAPER_DIR" commit -m "$COMMIT_MSG"
GIT_PUSH_BIN="git"
"$GIT_PUSH_BIN" -C "$PAPER_DIR" push origin "$BRANCH_NAME"

echo "Done: committed and pushed sonifold_paper to $BRANCH_NAME"
