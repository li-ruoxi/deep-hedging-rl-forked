#!/usr/bin/env bash
set -euo pipefail

# Build arXiv-like PDF using Pandoc defaults file
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building paper.pdf via Pandoc..."
pandoc --defaults "$ROOT_DIR/pandoc.yaml" \
  --lua-filter "$ROOT_DIR/filters/latex_fix_equations.lua" \
  --lua-filter "$ROOT_DIR/filters/latex_inline_tables.lua" \
  --lua-filter "$ROOT_DIR/filters/latex_code_listing.lua" \
  --lua-filter "$ROOT_DIR/filters/latex_figure_floats.lua"

echo "OK -> $ROOT_DIR/paper.pdf"
