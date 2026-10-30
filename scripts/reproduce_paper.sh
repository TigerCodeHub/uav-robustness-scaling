#!/usr/bin/env bash
# Regenerate every table and in-text number of the paper from the data archive, and check
# them against the committed copies in paper_outputs/.
#
# Usage:  scripts/reproduce_paper.sh [DATA_DIR] [OUT_DIR]     (defaults: data/results, outputs)
#   PYTHON=python3   (default: python)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
DATA="${1:-data/results}"
OUT="${2:-outputs}"
[ -d "$DATA" ] || { echo "$DATA not found: download and extract the data archive (see DATA_MANIFEST.md)"; exit 1; }

if [ "$DATA" = data/results ]; then
  sha256sum --quiet -c DATA_CHECKSUMS.txt
  echo "data checksums OK ($(wc -l < DATA_CHECKSUMS.txt) files)"
fi

"$PY" analysis/make_tables.py "$DATA" "$OUT/tables"
"$PY" analysis/heldout.py "$DATA" "$OUT/tables"
"$PY" analysis/prose_numbers.py "$DATA" "$OUT/prose_numbers.txt"
# The timing table is formatted from the committed measurements in paper_outputs/inference_timing/
# (re-measure with scripts/inference_timing.py; new timings are similar, not identical).
"$PY" analysis/inference_table.py paper_outputs/inference_timing "$OUT/tables"

if diff -r -x inference_timing paper_outputs "$OUT"; then
  echo "all outputs identical to paper_outputs/"
else
  echo "outputs differ from paper_outputs/ (see diff above)"; exit 1
fi
