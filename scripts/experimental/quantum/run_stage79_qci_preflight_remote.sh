#!/usr/bin/env bash
set -euo pipefail

ROOT="${STAGE79_ROOT:-$(pwd)}"
PYTHON_BIN="${STAGE79_PYTHON:-python}"
OUTPUT_ROOT="${STAGE79_OUTPUT_ROOT:-external_results/stage79_qci_dirac3_poc}"
ARCHIVE="${STAGE79_PREFLIGHT_ARCHIVE:-stage79_qci_dirac3_preflight_results_v1.tar.gz}"

cd "$ROOT"

"$PYTHON_BIN" scripts/experimental/quantum/run_stage79_qci_dirac3_poc.py \
  --phase validate \
  --output-root "$OUTPUT_ROOT"

"$PYTHON_BIN" scripts/experimental/quantum/run_stage79_qci_dirac3_poc.py \
  --phase preflight \
  --output-root "$OUTPUT_ROOT"

tar -czf "$ARCHIVE" "$OUTPUT_ROOT"
sha256sum "$ARCHIVE"

echo "Stage79 stopped after allocation-only preflight; no Dirac-3 job was submitted."
