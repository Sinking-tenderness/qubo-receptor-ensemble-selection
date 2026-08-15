#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "calibration" && "$1" != "confirmation" ) ]]; then
  echo "usage: $0 calibration|confirmation" >&2
  exit 2
fi

PHASE="$1"
ROOT="${STAGE79_ROOT:-$(pwd)}"
PYTHON_BIN="${STAGE79_PYTHON:-python}"
OUTPUT_ROOT="${STAGE79_OUTPUT_ROOT:-external_results/stage79_qci_dirac3_poc}"
ARCHIVE="${STAGE79_DEVICE_ARCHIVE:-stage79_qci_dirac3_${PHASE}_results_v1.tar.gz}"

cd "$ROOT"

"$PYTHON_BIN" scripts/experimental/quantum/run_stage79_qci_dirac3_poc.py \
  --phase "$PHASE" \
  --authorize-qci-device \
  --output-root "$OUTPUT_ROOT"

tar -czf "$ARCHIVE" "$OUTPUT_ROOT"
sha256sum "$ARCHIVE"

echo "Stage79 $PHASE completed and was packaged."
