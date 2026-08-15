#!/usr/bin/env bash
set -euo pipefail

ROOT="${STAGE78_ROOT:-$(pwd)}"
PYTHON_BIN="${STAGE78_PYTHON:-python}"
OUTPUT_ROOT="${STAGE78_OUTPUT_ROOT:-external_results/stage78_advantage2_reverse_annealing_poc}"
ARCHIVE="${STAGE78_PREFLIGHT_ARCHIVE:-stage78_advantage2_preflight_results_v1.tar.gz}"

cd "$ROOT"

if [[ -z "${DWAVE_API_TOKEN:-}" ]]; then
  echo "ERROR: DWAVE_API_TOKEN is not set" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/experimental/quantum/run_stage78_advantage2_reverse_annealing_poc.py \
  --phase local-validate \
  --output-root "$OUTPUT_ROOT"

preflight_args=(
  --phase preflight
  --output-root "$OUTPUT_ROOT"
)
if [[ -n "${STAGE78_SOLVER_NAME:-}" ]]; then
  preflight_args+=(--solver-name "$STAGE78_SOLVER_NAME")
fi

"$PYTHON_BIN" scripts/experimental/quantum/run_stage78_advantage2_reverse_annealing_poc.py \
  "${preflight_args[@]}"

tar -czf "$ARCHIVE" "$OUTPUT_ROOT"
sha256sum "$ARCHIVE"

echo "Stage78 stopped after the no-sampling Leap preflight."
echo "Do not run calibration until the preflight archive has been reviewed."
