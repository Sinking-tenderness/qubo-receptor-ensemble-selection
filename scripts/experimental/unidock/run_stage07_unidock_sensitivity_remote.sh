#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
unidock_bin="${UNIDOCK_BIN:-unidock}"
config="configs/stage07_mk14_unidock113_train160_search_sensitivity.json"
run_dir="results/runs/stage07_mk14_unidock113_train160_search_sensitivity"
evidence_dir="$run_dir/environment"
archive="stage07_mk14_unidock113_train160_search_sensitivity_core_v1.tar.gz"

mkdir -p "$evidence_dir"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv > "$evidence_dir/nvidia_smi.csv"
nvidia-smi > "$evidence_dir/nvidia_smi_full.txt"
"$unidock_bin" --version > "$evidence_dir/unidock_version.txt" 2>&1 || true
conda list unidock > "$evidence_dir/conda_unidock.txt"
conda list --explicit > "$evidence_dir/conda_explicit.txt"

"$python_bin" scripts/experimental/unidock/run_stage07_unidock_sensitivity.py \
  --config "$config" \
  --audit-only

"$python_bin" scripts/experimental/unidock/run_stage07_unidock_sensitivity.py \
  --config "$config" \
  --unidock "$unidock_bin" \
  --resume

"$python_bin" scripts/experimental/unidock/evaluate_stage07_unidock_sensitivity.py \
  --config "$config" \
  --overwrite

tar \
  --exclude='*/poses/*' \
  --exclude='*/ligands.index' \
  -czf "$archive" \
  "$config" \
  "$run_dir" \
  data/stage07_mk14_unidock113_train160_search_sensitivity_result.json \
  reports/stage-07/mk14_unidock113_train160_search_sensitivity.md

sha256sum "$archive"
