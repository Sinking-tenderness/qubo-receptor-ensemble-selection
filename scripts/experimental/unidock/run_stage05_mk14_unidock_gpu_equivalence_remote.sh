#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
unidock_bin="${UNIDOCK_BIN:-unidock}"
config="configs/stage05_mk14_unidock_train160_gpu_equivalence.json"
run_dir="results/runs/stage05_mk14_unidock_train160_gpu_equivalence"
evidence_dir="$run_dir/environment"
archive="stage05_mk14_unidock_train160_gpu_equivalence_core_results_v1.tar.gz"

mkdir -p "$evidence_dir"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv > "$evidence_dir/nvidia_smi.csv"
nvidia-smi > "$evidence_dir/nvidia_smi_full.txt"
"$unidock_bin" --version > "$evidence_dir/unidock_version.txt" 2>&1 || true
conda list unidock > "$evidence_dir/conda_unidock.txt"
conda list --explicit > "$evidence_dir/conda_explicit.txt"

"$python_bin" scripts/experimental/unidock/run_unidock_gpu_equivalence.py \
  --config "$config" \
  --audit-only

"$python_bin" scripts/experimental/unidock/run_unidock_gpu_equivalence.py \
  --config "$config" \
  --unidock "$unidock_bin" \
  --resume

"$python_bin" scripts/experimental/unidock/audit_unidock_gpu_equivalence.py \
  --config "$config" \
  --overwrite

tar --exclude='*/poses/*' --exclude='*/ligands.index' \
  -czf "$archive" \
  "$config" \
  "$run_dir/gpu_scores.csv" \
  "$run_dir/gpu_batch_runs.csv" \
  "$run_dir/gpu_run_summary.json" \
  "$run_dir/equivalence_pairwise.csv" \
  "$run_dir/equivalence_group_metrics.csv" \
  "$run_dir/equivalence_summary.json" \
  "$run_dir/environment" \
  "$run_dir/batches"

sha256sum "$archive"
