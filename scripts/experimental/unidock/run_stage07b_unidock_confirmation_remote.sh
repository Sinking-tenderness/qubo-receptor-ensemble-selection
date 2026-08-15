#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
unidock_bin="${UNIDOCK_BIN:-unidock}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
config="configs/stage07b_mk14_unidock113_train160_enhanced_confirmation.json"
run_dir="results/runs/stage07b_mk14_unidock113_train160_enhanced_confirmation"
evidence_dir="$run_dir/environment"
core_archive="$output_root/stage07b_mk14_unidock113_train160_enhanced_confirmation_core_v1.tar.gz"
pose_archive="$output_root/stage07b_mk14_unidock113_train160_pose_diagnostics_v1.tar.gz"
pose_summary="data/stage07b_mk14_unidock113_train160_pose_diagnostics_summary.json"

on_exit() {
  status=$?
  sync || true
  if [[ "${AUTO_POWEROFF:-0}" == "1" ]]; then
    echo "AUTO_POWEROFF=1; requesting instance poweroff after exit status $status"
    shutdown -h now || poweroff || true
  fi
  return "$status"
}
trap on_exit EXIT

mkdir -p "$evidence_dir" "$output_root"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv > "$evidence_dir/nvidia_smi.csv"
nvidia-smi > "$evidence_dir/nvidia_smi_full.txt"
"$unidock_bin" --version > "$evidence_dir/unidock_version.txt" 2>&1 || true
conda list unidock > "$evidence_dir/conda_unidock.txt"
conda list --explicit > "$evidence_dir/conda_explicit.txt"

"$python_bin" \
  scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py \
  --config "$config" \
  --audit-only

"$python_bin" \
  scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py \
  --config "$config" \
  --unidock "$unidock_bin" \
  --resume

"$python_bin" \
  scripts/experimental/unidock/evaluate_stage07b_unidock_enhanced_confirmation.py \
  --config "$config" \
  --overwrite

"$python_bin" \
  scripts/experimental/unidock/build_stage07b_pose_diagnostics.py \
  --output "$pose_archive" \
  --summary-output "$pose_summary"

tar \
  --exclude='*/poses/*' \
  --exclude='*/ligands.index' \
  -czf "$core_archive" \
  "$config" \
  "$run_dir" \
  data/stage07b_mk14_unidock113_train160_enhanced_confirmation_result.json \
  "$pose_summary" \
  reports/stage-07/mk14_unidock113_train160_enhanced_confirmation.md

sync
sha256sum "$core_archive" "$pose_archive"
