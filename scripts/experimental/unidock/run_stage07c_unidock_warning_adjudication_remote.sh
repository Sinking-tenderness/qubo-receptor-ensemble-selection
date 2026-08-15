#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
unidock_bin="${UNIDOCK_BIN:-unidock}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
config="configs/stage07c_mk14_unidock113_warning_adjudication.json"
run_dir="results/runs/stage07c_mk14_unidock113_warning_adjudication"
evidence_dir="$run_dir/environment"
result_json="data/stage07c_mk14_unidock113_warning_adjudication_result.json"
report_md="reports/stage-07/mk14_unidock113_warning_adjudication.md"
pose_summary="data/stage07c_mk14_unidock113_pose_diagnostics_summary.json"
core_archive="$output_root/stage07c_mk14_unidock113_warning_adjudication_core_v1.tar.gz"
pose_archive="$output_root/stage07c_mk14_unidock113_pose_diagnostics_v1.tar.gz"

on_exit() {
  status=$?
  sync || true
  if [[ "${AUTO_POWEROFF:-0}" == "1" ]]; then
    echo "AUTO_POWEROFF=1; requesting poweroff after exit status $status"
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
  scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py \
  --config "$config" \
  --audit-only

"$python_bin" \
  scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py \
  --config "$config" \
  --unidock "$unidock_bin" \
  --resume

"$python_bin" \
  scripts/experimental/unidock/evaluate_stage07c_unidock_warning_adjudication.py \
  --config "$config" \
  --overwrite

"$python_bin" \
  scripts/experimental/unidock/build_stage07c_pose_diagnostics.py \
  --output "$pose_archive" \
  --summary-output "$pose_summary"

tar -czf "$core_archive" \
  "$config" \
  "$run_dir/scores.csv" \
  "$run_dir/batch_runs.csv" \
  "$run_dir/run_summary.json" \
  "$run_dir/group_metrics.csv" \
  "$run_dir/seed_stability.csv" \
  "$run_dir/replay_comparison.csv" \
  data/processed/stage07c_mk14_unidock113_enhanced_seed012_scores.csv \
  data/processed/stage07c_mk14_unidock113_enhanced_warning_replay_reference.csv \
  data/stage07c_mk14_unidock113_existing_evidence_provenance.json \
  "$result_json" \
  "$pose_summary" \
  "$report_md"

sync
sha256sum "$core_archive" "$pose_archive"
