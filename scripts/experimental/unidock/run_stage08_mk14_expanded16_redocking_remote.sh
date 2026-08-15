#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
unidock_bin="${UNIDOCK_BIN:-unidock}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
config="configs/stage08_mk14_expanded16_unidock_redocking.json"
run_dir="results/runs/stage08_mk14_expanded16_unidock_redocking"
summary="data/stage08_mk14_expanded16_unidock_redocking_summary.json"
audit="data/stage08_mk14_expanded16_unidock_redocking_audit.json"
new_manifest="data/processed/stage08_mk14_expanded16_new_receptor_manifest.csv"
combined_manifest="data/processed/stage08_mk14_expanded16_receptor_manifest.csv"
redocking_results="data/processed/stage08_mk14_expanded16_unidock_redocking_results.csv"
core_archive="$output_root/stage08_mk14_expanded16_unidock_redocking_core_v1.tar.gz"
diagnostic_archive="$output_root/stage08_mk14_expanded16_unidock_redocking_diagnostics_v1.tar.gz"
failure_archive="$output_root/stage08_mk14_expanded16_unidock_redocking_failed_diagnostics_v1.tar.gz"

on_exit() {
  status=$?
  sync || true
  if [[ "$status" -ne 0 && -d "$run_dir" ]]; then
    tar -czf "$failure_archive" "$run_dir" "$config" 2>/dev/null || true
    sha256sum "$failure_archive" 2>/dev/null || true
  fi
  if [[ "${AUTO_POWEROFF:-0}" == "1" ]]; then
    echo "AUTO_POWEROFF=1; requesting poweroff after exit status $status"
    shutdown -h now || poweroff || true
  fi
  return "$status"
}
trap on_exit EXIT

mkdir -p "$run_dir/environment" "$output_root"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv > "$run_dir/environment/nvidia_smi.csv"
nvidia-smi > "$run_dir/environment/nvidia_smi_full.txt"
"$unidock_bin" --version > "$run_dir/environment/unidock_version.txt" 2>&1 || true
conda list unidock > "$run_dir/environment/conda_unidock.txt"
conda list | grep -E '^(rdkit|meeko|prody)[[:space:]]' \
  > "$run_dir/environment/conda_preparation_packages.txt" || true
conda list --explicit > "$run_dir/environment/conda_explicit.txt"

"$python_bin" \
  scripts/experimental/unidock/run_stage08_mk14_expanded16_redocking.py \
  --config "$config" \
  --audit-only

"$python_bin" \
  scripts/experimental/unidock/run_stage08_mk14_expanded16_redocking.py \
  --config "$config" \
  --unidock "$unidock_bin" \
  --resume

"$python_bin" \
  scripts/experimental/unidock/audit_stage08_mk14_expanded16_redocking.py \
  --config "$config"

mapfile -t core_runtime_files < <(
  find "$run_dir/preparation" -type f \
    \( -name '*_receptor.pdbqt' -o -name 'preparation_summary.json' \
       -o -name '*_common_frame.sdf' \) | sort
)

tar -czf "$core_archive" \
  "$config" \
  "$summary" \
  "$audit" \
  "$new_manifest" \
  "$combined_manifest" \
  "$redocking_results" \
  "$run_dir/environment" \
  "${core_runtime_files[@]}"

tar -czf "$diagnostic_archive" \
  "$config" \
  "$summary" \
  "$audit" \
  "$new_manifest" \
  "$combined_manifest" \
  "$redocking_results" \
  "$run_dir"

sync
sha256sum "$core_archive" "$diagnostic_archive"
