#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
unidock_bin="${UNIDOCK_BIN:-unidock}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
config="configs/stage42c_bace1_train266_unidock113_production.json"
run_dir="results/runs/stage42c_bace1_train266_unidock113_production"
progress="$run_dir/progress.json"
summary="$run_dir/summary.json"
audit="data/stage42c_bace1_train266_unidock113_production_audit.json"
scores="$run_dir/scores.csv"
batches="$run_dir/batch_runs.csv"
median_matrix="$run_dir/primary_median_score_matrix.csv"
minimum_matrix="$run_dir/sensitivity_minimum_score_matrix.csv"
receptor_manifest="data/processed/stage42c_bace1_redocking_qualified34_receptor_manifest.csv"
preregistration="configs/stage42_bace1_redocking_qualified_development_preregistration.json"
input_audit="data/stage42b_bace1_train266_unidock_input_independent_audit.json"
ligand_manifest="data/processed/stage42b_bace1_train266_unidock_pdbqt_manifest.csv"
ligand_summary="data/stage42b_bace1_train266_unidock_input_summary.json"

partition="${PARTITION_ID:-${SEED_IDS:-all}}"
partition="${partition//,/+}"
partition="${partition//[^A-Za-z0-9_+-]/_}"
core_archive="$output_root/stage42c_bace1_train266_unidock113_production_core_v1.tar.gz"
diagnostic_archive="$output_root/stage42c_bace1_train266_unidock113_production_diagnostics_v1.tar.gz"
checkpoint_archive="$output_root/stage42c_bace1_train266_unidock113_${partition}_checkpoint_v1.tar.gz"
failure_archive="$output_root/stage42c_bace1_train266_unidock113_${partition}_failed_diagnostics_v1.tar.gz"

on_exit() {
  status=$?
  sync || true
  if [[ "$status" -ne 0 ]]; then
    failure_paths=("$config")
    [[ -f "$progress" ]] && failure_paths+=("$progress")
    [[ -d "$run_dir" ]] && failure_paths+=("$run_dir")
    tar -czf "$failure_archive" "${failure_paths[@]}" 2>/dev/null || true
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
conda list | grep -E '^(python|numpy|scipy|rdkit|meeko|unidock)[[:space:]]' \
  > "$run_dir/environment/conda_core_packages.txt" || true
conda list --explicit > "$run_dir/environment/conda_explicit.txt"

runner=(
  "$python_bin"
  -m
  scripts.experimental.unidock.run_stage42c_bace1_train266_production
  --config "$config"
)

"${runner[@]}" --audit-only

runner+=(--unidock "$unidock_bin" --resume)
if [[ "${FINALIZE_ONLY:-0}" == "1" ]]; then
  runner+=(--finalize-only)
fi
if [[ -n "${SEED_IDS:-}" ]]; then
  IFS=',' read -r -a requested_seeds <<< "$SEED_IDS"
  for seed_id in "${requested_seeds[@]}"; do
    runner+=(--seed-id "$seed_id")
  done
fi
if [[ -n "${RECEPTOR_IDS:-}" ]]; then
  IFS=',' read -r -a requested_receptors <<< "$RECEPTOR_IDS"
  for receptor_id in "${requested_receptors[@]}"; do
    runner+=(--receptor-id "$receptor_id")
  done
fi

"${runner[@]}"

if [[ -f "$summary" ]]; then
  "$python_bin" -m scripts.experimental.unidock.audit_stage42c_bace1_train266_production \
    --config "$config"

  mapfile -t core_batch_files < <(
    find "$run_dir/batches" -type f \
      \( -name 'scores.csv' -o -name 'batch_summary.json' \
         -o -name 'unidock.log' \) | sort
  )

  tar -czf "$core_archive" \
    "$config" "$summary" "$audit" "$progress" "$scores" "$batches" \
    "$median_matrix" "$minimum_matrix" "$receptor_manifest" \
    "$preregistration" "$input_audit" "$ligand_manifest" "$ligand_summary" \
    "$run_dir/environment" "${core_batch_files[@]}"

  tar -czf "$diagnostic_archive" \
    "$config" "$audit" "$receptor_manifest" "$preregistration" "$input_audit" \
    "$ligand_manifest" "$ligand_summary" "$run_dir"

  sync
  sha256sum "$core_archive" "$diagnostic_archive"
else
  tar -czf "$checkpoint_archive" \
    "$config" "$progress" "$run_dir/environment" "$run_dir/batches"
  sync
  sha256sum "$checkpoint_archive"
fi
