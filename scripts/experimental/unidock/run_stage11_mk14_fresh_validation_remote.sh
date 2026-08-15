#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
unidock_bin="${UNIDOCK_BIN:-unidock}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
config="configs/stage11_mk14_fresh_validation_unidock113_confirmation.json"
run_dir="results/runs/stage11_mk14_fresh_validation_unidock113_confirmation"
progress="$run_dir/progress.json"
summary="$run_dir/summary.json"
audit="data/stage11_mk14_fresh_validation_unidock113_confirmation_audit.json"
evaluation="data/stage11_mk14_fresh_validation_unidock113_confirmation_result.json"
evaluation_report="reports/stage-11/mk14_fresh_validation_unidock113_result.md"
scores="$run_dir/scores.csv"
batches="$run_dir/batch_runs.csv"
median_matrix="$run_dir/primary_median_score_matrix.csv"
minimum_matrix="$run_dir/sensitivity_minimum_score_matrix.csv"
receptor_manifest="data/processed/stage11_mk14_fresh_validation_six_receptor_manifest.csv"
ligand_manifest="data/processed/stage11_mk14_fresh_validation_unidock_pdbqt_manifest.csv"
preparation_summary="data/stage11_mk14_fresh_validation_unidock_input_summary.json"
rigid_directory="results/runs/stage11_mk14_fresh_validation_unidock_inputs/rigid_macrocycles"

partition="${PARTITION_ID:-${SEED_IDS:-all}}"
partition="${partition//,/+}"
partition="${partition//[^A-Za-z0-9_+-]/_}"
core_archive="$output_root/stage11_mk14_fresh_validation_unidock113_confirmation_core_v1.tar.gz"
diagnostic_archive="$output_root/stage11_mk14_fresh_validation_unidock113_confirmation_diagnostics_v1.tar.gz"
checkpoint_archive="$output_root/stage11_mk14_fresh_validation_unidock113_${partition}_checkpoint_v1.tar.gz"
failure_archive="$output_root/stage11_mk14_fresh_validation_unidock113_${partition}_failed_diagnostics_v1.tar.gz"

on_exit() {
  status=$?
  sync || true
  if [[ "$status" -ne 0 ]]; then
    failure_paths=("$config")
    [[ -f "$preparation_summary" ]] && failure_paths+=("$preparation_summary")
    [[ -f "$receptor_manifest" ]] && failure_paths+=("$receptor_manifest")
    [[ -f "$ligand_manifest" ]] && failure_paths+=("$ligand_manifest")
    [[ -d "$rigid_directory" ]] && failure_paths+=("$rigid_directory")
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

source "$(conda info --base)/etc/profile.d/conda.sh"
environment_name="qubo-unidock-stage08"
if conda env list | awk '{print $1}' | grep -qx "$environment_name"; then
  conda activate "$environment_name"
else
  conda env create -f environment/stage08_unidock_gpu.yml
  conda activate "$environment_name"
fi

mkdir -p "$run_dir/environment" "$output_root"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv > "$run_dir/environment/nvidia_smi.csv"
nvidia-smi > "$run_dir/environment/nvidia_smi_full.txt"
"$unidock_bin" --version > "$run_dir/environment/unidock_version.txt" 2>&1 || true
conda list unidock > "$run_dir/environment/conda_unidock.txt"
conda list meeko > "$run_dir/environment/conda_meeko.txt"
conda list --explicit > "$run_dir/environment/conda_explicit.txt"

if [[ ! -f "$preparation_summary" ]]; then
  "$python_bin" \
    scripts/experimental/unidock/prepare_stage11_mk14_fresh_validation_inputs.py \
    --config "$config" \
    --overwrite
fi

runner=(
  "$python_bin"
  scripts/experimental/unidock/run_stage11_mk14_fresh_validation_confirmation.py
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
  "$python_bin" \
    scripts/experimental/unidock/audit_stage11_mk14_fresh_validation_confirmation.py \
    --config "$config"
  "$python_bin" \
    scripts/experimental/unidock/evaluate_stage11_mk14_fresh_validation_confirmation.py \
    --config "$config" \
    --overwrite

  mapfile -t core_batch_files < <(
    find "$run_dir/batches" -type f \
      \( -name 'scores.csv' -o -name 'batch_summary.json' \
         -o -name 'unidock.log' \) | sort
  )

  tar -czf "$core_archive" \
    "$config" \
    "$preparation_summary" \
    "$receptor_manifest" \
    "$ligand_manifest" \
    "$summary" \
    "$audit" \
    "$evaluation" \
    "$evaluation_report" \
    "$progress" \
    "$scores" \
    "$batches" \
    "$median_matrix" \
    "$minimum_matrix" \
    "$run_dir/environment" \
    "${core_batch_files[@]}"

  tar -czf "$diagnostic_archive" \
    "$config" \
    "$preparation_summary" \
    "$receptor_manifest" \
    "$ligand_manifest" \
    "$audit" \
    "$evaluation" \
    "$evaluation_report" \
    "$run_dir"

  sync
  sha256sum "$core_archive" "$diagnostic_archive"
else
  tar -czf "$checkpoint_archive" \
    "$config" \
    "$preparation_summary" \
    "$receptor_manifest" \
    "$ligand_manifest" \
    "$rigid_directory" \
    "$progress" \
    "$run_dir/environment" \
    "$run_dir/batches"
  sync
  sha256sum "$checkpoint_archive"
fi
