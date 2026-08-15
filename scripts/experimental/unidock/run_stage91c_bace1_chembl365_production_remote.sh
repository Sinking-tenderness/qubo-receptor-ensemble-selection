#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
environment_name="${CONDA_ENV:-qubo-unidock-stage08}"
config="configs/stage91c_bace1_chembl365_unidock113_production.json"
run_dir="results/runs/stage91c_bace1_chembl365_unidock113_production"
progress="$run_dir/progress.json"
summary="$run_dir/summary.json"
audit="data/stage91c_bace1_chembl365_unidock113_production_audit.json"
scores="$run_dir/scores.csv"
batches="$run_dir/batch_runs.csv"
median_matrix="$run_dir/primary_median_score_matrix.csv"
minimum_matrix="$run_dir/sensitivity_minimum_score_matrix.csv"
receptor_manifest="data/processed/stage42c_bace1_redocking_qualified34_receptor_manifest.csv"
ligand_manifest="data/processed/stage91b_bace1_chembl365_unidock_pdbqt_manifest.csv"
ligand_summary="data/stage91b_bace1_chembl365_unidock_input_summary.json"
input_audit="data/stage91b_bace1_chembl365_unidock_input_audit.json"
preregistration="configs/stage91c_bace1_group_robust_development_docking_preregistration.json"
core_archive="$output_root/stage91c_bace1_chembl365_unidock113_production_core_v1.tar.gz"
diagnostic_archive="$output_root/stage91c_bace1_chembl365_unidock113_production_diagnostics_v1.tar.gz"
failure_archive="$output_root/stage91c_bace1_chembl365_unidock113_production_failed_diagnostics_v1.tar.gz"

on_exit() {
  status=$?
  sync || true
  if [[ "$status" -ne 0 ]]; then
    paths=("$config")
    [[ -f "$progress" ]] && paths+=("$progress")
    [[ -d "$run_dir" ]] && paths+=("$run_dir")
    tar -czf "$failure_archive" "${paths[@]}" 2>/dev/null || true
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
conda activate "$environment_name"
python_bin="$(command -v python)"
unidock_bin="${UNIDOCK_BIN:-$(command -v unidock)}"

echo "environment_name=$environment_name"
echo "python_bin=$python_bin"
echo "unidock_bin=$unidock_bin"
nvidia-smi
"$unidock_bin" --version

mkdir -p "$run_dir/environment" "$output_root"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv > "$run_dir/environment/nvidia_smi.csv"
nvidia-smi > "$run_dir/environment/nvidia_smi_full.txt"
"$unidock_bin" --version > "$run_dir/environment/unidock_version.txt" 2>&1
conda list | grep -E '^(python|numpy|scipy|rdkit|meeko|unidock)[[:space:]]' \
  > "$run_dir/environment/conda_core_packages.txt" || true
conda list --explicit > "$run_dir/environment/conda_explicit.txt"

runner=(
  "$python_bin"
  -m scripts.experimental.unidock.run_stage91c_bace1_chembl365_production
  --config "$config"
)

"${runner[@]}" --audit-only
"${runner[@]}" --unidock "$unidock_bin" --resume

"$python_bin" -m scripts.experimental.unidock.audit_stage91c_bace1_chembl365_production \
  --config "$config"

mapfile -t core_batch_files < <(
  find "$run_dir/batches" -type f \
    \( -name 'scores.csv' -o -name 'batch_summary.json' \
       -o -name 'unidock.log' \) | sort
)

tar -czf "$core_archive" \
  "$config" "$preregistration" "$summary" "$audit" "$progress" \
  "$scores" "$batches" "$median_matrix" "$minimum_matrix" \
  "$receptor_manifest" "$ligand_manifest" "$ligand_summary" "$input_audit" \
  "$run_dir/environment" "${core_batch_files[@]}"

tar -czf "$diagnostic_archive" \
  "$config" "$preregistration" "$receptor_manifest" "$ligand_manifest" \
  "$ligand_summary" "$input_audit" "$audit" "$run_dir"

sync
sha256sum "$core_archive" "$diagnostic_archive"
