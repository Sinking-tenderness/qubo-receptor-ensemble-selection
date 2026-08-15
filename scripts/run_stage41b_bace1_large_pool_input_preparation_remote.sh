#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
config="configs/stage41b_bace1_large_pool_redocking_input_preparation.json"
run_dir="results/runs/stage41b_bace1_large_pool_redocking_input_preparation"
summary="data/stage41b_bace1_large_pool_redocking_input_preparation_summary.json"
receptors="data/processed/stage41b_bace1_large_pool_prepared_receptor_manifest.csv"
cases="data/processed/stage41b_bace1_large_pool_cognate_redocking_case_manifest.csv"
box="data/stage41b_bace1_large_pool_common_box.json"
core_archive="$output_root/stage41b_bace1_large_pool_redocking_inputs_core_v1.tar.gz"
diagnostic_archive="$output_root/stage41b_bace1_large_pool_redocking_inputs_diagnostics_v1.tar.gz"
failure_archive="$output_root/stage41b_bace1_large_pool_redocking_inputs_failed_runtime_v1.tar.gz"

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

source "$(conda info --base)/etc/profile.d/conda.sh"

probe_environment() {
  local environment_name="$1"
  conda run -n "$environment_name" python -c \
    'import gemmi, meeko, numpy, prody, rdkit, scipy; assert numpy.__version__.startswith("1.26.")' \
    >/dev/null 2>&1
}

environment_name="${STAGE41B_ENV_NAME:-}"
if [[ -n "$environment_name" ]]; then
  probe_environment "$environment_name"
else
  for candidate in qubo-unidock-stage13 qubo-unidock-stage08 qubo-unidock-stage07 qubo-bace1-stage41b; do
    if probe_environment "$candidate"; then
      environment_name="$candidate"
      break
    fi
  done
fi

if [[ -z "$environment_name" ]]; then
  environment_name="qubo-bace1-stage41b"
  conda env create -f environment/stage41b_bace1_input_preparation.yml
fi
conda activate "$environment_name"
probe_environment "$environment_name"
echo "environment_name=$environment_name"

mkdir -p "$run_dir/environment" "$output_root"
python --version > "$run_dir/environment/python_version.txt" 2>&1
conda list | grep -E '^(python|numpy|scipy|rdkit|gemmi|prody|meeko)[[:space:]]' \
  > "$run_dir/environment/conda_core_packages.txt" || true
conda list --explicit > "$run_dir/environment/conda_explicit.txt"

python -m scripts.prepare_stage41b_bace1_large_pool_redocking_inputs \
  --config "$config" --audit-only

python -m scripts.prepare_stage41b_bace1_large_pool_redocking_inputs \
  --config "$config" --resume

mapfile -t core_runtime_files < <(
  find "$run_dir/preparation" -type f \
    \( -name 'case_preparation_summary.json' \
       -o -name 'case_preparation_failure.json' \
       -o -name 'summary.json' \
       -o -name 'alignment_summary.json' \
       -o -name '*_receptor.pdbqt' \
       -o -name '*_ligand.pdbqt' \
       -o -name '*_common_frame.sdf' \) | sort
)

tar -czf "$core_archive" \
  "$config" "$summary" "$receptors" "$cases" "$box" \
  "$run_dir/environment" "${core_runtime_files[@]}"

tar -czf "$diagnostic_archive" \
  "$config" "$summary" "$receptors" "$cases" "$box" "$run_dir"

sync
sha256sum "$core_archive" "$diagnostic_archive"
