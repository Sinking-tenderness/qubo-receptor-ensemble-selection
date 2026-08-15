#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
config="configs/stage52a_ppara_train374_unidock_input_preparation.json"
run_dir="results/runs/stage52a_ppara_train374_unidock_inputs"
summary="data/stage52a_ppara_train374_unidock_input_summary.json"
manifest="data/processed/stage52a_ppara_train374_unidock_pdbqt_manifest.csv"
core_archive="$output_root/stage52a_ppara_train374_unidock_inputs_core_v1.tar.gz"
diagnostic_archive="$output_root/stage52a_ppara_train374_unidock_inputs_diagnostics_v1.tar.gz"
failure_archive="$output_root/stage52a_ppara_train374_unidock_inputs_failed_runtime_v1.tar.gz"

on_exit() {
  status=$?
  sync || true
  if [[ "$status" -ne 0 ]]; then
    tar -czf "$failure_archive" \
      configs/stage52*.json \
      data/stage51b_ppara_redocking_bias_diagnostic_*.json \
      data/stage49_ppara_ligand_panel_allocation_summary.json \
      data/processed/stage49_ppara_train374_ligand_manifest.csv \
      "$run_dir" 2>/dev/null || true
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
    'import importlib.metadata; assert importlib.metadata.version("rdkit") == "2026.3.1"; assert importlib.metadata.version("meeko") == "0.7.1"' \
    >/dev/null 2>&1
}

environment_name="${STAGE52_ENV_NAME:-}"
if [[ -n "$environment_name" ]]; then
  probe_environment "$environment_name"
else
  for candidate in qubo-unidock-stage08 qubo-unidock-stage13 qubo-unidock-stage07; do
    if probe_environment "$candidate"; then
      environment_name="$candidate"
      break
    fi
  done
fi
if [[ -z "$environment_name" ]]; then
  environment_name="qubo-unidock-stage08"
  conda env create -f environment/stage08_unidock_gpu.yml
fi
conda activate "$environment_name"
probe_environment "$environment_name"
echo "environment_name=$environment_name"

python_bin="${PYTHON_BIN:-python}"
mkdir -p "$run_dir/environment" "$output_root"
"$python_bin" --version > "$run_dir/environment/python_version.txt" 2>&1
conda list | grep -E '^(python|numpy|scipy|rdkit|meeko|unidock)[[:space:]]' \
  > "$run_dir/environment/conda_core_packages.txt" || true
conda list --explicit > "$run_dir/environment/conda_explicit.txt"

"$python_bin" -m scripts.experimental.unidock.prepare_development_ligand_inputs \
  --config "$config" --root . --audit-only

"$python_bin" -m scripts.experimental.unidock.prepare_development_ligand_inputs \
  --config "$config" --root . --resume

tar -czf "$core_archive" \
  configs/stage52_ppara_posthoc_exploratory_development_preregistration.json \
  "$config" "$summary" "$manifest" \
  "$run_dir/sdf" "$run_dir/pdbqt" "$run_dir/checkpoints" \
  "$run_dir/environment"

tar -czf "$diagnostic_archive" \
  configs/stage52*.json \
  data/stage51b_ppara_redocking_bias_diagnostic_*.json \
  data/stage49_ppara_ligand_panel_allocation_summary.json \
  data/processed/stage49_ppara_train374_ligand_manifest.csv \
  "$summary" "$manifest" "$run_dir"

sync
sha256sum "$core_archive" "$diagnostic_archive"
