#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
unidock_bin="${UNIDOCK_BIN:-unidock}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
config="configs/stage51_ppara_large_pool_cognate_redocking.json"
run_dir="results/runs/stage51_ppara_large_pool_cognate_redocking"
summary="data/stage51_ppara_large_pool_cognate_redocking_summary.json"
results="data/processed/stage51_ppara_large_pool_cognate_redocking_results.csv"
gate_results="data/processed/stage51_ppara_large_pool_receptor_gate_results.csv"
core_archive="$output_root/stage51_ppara_large_pool_cognate_redocking_core_v1.tar.gz"
diagnostic_archive="$output_root/stage51_ppara_large_pool_cognate_redocking_diagnostics_v1.tar.gz"
failure_archive="$output_root/stage51_ppara_large_pool_cognate_redocking_failed_runtime_v1.tar.gz"

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
    'import meeko, numpy, rdkit, scipy; assert numpy.__version__.startswith("1.26.")' \
    >/dev/null 2>&1 && \
  conda run -n "$environment_name" unidock --version >/dev/null 2>&1
}

environment_name="${STAGE51_ENV_NAME:-}"
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

mkdir -p "$run_dir/environment" "$output_root"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
  > "$run_dir/environment/nvidia_smi.csv"
nvidia-smi > "$run_dir/environment/nvidia_smi_full.txt"
"$unidock_bin" --version > "$run_dir/environment/unidock_version.txt" 2>&1 || true
conda list unidock > "$run_dir/environment/conda_unidock.txt"
conda list | grep -E '^(python|numpy|scipy|rdkit|meeko|unidock)[[:space:]]' \
  > "$run_dir/environment/conda_core_packages.txt" || true
conda list --explicit > "$run_dir/environment/conda_explicit.txt"

"$python_bin" -m scripts.experimental.unidock.run_stage51_ppara_large_pool_cognate_redocking \
  --config "$config" --audit-only

"$python_bin" -m scripts.experimental.unidock.run_stage51_ppara_large_pool_cognate_redocking \
  --config "$config" --unidock "$unidock_bin" --resume

mapfile -t core_runtime_files < <(
  find "$run_dir/batches" -type f \
    \( -name 'batch_summary.json' -o -name 'scores.csv' \
       -o -name 'unidock.log' -o -path '*/rmsd/summary.json' \
       -o -path '*/rmsd/poses.csv' \) | sort
)

tar -czf "$core_archive" \
  "$config" "$summary" "$results" "$gate_results" \
  "$run_dir/environment" "${core_runtime_files[@]}"

tar -czf "$diagnostic_archive" \
  "$config" "$summary" "$results" "$gate_results" "$run_dir"

sync
sha256sum "$core_archive" "$diagnostic_archive"
