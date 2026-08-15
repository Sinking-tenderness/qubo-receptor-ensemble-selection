#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
environment_name="${CONDA_ENV:-qubo-unidock-stage08}"
config="configs/stage91b_bace1_chembl365_unidock_input_preparation.json"
run_dir="results/runs/stage91b_bace1_chembl365_unidock_inputs"
core_archive="$output_root/stage91b_bace1_chembl365_unidock_inputs_core_v1.tar.gz"
diagnostic_archive="$output_root/stage91b_bace1_chembl365_unidock_inputs_diagnostics_v1.tar.gz"
failure_archive="$output_root/stage91b_bace1_chembl365_unidock_inputs_failed_runtime_v1.tar.gz"

on_exit() {
  status=$?
  sync || true
  if [[ "$status" -ne 0 ]]; then
    paths=("$config")
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

echo "environment_name=$environment_name"
"$python_bin" --version
"$python_bin" - <<'PY'
import importlib.metadata
for name in ("rdkit", "meeko"):
    print(f"{name}={importlib.metadata.version(name)}")
PY
command -v unidock
unidock --version

mkdir -p "$run_dir/environment" "$output_root"
conda list --explicit > "$run_dir/environment/conda_explicit.txt"
conda list | grep -E '^(python|numpy|scipy|rdkit|meeko|unidock)[[:space:]]' \
  > "$run_dir/environment/conda_core_packages.txt" || true

"$python_bin" -m scripts.experimental.unidock.prepare_stage91b_bace1_chembl365_inputs \
  --config "$config" --root . --overwrite

"$python_bin" -m scripts.experimental.unidock.audit_stage91b_bace1_chembl365_inputs \
  --config "$config" --root .

tar -czf "$core_archive" \
  "$config" \
  data/stage91_bace1_group_robust_rescue_preregistration_result.json \
  data/stage91b_bace1_development_manifest_freeze.json \
  data/stage91b_bace1_chembl365_unidock_input_summary.json \
  data/stage91b_bace1_chembl365_unidock_input_audit.json \
  data/processed/stage91b_bace1_chembl365_development_ligand_manifest.csv \
  data/processed/stage91b_bace1_chembl365_unidock_pdbqt_manifest.csv \
  "$run_dir/environment" "$run_dir/sdf" "$run_dir/pdbqt"

tar -czf "$diagnostic_archive" \
  "$config" \
  data/stage91_bace1_group_robust_rescue_preregistration_result.json \
  data/stage91b_bace1_development_manifest_freeze.json \
  data/stage91b_bace1_chembl365_unidock_input_summary.json \
  data/stage91b_bace1_chembl365_unidock_input_audit.json \
  data/processed/stage91b_bace1_chembl365_development_ligand_manifest.csv \
  data/processed/stage91b_bace1_chembl365_unidock_pdbqt_manifest.csv \
  "$run_dir"

sync
sha256sum "$core_archive" "$diagnostic_archive"
