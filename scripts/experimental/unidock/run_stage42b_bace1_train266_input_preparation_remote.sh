#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
run_dir="results/runs/stage42b_bace1_train266_unidock_inputs"
core_archive="$output_root/stage42b_bace1_train266_unidock_inputs_core_v1.tar.gz"
diagnostic_archive="$output_root/stage42b_bace1_train266_unidock_inputs_diagnostics_v1.tar.gz"
failure_archive="$output_root/stage42b_bace1_train266_unidock_inputs_failed_runtime_v1.tar.gz"

on_exit() {
  status=$?
  sync || true
  if [[ "$status" -ne 0 ]]; then
    tar -czf "$failure_archive" \
      configs/stage42*.json \
      data/stage42a_bace1_ligand_panel_allocation_summary.json \
      data/processed/stage42a_bace1_*manifest.csv \
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

mkdir -p "$run_dir/environment" "$output_root"
"$python_bin" --version > "$run_dir/environment/python_version.txt" 2>&1
conda list | grep -E '^(python|numpy|scipy|rdkit|meeko|unidock)[[:space:]]' \
  > "$run_dir/environment/conda_core_packages.txt" || true
conda list --explicit > "$run_dir/environment/conda_explicit.txt"

"$python_bin" scripts/allocate_stage42a_bace1_ligand_panels.py \
  --config configs/stage42a_bace1_ligand_panel_allocation.json \
  --overwrite

"$python_bin" -m scripts.experimental.unidock.prepare_stage42b_bace1_train266_inputs \
  --config configs/stage42b_bace1_train266_unidock_input_preparation.json \
  --overwrite

tar -czf "$core_archive" \
  configs/stage42_bace1_redocking_qualified_development_preregistration.json \
  configs/stage42a_bace1_ligand_panel_allocation.json \
  configs/stage42b_bace1_train266_unidock_input_preparation.json \
  data/stage42a_bace1_ligand_panel_allocation_summary.json \
  data/stage42b_bace1_train266_unidock_input_summary.json \
  data/processed/stage42a_bace1_selected_ligand_panel_manifest.csv \
  data/processed/stage42a_bace1_train266_ligand_manifest.csv \
  data/processed/stage42b_bace1_train266_unidock_pdbqt_manifest.csv \
  "$run_dir/environment" "$run_dir/sdf" "$run_dir/pdbqt"

tar -czf "$diagnostic_archive" \
  configs/stage42_bace1_redocking_qualified_development_preregistration.json \
  configs/stage42a_bace1_ligand_panel_allocation.json \
  configs/stage42b_bace1_train266_unidock_input_preparation.json \
  data/stage42a_bace1_ligand_panel_allocation_summary.json \
  data/stage42b_bace1_train266_unidock_input_summary.json \
  data/processed/stage42a_bace1_selected_ligand_panel_manifest.csv \
  data/processed/stage42a_bace1_train266_ligand_manifest.csv \
  data/processed/stage42b_bace1_train266_unidock_pdbqt_manifest.csv \
  "$run_dir"

sync
sha256sum "$core_archive" "$diagnostic_archive"
