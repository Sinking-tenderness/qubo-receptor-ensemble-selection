#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
prep_config="configs/stage57_ppard_cognate_redocking_input_preparation.json"
dock_config="configs/stage57_ppard_cognate_redocking.json"
prep_run="results/runs/stage57_ppard_cognate_redocking_input_preparation"
dock_run="results/runs/stage57_ppard_cognate_redocking"
prep_summary="data/stage57_ppard_cognate_redocking_input_preparation_summary.json"
dock_summary="data/stage57_ppard_cognate_redocking_summary.json"
receptors="data/processed/stage57_ppard_prepared_receptor_manifest.csv"
cases="data/processed/stage57_ppard_cognate_redocking_case_manifest.csv"
box="data/stage57_ppard_frozen_common_box.json"
results="data/processed/stage57_ppard_cognate_redocking_results.csv"
gate_results="data/processed/stage57_ppard_receptor_gate_results.csv"
core_archive="$output_root/stage57_ppard_cognate_redocking_core_v1.tar.gz"
diagnostic_archive="$output_root/stage57_ppard_cognate_redocking_diagnostics_v1.tar.gz"
failure_archive="$output_root/stage57_ppard_cognate_redocking_failed_runtime_v1.tar.gz"

on_exit() {
  status=$?
  sync || true
  if [[ "$status" -ne 0 ]]; then
    tar -czf "$failure_archive" \
      "$prep_config" "$dock_config" "$prep_run" "$dock_run" \
      "$prep_summary" "$dock_summary" "$receptors" "$cases" "$box" \
      "$results" "$gate_results" 2>/dev/null || true
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
    'import gemmi, meeko, numpy, openmm, pdbfixer, prody, rdkit, scipy; assert numpy.__version__ == "1.26.4"' \
    >/dev/null 2>&1 && \
  conda run -n "$environment_name" unidock --version >/dev/null 2>&1
}

environment_name="${STAGE57_ENV_NAME:-qubo-unidock-stage08}"
if ! conda env list | awk '{print $1}' | grep -qx "$environment_name"; then
  if [[ "$environment_name" != "qubo-unidock-stage08" ]]; then
    echo "The Stage57 runtime ledger requires qubo-unidock-stage08." >&2
    exit 1
  fi
  conda env create -f environment/stage57_ppard_unidock_gpu.yml
fi
if ! probe_environment "$environment_name"; then
  echo "Existing $environment_name lacks a Stage57 dependency; do not alter it while another job is running." >&2
  echo "Required imports: gemmi, meeko, numpy=1.26.4, openmm, pdbfixer, prody, rdkit, scipy, plus Uni-Dock 1.1.3." >&2
  exit 1
fi
conda activate "$environment_name"
echo "environment_name=$environment_name"

mkdir -p "$prep_run/environment" "$dock_run/environment" "$output_root"
python --version > "$prep_run/environment/python_version.txt" 2>&1
conda list | grep -E '^(python|numpy|scipy|rdkit|gemmi|prody|meeko|pdbfixer|openmm|unidock)[[:space:]]' \
  > "$prep_run/environment/conda_core_packages.txt" || true
conda list --explicit > "$prep_run/environment/conda_explicit.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
  > "$dock_run/environment/nvidia_smi.csv"
nvidia-smi > "$dock_run/environment/nvidia_smi_full.txt"
unidock --version > "$dock_run/environment/unidock_version.txt" 2>&1
cp "$prep_run/environment/conda_core_packages.txt" "$dock_run/environment/"
cp "$prep_run/environment/conda_explicit.txt" "$dock_run/environment/"

python -m scripts.prepare_stage57_ppard_redocking_inputs \
  --config "$prep_config" --audit-only
python -m scripts.prepare_stage57_ppard_redocking_inputs \
  --config "$prep_config" --resume

python -m scripts.experimental.unidock.run_stage57_ppard_cognate_redocking \
  --config "$dock_config" --audit-only
python -m scripts.experimental.unidock.run_stage57_ppard_cognate_redocking \
  --config "$dock_config" --unidock unidock --resume

mapfile -t prep_core_files < <(
  find "$prep_run/preparation" -type f \
    \( -name 'case_preparation_summary.json' \
       -o -name 'case_preparation_failure.json' \
       -o -name 'summary.json' \
       -o -name 'alignment_summary.json' \
       -o -name '*_completed.pdb' \
       -o -name '*_receptor.pdbqt' \
       -o -name '*_ligand.pdbqt' \
       -o -name '*_common_frame.sdf' \) | sort
)
mapfile -t dock_core_files < <(
  find "$dock_run/batches" -type f \
    \( -name 'batch_summary.json' -o -name 'scores.csv' \
       -o -name 'unidock.log' -o -path '*/rmsd/summary.json' \
       -o -path '*/rmsd/poses.csv' \) | sort
)

tar -czf "$core_archive" \
  "$prep_config" "$dock_config" "$prep_summary" "$dock_summary" \
  "$receptors" "$cases" "$box" "$results" "$gate_results" \
  "$prep_run/environment" "$dock_run/environment" \
  "${prep_core_files[@]}" "${dock_core_files[@]}"

tar -czf "$diagnostic_archive" \
  "$prep_config" "$dock_config" "$prep_summary" "$dock_summary" \
  "$receptors" "$cases" "$box" "$results" "$gate_results" \
  "$prep_run" "$dock_run"

sync
sha256sum "$core_archive" "$diagnostic_archive"
