#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
core_archive="$output_root/stage102a_egfr_fa10_phase_a_unidock113_production_core_v1.tar.gz"
diagnostic_archive="$output_root/stage102a_egfr_fa10_phase_a_unidock113_production_diagnostics_v1.tar.gz"
failure_archive="$output_root/stage102a_egfr_fa10_phase_a_failed_runtime_v1.tar.gz"

on_exit() {
  status=$?
  sync || true
  if [[ "$status" -ne 0 ]]; then
    paths=(configs/stage102_prospective_marginal_learning.json)
    [[ -f data/stage102a_phase_a_ligand_allocation_summary.json ]] && paths+=(data/stage102a_phase_a_ligand_allocation_summary.json)
    [[ -f data/stage102a_egfr_phase_a_input_summary.json ]] && paths+=(data/stage102a_egfr_phase_a_input_summary.json)
    [[ -f data/stage102a_fa10_phase_a_input_summary.json ]] && paths+=(data/stage102a_fa10_phase_a_input_summary.json)
    [[ -d results/runs/stage102a_egfr_phase_a_production ]] && paths+=(results/runs/stage102a_egfr_phase_a_production)
    [[ -d results/runs/stage102a_fa10_phase_a_production ]] && paths+=(results/runs/stage102a_fa10_phase_a_production)
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

probe_environment() {
  local environment_name="$1"
  conda run -n "$environment_name" python -c \
    'import importlib.metadata, numpy, rdkit, scipy; assert numpy.__version__.startswith("1.26."); assert importlib.metadata.version("meeko") == "0.7.1"' \
    >/dev/null 2>&1 && \
  conda run -n "$environment_name" unidock --version >/dev/null 2>&1
}

environment_name="${STAGE102_ENV_NAME:-}"
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

mkdir -p results/runs/stage102a_environment "$output_root"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
  > results/runs/stage102a_environment/nvidia_smi.csv
unidock --version > results/runs/stage102a_environment/unidock_version.txt 2>&1
conda list | grep -E '^(python|numpy|scipy|rdkit|meeko|unidock)[[:space:]]' \
  > results/runs/stage102a_environment/conda_core_packages.txt || true

python scripts/freeze_stage102a_phase_a_receptors.py --root .
python scripts/allocate_stage102a_phase_a_ligands.py --root .
python -m scripts.prepare_stage102a_phase_a_inputs --root . --resume

python -m scripts.experimental.unidock.run_stage102a_phase_a_production \
  --root . --audit-only
python -m scripts.experimental.unidock.run_stage102a_phase_a_production \
  --root . --unidock unidock --resume

core_paths=(
  configs/stage102_prospective_marginal_learning.json
  data/stage102a_phase_a_receptor_freeze_summary.json
  data/stage102a_phase_a_ligand_allocation_summary.json
  data/stage102a_egfr_phase_a_input_summary.json
  data/stage102a_fa10_phase_a_input_summary.json
  data/processed/stage102a_egfr_passing_receptor_manifest.csv
  data/processed/stage102a_fa10_passing_receptor_manifest.csv
  data/processed/stage102a_egfr_phase_a_ligand_manifest.csv
  data/processed/stage102a_fa10_phase_a_ligand_manifest.csv
  data/processed/stage102a_egfr_phase_a_pdbqt_manifest.csv
  data/processed/stage102a_fa10_phase_a_pdbqt_manifest.csv
  results/runs/stage102a_environment
)
for target in egfr fa10; do
  run_dir="results/runs/stage102a_${target}_phase_a_production"
  core_paths+=(
    "$run_dir/summary.json"
    "$run_dir/progress.json"
    "$run_dir/scores.csv"
    "$run_dir/batch_runs.csv"
    "$run_dir/primary_median_score_matrix.csv"
    "$run_dir/sensitivity_minimum_score_matrix.csv"
  )
  while IFS= read -r path; do core_paths+=("$path"); done < <(
    find "$run_dir/batches" -type f \( -name scores.csv -o -name batch_summary.json -o -name unidock.log \) | sort
  )
done

tar -czf "$core_archive" "${core_paths[@]}"
  tar --exclude='*_out.pdbqt' --exclude='*/poses/*' -czf "$diagnostic_archive" \
  configs/stage102_prospective_marginal_learning.json \
  data/stage102a_*.json data/processed/stage102a_*.csv \
  results/runs/stage102a_environment \
  results/runs/stage102a_egfr_phase_a_production \
  results/runs/stage102a_fa10_phase_a_production

sync
sha256sum "$core_archive" "$diagnostic_archive"
