#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-}"
if [[ -z "$python_bin" ]]; then
  for candidate in \
    /root/miniconda3/envs/qubo-unidock-stage08/bin/python \
    /root/miniconda3/envs/qubo-unidock-stage13/bin/python \
    /root/miniconda3/envs/qubo-unidock-stage07/bin/python \
    python; do
    if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then
      python_bin="$candidate"
      break
    fi
  done
fi

output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
config="configs/stage58b_ppard_pilot96_unidock113_production.json"
run_dir="results/runs/stage58b_ppard_pilot96_unidock113_production"
progress="$run_dir/progress.json"
summary="$run_dir/summary.json"
audit="data/stage58b_ppard_pilot96_unidock113_production_audit.json"
scores="$run_dir/scores.csv"
batches="$run_dir/batch_runs.csv"
median_matrix="$run_dir/primary_median_score_matrix.csv"
minimum_matrix="$run_dir/sensitivity_minimum_score_matrix.csv"
receptor_manifest="data/processed/stage58b_ppard_stage57_passing29_receptor_manifest.csv"
receptor_summary="data/stage58b_ppard_stage57_passing29_receptor_manifest_summary.json"
preregistration="configs/stage55_ppard_small_pilot_preregistration.json"
input_audit="data/stage58a_ppard_pilot96_unidock_input_independent_audit.json"
ligand_manifest="data/processed/stage58a_ppard_pilot96_unidock_pdbqt_manifest.csv"
ligand_summary="data/stage58a_ppard_pilot96_unidock_input_summary.json"
core_archive="$output_root/stage58b_ppard_pilot96_unidock113_production_core_v1.tar.gz"
diagnostic_archive="$output_root/stage58b_ppard_pilot96_unidock113_production_diagnostics_v1.tar.gz"
failure_archive="$output_root/stage58b_ppard_pilot96_packaging_recovery_failed_v1.tar.gz"

on_exit() {
  status=$?
  sync || true
  if [[ "$status" -ne 0 ]]; then
    tar -czf "$failure_archive" "$config" "$progress" "$summary" 2>/dev/null || true
    sha256sum "$failure_archive" 2>/dev/null || true
  fi
  if [[ "${AUTO_POWEROFF:-0}" == "1" ]]; then
    echo "AUTO_POWEROFF=1; requesting poweroff after exit status $status"
    shutdown -h now || poweroff || true
  fi
  return "$status"
}
trap on_exit EXIT

"$python_bin" -m scripts.experimental.unidock.repair_stage58b_ppard_pilot96_packaging \
  --config "$config"
"$python_bin" -m scripts.experimental.unidock.audit_stage58b_ppard_pilot96_production \
  --config "$config"

mapfile -t core_batch_files < <(
  find "$run_dir/batches" -type f \
    \( -name 'scores.csv' -o -name 'batch_summary.json' -o -name 'unidock.log' \) | sort
)
tar -czf "$core_archive" \
  "$config" "$summary" "$audit" "$progress" "$scores" "$batches" \
  "$median_matrix" "$minimum_matrix" "$receptor_manifest" \
  "$receptor_summary" "$preregistration" "$input_audit" \
  "$ligand_manifest" "$ligand_summary" "$run_dir/environment" \
  "${core_batch_files[@]}"
tar -czf "$diagnostic_archive" \
  "$config" "$audit" "$receptor_manifest" "$receptor_summary" \
  "$preregistration" "$input_audit" "$ligand_manifest" "$ligand_summary" \
  "$run_dir"

sync
sha256sum "$core_archive" "$diagnostic_archive"
