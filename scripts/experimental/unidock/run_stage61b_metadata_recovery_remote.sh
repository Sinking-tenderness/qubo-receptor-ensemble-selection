#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

output_root="${OUTPUT_ROOT:-/root/autodl-tmp}"
failure_archive="$output_root/stage61b_ppard_metadata_recovery_failed_v1.tar.gz"

on_exit() {
  status=$?
  sync || true
  if [[ "$status" -ne 0 ]]; then
    paths=(
      configs/stage61b_ppard_remaining144_unidock113_production.json
      results/runs/stage61b_ppard_remaining144_unidock113_production/progress.json
      results/runs/stage61b_ppard_remaining144_unidock113_production/summary.json
    )
    [[ -f data/stage61b_ppard_progress_descriptor_amendment01.json ]] && \
      paths+=(data/stage61b_ppard_progress_descriptor_amendment01.json)
    [[ -f data/stage61b_ppard_remaining144_unidock113_production_audit.json ]] && \
      paths+=(data/stage61b_ppard_remaining144_unidock113_production_audit.json)
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
    'import meeko, numpy, rdkit, scipy; assert numpy.__version__.startswith("1.26.")' \
    >/dev/null 2>&1
}

environment_name="${STAGE61_ENV_NAME:-}"
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
  echo "No compatible existing environment found; metadata recovery will not create one." >&2
  exit 1
fi
conda activate "$environment_name"
echo "environment_name=$environment_name"

python -m scripts.repair_stage61b_progress_descriptor --root .
python -m scripts.experimental.unidock.audit_stage61b_ppard_remaining144_production \
  --config configs/stage61b_ppard_remaining144_unidock113_production.json
python -m scripts.experimental.unidock.package_stage61b_recovered_results \
  --root . --output-root "$output_root"

sync
sha256sum \
  "$output_root/stage61b_ppard_remaining144_unidock113_production_core_v1.tar.gz" \
  "$output_root/stage61b_ppard_remaining144_unidock113_production_diagnostics_v1.tar.gz"
