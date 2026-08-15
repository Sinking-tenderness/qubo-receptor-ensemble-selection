#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
MD_ENV_NAME="${MD_ENV_NAME:-qubo-receptor-md}"
UNIDOCK_ENV_NAME="${UNIDOCK_ENV_NAME:-qubo-unidock-stage08}"
CONFIG="configs/stage43_pparg_md96_rank_sensitive_replication.json"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp}"
RUN_DIR="results/runs/stage43_pparg_md96_unidock113_production"
CORE="$OUTPUT_ROOT/stage43_pparg_md96_unidock113_production_core_v1.tar.gz"
FAILURE="$OUTPUT_ROOT/stage43_pparg_md96_unidock113_production_failed_diagnostics_v1.tar.gz"

cd "$ROOT"

on_exit() {
  status=$?
  sync || true
  if [[ "$status" -ne 0 ]]; then
    tar -czf "$FAILURE" "$CONFIG" data/stage43_pparg_md96_input_preparation_result.json "$RUN_DIR" 2>/dev/null || true
    sha256sum "$FAILURE" 2>/dev/null || true
  fi
  if [[ "${AUTO_POWEROFF:-0}" == "1" ]]; then
    echo "AUTO_POWEROFF=1; requesting poweroff after exit status $status"
    shutdown -h now || poweroff || true
  fi
  return "$status"
}
trap on_exit EXIT

if [[ -z "$TRAJECTORY_ROOT" ]]; then
  candidate="$(find /root/autodl-tmp -type f -path '*/results/runs/stage28b_pparg_md_ready_multistart_md/*/trajectory_qc/aligned_protein.dcd' -print -quit 2>/dev/null || true)"
  if [[ -z "$candidate" ]]; then
    echo "ERROR: no Stage28b trajectory was found; set TRAJECTORY_ROOT to the preserved Stage28b workspace" >&2
    exit 1
  fi
  TRAJECTORY_ROOT="${candidate%%/results/runs/stage28b_pparg_md_ready_multistart_md/*}"
fi

trajectory_count="$(find "$TRAJECTORY_ROOT/results/runs/stage28b_pparg_md_ready_multistart_md" -type f -name aligned_protein.dcd 2>/dev/null | wc -l)"
if [[ "$trajectory_count" -ne 8 ]]; then
  echo "ERROR: expected 8 aligned DCD files under $TRAJECTORY_ROOT, found $trajectory_count" >&2
  exit 1
fi
echo "trajectory_root=$TRAJECTORY_ROOT"
echo "aligned_dcd_count=$trajectory_count"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$MD_ENV_NAME"
MD_PYTHON="$(command -v python)"
"$MD_PYTHON" -c 'import mdtraj; print("mdtraj_version=" + mdtraj.__version__)'

conda activate "$UNIDOCK_ENV_NAME"
UNIDOCK_PYTHON="$(command -v python)"
UNIDOCK_EXECUTABLE="$(command -v unidock)"
"$UNIDOCK_EXECUTABLE" --version 2>&1 | head -n 5 || true

"$MD_PYTHON" scripts/prepare_stage43_pparg_md96_inputs.py \
  --config "$CONFIG" --root "$ROOT" --materialize-receptors \
  --trajectory-root "$TRAJECTORY_ROOT" \
  --receptor-preparation-python "$UNIDOCK_PYTHON"

mkdir -p "$RUN_DIR/environment" "$OUTPUT_ROOT"
nvidia-smi > "$RUN_DIR/environment/nvidia_smi_full.txt"
"$UNIDOCK_EXECUTABLE" --version > "$RUN_DIR/environment/unidock_version.txt" 2>&1 || true
conda list --explicit > "$RUN_DIR/environment/conda_explicit.txt"

RUNNER=("$UNIDOCK_PYTHON" -m scripts.experimental.unidock.run_stage43_pparg_md96_production --config "$CONFIG")
"${RUNNER[@]}" --audit-only
RUNNER+=(--unidock "$UNIDOCK_EXECUTABLE" --resume)
if [[ "${FINALIZE_ONLY:-0}" == "1" ]]; then RUNNER+=(--finalize-only); fi
if [[ -n "${SEED_IDS:-}" ]]; then
  IFS=',' read -r -a seeds <<< "$SEED_IDS"
  for value in "${seeds[@]}"; do RUNNER+=(--seed-id "$value"); done
fi
if [[ -n "${RECEPTOR_IDS:-}" ]]; then
  IFS=',' read -r -a receptors <<< "$RECEPTOR_IDS"
  for value in "${receptors[@]}"; do RUNNER+=(--receptor-id "$value"); done
fi
"${RUNNER[@]}"

if [[ -f "$RUN_DIR/summary.json" ]]; then
  "$UNIDOCK_PYTHON" scripts/audit_stage43_pparg_md96_production.py --config "$CONFIG"
  "$UNIDOCK_PYTHON" scripts/build_stage43_pparg_md96_result_bundle.py --root "$ROOT" --output "$CORE"
  sync
  sha256sum "$CORE"
else
  CHECKPOINT="$OUTPUT_ROOT/stage43_pparg_md96_unidock113_checkpoint_v1.tar.gz"
  tar -czf "$CHECKPOINT" "$CONFIG" data/processed/stage43_pparg_md96_prepared_receptor_manifest.csv data/stage43_pparg_md96_input_preparation_result.json "$RUN_DIR"
  sync
  sha256sum "$CHECKPOINT"
fi
