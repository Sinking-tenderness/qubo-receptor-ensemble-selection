#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
CONFIG="configs/stage32b_pparg_md_pair_fresh_validation.json"
STAGE32_WORKSPACE="${STAGE32_WORKSPACE:-}"
UNIDOCK_ENV_NAME="${UNIDOCK_ENV_NAME:-}"
SHUTDOWN_ON_SUCCESS=0

if [[ "${1:-}" == "--shutdown-on-success" ]]; then
  SHUTDOWN_ON_SUCCESS=1
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--shutdown-on-success]" >&2
  exit 2
fi

cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ -z "$STAGE32_WORKSPACE" ]]; then
  receptor="$(find /root/autodl-tmp -type f -path '*/results/runs/stage32_pparg_md_functional_pilot_input_preparation/receptors/PPARG_MD_00524_2ATH/PPARG_MD_00524_2ATH_receptor.pdbqt' -print -quit 2>/dev/null || true)"
  if [[ -z "$receptor" ]]; then
    echo "ERROR: preserved Stage32 receptor PDBQT was not found; set STAGE32_WORKSPACE" >&2
    exit 1
  fi
  STAGE32_WORKSPACE="${receptor%%/results/runs/stage32_pparg_md_functional_pilot_input_preparation/*}"
fi
echo "stage32_workspace=$STAGE32_WORKSPACE"

if [[ -z "$UNIDOCK_ENV_NAME" ]]; then
  for candidate in qubo-unidock-stage07 qubo-unidock-stage08 qubo-unidock-stage13; do
    prefix="$(conda env list --json | python -c "import json,sys; envs=json.load(sys.stdin)['envs']; print(next((p for p in envs if p.endswith('/$candidate')), ''))")"
    if [[ -n "$prefix" && -x "$prefix/bin/python" && -x "$prefix/bin/unidock" ]] && \
       "$prefix/bin/python" - <<'PY' >/dev/null 2>&1
import importlib.metadata
assert importlib.metadata.version("rdkit") == "2026.3.1"
assert importlib.metadata.version("meeko") == "0.7.1"
import numpy, scipy
PY
    then
      UNIDOCK_ENV_NAME="$candidate"
      break
    fi
  done
fi
if [[ -z "$UNIDOCK_ENV_NAME" ]]; then
  echo "ERROR: no existing environment has Uni-Dock 1.1.3, RDKit 2026.3.1, Meeko 0.7.1, NumPy, and SciPy" >&2
  exit 1
fi

conda activate "$UNIDOCK_ENV_NAME"
PYTHON="$(command -v python)"
UNIDOCK="$(command -v unidock)"
echo "unidock_env=$UNIDOCK_ENV_NAME"
"$PYTHON" - <<'PY'
import importlib.metadata
import numpy
import scipy
print("python_dependencies=ok")
for package in ("rdkit", "meeko"):
    print(f"{package}={importlib.metadata.version(package)}")
print("numpy=" + numpy.__version__)
print("scipy=" + scipy.__version__)
PY
"$UNIDOCK" --version 2>&1 | head -n 5 || true

"$PYTHON" scripts/prepare_stage32b_pparg_fresh_validation.py \
  --config "$CONFIG" \
  --root "$ROOT" \
  --stage32-workspace "$STAGE32_WORKSPACE"

"$PYTHON" scripts/experimental/unidock/run_stage32b_pparg_md_pair_fresh_validation.py \
  --config "$CONFIG" \
  --root "$ROOT" \
  --unidock "$UNIDOCK" \
  --audit-only

"$PYTHON" scripts/experimental/unidock/run_stage32b_pparg_md_pair_fresh_validation.py \
  --config "$CONFIG" \
  --root "$ROOT" \
  --unidock "$UNIDOCK" \
  --resume

"$PYTHON" scripts/evaluate_stage32b_pparg_md_pair_fresh_validation.py \
  --config "$CONFIG" \
  --root "$ROOT"

"$PYTHON" scripts/build_stage32b_pparg_md_pair_fresh_validation_result_bundle.py \
  --root "$ROOT" \
  --output "$ROOT/stage32b_pparg_md_pair_fresh_validation_core_v1.tar.gz"

sha256sum "$ROOT/stage32b_pparg_md_pair_fresh_validation_core_v1.tar.gz"
sync
if [[ "$SHUTDOWN_ON_SUCCESS" -eq 1 ]]; then
  echo "Stage32b completed and synced; shutting down now."
  sudo shutdown -h now 2>/dev/null || shutdown -h now
fi
