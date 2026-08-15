#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
MD_ENV_NAME="${MD_ENV_NAME:-qubo-receptor-md}"
UNIDOCK_ENV_NAME="${UNIDOCK_ENV_NAME:-qubo-unidock-stage08}"
CONFIG="configs/stage32_pparg_md_functional_complementarity_pilot.json"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-}"

cd "$ROOT"

if [[ -z "$TRAJECTORY_ROOT" ]]; then
  candidate="$(find /root/autodl-tmp -type f -path '*/results/runs/stage28b_pparg_md_ready_multistart_md/*/trajectory_qc/aligned_protein.dcd' -print -quit 2>/dev/null || true)"
  if [[ -z "$candidate" ]]; then
    echo "ERROR: no Stage28b aligned_protein.dcd was found; set TRAJECTORY_ROOT to the preserved Stage28b workspace" >&2
    exit 1
  fi
  TRAJECTORY_ROOT="${candidate%%/results/runs/stage28b_pparg_md_ready_multistart_md/*}"
fi

trajectory_count="$(find "$TRAJECTORY_ROOT/results/runs/stage28b_pparg_md_ready_multistart_md" -type f -name aligned_protein.dcd 2>/dev/null | wc -l)"
if [[ "$trajectory_count" -ne 8 ]]; then
  echo "ERROR: expected 8 Stage28b aligned DCD files under $TRAJECTORY_ROOT, found $trajectory_count" >&2
  exit 1
fi
echo "trajectory_root=$TRAJECTORY_ROOT"
echo "aligned_dcd_count=$trajectory_count"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$MD_ENV_NAME"
MD_PYTHON="$(command -v python)"
python - <<'PY'
import mdtraj
import numpy
print("mdtraj_version=" + mdtraj.__version__)
print("md_numpy_version=" + numpy.__version__)
PY

conda activate "$UNIDOCK_ENV_NAME"
UNIDOCK_PYTHON="$(command -v python)"
UNIDOCK_EXECUTABLE="$(command -v unidock)"
python - <<'PY'
import importlib.metadata
import numpy
import sys

for package in ("meeko", "prody"):
    print(f"{package}={importlib.metadata.version(package)}")
print("unidock_numpy_version=" + numpy.__version__)
print("python=" + sys.version.split()[0])
PY
"$UNIDOCK_EXECUTABLE" --version 2>&1 | head -n 5 || true

"$MD_PYTHON" scripts/prepare_stage32_pparg_md_functional_pilot.py \
  --config "$CONFIG" \
  --root "$ROOT" \
  --materialize-receptors \
  --trajectory-root "$TRAJECTORY_ROOT" \
  --receptor-preparation-python "$UNIDOCK_PYTHON"

"$UNIDOCK_PYTHON" scripts/experimental/unidock/run_stage32_pparg_md_functional_pilot.py \
  --config "$CONFIG" \
  --root "$ROOT" \
  --unidock "$UNIDOCK_EXECUTABLE" \
  --audit-only

"$UNIDOCK_PYTHON" scripts/experimental/unidock/run_stage32_pparg_md_functional_pilot.py \
  --config "$CONFIG" \
  --root "$ROOT" \
  --unidock "$UNIDOCK_EXECUTABLE" \
  --resume

"$UNIDOCK_PYTHON" scripts/build_stage32_pparg_md_functional_pilot_result_bundle.py \
  --root "$ROOT" \
  --output "$ROOT/stage32_pparg_md_functional_complementarity_pilot_core_v1.tar.gz"

sha256sum "$ROOT/stage32_pparg_md_functional_complementarity_pilot_core_v1.tar.gz"
