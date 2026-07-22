#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
unidock_bin="${UNIDOCK_BIN:-unidock}"
archive="stage05_mk14_unidock_rigid_train160_gpu_diagnostics_core_v1.tar.gz"
evidence_dir="results/runs/stage05_mk14_unidock_rigid_gpu_diagnostics_environment"
configs=(
  "configs/stage05_mk14_unidock_rigid_train160_detail_gpu_equivalence.json"
  "configs/stage05_mk14_unidock_rigid_train160_enhanced_gpu_equivalence.json"
)

mkdir -p "$evidence_dir"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv > "$evidence_dir/nvidia_smi.csv"
nvidia-smi > "$evidence_dir/nvidia_smi_full.txt"
"$unidock_bin" --version > "$evidence_dir/unidock_version.txt" 2>&1 || true
conda list unidock > "$evidence_dir/conda_unidock.txt"
conda list --explicit > "$evidence_dir/conda_explicit.txt"

run_dirs=()
for config in "${configs[@]}"; do
  run_dir="$($python_bin - "$config" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="ascii") as handle:
    print(json.load(handle)["outputs"]["run_directory"])
PY
)"
  run_dirs+=("$run_dir")
  "$python_bin" scripts/experimental/unidock/run_unidock_gpu_equivalence.py \
    --config "$config" \
    --audit-only
  "$python_bin" scripts/experimental/unidock/run_unidock_gpu_equivalence.py \
    --config "$config" \
    --unidock "$unidock_bin" \
    --resume
  "$python_bin" scripts/experimental/unidock/audit_unidock_gpu_equivalence.py \
    --config "$config" \
    --overwrite
done

tar_args=(
  --exclude='*/poses/*'
  --exclude='*/ligands.index'
  -czf "$archive"
  "${configs[@]}"
  "$evidence_dir"
)
for run_dir in "${run_dirs[@]}"; do
  tar_args+=("$run_dir")
done
tar "${tar_args[@]}"

sha256sum "$archive"
