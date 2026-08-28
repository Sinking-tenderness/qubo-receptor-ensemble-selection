#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

data_root="${DATA_ROOT:-/root/autodl-tmp/qubo_data_root}"
config="${CONFIG:-configs/experiments/ppara_pool30_fixed_k4_remote.json}"
run_directory="${RUN_DIRECTORY:-$data_root/results/runs/ppara_pool30_fixed_k4_remote}"
comparison_directory="${COMPARISON_DIRECTORY:-$run_directory/baseline_comparison}"
python_bin="${PYTHON_BIN:-python}"

"$python_bin" scripts/run_experiment.py plan \
  --config "$config" \
  --data-root "$data_root"

"$python_bin" scripts/run_experiment.py run \
  --config "$config" \
  --data-root "$data_root" \
  --from build_problem \
  --to persist \
  --resume

"$python_bin" scripts/compare_selection_methods_v5.py \
  --output-dir "$comparison_directory" \
  --target-fixed PPARA "$run_directory/problem.json" \
  --fixed-k 4

echo "PPARA pool30 fixed k=4 complete: run=$run_directory comparison=$comparison_directory"
