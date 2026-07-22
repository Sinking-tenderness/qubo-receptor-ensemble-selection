#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
vina_bin="environment/bin/vina_1.2.7_linux_x86_64"
chmod +x "$vina_bin"

echo "execution_profile=32vcpu"
echo "available_cpu=$(nproc)"
echo "layout=16_workers_x_2_vina_cpu"

for seed_index in 0 1 2; do
  "$python_bin" scripts/run_md_receptor_ligand_benchmark.py \
    --config "configs/stage05_mk14_fresh_validation_e32_seed${seed_index}_32vcpu_linux.json" \
    --audit-only
done

for seed_index in 0 1 2; do
  "$python_bin" scripts/run_md_receptor_ligand_benchmark.py \
    --config "configs/stage05_mk14_fresh_validation_e32_seed${seed_index}_32vcpu_linux.json" \
    --resume
done

"$python_bin" scripts/aggregate_seed_replicates.py \
  --config configs/stage05_mk14_fresh_validation_e32_32vcpu_seed_aggregation.json \
  --overwrite

result_archive="stage05_mk14_fresh_validation_e32_core_results_v1.tar.gz"
tar -czf "$result_archive" \
  results/runs/stage05_mk14_fresh_validation_e32_seed0_linux/summary.json \
  results/runs/stage05_mk14_fresh_validation_e32_seed0_linux/representative_scores.csv \
  results/runs/stage05_mk14_fresh_validation_e32_seed1_linux/summary.json \
  results/runs/stage05_mk14_fresh_validation_e32_seed1_linux/representative_scores.csv \
  results/runs/stage05_mk14_fresh_validation_e32_seed2_linux/summary.json \
  results/runs/stage05_mk14_fresh_validation_e32_seed2_linux/representative_scores.csv \
  results/runs/stage05_mk14_fresh_validation_e32_aggregated/aggregated_seed_scores.csv \
  results/runs/stage05_mk14_fresh_validation_e32_aggregated/primary_median_score_matrix.csv \
  results/runs/stage05_mk14_fresh_validation_e32_aggregated/sensitivity_minimum_score_matrix.csv \
  results/runs/stage05_mk14_fresh_validation_e32_aggregated/summary.json
sha256sum "$result_archive"
