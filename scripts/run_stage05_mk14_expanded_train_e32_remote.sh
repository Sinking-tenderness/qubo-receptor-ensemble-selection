#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
vina_bin="environment/bin/vina_1.2.7_linux_x86_64"
chmod +x "$vina_bin"

for seed_index in 0 1 2; do
  "$python_bin" scripts/run_md_receptor_ligand_benchmark.py \
    --config "configs/stage05_mk14_expanded8_train160_e32_seed${seed_index}_linux.json" \
    --audit-only
done

for seed_index in 0 1 2; do
  "$python_bin" scripts/run_md_receptor_ligand_benchmark.py \
    --config "configs/stage05_mk14_expanded8_train160_e32_seed${seed_index}_linux.json" \
    --resume
done

"$python_bin" scripts/aggregate_seed_replicates.py \
  --config configs/stage05_mk14_expanded8_train160_e32_seed_aggregation.json \
  --overwrite

audit_dir="results/runs/stage05_mk14_expanded8_train160_e32_matrix_admission"
mkdir -p "$audit_dir"
set +e
"$python_bin" scripts/audit_stage05_expanded_e32_matrix.py \
  --amendment configs/stage05_mk14_expanded_train_search_protocol_amendment01.json \
  --aggregation-summary results/runs/stage05_mk14_expanded8_train160_e32_aggregated/summary.json \
  --output-summary "$audit_dir/summary.json" \
  --flagged-output "$audit_dir/flagged_pairs.csv" \
  --overwrite
audit_exit=$?
set -e
if [[ "$audit_exit" -ne 0 && "$audit_exit" -ne 2 ]]; then
  exit "$audit_exit"
fi

result_archive="stage05_mk14_expanded8_train160_e32_core_results_v1.tar.gz"
tar -czf "$result_archive" \
  results/runs/stage05_mk14_expanded8_train160_e32_seed0_linux/summary.json \
  results/runs/stage05_mk14_expanded8_train160_e32_seed0_linux/representative_scores.csv \
  results/runs/stage05_mk14_expanded8_train160_e32_seed1_linux/summary.json \
  results/runs/stage05_mk14_expanded8_train160_e32_seed1_linux/representative_scores.csv \
  results/runs/stage05_mk14_expanded8_train160_e32_seed2_linux/summary.json \
  results/runs/stage05_mk14_expanded8_train160_e32_seed2_linux/representative_scores.csv \
  results/runs/stage05_mk14_expanded8_train160_e32_aggregated/aggregated_seed_scores.csv \
  results/runs/stage05_mk14_expanded8_train160_e32_aggregated/primary_median_score_matrix.csv \
  results/runs/stage05_mk14_expanded8_train160_e32_aggregated/sensitivity_minimum_score_matrix.csv \
  results/runs/stage05_mk14_expanded8_train160_e32_aggregated/summary.json \
  "$audit_dir/summary.json" \
  "$audit_dir/flagged_pairs.csv"
sha256sum "$result_archive"
if [[ "$audit_exit" -eq 2 ]]; then
  echo "e32 matrix admission was rejected; the result archive was still created"
fi
