#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
vina_bin="environment/bin/vina_1.2.7_linux_x86_64"
chmod +x "$vina_bin"

for seed_index in 0 1 2; do
  "$python_bin" scripts/run_md_receptor_ligand_benchmark.py \
    --config "configs/stage05_mk14_development_e16_seed${seed_index}_linux.json" \
    --audit-only
done

for seed_index in 0 1 2; do
  "$python_bin" scripts/run_md_receptor_ligand_benchmark.py \
    --config "configs/stage05_mk14_development_e16_seed${seed_index}_linux.json" \
    --resume
done

"$python_bin" scripts/aggregate_seed_replicates.py \
  --config configs/stage05_mk14_development_seed_aggregation.json \
  --overwrite
