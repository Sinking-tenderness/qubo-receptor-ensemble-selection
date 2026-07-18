#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
vina_bin="environment/bin/vina_1.2.7_linux_x86_64"
config="configs/stage05_mk14_expanded_train_matrix_e32_diagnostics.json"
chmod +x "$vina_bin"

"$python_bin" scripts/run_stage05_mk14_expanded_matrix_diagnostics.py \
  --config "$config" \
  --audit-only

"$python_bin" scripts/run_stage05_mk14_expanded_matrix_diagnostics.py \
  --config "$config" \
  --overwrite

result_archive="stage05_mk14_expanded8_train160_e32_matrix_diagnostics_v1.tar.gz"
tar -czf "$result_archive" \
  results/runs/stage05_mk14_expanded8_train160_e32_matrix_diagnostics
sha256sum "$result_archive"
