#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
vina_bin="environment/bin/vina_1.2.7_linux_x86_64"
config="configs/stage05_mk14_fresh_validation_e32_seed1_64vcpu_linux.json"
run_dir="results/runs/stage05_mk14_fresh_validation_e32_seed1_linux"
result_archive="stage05_mk14_fresh_validation_seed1_64vcpu_core_v1.tar.gz"

available_cpu="$(nproc)"
if (( available_cpu < 64 )); then
  echo "ERROR: this profile requires at least 64 visible CPUs; found ${available_cpu}" >&2
  exit 1
fi

chmod +x "$vina_bin"
echo "execution_profile=distributed_seed1_64vcpu"
echo "available_cpu=${available_cpu}"
echo "layout=32_workers_x_2_vina_cpu"
echo "base_seed=20260802"

"$python_bin" scripts/run_md_receptor_ligand_benchmark.py \
  --config "$config" \
  --audit-only

"$python_bin" scripts/run_md_receptor_ligand_benchmark.py \
  --config "$config" \
  --resume

"$python_bin" - "$run_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="ascii"))
assert summary["status"] in {"ok", "ok_with_search_warning"}
assert summary["receptor_count"] == 5
assert summary["ligand_count"] == 1576
assert summary["expected_receptor_ligand_pairs"] == 7880
assert summary["observed_receptor_ligand_pairs"] == 7880
assert summary["successful_receptor_ligand_pairs"] == 7880
assert summary["failed_receptor_ligand_pairs"] == 0
assert summary["docking_parameters"]["workers"] == 32
assert summary["docking_parameters"]["max_total_cpu"] == 64
assert summary["docking_parameters"]["base_seed"] == 20260802
print("seed1_completion_audit=ok")
PY

tar -czf "$result_archive" "$run_dir/summary.json" "$run_dir"/*.csv
sha256sum "$result_archive"
