#!/usr/bin/env bash
set -euo pipefail

# Run on the Linux docking host. All default artifact paths are remote paths.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

data_root="${DATA_ROOT:-/root/autodl-tmp/qubo_data_root}"
source_run="${SOURCE_RUN:-$data_root/results/runs/ppara_adaptive_remote}"
destination_run="${DESTINATION_RUN:-$data_root/results/runs/ppara_pool30_adaptive_remote}"
config="${CONFIG:-configs/experiments/ppara_pool30_adaptive_remote.json}"
python_bin="${PYTHON_BIN:-python}"

if [[ ! -d "$data_root" ]]; then
  echo "DATA_ROOT does not exist: $data_root" >&2
  exit 1
fi
if [[ ! -d "$source_run" ]]; then
  echo "source PPARA run does not exist: $source_run" >&2
  exit 1
fi
if [[ ! -f "$config" ]]; then
  echo "pool30 configuration does not exist: $config" >&2
  exit 1
fi

"$python_bin" scripts/run_experiment.py plan \
  --config "$config" \
  --data-root "$data_root"

prepare_args=()
if [[ -f "$destination_run/prepared_ligands.csv" ]]; then
  prepare_args+=(--resume)
fi
"$python_bin" scripts/run_experiment.py run \
  --config "$config" \
  --data-root "$data_root" \
  --to prepare \
  "${prepare_args[@]}"

"$python_bin" scripts/seed_ppara_pool30_score_reuse.py \
  --source-run "$source_run" \
  --destination-run "$destination_run" \
  --data-root "$data_root"

"$python_bin" scripts/run_experiment.py run \
  --config "$config" \
  --data-root "$data_root" \
  --from dock \
  --to persist \
  --resume

"$python_bin" - "$destination_run" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

run_directory = Path(sys.argv[1])
audit = json.loads((run_directory / "score_table_reuse_audit.json").read_text(encoding="utf-8"))
receptors = json.loads((run_directory / "receptor_preparation_audit.json").read_text(encoding="utf-8"))
manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
score_tables = list((run_directory / "score_tables").glob("seed_*__*.csv"))

if audit.get("status") != "ok" or len(audit.get("shared_receptor_ids", [])) != 15:
    raise SystemExit("score table reuse audit does not verify the original 15 receptors")
if int(audit.get("linked_table_count", 0)) != 45:
    raise SystemExit("score table reuse audit does not contain 45 verified source tables")
if int(receptors.get("selected_count", 0)) != 30:
    raise SystemExit("pool30 receptor audit does not contain 30 selected receptors")
if len(score_tables) != 90:
    raise SystemExit(f"expected 90 score tables, found {len(score_tables)}")
if manifest.get("status") != "completed":
    raise SystemExit("pool30 manifest is not completed")

print(f"pool30 verified: run={run_directory} score_tables={len(score_tables)}")
PY
