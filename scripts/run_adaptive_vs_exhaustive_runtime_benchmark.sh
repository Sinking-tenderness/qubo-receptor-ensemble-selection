#!/usr/bin/env bash
set -euo pipefail

# Run the frozen adaptive-vs-exhaustive timing comparison on a Linux host.
# The docking and aggregation artifacts are shared; every command starts at
# build_problem and ends at persist so the timing covers k selection plus the
# final solve/evaluate/persist work only.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/qubo_data_root}"
OUTPUT_DIR="${RUNTIME_BENCHMARK_DIR:-$DATA_ROOT/results/runtime_benchmark}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Repository directory does not exist: $REPO_ROOT" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/scripts/run_experiment.py" ]]; then
  echo "Cannot find scripts/run_experiment.py under: $REPO_ROOT" >&2
  exit 1
fi
if [[ ! -x "$TIME_BIN" ]]; then
  echo "GNU time executable is not available: $TIME_BIN" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable is not available: $PYTHON_BIN" >&2
  exit 1
fi

# target|mode|config relative to the repository root. Keep this list frozen:
# five targets x (one adaptive + six fixed-k configurations) = 35 runs.
CONFIG_SPECS=(
  "MK14|adaptive|configs/experiments/mk14_adaptive_rp09_remote.json"
  "MK14|fixed|configs/experiments/mk14_fixed_k1_remote.json"
  "MK14|fixed|configs/experiments/mk14_fixed_k2_remote.json"
  "MK14|fixed|configs/experiments/mk14_fixed_k3_remote.json"
  "MK14|fixed|configs/experiments/mk14_fixed_k4_remote.json"
  "MK14|fixed|configs/experiments/mk14_fixed_k5_remote.json"
  "MK14|fixed|configs/experiments/mk14_fixed_k6_remote.json"
  "PPARG|adaptive|configs/experiments/pparg_adaptive_remote.json"
  "PPARG|fixed|configs/experiments/pparg_fixed_k1_remote.json"
  "PPARG|fixed|configs/experiments/pparg_fixed_k2_remote.json"
  "PPARG|fixed|configs/experiments/pparg_fixed_k3_remote.json"
  "PPARG|fixed|configs/experiments/pparg_fixed_k4_remote.json"
  "PPARG|fixed|configs/experiments/pparg_fixed_k5_remote.json"
  "PPARG|fixed|configs/experiments/pparg_fixed_k6_remote.json"
  "BACE1|adaptive|configs/experiments/bace1_adaptive_remote.json"
  "BACE1|fixed|configs/experiments/bace1_fixed_k1_remote.json"
  "BACE1|fixed|configs/experiments/bace1_fixed_k2_remote.json"
  "BACE1|fixed|configs/experiments/bace1_fixed_k3_remote.json"
  "BACE1|fixed|configs/experiments/bace1_fixed_k4_remote.json"
  "BACE1|fixed|configs/experiments/bace1_fixed_k5_remote.json"
  "BACE1|fixed|configs/experiments/bace1_fixed_k6_remote.json"
  "ESR1|adaptive|configs/experiments/esr1_adaptive_remote.json"
  "ESR1|fixed|configs/experiments/esr1_fixed_k1_remote.json"
  "ESR1|fixed|configs/experiments/esr1_fixed_k2_remote.json"
  "ESR1|fixed|configs/experiments/esr1_fixed_k3_remote.json"
  "ESR1|fixed|configs/experiments/esr1_fixed_k4_remote.json"
  "ESR1|fixed|configs/experiments/esr1_fixed_k5_remote.json"
  "ESR1|fixed|configs/experiments/esr1_fixed_k6_remote.json"
  "PPARA|adaptive|configs/experiments/ppara_adaptive_remote.json"
  "PPARA|fixed|configs/experiments/ppara_fixed_k1_remote.json"
  "PPARA|fixed|configs/experiments/ppara_fixed_k2_remote.json"
  "PPARA|fixed|configs/experiments/ppara_fixed_k3_remote.json"
  "PPARA|fixed|configs/experiments/ppara_fixed_k4_remote.json"
  "PPARA|fixed|configs/experiments/ppara_fixed_k5_remote.json"
  "PPARA|fixed|configs/experiments/ppara_fixed_k6_remote.json"
)

mkdir -p "$OUTPUT_DIR"
MANIFEST_FILE="$OUTPUT_DIR/configurations.tsv"

# Resolve experiment IDs and result directories once, and fail before any
# expensive run if the frozen configuration set is incomplete.
"$PYTHON_BIN" - "$DATA_ROOT" "$REPO_ROOT" "$MANIFEST_FILE" "${CONFIG_SPECS[@]}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

data_root = Path(sys.argv[1]).resolve()
repo_root = Path(sys.argv[2]).resolve()
manifest_path = Path(sys.argv[3])
specs = sys.argv[4:]

if len(specs) != 35:
    raise SystemExit(f"expected 35 configuration specs, found {len(specs)}")

rows: list[dict[str, str]] = []
seen_targets: dict[str, dict[str, int]] = {}
for spec in specs:
    try:
        target, mode, relative_config = spec.split("|", 2)
    except ValueError as exc:
        raise SystemExit(f"invalid configuration spec: {spec}") from exc
    config_path = repo_root / relative_config
    if not config_path.is_file():
        raise SystemExit(f"configuration file does not exist: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON configuration: {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"configuration root is not an object: {config_path}")
    if str(payload.get("target_id", "")) != target:
        raise SystemExit(
            f"target mismatch for {config_path}: "
            f"expected {target}, got {payload.get('target_id')}"
        )
    experiment_id = str(payload.get("experiment_id", "")).strip()
    if not experiment_id:
        raise SystemExit(f"missing experiment_id: {config_path}")
    paths = payload.get("paths")
    if not isinstance(paths, dict) or not str(paths.get("run_directory", "")).strip():
        raise SystemExit(f"missing paths.run_directory: {config_path}")
    run_directory = Path(str(paths["run_directory"]))
    if not run_directory.is_absolute():
        run_directory = data_root / run_directory
    fixed_k = ""
    problem = payload.get("problem")
    if not isinstance(problem, dict):
        raise SystemExit(f"missing problem object: {config_path}")
    if mode == "fixed":
        fixed_k = str(problem.get("target_size", "")).strip()
        if fixed_k not in {str(k) for k in range(1, 7)}:
            raise SystemExit(f"fixed configuration has invalid target_size: {config_path}")
    elif mode == "adaptive":
        policy = problem.get("k_policy")
        if not isinstance(policy, dict) or policy.get("mode") != "adaptive":
            raise SystemExit(f"configuration is not adaptive: {config_path}")
        try:
            candidates = [int(value) for value in policy.get("candidates", [])]
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"adaptive candidates are invalid: {config_path}") from exc
        if candidates != list(range(1, 7)):
            raise SystemExit(
                f"adaptive candidates must be [1, 2, 3, 4, 5, 6]: {config_path}"
            )
    else:
        raise SystemExit(f"invalid mode in configuration spec: {spec}")
    target_counts = seen_targets.setdefault(target, {"adaptive": 0, "fixed": 0})
    target_counts[mode] += 1
    rows.append(
        {
            "target_id": target,
            "mode": mode,
            "fixed_k": fixed_k,
            "config_rel": relative_config,
            "config_path": str(config_path),
            "experiment_id": experiment_id,
            "run_directory": str(run_directory),
        }
    )

if set(seen_targets) != {"MK14", "PPARG", "BACE1", "ESR1", "PPARA"}:
    raise SystemExit(f"unexpected target set: {sorted(seen_targets)}")
for target, counts in seen_targets.items():
    if counts != {"adaptive": 1, "fixed": 6}:
        raise SystemExit(f"invalid configuration count for {target}: {counts}")

manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(rows[0]),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
PY

timing_is_complete() {
  local time_file="$1"
  local status_file="$2"
  local meta_file="$3"
  local summary_file="$4"

  [[ -s "$time_file" ]] || return 1
  grep -Eq '^elapsed_seconds=[0-9]' "$time_file" || return 1
  [[ -f "$summary_file" ]] || return 1

  if [[ -s "$status_file" ]]; then
    [[ "$(tr -d '[:space:]' < "$status_file")" == "0" ]] || return 1
  elif [[ -f "$meta_file" ]]; then
    grep -Eq '^action=(completed|skipped_existing)$' "$meta_file" || return 1
  fi

  "$PYTHON_BIN" - "$summary_file" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        payload = json.load(handle)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "completed" else 1)
PY
}

failures=0
while IFS=$'\t' read -r target mode fixed_k config_rel config_path experiment_id run_directory; do
  [[ "$target" == "target_id" ]] && continue

  key="${config_rel##*/}"
  key="${key%.json}"
  time_file="$OUTPUT_DIR/$key.time"
  status_file="$OUTPUT_DIR/$key.status"
  meta_file="$OUTPUT_DIR/$key.meta"
  log_file="$OUTPUT_DIR/$key.log"
  summary_file="$run_directory/summary.json"

  if timing_is_complete "$time_file" "$status_file" "$meta_file" "$summary_file"; then
    if [[ ! -s "$status_file" ]]; then
      printf '0\n' > "$status_file"
    fi
    if [[ ! -f "$meta_file" ]]; then
      printf 'action=skipped_existing\n' > "$meta_file"
    fi
    echo "[skip] $target $mode ${fixed_k:-adaptive} ($key): existing successful timing"
    continue
  fi

  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    printf 'action=running\n'
    printf 'started_at_utc=%s\n' "$started_at"
    printf 'target_id=%s\n' "$target"
    printf 'mode=%s\n' "$mode"
    printf 'fixed_k=%s\n' "$fixed_k"
    printf 'config_rel=%s\n' "$config_rel"
  } > "$meta_file"
  printf 'running\n' > "$status_file"

  echo "[run] $target $mode ${fixed_k:-adaptive} ($key)"
  exit_status=0
  if "$TIME_BIN" \
    -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S' \
    -o "$time_file" \
    "$PYTHON_BIN" "$REPO_ROOT/scripts/run_experiment.py" run \
    --config "$config_path" \
    --data-root "$DATA_ROOT" \
    --from build_problem \
    --to persist \
    > "$log_file" 2>&1; then
    exit_status=0
  else
    exit_status=$?
  fi

  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\n' "$exit_status" > "$status_file"
  {
    printf 'action=%s\n' "$([[ "$exit_status" == "0" ]] && echo completed || echo failed)"
    printf 'started_at_utc=%s\n' "$started_at"
    printf 'finished_at_utc=%s\n' "$finished_at"
    printf 'target_id=%s\n' "$target"
    printf 'mode=%s\n' "$mode"
    printf 'fixed_k=%s\n' "$fixed_k"
    printf 'config_rel=%s\n' "$config_rel"
    printf 'exit_status=%s\n' "$exit_status"
  } > "$meta_file"

  if [[ "$exit_status" != "0" ]]; then
    echo "[failed] $target $mode ${fixed_k:-adaptive}; see $log_file" >&2
    failures=$((failures + 1))
  else
    echo "[done] $target $mode ${fixed_k:-adaptive}"
  fi
done < "$MANIFEST_FILE"

# Convert the per-run records into a row-level report and a per-target
# adaptive-versus-exhaustive report. This uses only the Python standard
# library, so it does not add another project dependency.
"$PYTHON_BIN" - "$MANIFEST_FILE" "$OUTPUT_DIR" <<'PY'
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])


def read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_timing(path: Path) -> dict[str, str]:
    return read_key_value_file(path)


with manifest_path.open(encoding="utf-8", newline="") as handle:
    manifest_rows = list(csv.DictReader(handle, delimiter="\t"))

rows: list[dict[str, str]] = []
for item in manifest_rows:
    output_key = Path(item["config_rel"]).stem
    timing = parse_timing(output_dir / f"{output_key}.time")
    status_values = read_key_value_file(output_dir / f"{output_key}.status")
    meta = read_key_value_file(output_dir / f"{output_key}.meta")
    summary_path = Path(item["run_directory"]) / "summary.json"
    evaluation_path = Path(item["run_directory"]) / "evaluation.json"
    summary = load_json(summary_path)
    evaluation = load_json(evaluation_path)
    exit_status = status_values.get("value", "")
    if not exit_status:
        status_file = output_dir / f"{output_key}.status"
        if status_file.is_file():
            exit_status = status_file.read_text(encoding="utf-8").strip()
    summary_status = str(summary.get("status", ""))
    evaluation_status = str(evaluation.get("status", ""))
    completed = exit_status == "0" and summary_status == "completed"
    adaptive_cardinality = summary.get("adaptive_cardinality")
    if not isinstance(adaptive_cardinality, dict):
        adaptive_cardinality = evaluation.get("adaptive_cardinality")
    if not isinstance(adaptive_cardinality, dict):
        adaptive_cardinality = {}
    selected_k = adaptive_cardinality.get("selected_k", "")
    primary_value = evaluation.get("primary_metric_value", "")
    if primary_value is None:
        primary_value = ""
    subset = evaluation.get("subset", [])
    if not isinstance(subset, list):
        subset = []
    rows.append(
        {
            "target_id": item["target_id"],
            "mode": item["mode"],
            "fixed_k": item["fixed_k"],
            "experiment_id": item["experiment_id"],
            "config_rel": item["config_rel"],
            "status": "completed" if completed else ("failed" if exit_status not in {"", "running", "0"} else "incomplete"),
            "execution_action": meta.get("action", ""),
            "exit_status": exit_status,
            "elapsed_seconds": timing.get("elapsed_seconds", ""),
            "user_seconds": timing.get("user_seconds", ""),
            "system_seconds": timing.get("system_seconds", ""),
            "started_at_utc": meta.get("started_at_utc", ""),
            "finished_at_utc": meta.get("finished_at_utc", ""),
            "run_directory": item["run_directory"],
            "summary_json": str(summary_path),
            "evaluation_json": str(evaluation_path),
            "summary_status": summary_status,
            "evaluation_status": evaluation_status,
            "selected_k": str(selected_k),
            "primary_metric": str(evaluation.get("primary_metric", "")),
            "primary_metric_value": str(primary_value),
            "selected_receptors": ";".join(str(value) for value in subset),
            "log_file": str(output_dir / f"{output_key}.log"),
        }
    )

all_fields = [
    "target_id", "mode", "fixed_k", "experiment_id", "config_rel", "status",
    "execution_action", "exit_status", "elapsed_seconds", "user_seconds",
    "system_seconds", "started_at_utc", "finished_at_utc", "run_directory",
    "summary_json", "evaluation_json", "summary_status", "evaluation_status",
    "selected_k", "primary_metric", "primary_metric_value", "selected_receptors",
    "log_file",
]
with (output_dir / "all_configurations.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=all_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

by_target = {target: [row for row in rows if row["target_id"] == target] for target in {
    "MK14", "PPARG", "BACE1", "ESR1", "PPARA"
}}
comparison_rows: list[dict[str, str]] = []
for target in ("MK14", "PPARG", "BACE1", "ESR1", "PPARA"):
    target_rows = by_target[target]
    adaptive = next((row for row in target_rows if row["mode"] == "adaptive"), None)
    fixed = {int(row["fixed_k"]): row for row in target_rows if row["mode"] == "fixed" and row["fixed_k"]}
    row: dict[str, str] = {
        "target_id": target,
        "adaptive_status": adaptive["status"] if adaptive else "missing",
        "adaptive_selected_k": adaptive["selected_k"] if adaptive else "",
        "adaptive_seconds": adaptive["elapsed_seconds"] if adaptive else "",
        "adaptive_primary_metric": adaptive["primary_metric"] if adaptive else "",
        "adaptive_primary_metric_value": adaptive["primary_metric_value"] if adaptive else "",
    }
    fixed_elapsed: list[float] = []
    fixed_metric_rows: list[tuple[int, float]] = []
    all_fixed_complete = True
    for k in range(1, 7):
        record = fixed.get(k)
        row[f"fixed_k{k}_status"] = record["status"] if record else "missing"
        row[f"fixed_k{k}_seconds"] = record["elapsed_seconds"] if record else ""
        row[f"fixed_k{k}_primary_metric_value"] = record["primary_metric_value"] if record else ""
        if record is None or record["status"] != "completed":
            all_fixed_complete = False
        elapsed = parse_float(record["elapsed_seconds"]) if record else None
        if elapsed is None:
            all_fixed_complete = False
        else:
            fixed_elapsed.append(elapsed)
        metric = parse_float(record["primary_metric_value"]) if record else None
        if metric is not None:
            fixed_metric_rows.append((k, metric))
    adaptive_elapsed = parse_float(adaptive["elapsed_seconds"]) if adaptive else None
    complete = adaptive is not None and adaptive["status"] == "completed" and all_fixed_complete and adaptive_elapsed is not None
    exhaustive_total = sum(fixed_elapsed) if all_fixed_complete else None
    row["runtime_status"] = "complete" if complete else "incomplete"
    row["exhaustive_total_seconds"] = f"{exhaustive_total:.6f}" if exhaustive_total is not None else ""
    row["speedup"] = f"{exhaustive_total / adaptive_elapsed:.6f}" if complete and exhaustive_total is not None and adaptive_elapsed and adaptive_elapsed > 0 else ""
    if fixed_metric_rows:
        best_k, best_value = max(fixed_metric_rows, key=lambda pair: (pair[1], -pair[0]))
        row["exhaustive_best_k_by_primary_metric"] = str(best_k)
        row["exhaustive_best_primary_metric_value"] = f"{best_value:.12g}"
        row["adaptive_matches_exhaustive_best"] = str(adaptive is not None and adaptive["selected_k"] == str(best_k)).lower()
    else:
        row["exhaustive_best_k_by_primary_metric"] = ""
        row["exhaustive_best_primary_metric_value"] = ""
        row["adaptive_matches_exhaustive_best"] = ""
    comparison_rows.append(row)

comparison_fields = [
    "target_id", "runtime_status", "adaptive_status", "adaptive_selected_k",
    "adaptive_seconds", "adaptive_primary_metric", "adaptive_primary_metric_value",
    *[field for k in range(1, 7) for field in (
        f"fixed_k{k}_status", f"fixed_k{k}_seconds", f"fixed_k{k}_primary_metric_value"
    )],
    "exhaustive_total_seconds", "speedup", "exhaustive_best_k_by_primary_metric",
    "exhaustive_best_primary_metric_value", "adaptive_matches_exhaustive_best",
]
with (output_dir / "adaptive_vs_exhaustive.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=comparison_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(comparison_rows)

print(f"Wrote {output_dir / 'all_configurations.csv'}")
print(f"Wrote {output_dir / 'adaptive_vs_exhaustive.csv'}")
PY

if [[ "$failures" != "0" ]]; then
  echo "$failures configuration run(s) failed; inspect per-run .log files in $OUTPUT_DIR" >&2
  exit 1
fi

echo "Runtime benchmark complete. Reports are under: $OUTPUT_DIR"
