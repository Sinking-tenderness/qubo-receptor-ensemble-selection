"""Independently audit Stage80 aggregations and its no-hardware decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid Stage80 boolean: {value}")


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    result_path = root / config["outputs"]["result_json"]
    result = read_json(result_path)
    metrics_path = root / result["outputs"]["metrics_csv"]["path"]
    if sha256(metrics_path) != result["outputs"]["metrics_csv"]["sha256"]:
        raise ValueError("Stage80 metrics identity differs")
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {
        (row["target_id"], int(row["outer_fold"]), int(row["k"])) for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("Stage80 contains duplicate canonical subproblems")

    trap_count = 0
    multi_count = 0
    for row in rows:
        single = float(row["best_single_delta"]) < -TOLERANCE
        pair = float(row["best_pair_delta"]) < -TOLERANCE
        tabu = float(row["tabu_delta"]) < -TOLERANCE
        multi = tabu and int(row["tabu_selected_move_count"]) >= 2
        trap = (not single and (pair or tabu)) or (
            float(row["tabu_delta"]) < float(row["steepest_delta"]) - TOLERANCE
            and int(row["tabu_selected_move_count"]) >= 2
        )
        if single != as_bool(row["single_improvable"]):
            raise ValueError("Stage80 single-improvement label differs")
        if pair != as_bool(row["pair_improvable"]):
            raise ValueError("Stage80 pair-improvement label differs")
        if tabu != as_bool(row["tabu_improvable"]):
            raise ValueError("Stage80 tabu-improvement label differs")
        if multi != as_bool(row["multi_move_tabu_improvement"]):
            raise ValueError("Stage80 multi-move label differs")
        if trap != as_bool(row["local_trap_candidate"]):
            raise ValueError("Stage80 trap label differs")
        if int(row["qci_total_binary_levels"]) != 2 * int(row["encoded_move_count"]):
            raise ValueError("Stage80 QCI level accounting differs")
        trap_count += int(trap)
        multi_count += int(multi)

    summary = result["summary"]
    expected_targets = dict(sorted(Counter(row["target_id"] for row in rows).items()))
    checks = {
        "subproblem_count": len(rows) == int(summary["subproblem_count"]),
        "target_counts": expected_targets == summary["target_counts"],
        "single_count": sum(
            float(row["best_single_delta"]) < -TOLERANCE for row in rows
        )
        == int(summary["single_improvable_count"]),
        "pair_count": sum(float(row["best_pair_delta"]) < -TOLERANCE for row in rows)
        == int(summary["pair_improvable_count"]),
        "tabu_count": sum(float(row["tabu_delta"]) < -TOLERANCE for row in rows)
        == int(summary["tabu_improvable_count"]),
        "multi_count": multi_count == int(summary["multi_move_tabu_improvement_count"]),
        "trap_count": trap_count == int(summary["local_trap_candidate_count"]),
        "no_cloud_or_hardware": result["data_boundary"]["qci_cloud_queries"] == 0
        and result["data_boundary"]["quantum_hardware_jobs"] == 0,
        "decision_consistent": bool(
            result["decision"]["additional_qci_local_scaling_run_authorized"]
        )
        == (trap_count >= int(config["decision_gate"]["minimum_local_trap_candidate_count"])
            and multi_count >= int(config["decision_gate"]["minimum_multi_move_tabu_improvement_count"])),
    }
    if not all(checks.values()):
        raise ValueError(f"Stage80 independent audit failed: {checks}")
    audit = {
        "schema_version": "1.0",
        "status": "stage80_local_move_hardness_independent_audit_ok",
        "rows_audited": len(rows),
        "checks": checks,
        "local_trap_candidate_count": trap_count,
        "multi_move_tabu_improvement_count": multi_count,
        "qci_cloud_queries_observed": 0,
        "quantum_hardware_jobs_observed": 0,
    }
    output = root / config["outputs"]["audit_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage80_local_move_hardness_screen.json"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    audit = run((root / args.config).resolve(), root)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
