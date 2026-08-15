"""Independently audit the Stage105 scenario-robust portfolio diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage105_scenario_robust_portfolio_diagnosis as stage105


TARGETS = ("BACE1", "EGFR", "FA10", "PPARA", "PPARD", "PPARG")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_close(actual: float, expected: float, message: str) -> None:
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), message)


def as_bool(value: str) -> bool:
    require(value in {"True", "False"}, f"expected Boolean CSV value, received {value!r}")
    return value == "True"


def audit(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(root / config_path)
    stage105.validate_hashes(root, config)
    outputs = config["outputs"]
    result = read_json(root / outputs["result_json"])
    rows = read_csv(root / outputs["fold_csv"])
    summaries = read_csv(root / outputs["target_csv"])

    require(result["status"] == "stage105_scenario_robust_portfolio_diagnosis_complete", "unexpected result status")
    require(tuple(result["target_ids"]) == TARGETS, "unexpected target IDs")
    require(len(rows) == 104, f"expected 104 fold rows, received {len(rows)}")
    require(len(summaries) == 12, f"expected 12 target summaries, received {len(summaries)}")
    fold_counts = {"BACE1": 4, "PPARG": 4, "PPARA": 4, "PPARD": 4, "EGFR": 5, "FA10": 5}
    expected = {
        (target, fold, subset_size, solver)
        for target, count in fold_counts.items()
        for fold in (range(count) if count == 4 else range(1, count + 1))
        for subset_size in (2, 3)
        for solver in ("pair_off_baseline", "scenario_robust_milp_certificate")
    }
    observed = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"]), row["solver_id"])
        for row in rows
    }
    require(observed == expected and len(observed) == len(rows), "incomplete or duplicate fold coverage")
    certificates = [row for row in rows if row["solver_id"] == "scenario_robust_milp_certificate"]
    baselines = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in rows
        if row["solver_id"] == "pair_off_baseline"
    }
    require(len(certificates) == 52 == result["certificate_count"], "certificate count mismatch")
    for row in rows:
        require(not as_bool(row["uses_outer_labels_for_selection"]), "outer labels leaked to Stage105 selection")
        require(float(row["minimum_jackknife_quality_margin"]) >= -1e-10, "scenario quality constraint violated")
        require(math.isfinite(float(row["holdout_robust_bedroc"])), "non-finite robust BEDROC")
    require(all(float(row["milp_gap"]) == 0.0 for row in certificates), "nonzero MILP certificate gap")
    require(all(row["selected_subset"] == baselines[(row["target_id"], int(row["outer_fold"]), int(row["subset_size"]))]["selected_subset"] for row in certificates), "Stage105 should exactly collapse to baseline")

    target_stats: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        selected = [row for row in certificates if row["target_id"] == target]
        gains = [
            float(row["holdout_robust_bedroc"])
            - float(baselines[(target, int(row["outer_fold"]), int(row["subset_size"]))]["holdout_robust_bedroc"])
            for row in selected
        ]
        reductions = [
            float(baselines[(target, int(row["outer_fold"]), int(row["subset_size"]))]["stable_redundancy_mean"])
            - float(row["stable_redundancy_mean"])
            for row in selected
        ]
        target_stats[target] = {"gain": sum(gains) / len(gains), "redundancy": sum(reductions) / len(reductions)}
    require(result["target_mean_gain_and_redundancy"] == target_stats, "target summary does not reproduce")
    gate = config["evaluation"]["diagnostic_gate"]
    gains = [value["gain"] for value in target_stats.values()]
    reductions = [value["redundancy"] for value in target_stats.values()]
    expected_checks = {
        "mean_target_gain": sum(gains) / len(gains) >= float(gate["minimum_mean_target_gain_over_pair_off"]),
        "worst_target_gain": min(gains) >= float(gate["minimum_worst_target_gain_over_pair_off"]),
        "target_count_within_0p01": sum(value >= -0.01 for value in gains) >= int(gate["minimum_target_count_within_0p01"]),
        "mean_redundancy_reduction": sum(reductions) / len(reductions) >= float(gate["minimum_mean_stable_redundancy_reduction"]),
        "target_count_nonnegative_redundancy": sum(value >= 0.0 for value in reductions) >= int(gate["minimum_target_count_with_nonnegative_redundancy_reduction"]),
    }
    require(result["diagnostic_gate"]["checks"] == expected_checks, "diagnostic gate checks do not reproduce")
    require(result["diagnostic_gate"]["passes"] is False, "scenario-robust constraint should not pass a redundancy gate")
    require(result["decision"]["replacement_objective_authorized"] is False, "Stage105 incorrectly authorized retuning")
    require(result["decision"]["new_target_protocol_authorized"] is False, "Stage105 incorrectly authorized new target")
    require(result["decision"]["parp1_released"] is False, "Stage105 incorrectly released PARP1")
    require(result["decision"]["quantum_hardware_authorized"] is False, "Stage105 incorrectly authorized hardware")
    require(all(value == 0 for value in result["data_boundary"].values()), "Stage105 data boundary breached")
    return {
        "schema_version": "1.0",
        "status": "stage105_independent_audit_ok",
        "target_count": len(TARGETS),
        "fold_metric_count": len(rows),
        "certificate_count": len(certificates),
        "changed_subset_certificate_count": 0,
        "all_scenario_constraints_satisfied": True,
        "outer_labels_used_by_selector": False,
        "parp1_released": False,
        "quantum_hardware_authorized": False,
        "data_boundary": result["data_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage105_scenario_robust_portfolio_diagnosis.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    record = audit(root, args.config)
    config = read_json(root / args.config)
    output = root / config["outputs"]["audit_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
