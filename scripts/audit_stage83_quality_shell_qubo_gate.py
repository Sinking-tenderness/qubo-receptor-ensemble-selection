"""Independently audit Stage83 quality-shell aggregation and decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean: {value}")


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    result = read_json(root / config["outputs"]["result_json"])
    tables: dict[str, list[dict[str, str]]] = {}
    identities = []
    for label in (
        "trial_metrics_csv",
        "fixed_k_metrics_csv",
        "variable_k_metrics_csv",
    ):
        descriptor = result["outputs"][label]
        path = root / descriptor["path"]
        identities.append(
            path.is_file()
            and sha256(path) == descriptor["sha256"]
            and path.stat().st_size == int(descriptor["size_bytes"])
        )
        tables[label] = read_csv(path)
    trials = tables["trial_metrics_csv"]
    cells = tables["fixed_k_metrics_csv"]
    variable = tables["variable_k_metrics_csv"]
    summary = result["summary"]
    target_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in cells:
        target_groups[row["target_id"]].append(row)
    support_fraction = sum(
        truth(row["reference_supported_by_grid"]) for row in cells
    ) / len(cells)
    fixed_fraction = sum(truth(row["frontier_competitive"]) for row in cells) / len(cells)
    variable_fraction = sum(
        truth(row["frontier_competitive"]) for row in variable
    ) / len(variable)
    target_fractions = {
        target: sum(truth(row["frontier_competitive"]) for row in rows) / len(rows)
        for target, rows in sorted(target_groups.items())
    }
    aggregation = {
        "trial_count": len(trials) == int(summary["trial_condition_count"]),
        "fixed_k_count": len(cells) == int(summary["fixed_k_cell_count"]),
        "variable_k_count": len(variable) == int(summary["variable_k_model_count"]),
        "support_fraction": abs(
            support_fraction - float(summary["reference_supported_by_grid_fraction"])
        )
        <= 1e-12,
        "fixed_fraction": abs(
            fixed_fraction - float(summary["fixed_k_frontier_competitive_fraction"])
        )
        <= 1e-12,
        "variable_fraction": abs(
            variable_fraction - float(summary["variable_k_frontier_competitive_fraction"])
        )
        <= 1e-12,
        "target_fractions": all(
            abs(
                value
                - float(summary["per_target_frontier_competitive_fraction"][target])
            )
            <= 1e-12
            for target, value in target_fractions.items()
        ),
    }
    gate = config["decision_gate"]
    independent_gate = {
        "all_level_limits_ok": all(truth(row["qci_level_limit_ok"]) for row in trials),
        "all_penalties_dominate_flip_bound": all(
            truth(row["penalty_dominates_flip_bound"]) for row in trials
        ),
        "minimum_coefficient_retention": min(
            float(row["coefficient_retention_fraction"]) for row in trials
        )
        >= float(gate["minimum_coefficient_retention_fraction"]),
        "reference_support": support_fraction
        >= float(gate["minimum_reference_support_fraction"]),
        "fixed_k_competitiveness": fixed_fraction
        >= float(gate["minimum_fixed_k_frontier_competitive_fraction"]),
        "per_target_competitiveness": min(target_fractions.values())
        >= float(gate["minimum_per_target_frontier_competitive_fraction"]),
        "variable_k_competitiveness": variable_fraction
        >= float(gate["minimum_variable_k_frontier_competitive_fraction"]),
    }
    passed = all(independent_gate.values())
    checks = {
        "output_identity": all(identities),
        "aggregation": all(aggregation.values()),
        "gate_checks_match": independent_gate == summary["gate_checks"],
        "gate_decision_consistent": passed
        == bool(result["decision"]["limited_qci_calibration_authorized"]),
        "no_cloud_or_hardware": result["data_boundary"]["qci_cloud_queries"] == 0
        and result["data_boundary"]["quantum_hardware_jobs"] == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage83 independent audit failed: {checks}")
    audit = {
        "schema_version": "1.0",
        "status": "stage83_quality_shell_qubo_independent_audit_ok",
        "trial_rows_audited": len(trials),
        "fixed_k_rows_audited": len(cells),
        "variable_k_rows_audited": len(variable),
        "checks": checks,
        "limited_qci_calibration_authorized": passed,
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
        "--config", default="configs/stage83_quality_shell_qubo_gate.json"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    audit = run((root / args.config).resolve(), root)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
