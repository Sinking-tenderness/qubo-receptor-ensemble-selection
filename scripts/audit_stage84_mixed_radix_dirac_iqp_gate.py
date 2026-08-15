"""Independently audit Stage84 mixed-radix encoding metrics and decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def truth(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean: {value}")


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    result = read_json(root / config["outputs"]["result_json"])
    descriptor = result["outputs"]["metrics_csv"]
    metrics_path = root / descriptor["path"]
    identity_ok = (
        metrics_path.is_file()
        and sha256(metrics_path) == descriptor["sha256"]
        and metrics_path.stat().st_size == int(descriptor["size_bytes"])
    )
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = result["summary"]
    gate = config["decision_gate"]
    maximum_float32_error = max(
        float(row["maximum_float32_restored_energy_error"]) for row in rows
    )
    minimum_precision_margin = min(
        float(row["constraint_penalty_to_float32_error_ratio"]) for row in rows
    )
    independent_gate = {
        "all_level_limits_ok": all(truth(row["qci_level_limit_ok"]) for row in rows),
        "all_degree_two": all(int(row["polynomial_max_degree"]) == 2 for row in rows),
        "all_weights_dominate_objective_range": all(
            truth(row["constraint_weight_dominates_objective_range"])
            for row in rows
        ),
        "all_reference_residuals_zero": all(
            truth(row["reference_constraint_residual_zero"]) for row in rows
        ),
        "all_random_feasible_residuals_zero": all(
            int(row["feasible_zero_residual_count"])
            == int(row["feasible_assignment_count_tested"])
            for row in rows
        ),
        "all_random_infeasible_penalties_positive": all(
            int(row["infeasible_positive_penalty_count"])
            == int(row["infeasible_assignment_count_tested"])
            for row in rows
        ),
        "coefficient_retention": min(
            float(row["coefficient_retention_fraction"]) for row in rows
        )
        >= float(gate["minimum_coefficient_retention_fraction"]),
        "float64_identity": max(
            float(row["maximum_float64_energy_identity_error"]) for row in rows
        )
        <= float(gate["maximum_float64_energy_identity_error"]),
        "float32_error": maximum_float32_error
        <= float(gate["maximum_float32_restored_energy_error"]),
        "precision_margin": minimum_precision_margin
        >= float(gate["minimum_constraint_penalty_to_error_ratio"]),
    }
    aggregation = {
        "row_count": len(rows) == int(summary["fixed_k_encoding_count"]),
        "maximum_variables": max(int(row["integer_variable_count"]) for row in rows)
        == int(summary["maximum_integer_variable_count"]),
        "maximum_levels": max(int(row["total_qci_levels"]) for row in rows)
        == int(summary["maximum_total_qci_levels"]),
        "maximum_terms": max(int(row["polynomial_term_count"]) for row in rows)
        == int(summary["maximum_polynomial_term_count"]),
        "float32_error": abs(
            maximum_float32_error
            - float(summary["maximum_float32_restored_energy_error"])
        )
        <= 1e-12,
        "precision_margin": abs(
            minimum_precision_margin
            - float(summary["minimum_constraint_penalty_to_error_ratio"])
        )
        <= 1e-9,
    }
    passed = all(independent_gate.values())
    checks = {
        "output_identity": identity_ok,
        "aggregation": all(aggregation.values()),
        "gate_checks_match": independent_gate == summary["gate_checks"],
        "gate_decision_consistent": passed
        == bool(result["decision"]["external_dirac_calibration_preparation_authorized"]),
        "no_cloud_or_hardware": result["data_boundary"]["qci_cloud_queries"] == 0
        and result["data_boundary"]["quantum_hardware_jobs"] == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage84 independent audit failed: {checks}")
    audit = {
        "schema_version": "1.0",
        "status": "stage84_mixed_radix_dirac_iqp_independent_audit_ok",
        "encoding_rows_audited": len(rows),
        "checks": checks,
        "external_dirac_calibration_preparation_authorized": passed,
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
        "--config", default="configs/stage84_mixed_radix_dirac_iqp_gate.json"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    audit = run((root / args.config).resolve(), root)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
