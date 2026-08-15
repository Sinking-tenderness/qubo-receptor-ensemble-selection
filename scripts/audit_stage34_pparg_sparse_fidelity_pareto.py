"""Independently audit the Stage34 sparse-fidelity Pareto screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import file_sha256, read_json, rooted, write_json
from scripts.run_stage30_pparg_group_balanced_state_qubo import load_inputs as load_stage30_inputs
from scripts.run_stage33_pparg_sparse_hardware_qubo import build_domain_wall_qubo, qubo_energy, selected_to_domain_wall, sparse_objective
from scripts.run_stage34_pparg_sparse_fidelity_pareto import build_model, dense_quality_objective


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    return value.lower() == "true"


def parse_selected(value: str, model: dict[str, Any]) -> tuple[int, ...]:
    lookup = {frame_id: index for index, frame_id in enumerate(model["frame_ids"])}
    return tuple(sorted(lookup[frame_id] for frame_id in value.split("+")))


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    result = read_json(outputs["result_json"])
    summary_rows = read_csv(outputs["encoding_summary_csv"])
    cell_rows = read_csv(outputs["cell_metrics_csv"])
    solver_rows = read_csv(outputs["solver_results_csv"])
    batch_rows = read_csv(outputs["batch_results_csv"])
    loaded = load_stage30_inputs(root, config)
    stage30_rows = read_csv(rooted(root, config["inputs"]["stage30_solver_results"]))
    stage30_best = {
        count: max(float(row["objective"]) for row in stage30_rows if int(row["candidate_count"]) == count)
        for count in config["screening_scales"]["candidate_counts"]
    }
    checks: dict[str, bool] = {
        "result_status": result.get("status") == "stage34_pparg_sparse_fidelity_pareto_complete",
        "summary_row_count": len(summary_rows) == 7,
        "cell_row_count": len(cell_rows) == 21,
        "batch_row_count": len(batch_rows) == 63,
        "data_boundary_zero": all(int(value) == 0 for value in result["data_boundary"].values()),
    }
    maximum_differences = {"sparse_objective": 0.0, "dense_quality": 0.0, "qubo_energy": 0.0, "cell_metric": 0.0}
    cell_checks: dict[str, bool] = {}
    gate = config["pareto_gate"]
    for encoding in config["candidate_encodings"]:
        for per_start in config["screening_scales"]["frames_per_start"]:
            model = build_model(int(per_start), loaded, encoding, config)
            count = len(model["unary"])
            cell_id = f"{encoding['encoding_id']}__n{count:04d}"
            cell = next(row for row in cell_rows if row["encoding_id"] == encoding["encoding_id"] and int(row["candidate_count"]) == count)
            solvers = [row for row in solver_rows if row["encoding_id"] == encoding["encoding_id"] and int(row["candidate_count"]) == count]
            qubo = build_domain_wall_qubo(model, float(config["sparse_encoding"]["domain_wall_violation_penalty"]))
            metrics_ok = True
            for row in solvers:
                selected = parse_selected(row["selected_frame_ids"], model)
                sparse_difference = abs(sparse_objective(selected, model) - float(row["sparse_objective"]))
                dense_difference = abs(dense_quality_objective(selected, model) - float(row["dense_quality_objective"]))
                residual = abs(qubo_energy(selected_to_domain_wall(selected, model), qubo) + sparse_objective(selected, model))
                maximum_differences["sparse_objective"] = max(maximum_differences["sparse_objective"], sparse_difference)
                maximum_differences["dense_quality"] = max(maximum_differences["dense_quality"], dense_difference)
                maximum_differences["qubo_energy"] = max(maximum_differences["qubo_energy"], residual)
                metrics_ok &= sparse_difference <= 1e-10 and dense_difference <= 1e-10
            strong = next(row for row in solvers if row["method"] == "strong_classical")
            quality_loss = stage30_best[count] - float(strong["dense_quality_objective"])
            couplers = len(qubo["quadratic"])
            reduction = 1.0 - couplers / math.comb(count, 2)
            cell_differences = [
                abs(float(cell["dense_quality_loss"]) - quality_loss),
                abs(float(cell["coupler_reduction_fraction"]) - reduction),
                abs(int(cell["domain_wall_coupler_count"]) - couplers),
                abs(int(cell["domain_wall_variable_count"]) - int(qubo["logical_variable_count"])),
            ]
            maximum_differences["cell_metric"] = max(maximum_differences["cell_metric"], *cell_differences)
            cell_checks[cell_id] = metrics_ok and max(cell_differences) <= 1e-10
    checks["all_cell_recalculations"] = all(cell_checks.values())
    checks["qubo_equivalence"] = maximum_differences["qubo_energy"] <= float(gate["maximum_qubo_equivalence_residual"])
    recomputed_summary: dict[str, dict[str, Any]] = {}
    for encoding in config["candidate_encodings"]:
        cells = [row for row in cell_rows if row["encoding_id"] == encoding["encoding_id"]]
        small = next(row for row in cells if int(row["candidate_count"]) == 32)
        reference = next(row for row in cells if int(row["candidate_count"]) == int(gate["direct_qpu_reference_candidate_count"]))
        full = next(row for row in cells if int(row["candidate_count"]) == 1200)
        values = {
            "quality_gate_passed": all(float(row["dense_quality_loss"]) <= float(gate["maximum_dense_quality_loss_at_each_gate_scale"]) for row in cells),
            "equivalence_gate_passed": all(float(row["maximum_equivalence_residual"]) <= float(gate["maximum_qubo_equivalence_residual"]) for row in cells),
            "small_exactness_gate_passed": abs(float(small["small_exact_gap"])) <= float(gate["maximum_small_exact_gap"]),
            "stability_gate_passed": all(float(row["annealing_batch_fraction_within_tolerance"]) >= float(gate["minimum_batch_fraction_within_tolerance"]) for row in cells),
            "sparsity_gate_passed": float(full["coupler_reduction_fraction"]) >= float(gate["minimum_full_pool_coupler_reduction_fraction"]),
            "direct_qpu_reference_gate_passed": parse_bool(reference["direct_qpu_ready_under_frozen_thresholds"]),
        }
        values["pareto_eligible"] = all(values.values())
        recomputed_summary[encoding["encoding_id"]] = values
    checks["summary_gate_recalculation"] = all(
        all(parse_bool(row[key]) == value for key, value in recomputed_summary[row["encoding_id"]].items())
        for row in summary_rows
    )
    eligible = [encoding_id for encoding_id, values in recomputed_summary.items() if values["pareto_eligible"]]
    checks["decision_recalculation"] = (
        int(result["decision"]["pareto_eligible_encoding_count"]) == len(eligible)
        and result["decision"]["sparse_fidelity_gate_passed"] == bool(eligible)
        and result["decision"]["small_quantum_annealing_application_pilot_authorized"] == bool(eligible)
    )
    status = "stage34_pparg_sparse_fidelity_pareto_audit_ok" if all(checks.values()) else "stage34_pparg_sparse_fidelity_pareto_audit_failed"
    record = {
        "schema_version": "1.0", "status": status,
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "result": {"path": outputs["result_json"].relative_to(root).as_posix(), "sha256": file_sha256(outputs["result_json"])},
        "checks": checks, "cell_checks": cell_checks, "maximum_absolute_differences": maximum_differences,
        "recomputed_encoding_gates": recomputed_summary,
    }
    write_json(outputs["audit_json"], record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage34_pparg_sparse_fidelity_pareto.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    record = audit(rooted(root, args.config), root)
    print(json.dumps({"status": record["status"], "checks": record["checks"]}, indent=2, sort_keys=True))
    return 0 if record["status"].endswith("_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
