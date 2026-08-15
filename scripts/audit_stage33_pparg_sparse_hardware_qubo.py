"""Independently audit the Stage33 sparse PPARG hardware-QUBO outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import file_sha256, read_json, rooted, write_json
from scripts.run_stage30_pparg_group_balanced_state_qubo import load_inputs as load_stage30_inputs
from scripts.run_stage33_pparg_sparse_hardware_qubo import (
    build_domain_wall_qubo,
    build_model,
    dense_objective,
    qubo_energy,
    selected_to_domain_wall,
    sparse_objective,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_selected(value: str, model: dict[str, Any]) -> tuple[int, ...]:
    frame_to_index = {frame_id: index for index, frame_id in enumerate(model["frame_ids"])}
    return tuple(sorted(frame_to_index[frame_id] for frame_id in value.split("+")))


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    candidate_rows = read_csv(outputs["candidate_manifest_csv"])
    edge_rows = read_csv(outputs["sparse_edges_csv"])
    solver_rows = read_csv(outputs["solver_results_csv"])
    batch_rows = read_csv(outputs["batch_results_csv"])
    scaling_rows = read_csv(outputs["model_scaling_csv"])
    loaded = load_stage30_inputs(root, config)
    tolerance = 1e-10
    checks: dict[str, bool] = {
        "result_status": result.get("status") == "stage33_pparg_sparse_hardware_qubo_complete",
        "scaling_row_count": len(scaling_rows) == 8,
        "batch_row_count": len(batch_rows) == 32,
        "data_boundary_zero": all(int(value) == 0 for value in result["data_boundary"].values()),
    }
    maximum_objective_difference = 0.0
    maximum_dense_difference = 0.0
    maximum_qubo_residual = 0.0
    cell_checks: dict[str, Any] = {}
    recomputed_quality_losses: dict[str, float] = {}
    stage30_rows = read_csv(root / "results/runs/stage30_pparg_group_balanced_state_qubo/solver_results.csv")
    for per_start in config["candidate_scaling"]["frames_per_start"]:
        model = build_model(int(per_start), loaded, config)
        pool_id = f"sparse_m{int(per_start):03d}_n{len(model['unary']):04d}"
        candidates = [row for row in candidate_rows if row["pool_id"] == pool_id]
        edges = [row for row in edge_rows if row["pool_id"] == pool_id]
        solvers = [row for row in solver_rows if row["pool_id"] == pool_id]
        scaling = next(row for row in scaling_rows if row["pool_id"] == pool_id)
        qubo = build_domain_wall_qubo(model, float(config["sparse_objective"]["domain_wall_violation_penalty"]))
        selections_ok = True
        for row in solvers:
            selected = parse_selected(row["selected_frame_ids"], model)
            objective_difference = abs(sparse_objective(selected, model) - float(row["sparse_objective"]))
            dense_difference = abs(dense_objective(selected, model) - float(row["dense_quality_objective"]))
            residual = abs(qubo_energy(selected_to_domain_wall(selected, model), qubo) + sparse_objective(selected, model))
            maximum_objective_difference = max(maximum_objective_difference, objective_difference)
            maximum_dense_difference = max(maximum_dense_difference, dense_difference)
            maximum_qubo_residual = max(maximum_qubo_residual, residual)
            selections_ok &= objective_difference <= tolerance and dense_difference <= tolerance
        coefficient_values = [abs(value) for _, value in qubo["linear"]] + [abs(value) for _, _, value in qubo["quadratic"]]
        expected_couplers = len(qubo["quadratic"])
        expected_dense = math.comb(len(model["unary"]), 2)
        expected_reduction = 1.0 - expected_couplers / expected_dense
        scaling_ok = (
            int(scaling["sparse_x_edge_count"]) == len(model["pair"])
            and int(scaling["domain_wall_variable_count"]) == int(qubo["logical_variable_count"])
            and int(scaling["domain_wall_quadratic_coupler_count"]) == expected_couplers
            and abs(float(scaling["coupler_reduction_fraction"]) - expected_reduction) <= tolerance
            and abs(float(scaling["coefficient_dynamic_range"]) - max(coefficient_values) / min(coefficient_values)) <= tolerance
        )
        strong = next(row for row in solvers if row["method"] == "strong_classical")
        stage30_best = max(float(row["objective"]) for row in stage30_rows if int(row["candidate_count"]) == len(model["unary"]))
        recomputed_quality_losses[str(len(model["unary"]))] = stage30_best - float(strong["dense_quality_objective"])
        cell_checks[pool_id] = {
            "candidate_count_ok": len(candidates) == len(model["unary"]),
            "edge_count_ok": len(edges) == len(model["pair"]),
            "selection_metrics_ok": selections_ok,
            "scaling_metrics_ok": scaling_ok,
        }
    checks["all_cell_checks"] = all(all(values.values()) for values in cell_checks.values())
    checks["objective_recalculation"] = maximum_objective_difference <= tolerance
    checks["dense_quality_recalculation"] = maximum_dense_difference <= tolerance
    checks["domain_wall_energy_recalculation"] = maximum_qubo_residual <= float(config["gate"]["maximum_qubo_equivalence_residual"])
    checks["quality_loss_recalculation"] = all(
        abs(recomputed_quality_losses[key] - float(result["quality_loss_vs_stage30_dense_baseline"][key])) <= tolerance
        for key in recomputed_quality_losses
    )
    full_scaling = next(row for row in scaling_rows if int(row["candidate_count"]) == 1200)
    reference_scaling = next(row for row in scaling_rows if int(row["candidate_count"]) == int(config["gate"]["direct_qpu_reference_candidate_count"]))
    recomputed_decision = {
        "full_pool_sparsity_gate_passed": float(full_scaling["coupler_reduction_fraction"]) >= float(config["gate"]["minimum_full_pool_coupler_reduction_fraction"]),
        "dense_structural_quality_retention_gate_passed": max(recomputed_quality_losses.values()) <= float(config["gate"]["maximum_dense_quality_loss"]),
        "direct_qpu_reference_cell_ready": reference_scaling["direct_qpu_ready_under_frozen_thresholds"].lower() == "true",
    }
    checks["decision_recalculation"] = all(result["decision"][key] == value for key, value in recomputed_decision.items())
    status = "stage33_pparg_sparse_hardware_qubo_audit_ok" if all(checks.values()) else "stage33_pparg_sparse_hardware_qubo_audit_failed"
    record = {
        "schema_version": "1.0",
        "status": status,
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "result": {"path": result_path.relative_to(root).as_posix(), "sha256": file_sha256(result_path)},
        "checks": checks,
        "cell_checks": cell_checks,
        "maximum_absolute_differences": {
            "sparse_objective": maximum_objective_difference,
            "dense_quality": maximum_dense_difference,
            "domain_wall_energy": maximum_qubo_residual,
        },
        "recomputed_quality_losses": recomputed_quality_losses,
        "recomputed_decision": recomputed_decision,
    }
    write_json(outputs["audit_json"], record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage33_pparg_sparse_hardware_qubo.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    record = audit(rooted(root, args.config), root)
    print(json.dumps({"status": record["status"], "checks": record["checks"]}, indent=2, sort_keys=True))
    return 0 if record["status"].endswith("_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
