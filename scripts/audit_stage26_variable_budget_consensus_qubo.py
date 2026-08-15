"""Independently audit the Stage26 variable-budget consensus QUBO benchmark."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    distance_matrix,
    file_sha256,
    load_target,
    read_csv,
    read_json,
    rooted,
)
from scripts.run_stage22_structural_state_coverage_qubo import (
    build_coverage_terms,
    coefficient_stats,
    qubo_hash,
)
from scripts.run_stage26_variable_budget_consensus_qubo import (
    build_consensus_qubo,
    direct_greedy,
    evaluate,
    exact_oracle,
    nested_ids,
    subset_matrix,
    variable_local_search,
)


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def parse_subset(value: str, ids: list[str]) -> tuple[int, ...]:
    members = tuple(part for part in value.split("+") if part)
    if len(members) != len(set(members)) or not set(members).issubset(ids):
        raise ValueError("invalid recorded subset")
    return tuple(sorted(ids.index(value) for value in members))


def audit(config_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    if result.get("status") != "stage26_variable_budget_consensus_qubo_complete":
        raise ValueError("unexpected Stage26 status")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("config hash differs")
    if result["implementation"]["sha256"] != file_sha256(rooted(root, result["implementation"]["path"])):
        raise ValueError("implementation hash differs")
    output_paths: dict[str, Path] = {}
    for key, descriptor in result["outputs"].items():
        path = rooted(root, descriptor["path"])
        if descriptor["sha256"] != file_sha256(path):
            raise ValueError(f"output hash differs: {key}")
        output_paths[key] = path
    solver_rows = read_csv(output_paths["solver_csv"])
    batch_rows = read_csv(output_paths["batch_csv"])
    read_rows = read_csv(output_paths["read_csv"])
    model_rows = read_csv(output_paths["model_csv"])
    solver_index = {(row["target_id"], int(row["instance_size"]), row["method"], int(row["beam_width"])): row for row in solver_rows}
    model_index = {(row["target_id"], int(row["instance_size"])): row for row in model_rows}
    reads_by_instance: dict[tuple[str, int], list[dict[str, str]]] = {}
    batches_by_instance: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in read_rows:
        reads_by_instance.setdefault((row["target_id"], int(row["instance_size"])), []).append(row)
    for row in batch_rows:
        batches_by_instance.setdefault((row["target_id"], int(row["instance_size"])), []).append(row)
    objective = config["objective"]
    gate = config["gate"]
    checked_reads = 0
    checked_solvers = 0
    checked_models = 0
    full_records = []
    small_records = []
    for target_id, spec in config["targets"].items():
        target = load_target(root, target_id, spec)
        full_ids = target["ids"]
        full_matrix = distance_matrix(full_ids, target["distances"])
        order = nested_ids(target_id, full_ids, spec["reference_id"])
        recorded_target = result["target_records"][target_id]
        if recorded_target["candidate_count"] != len(full_ids):
            raise ValueError(f"{target_id}: candidate count differs")
        record_index = {int(row["instance_size"]): row for row in recorded_target["instances"]}
        for size_value in spec["instance_sizes"]:
            size = int(size_value)
            ids = order[:size]
            matrix = subset_matrix(full_ids, full_matrix, ids)
            terms = build_coverage_terms(ids, matrix, float(objective["neighborhood_fraction"]))
            masks = [int(terms["coverage_masks"][value]) for value in ids]
            record = record_index[size]
            classical_values = []
            direct = direct_greedy(size, masks, matrix, objective)
            direct, direct_metrics, _ = variable_local_search(direct, size, masks, matrix, objective)
            row = solver_index[(target_id, size, "direct_greedy_plus_variable_local_search", 0)]
            if parse_subset(row["selected_subset"], ids) != direct:
                raise ValueError(f"{target_id}/{size}: direct subset differs")
            close(float(row["composite_objective"]), float(direct_metrics["composite_objective"]), "direct objective")
            classical_values.append(float(direct_metrics["composite_objective"]))
            checked_solvers += 1
            for width_value in config["classical_baselines"]["beam_widths"]:
                width = int(width_value)
                row = solver_index[(target_id, size, "beam_plus_variable_local_search", width)]
                indices = parse_subset(row["selected_subset"], ids)
                metrics = evaluate(indices, masks, matrix, objective)
                close(float(row["composite_objective"]), float(metrics["composite_objective"]), f"beam {width} objective")
                classical_values.append(float(metrics["composite_objective"]))
                checked_solvers += 1
            exact = exact_oracle(size, masks, matrix, objective, int(config["instances"]["exact_oracle_state_limit"]))
            if exact is not None:
                row = solver_index[(target_id, size, "exact_oracle", 0)]
                if parse_subset(row["selected_subset"], ids) != exact["indices"]:
                    raise ValueError(f"{target_id}/{size}: exact subset differs")
                close(float(row["composite_objective"]), float(exact["metrics"]["composite_objective"]), "exact objective")
                checked_solvers += 1
            strong = max(classical_values)
            close(float(record["strong_classical_objective"]), strong, "strong classical")
            local_reads = reads_by_instance[(target_id, size)]
            expected_reads = int(config["sampler"]["batch_count"]) * int(config["sampler"]["reads_per_batch"])
            if len(local_reads) != expected_reads:
                raise ValueError(f"{target_id}/{size}: read count differs")
            best_by_batch: dict[int, float] = {}
            for read in local_reads:
                indices = parse_subset(read["selected_subset"], ids)
                metrics = evaluate(indices, masks, matrix, objective)
                close(float(read["composite_objective"]), float(metrics["composite_objective"]), "sample objective")
                close(float(read["delta_vs_strong_classical"]), float(metrics["composite_objective"]) - strong, "sample delta")
                batch = int(read["batch"])
                best_by_batch[batch] = max(best_by_batch.get(batch, -math.inf), float(metrics["composite_objective"]))
                checked_reads += 1
            local_batches = batches_by_instance[(target_id, size)]
            if len(local_batches) != int(config["sampler"]["batch_count"]):
                raise ValueError(f"{target_id}/{size}: batch count differs")
            for batch in local_batches:
                close(float(batch["best_objective"]), best_by_batch[int(batch["batch"])], "batch best")
            best = max(best_by_batch.values())
            close(float(record["best_sampler_objective"]), best, "sampler best")
            model = model_index[(target_id, size)]
            qubo = build_consensus_qubo(ids, matrix, terms, objective)
            stats = coefficient_stats(qubo)
            if model["qubo_sha256"] != qubo_hash(qubo):
                raise ValueError(f"{target_id}/{size}: QUBO hash differs")
            if int(model["variable_count"]) != len(qubo["variables"]):
                raise ValueError(f"{target_id}/{size}: variable count differs")
            close(float(model["coefficient_dynamic_range"]), float(stats["coefficient_dynamic_range"]), "dynamic range")
            checked_models += 1
            if record["full_instance"]:
                full_records.append(record)
            if exact is not None:
                small_records.append(record)
    stable = all(float(row["within_tolerance_batch_fraction"]) >= float(gate["minimum_full_instance_batch_fraction_within_tolerance"]) for row in full_records)
    winning = sum(float(row["delta_vs_strong_classical"]) > float(gate["minimum_gain_vs_strong_classical"]) for row in full_records)
    exact_ok = all(float(row["sampler_gap_to_exact"]) <= float(gate["maximum_small_instance_gap_to_exact"]) + 1e-12 for row in small_records)
    novelty = stable and exact_ok and winning >= int(gate["minimum_full_targets_strictly_above_strong_classical"])
    direct_ready = all(bool(row["direct_qpu_ready_under_frozen_thresholds"]) for row in full_records)
    decision = result["decision"]
    expected = {
        "sampler_stability_gate_passed": stable,
        "small_exactness_gate_passed": exact_ok,
        "full_targets_strictly_above_strong_classical": winning,
        "optimization_novelty_gate_passed": novelty,
        "direct_qpu_readiness_gate_passed": direct_ready,
    }
    for key, value in expected.items():
        if decision[key] != value:
            raise ValueError(f"decision differs: {key}")
    if any(int(value) != 0 for value in result["data_boundary"].values()):
        raise ValueError("nonzero data boundary")
    audit_result = {
        "schema_version": "1.0",
        "status": "stage26_variable_budget_consensus_qubo_audit_ok",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "result": {"path": result_path.relative_to(root).as_posix(), "sha256": file_sha256(result_path)},
        "coverage": {"read_rows_recomputed": checked_reads, "solver_rows_recomputed": checked_solvers, "model_rows_recomputed": checked_models, "full_target_count": len(full_records), "exact_instance_count": len(small_records)},
        "decision_recomputed": expected,
        "checks": {"all_output_hashes_verified": True, "all_read_objectives_recomputed": True, "direct_and_exact_baselines_recomputed": True, "qubo_hashes_and_scaling_recomputed": True, "data_boundary_zero": True, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
    }
    output = rooted(root, output_path.as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage26_variable_budget_consensus_qubo.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/stage26_variable_budget_consensus_qubo_audit.json"))
    args = parser.parse_args()
    audit(args.config, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
