"""Independently audit the Stage27 fixed-k Pareto frontier."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import file_sha256, read_csv, read_json, rooted
from scripts.run_stage22_structural_state_coverage_qubo import build_coverage_terms, coefficient_stats, qubo_hash
from scripts.run_stage27_fixed_k_pareto_frontier import (
    benefit,
    build_fixed_k_qubo,
    exact_fixed_k,
    fixed_swap_search,
    greedy_frontier,
    load_stage27_target,
    supported_cost_intervals,
)


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def parse_subset(value: str, ids: list[str], target_k: int) -> tuple[int, ...]:
    members = tuple(part for part in value.split("+") if part)
    if len(members) != target_k or len(members) != len(set(members)) or not set(members).issubset(ids):
        raise ValueError("invalid fixed-k subset")
    return tuple(sorted(ids.index(value) for value in members))


def audit(config_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    if result.get("status") != "stage27_fixed_k_pareto_frontier_complete":
        raise ValueError("unexpected Stage27 status")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("config hash differs")
    if result["implementation"]["sha256"] != file_sha256(rooted(root, result["implementation"]["path"])):
        raise ValueError("implementation hash differs")
    paths: dict[str, Path] = {}
    for key, descriptor in result["outputs"].items():
        path = rooted(root, descriptor["path"])
        if descriptor["sha256"] != file_sha256(path):
            raise ValueError(f"output hash differs: {key}")
        paths[key] = path
    frontiers = read_csv(paths["frontier_csv"])
    solvers = read_csv(paths["solver_csv"])
    batches = read_csv(paths["batch_csv"])
    reads = read_csv(paths["read_csv"])
    intervals = read_csv(paths["cost_interval_csv"])
    models = read_csv(paths["model_csv"])
    frontier_index = {(row["target_id"], int(row["k"])): row for row in frontiers}
    model_index = {(row["target_id"], int(row["k"])): row for row in models}
    solver_groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    read_groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    batch_groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    interval_groups: dict[str, list[dict[str, str]]] = {}
    for row in solvers:
        solver_groups.setdefault((row["target_id"], int(row["k"])), []).append(row)
    for row in reads:
        read_groups.setdefault((row["target_id"], int(row["k"])), []).append(row)
    for row in batches:
        batch_groups.setdefault((row["target_id"], int(row["k"])), []).append(row)
    for row in intervals:
        interval_groups.setdefault(row["target_id"], []).append(row)
    objective = config["objective"]
    gate = config["gate"]
    checked_reads = 0
    checked_solvers = 0
    checked_models = 0
    checked_exact = 0
    all_stable = True
    all_exact = True
    strict_wins = 0
    nontrivial_targets = 0
    for target_id, spec in config["targets"].items():
        target = load_stage27_target(root, target_id, spec)
        ids, matrix = target["ids"], target["matrix"]
        terms = build_coverage_terms(ids, matrix, float(objective["neighborhood_fraction"]))
        masks = [int(terms["coverage_masks"][value]) for value in ids]
        direct_frontier = greedy_frontier(len(ids), masks, matrix, objective)
        benefits = {0: 0.0}
        for target_k_value in objective["k_values"]:
            target_k = int(target_k_value)
            key = (target_id, target_k)
            solver_rows = solver_groups[key]
            classical = []
            for row in solver_rows:
                indices = parse_subset(row["selected_subset"], ids, target_k)
                metrics = benefit(indices, masks, matrix, objective)
                close(float(row["composite_objective"]), float(metrics["composite_objective"]), "solver objective")
                if row["method"] != "exact_oracle":
                    classical.append(float(metrics["composite_objective"]))
                checked_solvers += 1
            direct, direct_metrics, _ = fixed_swap_search(direct_frontier[target_k], len(ids), masks, matrix, objective)
            direct_row = next(row for row in solver_rows if row["method"] == "direct_greedy_plus_fixed_swap")
            if parse_subset(direct_row["selected_subset"], ids, target_k) != direct:
                raise ValueError(f"{target_id}/k{target_k}: direct solver differs")
            close(float(direct_row["composite_objective"]), float(direct_metrics["composite_objective"]), "direct solver")
            exact = None
            if len(ids) <= int(config["exact_oracle"]["maximum_candidate_count"]):
                exact = exact_fixed_k(len(ids), target_k, masks, matrix, objective, int(config["exact_oracle"]["state_limit_per_k"]))
                exact_row = next(row for row in solver_rows if row["method"] == "exact_oracle")
                if exact is None or parse_subset(exact_row["selected_subset"], ids, target_k) != exact["indices"]:
                    raise ValueError(f"{target_id}/k{target_k}: exact oracle differs")
                checked_exact += 1
            strong = max(classical)
            local_reads = read_groups[key]
            expected_reads = int(config["sampler"]["batch_count"]) * int(config["sampler"]["reads_per_batch"])
            if len(local_reads) != expected_reads:
                raise ValueError(f"{target_id}/k{target_k}: read count differs")
            best_by_batch: dict[int, float] = {}
            for row in local_reads:
                indices = parse_subset(row["selected_subset"], ids, target_k)
                metrics = benefit(indices, masks, matrix, objective)
                close(float(row["composite_objective"]), float(metrics["composite_objective"]), "read objective")
                close(float(row["delta_vs_strong_classical"]), float(metrics["composite_objective"]) - strong, "read delta")
                batch = int(row["batch"])
                best_by_batch[batch] = max(best_by_batch.get(batch, -math.inf), float(metrics["composite_objective"]))
                checked_reads += 1
            local_batches = batch_groups[key]
            if len(local_batches) != int(config["sampler"]["batch_count"]):
                raise ValueError("batch count differs")
            for row in local_batches:
                close(float(row["best_objective"]), best_by_batch[int(row["batch"])], "batch best")
            sampler_value = max(best_by_batch.values())
            exact_value = None if exact is None else float(exact["metrics"]["composite_objective"])
            best_known = max([strong, sampler_value] + ([] if exact_value is None else [exact_value]))
            frontier = frontier_index[key]
            close(float(frontier["best_known_benefit"]), best_known, "best known")
            close(float(frontier["strong_classical_benefit"]), strong, "strong classical")
            close(float(frontier["sampler_benefit"]), sampler_value, "sampler")
            within = sum(value >= sampler_value - float(gate["objective_tolerance"]) for value in best_by_batch.values()) / len(best_by_batch)
            close(float(frontier["within_tolerance_batch_fraction"]), within, "stability")
            stable = within >= float(gate["minimum_batch_fraction_within_tolerance"])
            exact_gap = None if exact_value is None else exact_value - sampler_value
            exact_ok = exact_gap is None or exact_gap <= float(gate["maximum_exact_gap"]) + 1e-12
            all_stable = all_stable and stable
            all_exact = all_exact and exact_ok
            if sampler_value - strong > float(gate["minimum_gain_vs_strong_classical"]):
                strict_wins += 1
            benefits[target_k] = best_known
            model = model_index[key]
            qubo = build_fixed_k_qubo(ids, matrix, terms, target_k, objective)
            stats = coefficient_stats(qubo)
            if model["qubo_sha256"] != qubo_hash(qubo):
                raise ValueError(f"{target_id}/k{target_k}: QUBO hash differs")
            if int(model["variable_count"]) != len(qubo["variables"]):
                raise ValueError("QUBO variable count differs")
            close(float(model["coefficient_dynamic_range"]), float(stats["coefficient_dynamic_range"]), "QUBO dynamic range")
            checked_models += 1
        expected_intervals = supported_cost_intervals(benefits)
        recorded_intervals = sorted(interval_groups[target_id], key=lambda row: int(row["k"]))
        if len(expected_intervals) != len(recorded_intervals):
            raise ValueError(f"{target_id}: interval count differs")
        for expected, recorded in zip(expected_intervals, recorded_intervals):
            if int(recorded["k"]) != int(expected["k"]):
                raise ValueError(f"{target_id}: supported k differs")
            close(float(recorded["cost_lower_inclusive"]), float(expected["cost_lower_inclusive"]), "cost lower")
            if expected["cost_upper_inclusive"] == "inf":
                if recorded["cost_upper_inclusive"] != "inf":
                    raise ValueError("infinite cost interval differs")
            else:
                close(float(recorded["cost_upper_inclusive"]), float(expected["cost_upper_inclusive"]), "cost upper")
        if any(0 < int(row["k"]) < int(objective["maximum_selected"]) for row in expected_intervals):
            nontrivial_targets += 1
    frontier_gate = all_stable and all_exact and nontrivial_targets >= int(gate["minimum_targets_with_nontrivial_supported_positive_k"])
    solver_novelty = strict_wins >= int(gate["minimum_target_k_cells_strictly_above_strong_classical"])
    direct_qpu = all(row["direct_qpu_ready_under_frozen_thresholds"].lower() == "true" for row in frontiers)
    expected_decision = {"sampler_stability_gate_passed": all_stable, "exactness_gate_passed": all_exact, "targets_with_nontrivial_supported_positive_k": nontrivial_targets, "frontier_validity_gate_passed": frontier_gate, "target_k_cells_strictly_above_strong_classical": strict_wins, "solver_novelty_gate_passed": solver_novelty, "direct_qpu_readiness_gate_passed": direct_qpu}
    for key, value in expected_decision.items():
        if result["decision"][key] != value:
            raise ValueError(f"decision differs: {key}")
    if any(int(value) != 0 for value in result["data_boundary"].values()):
        raise ValueError("nonzero data boundary")
    audit_result = {"schema_version": "1.0", "status": "stage27_fixed_k_pareto_frontier_audit_ok", "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)}, "result": {"path": result_path.relative_to(root).as_posix(), "sha256": file_sha256(result_path)}, "coverage": {"read_rows_recomputed": checked_reads, "solver_rows_recomputed": checked_solvers, "exact_cells_recomputed": checked_exact, "model_rows_recomputed": checked_models, "target_count": len(config["targets"]), "frontier_cell_count": len(frontiers)}, "decision_recomputed": expected_decision, "checks": {"all_output_hashes_verified": True, "all_read_and_solver_objectives_recomputed": True, "direct_and_exact_solvers_recomputed": True, "all_cost_intervals_recomputed": True, "all_qubo_hashes_recomputed": True, "data_boundary_zero": True, "new_docking_jobs": 0, "quantum_hardware_jobs": 0}}
    output = rooted(root, output_path.as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage27_fixed_k_pareto_frontier.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/stage27_fixed_k_pareto_frontier_audit.json"))
    args = parser.parse_args()
    audit(args.config, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
