"""Run the Stage27 fixed-k structural-benefit Pareto frontier benchmark."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    descriptor,
    distance_matrix,
    file_sha256,
    load_target,
    numeric,
    read_csv,
    read_json,
    rooted,
    write_csv,
    write_json,
)
from scripts.run_stage22_structural_state_coverage_qubo import (
    add_linear,
    add_quadratic,
    add_square,
    binary_assignment,
    build_coverage_terms,
    coefficient_stats,
    qubo_energy,
    qubo_hash,
    slack_weights,
)
from scripts.run_stage26_variable_budget_consensus_qubo import (
    consensus_components,
    nested_ids,
)


def matrix_from_distance_rows(
    ids: list[str], rows: list[dict[str, str]], distance_column: str
) -> np.ndarray:
    id_set = set(ids)
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        first, second = row["conformer_id_a"], row["conformer_id_b"]
        if first not in id_set or second not in id_set or first == second:
            continue
        pair = tuple(sorted((first, second)))
        if pair in values:
            raise ValueError(f"duplicate distance pair: {pair}")
        value = numeric(row, distance_column)
        if value < 0:
            raise ValueError("negative structural distance")
        values[pair] = value
    expected = len(ids) * (len(ids) - 1) // 2
    if len(values) != expected:
        raise ValueError(f"incomplete distance matrix: {len(values)} != {expected}")
    maximum = max(values.values(), default=1.0)
    scale = maximum if maximum > 0 else 1.0
    index = {value: position for position, value in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=float)
    for (first, second), value in values.items():
        i, j = index[first], index[second]
        matrix[i, j] = matrix[j, i] = value / scale
    return matrix


def load_stage27_target(root: Path, target_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    if spec["loader"] == "stage21_hard_gate":
        target = load_target(root, target_id, spec)
        ids = nested_ids(target_id, target["ids"], spec["reference_id"])
        source_ids = target["ids"]
        source_matrix = distance_matrix(source_ids, target["distances"])
        positions = [source_ids.index(value) for value in ids]
        matrix = source_matrix[np.ix_(positions, positions)]
        return {
            "ids": ids,
            "matrix": matrix,
            "input_paths": target["input_paths"],
            "hard_gate_excluded_count": len(target["excluded_hard_gate"]),
        }
    input_paths = {key: rooted(root, value) for key, value in spec["inputs"].items()}
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = read_json(input_paths["summary_json"])
    if summary.get("status") != spec["summary_status"]:
        raise ValueError(f"{target_id}: unexpected structural-pool status")
    if spec["loader"] == "summary_distance_pool":
        ids = sorted(
            {
                summary["selection_seed"],
                *summary["selected_receptor_ids"],
                *summary["reserve_receptor_ids"],
            }
        )
    elif spec["loader"] == "csv_distance_pool":
        rows = read_csv(input_paths["eligible_pool"])
        ids = sorted(row["conformer_id"] for row in rows)
        if len(ids) != len(set(ids)):
            raise ValueError(f"{target_id}: duplicate preparation-ready ID")
    else:
        raise ValueError(f"unknown loader: {spec['loader']}")
    ids = nested_ids(target_id, ids, spec["reference_id"])
    matrix = matrix_from_distance_rows(
        ids, read_csv(input_paths["pairwise_distances"]), spec["distance_column"]
    )
    return {
        "ids": ids,
        "matrix": matrix,
        "input_paths": input_paths,
        "hard_gate_excluded_count": 0,
    }


def benefit(
    selected: Iterable[int], masks: list[int], matrix: np.ndarray, objective: dict[str, Any]
) -> dict[str, float]:
    return consensus_components(
        selected,
        masks,
        matrix,
        int(objective["maximum_selected"]),
        float(objective["single_coverage_weight"]),
        float(objective["double_coverage_weight"]),
        float(objective["pair_diversity_weight"]),
        0.0,
    )


def better(
    candidate: tuple[int, ...], candidate_metrics: dict[str, float],
    current: tuple[int, ...], current_metrics: dict[str, float]
) -> bool:
    left = float(candidate_metrics["composite_objective"])
    right = float(current_metrics["composite_objective"])
    return left > right + 1e-12 or (math.isclose(left, right, abs_tol=1e-12) and candidate < current)


def greedy_frontier(
    candidate_count: int, masks: list[int], matrix: np.ndarray, objective: dict[str, Any]
) -> dict[int, tuple[int, ...]]:
    current: tuple[int, ...] = ()
    result: dict[int, tuple[int, ...]] = {}
    for target_k in range(1, int(objective["maximum_selected"]) + 1):
        selected = set(current)
        candidates = [tuple(sorted((*current, value))) for value in range(candidate_count) if value not in selected]
        current = max(
            candidates,
            key=lambda value: (float(benefit(value, masks, matrix, objective)["composite_objective"]), tuple(-v for v in value)),
        )
        result[target_k] = current
    return result


def fixed_swap_search(
    start: tuple[int, ...], candidate_count: int, masks: list[int], matrix: np.ndarray,
    objective: dict[str, Any]
) -> tuple[tuple[int, ...], dict[str, float], int]:
    current = tuple(sorted(start))
    metrics = benefit(current, masks, matrix, objective)
    iterations = 0
    while True:
        selected = set(current)
        best, best_metrics = current, metrics
        for outgoing in current:
            for incoming in range(candidate_count):
                if incoming in selected:
                    continue
                candidate = tuple(sorted((selected - {outgoing}) | {incoming}))
                candidate_metrics = benefit(candidate, masks, matrix, objective)
                if better(candidate, candidate_metrics, best, best_metrics):
                    best, best_metrics = candidate, candidate_metrics
        if float(best_metrics["composite_objective"]) <= float(metrics["composite_objective"]) + 1e-12:
            return current, metrics, iterations
        current, metrics = best, best_metrics
        iterations += 1
        if iterations > candidate_count * len(current):
            raise RuntimeError("fixed swap search exceeded iteration guard")


def beam_frontier(
    candidate_count: int, masks: list[int], matrix: np.ndarray,
    objective: dict[str, Any], width: int
) -> dict[int, tuple[int, ...]]:
    layer: list[tuple[int, ...]] = [()]
    result: dict[int, tuple[int, ...]] = {}
    for target_k in range(1, int(objective["maximum_selected"]) + 1):
        proposals = {
            tuple(sorted((*selected, incoming)))
            for selected in layer
            for incoming in range(candidate_count)
            if incoming not in selected
        }
        layer = sorted(
            proposals,
            key=lambda value: (-float(benefit(value, masks, matrix, objective)["composite_objective"]), value),
        )[:width]
        result[target_k] = layer[0]
    return result


def exact_fixed_k(
    candidate_count: int, target_k: int, masks: list[int], matrix: np.ndarray,
    objective: dict[str, Any], state_limit: int
) -> dict[str, Any] | None:
    state_count = math.comb(candidate_count, target_k)
    if state_count > state_limit:
        return None
    best: tuple[int, ...] | None = None
    best_metrics: dict[str, float] | None = None
    for selected in itertools.combinations(range(candidate_count), target_k):
        metrics = benefit(selected, masks, matrix, objective)
        if best is None or better(selected, metrics, best, best_metrics):
            best, best_metrics = selected, metrics
    if best is None or best_metrics is None:
        raise ValueError("exact fixed-k enumeration produced no state")
    return {"indices": best, "metrics": best_metrics, "state_count": state_count}


def anneal_fixed_k(
    candidate_count: int, target_k: int, masks: list[int], matrix: np.ndarray,
    objective: dict[str, Any], sweeps: int, temperature_start: float,
    temperature_end: float, rng: random.Random
) -> dict[str, Any]:
    current = tuple(sorted(rng.sample(range(candidate_count), target_k)))
    current_value = float(benefit(current, masks, matrix, objective)["composite_objective"])
    best, best_value = current, current_value
    accepted = 0
    for step in range(sweeps):
        selected = set(current)
        outgoing = rng.choice(current)
        incoming = rng.choice([value for value in range(candidate_count) if value not in selected])
        proposal = tuple(sorted((selected - {outgoing}) | {incoming}))
        proposed = float(benefit(proposal, masks, matrix, objective)["composite_objective"])
        progress = step / max(1, sweeps - 1)
        temperature = max(temperature_end, temperature_start * (temperature_end / temperature_start) ** progress)
        delta = proposed - current_value
        if delta >= 0 or rng.random() < math.exp(max(-700.0, delta / temperature)):
            current, current_value = proposal, proposed
            accepted += 1
            if current_value > best_value + 1e-12 or (
                math.isclose(current_value, best_value, abs_tol=1e-12) and current < best
            ):
                best, best_value = current, current_value
    return {"indices": best, "metrics": benefit(best, masks, matrix, objective), "acceptance_fraction": accepted / max(1, sweeps)}


def build_fixed_k_qubo(
    ids: list[str], matrix: np.ndarray, terms: dict[str, Any], target_k: int,
    objective: dict[str, Any]
) -> dict[str, Any]:
    coefficients: dict[str, Any] = {"constant": 0.0, "linear": {}, "quadratic": {}}
    x_names = {value: f"x__{value}" for value in ids}
    u_names = {value: f"u__{value}" for value in ids}
    v_names = {value: f"v__{value}" for value in ids}
    add_square(
        coefficients, -target_k, {x_names[value]: 1.0 for value in ids},
        float(objective["cardinality_penalty"]),
    )
    single_slack: list[str] = []
    double_slack: list[str] = []
    single_weights = slack_weights(int(terms["neighbor_count"]) - 1)
    double_weights = slack_weights(int(terms["neighbor_count"]))
    for state_id in ids:
        incidence = terms["incidence"][state_id]
        first = {u_names[state_id]: 1.0, **{x_names[value]: -1.0 for value in incidence}}
        for index, weight in enumerate(single_weights):
            name = f"su__{state_id}__{index}"
            single_slack.append(name)
            first[name] = float(weight)
        add_square(coefficients, 0.0, first, float(objective["coverage_constraint_penalty"]))
        second = {v_names[state_id]: 2.0, **{x_names[value]: -1.0 for value in incidence}}
        for index, weight in enumerate(double_weights):
            name = f"sv__{state_id}__{index}"
            double_slack.append(name)
            second[name] = float(weight)
        add_square(coefficients, 0.0, second, float(objective["coverage_constraint_penalty"]))
        add_linear(coefficients, u_names[state_id], -float(objective["single_coverage_weight"]) / len(ids))
        add_linear(coefficients, v_names[state_id], -float(objective["double_coverage_weight"]) / len(ids))
    denominator = max(1, math.comb(int(objective["maximum_selected"]), 2))
    for first in range(len(ids)):
        for second in range(first + 1, len(ids)):
            add_quadratic(
                coefficients, x_names[ids[first]], x_names[ids[second]],
                -float(objective["pair_diversity_weight"]) * float(matrix[first, second]) / denominator,
            )
    variables = sorted(set(coefficients["linear"]) | {value for key in coefficients["quadratic"] for value in key.split("::", 1)})
    return {
        "constant": float(coefficients["constant"]),
        "linear": {key: float(value) for key, value in coefficients["linear"].items()},
        "quadratic": {key: float(value) for key, value in coefficients["quadratic"].items()},
        "variables": variables,
        "variable_groups": {"x": sorted(x_names.values()), "single_u": sorted(u_names.values()), "double_v": sorted(v_names.values()), "single_slack": sorted(single_slack), "double_slack": sorted(double_slack)},
        "target_k": target_k,
        "convention": "minimize Q; feasible auxiliary minimum equals negative fixed-k structural benefit",
    }


def assignment_for_fixed_subset(
    selected: tuple[int, ...], ids: list[str], terms: dict[str, Any], qubo: dict[str, Any]
) -> dict[str, int]:
    if len(selected) != int(qubo["target_k"]):
        raise ValueError("assignment has wrong cardinality")
    assignment = {value: 0 for value in qubo["variables"]}
    chosen = {ids[index] for index in selected}
    for value in chosen:
        assignment[f"x__{value}"] = 1
    single_weights = slack_weights(int(terms["neighbor_count"]) - 1)
    double_weights = slack_weights(int(terms["neighbor_count"]))
    for state_id in ids:
        count = sum(value in chosen for value in terms["incidence"][state_id])
        u, v = int(count >= 1), int(count >= 2)
        assignment[f"u__{state_id}"] = u
        assignment[f"v__{state_id}"] = v
        for index, bit in binary_assignment(single_weights, count - u).items():
            assignment[f"su__{state_id}__{index}"] = bit
        for index, bit in binary_assignment(double_weights, count - 2 * v).items():
            assignment[f"sv__{state_id}__{index}"] = bit
    return assignment


def supported_cost_intervals(benefits: dict[int, float]) -> list[dict[str, Any]]:
    rows = []
    for target_k, value in sorted(benefits.items()):
        lower = 0.0
        upper = math.inf
        for other_k, other_value in benefits.items():
            if other_k > target_k:
                lower = max(lower, (other_value - value) / (other_k - target_k))
            elif other_k < target_k:
                upper = min(upper, (value - other_value) / (target_k - other_k))
        lower = max(0.0, lower)
        if upper + 1e-12 >= lower and upper >= 0.0:
            rows.append({
                "k": target_k,
                "benefit": value,
                "cost_lower_inclusive": lower,
                "cost_upper_inclusive": "inf" if math.isinf(upper) else max(0.0, upper),
            })
    return rows


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if outputs["result_json"].exists() and not overwrite:
        raise FileExistsError(f"result exists: {outputs['result_json']}; pass --overwrite")
    objective = config["objective"]
    sampler = config["sampler"]
    gate = config["gate"]
    k_values = [int(value) for value in objective["k_values"]]
    frontier_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    read_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    target_records: dict[str, Any] = {}
    input_records: dict[str, Any] = {}
    strict_wins = 0
    all_stable = True
    all_exact = True
    nontrivial_targets = 0
    for target_id, spec in config["targets"].items():
        target = load_stage27_target(root, target_id, spec)
        ids, matrix = target["ids"], target["matrix"]
        terms = build_coverage_terms(ids, matrix, float(objective["neighborhood_fraction"]))
        masks = [int(terms["coverage_masks"][value]) for value in ids]
        print(json.dumps({"target_id": target_id, "candidate_count": len(ids), "status": "running"}), flush=True)
        started = time.perf_counter()
        greedy = greedy_frontier(len(ids), masks, matrix, objective)
        greedy_runtime = time.perf_counter() - started
        beam_results: dict[int, dict[int, tuple[int, ...]]] = {}
        beam_runtimes: dict[int, float] = {}
        for width_value in config["classical_baselines"]["beam_widths"]:
            width = int(width_value)
            started = time.perf_counter()
            beam_results[width] = beam_frontier(len(ids), masks, matrix, objective, width)
            beam_runtimes[width] = time.perf_counter() - started
        records = []
        benefits = {0: 0.0}
        for target_k in k_values:
            classical_values = []
            selected, metrics, _ = fixed_swap_search(greedy[target_k], len(ids), masks, matrix, objective)
            solver_rows.append({"target_id": target_id, "candidate_count": len(ids), "k": target_k, "method": "direct_greedy_plus_fixed_swap", "beam_width": 0, "runtime_seconds": greedy_runtime, "selected_subset": "+".join(ids[i] for i in selected), **metrics, "state_count": ""})
            classical_values.append(float(metrics["composite_objective"]))
            for width, frontier in beam_results.items():
                selected, metrics, _ = fixed_swap_search(frontier[target_k], len(ids), masks, matrix, objective)
                solver_rows.append({"target_id": target_id, "candidate_count": len(ids), "k": target_k, "method": "beam_plus_fixed_swap", "beam_width": width, "runtime_seconds": beam_runtimes[width], "selected_subset": "+".join(ids[i] for i in selected), **metrics, "state_count": ""})
                classical_values.append(float(metrics["composite_objective"]))
            exact = None
            if len(ids) <= int(config["exact_oracle"]["maximum_candidate_count"]):
                started = time.perf_counter()
                exact = exact_fixed_k(len(ids), target_k, masks, matrix, objective, int(config["exact_oracle"]["state_limit_per_k"]))
                runtime = time.perf_counter() - started
                if exact is not None:
                    solver_rows.append({"target_id": target_id, "candidate_count": len(ids), "k": target_k, "method": "exact_oracle", "beam_width": 0, "runtime_seconds": runtime, "selected_subset": "+".join(ids[i] for i in exact["indices"]), **exact["metrics"], "state_count": exact["state_count"]})
            strong = max(classical_values)
            local_batches = []
            for batch in range(int(sampler["batch_count"])):
                seed = int(sampler["base_seed"]) + sum(ord(c) for c in target_id) + target_k * 1009 + batch * 1000003
                rng = random.Random(seed)
                local_reads = []
                started = time.perf_counter()
                for read_index in range(int(sampler["reads_per_batch"])):
                    sampled = anneal_fixed_k(len(ids), target_k, masks, matrix, objective, int(sampler["sweeps_per_read"]), float(sampler["temperature_start"]), float(sampler["temperature_end"]), rng)
                    row = {"target_id": target_id, "candidate_count": len(ids), "k": target_k, "batch": batch, "read": read_index, "seed": seed, "selected_subset": "+".join(ids[i] for i in sampled["indices"]), **sampled["metrics"], "acceptance_fraction": sampled["acceptance_fraction"], "delta_vs_strong_classical": float(sampled["metrics"]["composite_objective"]) - strong}
                    read_rows.append(row)
                    local_reads.append(row)
                runtime = time.perf_counter() - started
                best = max(local_reads, key=lambda row: (float(row["composite_objective"]), row["selected_subset"]))
                batch_row = {"target_id": target_id, "candidate_count": len(ids), "k": target_k, "batch": batch, "seed": seed, "runtime_seconds": runtime, "best_subset": best["selected_subset"], "best_objective": float(best["composite_objective"]), "delta_vs_strong_classical": float(best["composite_objective"]) - strong}
                batch_rows.append(batch_row)
                local_batches.append(batch_row)
            best_batch = max(local_batches, key=lambda row: (float(row["best_objective"]), row["best_subset"]))
            sampler_value = float(best_batch["best_objective"])
            exact_value = None if exact is None else float(exact["metrics"]["composite_objective"])
            best_known = max([strong, sampler_value] + ([] if exact_value is None else [exact_value]))
            within = sum(float(row["best_objective"]) >= sampler_value - float(gate["objective_tolerance"]) for row in local_batches) / len(local_batches)
            exact_gap = None if exact_value is None else exact_value - sampler_value
            stable = within >= float(gate["minimum_batch_fraction_within_tolerance"])
            exact_ok = exact_gap is None or exact_gap <= float(gate["maximum_exact_gap"]) + 1e-12
            all_stable = all_stable and stable
            all_exact = all_exact and exact_ok
            if sampler_value - strong > float(gate["minimum_gain_vs_strong_classical"]):
                strict_wins += 1
            selected_ids = tuple(part for part in best_batch["best_subset"].split("+") if part)
            indices = tuple(ids.index(value) for value in selected_ids)
            selected_metrics = benefit(indices, masks, matrix, objective)
            qubo = build_fixed_k_qubo(ids, matrix, terms, target_k, objective)
            energy = qubo_energy(qubo, assignment_for_fixed_subset(indices, ids, terms, qubo))
            residual = abs(energy + float(selected_metrics["composite_objective"]))
            if residual > 1e-8:
                raise ValueError(f"{target_id}/k{target_k}: QUBO equivalence failed")
            stats = coefficient_stats(qubo)
            direct_ready = len(qubo["variables"]) <= int(gate["direct_qpu_max_variables"]) and float(stats["coefficient_dynamic_range"]) <= float(gate["direct_qpu_max_coefficient_dynamic_range"])
            model_rows.append({"target_id": target_id, "candidate_count": len(ids), "k": target_k, "variable_count": len(qubo["variables"]), "quadratic_coupler_count": len(qubo["quadratic"]), "qubo_sha256": qubo_hash(qubo), "equivalence_residual": residual, "selected_energy": energy, "direct_qpu_ready_under_frozen_thresholds": direct_ready, **stats})
            row = {"target_id": target_id, "candidate_count": len(ids), "k": target_k, "best_known_benefit": best_known, "sampler_benefit": sampler_value, "strong_classical_benefit": strong, "delta_vs_strong_classical": sampler_value - strong, "exact_oracle_benefit": exact_value, "sampler_gap_to_exact": exact_gap, "within_tolerance_batch_fraction": within, "sampler_stable": stable, "exact_gate_passed": exact_ok, "best_sampler_subset": "+".join(selected_ids), "qubo_variable_count": len(qubo["variables"]), "direct_qpu_ready_under_frozen_thresholds": direct_ready}
            frontier_rows.append(row)
            records.append(row)
            benefits[target_k] = best_known
        intervals = supported_cost_intervals(benefits)
        supported_positive = [int(row["k"]) for row in intervals if 0 < int(row["k"]) < int(objective["maximum_selected"])]
        if supported_positive:
            nontrivial_targets += 1
        for row in intervals:
            interval_rows.append({"target_id": target_id, "candidate_count": len(ids), **row})
        target_records[target_id] = {"candidate_count": len(ids), "hard_gate_excluded_count": target["hard_gate_excluded_count"], "frontier": records, "supported_cost_intervals": intervals, "nontrivial_supported_positive_k": supported_positive}
        input_records[target_id] = {key: descriptor(root, path) for key, path in target["input_paths"].items()}
    write_csv(outputs["frontier_csv"], frontier_rows)
    write_csv(outputs["solver_csv"], solver_rows)
    write_csv(outputs["batch_csv"], batch_rows)
    write_csv(outputs["read_csv"], read_rows)
    write_csv(outputs["cost_interval_csv"], interval_rows)
    write_csv(outputs["model_csv"], model_rows)
    frontier_gate = all_stable and all_exact and nontrivial_targets >= int(gate["minimum_targets_with_nontrivial_supported_positive_k"])
    solver_novelty = strict_wins >= int(gate["minimum_target_k_cells_strictly_above_strong_classical"])
    direct_qpu = all(bool(row["direct_qpu_ready_under_frozen_thresholds"]) for row in frontier_rows)
    report = ["# Stage 27: fixed-k Pareto frontier", "", "Frozen benefit: 0.25 single coverage + 0.75 double coverage + 0.10 pair diversity with one common k=8 pair denominator. No per-conformer cost is optimized.", "", "| Target | n | Supported k under nonnegative cost | Strict sampler wins | Stable cells | Direct-QPU-ready cells |", "|---|---:|---|---:|---:|---:|"]
    for target_id, record in target_records.items():
        cells = record["frontier"]
        report.append(f"| {target_id} | {record['candidate_count']} | {','.join(str(row['k']) for row in record['supported_cost_intervals'])} | {sum(float(row['delta_vs_strong_classical']) > float(gate['minimum_gain_vs_strong_classical']) for row in cells)} | {sum(bool(row['sampler_stable']) for row in cells)}/{len(cells)} | {sum(bool(row['direct_qpu_ready_under_frozen_thresholds']) for row in cells)}/{len(cells)} |")
    report += ["", f"Frontier-validity gate: **{'PASS' if frontier_gate else 'NO-GO'}**.", f"Solver-novelty gate: **{'PASS' if solver_novelty else 'NO-GO'}**.", f"Direct-QPU readiness gate: **{'PASS' if direct_qpu else 'NO-GO'}**.", "", "No docking scores, ligand labels, fresh-validation rows, test rows, or quantum-hardware outputs were read."]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report) + "\n", encoding="ascii")
    result = {"schema_version": "1.0", "status": "stage27_fixed_k_pareto_frontier_complete", "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)}, "implementation": descriptor(root, Path(__file__).resolve()), "inputs": input_records, "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key != "result_json"}, "target_records": target_records, "decision": {"sampler_stability_gate_passed": all_stable, "exactness_gate_passed": all_exact, "targets_with_nontrivial_supported_positive_k": nontrivial_targets, "frontier_validity_gate_passed": frontier_gate, "target_k_cells_strictly_above_strong_classical": strict_wins, "solver_novelty_gate_passed": solver_novelty, "direct_qpu_readiness_gate_passed": direct_qpu, "new_docking_jobs_authorized_by_this_stage": False, "quantum_hardware_authorized": False}, "data_boundary": {"docking_scores_read": 0, "ligand_labels_read": 0, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0}, "interpretation_boundary": config["interpretation_boundary"]}
    write_json(outputs["result_json"], result)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage27_fixed_k_pareto_frontier.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
