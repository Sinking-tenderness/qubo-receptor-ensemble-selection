"""Run the Stage26 variable-budget two-hit structural consensus QUBO."""

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


def nested_ids(target_id: str, ids: list[str], reference_id: str) -> list[str]:
    if reference_id not in ids:
        raise ValueError(f"{target_id}: reference is absent from preparation-ready pool")
    remainder = [value for value in ids if value != reference_id]
    remainder.sort(
        key=lambda value: (
            hashlib.sha256(f"{target_id}:{value}".encode("ascii")).hexdigest(),
            value,
        )
    )
    return [reference_id, *remainder]


def subset_matrix(full_ids: list[str], full_matrix: np.ndarray, ids: list[str]) -> np.ndarray:
    index = {value: position for position, value in enumerate(full_ids)}
    positions = [index[value] for value in ids]
    return full_matrix[np.ix_(positions, positions)]


def consensus_components(
    selected: Iterable[int],
    masks: list[int],
    matrix: np.ndarray,
    maximum_selected: int,
    single_weight: float,
    double_weight: float,
    diversity_weight: float,
    per_conformer_cost: float,
) -> dict[str, float]:
    values = sorted(set(int(value) for value in selected))
    once = 0
    twice = 0
    pair_sum = 0.0
    for position, index in enumerate(values):
        mask = int(masks[index])
        twice |= once & mask
        once |= mask
        for other in values[:position]:
            pair_sum += float(matrix[index, other])
    state_count = len(masks)
    denominator = max(1, math.comb(maximum_selected, 2))
    single = once.bit_count() / state_count
    double = twice.bit_count() / state_count
    diversity = pair_sum / denominator
    cost = per_conformer_cost * len(values)
    objective = (
        single_weight * single
        + double_weight * double
        + diversity_weight * diversity
        - cost
    )
    return {
        "selected_count": float(len(values)),
        "single_coverage_fraction": float(single),
        "double_coverage_fraction": float(double),
        "normalized_pair_diversity": float(diversity),
        "selection_cost": float(cost),
        "composite_objective": float(objective),
    }


def value_key(indices: tuple[int, ...], metrics: dict[str, float]) -> tuple[float, tuple[int, ...]]:
    return (-float(metrics["composite_objective"]), indices)


def evaluate(
    indices: tuple[int, ...], masks: list[int], matrix: np.ndarray, objective: dict[str, Any]
) -> dict[str, float]:
    return consensus_components(
        indices,
        masks,
        matrix,
        int(objective["maximum_selected"]),
        float(objective["single_coverage_weight"]),
        float(objective["double_coverage_weight"]),
        float(objective["pair_diversity_weight"]),
        float(objective["per_conformer_cost"]),
    )


def direct_greedy(
    candidate_count: int, masks: list[int], matrix: np.ndarray, objective: dict[str, Any]
) -> tuple[int, ...]:
    maximum = int(objective["maximum_selected"])
    current: tuple[int, ...] = ()
    current_value = float(evaluate(current, masks, matrix, objective)["composite_objective"])
    for _ in range(maximum):
        proposals = [
            tuple(sorted((*current, incoming)))
            for incoming in range(candidate_count)
            if incoming not in current
        ]
        best = min(proposals, key=lambda item: value_key(item, evaluate(item, masks, matrix, objective)))
        best_value = float(evaluate(best, masks, matrix, objective)["composite_objective"])
        if best_value <= current_value + 1e-12:
            break
        current, current_value = best, best_value
    return current


def variable_local_search(
    start: tuple[int, ...], candidate_count: int, masks: list[int], matrix: np.ndarray,
    objective: dict[str, Any]
) -> tuple[tuple[int, ...], dict[str, float], int]:
    maximum = int(objective["maximum_selected"])
    current = tuple(sorted(start))
    current_metrics = evaluate(current, masks, matrix, objective)
    iterations = 0
    while True:
        selected = set(current)
        proposals: set[tuple[int, ...]] = set()
        if len(current) < maximum:
            proposals.update(tuple(sorted((*current, value))) for value in range(candidate_count) if value not in selected)
        if current:
            proposals.update(tuple(value for value in current if value != outgoing) for outgoing in current)
        if current and len(current) <= maximum:
            for outgoing in current:
                proposals.update(
                    tuple(sorted((selected - {outgoing}) | {incoming}))
                    for incoming in range(candidate_count) if incoming not in selected
                )
        if not proposals:
            return current, current_metrics, iterations
        best = min(proposals, key=lambda item: value_key(item, evaluate(item, masks, matrix, objective)))
        best_metrics = evaluate(best, masks, matrix, objective)
        if float(best_metrics["composite_objective"]) <= float(current_metrics["composite_objective"]) + 1e-12:
            return current, current_metrics, iterations
        current, current_metrics = best, best_metrics
        iterations += 1
        if iterations > candidate_count * (maximum + 1):
            raise RuntimeError("variable local search exceeded iteration guard")


def beam_search(
    candidate_count: int, masks: list[int], matrix: np.ndarray, objective: dict[str, Any], width: int
) -> tuple[int, ...]:
    maximum = int(objective["maximum_selected"])
    layer: list[tuple[int, ...]] = [()]
    best = ()
    best_metrics = evaluate(best, masks, matrix, objective)
    for _ in range(maximum):
        proposals = {
            tuple(sorted((*subset, incoming)))
            for subset in layer
            for incoming in range(candidate_count)
            if incoming not in subset
        }
        ranked = sorted(proposals, key=lambda item: value_key(item, evaluate(item, masks, matrix, objective)))
        layer = ranked[:width]
        candidate = layer[0]
        metrics = evaluate(candidate, masks, matrix, objective)
        if value_key(candidate, metrics) < value_key(best, best_metrics):
            best, best_metrics = candidate, metrics
    return best


def exact_oracle(
    candidate_count: int, masks: list[int], matrix: np.ndarray, objective: dict[str, Any], state_limit: int
) -> dict[str, Any] | None:
    maximum = min(int(objective["maximum_selected"]), candidate_count)
    state_count = sum(math.comb(candidate_count, size) for size in range(maximum + 1))
    if state_count > state_limit:
        return None
    best: tuple[int, ...] = ()
    best_metrics = evaluate(best, masks, matrix, objective)
    for size in range(1, maximum + 1):
        for subset in itertools.combinations(range(candidate_count), size):
            metrics = evaluate(subset, masks, matrix, objective)
            if value_key(subset, metrics) < value_key(best, best_metrics):
                best, best_metrics = subset, metrics
    return {"indices": best, "metrics": best_metrics, "state_count": state_count}


def anneal_read(
    candidate_count: int, masks: list[int], matrix: np.ndarray, objective: dict[str, Any],
    sweeps: int, temperature_start: float, temperature_end: float, rng: random.Random
) -> dict[str, Any]:
    maximum = int(objective["maximum_selected"])
    initial_size = rng.randrange(maximum + 1)
    current = tuple(sorted(rng.sample(range(candidate_count), initial_size)))
    current_value = float(evaluate(current, masks, matrix, objective)["composite_objective"])
    best, best_value = current, current_value
    accepted = 0
    for step in range(sweeps):
        selected = set(current)
        moves = []
        if len(current) < maximum:
            moves.append("add")
        if current:
            moves.extend(("remove", "swap"))
        move = rng.choice(moves)
        if move == "add":
            incoming = rng.choice([value for value in range(candidate_count) if value not in selected])
            proposal = tuple(sorted((*current, incoming)))
        elif move == "remove":
            outgoing = rng.choice(current)
            proposal = tuple(value for value in current if value != outgoing)
        else:
            outgoing = rng.choice(current)
            incoming = rng.choice([value for value in range(candidate_count) if value not in selected])
            proposal = tuple(sorted((selected - {outgoing}) | {incoming}))
        proposed = float(evaluate(proposal, masks, matrix, objective)["composite_objective"])
        progress = step / max(1, sweeps - 1)
        temperature = max(temperature_end, temperature_start * (temperature_end / temperature_start) ** progress)
        delta = proposed - current_value
        if delta >= 0.0 or rng.random() < math.exp(max(-700.0, delta / temperature)):
            current, current_value = proposal, proposed
            accepted += 1
            if current_value > best_value + 1e-12 or (
                math.isclose(current_value, best_value, abs_tol=1e-12) and current < best
            ):
                best, best_value = current, current_value
    return {
        "indices": best,
        "metrics": evaluate(best, masks, matrix, objective),
        "acceptance_fraction": accepted / max(1, sweeps),
    }


def build_consensus_qubo(
    ids: list[str], matrix: np.ndarray, terms: dict[str, Any], objective: dict[str, Any]
) -> dict[str, Any]:
    coefficients: dict[str, Any] = {"constant": 0.0, "linear": {}, "quadratic": {}}
    maximum = int(objective["maximum_selected"])
    x_names = {value: f"x__{value}" for value in ids}
    u_names = {value: f"u__{value}" for value in ids}
    v_names = {value: f"v__{value}" for value in ids}
    budget_names: list[str] = []
    budget_terms = {x_names[value]: 1.0 for value in ids}
    for index, weight in enumerate(slack_weights(maximum)):
        name = f"b__{index}"
        budget_names.append(name)
        budget_terms[name] = float(weight)
    add_square(coefficients, -maximum, budget_terms, float(objective["budget_constraint_penalty"]))
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
    for value in ids:
        add_linear(coefficients, x_names[value], float(objective["per_conformer_cost"]))
    denominator = max(1, math.comb(maximum, 2))
    for first in range(len(ids)):
        for second in range(first + 1, len(ids)):
            add_quadratic(
                coefficients, x_names[ids[first]], x_names[ids[second]],
                -float(objective["pair_diversity_weight"]) * float(matrix[first, second]) / denominator,
            )
    variables = sorted(set(coefficients["linear"]) | {v for key in coefficients["quadratic"] for v in key.split("::", 1)})
    return {
        "constant": float(coefficients["constant"]),
        "linear": {key: float(value) for key, value in coefficients["linear"].items()},
        "quadratic": {key: float(value) for key, value in coefficients["quadratic"].items()},
        "variables": variables,
        "variable_groups": {
            "x": sorted(x_names.values()), "single_u": sorted(u_names.values()),
            "double_v": sorted(v_names.values()), "single_slack": sorted(single_slack),
            "double_slack": sorted(double_slack), "budget_slack": sorted(budget_names),
        },
        "maximum_selected": maximum,
        "convention": "minimize Q; feasible auxiliary minimum equals negative reduced objective",
    }


def assignment_for_subset(
    selected: tuple[int, ...], ids: list[str], terms: dict[str, Any], qubo: dict[str, Any]
) -> dict[str, int]:
    assignment = {value: 0 for value in qubo["variables"]}
    chosen = {ids[index] for index in selected}
    for value in chosen:
        assignment[f"x__{value}"] = 1
    budget_weights = slack_weights(int(qubo["maximum_selected"]))
    for index, bit in binary_assignment(budget_weights, int(qubo["maximum_selected"]) - len(chosen)).items():
        assignment[f"b__{index}"] = bit
    single_weights = slack_weights(int(terms["neighbor_count"]) - 1)
    double_weights = slack_weights(int(terms["neighbor_count"]))
    for state_id in ids:
        count = sum(value in chosen for value in terms["incidence"][state_id])
        u = int(count >= 1)
        v = int(count >= 2)
        assignment[f"u__{state_id}"] = u
        assignment[f"v__{state_id}"] = v
        for index, bit in binary_assignment(single_weights, count - u).items():
            assignment[f"su__{state_id}__{index}"] = bit
        for index, bit in binary_assignment(double_weights, count - 2 * v).items():
            assignment[f"sv__{state_id}__{index}"] = bit
    return assignment


def instance_record(
    target_id: str, ids: list[str], matrix: np.ndarray, objective: dict[str, Any], config: dict[str, Any],
    read_rows: list[dict[str, Any]], batch_rows: list[dict[str, Any]], solver_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]], full_size: int,
) -> dict[str, Any]:
    terms = build_coverage_terms(ids, matrix, float(objective["neighborhood_fraction"]))
    masks = [int(terms["coverage_masks"][value]) for value in ids]
    size = len(ids)
    instance_id = f"{target_id}_n{size}"
    exact_start = time.perf_counter()
    exact = exact_oracle(
        size, masks, matrix, objective,
        int(config["instances"]["exact_oracle_state_limit"]),
    )
    exact_runtime = time.perf_counter() - exact_start
    if exact is not None:
        solver_rows.append({
            "target_id": target_id, "instance_size": size, "method": "exact_oracle", "beam_width": 0,
            "runtime_seconds": exact_runtime, "selected_subset": "+".join(ids[i] for i in exact["indices"]),
            **exact["metrics"], "state_count": exact["state_count"],
        })
    starts: list[tuple[str, int, tuple[int, ...], float]] = []
    started = time.perf_counter()
    greedy = direct_greedy(size, masks, matrix, objective)
    refined, _, _ = variable_local_search(greedy, size, masks, matrix, objective)
    starts.append(("direct_greedy_plus_variable_local_search", 0, refined, time.perf_counter() - started))
    for width_value in config["classical_baselines"]["beam_widths"]:
        width = int(width_value)
        started = time.perf_counter()
        beam = beam_search(size, masks, matrix, objective, width)
        refined, _, _ = variable_local_search(beam, size, masks, matrix, objective)
        starts.append(("beam_plus_variable_local_search", width, refined, time.perf_counter() - started))
    for method, width, selected, runtime in starts:
        solver_rows.append({
            "target_id": target_id, "instance_size": size, "method": method, "beam_width": width,
            "runtime_seconds": runtime, "selected_subset": "+".join(ids[i] for i in selected),
            **evaluate(selected, masks, matrix, objective), "state_count": "",
        })
    strong = max(float(row["composite_objective"]) for row in solver_rows if row["target_id"] == target_id and int(row["instance_size"]) == size and row["method"] != "exact_oracle")
    sampler = config["sampler"]
    local_batches = []
    all_read_runtime = 0.0
    for batch in range(int(sampler["batch_count"])):
        seed = int(sampler["base_seed"]) + sum(ord(c) for c in target_id) + size * 1009 + batch * 1000003
        rng = random.Random(seed)
        local_reads = []
        started = time.perf_counter()
        for read_index in range(int(sampler["reads_per_batch"])):
            result = anneal_read(
                size, masks, matrix, objective, int(sampler["sweeps_per_read"]),
                float(sampler["temperature_start"]), float(sampler["temperature_end"]), rng,
            )
            row = {
                "target_id": target_id, "instance_size": size, "batch": batch, "read": read_index,
                "seed": seed, "selected_subset": "+".join(ids[i] for i in result["indices"]),
                **result["metrics"], "acceptance_fraction": result["acceptance_fraction"],
                "delta_vs_strong_classical": float(result["metrics"]["composite_objective"]) - strong,
            }
            read_rows.append(row)
            local_reads.append(row)
        runtime = time.perf_counter() - started
        all_read_runtime += runtime
        best = max(local_reads, key=lambda row: (float(row["composite_objective"]), row["selected_subset"]))
        batch_row = {
            "target_id": target_id, "instance_size": size, "batch": batch, "seed": seed,
            "runtime_seconds": runtime, "best_subset": best["selected_subset"],
            "best_selected_count": int(float(best["selected_count"])),
            "best_objective": float(best["composite_objective"]),
            "delta_vs_strong_classical": float(best["composite_objective"]) - strong,
        }
        batch_rows.append(batch_row)
        local_batches.append(batch_row)
    best_batch = max(local_batches, key=lambda row: (float(row["best_objective"]), row["best_subset"]))
    selected_ids = tuple(part for part in best_batch["best_subset"].split("+") if part)
    selected_indices = tuple(ids.index(value) for value in selected_ids)
    metrics = evaluate(selected_indices, masks, matrix, objective)
    qubo = build_consensus_qubo(ids, matrix, terms, objective)
    energy = qubo_energy(qubo, assignment_for_subset(selected_indices, ids, terms, qubo))
    residual = abs(energy + float(metrics["composite_objective"]))
    if residual > 1e-8:
        raise ValueError(f"{instance_id}: QUBO energy equivalence failed: {residual}")
    stats = coefficient_stats(qubo)
    gate = config["gate"]
    direct_qpu_ready = (
        len(qubo["variables"]) <= int(gate["direct_qpu_max_variables"])
        and float(stats["coefficient_dynamic_range"]) <= float(gate["direct_qpu_max_coefficient_dynamic_range"])
    )
    model_rows.append({
        "target_id": target_id, "instance_size": size, "full_instance": size == full_size,
        "x_count": size, "variable_count": len(qubo["variables"]),
        "quadratic_coupler_count": len(qubo["quadratic"]), "qubo_sha256": qubo_hash(qubo),
        "equivalence_residual": residual, "selected_energy": energy,
        "direct_qpu_ready_under_frozen_thresholds": direct_qpu_ready, **stats,
    })
    best_value = float(best_batch["best_objective"])
    within = sum(float(row["best_objective"]) >= best_value - float(gate["objective_tolerance"]) for row in local_batches) / len(local_batches)
    oracle_value = None if exact is None else float(exact["metrics"]["composite_objective"])
    return {
        "instance_id": instance_id, "instance_size": size, "full_instance": size == full_size,
        "neighbor_count": int(terms["neighbor_count"]), "strong_classical_objective": strong,
        "best_sampler_objective": best_value, "delta_vs_strong_classical": best_value - strong,
        "best_sampler_subset": list(selected_ids), "best_sampler_selected_count": len(selected_ids),
        "within_tolerance_batch_fraction": within, "exact_oracle_objective": oracle_value,
        "sampler_gap_to_exact": None if oracle_value is None else oracle_value - best_value,
        "sampler_total_runtime_seconds": all_read_runtime, "qubo_variable_count": len(qubo["variables"]),
        "qubo_coupler_count": len(qubo["quadratic"]), "qubo_coefficient_dynamic_range": stats["coefficient_dynamic_range"],
        "direct_qpu_ready_under_frozen_thresholds": direct_qpu_ready,
    }


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if outputs["result_json"].exists() and not overwrite:
        raise FileExistsError(f"result exists: {outputs['result_json']}; pass --overwrite")
    objective = config["objective"]
    read_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    target_records: dict[str, Any] = {}
    input_records: dict[str, Any] = {}
    for target_id, spec in config["targets"].items():
        target = load_target(root, target_id, spec)
        full_ids = target["ids"]
        full_matrix = distance_matrix(full_ids, target["distances"])
        order = nested_ids(target_id, full_ids, spec["reference_id"])
        sizes = [int(value) for value in spec["instance_sizes"]]
        if sizes[-1] != len(full_ids) or sizes != sorted(set(sizes)):
            raise ValueError(f"{target_id}: instance sizes must be unique and end at full pool size")
        records = []
        for size in sizes:
            ids = order[:size]
            matrix = subset_matrix(full_ids, full_matrix, ids)
            print(json.dumps({"target_id": target_id, "instance_size": size, "status": "running"}), flush=True)
            records.append(instance_record(
                target_id, ids, matrix, objective, config, read_rows, batch_rows, solver_rows, model_rows, len(full_ids)
            ))
        target_records[target_id] = {
            "candidate_count": len(full_ids), "hard_gate_excluded_count": len(target["excluded_hard_gate"]),
            "nested_order_sha256": hashlib.sha256("\n".join(order).encode("ascii")).hexdigest().upper(),
            "instances": records,
        }
        input_records[target_id] = {key: descriptor(root, path) for key, path in target["input_paths"].items()}
    write_csv(outputs["solver_csv"], solver_rows)
    write_csv(outputs["batch_csv"], batch_rows)
    write_csv(outputs["read_csv"], read_rows)
    write_csv(outputs["model_csv"], model_rows)
    gate = config["gate"]
    full_records = [record for target in target_records.values() for record in target["instances"] if record["full_instance"]]
    stable = all(float(record["within_tolerance_batch_fraction"]) >= float(gate["minimum_full_instance_batch_fraction_within_tolerance"]) for record in full_records)
    winning = sum(float(record["delta_vs_strong_classical"]) > float(gate["minimum_gain_vs_strong_classical"]) for record in full_records)
    small_records = [record for target in target_records.values() for record in target["instances"] if record["exact_oracle_objective"] is not None]
    exact_ok = all(float(record["sampler_gap_to_exact"]) <= float(gate["maximum_small_instance_gap_to_exact"]) + 1e-12 for record in small_records)
    novelty = stable and exact_ok and winning >= int(gate["minimum_full_targets_strictly_above_strong_classical"])
    direct_ready = all(bool(record["direct_qpu_ready_under_frozen_thresholds"]) for record in full_records)
    report_lines = [
        "# Stage 26: variable-budget two-hit consensus QUBO", "",
        "Frozen structure-only objective: 0.25 single coverage + 0.75 double coverage + 0.10 pair diversity - 0.04 per selected conformer; at most eight conformers.", "",
        "| Target | n | Full | Selected k | Sampler | Strong classical | Delta | Stable | QUBO variables | Direct QPU ready |", "|---|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for target_id, target in target_records.items():
        for record in target["instances"]:
            report_lines.append(
                f"| {target_id} | {record['instance_size']} | {'yes' if record['full_instance'] else 'no'} | {record['best_sampler_selected_count']} | {record['best_sampler_objective']:.6f} | {record['strong_classical_objective']:.6f} | {record['delta_vs_strong_classical']:.6f} | {record['within_tolerance_batch_fraction']:.2f} | {record['qubo_variable_count']} | {'yes' if record['direct_qpu_ready_under_frozen_thresholds'] else 'no'} |"
            )
    report_lines += ["", f"Optimization-novelty gate: **{'PASS' if novelty else 'NO-GO'}**.", f"Direct-QPU readiness under frozen thresholds: **{'PASS' if direct_ready else 'NO-GO'}**.", "", "No docking scores, ligand labels, fresh validation rows, test rows, or quantum-hardware outputs were read."]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report_lines) + "\n", encoding="ascii")
    result = {
        "schema_version": "1.0", "status": "stage26_variable_budget_consensus_qubo_complete",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "implementation": descriptor(root, Path(__file__).resolve()), "inputs": input_records,
        "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key != "result_json"},
        "target_records": target_records,
        "decision": {
            "sampler_stability_gate_passed": stable, "small_exactness_gate_passed": exact_ok,
            "full_targets_strictly_above_strong_classical": winning,
            "optimization_novelty_gate_passed": novelty,
            "direct_qpu_readiness_gate_passed": direct_ready,
            "functional_complementarity_preregistration_authorized": novelty,
            "new_docking_jobs_authorized_by_this_stage": False, "quantum_hardware_authorized": False,
        },
        "data_boundary": {"docking_scores_read": 0, "ligand_labels_read": 0, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage26_variable_budget_consensus_qubo.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
