"""Run the Stage33 sparse, domain-wall PPARG MD QUBO feasibility screen."""

from __future__ import annotations

import argparse
import csv
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

from scripts.run_stage21_structure_aware_qubo import file_sha256, read_json, rooted, write_csv, write_json
from scripts.run_stage30_pparg_group_balanced_state_qubo import load_inputs as load_stage30_inputs


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def validate_upstream(config: dict[str, Any], root: Path) -> None:
    expected = {
        "stage28b_audit": "stage28_pparg_multistart_md_ensemble_audit_ok",
        "stage29_audit": "stage29_pparg_md_qubo_solver_scaling_audit_ok",
        "stage30_audit": "stage30_pparg_group_balanced_state_qubo_audit_ok",
        "stage32c_audit": "stage32c_pparg_md_pair_failure_diagnostic_audit_ok",
    }
    for key, status in expected.items():
        record = read_json(rooted(root, config["inputs"][key]))
        if record.get("status") != status:
            raise ValueError(f"{key} status differs: {record.get('status')}")


def x_affine(group: int, position: int, width: int) -> tuple[float, dict[int, float]]:
    offset = group * (width - 1)
    if position == 0:
        return 1.0, {offset: -1.0}
    if position == width - 1:
        return 0.0, {offset + position - 1: 1.0}
    return 0.0, {offset + position - 1: 1.0, offset + position: -1.0}


def add_polynomial_product(
    linear: dict[int, float],
    quadratic: dict[tuple[int, int], float],
    left: tuple[float, dict[int, float]],
    right: tuple[float, dict[int, float]],
    scale: float,
) -> float:
    lc, lt = left
    rc, rt = right
    constant = scale * lc * rc
    for index, coefficient in lt.items():
        linear[index] = linear.get(index, 0.0) + scale * coefficient * rc
    for index, coefficient in rt.items():
        linear[index] = linear.get(index, 0.0) + scale * coefficient * lc
    for left_index, left_coefficient in lt.items():
        for right_index, right_coefficient in rt.items():
            value = scale * left_coefficient * right_coefficient
            if left_index == right_index:
                linear[left_index] = linear.get(left_index, 0.0) + value
            else:
                edge = tuple(sorted((left_index, right_index)))
                quadratic[edge] = quadratic.get(edge, 0.0) + value
    return constant


def build_model(per_start: int, loaded: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    objective = config["sparse_objective"]
    groups_global = [tuple(loaded["ordered_global"][start][:per_start]) for start in sorted(loaded["ordered_global"])]
    global_indices = np.asarray([value for group in groups_global for value in group], dtype=int)
    groups = [tuple(range(group * per_start, (group + 1) * per_start)) for group in range(8)]
    local_distance = loaded["distance"][np.ix_(global_indices, global_indices)]
    local_state = loaded["state_separation"][np.ix_(global_indices, global_indices)]
    centrality = loaded["centrality"][global_indices]
    complete = loaded["distance"]
    radius = float(np.quantile(complete[np.triu_indices(len(complete), 1)], objective["distance_radius_complete_pool_quantile"]))
    nearest = int(objective["cross_start_nearest_neighbors_per_direction"])
    edge_pairs: set[tuple[int, int]] = set()
    for left_group in range(8):
        for right_group in range(left_group + 1, 8):
            left = np.asarray(groups[left_group], dtype=int)
            right = np.asarray(groups[right_group], dtype=int)
            block = local_distance[np.ix_(left, right)]
            for row, local_left in enumerate(left):
                for column in np.argsort(block[row], kind="stable")[:nearest]:
                    edge_pairs.add((int(local_left), int(right[column])))
            for column, local_right in enumerate(right):
                for row in np.argsort(block[:, column], kind="stable")[:nearest]:
                    edge_pairs.add((int(left[row]), int(local_right)))
    edges: dict[tuple[int, int], float] = {}
    for left, right in sorted(edge_pairs):
        distance = float(local_distance[left, right])
        if distance > radius:
            continue
        distance_similarity = max(0.0, 1.0 - distance / radius)
        state_similarity = 1.0 - float(local_state[left, right])
        similarity = (
            float(objective["distance_similarity_fraction"]) * distance_similarity
            + float(objective["state_similarity_fraction"]) * state_similarity
        )
        if similarity >= float(objective["minimum_retained_similarity"]):
            edges[(left, right)] = similarity
    k = int(objective["selected_count"])
    unary = float(objective["within_start_centrality_weight"]) * centrality / k
    pair_scale = -float(objective["local_redundancy_weight"]) / float(objective["pair_normalization"])
    pair = {edge: pair_scale * similarity for edge, similarity in edges.items()}
    pair_matrix = np.zeros((len(global_indices), len(global_indices)), dtype=float)
    for (left, right), value in pair.items():
        pair_matrix[left, right] = value
        pair_matrix[right, left] = value
    return {
        "per_start": per_start,
        "global_indices": global_indices,
        "groups": groups,
        "unary": unary,
        "pair": pair,
        "pair_matrix": pair_matrix,
        "similarity": edges,
        "distance": local_distance,
        "state_separation": local_state,
        "frame_ids": tuple(loaded["frames"][index]["frame_id"] for index in global_indices),
        "source_ids": tuple(loaded["frames"][index]["conformer_id"] for index in global_indices),
        "radius": radius,
        "k": k,
    }


def validate_selected(selected: Iterable[int], model: dict[str, Any]) -> tuple[int, ...]:
    chosen = tuple(sorted(int(value) for value in selected))
    chosen_set = set(chosen)
    if len(chosen) != 8 or any(sum(index in chosen_set for index in group) != 1 for group in model["groups"]):
        raise ValueError("selection must contain exactly one frame per start")
    return chosen


def sparse_objective(selected: Iterable[int], model: dict[str, Any]) -> float:
    chosen = validate_selected(selected, model)
    chosen_array = np.asarray(chosen, dtype=int)
    upper = np.triu_indices(len(chosen_array), 1)
    return float(
        model["unary"][chosen_array].sum()
        + model["pair_matrix"][np.ix_(chosen_array, chosen_array)][upper].sum()
    )


def dense_objective(selected: Iterable[int], model: dict[str, Any]) -> float:
    chosen = np.asarray(validate_selected(selected, model), dtype=int)
    upper = np.triu_indices(len(chosen), 1)
    return float(
        0.3 * model["unary"][chosen].sum() / 0.3
        + 0.5 * model["distance"][np.ix_(chosen, chosen)][upper].mean()
        + 0.2 * model["state_separation"][np.ix_(chosen, chosen)][upper].mean()
    )


def cyclic_orders() -> list[tuple[int, ...]]:
    forward = tuple(range(8))
    reverse = tuple(reversed(forward))
    return [base[offset:] + base[:offset] for base in (forward, reverse) for offset in range(8)]


def greedy_for_order(model: dict[str, Any], order: Iterable[int]) -> tuple[int, ...]:
    selected: list[int] = []
    selected_set: set[int] = set()
    for group_index in order:
        best: int | None = None
        best_value = -math.inf
        for candidate in model["groups"][group_index]:
            value = float(model["unary"][candidate])
            if selected_set:
                value += float(model["pair_matrix"][candidate, list(selected_set)].sum())
            if value > best_value + 1e-12 or (math.isclose(value, best_value, abs_tol=1e-12) and (best is None or candidate < best)):
                best, best_value = candidate, value
        assert best is not None
        selected.append(best)
        selected_set.add(best)
    return tuple(sorted(selected))


def coordinate_descent(start: Iterable[int], model: dict[str, Any]) -> tuple[tuple[int, ...], float, int]:
    current = validate_selected(start, model)
    current_value = sparse_objective(current, model)
    iterations = 0
    while True:
        current_set = set(current)
        best, best_value = current, current_value
        for group in model["groups"]:
            outgoing = next(index for index in group if index in current_set)
            retained = tuple(index for index in current if index != outgoing)
            retained_array = np.asarray(retained, dtype=int)
            group_array = np.asarray(group, dtype=int)
            base = float(model["unary"][retained_array].sum())
            retained_pair = model["pair_matrix"][np.ix_(retained_array, retained_array)]
            base += float(retained_pair[np.triu_indices(len(retained_array), 1)].sum())
            scores = base + model["unary"][group_array] + model["pair_matrix"][np.ix_(group_array, retained_array)].sum(axis=1)
            incoming = int(group_array[int(np.argmax(scores))])
            candidate = tuple(sorted((*retained, incoming)))
            value = float(np.max(scores))
            if value > best_value + 1e-12 or (math.isclose(value, best_value, abs_tol=1e-12) and candidate < best):
                best, best_value = candidate, value
        if best_value <= current_value + 1e-12:
            return current, current_value, iterations
        current, current_value = best, best_value
        iterations += 1


def strong_classical(model: dict[str, Any], restart_count: int, rng: random.Random) -> dict[str, Any]:
    starts = [greedy_for_order(model, order) for order in cyclic_orders()]
    starts.extend(tuple(sorted(rng.choice(group) for group in model["groups"])) for _ in range(restart_count))
    optima: dict[tuple[int, ...], float] = {}
    total_iterations = 0
    for start in starts:
        selected, value, iterations = coordinate_descent(start, model)
        optima[selected] = value
        total_iterations += iterations
    best = max(optima, key=lambda selected: (optima[selected], tuple(-value for value in selected)))
    return {"selected": best, "objective": optima[best], "unique_local_optima": len(optima), "iterations": total_iterations}


def exact_oracle(model: dict[str, Any], maximum_states: int) -> dict[str, Any] | None:
    state_count = math.prod(len(group) for group in model["groups"])
    if state_count > maximum_states:
        return None
    best: tuple[int, ...] | None = None
    best_value = -math.inf
    for selected in itertools.product(*model["groups"]):
        value = sparse_objective(selected, model)
        candidate = tuple(sorted(selected))
        if value > best_value + 1e-12 or (math.isclose(value, best_value, abs_tol=1e-12) and (best is None or candidate < best)):
            best, best_value = candidate, value
    return {"selected": best, "objective": best_value, "state_count": state_count}


def anneal(model: dict[str, Any], sampler: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    selected = [rng.choice(group) for group in model["groups"]]
    value = sparse_objective(selected, model)
    best, best_value = tuple(sorted(selected)), value
    sweeps = max(int(sampler["minimum_sweeps_per_read"]), int(sampler["sweeps_per_candidate"]) * len(model["unary"]))
    for step in range(sweeps):
        group_index = rng.randrange(8)
        outgoing = selected[group_index]
        incoming = rng.choice(model["groups"][group_index])
        others = selected[:group_index] + selected[group_index + 1:]
        proposal_value = value + float(model["unary"][incoming] - model["unary"][outgoing])
        proposal_value += float(model["pair_matrix"][incoming, others].sum() - model["pair_matrix"][outgoing, others].sum())
        fraction = step / max(1, sweeps - 1)
        temperature = float(sampler["temperature_start"]) * (float(sampler["temperature_end"]) / float(sampler["temperature_start"])) ** fraction
        if proposal_value >= value or rng.random() < math.exp((proposal_value - value) / temperature):
            selected[group_index], value = incoming, proposal_value
        candidate = tuple(sorted(selected))
        if value > best_value + 1e-12 or (math.isclose(value, best_value, abs_tol=1e-12) and candidate < best):
            best, best_value = candidate, value
    return {"selected": best, "objective": best_value, "sweeps": sweeps}


def build_domain_wall_qubo(model: dict[str, Any], penalty: float) -> dict[str, Any]:
    width = int(model["per_start"])
    if width < 2:
        raise ValueError("domain-wall encoding requires at least two candidates per group")
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}
    constant = 0.0
    for index, value in enumerate(model["unary"]):
        group, position = divmod(index, width)
        constant += add_polynomial_product(linear, quadratic, x_affine(group, position, width), (1.0, {}), -float(value))
    for (left, right), value in model["pair"].items():
        left_group, left_position = divmod(left, width)
        right_group, right_position = divmod(right, width)
        constant += add_polynomial_product(
            linear,
            quadratic,
            x_affine(left_group, left_position, width),
            x_affine(right_group, right_position, width),
            -float(value),
        )
    for group in range(8):
        offset = group * (width - 1)
        for position in range(1, width - 1):
            current = offset + position
            previous = offset + position - 1
            linear[current] = linear.get(current, 0.0) + penalty
            edge = (previous, current)
            quadratic[edge] = quadratic.get(edge, 0.0) - penalty
    linear = {key: value for key, value in linear.items() if abs(value) > 1e-12}
    quadratic = {key: value for key, value in quadratic.items() if abs(value) > 1e-12}
    record = {
        "logical_variable_count": 8 * (width - 1),
        "linear": sorted((key, value) for key, value in linear.items()),
        "quadratic": sorted((left, right, value) for (left, right), value in quadratic.items()),
        "constant": constant,
    }
    record["sha256"] = stable_hash(record)
    return record


def selected_to_domain_wall(selected: Iterable[int], model: dict[str, Any]) -> np.ndarray:
    width = int(model["per_start"])
    chosen = validate_selected(selected, model)
    bits = np.zeros(8 * (width - 1), dtype=int)
    for index in chosen:
        group, position = divmod(index, width)
        bits[group * (width - 1):group * (width - 1) + position] = 1
    return bits


def qubo_energy(bits: np.ndarray, qubo: dict[str, Any]) -> float:
    value = float(qubo["constant"])
    value += sum(coefficient * bits[index] for index, coefficient in qubo["linear"])
    value += sum(coefficient * bits[left] * bits[right] for left, right, coefficient in qubo["quadratic"])
    return float(value)


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    validate_upstream(config, root)
    loaded = load_stage30_inputs(root, config)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if outputs["result_json"].exists() and not overwrite:
        raise FileExistsError(outputs["result_json"])
    sampler = config["annealing_sampler"]
    gate = config["gate"]
    candidate_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []
    cell_records: dict[str, Any] = {}
    strict_wins = 0
    all_residuals: list[float] = []
    for cell_index, per_start in enumerate(config["candidate_scaling"]["frames_per_start"]):
        model = build_model(int(per_start), loaded, config)
        pool_id = f"sparse_m{int(per_start):03d}_n{len(model['unary']):04d}"
        for local_index, global_index in enumerate(model["global_indices"]):
            candidate_rows.append({"pool_id": pool_id, "local_candidate_index": local_index, "global_frame_index": int(global_index), "frame_id": model["frame_ids"][local_index], "conformer_id": model["source_ids"][local_index], "start_group": local_index // int(per_start), "within_group_position": local_index % int(per_start)})
        for (left, right), similarity in model["similarity"].items():
            edge_rows.append({"pool_id": pool_id, "left_local_index": left, "right_local_index": right, "left_frame_id": model["frame_ids"][left], "right_frame_id": model["frame_ids"][right], "normalized_distance": model["distance"][left, right], "state_similarity": 1.0 - model["state_separation"][left, right], "combined_similarity": similarity, "objective_coefficient": model["pair"][(left, right)]})
        rng = random.Random(int(sampler["base_seed"]) + cell_index * 100003)
        started = time.perf_counter()
        strong = strong_classical(model, int(config["classical_baselines"]["random_coordinate_restart_count"]), rng)
        strong_runtime = time.perf_counter() - started
        exact = exact_oracle(model, int(config["exact_oracle"]["maximum_states"]))
        reference = exact or strong
        methods = [("strong_classical", strong, strong_runtime)]
        if exact:
            methods.append(("exact_oracle", exact, 0.0))
        best_annealed: dict[str, Any] | None = None
        batch_best: list[float] = []
        for batch in range(int(sampler["batch_count"])):
            batch_started = time.perf_counter()
            reads = [anneal(model, sampler, random.Random(int(sampler["base_seed"]) + cell_index * 100003 + batch * 1009 + read)) for read in range(int(sampler["reads_per_batch"]))]
            batch_record = max(reads, key=lambda item: (item["objective"], tuple(-value for value in item["selected"])))
            batch_best.append(float(batch_record["objective"]))
            if best_annealed is None or batch_record["objective"] > best_annealed["objective"] + 1e-12:
                best_annealed = batch_record
            batch_rows.append({"pool_id": pool_id, "batch_index": batch, "best_objective": batch_record["objective"], "reference_gap": float(reference["objective"]) - float(batch_record["objective"]), "within_tolerance": float(reference["objective"]) - float(batch_record["objective"]) <= float(gate["objective_tolerance"]), "runtime_seconds": time.perf_counter() - batch_started})
        assert best_annealed is not None
        methods.append(("group_feasible_annealing", best_annealed, 0.0))
        if best_annealed["objective"] > strong["objective"] + float(gate["minimum_strict_gain"]):
            strict_wins += 1
        for method, record, runtime in methods:
            solver_rows.append({"pool_id": pool_id, "frames_per_start": per_start, "candidate_count": len(model["unary"]), "method": method, "runtime_seconds": runtime, "selected_frame_ids": "+".join(model["frame_ids"][index] for index in record["selected"]), "selected_source_ids": "+".join(model["source_ids"][index] for index in record["selected"]), "sparse_objective": record["objective"], "dense_quality_objective": dense_objective(record["selected"], model), "reference_gap": float(reference["objective"]) - float(record["objective"]), "unique_local_optima": record.get("unique_local_optima", ""), "state_count": record.get("state_count", "")})
        qubo = build_domain_wall_qubo(model, float(config["sparse_objective"]["domain_wall_violation_penalty"]))
        residuals = []
        sample_sets = [reference["selected"], strong["selected"], best_annealed["selected"]]
        sample_sets.extend(tuple(group[rng.randrange(len(group))] for group in model["groups"]) for _ in range(32))
        for selected in sample_sets:
            bits = selected_to_domain_wall(selected, model)
            residuals.append(abs(qubo_energy(bits, qubo) + sparse_objective(selected, model)))
        all_residuals.extend(residuals)
        coefficients = [abs(value) for _, value in qubo["linear"]] + [abs(value) for _, _, value in qubo["quadratic"]]
        variable_count = int(qubo["logical_variable_count"])
        coupler_count = len(qubo["quadratic"])
        dense_couplers = math.comb(len(model["unary"]), 2)
        reduction = 1.0 - coupler_count / dense_couplers
        dynamic_range = max(coefficients) / min(coefficients)
        direct_ready = variable_count <= int(gate["direct_qpu_max_logical_variables"]) and coupler_count <= int(gate["direct_qpu_max_quadratic_couplers"]) and dynamic_range <= float(gate["direct_qpu_max_coefficient_dynamic_range"])
        dense_strong = max((row for row in solver_rows if row["pool_id"] == pool_id), key=lambda row: float(row["dense_quality_objective"]))
        scaling_rows.append({"pool_id": pool_id, "frames_per_start": per_start, "candidate_count": len(model["unary"]), "selected_count": 8, "sparse_x_edge_count": len(model["pair"]), "domain_wall_variable_count": variable_count, "domain_wall_quadratic_coupler_count": coupler_count, "dense_one_hot_coupler_count": dense_couplers, "coupler_reduction_fraction": reduction, "coefficient_minimum_absolute_nonzero": min(coefficients), "coefficient_maximum_absolute": max(coefficients), "coefficient_dynamic_range": dynamic_range, "maximum_equivalence_residual": max(residuals), "annealing_batch_fraction_within_tolerance": sum(float(reference["objective"]) - value <= float(gate["objective_tolerance"]) for value in batch_best) / len(batch_best), "direct_qpu_ready_under_frozen_thresholds": direct_ready, "qubo_sha256": qubo["sha256"], "distance_radius": model["radius"]})
        cell_records[pool_id] = {"reference_method": "exact_oracle" if exact else "strong_classical", "reference_objective": reference["objective"], "strong_classical_objective": strong["objective"], "annealing_objective": best_annealed["objective"], "best_observed_dense_quality": dense_strong["dense_quality_objective"]}
    write_csv(outputs["candidate_manifest_csv"], candidate_rows)
    write_csv(outputs["sparse_edges_csv"], edge_rows)
    write_csv(outputs["solver_results_csv"], solver_rows)
    write_csv(outputs["batch_results_csv"], batch_rows)
    write_csv(outputs["model_scaling_csv"], scaling_rows)
    reference_count = int(gate["direct_qpu_reference_candidate_count"])
    reference_scaling = next(row for row in scaling_rows if int(row["candidate_count"]) == reference_count)
    full_scaling = scaling_rows[-1]
    exact_rows = [row for row in solver_rows if row["method"] == "exact_oracle"]
    exactness = all(abs(float(row["reference_gap"])) <= float(gate["maximum_exact_gap"]) for row in exact_rows)
    equivalence = max(all_residuals) <= float(gate["maximum_qubo_equivalence_residual"])
    sparse_gate = float(full_scaling["coupler_reduction_fraction"]) >= float(gate["minimum_full_pool_coupler_reduction_fraction"])
    stability = sum(float(row["annealing_batch_fraction_within_tolerance"]) >= float(gate["minimum_batch_fraction_within_tolerance"]) for row in scaling_rows) >= 6
    quality_rows = [row for row in solver_rows if row["method"] == "strong_classical"]
    stage30_path = root / "results/runs/stage30_pparg_group_balanced_state_qubo/solver_results.csv"
    with stage30_path.open("r", encoding="utf-8", newline="") as handle:
        stage30_rows = list(csv.DictReader(handle))
    stage30_best = {int(row["candidate_count"]): max(float(candidate["objective"]) for candidate in stage30_rows if int(candidate["candidate_count"]) == int(row["candidate_count"])) for row in quality_rows}
    quality_losses = {int(row["candidate_count"]): stage30_best[int(row["candidate_count"])] - float(row["dense_quality_objective"]) for row in quality_rows}
    quality_gate = max(quality_losses.values()) <= float(gate["maximum_dense_quality_loss"])
    hardware_pilot = bool(exactness and equivalence and sparse_gate and stability and quality_gate and reference_scaling["direct_qpu_ready_under_frozen_thresholds"])
    result = {
        "schema_version": "1.0",
        "status": "stage33_pparg_sparse_hardware_qubo_complete",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "inputs": {key: {"path": value, "sha256": file_sha256(rooted(root, value))} for key, value in config["inputs"].items()},
        "input_statistics": {"frame_count": 1200, "start_count": 8, "frames_per_start": 150},
        "cell_records": cell_records,
        "quality_loss_vs_stage30_dense_baseline": {str(key): value for key, value in quality_losses.items()},
        "decision": {
            "input_gate_passed": True,
            "scaling_complete": len(scaling_rows) == int(gate["required_scaling_cell_count"]),
            "small_cell_exactness_gate_passed": exactness,
            "domain_wall_equivalence_gate_passed": equivalence,
            "full_pool_sparsity_gate_passed": sparse_gate,
            "annealing_stability_gate_passed": stability,
            "dense_structural_quality_retention_gate_passed": quality_gate,
            "direct_qpu_reference_cell_ready": bool(reference_scaling["direct_qpu_ready_under_frozen_thresholds"]),
            "cells_strictly_above_strong_classical": strict_wins,
            "solver_novelty_gate_passed": strict_wins >= int(gate["minimum_cells_strictly_above_strong_classical"]),
            "small_quantum_annealing_application_pilot_authorized": hardware_pilot,
            "quantum_advantage_claim_authorized": False,
            "new_docking_jobs_authorized_by_this_stage": False,
        },
        "data_boundary": {"docking_scores_read": 0, "ligand_labels_read": 0, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "outputs": {key: value for key, value in config["outputs"].items() if key not in {"result_json", "audit_json"}},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    report = [
        "# Stage33 PPARG sparse hardware-oriented QUBO",
        "",
        "## Decision",
        "",
        f"- Small quantum-annealing application pilot authorized: **{hardware_pilot}**",
        f"- Full-pool coupler reduction: **{float(full_scaling['coupler_reduction_fraction']):.3%}**",
        f"- Direct-QPU reference cell ({reference_count} candidates) ready: **{reference_scaling['direct_qpu_ready_under_frozen_thresholds']}**",
        f"- Maximum dense structural-quality loss: **{max(quality_losses.values()):.6f}**",
        f"- Annealing cells strictly above strong classical: **{strict_wins}**",
        "",
        "## Interpretation",
        "",
        "This is a structure-only hardware feasibility screen. Passing authorizes only a separately controlled small quantum-annealing application pilot; it does not authorize a quantum advantage or docking-efficacy claim.",
    ]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage33_pparg_sparse_hardware_qubo.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run(rooted(root, args.config), root, args.overwrite)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
