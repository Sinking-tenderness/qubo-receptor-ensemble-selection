"""Run a structure-state coverage QUBO over full eligible conformer pools.

For each structural state j, the nearest fixed fraction of candidate receptors
defines a label-independent coverage neighborhood N_j.  A selected subset S
receives credit when S intersects N_j.  The reduced objective is

    F(S) = mean_j 1[S intersects N_j]
           + lambda_diversity * mean_{i<l in S} d_i,l.

The exact QUBO uses receptor variables x_i, state-coverage variables y_j,
binary slack variables, a cardinality constraint, and pairwise structural
diversity.  This stage reads no docking result, ligand label, validation row,
test row, or quantum-hardware result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    descriptor,
    distance_matrix,
    file_sha256,
    load_target,
    maxmin_seeded,
    maxsum_greedy,
    read_json,
    rooted,
    subset_metrics,
    write_csv,
    write_json,
)


CANDIDATE_METHOD = "coverage_qubo_exact_or_multistart"
BASELINE_METHODS = ("direct_greedy", "maxmin_seeded", "maxsum_greedy")


def add_linear(coefficients: dict[str, Any], variable: str, value: float) -> None:
    coefficients["linear"][variable] = (
        float(coefficients["linear"].get(variable, 0.0)) + float(value)
    )


def add_quadratic(
    coefficients: dict[str, Any], first: str, second: str, value: float
) -> None:
    if first == second:
        add_linear(coefficients, first, value)
        return
    key = "::".join(sorted((first, second)))
    coefficients["quadratic"][key] = (
        float(coefficients["quadratic"].get(key, 0.0)) + float(value)
    )


def add_square(
    coefficients: dict[str, Any],
    constant: float,
    terms: dict[str, float],
    weight: float,
) -> None:
    coefficients["constant"] += float(weight) * constant * constant
    for variable, value in terms.items():
        add_linear(
            coefficients,
            variable,
            float(weight) * (2.0 * constant * value + value * value),
        )
    variables = list(terms)
    for first_index, first in enumerate(variables):
        for second in variables[first_index + 1 :]:
            add_quadratic(
                coefficients,
                first,
                second,
                2.0 * float(weight) * terms[first] * terms[second],
            )


def slack_weights(maximum: int) -> list[int]:
    if maximum < 1:
        return []
    weights: list[int] = []
    represented = 0
    next_weight = 1
    while represented < maximum:
        value = min(next_weight, maximum - represented)
        weights.append(value)
        represented += value
        next_weight *= 2
    return weights


def binary_assignment(weights: list[int], value: int) -> dict[int, int]:
    if value < 0 or value > sum(weights):
        raise ValueError("slack value is outside its encoding range")
    for bits in itertools.product((0, 1), repeat=len(weights)):
        if sum(weight * bit for weight, bit in zip(weights, bits)) == value:
            return {index: int(bit) for index, bit in enumerate(bits)}
    raise ValueError("slack encoding cannot represent value")


def build_coverage_terms(
    ids: list[str], matrix: np.ndarray, neighborhood_fraction: float
) -> dict[str, Any]:
    if not 0.0 < neighborhood_fraction <= 1.0:
        raise ValueError("neighborhood fraction must be in (0, 1]")
    candidate_count = len(ids)
    neighbor_count = max(2, math.ceil(neighborhood_fraction * candidate_count))
    incidence: dict[str, list[str]] = {}
    coverage_masks = {conformer_id: 0 for conformer_id in ids}
    for state_index, state_id in enumerate(ids):
        nearest = sorted(
            range(candidate_count),
            key=lambda candidate_index: (
                float(matrix[candidate_index, state_index]),
                ids[candidate_index],
            ),
        )[:neighbor_count]
        incidence[state_id] = [ids[index] for index in nearest]
        for candidate_index in nearest:
            coverage_masks[ids[candidate_index]] |= 1 << state_index
    candidate_cover_counts = {
        conformer_id: mask.bit_count()
        for conformer_id, mask in coverage_masks.items()
    }
    return {
        "candidate_ids": ids,
        "state_ids": ids,
        "neighborhood_fraction": float(neighborhood_fraction),
        "neighbor_count": neighbor_count,
        "state_weight": 1.0 / candidate_count,
        "incidence": incidence,
        "coverage_masks": coverage_masks,
        "candidate_cover_counts": candidate_cover_counts,
    }


def coverage_mask(subset: Iterable[str], terms: dict[str, Any]) -> int:
    value = 0
    for conformer_id in subset:
        value |= int(terms["coverage_masks"][conformer_id])
    return value


def objective_components(
    subset: tuple[str, ...],
    ids: list[str],
    matrix: np.ndarray,
    terms: dict[str, Any],
    diversity_weight: float,
) -> dict[str, float]:
    if not subset:
        return {
            "coverage_fraction": 0.0,
            "mean_pair_distance_normalized": 0.0,
            "composite_objective": 0.0,
        }
    covered = coverage_mask(subset, terms).bit_count()
    coverage = covered / len(terms["state_ids"])
    structural = subset_metrics(subset, ids, matrix, ids[0])
    diversity = float(structural["mean_pair_distance_normalized"])
    return {
        "coverage_fraction": float(coverage),
        "mean_pair_distance_normalized": diversity,
        "minimum_pair_distance_normalized": float(
            structural["minimum_pair_distance_normalized"]
        ),
        "composite_objective": float(coverage + diversity_weight * diversity),
    }


def objective_key(
    subset: tuple[str, ...],
    ids: list[str],
    matrix: np.ndarray,
    terms: dict[str, Any],
    diversity_weight: float,
) -> tuple[float, tuple[str, ...]]:
    value = objective_components(subset, ids, matrix, terms, diversity_weight)
    return (-float(value["composite_objective"]), subset)


def direct_greedy(
    ids: list[str],
    matrix: np.ndarray,
    terms: dict[str, Any],
    target_k: int,
    diversity_weight: float,
) -> tuple[str, ...]:
    selected: tuple[str, ...] = ()
    for _ in range(target_k):
        candidates = [
            tuple(sorted((*selected, conformer_id)))
            for conformer_id in ids
            if conformer_id not in selected
        ]
        selected = min(
            candidates,
            key=lambda subset: objective_key(
                subset, ids, matrix, terms, diversity_weight
            ),
        )
    return selected


def improve_by_swaps(
    start: tuple[str, ...],
    ids: list[str],
    matrix: np.ndarray,
    terms: dict[str, Any],
    target_k: int,
    diversity_weight: float,
) -> tuple[tuple[str, ...], dict[str, float], int]:
    current = tuple(sorted(start))
    if len(current) != target_k:
        raise ValueError("local-search start has wrong cardinality")
    current_metrics = objective_components(
        current, ids, matrix, terms, diversity_weight
    )
    iterations = 0
    while True:
        selected = set(current)
        best = current
        best_metrics = current_metrics
        for outgoing in current:
            for incoming in ids:
                if incoming in selected:
                    continue
                candidate = tuple(sorted((selected - {outgoing}) | {incoming}))
                metrics = objective_components(
                    candidate, ids, matrix, terms, diversity_weight
                )
                value = float(metrics["composite_objective"])
                best_value = float(best_metrics["composite_objective"])
                if value > best_value + 1e-12 or (
                    math.isclose(value, best_value, abs_tol=1e-12)
                    and candidate < best
                ):
                    best, best_metrics = candidate, metrics
        if float(best_metrics["composite_objective"]) > float(
            current_metrics["composite_objective"]
        ) + 1e-12:
            current, current_metrics = best, best_metrics
            iterations += 1
            if iterations > len(ids) * target_k:
                raise RuntimeError("local search exceeded iteration guard")
            continue
        return current, current_metrics, iterations


def run_restarts(
    ids: list[str],
    matrix: np.ndarray,
    terms: dict[str, Any],
    target_k: int,
    reference_id: str,
    diversity_weight: float,
    restart_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    greedy = direct_greedy(ids, matrix, terms, target_k, diversity_weight)
    starts = [
        greedy,
        maxmin_seeded(ids, matrix, target_k, reference_id),
        maxsum_greedy(ids, matrix, target_k),
        tuple(ids[:target_k]),
    ]
    rng = random.Random(seed)
    for _ in range(max(0, restart_count - len(starts))):
        starts.append(tuple(sorted(rng.sample(ids, target_k))))
    rows = []
    for restart, start in enumerate(starts):
        selected, metrics, iterations = improve_by_swaps(
            start, ids, matrix, terms, target_k, diversity_weight
        )
        rows.append(
            {
                "restart": restart,
                "start_subset": "+".join(start),
                "selected_subset": "+".join(selected),
                "iterations": iterations,
                **metrics,
            }
        )
    return rows


def exact_oracle(
    ids: list[str],
    matrix: np.ndarray,
    terms: dict[str, Any],
    target_k: int,
    diversity_weight: float,
    state_limit: int,
) -> dict[str, Any] | None:
    state_count = math.comb(len(ids), target_k)
    if state_count > state_limit:
        return None
    best_subset: tuple[str, ...] | None = None
    best_metrics: dict[str, float] | None = None
    second_value = -math.inf
    for subset in itertools.combinations(ids, target_k):
        subset = tuple(subset)
        metrics = objective_components(
            subset, ids, matrix, terms, diversity_weight
        )
        value = float(metrics["composite_objective"])
        if best_subset is None or value > float(
            best_metrics["composite_objective"]
        ) + 1e-12 or (
            math.isclose(value, float(best_metrics["composite_objective"]), abs_tol=1e-12)
            and subset < best_subset
        ):
            if best_metrics is not None:
                second_value = max(
                    second_value, float(best_metrics["composite_objective"])
                )
            best_subset, best_metrics = subset, metrics
        elif value < float(best_metrics["composite_objective"]) - 1e-12:
            second_value = max(second_value, value)
    if best_subset is None or best_metrics is None:
        raise ValueError("exact enumeration produced no states")
    if not math.isfinite(second_value):
        second_value = float(best_metrics["composite_objective"])
    return {
        "selected_subset": best_subset,
        "metrics": best_metrics,
        "state_count": state_count,
        "best_second_gap": float(
            best_metrics["composite_objective"] - second_value
        ),
    }


def build_auxiliary_qubo(
    ids: list[str],
    matrix: np.ndarray,
    terms: dict[str, Any],
    target_k: int,
    diversity_weight: float,
    cardinality_penalty: float,
    constraint_penalty: float,
) -> dict[str, Any]:
    coefficients: dict[str, Any] = {
        "constant": 0.0,
        "linear": {},
        "quadratic": {},
    }
    x_names = {conformer_id: f"x__{conformer_id}" for conformer_id in ids}
    y_names = {state_id: f"y__{state_id}" for state_id in terms["state_ids"]}
    add_square(
        coefficients,
        -target_k,
        {x_names[conformer_id]: 1.0 for conformer_id in ids},
        cardinality_penalty,
    )
    slack_names: list[str] = []
    weights = slack_weights(int(terms["neighbor_count"]) - 1)
    for state_id in terms["state_ids"]:
        expression = {y_names[state_id]: 1.0}
        for index, weight in enumerate(weights):
            name = f"s__{state_id}__{index}"
            slack_names.append(name)
            expression[name] = float(weight)
        for conformer_id in terms["incidence"][state_id]:
            expression[x_names[conformer_id]] = -1.0
        add_square(coefficients, 0.0, expression, constraint_penalty)
        add_linear(
            coefficients,
            y_names[state_id],
            -float(terms["state_weight"]),
        )
    pair_denominator = max(1, math.comb(target_k, 2))
    for first_index, first in enumerate(ids):
        for second_index in range(first_index + 1, len(ids)):
            second = ids[second_index]
            add_quadratic(
                coefficients,
                x_names[first],
                x_names[second],
                -diversity_weight
                * float(matrix[first_index, second_index])
                / pair_denominator,
            )
    variables = sorted(
        set(coefficients["linear"])
        | {
            variable
            for key in coefficients["quadratic"]
            for variable in key.split("::", 1)
        }
    )
    return {
        "constant": float(coefficients["constant"]),
        "linear": {
            key: float(value) for key, value in coefficients["linear"].items()
        },
        "quadratic": {
            key: float(value) for key, value in coefficients["quadratic"].items()
        },
        "variables": variables,
        "variable_groups": {
            "x": sorted(x_names.values()),
            "state_y": sorted(y_names.values()),
            "state_slack": sorted(set(slack_names)),
        },
        "target_size": target_k,
        "neighborhood_fraction": float(terms["neighborhood_fraction"]),
        "neighbor_count": int(terms["neighbor_count"]),
        "diversity_weight": diversity_weight,
        "cardinality_penalty": cardinality_penalty,
        "constraint_penalty": constraint_penalty,
        "convention": (
            "Q(b)=constant+sum_v linear[v]*b_v+"
            "sum_u<v quadratic[u::v]*b_u*b_v; minimize Q"
        ),
    }


def assignment_for_subset(
    subset: tuple[str, ...], terms: dict[str, Any], qubo: dict[str, Any]
) -> dict[str, int]:
    selected = set(subset)
    assignment = {variable: 0 for variable in qubo["variables"]}
    for conformer_id in selected:
        assignment[f"x__{conformer_id}"] = 1
    weights = slack_weights(int(terms["neighbor_count"]) - 1)
    for state_id in terms["state_ids"]:
        count = sum(
            conformer_id in selected
            for conformer_id in terms["incidence"][state_id]
        )
        covered = int(count > 0)
        assignment[f"y__{state_id}"] = covered
        bits = binary_assignment(weights, count - covered)
        for index, bit in bits.items():
            assignment[f"s__{state_id}__{index}"] = bit
    return assignment


def qubo_energy(qubo: dict[str, Any], assignment: dict[str, int]) -> float:
    value = float(qubo["constant"])
    value += sum(
        float(coefficient) * int(assignment.get(variable, 0))
        for variable, coefficient in qubo["linear"].items()
    )
    value += sum(
        float(coefficient)
        * int(assignment.get(first, 0))
        * int(assignment.get(second, 0))
        for key, coefficient in qubo["quadratic"].items()
        for first, second in [key.split("::", 1)]
    )
    return float(value)


def qubo_hash(qubo: dict[str, Any]) -> str:
    payload = json.dumps(
        qubo, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def coefficient_stats(qubo: dict[str, Any]) -> dict[str, float]:
    coefficients = [
        abs(float(value))
        for value in list(qubo["linear"].values())
        + list(qubo["quadratic"].values())
    ]
    nonzero = [value for value in coefficients if value > 0.0]
    maximum = max(coefficients, default=0.0)
    minimum = min(nonzero, default=0.0)
    return {
        "max_abs_coefficient": maximum,
        "min_nonzero_abs_coefficient": minimum,
        "coefficient_dynamic_range": maximum / minimum if minimum else 0.0,
    }


def parse_subset(value: str) -> tuple[str, ...]:
    return tuple(sorted(part for part in value.split("+") if part))


def method_row(
    target_id: str,
    neighborhood_fraction: float,
    target_k: int,
    method: str,
    subset: tuple[str, ...],
    ids: list[str],
    matrix: np.ndarray,
    terms: dict[str, Any],
    diversity_weight: float,
    greedy_objective: float,
    solver: str,
) -> dict[str, Any]:
    metrics = objective_components(
        subset, ids, matrix, terms, diversity_weight
    )
    return {
        "target_id": target_id,
        "neighborhood_fraction": neighborhood_fraction,
        "k": target_k,
        "method": method,
        "solver": solver,
        "selected_subset": "+".join(subset),
        **metrics,
        "delta_composite_vs_direct_greedy": float(
            metrics["composite_objective"] - greedy_objective
        ),
    }


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported Stage22 schema")
    if config["evidence_timing"]["new_docking_jobs"]:
        raise ValueError("Stage22 cannot launch docking")
    outputs = {
        key: rooted(root, value) for key, value in config["outputs"].items()
    }
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage22 outputs exist; pass --overwrite")

    diagnostic = config["diagnostic"]
    k_values = [int(value) for value in diagnostic["k_values"]]
    fractions = [float(value) for value in diagnostic["neighborhood_fractions"]]
    primary_fraction = float(diagnostic["primary_neighborhood_fraction"])
    diversity_weight = float(diagnostic["diversity_weight"])
    restart_count = int(diagnostic["restart_count"])
    base_seed = int(diagnostic["base_seed"])
    exact_state_limit = int(diagnostic["exact_state_limit"])
    cardinality_penalty = float(diagnostic["cardinality_penalty"])
    constraint_penalty = float(diagnostic["constraint_penalty"])
    if primary_fraction not in fractions:
        raise ValueError("primary fraction is not in sensitivity schedule")
    if diversity_weight <= 0 or restart_count < 4:
        raise ValueError("invalid Stage22 settings")

    selection_rows: list[dict[str, Any]] = []
    restart_rows: list[dict[str, Any]] = []
    input_records: dict[str, Any] = {}
    target_models: dict[str, Any] = {}
    for target_id, spec in config["targets"].items():
        target = load_target(root, target_id, spec)
        ids = target["ids"]
        matrix = distance_matrix(ids, target["distances"])
        reference_id = str(spec["reference_id"])
        input_records[target_id] = {
            key: descriptor(root, path)
            for key, path in target["input_paths"].items()
        }
        target_record: dict[str, Any] = {
            "candidate_count": len(ids),
            "hard_gate_excluded_count": len(target["excluded_hard_gate"]),
            "reference_id": reference_id,
            "fraction_models": {},
        }
        for fraction in fractions:
            terms = build_coverage_terms(ids, matrix, fraction)
            fraction_key = f"{fraction:.6f}"
            fraction_record: dict[str, Any] = {
                "neighbor_count": terms["neighbor_count"],
                "candidate_cover_count_min": min(
                    terms["candidate_cover_counts"].values()
                ),
                "candidate_cover_count_median": statistics.median(
                    terms["candidate_cover_counts"].values()
                ),
                "candidate_cover_count_max": max(
                    terms["candidate_cover_counts"].values()
                ),
                "k_models": {},
            }
            for k in k_values:
                greedy = direct_greedy(
                    ids, matrix, terms, k, diversity_weight
                )
                greedy_metrics = objective_components(
                    greedy, ids, matrix, terms, diversity_weight
                )
                restarts = run_restarts(
                    ids,
                    matrix,
                    terms,
                    k,
                    reference_id,
                    diversity_weight,
                    restart_count,
                    base_seed
                    + k
                    + int(round(fraction * 10000))
                    + sum(ord(value) for value in target_id),
                )
                restart_rows.extend(
                    {
                        "target_id": target_id,
                        "neighborhood_fraction": fraction,
                        "k": k,
                        **row,
                    }
                    for row in restarts
                )
                best_restart_value = max(
                    float(row["composite_objective"]) for row in restarts
                )
                best_restart_subset = min(
                    row["selected_subset"]
                    for row in restarts
                    if math.isclose(
                        float(row["composite_objective"]),
                        best_restart_value,
                        abs_tol=1e-12,
                    )
                )
                exact = exact_oracle(
                    ids,
                    matrix,
                    terms,
                    k,
                    diversity_weight,
                    exact_state_limit,
                )
                selected = (
                    tuple(exact["selected_subset"])
                    if exact is not None
                    else parse_subset(best_restart_subset)
                )
                selected_metrics = objective_components(
                    selected, ids, matrix, terms, diversity_weight
                )
                best_frequency = sum(
                    row["selected_subset"] == "+".join(selected)
                    for row in restarts
                )
                maxmin = maxmin_seeded(ids, matrix, k, reference_id)
                maxsum = maxsum_greedy(ids, matrix, k)
                methods = {
                    CANDIDATE_METHOD: (
                        selected,
                        "exact_enumeration"
                        if exact is not None
                        else "deterministic_multistart_one_swap",
                    ),
                    "direct_greedy": (greedy, "deterministic_forward_greedy"),
                    "maxmin_seeded": (maxmin, "deterministic_seeded_maxmin"),
                    "maxsum_greedy": (maxsum, "deterministic_maxsum_greedy"),
                }
                method_records: dict[str, Any] = {}
                for method, (subset, solver) in methods.items():
                    row = method_row(
                        target_id,
                        fraction,
                        k,
                        method,
                        subset,
                        ids,
                        matrix,
                        terms,
                        diversity_weight,
                        float(greedy_metrics["composite_objective"]),
                        solver,
                    )
                    if method == CANDIDATE_METHOD:
                        row.update(
                            {
                                "best_restart_frequency": best_frequency,
                                "best_restart_fraction": best_frequency
                                / len(restarts),
                                "exact_oracle_available": exact is not None,
                                "multistart_matches_exact": (
                                    best_restart_subset == "+".join(selected)
                                    if exact is not None
                                    else None
                                ),
                            }
                        )
                    selection_rows.append(row)
                    method_records[method] = row

                qubo = build_auxiliary_qubo(
                    ids,
                    matrix,
                    terms,
                    k,
                    diversity_weight,
                    cardinality_penalty,
                    constraint_penalty,
                )
                assignment = assignment_for_subset(selected, terms, qubo)
                energy = qubo_energy(qubo, assignment)
                residual = abs(
                    energy + float(selected_metrics["composite_objective"])
                )
                if residual > 1e-7:
                    raise ValueError("coverage QUBO does not match reduced objective")
                coefficient = coefficient_stats(qubo)
                exact_gap = (
                    float(exact["best_second_gap"])
                    if exact is not None
                    else 0.0
                )
                fraction_record["k_models"][str(k)] = {
                    "selected_subset": list(selected),
                    "selected_metrics": selected_metrics,
                    "methods": method_records,
                    "state_count": math.comb(len(ids), k),
                    "exact_state_count": (
                        int(exact["state_count"]) if exact is not None else 0
                    ),
                    "exact_best_second_gap": exact_gap,
                    "best_restart_frequency": best_frequency,
                    "best_restart_fraction": best_frequency / len(restarts),
                    "unique_restart_solution_count": len(
                        {row["selected_subset"] for row in restarts}
                    ),
                    "qubo": {
                        "sha256": qubo_hash(qubo),
                        "variable_count": len(qubo["variables"]),
                        "x_count": len(qubo["variable_groups"]["x"]),
                        "state_y_count": len(
                            qubo["variable_groups"]["state_y"]
                        ),
                        "state_slack_count": len(
                            qubo["variable_groups"]["state_slack"]
                        ),
                        "linear_count": len(qubo["linear"]),
                        "quadratic_count": len(qubo["quadratic"]),
                        "selected_energy": energy,
                        "equivalence_residual": residual,
                        **coefficient,
                        "scaled_exact_best_second_gap": (
                            exact_gap / coefficient["max_abs_coefficient"]
                            if coefficient["max_abs_coefficient"]
                            else 0.0
                        ),
                    },
                }
            target_record["fraction_models"][fraction_key] = fraction_record
        target_models[target_id] = target_record

    write_csv(outputs["selection_csv"], selection_rows)
    write_csv(outputs["restart_csv"], restart_rows)
    data_boundary = {
        "ligand_labels_read": 0,
        "docking_scores_read": 0,
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
    model_record = {
        "schema_version": "1.0",
        "status": "stage22_structural_state_coverage_qubo_model_record",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "inputs": input_records,
        "target_models": target_models,
        "data_boundary": data_boundary,
    }
    write_json(outputs["model_record_json"], model_record)

    primary_rows = [
        row
        for row in selection_rows
        if row["method"] == CANDIDATE_METHOD
        and math.isclose(
            float(row["neighborhood_fraction"]), primary_fraction, abs_tol=1e-12
        )
    ]
    required_k = int(config["go_no_go"]["required_common_k"])
    primary_required = [row for row in primary_rows if int(row["k"]) == required_k]
    strict_primary_targets = sorted(
        row["target_id"]
        for row in primary_required
        if float(row["delta_composite_vs_direct_greedy"])
        > float(config["go_no_go"]["minimum_objective_gain"])
        and float(row["best_restart_fraction"])
        >= float(config["go_no_go"]["minimum_best_restart_fraction"])
    )
    sensitivity_pass: dict[str, Any] = {}
    for target_id in sorted(config["targets"]):
        rows = [
            row
            for row in selection_rows
            if row["target_id"] == target_id
            and row["method"] == CANDIDATE_METHOD
            and int(row["k"]) == required_k
        ]
        positive = sum(
            float(row["delta_composite_vs_direct_greedy"])
            > float(config["go_no_go"]["minimum_objective_gain"])
            for row in rows
        )
        sensitivity_pass[target_id] = {
            "positive_fraction_count": positive,
            "fraction_count": len(rows),
            "passed": positive
            >= int(config["go_no_go"]["minimum_positive_sensitivity_fractions"]),
        }
    go_passed = (
        strict_primary_targets == sorted(config["targets"])
        and all(value["passed"] for value in sensitivity_pass.values())
    )
    decision = {
        "required_common_k": required_k,
        "strict_primary_targets": strict_primary_targets,
        "sensitivity": sensitivity_pass,
        "structural_coverage_gate_passed": go_passed,
        "matched_small_docking_preregistration_authorized": go_passed,
        "new_docking_jobs_authorized_by_this_stage": False,
        "quantum_hardware_authorized": False,
    }

    report_lines = [
        "# Stage 22: structural-state coverage QUBO",
        "",
        "This is a post-hoc, structure-only model-design screen. No ligand label or docking result was read.",
        "",
        "## Frozen objective",
        "",
        "`F(S) = covered structural-state fraction + 0.15 * mean pairwise structural distance`",
        "",
        f"Primary neighborhood fraction: {primary_fraction:.2f}; sensitivity fractions: {', '.join(f'{value:.2f}' for value in fractions)}.",
        "",
        "## Primary results",
        "",
        "| Target | k | Coverage QUBO | Direct greedy | Delta | Coverage | Diversity | Restart agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selection_index = {
        (
            row["target_id"],
            float(row["neighborhood_fraction"]),
            int(row["k"]),
            row["method"],
        ): row
        for row in selection_rows
    }
    for row in sorted(primary_rows, key=lambda value: (value["target_id"], int(value["k"]))):
        greedy = selection_index[
            (
                row["target_id"],
                primary_fraction,
                int(row["k"]),
                "direct_greedy",
            )
        ]
        report_lines.append(
            f"| {row['target_id']} | {row['k']} | {float(row['composite_objective']):.6f} | {float(greedy['composite_objective']):.6f} | {float(row['delta_composite_vs_direct_greedy']):.6f} | {float(row['coverage_fraction']):.6f} | {float(row['mean_pair_distance_normalized']):.6f} | {float(row['best_restart_fraction']):.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Structural coverage gate passed: `{str(go_passed).lower()}`.",
            "Passing authorizes only preparation of a separate small matched-docking preregistration. It does not authorize docking execution or quantum hardware.",
            "",
            "## Interpretation boundary",
            "",
            config["interpretation_boundary"],
        ]
    )
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text(
        "\n".join(report_lines) + "\n", encoding="ascii"
    )

    result = {
        "schema_version": "1.0",
        "status": "stage22_structural_state_coverage_qubo_complete",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "implementation": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "inputs": input_records,
        "target_models": target_models,
        "decision": decision,
        "data_boundary": data_boundary,
        "outputs": {
            key: descriptor(root, path)
            for key, path in outputs.items()
            if key != "result_json"
        },
        "result_json": {
            "path": outputs["result_json"].relative_to(root).as_posix()
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": decision,
                "result_json": result["result_json"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage22_structural_state_coverage_qubo.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
