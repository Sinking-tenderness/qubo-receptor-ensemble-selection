"""Screen a frozen adaptive receptor-pool cardinality staircase."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage19e_cross_target_qubo_v2 import (
    descending_rank,
    load_target,
    output_descriptor,
    read_json,
    rooted,
    score_subsets,
    verified,
    write_csv,
    write_json,
)
from scripts.diagnose_stage19i_objective_adequacy_noise_screen import (
    MATRIX_IDS,
    METRIC_IDS,
    all_subsets,
    assignment,
    build_qubo,
    build_terms,
    minmax,
    qubo_energy,
    reduced_objective,
    score_mixed_subsets,
    validate_assignment,
)
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import make_context


BASELINE_METHODS = ("direct_greedy", "additive_top_k", "exact_robust_oracle")
CANDIDATE_METHOD = "adaptive_qubo"


def write_singleton_qubo(
    receptor_ids: list[str],
    singleton_values: dict[str, float],
    penalty: float,
) -> dict[str, Any]:
    normalized = minmax(singleton_values)
    coefficients = {
        "constant": float(penalty),
        "linear": {
            receptor_id: -float(normalized[receptor_id]) - float(penalty)
            for receptor_id in receptor_ids
        },
        "quadratic": {
            f"{first}::{second}": 2.0 * float(penalty)
            for first, second in itertools.combinations(receptor_ids, 2)
        },
    }
    return {
        **coefficients,
        "variables": list(receptor_ids),
        "variable_groups": {
            "x": list(receptor_ids),
            "active_y": [],
            "decoy_z": [],
            "active_slack": [],
        },
        "target_size": 1,
        "singleton_normalized": normalized,
        "cardinality_penalty": float(penalty),
        "convention": (
            "Q(x)=constant+sum_i linear[i]*x_i+"
            "sum_i<j quadratic[i::j]*x_i*x_j; minimize Q"
        ),
    }


def singleton_assignment(
    qubo: dict[str, Any], selected: tuple[str, ...]
) -> dict[str, int]:
    values = {variable: 0 for variable in qubo["variables"]}
    for receptor_id in selected:
        values[receptor_id] = 1
    return values


def coefficient_stats(
    qubo: dict[str, Any], best_energy: float, second_energy: float
) -> dict[str, float]:
    coefficients = list(qubo["linear"].values()) + list(qubo["quadratic"].values())
    absolute = [abs(float(value)) for value in coefficients]
    nonzero = [value for value in absolute if value > 0.0]
    maximum = max(absolute, default=0.0)
    raw_gap = float(second_energy - best_energy)
    return {
        "max_abs_coefficient": maximum,
        "min_nonzero_abs_coefficient": min(nonzero, default=0.0),
        "coefficient_dynamic_range": maximum / min(nonzero) if nonzero else 0.0,
        "raw_best_second_gap": raw_gap,
        "scaled_best_second_gap": raw_gap / maximum if maximum else 0.0,
    }


def singleton_certificate(
    qubo: dict[str, Any], receptor_ids: list[str], singleton_values: dict[str, float]
) -> dict[str, Any]:
    subsets = [(receptor_id,) for receptor_id in receptor_ids]
    energies = [
        qubo_energy(qubo, singleton_assignment(qubo, subset)) for subset in subsets
    ]
    order = sorted(range(len(subsets)), key=lambda index: (energies[index], subsets[index]))
    selected_index = order[0]
    stats = coefficient_stats(qubo, energies[order[0]], energies[order[1]])
    return {
        "subsets": subsets,
        "selected_index": selected_index,
        "selected_subset": list(subsets[selected_index]),
        "selected_objective": float(singleton_values[subsets[selected_index][0]]),
        "selected_energy": float(energies[selected_index]),
        "state_count": len(subsets),
        "equivalence_residual": 0.0,
        **stats,
    }


def threshold_certificate(
    terms: dict[str, Any],
    qubo: dict[str, Any],
    receptor_ids: list[str],
    target_size: int,
    decoy_weight: float,
    redundancy_weight: float,
) -> dict[str, Any]:
    subsets = [
        tuple(sorted(value))
        for value in itertools.combinations(receptor_ids, target_size)
    ]
    best_subset: tuple[str, ...] | None = None
    second_subset: tuple[str, ...] | None = None
    best_objective = -math.inf
    second_objective = -math.inf
    for subset in subsets:
        objective = reduced_objective(
            terms, subset, decoy_weight, redundancy_weight
        )
        if best_subset is None or objective > best_objective or (
            objective == best_objective and subset < best_subset
        ):
            second_subset = best_subset
            second_objective = best_objective
            best_subset = subset
            best_objective = objective
        elif objective > second_objective:
            second_objective = objective
            second_subset = subset
    if best_subset is None or second_subset is None:
        raise ValueError("no receptor subsets were generated")
    selected_assignment = assignment(terms, qubo, best_subset)
    validate_assignment(terms, selected_assignment, target_size)
    selected_energy = qubo_energy(qubo, selected_assignment)
    second_assignment = assignment(terms, qubo, second_subset)
    validate_assignment(terms, second_assignment, target_size)
    second_energy = qubo_energy(qubo, second_assignment)
    residual = abs((selected_energy + best_objective) - (second_energy + second_objective))
    stats = coefficient_stats(qubo, selected_energy, second_energy)
    return {
        "selected_subset": list(best_subset),
        "selected_objective": float(best_objective),
        "selected_energy": float(selected_energy),
        "second_subset": list(second_subset),
        "second_objective": float(second_objective),
        "second_energy": float(second_energy),
        "state_count": len(subsets),
        "equivalence_residual_sample": float(residual),
        "equivalence_states_evaluated": 2,
        **stats,
    }


def score_all_sizes(
    context: dict[str, Any],
    receptor_ids: list[str],
    maximum_k: int,
    split: str,
    alpha: float,
) -> tuple[list[tuple[str, ...]], dict[str, np.ndarray]]:
    subsets = all_subsets(receptor_ids, maximum_k)
    return subsets, score_mixed_subsets(context, subsets, receptor_ids, split, alpha)


def utility_map(
    subsets: list[tuple[str, ...]], values: dict[str, np.ndarray]
) -> dict[tuple[str, ...], float]:
    return {
        subset: float(values["robust_composite"][index])
        for index, subset in enumerate(subsets)
    }


def greedy_path(
    utilities: dict[tuple[str, ...], float],
    receptor_ids: list[str],
    maximum_k: int,
) -> dict[int, tuple[str, ...]]:
    def best(candidates: list[tuple[str, ...]]) -> tuple[str, ...]:
        return min(candidates, key=lambda subset: (-utilities[subset], subset))

    current = best([(receptor_id,) for receptor_id in receptor_ids])
    path = {1: current}
    for size in range(2, maximum_k + 1):
        current = best(
            [
                tuple(sorted((*current, receptor_id)))
                for receptor_id in receptor_ids
                if receptor_id not in current
            ]
        )
        path[size] = current
    return path


def metrics_for_subset(
    values: dict[str, np.ndarray],
    subset_index: dict[tuple[str, ...], int],
    subset: tuple[str, ...],
) -> dict[str, float]:
    index = subset_index[tuple(sorted(subset))]
    return {key: float(values[key][index]) for key in METRIC_IDS}


def make_method_row(
    target_id: str,
    outer_fold: int,
    k: int,
    objective_id: str,
    method: str,
    subset: tuple[str, ...],
    subset_index: dict[tuple[str, ...], int],
    train_values: dict[str, np.ndarray],
    holdout_values: dict[str, np.ndarray],
    **extra: Any,
) -> dict[str, Any]:
    subset = tuple(sorted(subset))
    index = subset_index[subset]
    return {
        "target_id": target_id,
        "outer_fold": outer_fold,
        "k": k,
        "objective_id": objective_id,
        "method": method,
        "selected_subset": "+".join(subset),
        "holdout_rank_at_k": descending_rank(
            holdout_values["robust_composite"], index
        ),
        **{
            f"train_{key}": value
            for key, value in metrics_for_subset(
                train_values, subset_index, subset
            ).items()
        },
        **{
            f"holdout_{key}": value
            for key, value in metrics_for_subset(
                holdout_values, subset_index, subset
            ).items()
        },
        **extra,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["target_id"], int(row["k"]), row["method"])].append(row)
    output: list[dict[str, Any]] = []
    for (target_id, k, method), selected in sorted(grouped.items()):
        output.append(
            {
                "target_id": target_id,
                "k": k,
                "method": method,
                "objective_ids": sorted({row["objective_id"] for row in selected}),
                "fold_count": len(selected),
                "mean_holdout_robust_composite": statistics.fmean(
                    float(row["holdout_robust_composite"]) for row in selected
                ),
                "standard_error_holdout_robust_composite": (
                    statistics.stdev(
                        float(row["holdout_robust_composite"]) for row in selected
                    )
                    / math.sqrt(len(selected))
                    if len(selected) > 1
                    else 0.0
                ),
                "worst_holdout_robust_composite": min(
                    float(row["holdout_robust_composite"]) for row in selected
                ),
                "mean_holdout_primary": statistics.fmean(
                    float(row["holdout_primary"]) for row in selected
                ),
                "mean_holdout_rank": statistics.fmean(
                    int(row["holdout_rank_at_k"]) for row in selected
                ),
                "selected_subsets": [row["selected_subset"] for row in selected],
            }
        )
    return output


def increment_rows(
    candidate_aggregate: list[dict[str, Any]],
    target_ids: list[str],
    maximum_k: int,
    minimum_mean_delta: float,
    minimum_positive_folds: int,
    stop_after_failures: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexed = {
        (row["target_id"], int(row["k"])): row
        for row in candidate_aggregate
        if row["method"] == CANDIDATE_METHOD
    }
    rows: list[dict[str, Any]] = []
    for target_id in target_ids:
        failures = 0
        for k in range(1, maximum_k + 1):
            current = indexed[(target_id, k)]
            previous = indexed.get((target_id, k - 1))
            if previous is None:
                rows.append(
                    {
                        "target_id": target_id,
                        "k": k,
                        "previous_k": None,
                        "mean_delta": None,
                        "positive_fold_count": None,
                        "continue_condition_passed": None,
                        "consecutive_failure_count": 0,
                    }
                )
                continue
            current_values = [
                value
                for value in current["selected_subsets"]
            ]
            previous_values = [
                value
                for value in previous["selected_subsets"]
            ]
            # The fold-level delta is computed separately below from rows; this
            # placeholder is replaced by the caller's paired values.
            del current_values, previous_values
            rows.append(
                {
                    "target_id": target_id,
                    "k": k,
                    "previous_k": k - 1,
                    "mean_delta": 0.0,
                    "positive_fold_count": 0,
                    "continue_condition_passed": False,
                    "consecutive_failure_count": failures + 1,
                }
            )
            failures += 1
    return rows, {}


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 20 implementation path differs")
    input_paths = {
        key: verified(root, descriptor)
        for key, descriptor in config["inputs"].items()
    }
    stage19e_config = read_json(input_paths["stage19e_config"])
    stage19e_result = read_json(input_paths["stage19e_result"])
    stage19e_audit = read_json(input_paths["stage19e_audit"])
    stage19i_result = read_json(input_paths["stage19i_result"])
    stage19i_audit = read_json(input_paths["stage19i_audit"])
    if stage19e_result.get("status") != "stage19e_quadratic_v2_not_supported_do_not_amend_bace1":
        raise ValueError("Stage 19e status differs")
    if stage19e_audit.get("status") != "stage19e_cross_target_qubo_v2_nested_diagnostic_audit_ok":
        raise ValueError("Stage 19e audit differs")
    if stage19i_result.get("status") != "stage19i_no_candidate_hardware_ready_do_not_execute_quantum":
        raise ValueError("Stage 19i status differs")
    if stage19i_audit.get("status") != "stage19i_objective_adequacy_noise_screen_audit_ok":
        raise ValueError("Stage 19i audit differs")

    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage 20 outputs exist; pass --overwrite")
    diagnostic = config["diagnostic"]
    minimum_k = int(diagnostic["minimum_k"])
    maximum_k = int(diagnostic["maximum_k"])
    outer_count = int(diagnostic["outer_fold_count"])
    fold_seed = int(diagnostic["fold_seed"])
    alpha = float(diagnostic["bedroc_alpha"])
    coverage_fraction = float(diagnostic["coverage_fraction"])
    decoy_weight = float(diagnostic["decoy_weight"])
    cardinality_penalty = float(diagnostic["cardinality_penalty"])
    constraint_penalty = float(diagnostic["constraint_penalty"])
    schedule = {int(item["k"]): item for item in diagnostic["k_schedule"]}
    if sorted(schedule) != list(range(minimum_k, maximum_k + 1)):
        raise ValueError("adaptive k schedule is incomplete")

    fold_rows: list[dict[str, Any]] = []
    full_models: dict[str, dict[str, Any]] = {}
    target_dimensions: dict[str, Any] = {}

    for target_id, target_spec in stage19e_config["targets"].items():
        print(f"loading_target={target_id}", flush=True)
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        all_ids = {row["ligand_id"] for row in ligands}
        assignments = make_frozen_group_folds(ligands, outer_count, fold_seed)
        target_dimensions[target_id] = {
            "ligand_count": len(ligands),
            "active_count": sum(row["label"] == "active" for row in ligands),
            "receptor_count": len(receptor_ids),
            "subset_counts_by_k": {
                str(k): math.comb(len(receptor_ids), k)
                for k in range(minimum_k, maximum_k + 1)
            },
        }
        full_context = make_context(
            all_ids,
            set(),
            matrices,
            receptor_ids,
            {
                "coverage_fraction": float(target_spec["v1_qubo"]["coverage_fraction"]),
                "utility_metric": "bedroc",
            },
        )
        full_subsets, full_values = score_all_sizes(
            full_context, receptor_ids, maximum_k, "train", alpha
        )
        full_subset_index = {subset: index for index, subset in enumerate(full_subsets)}
        full_utility = utility_map(full_subsets, full_values)
        full_single_values = {
            receptor_id: full_utility[(receptor_id,)] for receptor_id in receptor_ids
        }
        full_models[target_id] = {}
        for k in range(minimum_k, maximum_k + 1):
            item = schedule[k]
            objective_id = str(item["objective_id"])
            if k == 1:
                qubo = write_singleton_qubo(
                    receptor_ids, full_single_values, cardinality_penalty
                )
                selected = min(
                    [(receptor_id,) for receptor_id in receptor_ids],
                    key=lambda subset: (-full_utility[subset], subset),
                )
                cert = singleton_certificate(qubo, receptor_ids, full_single_values)
                cert["selected_subset"] = list(selected)
            else:
                terms = build_terms(
                    full_context,
                    receptor_ids,
                    coverage_fraction,
                    int(item["active_threshold"]),
                    alpha,
                )
                qubo = build_qubo(
                    terms,
                    receptor_ids,
                    k,
                    decoy_weight,
                    float(item["redundancy_weight"]),
                    cardinality_penalty,
                    constraint_penalty,
                )
                cert = threshold_certificate(
                    terms,
                    qubo,
                    receptor_ids,
                    k,
                    decoy_weight,
                    float(item["redundancy_weight"]),
                )
            full_models[target_id][str(k)] = {
                "k": k,
                "objective_id": objective_id,
                "schedule": item,
                "selected_subset": cert["selected_subset"],
                "full_train_metrics": metrics_for_subset(
                    full_values,
                    full_subset_index,
                    tuple(cert["selected_subset"]),
                ),
                "selected_objective": cert["selected_objective"],
                "selected_energy": cert["selected_energy"],
                "state_count": cert["state_count"],
                "equivalence_residual_sample": cert.get(
                    "equivalence_residual", cert.get("equivalence_residual_sample", 0.0)
                ),
                "equivalence_states_evaluated": cert.get(
                    "equivalence_states_evaluated", 2
                ),
                "variable_count": len(qubo["variables"]),
                "linear_count": len(qubo["linear"]),
                "quadratic_count": len(qubo["quadratic"]),
                "max_abs_coefficient": cert["max_abs_coefficient"],
                "min_nonzero_abs_coefficient": cert["min_nonzero_abs_coefficient"],
                "coefficient_dynamic_range": cert["coefficient_dynamic_range"],
                "raw_best_second_gap": cert["raw_best_second_gap"],
                "scaled_best_second_gap": cert["scaled_best_second_gap"],
                "terms": {} if k == 1 else terms,
                "qubo": qubo,
            }

        for outer_fold in range(outer_count):
            print(f"target={target_id} outer_fold={outer_fold}", flush=True)
            holdout_ids = {
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == outer_fold
            }
            train_ids = all_ids - holdout_ids
            context = make_context(
                train_ids,
                holdout_ids,
                matrices,
                receptor_ids,
                {
                    "coverage_fraction": float(target_spec["v1_qubo"]["coverage_fraction"]),
                    "utility_metric": "bedroc",
                },
            )
            subsets, train_values = score_all_sizes(
                context, receptor_ids, maximum_k, "train", alpha
            )
            _, holdout_values = score_all_sizes(
                context, receptor_ids, maximum_k, "validation", alpha
            )
            subset_index = {subset: index for index, subset in enumerate(subsets)}
            utilities = utility_map(subsets, train_values)
            path = greedy_path(utilities, receptor_ids, maximum_k)
            for k in range(minimum_k, maximum_k + 1):
                exact_subset = min(
                    [subset for subset in subsets if len(subset) == k],
                    key=lambda subset: (-utilities[subset], subset),
                )
                additive_subset = tuple(
                    sorted(
                        sorted(
                            receptor_ids,
                            key=lambda receptor_id: (
                                -utilities[(receptor_id,)],
                                receptor_id,
                            ),
                        )[:k]
                    )
                )
                item = schedule[k]
                objective_id = str(item["objective_id"])
                for method, subset in (
                    ("direct_greedy", path[k]),
                    ("additive_top_k", additive_subset),
                    ("exact_robust_oracle", exact_subset),
                ):
                    fold_rows.append(
                        make_method_row(
                            target_id,
                            outer_fold,
                            k,
                            objective_id,
                            method,
                            subset,
                            subset_index,
                            train_values,
                            holdout_values,
                        )
                    )
                if k == 1:
                    singleton_values = {
                        receptor_id: utilities[(receptor_id,)]
                        for receptor_id in receptor_ids
                    }
                    qubo = write_singleton_qubo(
                        receptor_ids, singleton_values, cardinality_penalty
                    )
                    subset = min(
                        [(receptor_id,) for receptor_id in receptor_ids],
                        key=lambda value: (-utilities[value], value),
                    )
                    cert = singleton_certificate(qubo, receptor_ids, singleton_values)
                else:
                    terms = build_terms(
                        context,
                        receptor_ids,
                        coverage_fraction,
                        int(item["active_threshold"]),
                        alpha,
                    )
                    qubo = build_qubo(
                        terms,
                        receptor_ids,
                        k,
                        decoy_weight,
                        float(item["redundancy_weight"]),
                        cardinality_penalty,
                        constraint_penalty,
                    )
                    cert = threshold_certificate(
                        terms,
                        qubo,
                        receptor_ids,
                        k,
                        decoy_weight,
                        float(item["redundancy_weight"]),
                    )
                    subset = tuple(cert["selected_subset"])
                fold_rows.append(
                    make_method_row(
                        target_id,
                        outer_fold,
                        k,
                        objective_id,
                        CANDIDATE_METHOD,
                        subset,
                        subset_index,
                        train_values,
                        holdout_values,
                        qubo_variable_count=len(qubo["variables"]),
                        qubo_state_count=cert["state_count"],
                        qubo_max_abs_coefficient=cert["max_abs_coefficient"],
                        qubo_scaled_best_second_gap=cert["scaled_best_second_gap"],
                        qubo_equivalence_residual=cert.get(
                            "equivalence_residual", cert.get("equivalence_residual_sample", 0.0)
                        ),
                        selected_objective=cert["selected_objective"],
                        selected_energy=cert["selected_energy"],
                    )
                )

    aggregates = aggregate_rows(fold_rows)
    candidate_aggregates = [
        row for row in aggregates if row["method"] == CANDIDATE_METHOD
    ]
    target_ids = sorted(target_dimensions)
    candidate_index = {
        (row["target_id"], int(row["k"])): row for row in candidate_aggregates
    }
    incremental_rows: list[dict[str, Any]] = []
    for target_id in target_ids:
        failure_count = 0
        for k in range(minimum_k, maximum_k + 1):
            current = candidate_index[(target_id, k)]
            if k == minimum_k:
                incremental_rows.append(
                    {
                        "target_id": target_id,
                        "k": k,
                        "previous_k": None,
                        "mean_delta": None,
                        "positive_fold_count": None,
                        "continue_condition_passed": None,
                        "consecutive_failure_count": 0,
                    }
                )
                continue
            previous = candidate_index[(target_id, k - 1)]
            current_rows = [
                row for row in fold_rows
                if row["target_id"] == target_id
                and int(row["k"]) == k
                and row["method"] == CANDIDATE_METHOD
            ]
            previous_rows = [
                row for row in fold_rows
                if row["target_id"] == target_id
                and int(row["k"]) == k - 1
                and row["method"] == CANDIDATE_METHOD
            ]
            previous_by_fold = {int(row["outer_fold"]): row for row in previous_rows}
            deltas = [
                float(row["holdout_robust_composite"])
                - float(previous_by_fold[int(row["outer_fold"])] ["holdout_robust_composite"])
                for row in current_rows
            ]
            mean_delta = statistics.fmean(deltas)
            positive_count = sum(value > 0.0 for value in deltas)
            passed = (
                mean_delta
                > float(diagnostic["minimum_target_mean_delta_to_continue"])
                and positive_count
                >= int(diagnostic["minimum_positive_folds_of_eight_to_continue"])
            )
            failure_count = 0 if passed else failure_count + 1
            incremental_rows.append(
                {
                    "target_id": target_id,
                    "k": k,
                    "previous_k": k - 1,
                    "mean_delta": mean_delta,
                    "positive_fold_count": positive_count,
                    "fold_deltas": deltas,
                    "continue_condition_passed": passed,
                    "consecutive_failure_count": failure_count,
                    "current_mean": current["mean_holdout_robust_composite"],
                    "previous_mean": previous["mean_holdout_robust_composite"],
                }
            )

    global_by_k: dict[int, list[float]] = defaultdict(list)
    for row in fold_rows:
        if row["method"] == CANDIDATE_METHOD:
            global_by_k[int(row["k"])].append(float(row["holdout_robust_composite"]))
    global_curve = []
    for k in range(minimum_k, maximum_k + 1):
        values = global_by_k[k]
        global_curve.append(
            {
                "k": k,
                "mean": statistics.fmean(values),
                "standard_error": statistics.stdev(values) / math.sqrt(len(values)),
                "fold_count": len(values),
            }
        )
    best_global = max(global_curve, key=lambda row: (row["mean"], -row["k"]))
    one_se_threshold = best_global["mean"] - best_global["standard_error"]
    one_se_k = min(
        row["k"] for row in global_curve if row["mean"] >= one_se_threshold
    )
    stop_k = maximum_k
    for target_id in target_ids:
        target_rows = [
            row for row in incremental_rows if row["target_id"] == target_id
        ]
        for row in target_rows:
            if row["consecutive_failure_count"] is not None and row[
                "consecutive_failure_count"
            ] >= int(diagnostic["stop_after_consecutive_failures"]):
                stop_k = min(stop_k, int(row["k"]))
                break

    data_boundary = {
        "train_rows_read_by_target": {
            target_id: int(spec["expected"]["ligand_count"])
            for target_id, spec in stage19e_config["targets"].items()
        },
        "new_docking_jobs": 0,
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
        "bace1_docking_rows_read": 0,
        "quantum_hardware_jobs": 0,
    }
    write_csv(outputs["fold_methods_csv"], fold_rows)
    write_csv(outputs["incremental_csv"], incremental_rows)
    write_json(
        outputs["model_record_json"],
        {
            "schema_version": "1.0",
            "status": "stage20_adaptive_k_train_only_models_complete",
            "data_boundary": data_boundary,
            "target_models": full_models,
        },
    )
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "stage20_adaptive_k_train_only_screen_complete",
        "experiment_id": config["experiment_id"],
        "experiment_class": config["evidence_timing"]["experiment_class"],
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "data_boundary": data_boundary,
        "target_dimensions": target_dimensions,
        "fold_method_aggregate": aggregates,
        "incremental_results": incremental_rows,
        "global_candidate_curve": global_curve,
        "one_standard_error": {
            "best_k": best_global["k"],
            "best_mean": best_global["mean"],
            "best_standard_error": best_global["standard_error"],
            "threshold": one_se_threshold,
            "recommended_smallest_k": one_se_k,
        },
        "stopping_recommendation": {
            "recommended_stop_k": stop_k,
            "rule": config["stopping_rule"],
        },
        "full_train_models": {
            target_id: {
                k: {
                    key: value
                    for key, value in model.items()
                    if key not in {"terms", "qubo"}
                }
                for k, model in models.items()
            }
            for target_id, models in full_models.items()
        },
        "outputs": {
            key: output_descriptor(root, path)
            for key, path in outputs.items()
            if key not in {"result_json", "report_md"}
        },
        "next_gate": "freeze_recommended_k_and_objective_for_independent_target_review_before_hardware",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    report_lines = [
        "# Stage 20 adaptive receptor-pool cardinality screen",
        "",
        "Post-hoc train-only review on MK14 and PPARG. Every k uses the same external robust BEDROC20 comparison metric; no new docking or quantum hardware was used.",
        "",
        "## Candidate curve",
        "",
        "| k | Mean holdout robust composite | Standard error |",
        "|---:|---:|---:|",
    ]
    for row in global_curve:
        report_lines.append(
            f"| {row['k']} | {row['mean']:.6f} | {row['standard_error']:.6f} |"
        )
    report_lines.extend(
        [
            "",
            f"- Best observed k: `{best_global['k']}`",
            f"- One-standard-error smallest k: `{one_se_k}`",
            f"- Consecutive-failure stop recommendation: `{stop_k}`",
            "",
            "## Increment rule",
            "",
            "| Target | k | Mean increment | Positive folds | Continue? |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in incremental_rows:
        if row["previous_k"] is not None:
            report_lines.append(
                f"| {row['target_id']} | {row['k']} | {row['mean_delta']:.6f} | "
                f"{row['positive_fold_count']} | {row['continue_condition_passed']} |"
            )
    report_lines.extend(
        [
            "",
            "The recommendation is exploratory and cannot amend a prior validation gate or establish quantum advantage.",
        ]
    )
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report_lines) + "\n", encoding="ascii")
    print(
        json.dumps(
            {
                "status": result["status"],
                "global_candidate_curve": result["global_candidate_curve"],
                "one_standard_error": result["one_standard_error"],
                "stopping_recommendation": result["stopping_recommendation"],
                "outputs": result["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
