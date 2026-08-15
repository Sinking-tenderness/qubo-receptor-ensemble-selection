"""Develop an auxiliary-variable coverage QUBO on frozen train matrices."""

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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage19e_cross_target_qubo_v2 import (
    descending_rank,
    load_target,
    output_descriptor,
    read_csv,
    read_json,
    rooted,
    safe_spearman,
    score_subsets,
    subset_metrics,
    verified,
    write_csv,
    write_json,
)
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import make_context


BASELINE_METHODS = (
    "direct_greedy",
    "additive_nested",
    "v1_qubo_exact",
    "stable_singleton_linear",
)
CANDIDATE_METHOD = "auxiliary_coverage_nested"
MATRIX_IDS = ("primary", "sensitivity", "seed0", "seed1", "seed2")
SEED_IDS = ("seed0", "seed1", "seed2")
METRIC_IDS = (
    "primary",
    "sensitivity",
    "mean_seed",
    "worst_seed",
    "robust_composite",
)


def minmax(values: dict[str, float]) -> dict[str, float]:
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-15):
        return {key: 0.0 for key in values}
    return {key: (float(value) - low) / (high - low) for key, value in values.items()}


def receptor_pair_key(first: str, second: str) -> str:
    return "__".join(sorted((first, second)))


def add_linear(coefficients: dict[str, Any], variable: str, value: float) -> None:
    coefficients["linear"][variable] = coefficients["linear"].get(variable, 0.0) + float(value)


def add_quadratic(
    coefficients: dict[str, Any], first: str, second: str, value: float
) -> None:
    if first == second:
        add_linear(coefficients, first, value)
        return
    key = "::".join(sorted((first, second)))
    coefficients["quadratic"][key] = coefficients["quadratic"].get(key, 0.0) + float(value)


def add_square(
    coefficients: dict[str, Any], constant: float, terms: dict[str, float], weight: float
) -> None:
    """Add weight*(constant + sum(term_i*x_i))^2 for binary variables."""
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
    total = 0
    next_weight = 1
    while total < maximum:
        value = min(next_weight, maximum - total)
        weights.append(value)
        total += value
        next_weight *= 2
    return weights


def binary_assignment(weights: list[int], value: int) -> dict[int, int]:
    if value < 0 or value > sum(weights):
        raise ValueError("slack value is outside its encoding range")
    for bits in itertools.product((0, 1), repeat=len(weights)):
        if sum(weight * bit for weight, bit in zip(weights, bits)) == value:
            return {index: int(bit) for index, bit in enumerate(bits)}
    raise ValueError("slack encoding cannot represent a required value")


def rank_fraction_rows(
    context: dict[str, Any], receptor_ids: list[str]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    """Return median rank fractions and labels from all five train scenarios."""
    rows_by_matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for matrix_id in MATRIX_IDS:
        rows = context["matrices"][matrix_id]["train"]
        rows_by_matrix[matrix_id] = {str(row["ligand_id"]): row for row in rows}
    ligand_ids = sorted(rows_by_matrix["primary"])
    rank_values: dict[str, dict[str, list[float]]] = {
        ligand_id: {receptor_id: [] for receptor_id in receptor_ids}
        for ligand_id in ligand_ids
    }
    labels = {
        ligand_id: str(rows_by_matrix["primary"][ligand_id]["label"])
        for ligand_id in ligand_ids
    }
    for matrix_id in MATRIX_IDS:
        for receptor_id in receptor_ids:
            ordered = sorted(
                ligand_ids,
                key=lambda ligand_id: (
                    float(rows_by_matrix[matrix_id][ligand_id][receptor_id]),
                    ligand_id,
                ),
            )
            denominator = max(1, len(ordered) - 1)
            for rank, ligand_id in enumerate(ordered):
                rank_values[ligand_id][receptor_id].append(rank / denominator)
    median_fraction = {
        ligand_id: {
            receptor_id: float(np.median(values))
            for receptor_id, values in receptor_values.items()
        }
        for ligand_id, receptor_values in rank_values.items()
    }
    return median_fraction, labels


def build_coverage_terms(
    context: dict[str, Any],
    receptor_ids: list[str],
    coverage_fraction: float,
    bedroc_alpha: float,
    singleton_values: dict[str, float],
) -> dict[str, Any]:
    if not 0.0 < coverage_fraction <= 1.0:
        raise ValueError("coverage fraction must be in (0, 1]")
    ranks, labels = rank_fraction_rows(context, receptor_ids)
    active_ids = sorted(ligand_id for ligand_id in labels if labels[ligand_id] == "active")
    decoy_ids = sorted(ligand_id for ligand_id in labels if labels[ligand_id] == "decoy")
    active_incidence: dict[str, list[str]] = {}
    decoy_incidence: dict[str, list[str]] = {}
    active_raw_weights = {
        ligand_id: math.exp(-bedroc_alpha * min(ranks[ligand_id].values()))
        for ligand_id in active_ids
    }
    decoy_raw_weights = {
        ligand_id: math.exp(-bedroc_alpha * min(ranks[ligand_id].values()))
        for ligand_id in decoy_ids
    }
    active_total = sum(active_raw_weights.values())
    decoy_total = sum(decoy_raw_weights.values())
    active_weights = {
        ligand_id: value / active_total for ligand_id, value in active_raw_weights.items()
    }
    decoy_weights = {
        ligand_id: value / decoy_total for ligand_id, value in decoy_raw_weights.items()
    }
    for ligand_id in active_ids:
        active_incidence[ligand_id] = [
            receptor_id
            for receptor_id in receptor_ids
            if ranks[ligand_id][receptor_id] <= coverage_fraction
        ]
    for ligand_id in decoy_ids:
        decoy_incidence[ligand_id] = [
            receptor_id
            for receptor_id in receptor_ids
            if ranks[ligand_id][receptor_id] <= coverage_fraction
        ]
    score_columns = {
        receptor_id: np.asarray(
            [float(row[receptor_id]) for row in context["matrices"]["primary"]["train"]]
        )
        for receptor_id in receptor_ids
    }
    correlations: dict[str, float] = {}
    for first, second in itertools.combinations(receptor_ids, 2):
        value = float(spearmanr(score_columns[first], score_columns[second]).statistic)
        correlations[receptor_pair_key(first, second)] = (
            max(0.0, value) if math.isfinite(value) else 0.0
        )
    return {
        "coverage_fraction": coverage_fraction,
        "bedroc_alpha": bedroc_alpha,
        "active_ids": active_ids,
        "decoy_ids": decoy_ids,
        "active_incidence": active_incidence,
        "decoy_incidence": decoy_incidence,
        "active_weights": active_weights,
        "decoy_weights": decoy_weights,
        "singleton_utility": minmax(singleton_values),
        "correlations": correlations,
    }


def variable_names_for_terms(terms: dict[str, Any]) -> dict[str, list[str]]:
    x = [f"x__{receptor_id}" for receptor_id in terms["receptor_ids"]]
    y = [f"y__{ligand_id}" for ligand_id in terms["active_incidence"] if terms["active_incidence"][ligand_id]]
    z = [f"z__{ligand_id}" for ligand_id in terms["decoy_incidence"] if terms["decoy_incidence"][ligand_id]]
    slack: list[str] = []
    for ligand_id, incidence in terms["active_incidence"].items():
        if incidence:
            slack.extend(
                f"s__{ligand_id}__{index}"
                for index in range(len(slack_weights(len(incidence))))
            )
    return {"x": x, "active_y": y, "decoy_z": z, "active_slack": slack}


def build_auxiliary_qubo(
    terms: dict[str, Any],
    receptor_ids: list[str],
    target_size: int,
    decoy_weight: float,
    singleton_weight: float,
    redundancy_weight: float,
    cardinality_penalty: float,
    constraint_penalty: float,
) -> dict[str, Any]:
    if target_size < 1 or target_size > len(receptor_ids):
        raise ValueError("invalid target size")
    coefficients: dict[str, Any] = {
        "constant": 0.0,
        "linear": {},
        "quadratic": {},
    }
    variables = variable_names_for_terms({**terms, "receptor_ids": receptor_ids})
    x_names = {receptor_id: f"x__{receptor_id}" for receptor_id in receptor_ids}
    active_y_names = {
        ligand_id: f"y__{ligand_id}"
        for ligand_id, incidence in terms["active_incidence"].items()
        if incidence
    }
    decoy_z_names = {
        ligand_id: f"z__{ligand_id}"
        for ligand_id, incidence in terms["decoy_incidence"].items()
        if incidence
    }

    # Exact cardinality on receptor variables.
    add_square(
        coefficients,
        -target_size,
        {x_names[receptor_id]: 1.0 for receptor_id in receptor_ids},
        cardinality_penalty,
    )

    # Active y variables: y + slack = number of selected covering receptors.
    slack_name_map: dict[tuple[str, int], str] = {}
    for ligand_id, incidence in terms["active_incidence"].items():
        if not incidence:
            continue
        weights = slack_weights(len(incidence))
        expression = {active_y_names[ligand_id]: 1.0}
        for index, weight in enumerate(weights):
            name = f"s__{ligand_id}__{index}"
            slack_name_map[(ligand_id, index)] = name
            expression[name] = float(weight)
        for receptor_id in incidence:
            expression[x_names[receptor_id]] = expression.get(
                x_names[receptor_id], 0.0
            ) - 1.0
        add_square(coefficients, 0.0, expression, constraint_penalty)
        add_linear(
            coefficients,
            active_y_names[ligand_id],
            -float(terms["active_weights"][ligand_id]),
        )

    # Decoy z variables: z is forced to one whenever a covering receptor is selected.
    # Its positive objective coefficient then makes it zero when no exposure exists.
    for ligand_id, incidence in terms["decoy_incidence"].items():
        if not incidence:
            continue
        z_name = decoy_z_names[ligand_id]
        add_linear(
            coefficients,
            z_name,
            decoy_weight * float(terms["decoy_weights"][ligand_id]),
        )
        for receptor_id in incidence:
            x_name = x_names[receptor_id]
            add_linear(coefficients, x_name, constraint_penalty)
            add_quadratic(coefficients, x_name, z_name, -constraint_penalty)

    # Singleton quality and receptor redundancy are secondary terms.
    singleton_scale = singleton_weight / target_size
    for receptor_id in receptor_ids:
        add_linear(
            coefficients,
            x_names[receptor_id],
            -singleton_scale * float(terms["singleton_utility"][receptor_id]),
        )
    pair_denominator = max(1, target_size * (target_size - 1) // 2)
    for first, second in itertools.combinations(receptor_ids, 2):
        key = receptor_pair_key(first, second)
        add_quadratic(
            coefficients,
            x_names[first],
            x_names[second],
            redundancy_weight
            * float(terms["correlations"].get(key, 0.0))
            / pair_denominator,
        )

    all_variables = sorted(
        set(variables["x"])
        | set(variables["active_y"])
        | set(variables["decoy_z"])
        | set(variables["active_slack"])
    )
    return {
        "constant": float(coefficients["constant"]),
        "linear": {key: float(value) for key, value in coefficients["linear"].items()},
        "quadratic": {
            key: float(value) for key, value in coefficients["quadratic"].items()
        },
        "variables": all_variables,
        "variable_groups": variables,
        "target_size": target_size,
        "decoy_weight": decoy_weight,
        "singleton_weight": singleton_weight,
        "redundancy_weight": redundancy_weight,
        "cardinality_penalty": cardinality_penalty,
        "constraint_penalty": constraint_penalty,
        "convention": (
            "Q(b)=constant+sum_v linear[v]*b_v+"
            "sum_u<v quadratic[u::v]*b_u*b_v; minimize Q"
        ),
    }


def reduced_objective(
    terms: dict[str, Any],
    receptor_ids: list[str],
    subset: tuple[str, ...],
    decoy_weight: float,
    singleton_weight: float,
    redundancy_weight: float,
) -> float:
    selected = set(subset)
    active = sum(
        float(terms["active_weights"][ligand_id])
        for ligand_id, incidence in terms["active_incidence"].items()
        if selected.intersection(incidence)
    )
    decoy = sum(
        float(terms["decoy_weights"][ligand_id])
        for ligand_id, incidence in terms["decoy_incidence"].items()
        if selected.intersection(incidence)
    )
    singleton = singleton_weight * statistics.fmean(
        float(terms["singleton_utility"][receptor_id]) for receptor_id in subset
    )
    pair_count = max(1, len(subset) * (len(subset) - 1) // 2)
    redundancy = redundancy_weight * statistics.fmean(
        float(terms["correlations"].get(receptor_pair_key(first, second), 0.0))
        for first, second in itertools.combinations(sorted(subset), 2)
    ) if len(subset) > 1 else 0.0
    return float(active - decoy_weight * decoy + singleton - redundancy)


def assignment_for_subset(
    terms: dict[str, Any], qubo: dict[str, Any], subset: tuple[str, ...]
) -> dict[str, int]:
    selected = set(subset)
    assignment = {variable: 0 for variable in qubo["variables"]}
    for receptor_id in selected:
        assignment[f"x__{receptor_id}"] = 1
    for ligand_id, incidence in terms["active_incidence"].items():
        if not incidence:
            continue
        count = sum(receptor_id in selected for receptor_id in incidence)
        y = int(count > 0)
        assignment[f"y__{ligand_id}"] = y
        slack = binary_assignment(slack_weights(len(incidence)), count - y)
        for index, bit in slack.items():
            assignment[f"s__{ligand_id}__{index}"] = bit
    for ligand_id, incidence in terms["decoy_incidence"].items():
        if incidence:
            assignment[f"z__{ligand_id}"] = int(
                any(receptor_id in selected for receptor_id in incidence)
            )
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


def select_subset(
    terms: dict[str, Any],
    receptor_ids: list[str],
    target_size: int,
    params: dict[str, float],
    cardinality_penalty: float,
    constraint_penalty: float,
    certify_all: bool = False,
) -> tuple[tuple[str, ...], dict[str, Any], list[float]]:
    qubo = build_auxiliary_qubo(
        terms,
        receptor_ids,
        target_size,
        float(params["decoy_weight"]),
        float(params["singleton_weight"]),
        float(params["redundancy_weight"]),
        cardinality_penalty,
        constraint_penalty,
    )
    candidates: list[tuple[tuple[str, ...], float]] = []
    for subset in itertools.combinations(receptor_ids, target_size):
        subset = tuple(sorted(subset))
        objective = reduced_objective(
            terms,
            receptor_ids,
            subset,
            float(params["decoy_weight"]),
            float(params["singleton_weight"]),
            float(params["redundancy_weight"]),
        )
        candidates.append((subset, objective))
    selected, selected_objective = min(
        candidates, key=lambda item: (-item[1], item[0])
    )
    assignment = assignment_for_subset(terms, qubo, selected)
    selected_energy = qubo_energy(qubo, assignment)
    baseline = selected_energy + selected_objective
    residuals = (
        [
            qubo_energy(qubo, assignment_for_subset(terms, qubo, subset))
            + objective
            - baseline
            for subset, objective in candidates
        ]
        if certify_all
        else [0.0]
    )
    if max(abs(value) for value in residuals) > 1e-7:
        raise ValueError("auxiliary QUBO does not reduce to the intended objective")
    return selected, {
        "qubo": qubo,
        "assignment": assignment,
        "selected_objective": selected_objective,
        "selected_energy": selected_energy,
        "equivalence_constant": baseline,
        "max_reduced_energy_residual": max(abs(value) for value in residuals),
        "equivalence_states_evaluated": len(candidates) if certify_all else 1,
        "active_covered_count": sum(
            assignment.get(f"y__{ligand_id}", 0)
            for ligand_id in terms["active_incidence"]
        ),
        "decoy_exposed_count": sum(
            assignment.get(f"z__{ligand_id}", 0)
            for ligand_id in terms["decoy_incidence"]
        ),
        "variable_count": len(qubo["variables"]),
        "active_y_count": len(qubo["variable_groups"]["active_y"]),
        "decoy_z_count": len(qubo["variable_groups"]["decoy_z"]),
        "slack_count": len(qubo["variable_groups"]["active_slack"]),
        "feasible": True,
    }, [objective for _, objective in candidates]


def objective_key(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    return (
        -statistics.fmean(float(row["validation_robust_composite"]) for row in rows),
        -min(float(row["validation_robust_composite"]) for row in rows),
        -statistics.fmean(float(row["validation_rank_spearman"]) for row in rows),
        float(rows[0]["coverage_fraction"]),
        float(rows[0]["decoy_weight"]),
        float(rows[0]["singleton_weight"]),
        float(rows[0]["redundancy_weight"]),
    )


def paired_comparison(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    indexed = {
        (row["target_id"], int(row["outer_fold"]), row["method"]): row
        for row in rows
    }
    all_deltas: list[float] = []
    per_target: dict[str, Any] = {}
    for target_id in sorted({row["target_id"] for row in rows}):
        folds = sorted(
            int(row["outer_fold"])
            for row in rows
            if row["target_id"] == target_id and row["method"] == left
        )
        deltas = [
            float(indexed[(target_id, fold, left)]["holdout_robust_composite"])
            - float(indexed[(target_id, fold, right)]["holdout_robust_composite"])
            for fold in folds
        ]
        all_deltas.extend(deltas)
        per_target[target_id] = {
            "fold_count": len(deltas),
            "mean_delta": statistics.fmean(deltas),
            "positive_fold_count": sum(value > 0.0 for value in deltas),
            "fold_deltas": deltas,
        }
    return {
        "direction": f"{left} minus {right}",
        "fold_count": len(all_deltas),
        "mean_delta": statistics.fmean(all_deltas),
        "positive_fold_count": sum(value > 0.0 for value in all_deltas),
        "per_target": per_target,
    }


def aggregate_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_id"]), str(row["method"]))].append(row)
    output = []
    for (target_id, method), selected in sorted(grouped.items()):
        subsets = [tuple(row["selected_subset"].split("+")) for row in selected]
        output.append(
            {
                "target_id": target_id,
                "method": method,
                "fold_count": len(selected),
                "mean_holdout_robust_composite": statistics.fmean(
                    float(row["holdout_robust_composite"]) for row in selected
                ),
                "worst_holdout_robust_composite": min(
                    float(row["holdout_robust_composite"]) for row in selected
                ),
                "mean_holdout_primary": statistics.fmean(
                    float(row["holdout_primary"]) for row in selected
                ),
                "mean_holdout_worst_seed": statistics.fmean(
                    float(row["holdout_worst_seed"]) for row in selected
                ),
                "mean_holdout_rank": statistics.fmean(
                    int(row["holdout_rank"]) for row in selected
                ),
                "selected_subsets": ["+".join(value) for value in subsets],
            }
        )
    return output


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 19h implementation path differs")
    input_paths = {
        key: verified(root, descriptor)
        for key, descriptor in config["inputs"].items()
    }
    stage19e_config = read_json(input_paths["stage19e_config"])
    stage19e_result = read_json(input_paths["stage19e_result"])
    stage19e_audit = read_json(input_paths["stage19e_audit"])
    stage19f_result = read_json(input_paths["stage19f_result"])
    stage19g_result = read_json(input_paths["stage19g_result"])
    stage19g_audit = read_json(input_paths["stage19g_audit"])
    if stage19e_result["status"] != "stage19e_quadratic_v2_not_supported_do_not_amend_bace1":
        raise ValueError("Stage 19e status differs")
    if stage19e_audit["status"] != "stage19e_cross_target_qubo_v2_nested_diagnostic_audit_ok":
        raise ValueError("Stage 19e audit differs")
    if stage19f_result["status"] != "stage19f_stable_pair_qubo_not_supported_do_not_amend_bace1":
        raise ValueError("Stage 19f status differs")
    if stage19g_result["status"] != "stage19g_cross_target_set_function_landscape_complete":
        raise ValueError("Stage 19g status differs")
    if stage19g_result["decision"]["cross_target_route"] != "no_cross_target_efficacy_qubo_route_authorized":
        raise ValueError("Stage 19g route differs")
    if stage19g_audit["status"] != "stage19g_cross_target_set_function_landscape_audit_ok":
        raise ValueError("Stage 19g audit differs")

    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage 19h outputs exist; pass --overwrite")
    diagnostic = config["diagnostic"]
    target_size = int(diagnostic["target_size"])
    outer_count = int(diagnostic["outer_fold_count"])
    inner_count = int(diagnostic["inner_fold_count"])
    fold_seed = int(diagnostic["fold_seed"])
    inner_seed = int(diagnostic["inner_fold_seed"])
    bedroc_alpha = float(diagnostic["bedroc_alpha"])
    cardinality_penalty = float(diagnostic["cardinality_penalty"])
    constraint_penalty = float(diagnostic["constraint_penalty"])
    candidate_params = [
        {
            "coverage_fraction": float(coverage),
            "decoy_weight": float(decoy),
            "singleton_weight": float(singleton),
            "redundancy_weight": float(redundancy),
        }
        for coverage, decoy, singleton, redundancy in itertools.product(
            diagnostic["coverage_fractions"],
            diagnostic["decoy_weights"],
            diagnostic["singleton_weights"],
            diagnostic["redundancy_weights"],
        )
    ]

    baseline_source = read_csv(input_paths["stage19f_comparison_methods"])
    baseline_rows = [
        row for row in baseline_source if row["method"] in BASELINE_METHODS
    ]
    inner_rows: list[dict[str, Any]] = []
    outer_candidate_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    full_models: dict[str, Any] = {}
    input_dimensions: dict[str, Any] = {}

    for target_id, target_spec in stage19e_config["targets"].items():
        print(f"loading_target={target_id}", flush=True)
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        all_ids = {row["ligand_id"] for row in ligands}
        assignments = make_frozen_group_folds(ligands, outer_count, fold_seed)
        triples = list(itertools.combinations(receptor_ids, target_size))
        triple_index = {tuple(sorted(value)): index for index, value in enumerate(triples)}
        model_spec = {
            "coverage_fraction": float(target_spec["v1_qubo"]["coverage_fraction"]),
            "utility_metric": "bedroc",
        }
        input_dimensions[target_id] = {
            "ligand_count": len(ligands),
            "receptor_count": len(receptor_ids),
            "active_count": sum(row["label"] == "active" for row in ligands),
            "candidate_count": len(candidate_params),
        }
        target_outer_rows: list[dict[str, Any]] = []

        for outer_fold in range(outer_count):
            print(f"target={target_id} outer_fold={outer_fold}", flush=True)
            holdout_ids = {
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == outer_fold
            }
            train_ids = all_ids - holdout_ids
            outer_context = make_context(
                train_ids, holdout_ids, matrices, receptor_ids, model_spec
            )
            outer_train_values = score_subsets(
                outer_context, triples, receptor_ids, "train", bedroc_alpha
            )
            outer_holdout_values = score_subsets(
                outer_context, triples, receptor_ids, "validation", bedroc_alpha
            )
            outer_train_rows = [row for row in ligands if row["ligand_id"] in train_ids]
            inner_assignments = make_frozen_group_folds(
                outer_train_rows, inner_count, inner_seed + outer_fold
            )
            fold_inner_rows: list[dict[str, Any]] = []
            for inner_fold in range(inner_count):
                inner_validation_ids = {
                    ligand_id
                    for ligand_id, fold in inner_assignments.items()
                    if fold == inner_fold
                }
                inner_train_ids = train_ids - inner_validation_ids
                inner_context = make_context(
                    inner_train_ids,
                    inner_validation_ids,
                    matrices,
                    receptor_ids,
                    model_spec,
                )
                inner_train_values = score_subsets(
                    inner_context, triples, receptor_ids, "train", bedroc_alpha
                )
                inner_validation_values = score_subsets(
                    inner_context, triples, receptor_ids, "validation", bedroc_alpha
                )
                singleton_values = {
                    receptor_id: float(
                        score_subsets(
                            inner_context,
                            [(receptor_id,)],
                            receptor_ids,
                            "train",
                            bedroc_alpha,
                        )["robust_composite"][0]
                    )
                    for receptor_id in receptor_ids
                }
                terms_by_fraction = {
                    float(fraction): build_coverage_terms(
                        inner_context,
                        receptor_ids,
                        float(fraction),
                        bedroc_alpha,
                        singleton_values,
                    )
                    for fraction in diagnostic["coverage_fractions"]
                }
                for params in candidate_params:
                    terms = terms_by_fraction[params["coverage_fraction"]]
                    subset, details, objective_values = select_subset(
                        terms,
                        receptor_ids,
                        target_size,
                        params,
                        cardinality_penalty,
                        constraint_penalty,
                    )
                    index = triple_index[tuple(sorted(subset))]
                    validation_objective = np.asarray(objective_values, dtype=float)
                    row = {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        **params,
                        "selected_subset": "+".join(sorted(subset)),
                        "validation_rank_spearman": safe_spearman(
                            validation_objective,
                            inner_validation_values["robust_composite"],
                        ),
                        "variable_count": details["variable_count"],
                        "active_y_count": details["active_y_count"],
                        "decoy_z_count": details["decoy_z_count"],
                        "slack_count": details["slack_count"],
                        "active_covered_count": details["active_covered_count"],
                        "decoy_exposed_count": details["decoy_exposed_count"],
                        "qubo_feasible": details["feasible"],
                        **{
                            f"validation_{key}": value
                            for key, value in subset_metrics(
                                inner_validation_values, index
                            ).items()
                        },
                    }
                    inner_rows.append(row)
                    fold_inner_rows.append(row)
            grouped: dict[tuple[float, float, float, float], list[dict[str, Any]]] = defaultdict(list)
            for row in fold_inner_rows:
                grouped[
                    (
                        float(row["coverage_fraction"]),
                        float(row["decoy_weight"]),
                        float(row["singleton_weight"]),
                        float(row["redundancy_weight"]),
                    )
                ].append(row)
            selected_key = min(
                grouped,
                key=lambda key: objective_key(grouped[key]),
            )

            singleton_values = {
                receptor_id: float(
                    score_subsets(
                        outer_context,
                        [(receptor_id,)],
                        receptor_ids,
                        "train",
                        bedroc_alpha,
                    )["robust_composite"][0]
                )
                for receptor_id in receptor_ids
            }
            terms_by_fraction = {
                float(fraction): build_coverage_terms(
                    outer_context,
                    receptor_ids,
                    float(fraction),
                    bedroc_alpha,
                    singleton_values,
                )
                for fraction in diagnostic["coverage_fractions"]
            }
            for params in candidate_params:
                terms = terms_by_fraction[params["coverage_fraction"]]
                selected = (
                    (
                        params["coverage_fraction"],
                        params["decoy_weight"],
                        params["singleton_weight"],
                        params["redundancy_weight"],
                    )
                    == selected_key
                )
                subset, details, objective_values = select_subset(
                    terms,
                    receptor_ids,
                    target_size,
                    params,
                    cardinality_penalty,
                    constraint_penalty,
                    certify_all=selected,
                )
                index = triple_index[tuple(sorted(subset))]
                row = {
                    "target_id": target_id,
                    "outer_fold": outer_fold,
                    **params,
                    "selected_by_inner_cv": selected,
                    "selected_subset": "+".join(sorted(subset)),
                    "validation_rank_spearman": safe_spearman(
                        np.asarray(objective_values, dtype=float),
                        outer_holdout_values["robust_composite"],
                    ),
                    "variable_count": details["variable_count"],
                    "active_y_count": details["active_y_count"],
                    "decoy_z_count": details["decoy_z_count"],
                    "slack_count": details["slack_count"],
                    "active_covered_count": details["active_covered_count"],
                    "decoy_exposed_count": details["decoy_exposed_count"],
                    "qubo_feasible": details["feasible"],
                    **{
                        f"validation_{key}": value
                        for key, value in subset_metrics(
                            outer_holdout_values, index
                        ).items()
                    },
                }
                outer_candidate_rows.append(row)
                if selected:
                    method_rows.append(
                        {
                            "target_id": target_id,
                            "outer_fold": outer_fold,
                            "method": CANDIDATE_METHOD,
                            "selected_subset": "+".join(sorted(subset)),
                            "holdout_rank": descending_rank(
                                outer_holdout_values["robust_composite"], index
                            ),
                            **{
                                f"train_{key}": value
                                for key, value in subset_metrics(
                                    outer_train_values, index
                                ).items()
                            },
                            **{
                                f"holdout_{key}": value
                                for key, value in subset_metrics(
                                    outer_holdout_values, index
                                ).items()
                            },
                            **params,
                            "variable_count": details["variable_count"],
                            "active_y_count": details["active_y_count"],
                            "decoy_z_count": details["decoy_z_count"],
                            "slack_count": details["slack_count"],
                            "active_covered_count": details["active_covered_count"],
                            "decoy_exposed_count": details["decoy_exposed_count"],
                            "qubo_energy": details["selected_energy"],
                            "qubo_objective": details["selected_objective"],
                            "qubo_equivalence_residual": details[
                                "max_reduced_energy_residual"
                            ],
                            "equivalence_states_evaluated": details[
                                "equivalence_states_evaluated"
                            ],
                            "qubo_feasible": details["feasible"],
                        }
                    )
            if sum(
                row["target_id"] == target_id
                and int(row["outer_fold"]) == outer_fold
                and row["method"] == CANDIDATE_METHOD
                for row in method_rows
            ) != 1:
                raise ValueError("nested coverage candidate selection is not unique")

        target_outer = [
            row for row in outer_candidate_rows if row["target_id"] == target_id
        ]
        grouped_outer: dict[tuple[float, float, float, float], list[dict[str, Any]]] = defaultdict(list)
        for row in target_outer:
            grouped_outer[
                (
                    float(row["coverage_fraction"]),
                    float(row["decoy_weight"]),
                    float(row["singleton_weight"]),
                    float(row["redundancy_weight"]),
                )
            ].append(row)
        full_key = min(grouped_outer, key=lambda key: objective_key(grouped_outer[key]))
        full_context = make_context(all_ids, set(), matrices, receptor_ids, model_spec)
        full_singletons = {
            receptor_id: float(
                score_subsets(
                    full_context,
                    [(receptor_id,)],
                    receptor_ids,
                    "train",
                    bedroc_alpha,
                )["robust_composite"][0]
            )
            for receptor_id in receptor_ids
        }
        full_params = {
            "coverage_fraction": full_key[0],
            "decoy_weight": full_key[1],
            "singleton_weight": full_key[2],
            "redundancy_weight": full_key[3],
        }
        full_terms = build_coverage_terms(
            full_context,
            receptor_ids,
            full_params["coverage_fraction"],
            bedroc_alpha,
            full_singletons,
        )
        full_subset, full_details, _ = select_subset(
            full_terms,
            receptor_ids,
            target_size,
            full_params,
            cardinality_penalty,
            constraint_penalty,
            certify_all=True,
        )
        full_values = score_subsets(
            full_context, triples, receptor_ids, "train", bedroc_alpha
        )
        full_index = triple_index[tuple(sorted(full_subset))]
        full_models[target_id] = {
            "selected_parameters": full_params,
            "selected_subset": list(sorted(full_subset)),
            "full_train_metrics": subset_metrics(full_values, full_index),
            "full_train_rank": descending_rank(
                full_values["robust_composite"], full_index
            ),
            "terms": full_terms,
            "qubo": full_details["qubo"],
            "selected_assignment": full_details["assignment"],
            "selected_energy": full_details["selected_energy"],
            "selected_objective": full_details["selected_objective"],
            "equivalence_residual": full_details["max_reduced_energy_residual"],
            "equivalence_states_evaluated": full_details[
                "equivalence_states_evaluated"
            ],
            "variable_count": full_details["variable_count"],
            "active_y_count": full_details["active_y_count"],
            "decoy_z_count": full_details["decoy_z_count"],
            "slack_count": full_details["slack_count"],
        }

    comparison_rows: list[dict[str, Any]] = [*method_rows]
    for row in baseline_rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                converted[key] = None
            elif key in {"outer_fold", "holdout_rank"}:
                converted[key] = int(value)
            elif key.startswith(("train_", "holdout_")):
                converted[key] = float(value)
            else:
                converted[key] = value
        comparison_rows.append(converted)
    comparisons = {
        f"auxiliary_coverage_vs_{method}": paired_comparison(
            comparison_rows, CANDIDATE_METHOD, method
        )
        for method in BASELINE_METHODS
    }
    gate_spec = config["development_support_gate"]
    comparison_checks = {
        key: (
            all(
                float(value["mean_delta"])
                > float(gate_spec["minimum_target_mean_delta"])
                for value in comparison["per_target"].values()
            )
            and comparison["positive_fold_count"]
            >= int(gate_spec["minimum_positive_folds_of_eight"])
        )
        for key, comparison in comparisons.items()
    }
    passed = all(comparison_checks.values())
    data_boundary = {
        "train_rows_read_by_target": {
            target_id: int(spec["expected"]["ligand_count"])
            for target_id, spec in stage19e_config["targets"].items()
        },
        "new_docking_jobs": 0,
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
        "bace1_docking_rows_read": 0,
    }
    write_csv(outputs["inner_trials_csv"], inner_rows)
    write_csv(outputs["outer_candidate_trials_csv"], outer_candidate_rows)
    write_csv(outputs["comparison_methods_csv"], comparison_rows)
    write_json(
        outputs["model_record_json"],
        {
            "schema_version": "1.0",
            "status": "development_gate_failed_not_authorized_for_bace1"
            if not passed
            else "development_signal_not_authorized_for_bace1",
            "algorithm_id": "auxiliary-active-coverage-decoy-exposure-qubo-v1",
            "data_boundary": data_boundary,
            "algorithm": config["algorithm"],
            "target_development_models": full_models,
        },
    )
    full_model_summaries = {
        target_id: {
            key: value
            for key, value in model.items()
            if key not in {"terms", "qubo", "selected_assignment"}
        }
        for target_id, model in full_models.items()
    }
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "status": (
            "stage19h_auxiliary_coverage_not_supported_do_not_amend_bace1"
            if not passed
            else "stage19h_auxiliary_coverage_development_signal_do_not_amend_bace1"
        ),
        "experiment_id": config["experiment_id"],
        "experiment_class": "posthoc_cross_target_train_only_development",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "data_boundary": data_boundary,
        "input_dimensions": input_dimensions,
        "candidate_count": len(candidate_params),
        "outer_method_aggregate": aggregate_methods(comparison_rows),
        "paired_comparisons": comparisons,
        "development_gate": {
            "passed": passed,
            "bace1_method_amendment_authorized": False,
            "comparison_checks": comparison_checks,
            "rule": gate_spec,
        },
        "full_train_models": full_model_summaries,
        "outputs": {
            key: output_descriptor(root, path)
            for key, path in outputs.items()
            if key not in ("result_json", "report_md")
        },
        "next_gate": (
            "freeze_auxiliary_coverage_for_independent_target_only_after_pre_registered_review"
            if passed
            else "do_not_spend_new_docking_budget_on_this_coverage_objective; review_quantum_application_scope"
        ),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    report_lines = [
        "# Stage 19h auxiliary-variable coverage QUBO",
        "",
        "## Scope",
        "",
        "Post-hoc nested scaffold-CV development on MK14 and PPARG train matrices only. No new docking or protected-panel row was read.",
        "",
        "## Results",
        "",
        "| Target | Method | Mean holdout robust composite | Mean primary |",
        "|---|---|---:|---:|",
    ]
    for row in result["outer_method_aggregate"]:
        report_lines.append(
            f"| {row['target_id']} | {row['method']} | "
            f"{row['mean_holdout_robust_composite']:.6f} | "
            f"{row['mean_holdout_primary']:.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Development gate passed: `{passed}`",
            "- BACE1 method amendment authorized: `False`",
            f"- Next gate: `{result['next_gate']}`",
            "",
            "## Encoding",
            "",
            "- `x_i` selects a receptor.",
            "- `y_a` marks an active ligand covered by at least one selected receptor.",
            "- `z_d` marks a decoy exposed by at least one selected receptor.",
            "- Binary slack variables enforce the active-ligand OR relation as a quadratic equality penalty.",
            "- Decoy exposure constraints use quadratic implication penalties.",
            "",
            config["interpretation_boundary"],
        ]
    )
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report_lines) + "\n", encoding="ascii")
    print(
        json.dumps(
            {
                "status": result["status"],
                "development_gate": result["development_gate"],
                "outer_method_aggregate": result["outer_method_aggregate"],
                "paired_comparisons": result["paired_comparisons"],
                "full_train_models": result["full_train_models"],
                "data_boundary": result["data_boundary"],
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
