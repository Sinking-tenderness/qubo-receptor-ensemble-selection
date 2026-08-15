"""Screen frozen receptor-ensemble objectives before quantum execution."""

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
    read_json,
    rooted,
    safe_spearman,
    score_subsets,
    verified,
    write_csv,
    write_json,
)
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import make_context


BASELINE_METHODS = ("direct_greedy", "additive_top3", "exact_robust_oracle")
CANDIDATE_METHOD = "frozen_qubo_objective"
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


def pair_key(first: str, second: str) -> str:
    return "__".join(sorted((first, second)))


def slack_weights(maximum: int) -> list[int]:
    weights: list[int] = []
    total = 0
    next_weight = 1
    while total < maximum:
        value = min(next_weight, maximum - total)
        weights.append(value)
        total += value
        next_weight *= 2
    return weights


def binary_slack(weights: list[int], value: int) -> list[int]:
    for bits in itertools.product((0, 1), repeat=len(weights)):
        if sum(weight * bit for weight, bit in zip(weights, bits)) == value:
            return list(bits)
    raise ValueError("slack value is not representable")


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


def rank_fraction_rows(
    context: dict[str, Any], receptor_ids: list[str]
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    rows_by_matrix = {
        matrix_id: {
            str(row["ligand_id"]): row
            for row in context["matrices"][matrix_id]["train"]
        }
        for matrix_id in MATRIX_IDS
    }
    ligand_ids = sorted(rows_by_matrix["primary"])
    ranks = {
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
                ranks[ligand_id][receptor_id].append(rank / denominator)
    return (
        {
            ligand_id: {
                receptor_id: float(np.median(values))
                for receptor_id, values in receptor_values.items()
            }
            for ligand_id, receptor_values in ranks.items()
        },
        labels,
    )


def build_terms(
    context: dict[str, Any],
    receptor_ids: list[str],
    coverage_fraction: float,
    active_threshold: int,
    bedroc_alpha: float,
) -> dict[str, Any]:
    ranks, labels = rank_fraction_rows(context, receptor_ids)
    active_ids = sorted(key for key, value in labels.items() if value == "active")
    decoy_ids = sorted(key for key, value in labels.items() if value == "decoy")
    active_raw = {
        ligand_id: math.exp(-bedroc_alpha * min(ranks[ligand_id].values()))
        for ligand_id in active_ids
    }
    decoy_raw = {
        ligand_id: math.exp(-bedroc_alpha * min(ranks[ligand_id].values()))
        for ligand_id in decoy_ids
    }
    active_total = sum(active_raw.values())
    decoy_total = sum(decoy_raw.values())
    score_columns = {
        receptor_id: np.asarray(
            [float(row[receptor_id]) for row in context["matrices"]["primary"]["train"]]
        )
        for receptor_id in receptor_ids
    }
    correlations: dict[str, float] = {}
    for first, second in itertools.combinations(receptor_ids, 2):
        value = float(spearmanr(score_columns[first], score_columns[second]).statistic)
        correlations[pair_key(first, second)] = (
            max(0.0, value) if math.isfinite(value) else 0.0
        )
    return {
        "coverage_fraction": float(coverage_fraction),
        "active_threshold": int(active_threshold),
        "decoy_threshold": 1,
        "bedroc_alpha": float(bedroc_alpha),
        "active_ids": active_ids,
        "decoy_ids": decoy_ids,
        "active_incidence": {
            ligand_id: [
                receptor_id
                for receptor_id in receptor_ids
                if ranks[ligand_id][receptor_id] <= coverage_fraction
            ]
            for ligand_id in active_ids
        },
        "decoy_incidence": {
            ligand_id: [
                receptor_id
                for receptor_id in receptor_ids
                if ranks[ligand_id][receptor_id] <= coverage_fraction
            ]
            for ligand_id in decoy_ids
        },
        "active_weights": {
            ligand_id: value / active_total for ligand_id, value in active_raw.items()
        },
        "decoy_weights": {
            ligand_id: value / decoy_total for ligand_id, value in decoy_raw.items()
        },
        "correlations": correlations,
    }


def build_qubo(
    terms: dict[str, Any],
    receptor_ids: list[str],
    target_size: int,
    decoy_weight: float,
    redundancy_weight: float,
    cardinality_penalty: float,
    constraint_penalty: float,
) -> dict[str, Any]:
    coefficients: dict[str, Any] = {
        "constant": 0.0,
        "linear": {},
        "quadratic": {},
    }
    x_names = {receptor_id: f"x__{receptor_id}" for receptor_id in receptor_ids}
    active_y = {
        ligand_id: f"y__{ligand_id}"
        for ligand_id, incidence in terms["active_incidence"].items()
        if incidence
    }
    decoy_z = {
        ligand_id: f"z__{ligand_id}"
        for ligand_id, incidence in terms["decoy_incidence"].items()
        if incidence
    }
    slack_names: list[str] = []
    add_square(
        coefficients,
        -target_size,
        {name: 1.0 for name in x_names.values()},
        cardinality_penalty,
    )
    threshold = int(terms["active_threshold"])
    for ligand_id, incidence in terms["active_incidence"].items():
        if not incidence:
            continue
        expression = {active_y[ligand_id]: float(threshold)}
        for index, weight in enumerate(slack_weights(len(incidence))):
            name = f"s__{ligand_id}__{index}"
            slack_names.append(name)
            expression[name] = float(weight)
        for receptor_id in incidence:
            expression[x_names[receptor_id]] = -1.0
        add_square(coefficients, 0.0, expression, constraint_penalty)
        add_linear(
            coefficients,
            active_y[ligand_id],
            -float(terms["active_weights"][ligand_id]),
        )
    for ligand_id, incidence in terms["decoy_incidence"].items():
        if not incidence:
            continue
        z_name = decoy_z[ligand_id]
        add_linear(
            coefficients,
            z_name,
            decoy_weight * float(terms["decoy_weights"][ligand_id]),
        )
        for receptor_id in incidence:
            add_linear(coefficients, x_names[receptor_id], constraint_penalty)
            add_quadratic(
                coefficients, x_names[receptor_id], z_name, -constraint_penalty
            )
    pair_denominator = max(1, target_size * (target_size - 1) // 2)
    for first, second in itertools.combinations(receptor_ids, 2):
        add_quadratic(
            coefficients,
            x_names[first],
            x_names[second],
            float(redundancy_weight)
            * float(terms["correlations"].get(pair_key(first, second), 0.0))
            / pair_denominator,
        )
    groups = {
        "x": [x_names[receptor_id] for receptor_id in receptor_ids],
        "active_y": list(active_y.values()),
        "decoy_z": list(decoy_z.values()),
        "active_slack": slack_names,
    }
    return {
        "constant": float(coefficients["constant"]),
        "linear": {key: float(value) for key, value in coefficients["linear"].items()},
        "quadratic": {
            key: float(value) for key, value in coefficients["quadratic"].items()
        },
        "variables": sorted(set().union(*[set(values) for values in groups.values()])),
        "variable_groups": groups,
        "target_size": int(target_size),
        "decoy_weight": float(decoy_weight),
        "redundancy_weight": float(redundancy_weight),
        "cardinality_penalty": float(cardinality_penalty),
        "constraint_penalty": float(constraint_penalty),
        "convention": (
            "Q(b)=constant+sum_v linear[v]*b_v+"
            "sum_u<v quadratic[u::v]*b_u*b_v; minimize Q"
        ),
    }


def reduced_objective(
    terms: dict[str, Any],
    subset: tuple[str, ...],
    decoy_weight: float,
    redundancy_weight: float,
) -> float:
    selected = set(subset)
    active = sum(
        float(terms["active_weights"][ligand_id])
        for ligand_id, incidence in terms["active_incidence"].items()
        if sum(receptor_id in selected for receptor_id in incidence)
        >= int(terms["active_threshold"])
    )
    decoy = sum(
        float(terms["decoy_weights"][ligand_id])
        for ligand_id, incidence in terms["decoy_incidence"].items()
        if selected.intersection(incidence)
    )
    redundancy = redundancy_weight * statistics.fmean(
        float(terms["correlations"].get(pair_key(first, second), 0.0))
        for first, second in itertools.combinations(sorted(subset), 2)
    )
    return float(active - decoy_weight * decoy - redundancy)


def assignment(
    terms: dict[str, Any], qubo: dict[str, Any], subset: tuple[str, ...]
) -> dict[str, int]:
    selected = set(subset)
    values = {variable: 0 for variable in qubo["variables"]}
    for receptor_id in selected:
        values[f"x__{receptor_id}"] = 1
    threshold = int(terms["active_threshold"])
    for ligand_id, incidence in terms["active_incidence"].items():
        if not incidence:
            continue
        count = sum(receptor_id in selected for receptor_id in incidence)
        y_value = int(count >= threshold)
        values[f"y__{ligand_id}"] = y_value
        for index, bit in enumerate(
            binary_slack(slack_weights(len(incidence)), count - threshold * y_value)
        ):
            values[f"s__{ligand_id}__{index}"] = bit
    for ligand_id, incidence in terms["decoy_incidence"].items():
        if incidence:
            values[f"z__{ligand_id}"] = int(
                any(receptor_id in selected for receptor_id in incidence)
            )
    return values


def validate_assignment(
    terms: dict[str, Any], values: dict[str, int], target_size: int
) -> None:
    if any(value not in (0, 1) for value in values.values()):
        raise ValueError("nonbinary auxiliary assignment")
    if sum(value for key, value in values.items() if key.startswith("x__")) != target_size:
        raise ValueError("cardinality constraint is not satisfied")
    threshold = int(terms["active_threshold"])
    for ligand_id, incidence in terms["active_incidence"].items():
        if not incidence:
            continue
        count = sum(values[f"x__{receptor_id}"] for receptor_id in incidence)
        y_value = values[f"y__{ligand_id}"]
        slack = sum(
            weight * values[f"s__{ligand_id}__{index}"]
            for index, weight in enumerate(slack_weights(len(incidence)))
        )
        if threshold * y_value + slack != count or y_value != int(count >= threshold):
            raise ValueError(f"active threshold constraint differs: {ligand_id}")
    for ligand_id, incidence in terms["decoy_incidence"].items():
        if not incidence:
            continue
        expected = int(any(values[f"x__{receptor_id}"] for receptor_id in incidence))
        if values[f"z__{ligand_id}"] != expected:
            raise ValueError(f"decoy implication differs: {ligand_id}")


def qubo_energy(qubo: dict[str, Any], values: dict[str, int]) -> float:
    energy = float(qubo["constant"])
    energy += sum(
        float(coefficient) * values.get(variable, 0)
        for variable, coefficient in qubo["linear"].items()
    )
    energy += sum(
        float(coefficient) * values.get(first, 0) * values.get(second, 0)
        for key, coefficient in qubo["quadratic"].items()
        for first, second in [key.split("::", 1)]
    )
    return float(energy)


def certify_states(
    terms: dict[str, Any],
    qubo: dict[str, Any],
    receptor_ids: list[str],
    target_size: int,
    decoy_weight: float,
    redundancy_weight: float,
) -> dict[str, Any]:
    subsets = [tuple(sorted(value)) for value in itertools.combinations(receptor_ids, target_size)]
    objectives: list[float] = []
    energies: list[float] = []
    assignments: list[dict[str, int]] = []
    for subset in subsets:
        values = assignment(terms, qubo, subset)
        validate_assignment(terms, values, target_size)
        objective = reduced_objective(terms, subset, decoy_weight, redundancy_weight)
        objectives.append(objective)
        energies.append(qubo_energy(qubo, values))
        assignments.append(values)
    objective_array = np.asarray(objectives, dtype=float)
    energy_array = np.asarray(energies, dtype=float)
    baseline = energy_array + objective_array
    residual = float(np.max(np.abs(baseline - baseline[0])))
    objective_best = min(
        range(len(subsets)), key=lambda index: (-objectives[index], subsets[index])
    )
    energy_best = min(
        range(len(subsets)), key=lambda index: (energies[index], subsets[index])
    )
    if objective_best != energy_best or residual > 1e-7:
        raise ValueError("QUBO does not reduce to the intended threshold objective")
    order = np.argsort(energy_array, kind="stable")
    maximum = max(
        [abs(value) for value in qubo["linear"].values()]
        + [abs(value) for value in qubo["quadratic"].values()]
        + [0.0]
    )
    nonzero = [
        abs(value)
        for value in list(qubo["linear"].values()) + list(qubo["quadratic"].values())
        if abs(value) > 0.0
    ]
    raw_gap = float(energy_array[order[1]] - energy_array[order[0]])
    return {
        "subsets": subsets,
        "objectives": objective_array,
        "energies": energy_array,
        "assignments": assignments,
        "selected_index": int(objective_best),
        "selected_subset": list(subsets[objective_best]),
        "selected_objective": float(objectives[objective_best]),
        "selected_energy": float(energies[objective_best]),
        "state_count": len(subsets),
        "equivalence_residual": residual,
        "max_abs_coefficient": maximum,
        "min_nonzero_abs_coefficient": min(nonzero, default=0.0),
        "coefficient_dynamic_range": (
            maximum / min(nonzero) if nonzero else 0.0
        ),
        "raw_best_second_gap": raw_gap,
        "scaled_best_second_gap": raw_gap / maximum if maximum else 0.0,
    }


def state_features(
    qubo: dict[str, Any], assignments: list[dict[str, int]]
) -> tuple[np.ndarray, np.ndarray]:
    linear_keys = sorted(qubo["linear"])
    quadratic_keys = sorted(qubo["quadratic"])
    variable_index = {
        variable: index for index, variable in enumerate(qubo["variables"])
    }
    assignment_matrix = np.asarray(
        [
            [assignment.get(variable, 0) for variable in qubo["variables"]]
            for assignment in assignments
        ],
        dtype=float,
    )
    parts: list[np.ndarray] = []
    coefficients: list[float] = []
    for variable in linear_keys:
        parts.append(assignment_matrix[:, variable_index[variable]])
        coefficients.append(float(qubo["linear"][variable]))
    for key in quadratic_keys:
        first, second = key.split("::", 1)
        parts.append(
            assignment_matrix[:, variable_index[first]]
            * assignment_matrix[:, variable_index[second]]
        )
        coefficients.append(float(qubo["quadratic"][key]))
    return np.column_stack(parts), np.asarray(coefficients, dtype=float)


def noise_screen(
    target_id: str,
    candidate_id: str,
    qubo: dict[str, Any],
    certificate: dict[str, Any],
    noise_levels: list[float],
    noise_models: list[dict[str, Any]],
    repeats: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features, coefficients = state_features(qubo, certificate["assignments"])
    maximum = float(certificate["max_abs_coefficient"])
    if maximum <= 0.0:
        raise ValueError("cannot scale a zero QUBO")
    scaled = coefficients / maximum
    base_energies = features @ scaled
    if int(np.argmin(base_energies)) != int(certificate["selected_index"]):
        raise ValueError("feature energy differs from certificate")
    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    best_objective = float(np.max(certificate["objectives"]))
    for model_index, noise_model in enumerate(noise_models):
        noise_model_id = str(noise_model["noise_model_id"])
        mode = str(noise_model["mode"])
        if mode not in {"absolute", "relative"}:
            raise ValueError(f"unknown noise model: {mode}")
        rng = np.random.default_rng(seed + model_index * 1000003)
        for noise_level in noise_levels:
            matches = 0
            regrets: list[float] = []
            ranks: list[int] = []
            for repeat in range(repeats):
                raw_noise = rng.normal(0.0, noise_level, size=len(scaled))
                perturbation = (
                    raw_noise
                    if mode == "absolute"
                    else raw_noise * np.abs(scaled)
                )
                noisy_energies = features @ (scaled + perturbation)
                selected_index = int(np.argmin(noisy_energies))
                selected_objective = float(certificate["objectives"][selected_index])
                regret = best_objective - selected_objective
                match = selected_index == int(certificate["selected_index"])
                matches += int(match)
                regrets.append(regret)
                ranks.append(
                    1
                    + int(
                        np.sum(
                            certificate["objectives"]
                            > certificate["objectives"][selected_index]
                        )
                    )
                )
                trial_rows.append(
                    {
                        "target_id": target_id,
                        "candidate_id": candidate_id,
                        "noise_model_id": noise_model_id,
                        "noise_level": noise_level,
                        "repeat": repeat,
                        "selected_subset": "+".join(
                            certificate["subsets"][selected_index]
                        ),
                        "matches_baseline": match,
                        "original_objective_regret": regret,
                        "original_objective_rank": ranks[-1],
                    }
                )
            summary_rows.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "noise_model_id": noise_model_id,
                    "noise_level": noise_level,
                    "repeat_count": repeats,
                    "selection_stability": matches / repeats,
                    "mean_original_objective_regret": statistics.fmean(regrets),
                    "worst_original_objective_regret": max(regrets),
                    "mean_original_objective_rank": statistics.fmean(ranks),
                }
            )
    return trial_rows, summary_rows


def all_subsets(receptor_ids: list[str], maximum_size: int) -> list[tuple[str, ...]]:
    return [
        tuple(sorted(value))
        for size in range(1, maximum_size + 1)
        for value in itertools.combinations(receptor_ids, size)
    ]


def score_mixed_subsets(
    context: dict[str, Any],
    subsets: list[tuple[str, ...]],
    receptor_ids: list[str],
    split: str,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Score mixed cardinalities while preserving the supplied subset order."""
    grouped: dict[int, list[tuple[int, tuple[str, ...]]]] = defaultdict(list)
    for index, subset in enumerate(subsets):
        grouped[len(subset)].append((index, subset))
    output = {
        key: np.empty(len(subsets), dtype=float)
        for key in (*MATRIX_IDS, "mean_seed", "worst_seed", "robust_composite")
    }
    for indexed in grouped.values():
        positions = [index for index, _ in indexed]
        values = score_subsets(
            context,
            [subset for _, subset in indexed],
            receptor_ids,
            split,
            alpha,
        )
        for key, array in values.items():
            output[key][positions] = array
    return output


def choose_highest(
    values: dict[tuple[str, ...], float], subsets: list[tuple[str, ...]]
) -> tuple[str, ...]:
    return min(subsets, key=lambda subset: (-values[subset], subset))


def greedy_selection(
    values: dict[tuple[str, ...], float], receptor_ids: list[str], target_size: int
) -> tuple[str, ...]:
    current = choose_highest(values, [(value,) for value in receptor_ids])
    while len(current) < target_size:
        candidates = [
            tuple(sorted((*current, receptor_id)))
            for receptor_id in receptor_ids
            if receptor_id not in current
        ]
        current = choose_highest(values, candidates)
    return current


def metrics_for_subset(
    values: dict[str, np.ndarray], subset_index: dict[tuple[str, ...], int], subset: tuple[str, ...]
) -> dict[str, float]:
    index = subset_index[tuple(sorted(subset))]
    return {key: float(values[key][index]) for key in METRIC_IDS}


def method_row(
    target_id: str,
    outer_fold: int,
    method: str,
    subset: tuple[str, ...],
    subset_index: dict[tuple[str, ...], int],
    train_values: dict[str, np.ndarray],
    holdout_values: dict[str, np.ndarray],
    **extra: Any,
) -> dict[str, Any]:
    canonical = tuple(sorted(subset))
    index = subset_index[canonical]
    return {
        "target_id": target_id,
        "outer_fold": outer_fold,
        "method": method,
        "candidate_id": extra.pop("candidate_id", ""),
        "selected_subset": "+".join(canonical),
        "holdout_rank": descending_rank(holdout_values["robust_composite"], index),
        **{
            f"train_{key}": value
            for key, value in metrics_for_subset(train_values, subset_index, canonical).items()
        },
        **{
            f"holdout_{key}": value
            for key, value in metrics_for_subset(holdout_values, subset_index, canonical).items()
        },
        **extra,
    }


def paired_comparison(
    rows: list[dict[str, Any]], candidate_id: str, baseline_method: str
) -> dict[str, Any]:
    indexed = {
        (row["target_id"], int(row["outer_fold"]), row["method"], row["candidate_id"]): row
        for row in rows
    }
    all_deltas: list[float] = []
    per_target: dict[str, Any] = {}
    for target_id in sorted({row["target_id"] for row in rows}):
        folds = range(4)
        deltas = []
        for fold in folds:
            candidate = indexed[(target_id, fold, CANDIDATE_METHOD, candidate_id)]
            baseline = indexed[(target_id, fold, baseline_method, "")]
            deltas.append(
                float(candidate["holdout_robust_composite"])
                - float(baseline["holdout_robust_composite"])
            )
        all_deltas.extend(deltas)
        per_target[target_id] = {
            "fold_count": len(deltas),
            "mean_delta": statistics.fmean(deltas),
            "positive_fold_count": sum(value > 0.0 for value in deltas),
            "fold_deltas": deltas,
        }
    return {
        "direction": f"{candidate_id} minus {baseline_method}",
        "fold_count": len(all_deltas),
        "mean_delta": statistics.fmean(all_deltas),
        "positive_fold_count": sum(value > 0.0 for value in all_deltas),
        "per_target": per_target,
    }


def aggregate_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["target_id"], row["method"], row["candidate_id"])].append(row)
    output: list[dict[str, Any]] = []
    for (target_id, method, candidate_id), selected in sorted(grouped.items()):
        output.append(
            {
                "target_id": target_id,
                "method": method,
                "candidate_id": candidate_id,
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
                "mean_holdout_rank": statistics.fmean(
                    int(row["holdout_rank"]) for row in selected
                ),
            }
        )
    return output


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 19i implementation path differs")
    input_paths = {
        key: verified(root, descriptor)
        for key, descriptor in config["inputs"].items()
    }
    stage19e_config = read_json(input_paths["stage19e_config"])
    stage19e_result = read_json(input_paths["stage19e_result"])
    stage19e_audit = read_json(input_paths["stage19e_audit"])
    stage19h_result = read_json(input_paths["stage19h_result"])
    stage19h_audit = read_json(input_paths["stage19h_audit"])
    if stage19e_result.get("status") != "stage19e_quadratic_v2_not_supported_do_not_amend_bace1":
        raise ValueError("Stage 19e status differs")
    if stage19e_audit.get("status") != "stage19e_cross_target_qubo_v2_nested_diagnostic_audit_ok":
        raise ValueError("Stage 19e audit differs")
    if stage19h_result.get("status") != "stage19h_auxiliary_coverage_not_supported_do_not_amend_bace1":
        raise ValueError("Stage 19h status differs")
    if stage19h_audit.get("status") != "stage19h_auxiliary_coverage_qubo_audit_ok":
        raise ValueError("Stage 19h audit differs")

    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage 19i outputs exist; pass --overwrite")
    diagnostic = config["diagnostic"]
    target_size = int(diagnostic["target_size"])
    outer_count = int(diagnostic["outer_fold_count"])
    fold_seed = int(diagnostic["fold_seed"])
    alpha = float(diagnostic["bedroc_alpha"])
    coverage_fraction = float(diagnostic["coverage_fraction"])
    decoy_weight = float(diagnostic["decoy_weight"])
    cardinality_penalty = float(diagnostic["cardinality_penalty"])
    constraint_penalty = float(diagnostic["constraint_penalty"])
    candidates = list(diagnostic["candidates"])
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs are not unique")
    if any(int(candidate["decoy_threshold"]) != 1 for candidate in candidates):
        raise ValueError("Stage 19i currently requires decoy threshold one")

    fold_rows: list[dict[str, Any]] = []
    noise_trials: list[dict[str, Any]] = []
    noise_summaries: list[dict[str, Any]] = []
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
            "triple_count": math.comb(len(receptor_ids), target_size),
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
        triples = [
            tuple(sorted(value)) for value in itertools.combinations(receptor_ids, target_size)
        ]
        full_single_values = score_subsets(
            full_context,
            [(receptor_id,) for receptor_id in receptor_ids],
            receptor_ids,
            "train",
            alpha,
        )
        full_models[target_id] = {}
        for candidate_index, candidate in enumerate(candidates):
            candidate_id = str(candidate["candidate_id"])
            full_terms = build_terms(
                full_context,
                receptor_ids,
                coverage_fraction,
                int(candidate["active_threshold"]),
                alpha,
            )
            full_qubo = build_qubo(
                full_terms,
                receptor_ids,
                target_size,
                decoy_weight,
                float(candidate["redundancy_weight"]),
                cardinality_penalty,
                constraint_penalty,
            )
            full_certificate = certify_states(
                full_terms,
                full_qubo,
                receptor_ids,
                target_size,
                decoy_weight,
                float(candidate["redundancy_weight"]),
            )
            trial_rows, summary_rows = noise_screen(
                target_id,
                candidate_id,
                full_qubo,
                full_certificate,
                [float(value) for value in diagnostic["noise_levels"]],
                list(diagnostic["noise_models"]),
                int(diagnostic["noise_repeats"]),
                int(diagnostic["noise_seed"]) + candidate_index * 1009,
            )
            noise_trials.extend(trial_rows)
            noise_summaries.extend(summary_rows)
            full_models[target_id][candidate_id] = {
                "candidate": candidate,
                "selected_subset": full_certificate["selected_subset"],
                "selected_objective": full_certificate["selected_objective"],
                "selected_energy": full_certificate["selected_energy"],
                "state_count": full_certificate["state_count"],
                "equivalence_residual": full_certificate["equivalence_residual"],
                "variable_count": len(full_qubo["variables"]),
                "linear_count": len(full_qubo["linear"]),
                "quadratic_count": len(full_qubo["quadratic"]),
                "max_abs_coefficient": full_certificate["max_abs_coefficient"],
                "min_nonzero_abs_coefficient": full_certificate[
                    "min_nonzero_abs_coefficient"
                ],
                "coefficient_dynamic_range": full_certificate["coefficient_dynamic_range"],
                "raw_best_second_gap": full_certificate["raw_best_second_gap"],
                "scaled_best_second_gap": full_certificate["scaled_best_second_gap"],
                "terms": full_terms,
                "qubo": full_qubo,
                "selected_assignment": full_certificate["assignments"][
                    full_certificate["selected_index"]
                ],
                "noise_summary": [
                    row
                    for row in summary_rows
                    if row["noise_level"]
                    == float(diagnostic["hardware_reference_noise_level"])
                ],
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
            subsets = all_subsets(receptor_ids, target_size)
            values_train = score_mixed_subsets(
                context, subsets, receptor_ids, "train", alpha
            )
            values_holdout = score_mixed_subsets(
                context, subsets, receptor_ids, "validation", alpha
            )
            subset_index = {subset: index for index, subset in enumerate(subsets)}
            utility_train = {
                subset: float(values_train["robust_composite"][index])
                for subset, index in subset_index.items()
            }
            triple_values = [subset for subset in subsets if len(subset) == target_size]
            greedy_subset = greedy_selection(utility_train, receptor_ids, target_size)
            additive_subset = tuple(
                sorted(
                    receptor_ids,
                    key=lambda receptor_id: (
                        -utility_train[(receptor_id,)],
                        receptor_id,
                    ),
                )[:target_size]
            )
            exact_subset = choose_highest(utility_train, triple_values)
            for method, subset in (
                ("direct_greedy", greedy_subset),
                ("additive_top3", additive_subset),
                ("exact_robust_oracle", exact_subset),
            ):
                fold_rows.append(
                    method_row(
                        target_id,
                        outer_fold,
                        method,
                        subset,
                        subset_index,
                        values_train,
                        values_holdout,
                    )
                )
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                terms = build_terms(
                    context,
                    receptor_ids,
                    coverage_fraction,
                    int(candidate["active_threshold"]),
                    alpha,
                )
                qubo = build_qubo(
                    terms,
                    receptor_ids,
                    target_size,
                    decoy_weight,
                    float(candidate["redundancy_weight"]),
                    cardinality_penalty,
                    constraint_penalty,
                )
                certificate = certify_states(
                    terms,
                    qubo,
                    receptor_ids,
                    target_size,
                    decoy_weight,
                    float(candidate["redundancy_weight"]),
                )
                selected_subset = tuple(certificate["selected_subset"])
                selected_index = subset_index[selected_subset]
                fold_rows.append(
                    method_row(
                        target_id,
                        outer_fold,
                        CANDIDATE_METHOD,
                        selected_subset,
                        subset_index,
                        values_train,
                        values_holdout,
                        candidate_id=candidate_id,
                        active_threshold=int(candidate["active_threshold"]),
                        redundancy_weight=float(candidate["redundancy_weight"]),
                        qubo_variable_count=len(qubo["variables"]),
                        qubo_max_abs_coefficient=certificate["max_abs_coefficient"],
                        qubo_scaled_best_second_gap=certificate["scaled_best_second_gap"],
                        qubo_equivalence_residual=certificate["equivalence_residual"],
                        qubo_state_count=certificate["state_count"],
                        selected_objective=certificate["selected_objective"],
                        selected_energy=certificate["selected_energy"],
                    )
                )

    comparisons: dict[str, Any] = {}
    classical_checks: dict[str, Any] = {}
    hardware_checks: dict[str, Any] = {}
    reference_noise = float(diagnostic["hardware_reference_noise_level"])
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        comparisons[candidate_id] = {
            f"{candidate_id}_vs_{method}": paired_comparison(
                fold_rows, candidate_id, method
            )
            for method in ("direct_greedy", "additive_top3")
        }
        greedy_comparison = comparisons[candidate_id][f"{candidate_id}_vs_direct_greedy"]
        classical_checks[candidate_id] = (
            all(
                float(value["mean_delta"])
                > float(diagnostic["classical_minimum_target_mean_delta"])
                for value in greedy_comparison["per_target"].values()
            )
            and int(greedy_comparison["positive_fold_count"])
            >= int(diagnostic["classical_minimum_positive_folds_of_eight"])
        )
        per_target_hardware: dict[str, Any] = {}
        for target_id, models in full_models.items():
            model = models[candidate_id]
            noise_rows = [
                row
                for row in noise_summaries
                if row["target_id"] == target_id
                and row["candidate_id"] == candidate_id
                and float(row["noise_level"]) == reference_noise
            ]
            if len(noise_rows) != len(diagnostic["noise_models"]):
                raise ValueError("reference noise model rows differ")
            gap_ok = (
                float(model["scaled_best_second_gap"])
                >= float(diagnostic["hardware_gap_threshold_scaled"])
            )
            stability_by_model = {
                str(row["noise_model_id"]): float(row["selection_stability"])
                for row in noise_rows
            }
            stability_ok = all(
                value >= float(diagnostic["hardware_stability_threshold"])
                for value in stability_by_model.values()
            )
            per_target_hardware[target_id] = {
                "scaled_best_second_gap": model["scaled_best_second_gap"],
                "gap_threshold": diagnostic["hardware_gap_threshold_scaled"],
                "gap_passed": gap_ok,
                "noise_level": reference_noise,
                "selection_stability_by_model": stability_by_model,
                "stability_threshold": diagnostic["hardware_stability_threshold"],
                "stability_passed": stability_ok,
            }
        hardware_checks[candidate_id] = {
            "per_target": per_target_hardware,
            "all_gap_checks_passed": all(value["gap_passed"] for value in per_target_hardware.values()),
            "all_stability_checks_passed": all(value["stability_passed"] for value in per_target_hardware.values()),
            "passed": all(
                value["gap_passed"] and value["stability_passed"]
                for value in per_target_hardware.values()
            ),
        }

    candidate_gate = {
        candidate_id: {
            "classical_passed": classical_checks[candidate_id],
            "hardware_passed": hardware_checks[candidate_id]["passed"],
            "ready_for_hardware_pilot": (
                classical_checks[candidate_id]
                and hardware_checks[candidate_id]["passed"]
            ),
        }
        for candidate_id in candidate_ids
    }
    ready_candidates = [
        candidate_id
        for candidate_id, checks in candidate_gate.items()
        if checks["ready_for_hardware_pilot"]
    ]
    result_status = (
        "stage19i_candidate_hardware_readiness_signal_do_not_claim_quantum_advantage"
        if ready_candidates
        else "stage19i_no_candidate_hardware_ready_do_not_execute_quantum"
    )
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
    write_csv(outputs["noise_trials_csv"], noise_trials)
    write_csv(outputs["noise_summary_csv"], noise_summaries)
    write_json(
        outputs["model_record_json"],
        {
            "schema_version": "1.0",
            "status": result_status,
            "algorithm_id": "frozen-threshold-consensus-coverage-qubo-v1",
            "data_boundary": data_boundary,
            "target_models": full_models,
        },
    )
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "status": result_status,
        "experiment_id": config["experiment_id"],
        "experiment_class": config["evidence_timing"]["experiment_class"],
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "data_boundary": data_boundary,
        "target_dimensions": target_dimensions,
        "candidate_ids": candidate_ids,
        "fold_method_aggregate": aggregate_method(fold_rows),
        "paired_comparisons": comparisons,
        "classical_checks": classical_checks,
        "hardware_checks": hardware_checks,
        "candidate_gate": candidate_gate,
        "ready_candidates": ready_candidates,
        "full_train_models": {
            target_id: {
                candidate_id: {
                    key: value
                    for key, value in model.items()
                    if key not in {"terms", "qubo", "selected_assignment"}
                }
                for candidate_id, model in models.items()
            }
            for target_id, models in full_models.items()
        },
        "outputs": {
            key: output_descriptor(root, path)
            for key, path in outputs.items()
            if key not in {"result_json", "report_md"}
        },
        "next_gate": (
            "review_one_candidate_for_preregistered_hardware_pilot"
            if ready_candidates
            else "redesign_objective_or_reframe_quantum_application_before_hardware"
        ),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    report_lines = [
        "# Stage 19i objective adequacy and noise screen",
        "",
        "Post-hoc train-only design review on MK14 and PPARG. No new docking, protected-panel rows, or quantum hardware jobs were used.",
        "",
        "## Classical outer-fold results",
        "",
        "| Target | Method | Candidate | Mean holdout robust composite | Mean primary |",
        "|---|---|---|---:|---:|",
    ]
    for row in result["fold_method_aggregate"]:
        report_lines.append(
            f"| {row['target_id']} | {row['method']} | {row['candidate_id'] or '-'} | "
            f"{row['mean_holdout_robust_composite']:.6f} | "
            f"{row['mean_holdout_primary']:.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Hardware-readiness checks",
            "",
            "| Candidate | Classical gate | Hardware gate | Ready |",
            "|---|---:|---:|---:|",
        ]
    )
    for candidate_id in candidate_ids:
        gate = candidate_gate[candidate_id]
        report_lines.append(
            f"| {candidate_id} | {gate['classical_passed']} | "
            f"{gate['hardware_passed']} | {gate['ready_for_hardware_pilot']} |"
        )
    report_lines.extend(
        [
            "",
            f"- Ready candidates: `{', '.join(ready_candidates) if ready_candidates else 'none'}`",
            f"- Next gate: `{result['next_gate']}`",
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
                "candidate_gate": result["candidate_gate"],
                "hardware_checks": result["hardware_checks"],
                "paired_comparisons": result["paired_comparisons"],
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
