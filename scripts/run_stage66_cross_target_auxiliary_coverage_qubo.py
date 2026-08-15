"""Develop a multiscale auxiliary-variable coverage QUBO across four targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_stage42d_bace1_large_pool_qubo_screen import bedroc_metrics, rank_cube
from scripts.run_stage42f_bace1_rank_sensitive_pair_qubo import pair_coefficients
from scripts.run_stage64_cross_target_uncertainty_shrunk_qubo import (
    K_VALUES,
    TOLERANCE,
    load_target,
    pairwise_jaccard,
)


SOLVER_QUBO = "auxiliary_qubo_beam_swap"
SOLVER_GREEDY = "same_objective_direct_greedy"
SOLVER_PAIR_OFF = "pair_off_baseline"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage66 frozen identity differs: {path}")
    return path


def subset_name(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(receptor_ids[index] for index in subset)


def normalized(values: np.ndarray) -> np.ndarray:
    low = float(np.min(values))
    high = float(np.max(values))
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-15):
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def bitset(values: np.ndarray) -> int:
    output = 0
    for index in np.flatnonzero(values):
        output |= 1 << int(index)
    return output


def seed_consensus(hits: np.ndarray, rule: str) -> np.ndarray:
    thresholds = {"any": 1, "majority": 2, "all": 3}
    if rule not in thresholds:
        raise ValueError(f"unknown seed consensus rule: {rule}")
    return np.sum(hits, axis=0) >= thresholds[rule]


def build_coverage_terms(
    ranks: np.ndarray,
    labels: np.ndarray,
    ligand_ids: list[str],
    mask: np.ndarray,
    candidate: dict[str, Any],
    fractions: list[float],
    scale_weights: list[float],
    bedroc_alpha: float,
) -> dict[str, Any]:
    if len(fractions) != len(scale_weights) or not math.isclose(
        sum(scale_weights), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Stage66 scale schedule differs")
    local_ranks = ranks[:, mask, :]
    local_labels = labels[mask]
    local_ids = [value for value, keep in zip(ligand_ids, mask) if keep]
    active_positions = np.flatnonzero(local_labels == 1)
    decoy_positions = np.flatnonzero(local_labels == 0)
    if not len(active_positions) or not len(decoy_positions):
        raise ValueError("Stage66 coverage terms require both label classes")
    singleton, _ = pair_coefficients(local_ranks, local_labels, bedroc_alpha)
    singleton = normalized(singleton)
    receptor_count = ranks.shape[2]
    scales: list[dict[str, Any]] = []
    active_states: list[dict[str, Any]] = []
    decoy_states: list[dict[str, Any]] = []
    for scale_index, (fraction, weight) in enumerate(
        zip(fractions, scale_weights)
    ):
        raw_hits = local_ranks <= float(fraction) + TOLERANCE
        active_hits = seed_consensus(
            raw_hits[:, active_positions, :], str(candidate["active_seed_rule"])
        )
        decoy_hits = seed_consensus(
            raw_hits[:, decoy_positions, :], str(candidate["decoy_seed_rule"])
        )
        active_bits = [
            bitset(active_hits[:, receptor]) for receptor in range(receptor_count)
        ]
        decoy_bits = [
            bitset(decoy_hits[:, receptor]) for receptor in range(receptor_count)
        ]
        scales.append(
            {
                "fraction": float(fraction),
                "weight": float(weight),
                "active_receptor_bits": active_bits,
                "decoy_receptor_bits": decoy_bits,
            }
        )
        for local_index, position in enumerate(active_positions):
            incidence = tuple(int(value) for value in np.flatnonzero(active_hits[local_index]))
            if incidence:
                active_states.append(
                    {
                        "state_id": f"a{scale_index}_{local_index}",
                        "ligand_id": local_ids[int(position)],
                        "scale_index": scale_index,
                        "fraction": float(fraction),
                        "objective_weight": float(weight) / len(active_positions),
                        "incidence": incidence,
                    }
                )
        for local_index, position in enumerate(decoy_positions):
            incidence = tuple(int(value) for value in np.flatnonzero(decoy_hits[local_index]))
            if incidence:
                decoy_states.append(
                    {
                        "state_id": f"d{scale_index}_{local_index}",
                        "ligand_id": local_ids[int(position)],
                        "scale_index": scale_index,
                        "fraction": float(fraction),
                        "objective_weight": float(weight) / len(decoy_positions),
                        "incidence": incidence,
                    }
                )
    return {
        "receptor_count": receptor_count,
        "active_count": len(active_positions),
        "decoy_count": len(decoy_positions),
        "singleton_utility": singleton,
        "scales": scales,
        "active_states": active_states,
        "decoy_states": decoy_states,
    }


class CoverageObjective:
    def __init__(self, terms: dict[str, Any], candidate: dict[str, Any]):
        self.terms = terms
        self.candidate = candidate
        self.receptor_count = int(terms["receptor_count"])
        self.cache: dict[tuple[int, ...], tuple[float, dict[str, float]]] = {}

    def score(
        self, subset: tuple[int, ...]
    ) -> tuple[float, dict[str, float]]:
        subset = tuple(sorted(subset))
        cached = self.cache.get(subset)
        if cached is not None:
            return cached
        active_coverage = 0.0
        decoy_exposure = 0.0
        for scale in self.terms["scales"]:
            active_union = 0
            decoy_union = 0
            for receptor in subset:
                active_union |= int(scale["active_receptor_bits"][receptor])
                decoy_union |= int(scale["decoy_receptor_bits"][receptor])
            active_coverage += (
                float(scale["weight"])
                * active_union.bit_count()
                / int(self.terms["active_count"])
            )
            decoy_exposure += (
                float(scale["weight"])
                * decoy_union.bit_count()
                / int(self.terms["decoy_count"])
            )
        singleton_quality = (
            float(np.mean(self.terms["singleton_utility"][list(subset)]))
            if subset
            else 0.0
        )
        value = (
            active_coverage
            - float(self.candidate["decoy_weight"]) * decoy_exposure
            + float(self.candidate["singleton_weight"]) * singleton_quality
        )
        components = {
            "active_multiscale_coverage": float(active_coverage),
            "decoy_multiscale_exposure": float(decoy_exposure),
            "singleton_quality": float(singleton_quality),
        }
        result = (float(value), components)
        self.cache[subset] = result
        return result


def objective_key(
    scorer: CoverageObjective, subset: tuple[int, ...]
) -> tuple[Any, ...]:
    return (-scorer.score(subset)[0], subset)


def direct_greedy_by_size(
    scorer: CoverageObjective, maximum_size: int
) -> dict[int, tuple[int, ...]]:
    current: tuple[int, ...] = tuple()
    output: dict[int, tuple[int, ...]] = {}
    for size in range(1, maximum_size + 1):
        selected = set(current)
        candidates = [
            tuple(sorted((*current, added)))
            for added in range(scorer.receptor_count)
            if added not in selected
        ]
        current = min(candidates, key=lambda value: objective_key(scorer, value))
        output[size] = current
    return output


def local_swap(
    scorer: CoverageObjective, subset: tuple[int, ...]
) -> tuple[int, ...]:
    current = subset
    while True:
        current_value = scorer.score(current)[0]
        selected = set(current)
        neighbors = {
            tuple(sorted((selected - {removed}) | {added}))
            for removed in current
            for added in range(scorer.receptor_count)
            if added not in selected
        }
        improving = [
            value
            for value in neighbors
            if scorer.score(value)[0] > current_value + TOLERANCE
        ]
        if not improving:
            return current
        current = min(improving, key=lambda value: objective_key(scorer, value))


def beam_swap_by_size(
    scorer: CoverageObjective, maximum_size: int, beam_width: int
) -> tuple[dict[int, tuple[int, ...]], dict[int, dict[str, int]]]:
    starts: dict[int, set[tuple[int, ...]]] = {
        1: {(index,) for index in range(scorer.receptor_count)}
    }
    beam = sorted(starts[1], key=lambda value: objective_key(scorer, value))[
        :beam_width
    ]
    for size in range(2, maximum_size + 1):
        expanded = {
            tuple(sorted((*subset, added)))
            for subset in beam
            for added in range(scorer.receptor_count)
            if added not in subset
        }
        beam = sorted(
            expanded, key=lambda value: objective_key(scorer, value)
        )[:beam_width]
        starts[size] = set(beam)
    for initial in range(scorer.receptor_count):
        current = (initial,)
        for size in range(2, maximum_size + 1):
            selected = set(current)
            candidates = [
                tuple(sorted((*current, added)))
                for added in range(scorer.receptor_count)
                if added not in selected
            ]
            current = min(
                candidates, key=lambda value: objective_key(scorer, value)
            )
            starts[size].add(current)
    selected: dict[int, tuple[int, ...]] = {}
    records: dict[int, dict[str, int]] = {}
    for size in range(1, maximum_size + 1):
        endpoints = {local_swap(scorer, value) for value in starts[size]}
        selected[size] = min(
            endpoints, key=lambda value: objective_key(scorer, value)
        )
        records[size] = {
            "start_state_count": len(starts[size]),
            "local_endpoint_count": len(endpoints),
        }
    return selected, records


def add_linear(coefficients: dict[str, Any], variable: str, value: float) -> None:
    coefficients["linear"][variable] = (
        coefficients["linear"].get(variable, 0.0) + float(value)
    )


def add_quadratic(
    coefficients: dict[str, Any], first: str, second: str, value: float
) -> None:
    if first == second:
        add_linear(coefficients, first, value)
        return
    key = "::".join(sorted((first, second)))
    coefficients["quadratic"][key] = (
        coefficients["quadratic"].get(key, 0.0) + float(value)
    )


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


def slack_weights(maximum: int) -> list[int]:
    if maximum < 1:
        return []
    weights: list[int] = []
    total = 0
    value = 1
    while total < maximum:
        weight = min(value, maximum - total)
        weights.append(weight)
        total += weight
        value *= 2
    return weights


def binary_assignment(weights: list[int], value: int) -> dict[int, int]:
    for bits in itertools.product((0, 1), repeat=len(weights)):
        if sum(weight * bit for weight, bit in zip(weights, bits)) == value:
            return {index: int(bit) for index, bit in enumerate(bits)}
    raise ValueError("Stage66 slack encoding cannot represent the value")


def build_sparse_qubo(
    terms: dict[str, Any],
    receptor_ids: list[str],
    target_size: int,
    candidate: dict[str, Any],
    cardinality_penalty: float,
    constraint_penalty: float,
) -> dict[str, Any]:
    coefficients: dict[str, Any] = {"constant": 0.0, "linear": {}, "quadratic": {}}
    x_names = [f"x__{value}" for value in receptor_ids]
    add_square(
        coefficients,
        -target_size,
        {variable: 1.0 for variable in x_names},
        cardinality_penalty,
    )
    active_variables: list[str] = []
    decoy_variables: list[str] = []
    slack_variables: list[str] = []
    for state in terms["active_states"]:
        y_name = f"y__{state['state_id']}"
        active_variables.append(y_name)
        expression = {y_name: 1.0}
        for index, weight in enumerate(slack_weights(len(state["incidence"]))):
            name = f"s__{state['state_id']}__{index}"
            slack_variables.append(name)
            expression[name] = float(weight)
        for receptor in state["incidence"]:
            expression[x_names[int(receptor)]] = -1.0
        add_square(coefficients, 0.0, expression, constraint_penalty)
        add_linear(coefficients, y_name, -float(state["objective_weight"]))
    for state in terms["decoy_states"]:
        z_name = f"z__{state['state_id']}"
        decoy_variables.append(z_name)
        add_linear(
            coefficients,
            z_name,
            float(candidate["decoy_weight"]) * float(state["objective_weight"]),
        )
        for receptor in state["incidence"]:
            x_name = x_names[int(receptor)]
            add_linear(coefficients, x_name, constraint_penalty)
            add_quadratic(coefficients, x_name, z_name, -constraint_penalty)
    singleton_scale = float(candidate["singleton_weight"]) / target_size
    for index, x_name in enumerate(x_names):
        add_linear(
            coefficients,
            x_name,
            -singleton_scale * float(terms["singleton_utility"][index]),
        )
    variables = sorted(
        set(x_names) | set(active_variables) | set(decoy_variables) | set(slack_variables)
    )
    return {
        "constant": float(coefficients["constant"]),
        "linear": {key: float(value) for key, value in coefficients["linear"].items()},
        "quadratic": {
            key: float(value) for key, value in coefficients["quadratic"].items()
        },
        "variables": variables,
        "variable_groups": {
            "x": x_names,
            "active_y": active_variables,
            "decoy_z": decoy_variables,
            "active_slack": slack_variables,
        },
        "target_size": target_size,
        "candidate_id": candidate["candidate_id"],
        "cardinality_penalty": cardinality_penalty,
        "constraint_penalty": constraint_penalty,
        "convention": (
            "Q(b)=constant+sum_v linear[v]*b_v+"
            "sum_u<v quadratic[u::v]*b_u*b_v; minimize Q"
        ),
    }


def assignment_for_subset(
    terms: dict[str, Any],
    qubo: dict[str, Any],
    receptor_ids: list[str],
    subset: tuple[int, ...],
) -> dict[str, int]:
    selected = set(subset)
    assignment = {variable: 0 for variable in qubo["variables"]}
    for index in selected:
        assignment[f"x__{receptor_ids[index]}"] = 1
    for state in terms["active_states"]:
        count = sum(int(index) in selected for index in state["incidence"])
        y = int(count > 0)
        assignment[f"y__{state['state_id']}"] = y
        slack = binary_assignment(slack_weights(len(state["incidence"])), count - y)
        for index, value in slack.items():
            assignment[f"s__{state['state_id']}__{index}"] = value
    for state in terms["decoy_states"]:
        assignment[f"z__{state['state_id']}"] = int(
            any(int(index) in selected for index in state["incidence"])
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


def metric_row(
    target_id: str,
    outer_fold: int,
    candidate: dict[str, Any],
    solver_id: str,
    subset_size: int,
    subset: tuple[int, ...],
    receptor_ids: list[str],
    train_scorer: CoverageObjective,
    holdout_scorer: CoverageObjective,
    ranks: np.ndarray,
    labels: np.ndarray,
    holdout_mask: np.ndarray,
    bedroc_alpha: float,
    search_record: dict[str, int],
) -> dict[str, Any]:
    train_value, train_components = train_scorer.score(subset)
    holdout_value, holdout_components = holdout_scorer.score(subset)
    metrics = bedroc_metrics(
        ranks[:, holdout_mask, :], labels[holdout_mask], subset, bedroc_alpha
    )
    return {
        "target_id": target_id,
        "outer_fold": outer_fold,
        "candidate_id": candidate["candidate_id"],
        "solver_id": solver_id,
        "subset_size": subset_size,
        "selected_subset": subset_name(subset, receptor_ids),
        "train_set_objective": train_value,
        "train_active_multiscale_coverage": train_components[
            "active_multiscale_coverage"
        ],
        "train_decoy_multiscale_exposure": train_components[
            "decoy_multiscale_exposure"
        ],
        "train_singleton_quality": train_components["singleton_quality"],
        "holdout_set_objective": holdout_value,
        "holdout_active_multiscale_coverage": holdout_components[
            "active_multiscale_coverage"
        ],
        "holdout_decoy_multiscale_exposure": holdout_components[
            "decoy_multiscale_exposure"
        ],
        "holdout_singleton_quality": holdout_components["singleton_quality"],
        "holdout_primary_bedroc": metrics["primary_bedroc"],
        "holdout_mean_seed_bedroc": metrics["mean_seed_bedroc"],
        "holdout_worst_seed_bedroc": metrics["worst_seed_bedroc"],
        "holdout_robust_bedroc": metrics["robust_bedroc_composite"],
        "search_start_state_count": search_record["start_state_count"],
        "search_local_endpoint_count": search_record["local_endpoint_count"],
    }


def summarize_candidates(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    target_order: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_off = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): float(
            row["holdout_robust_bedroc"]
        )
        for row in rows
        if row["solver_id"] == SOLVER_PAIR_OFF
    }
    greedy = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            row["candidate_id"],
            int(row["subset_size"]),
        ): row
        for row in rows
        if row["solver_id"] == SOLVER_GREEDY
    }
    target_rows: list[dict[str, Any]] = []
    for target_id in target_order:
        for candidate in candidates:
            selected = [
                row
                for row in rows
                if row["target_id"] == target_id
                and row["candidate_id"] == candidate["candidate_id"]
                and row["solver_id"] == SOLVER_QUBO
                and int(row["subset_size"]) >= 2
            ]
            pair_gains = [
                float(row["holdout_robust_bedroc"])
                - pair_off[
                    (target_id, int(row["outer_fold"]), int(row["subset_size"]))
                ]
                for row in selected
            ]
            greedy_rows = [
                greedy[
                    (
                        target_id,
                        int(row["outer_fold"]),
                        candidate["candidate_id"],
                        int(row["subset_size"]),
                    )
                ]
                for row in selected
            ]
            greedy_gains = [
                float(row["holdout_robust_bedroc"])
                - float(greedy_row["holdout_robust_bedroc"])
                for row, greedy_row in zip(selected, greedy_rows)
            ]
            objective_gains = [
                float(row["train_set_objective"])
                - float(greedy_row["train_set_objective"])
                for row, greedy_row in zip(selected, greedy_rows)
            ]
            target_rows.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate["candidate_id"],
                    "fixed_k_cell_count": len(selected),
                    "mean_fixed_k_holdout_robust_bedroc": statistics.fmean(
                        float(row["holdout_robust_bedroc"]) for row in selected
                    ),
                    "mean_gain_over_pair_off": statistics.fmean(pair_gains),
                    "minimum_fold_k_gain_over_pair_off": min(pair_gains),
                    "nonnegative_fold_k_gain_over_pair_off_count": sum(
                        value >= -TOLERANCE for value in pair_gains
                    ),
                    "mean_gain_over_same_objective_greedy": statistics.fmean(
                        greedy_gains
                    ),
                    "mean_train_objective_gain_over_greedy": statistics.fmean(
                        objective_gains
                    ),
                    "minimum_train_objective_gain_over_greedy": min(objective_gains),
                    "selection_difference_count_vs_greedy": sum(
                        row["selected_subset"] != greedy_row["selected_subset"]
                        for row, greedy_row in zip(selected, greedy_rows)
                    ),
                    "mean_fixed_k_selection_jaccard": statistics.fmean(
                        pairwise_jaccard(
                            [
                                str(row["selected_subset"])
                                for row in selected
                                if int(row["subset_size"]) == subset_size
                            ]
                        )
                        for subset_size in range(2, 7)
                    ),
                }
            )
    lookup = {
        (row["target_id"], row["candidate_id"]): row for row in target_rows
    }
    global_rows: list[dict[str, Any]] = []
    for order, candidate in enumerate(candidates):
        selected = [
            lookup[(target_id, candidate["candidate_id"])]
            for target_id in target_order
        ]
        gains = [float(row["mean_gain_over_pair_off"]) for row in selected]
        global_rows.append(
            {
                "candidate_order": order,
                "candidate_id": candidate["candidate_id"],
                "active_seed_rule": candidate["active_seed_rule"],
                "decoy_seed_rule": candidate["decoy_seed_rule"],
                "decoy_weight": candidate["decoy_weight"],
                "singleton_weight": candidate["singleton_weight"],
                "eligible_for_freeze": candidate["eligible_for_freeze"],
                "mean_target_gain_over_pair_off": statistics.fmean(gains),
                "worst_target_gain_over_pair_off": min(gains),
                "nonnegative_target_count_over_pair_off": sum(
                    value >= -TOLERANCE for value in gains
                ),
                "positive_target_count_over_pair_off": sum(
                    value > TOLERANCE for value in gains
                ),
                "mean_target_gain_over_same_objective_greedy": statistics.fmean(
                    float(row["mean_gain_over_same_objective_greedy"])
                    for row in selected
                ),
                "minimum_train_objective_gain_over_greedy": min(
                    float(row["minimum_train_objective_gain_over_greedy"])
                    for row in selected
                ),
                "selection_difference_count_vs_greedy": sum(
                    int(row["selection_difference_count_vs_greedy"])
                    for row in selected
                ),
                "mean_target_selection_jaccard": statistics.fmean(
                    float(row["mean_fixed_k_selection_jaccard"])
                    for row in selected
                ),
            }
        )
    return target_rows, global_rows


def candidate_key(
    row: dict[str, Any], order_lookup: dict[str, int]
) -> tuple[Any, ...]:
    return (
        -float(row["worst_target_gain_over_pair_off"]),
        -float(row["mean_target_gain_over_pair_off"]),
        -int(row["nonnegative_target_count_over_pair_off"]),
        order_lookup[str(row["candidate_id"])],
    )


def loto_rows(
    target_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    target_order: list[str],
) -> list[dict[str, Any]]:
    eligible = [row for row in candidates if bool(row["eligible_for_freeze"])]
    order = {row["candidate_id"]: index for index, row in enumerate(candidates)}
    lookup = {
        (row["target_id"], row["candidate_id"]): row for row in target_rows
    }
    output: list[dict[str, Any]] = []
    for held_target in target_order:
        development_targets = [value for value in target_order if value != held_target]
        summaries: list[dict[str, Any]] = []
        for candidate in eligible:
            rows = [
                lookup[(target_id, candidate["candidate_id"])]
                for target_id in development_targets
            ]
            gains = [float(row["mean_gain_over_pair_off"]) for row in rows]
            summaries.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "worst_target_gain_over_pair_off": min(gains),
                    "mean_target_gain_over_pair_off": statistics.fmean(gains),
                    "nonnegative_target_count_over_pair_off": sum(
                        value >= -TOLERANCE for value in gains
                    ),
                }
            )
        selected = min(summaries, key=lambda row: candidate_key(row, order))
        held = lookup[(held_target, selected["candidate_id"])]
        output.append(
            {
                "held_target_id": held_target,
                "selected_candidate_id": selected["candidate_id"],
                "development_worst_target_gain_over_pair_off": selected[
                    "worst_target_gain_over_pair_off"
                ],
                "development_mean_target_gain_over_pair_off": selected[
                    "mean_target_gain_over_pair_off"
                ],
                "held_target_gain_over_pair_off": held["mean_gain_over_pair_off"],
                "held_target_gain_over_same_objective_greedy": held[
                    "mean_gain_over_same_objective_greedy"
                ],
            }
        )
    return output


def qubo_model_record(
    target_order: list[str],
    targets: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    development: dict[str, Any],
    qubo_config: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    models: dict[str, Any] = {}
    maximum_residual = 0.0
    for target_index, target_id in enumerate(target_order):
        target = targets[target_id]
        mask = np.ones(len(target["ligand_ids"]), dtype=bool)
        ranks = rank_cube(target["scores"], mask)
        terms = build_coverage_terms(
            ranks,
            target["labels"],
            target["ligand_ids"],
            mask,
            candidate,
            [float(value) for value in development["coverage_fractions"]],
            [float(value) for value in development["scale_weights"]],
            float(development["bedroc_alpha"]),
        )
        scorer = CoverageObjective(terms, candidate)
        selected, search_records = beam_swap_by_size(
            scorer,
            max(K_VALUES),
            int(development["classical_beam_width"]),
        )
        reference_k = int(qubo_config["reference_model_k"])
        qubo = build_sparse_qubo(
            terms,
            target["receptor_ids"],
            reference_k,
            candidate,
            float(qubo_config["cardinality_penalty"]),
            float(qubo_config["constraint_penalty"]),
        )
        generator = random.Random(
            int(qubo_config["certificate_seed_base"]) + target_index
        )
        subsets = {selected[reference_k]}
        for _ in range(int(qubo_config["certificate_random_subset_count"])):
            subsets.add(
                tuple(
                    sorted(
                        generator.sample(
                            range(len(target["receptor_ids"])), reference_k
                        )
                    )
                )
            )
        residuals: list[float] = []
        for subset in sorted(subsets):
            assignment = assignment_for_subset(
                terms, qubo, target["receptor_ids"], subset
            )
            residuals.append(qubo_energy(qubo, assignment) + scorer.score(subset)[0])
        residual = max(abs(value) for value in residuals)
        maximum_residual = max(maximum_residual, residual)
        selected_subset = selected[reference_k]
        selected_assignment = assignment_for_subset(
            terms, qubo, target["receptor_ids"], selected_subset
        )
        models[target_id] = {
            "candidate_id": candidate["candidate_id"],
            "reference_k": reference_k,
            "selected_subset": subset_name(selected_subset, target["receptor_ids"]),
            "selected_objective": scorer.score(selected_subset)[0],
            "selected_energy": qubo_energy(qubo, selected_assignment),
            "equivalence_max_residual": residual,
            "equivalence_state_count": len(subsets),
            "search_record": search_records[reference_k],
            "variable_count": len(qubo["variables"]),
            "linear_coefficient_count": len(qubo["linear"]),
            "quadratic_coefficient_count": len(qubo["quadratic"]),
            "active_state_count": len(terms["active_states"]),
            "decoy_state_count": len(terms["decoy_states"]),
            "qubo_sha256": canonical_sha256(qubo),
            "selected_assignment": selected_assignment,
            "qubo": qubo,
        }
    return {
        "schema_version": "1.0",
        "algorithm_id": "multiscale-seed-robust-active-coverage-decoy-exposure-qubo-v2",
        "status": "posthoc_full_train_model_record",
        "selected_candidate": candidate,
        "objective": {
            "coverage_fractions": development["coverage_fractions"],
            "scale_weights": development["scale_weights"],
            "bedroc_alpha": development["bedroc_alpha"],
            "reduced_set_objective": (
                "sum_t w_t(active_OR_coverage_t-decoy_weight*decoy_OR_exposure_t)"
                "+singleton_weight*mean_normalized_singleton_quality"
            ),
        },
        "variables": {
            "x_i": "receptor selected",
            "y_at": "active ligand covered at scale t",
            "z_dt": "decoy exposed at scale t",
            "s_at": "binary slack enforcing y_at plus slack equals hit count",
        },
        "constraints": {
            "cardinality": "cardinality_penalty*(sum_i x_i-k)^2",
            "active_or": "constraint_penalty*(y_at+s_at-sum_i h_ati*x_i)^2",
            "decoy_or": "constraint_penalty*x_i*(1-z_dt) for each exposure edge",
        },
        "hardware_execution": False,
        "targets": models,
    }, maximum_residual


def report_text(result: dict[str, Any]) -> str:
    selected = result["selected_candidate"]
    gate = result["freeze_gate"]
    return f"""# Stage66 cross-target auxiliary-variable coverage QUBO

## Scope

Four consumed historical development matrices were analyzed with no new docking,
protected-data read, or quantum-hardware job.

## Objective

The QUBO rewards multiscale active-ligand OR coverage at 5%, 10%, and 20% rank
thresholds, penalizes decoy OR exposure, and anchors selection with robust
singleton quality. Active and decoy coverage are represented by explicit binary
auxiliary variables rather than receptor-pair residuals.

## Selected historical-development candidate

- Candidate: `{selected['candidate_id']}`
- Mean target gain over pair-off: {selected['mean_target_gain_over_pair_off']:+.6f}
- Worst-target gain over pair-off: {selected['worst_target_gain_over_pair_off']:+.6f}
- Nonnegative targets: {selected['nonnegative_target_count_over_pair_off']}/4
- Mean target gain over same-objective direct greedy: {selected['mean_target_gain_over_same_objective_greedy']:+.6f}
- Selection differences from same-objective greedy: {selected['selection_difference_count_vs_greedy']}

## Decision

Coverage objective freeze gate: **{'PASS' if gate['coverage_objective_freeze_authorized'] else 'NO-GO'}**.

The same-objective classical comparison is a solver audit, not a claim that a
QUBO must beat the optimum of its own objective. Any freeze only authorizes
preregistration on a genuinely new target; it does not authorize hardware or
establish quantum advantage.
"""


def compute_analysis(config: dict[str, Any], root: Path) -> dict[str, Any]:
    stage64_config_path = verified(root, config["inputs"]["stage64_config"])
    stage65_result_path = verified(root, config["inputs"]["stage65_result"])
    stage65_audit_path = verified(root, config["inputs"]["stage65_audit"])
    stage65_metrics_path = verified(root, config["inputs"]["stage65_fixed_k_metrics"])
    stage64_config = read_json(stage64_config_path)
    stage65_result = read_json(stage65_result_path)
    stage65_audit = read_json(stage65_audit_path)
    if stage65_result.get("status") != "stage65_cross_target_pair_sign_mechanism_complete":
        raise ValueError("Stage66 source Stage65 result did not complete")
    if stage65_audit.get("status") != "stage65_cross_target_pair_sign_mechanism_independent_audit_ok":
        raise ValueError("Stage66 source Stage65 audit did not pass")
    if not stage65_result["decision"]["auxiliary_coverage_qubo_design_authorized"]:
        raise ValueError("Stage65 did not authorize auxiliary coverage design")
    development = config["development"]
    candidates = [dict(value) for value in config["candidate_grid"]]
    target_order = [str(value) for value in development["target_order"]]
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("Stage66 candidate IDs are not unique")
    targets = {
        target_id: load_target(root, target_id, stage64_config["targets"][target_id])
        for target_id in target_order
    }
    rows: list[dict[str, Any]] = []
    for target_id in target_order:
        target = targets[target_id]
        ligand_ids = target["ligand_ids"]
        labels = target["labels"]
        receptor_ids = target["receptor_ids"]
        for outer_fold in range(int(development["outer_fold_count"])):
            train_mask = np.asarray(
                [target["outer"][ligand_id] != outer_fold for ligand_id in ligand_ids]
            )
            holdout_mask = ~train_mask
            ranks = rank_cube(target["scores"], train_mask)
            for candidate in candidates:
                train_terms = build_coverage_terms(
                    ranks,
                    labels,
                    ligand_ids,
                    train_mask,
                    candidate,
                    [float(value) for value in development["coverage_fractions"]],
                    [float(value) for value in development["scale_weights"]],
                    float(development["bedroc_alpha"]),
                )
                holdout_terms = build_coverage_terms(
                    ranks,
                    labels,
                    ligand_ids,
                    holdout_mask,
                    candidate,
                    [float(value) for value in development["coverage_fractions"]],
                    [float(value) for value in development["scale_weights"]],
                    float(development["bedroc_alpha"]),
                )
                train_scorer = CoverageObjective(train_terms, candidate)
                holdout_scorer = CoverageObjective(holdout_terms, candidate)
                greedy = direct_greedy_by_size(train_scorer, max(K_VALUES))
                selected, records = beam_swap_by_size(
                    train_scorer,
                    max(K_VALUES),
                    int(development["classical_beam_width"]),
                )
                for subset_size in K_VALUES:
                    rows.append(
                        metric_row(
                            target_id,
                            outer_fold,
                            candidate,
                            SOLVER_QUBO,
                            subset_size,
                            selected[subset_size],
                            receptor_ids,
                            train_scorer,
                            holdout_scorer,
                            ranks,
                            labels,
                            holdout_mask,
                            float(development["bedroc_alpha"]),
                            records[subset_size],
                        )
                    )
                    rows.append(
                        metric_row(
                            target_id,
                            outer_fold,
                            candidate,
                            SOLVER_GREEDY,
                            subset_size,
                            greedy[subset_size],
                            receptor_ids,
                            train_scorer,
                            holdout_scorer,
                            ranks,
                            labels,
                            holdout_mask,
                            float(development["bedroc_alpha"]),
                            {"start_state_count": 1, "local_endpoint_count": 1},
                        )
                    )
            print(
                json.dumps(
                    {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "candidate_count": len(candidates),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    source_pair_off = [
        row for row in read_csv(stage65_metrics_path) if row["candidate_id"] == "pair_off"
    ]
    for source in source_pair_off:
        rows.append(
            {
                "target_id": source["target_id"],
                "outer_fold": int(source["outer_fold"]),
                "candidate_id": "pair_off",
                "solver_id": SOLVER_PAIR_OFF,
                "subset_size": int(source["subset_size"]),
                "selected_subset": source["selected_subset"],
                "train_set_objective": source["train_qubo_objective"],
                "train_active_multiscale_coverage": "",
                "train_decoy_multiscale_exposure": "",
                "train_singleton_quality": "",
                "holdout_set_objective": "",
                "holdout_active_multiscale_coverage": "",
                "holdout_decoy_multiscale_exposure": "",
                "holdout_singleton_quality": "",
                "holdout_primary_bedroc": source["holdout_primary_bedroc"],
                "holdout_mean_seed_bedroc": source["holdout_mean_seed_bedroc"],
                "holdout_worst_seed_bedroc": source["holdout_worst_seed_bedroc"],
                "holdout_robust_bedroc": source["holdout_robust_bedroc"],
                "search_start_state_count": source["search_start_state_count"],
                "search_local_endpoint_count": source["search_local_endpoint_count"],
            }
        )
    expected = (
        len(target_order)
        * int(development["outer_fold_count"])
        * len(candidates)
        * len(K_VALUES)
        * 2
        + len(target_order) * int(development["outer_fold_count"]) * len(K_VALUES)
    )
    if len(rows) != expected or len(source_pair_off) != 96:
        raise ValueError("Stage66 metric dimensions differ")
    target_rows, global_rows = summarize_candidates(rows, candidates, target_order)
    order = {row["candidate_id"]: index for index, row in enumerate(candidates)}
    eligible = [row for row in global_rows if str(row["eligible_for_freeze"]) == "True"]
    selected = min(eligible, key=lambda row: candidate_key(row, order))
    selected_candidate = next(
        row for row in candidates if row["candidate_id"] == selected["candidate_id"]
    )
    loto = loto_rows(target_rows, candidates, target_order)
    model_record, maximum_residual = qubo_model_record(
        target_order,
        targets,
        selected_candidate,
        development,
        config["qubo_encoding"],
    )
    thresholds = config["freeze_gate"]
    qubo_rows = [row for row in rows if row["solver_id"] == SOLVER_QUBO]
    greedy_lookup = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            row["candidate_id"],
            int(row["subset_size"]),
        ): row
        for row in rows
        if row["solver_id"] == SOLVER_GREEDY
    }
    noninferior = sum(
        float(row["train_set_objective"])
        + TOLERANCE
        >= float(
            greedy_lookup[
                (
                    row["target_id"],
                    int(row["outer_fold"]),
                    row["candidate_id"],
                    int(row["subset_size"]),
                )
            ]["train_set_objective"]
        )
        for row in qubo_rows
    )
    selected_difference_fraction = float(selected["selection_difference_count_vs_greedy"]) / (
        len(target_order) * int(development["outer_fold_count"]) * 5
    )
    loto_gains = [float(row["held_target_gain_over_pair_off"]) for row in loto]
    checks = {
        "minimum_mean_target_gain_over_pair_off": float(
            selected["mean_target_gain_over_pair_off"]
        )
        >= float(thresholds["minimum_mean_target_gain_over_pair_off"]) - TOLERANCE,
        "minimum_worst_target_gain_over_pair_off": float(
            selected["worst_target_gain_over_pair_off"]
        )
        >= float(thresholds["minimum_worst_target_gain_over_pair_off"]) - TOLERANCE,
        "minimum_nonnegative_target_count_over_pair_off": int(
            selected["nonnegative_target_count_over_pair_off"]
        )
        >= int(thresholds["minimum_nonnegative_target_count_over_pair_off"]),
        "minimum_loto_mean_gain_over_pair_off": statistics.fmean(loto_gains)
        >= float(thresholds["minimum_loto_mean_gain_over_pair_off"]) - TOLERANCE,
        "minimum_positive_loto_target_count": sum(value > TOLERANCE for value in loto_gains)
        >= int(thresholds["minimum_positive_loto_target_count"]),
        "all_same_objective_search_cells_noninferior": noninferior == len(qubo_rows),
        "minimum_selection_difference_fraction_vs_greedy": selected_difference_fraction
        >= float(thresholds["minimum_selection_difference_fraction_vs_greedy"])
        - TOLERANCE,
        "maximum_qubo_energy_equivalence_residual": maximum_residual
        <= float(thresholds["maximum_qubo_energy_equivalence_residual"])
        + TOLERANCE,
    }
    passed = all(checks.values())
    model_record["status"] = (
        "coverage_objective_candidate_frozen_for_new_target_preregistration"
        if passed
        else "coverage_objective_not_frozen"
    )
    return {
        "rows": rows,
        "target_rows": target_rows,
        "global_rows": global_rows,
        "loto_rows": loto,
        "model_record": model_record,
        "result": {
            "schema_version": "1.0",
            "status": "stage66_cross_target_auxiliary_coverage_qubo_complete",
            "experiment_class": "posthoc cross-target train-only objective development",
            "candidate_count": len(candidates),
            "fixed_k_metric_count": len(rows),
            "pair_off_reproduction_cell_count": len(source_pair_off),
            "target_input_audits": {
                target_id: {
                    "ligand_count": len(targets[target_id]["ligand_ids"]),
                    "receptor_count": len(targets[target_id]["receptor_ids"]),
                    "score_row_count": 3
                    * len(targets[target_id]["ligand_ids"])
                    * len(targets[target_id]["receptor_ids"]),
                    "input_descriptors": targets[target_id]["input_descriptors"],
                }
                for target_id in target_order
            },
            "selected_candidate": selected,
            "loto_transfer": {
                "mean_gain_over_pair_off": statistics.fmean(loto_gains),
                "worst_gain_over_pair_off": min(loto_gains),
                "positive_target_count": sum(value > TOLERANCE for value in loto_gains),
                "nonnegative_target_count": sum(value >= -TOLERANCE for value in loto_gains),
            },
            "solver_audit": {
                "same_objective_cell_count": len(qubo_rows),
                "beam_swap_noninferior_objective_cell_count": noninferior,
                "selected_candidate_selection_difference_fraction_vs_greedy": selected_difference_fraction,
                "maximum_qubo_energy_equivalence_residual": maximum_residual,
                "same_objective_classical_ceiling_acknowledged": True,
            },
            "freeze_gate": {
                "checks": checks,
                "coverage_objective_freeze_authorized": passed,
            },
            "decision": {
                "new_target_preregistration_authorized": passed,
                "fresh_validation_authorized": False,
                "new_docking_authorized": False,
                "quantum_hardware_authorized": False,
                "same_target_retuning_authorized": False,
                "next_action": (
                    "freeze the selected coverage objective and preregister one genuinely new target"
                    if passed
                    else "do not retune this coverage family on the same four targets; review objective scope"
                ),
            },
            "data_boundary": {
                "historical_development_targets_read": 4,
                "fresh_validation_rows_read": 0,
                "locked_test_rows_read": 0,
                "new_docking_jobs": 0,
                "quantum_hardware_jobs": 0,
            },
            "interpretation_boundary": config["interpretation_boundary"],
        },
    }


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    runner = verified(root, config["implementation"]["runner"])
    if runner.resolve() != Path(__file__).resolve():
        raise ValueError("Stage66 runner identity differs")
    for value in config["implementation"].values():
        verified(root, value)
    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage66 outputs exist; pass --overwrite")
    analysis = compute_analysis(config, root)
    write_csv(outputs["fixed_k_metrics_csv"], analysis["rows"])
    write_csv(outputs["target_summary_csv"], analysis["target_rows"])
    write_csv(outputs["global_summary_csv"], analysis["global_rows"])
    write_csv(outputs["loto_summary_csv"], analysis["loto_rows"])
    write_json(outputs["model_record_json"], analysis["model_record"])
    result = analysis["result"]
    result["config"] = descriptor(root, config_path)
    result["implementation"] = descriptor(root, runner)
    result["outputs"] = {
        key: descriptor(root, path)
        for key, path in outputs.items()
        if key not in {"result_json", "audit_json", "report_md"}
    }
    result["analysis_payload_sha256"] = canonical_sha256(
        {
            "target_summary": analysis["target_rows"],
            "global_summary": analysis["global_rows"],
            "loto_summary": analysis["loto_rows"],
            "selected_candidate": result["selected_candidate"],
            "freeze_gate": result["freeze_gate"],
        }
    )
    write_json(outputs["result_json"], result)
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text(report_text(result), encoding="ascii")
    result["outputs"]["report_md"] = descriptor(root, outputs["report_md"])
    write_json(outputs["result_json"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage66_cross_target_auxiliary_coverage_qubo.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
