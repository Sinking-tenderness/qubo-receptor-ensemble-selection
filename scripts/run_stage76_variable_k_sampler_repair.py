"""Repair Stage75 variable-k sampler fidelity without changing its CQM."""

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

import dimod
import numpy as np

try:
    import scripts.run_stage75_explicit_variable_k_cqm as s75
except ImportError:
    import run_stage75_explicit_variable_k_cqm as s75


TOLERANCE = 1e-12
SOURCE_METHOD = "stage75_cold_global_annealing_reference"
NEW_METHODS = (
    "cold_adjacent_variable_annealing",
    "decomposed_warm_adjacent_annealing",
    "decomposed_warm_parallel_tempering",
    "frontier_warm_parallel_tempering",
)
METHOD_ORDER = (SOURCE_METHOD,) + NEW_METHODS


def normalization(cell: dict[str, Any]) -> float:
    return float(
        cell["model"]["pair_scale"]
        * math.comb(max(cell["frontiers"]), 2)
    )


def adjacent_proposal(
    cell: dict[str, Any], current: tuple[int, ...], rng: np.random.Generator
) -> tuple[int, ...]:
    protocol = cell["solver_protocol"]
    if rng.random() < float(protocol["swap_move_probability"]):
        outgoing = current[int(rng.integers(0, len(current)))]
        selected = set(current)
        available = [
            index
            for index in range(cell["model"]["count"])
            if index not in selected
        ]
        incoming = available[int(rng.integers(0, len(available)))]
        return tuple(sorted((selected - {outgoing}) | {incoming}))

    allowed = tuple(cell["frontiers"])
    position = allowed.index(len(current))
    neighbor_positions = []
    if position > 0:
        neighbor_positions.append(position - 1)
    if position + 1 < len(allowed):
        neighbor_positions.append(position + 1)
    target_k = allowed[
        neighbor_positions[int(rng.integers(0, len(neighbor_positions)))]
    ]
    selected = set(current)
    if target_k > len(current):
        available = np.asarray(
            [
                index
                for index in range(cell["model"]["count"])
                if index not in selected
            ],
            dtype=int,
        )
        additions = rng.choice(
            available, target_k - len(current), replace=False
        )
        selected.update(int(value) for value in additions)
    else:
        removals = rng.choice(
            np.asarray(current, dtype=int),
            len(current) - target_k,
            replace=False,
        )
        selected.difference_update(int(value) for value in removals)
    return tuple(sorted(selected))


def starting_subset(
    cell: dict[str, Any], source: str, rng: np.random.Generator
) -> tuple[tuple[int, ...], int, int, int]:
    if source == "cold_random":
        subset, attempts, fallback = s75.random_feasible_start(
            cell,
            rng,
            int(cell["solver_protocol"]["maximum_initialization_attempts"]),
        )
        return subset, attempts, int(fallback), 1
    if source == "decomposed_deterministic":
        subset, _, _ = s75.fixed_candidate(cell, "deterministic")
        return subset, 0, 0, len(cell["frontiers"])
    if source == "frozen_frontier":
        subset, _, _ = s75.fixed_candidate(cell, "reference")
        return subset, 0, 0, len(cell["frontiers"])
    raise ValueError(f"unknown Stage76 initialization source: {source}")


def adjacent_annealing(
    cell: dict[str, Any], rng: np.random.Generator, source: str
) -> dict[str, Any]:
    protocol = cell["solver_protocol"]
    budget = int(protocol["proposal_budget"])
    beta_minimum, beta_maximum = (
        float(value) for value in protocol["annealing_beta_range"]
    )
    current, attempts, fallback, candidate_count = starting_subset(
        cell, source, rng
    )
    current_energy = s75.variable_energy(
        cell["model"], current, cell["reward"]
    )
    best, best_energy = current, current_energy
    feasible_proposals = accepted = 0
    evaluations = 1
    scale = normalization(cell)
    for step in range(budget):
        candidate = adjacent_proposal(cell, current, rng)
        if not s75.valid(cell, candidate):
            continue
        feasible_proposals += 1
        evaluations += 1
        candidate_energy = s75.variable_energy(
            cell["model"], candidate, cell["reward"]
        )
        delta = (candidate_energy - current_energy) / scale
        fraction = step / max(1, budget - 1)
        beta = beta_minimum * (beta_maximum / beta_minimum) ** fraction
        if delta <= 0.0 or rng.random() < math.exp(-beta * delta):
            current, current_energy = candidate, candidate_energy
            accepted += 1
            if (current_energy, current) < (best_energy, best):
                best, best_energy = current, current_energy
    return {
        "subset": best,
        "proposal_count": budget,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "exchange_attempt_count": 0,
        "accepted_exchange_count": 0,
        "initialization_attempt_count": attempts,
        "initialization_fallback_count": fallback,
        "initialization_candidate_count": candidate_count,
        "initialization_source": source,
    }


def warm_replica_states(
    cell: dict[str, Any], source_field: str, replica_count: int
) -> list[tuple[int, ...]]:
    states = [
        tuple(record[f"{source_field}_subset"])
        for record in cell["frontiers"].values()
    ]
    states = sorted(
        states,
        key=lambda subset: (
            s75.variable_energy(cell["model"], subset, cell["reward"]),
            subset,
        ),
        reverse=True,
    )
    best = states[-1]
    while len(states) < replica_count:
        states.insert(0, best)
    if len(states) > replica_count:
        states = states[-replica_count:]
        states.sort(
            key=lambda subset: (
                s75.variable_energy(
                    cell["model"], subset, cell["reward"]
                ),
                subset,
            ),
            reverse=True,
        )
    return states


def parallel_tempering(
    cell: dict[str, Any], rng: np.random.Generator, source_field: str
) -> dict[str, Any]:
    protocol = cell["solver_protocol"]
    budget = int(protocol["proposal_budget"])
    replica_count = int(protocol["replica_count"])
    exchange_interval = int(protocol["exchange_interval_sweeps"])
    beta_minimum, beta_maximum = (
        float(value) for value in protocol["parallel_tempering_beta_range"]
    )
    betas = np.geomspace(beta_minimum, beta_maximum, replica_count)
    states = warm_replica_states(cell, source_field, replica_count)
    energies = [
        s75.variable_energy(cell["model"], subset, cell["reward"])
        for subset in states
    ]
    best_index = min(range(replica_count), key=lambda index: (energies[index], states[index]))
    best, best_energy = states[best_index], energies[best_index]
    feasible_proposals = accepted = 0
    exchange_attempts = accepted_exchanges = 0
    evaluations = replica_count
    scale = normalization(cell)
    sweep = 0
    for step in range(budget):
        replica = step % replica_count
        candidate = adjacent_proposal(cell, states[replica], rng)
        if s75.valid(cell, candidate):
            feasible_proposals += 1
            evaluations += 1
            candidate_energy = s75.variable_energy(
                cell["model"], candidate, cell["reward"]
            )
            delta = (candidate_energy - energies[replica]) / scale
            if delta <= 0.0 or rng.random() < math.exp(-betas[replica] * delta):
                states[replica] = candidate
                energies[replica] = candidate_energy
                accepted += 1
                if (candidate_energy, candidate) < (best_energy, best):
                    best, best_energy = candidate, candidate_energy
        if (step + 1) % replica_count == 0:
            sweep += 1
            if sweep % exchange_interval == 0:
                parity = (sweep // exchange_interval) % 2
                for left in range(parity, replica_count - 1, 2):
                    right = left + 1
                    exchange_attempts += 1
                    log_acceptance = (
                        (betas[left] - betas[right])
                        * (energies[left] - energies[right])
                        / scale
                    )
                    if log_acceptance >= 0.0 or rng.random() < math.exp(
                        log_acceptance
                    ):
                        states[left], states[right] = states[right], states[left]
                        energies[left], energies[right] = energies[right], energies[left]
                        accepted_exchanges += 1
    return {
        "subset": best,
        "proposal_count": budget,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "exchange_attempt_count": exchange_attempts,
        "accepted_exchange_count": accepted_exchanges,
        "initialization_attempt_count": 0,
        "initialization_fallback_count": 0,
        "initialization_candidate_count": len(cell["frontiers"]),
        "initialization_source": (
            "decomposed_deterministic_fronts"
            if source_field == "deterministic"
            else "frozen_fixed_k_frontiers"
        ),
    }


def trial_row(
    cell: dict[str, Any], method: str, repeat: int, seed: int, solved: dict[str, Any]
) -> dict[str, Any]:
    subset = tuple(solved["subset"])
    if not s75.valid(cell, subset):
        raise ValueError("Stage76 solver returned an infeasible state")
    energy = s75.variable_energy(cell["model"], subset, cell["reward"])
    row = {
        "target_id": cell["model"]["record"]["target_id"],
        "outer_fold": int(cell["model"]["record"]["outer_fold"]),
        "reward_quantile": float(cell["reward_quantile"]),
        "reward_value": float(cell["reward"]),
        "method": method,
        "repeat": repeat,
        "seed": seed,
    }
    row.update({key: value for key, value in solved.items() if key != "subset"})
    row.update(
        {
            "selected_k": len(subset),
            "solution_subset": s75.subset_name(cell["model"], subset),
            "solution_deficit": s75.subset_deficit(cell["model"], subset),
            "solution_raw_pair_objective": s75.raw_objective(
                cell["model"], subset
            ),
            "solution_variable_energy": energy,
        }
    )
    return row


def frozen_source_rows(
    cell: dict[str, Any], source_trials: list[dict[str, str]]
) -> list[dict[str, Any]]:
    target = str(cell["model"]["record"]["target_id"])
    fold = int(cell["model"]["record"]["outer_fold"])
    quantile = float(cell["reward_quantile"])
    selected = [
        row
        for row in source_trials
        if row["target_id"] == target
        and int(row["outer_fold"]) == fold
        and math.isclose(float(row["reward_quantile"]), quantile)
        and row["method"] == "constraint_native_variable_annealing"
    ]
    if len(selected) != 8:
        raise ValueError("Stage76 Stage75 source trial count differs")
    output = []
    for row in selected:
        subset = s75.parse_subset(cell["model"], row["solution_subset"])
        solved = {
            "subset": subset,
            "proposal_count": int(row["proposal_count"]),
            "feasible_proposal_count": int(row["feasible_proposal_count"]),
            "objective_evaluation_count": int(row["objective_evaluation_count"]),
            "accepted_move_count": int(row["accepted_move_count"]),
            "exchange_attempt_count": 0,
            "accepted_exchange_count": 0,
            "initialization_attempt_count": int(row["initialization_attempt_count"]),
            "initialization_fallback_count": int(row["initialization_fallback_count"]),
            "initialization_candidate_count": 1,
            "initialization_source": "stage75_cold_random_frozen",
        }
        recovered = trial_row(
            cell,
            SOURCE_METHOD,
            int(row["repeat"]),
            int(row["seed"]),
            solved,
        )
        if not math.isclose(
            float(recovered["solution_variable_energy"]),
            float(row["solution_variable_energy"]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError("Stage76 Stage75 source energy differs")
        output.append(recovered)
    return output


def reference_values(
    cell: dict[str, Any], source_trials: list[dict[str, str]]
) -> dict[str, Any]:
    frontier_subset, frontier_energy, _ = s75.fixed_candidate(cell, "reference")
    target = str(cell["model"]["record"]["target_id"])
    fold = int(cell["model"]["record"]["outer_fold"])
    quantile = float(cell["reward_quantile"])
    tabu = [
        row
        for row in source_trials
        if row["target_id"] == target
        and int(row["outer_fold"]) == fold
        and math.isclose(float(row["reward_quantile"]), quantile)
        and row["method"] == "budgeted_variable_tabu"
    ]
    joint = min(
        tabu,
        key=lambda row: (
            float(row["solution_variable_energy"]), row["solution_subset"]
        ),
    )
    joint_subset = s75.parse_subset(cell["model"], joint["solution_subset"])
    joint_energy = s75.variable_energy(
        cell["model"], joint_subset, cell["reward"]
    )
    exact = all(
        value["reference_type"] == "exact_enumeration"
        for value in cell["frontiers"].values()
    )
    return {
        "frontier_subset": frontier_subset,
        "frontier_energy": frontier_energy,
        "joint_subset": joint_subset,
        "joint_energy": joint_energy,
        "exact_frontier_available": exact,
    }


def compare_methods(
    cell: dict[str, Any], rows: list[dict[str, Any]], references: dict[str, Any], tolerance: float
) -> list[dict[str, Any]]:
    output = []
    scale = normalization(cell)
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        best = min(
            selected,
            key=lambda row: (
                float(row["solution_variable_energy"]), row["solution_subset"]
            ),
        )
        energy = float(best["solution_variable_energy"])
        delta_joint = (energy - references["joint_energy"]) / scale
        delta_frontier = (energy - references["frontier_energy"]) / scale
        output.append(
            {
                "target_id": cell["model"]["record"]["target_id"],
                "outer_fold": int(cell["model"]["record"]["outer_fold"]),
                "reward_quantile": float(cell["reward_quantile"]),
                "reward_value": float(cell["reward"]),
                "method": method,
                "trial_count": len(selected),
                "best_repeat": int(best["repeat"]),
                "best_selected_k": int(best["selected_k"]),
                "best_subset": best["solution_subset"],
                "best_energy": energy,
                "mean_energy": statistics.fmean(
                    float(row["solution_variable_energy"]) for row in selected
                ),
                "frozen_frontier_energy": references["frontier_energy"],
                "stage75_joint_tabu_energy": references["joint_energy"],
                "delta_vs_frontier_normalized": delta_frontier,
                "delta_vs_joint_normalized": delta_joint,
                "within_frontier_tolerance": delta_frontier <= tolerance,
                "within_joint_tolerance": delta_joint <= tolerance,
                "strict_frontier_improvement": delta_frontier < -tolerance,
                "exact_frontier_available": references[
                    "exact_frontier_available"
                ],
                "exact_frontier_match": bool(
                    references["exact_frontier_available"]
                    and energy <= references["frontier_energy"] + TOLERANCE
                ),
                "unique_solution_count": len(
                    {row["solution_subset"] for row in selected}
                ),
                "mean_feasible_proposal_fraction": statistics.fmean(
                    int(row["feasible_proposal_count"])
                    / max(1, int(row["proposal_count"]))
                    for row in selected
                ),
                "mean_exchange_acceptance_fraction": statistics.fmean(
                    int(row["accepted_exchange_count"])
                    / max(1, int(row["exchange_attempt_count"]))
                    for row in selected
                ),
            }
        )
    return output


def summarize(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    exact_total = sum(
        row["exact_frontier_available"]
        for row in comparisons
        if row["method"] == SOURCE_METHOD
    )
    for method in METHOD_ORDER:
        rows = [row for row in comparisons if row["method"] == method]
        exact = [row for row in rows if row["exact_frontier_available"]]
        output.append(
            {
                "method": method,
                "cell_count": len(rows),
                "trial_count": sum(int(row["trial_count"]) for row in rows),
                "exact_frontier_cell_count": exact_total,
                "exact_frontier_match_rate": statistics.fmean(
                    bool(row["exact_frontier_match"]) for row in exact
                ),
                "joint_competitive_fraction": statistics.fmean(
                    bool(row["within_joint_tolerance"]) for row in rows
                ),
                "frontier_competitive_fraction": statistics.fmean(
                    bool(row["within_frontier_tolerance"]) for row in rows
                ),
                "strict_frontier_improvement_cell_count": sum(
                    bool(row["strict_frontier_improvement"]) for row in rows
                ),
                "mean_delta_vs_joint_normalized": statistics.fmean(
                    float(row["delta_vs_joint_normalized"]) for row in rows
                ),
                "mean_delta_vs_frontier_normalized": statistics.fmean(
                    float(row["delta_vs_frontier_normalized"])
                    for row in rows
                ),
                "mean_unique_solution_count": statistics.fmean(
                    int(row["unique_solution_count"]) for row in rows
                ),
                "mean_feasible_proposal_fraction": statistics.fmean(
                    float(row["mean_feasible_proposal_fraction"])
                    for row in rows
                ),
                "mean_exchange_acceptance_fraction": statistics.fmean(
                    float(row["mean_exchange_acceptance_fraction"])
                    for row in rows
                ),
            }
        )
    return output


def pairwise_ablation(
    comparisons: list[dict[str, Any]], left: str, right: str, tolerance: float
) -> dict[str, Any]:
    lookup = {
        (row["target_id"], row["outer_fold"], row["reward_quantile"], row["method"]): row
        for row in comparisons
    }
    deltas = []
    wins = losses = ties = 0
    for target, fold, quantile in sorted(
        {
            (row["target_id"], row["outer_fold"], row["reward_quantile"])
            for row in comparisons
        }
    ):
        left_row = lookup[(target, fold, quantile, left)]
        right_row = lookup[(target, fold, quantile, right)]
        delta = float(right_row["best_energy"]) - float(left_row["best_energy"])
        deltas.append(delta)
        if float(right_row["delta_vs_joint_normalized"]) < float(
            left_row["delta_vs_joint_normalized"]
        ) - tolerance:
            wins += 1
        elif float(right_row["delta_vs_joint_normalized"]) > float(
            left_row["delta_vs_joint_normalized"]
        ) + tolerance:
            losses += 1
        else:
            ties += 1
    return {
        "left_method": left,
        "right_method": right,
        "right_strict_win_cell_count": wins,
        "left_strict_win_cell_count": losses,
        "tie_cell_count": ties,
        "mean_raw_energy_change_right_minus_left": statistics.fmean(deltas),
    }


def aggregate(
    model_checks: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    summaries = summarize(comparisons)
    summary = {row["method"]: row for row in summaries}
    tolerance = float(config["benchmark_gate"]["normalized_energy_tolerance"])
    ablations = [
        pairwise_ablation(comparisons, SOURCE_METHOD, NEW_METHODS[0], tolerance),
        pairwise_ablation(comparisons, NEW_METHODS[0], NEW_METHODS[1], tolerance),
        pairwise_ablation(comparisons, NEW_METHODS[1], NEW_METHODS[2], tolerance),
        pairwise_ablation(comparisons, NEW_METHODS[2], NEW_METHODS[3], tolerance),
    ]
    gate = config["benchmark_gate"]
    cold = summary[NEW_METHODS[0]]
    final = summary[NEW_METHODS[3]]
    source = summary[SOURCE_METHOD]
    encoding_passed = bool(
        len(model_checks) == int(gate["required_cqm_model_count"])
        and all(row["cqm_hash_match"] for row in model_checks)
        and max(row["energy_residual"] for row in model_checks)
        <= float(gate["maximum_energy_encoding_residual"])
    )
    cold_passed = bool(
        cold["exact_frontier_match_rate"]
        >= float(gate["minimum_cold_adjacent_exact_match_rate"])
        and cold["joint_competitive_fraction"]
        >= float(gate["minimum_cold_adjacent_joint_competitive_fraction"])
    )
    warm_passed = bool(
        final["exact_frontier_match_rate"]
        >= float(gate["minimum_warm_pt_exact_match_rate"])
        and final["joint_competitive_fraction"]
        >= float(gate["minimum_warm_pt_joint_competitive_fraction"])
        and final["frontier_competitive_fraction"]
        >= float(gate["minimum_warm_pt_frontier_competitive_fraction"])
        and final["strict_frontier_improvement_cell_count"]
        >= int(gate["minimum_warm_pt_frontier_improvement_cell_count"])
        and final["joint_competitive_fraction"]
        - source["joint_competitive_fraction"]
        >= float(gate["minimum_joint_competitive_gain_over_stage75_cold"])
    )
    return {
        "encoding_summary": {
            "cqm_model_count": len(model_checks),
            "cqm_hash_match_count": sum(row["cqm_hash_match"] for row in model_checks),
            "maximum_energy_identity_residual": max(
                row["energy_residual"] for row in model_checks
            ),
            "maximum_logical_variable_count": max(
                row["logical_variable_count"] for row in model_checks
            ),
            "objective_or_constraint_changes": 0,
        },
        "benchmark_summary": {
            "comparison_cell_count": len(comparisons) // len(METHOD_ORDER),
            "method_cell_count": len(comparisons),
            "solver_trial_count": len(trials),
            "method_count": len(METHOD_ORDER),
            "exact_frontier_cell_count": int(
                summary[SOURCE_METHOD]["exact_frontier_cell_count"]
            ),
        },
        "method_summaries": summaries,
        "ablation_summary": ablations,
        "route_gate": {
            "stage75_cqm_identity_preserved": encoding_passed,
            "cold_start_sampler_repair_passed": cold_passed,
            "warm_start_parallel_tempering_fidelity_passed": warm_passed,
        },
        "decision": {
            "objective_redesign_required": False,
            "explicit_variable_k_cqm_remains_frozen": encoding_passed,
            "standalone_cold_sampler_ready": bool(encoding_passed and cold_passed),
            "local_warm_start_hardware_shaped_emulation_authorized": bool(
                encoding_passed and warm_passed
            ),
            "cloud_cqm_execution_authorized": False,
            "direct_qpu_execution_authorized": False,
            "quantum_scaling_claim_authorized": False,
            "quantum_advantage_claim_authorized": False,
        },
    }


def report_text(result: dict[str, Any]) -> str:
    summaries = {row["method"]: row for row in result["method_summaries"]}
    source = summaries[SOURCE_METHOD]
    cold = summaries[NEW_METHODS[0]]
    warm = summaries[NEW_METHODS[3]]
    return f"""# Stage76 variable-k sampler repair

## Frozen scientific object

Stage76 reuses all 80 Stage75 CQMs without changing the pair objective, reward levels, allowed budgets, conditional quality thresholds, or three explicit constraints. Only initialization, cardinality moves, and temperature coupling change.

## Matched-budget comparison

Each stochastic method uses 8 repeats and 8192 state proposals per cell. Parallel-tempering exchanges reuse already evaluated energies and are reported separately. Classical warm-start construction cost is explicit and is not presented as a quantum speed advantage.

| Method | Exact-frontier match | Joint-tabu competitive | Frontier competitive | Frontier improvements |
|---|---:|---:|---:|---:|
| Stage75 cold global annealing | {source['exact_frontier_match_rate']:.3f} | {source['joint_competitive_fraction']:.3f} | {source['frontier_competitive_fraction']:.3f} | {source['strict_frontier_improvement_cell_count']} |
| Stage76 cold adjacent annealing | {cold['exact_frontier_match_rate']:.3f} | {cold['joint_competitive_fraction']:.3f} | {cold['frontier_competitive_fraction']:.3f} | {cold['strict_frontier_improvement_cell_count']} |
| Stage76 frontier-warm parallel tempering | {warm['exact_frontier_match_rate']:.3f} | {warm['joint_competitive_fraction']:.3f} | {warm['frontier_competitive_fraction']:.3f} | {warm['strict_frontier_improvement_cell_count']} |

## Decision

- Stage75 CQM identity preserved: `{result['route_gate']['stage75_cqm_identity_preserved']}`.
- Standalone cold-start sampler repaired: `{result['route_gate']['cold_start_sampler_repair_passed']}`.
- Warm-start parallel-tempering fidelity gate: `{result['route_gate']['warm_start_parallel_tempering_fidelity_passed']}`.
- Local warm-start hardware-shaped emulation authorized: `{result['decision']['local_warm_start_hardware_shaped_emulation_authorized']}`.
- Cloud CQM, direct QPU, quantum scaling, and quantum advantage claims: `False`.

The frontier-warm result is a hybrid refinement route. It inherits classical fixed-k frontier information and therefore cannot be used as evidence of standalone quantum superiority.
"""


def compute(config: dict[str, Any], root: Path) -> dict[str, Any]:
    implementation = {
        key: s75.verified(root, value, key)
        for key, value in config["implementation"].items()
    }
    inputs = {
        key: s75.verified(root, value, key)
        for key, value in config["inputs"].items()
    }
    stage75_result = s75.read_json(inputs["stage75_result"])
    stage75_audit = s75.read_json(inputs["stage75_audit"])
    if not stage75_result["decision"]["explicit_variable_k_cqm_freeze_authorized"]:
        raise ValueError("Stage76 requires the frozen Stage75 explicit CQM")
    if stage75_audit["status"] != "stage75_explicit_variable_k_cqm_independent_audit_ok":
        raise ValueError("Stage76 requires the Stage75 independent audit")
    source = s75.read_json(inputs["stage72_model_record"])
    stage75_models = s75.read_json(inputs["stage75_model_record"])
    frozen_hashes = {
        (row["target_id"], int(row["outer_fold"]), float(row["reward_quantile"])): row["cqm_sha256"]
        for row in stage75_models["models"]
    }
    workloads = s75.read_csv(inputs["stage74_workload_metrics"])
    source_comparisons = s75.read_csv(inputs["stage74_cell_comparison"])
    source_stage74_trials = s75.read_csv(inputs["stage74_solver_trials"])
    source_stage75_trials = s75.read_csv(inputs["stage75_solver_trials"])
    models = [s75.load_model(record) for record in source["models"]]
    trial_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    model_checks: list[dict[str, Any]] = []
    repeats = int(config["solver_protocol"]["stochastic_repeats"])
    base_seed = int(config["solver_protocol"]["seed_base"])
    cell_index = 0
    for model in models:
        frontiers = s75.source_frontiers(
            model,
            workloads,
            source_comparisons,
            source_stage74_trials,
            config["variable_k_cqm"]["quality_regime"],
        )
        for quantile in config["variable_k_cqm"]["reward_quantiles"]:
            reward = s75.reward_order_statistic(model, float(quantile))["reward"]
            cell = {
                "model": model,
                "frontiers": frontiers,
                "reward_quantile": float(quantile),
                "reward": float(reward),
                "solver_protocol": config["solver_protocol"],
            }
            key = (
                str(model["record"]["target_id"]),
                int(model["record"]["outer_fold"]),
                float(quantile),
            )
            canonical_hash = s75.canonical_sha256(
                s75.cqm_canonical(model, frontiers, float(reward))
            )
            frontier_subset, frontier_energy, _ = s75.fixed_candidate(
                cell, "reference"
            )
            cqm = s75.build_cqm(model, frontiers, float(reward))
            sample = s75.assignment(model, frontiers, frontier_subset)
            residual = abs(float(cqm.objective.energy(sample)) - frontier_energy)
            model_checks.append(
                {
                    "target_id": key[0],
                    "outer_fold": key[1],
                    "reward_quantile": key[2],
                    "cqm_sha256": canonical_hash,
                    "cqm_hash_match": canonical_hash == frozen_hashes[key],
                    "energy_residual": residual,
                    "assignment_feasible": bool(cqm.check_feasible(sample)),
                    "logical_variable_count": int(cqm.num_variables()),
                    "quadratic_coupler_count": len(model["pairs"]),
                    "explicit_constraint_count": len(cqm.constraints),
                }
            )
            local_rows = frozen_source_rows(cell, source_stage75_trials)
            method_specs = (
                (NEW_METHODS[0], "single", "cold_random"),
                (NEW_METHODS[1], "single", "decomposed_deterministic"),
                (NEW_METHODS[2], "pt", "deterministic"),
                (NEW_METHODS[3], "pt", "reference"),
            )
            for method_index, (method, family, source_name) in enumerate(
                method_specs, start=1
            ):
                for repeat in range(repeats):
                    seed = base_seed + cell_index * 100_000 + method_index * 1_000 + repeat
                    rng = np.random.default_rng(seed)
                    solved = (
                        adjacent_annealing(cell, rng, source_name)
                        if family == "single"
                        else parallel_tempering(cell, rng, source_name)
                    )
                    local_rows.append(
                        trial_row(cell, method, repeat, seed, solved)
                    )
            references = reference_values(cell, source_stage75_trials)
            scale = normalization(cell)
            for row in local_rows:
                row["frozen_frontier_energy"] = references["frontier_energy"]
                row["stage75_joint_tabu_energy"] = references["joint_energy"]
                row["delta_vs_frontier_normalized"] = (
                    float(row["solution_variable_energy"])
                    - references["frontier_energy"]
                ) / scale
                row["delta_vs_joint_normalized"] = (
                    float(row["solution_variable_energy"])
                    - references["joint_energy"]
                ) / scale
            trial_rows.extend(local_rows)
            comparison_rows.extend(
                compare_methods(
                    cell,
                    local_rows,
                    references,
                    float(config["benchmark_gate"]["normalized_energy_tolerance"]),
                )
            )
            cell_index += 1
        print(
            json.dumps(
                {
                    "target_id": model["record"]["target_id"],
                    "outer_fold": model["record"]["outer_fold"],
                    "cells_completed": cell_index,
                    "trials_completed": len(trial_rows),
                }
            ),
            flush=True,
        )
    aggregate_value = aggregate(
        model_checks, trial_rows, comparison_rows, config
    )
    output_paths = {
        key: root / str(value) for key, value in config["outputs"].items()
    }
    s75.write_csv(output_paths["cqm_identity_csv"], model_checks)
    s75.write_csv(output_paths["solver_trials_csv"], trial_rows)
    s75.write_csv(output_paths["cell_method_comparison_csv"], comparison_rows)
    s75.write_csv(output_paths["method_summary_csv"], aggregate_value["method_summaries"])
    payload = {
        **aggregate_value,
        "cqm_identity_sha256": s75.sha256(output_paths["cqm_identity_csv"]),
        "solver_trials_sha256": s75.sha256(output_paths["solver_trials_csv"]),
        "cell_method_comparison_sha256": s75.sha256(
            output_paths["cell_method_comparison_csv"]
        ),
        "method_summary_sha256": s75.sha256(output_paths["method_summary_csv"]),
    }
    result = {
        "schema_version": "1.0",
        "status": "stage76_variable_k_sampler_repair_complete",
        "experiment_class": "post-hoc solver-mechanism ablation on frozen Stage75 CQMs",
        "config": s75.descriptor(
            root, root / "configs/stage76_variable_k_sampler_repair.json"
        ),
        "implementation": {
            key: s75.descriptor(root, path) for key, path in implementation.items()
        },
        "inputs": {
            key: s75.descriptor(root, path) for key, path in inputs.items()
        },
        "runtime": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": np.__version__,
            "dimod": dimod.__version__,
            "wall_clock_used_for_decision": False,
        },
        **aggregate_value,
        "data_boundary": {
            "historical_development_targets_read": 4,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "cloud_cqm_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "analysis_payload_sha256": s75.canonical_sha256(payload),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    result["outputs"] = {
        key: s75.descriptor(root, output_paths[key])
        for key in (
            "cqm_identity_csv",
            "solver_trials_csv",
            "cell_method_comparison_csv",
            "method_summary_csv",
        )
    }
    s75.write_json(output_paths["result_json"], result)
    output_paths["report_md"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["report_md"].write_text(
        report_text(result), encoding="utf-8", newline="\n"
    )
    result["outputs"]["report_md"] = s75.descriptor(
        root, output_paths["report_md"]
    )
    s75.write_json(output_paths["result_json"], result)
    return result


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    expected = root / "configs/stage76_variable_k_sampler_repair.json"
    if config_path != expected.resolve():
        raise ValueError("Stage76 must run from its frozen repository config")
    config = s75.read_json(config_path)
    result_path = root / str(config["outputs"]["result_json"])
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Stage76 result exists: {result_path}")
    result = compute(config, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage76_variable_k_sampler_repair.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
