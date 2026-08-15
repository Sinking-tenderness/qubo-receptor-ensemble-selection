"""Independently replay and audit the Stage76 sampler-repair benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import scripts.audit_stage75_explicit_variable_k_cqm as a75
except ImportError:
    import audit_stage75_explicit_variable_k_cqm as a75


SOURCE_METHOD = "stage75_cold_global_annealing_reference"
METHODS = (
    SOURCE_METHOD,
    "cold_adjacent_variable_annealing",
    "decomposed_warm_adjacent_annealing",
    "decomposed_warm_parallel_tempering",
    "frontier_warm_parallel_tempering",
)
TOLERANCE = 1e-12


def scale(cell: dict[str, Any]) -> float:
    return cell["model"]["scale"] * math.comb(max(cell["frontiers"]), 2)


def neighboring_candidate(
    cell: dict[str, Any], state: tuple[int, ...], rng: np.random.Generator
) -> tuple[int, ...]:
    protocol = cell["protocol"]
    if rng.random() < float(protocol["swap_move_probability"]):
        removed = state[int(rng.integers(0, len(state)))]
        chosen = set(state)
        available = [
            index
            for index in range(cell["model"]["count"])
            if index not in chosen
        ]
        added = available[int(rng.integers(0, len(available)))]
        return tuple(sorted((chosen - {removed}) | {added}))
    allowed = tuple(cell["frontiers"])
    position = allowed.index(len(state))
    adjacent = []
    if position:
        adjacent.append(position - 1)
    if position < len(allowed) - 1:
        adjacent.append(position + 1)
    target = allowed[adjacent[int(rng.integers(0, len(adjacent)))]]
    chosen = set(state)
    if target > len(state):
        pool = np.asarray(
            [
                index
                for index in range(cell["model"]["count"])
                if index not in chosen
            ],
            dtype=int,
        )
        values = rng.choice(pool, target - len(state), replace=False)
        chosen.update(int(value) for value in values)
    else:
        values = rng.choice(
            np.asarray(state, dtype=int), len(state) - target, replace=False
        )
        chosen.difference_update(int(value) for value in values)
    return tuple(sorted(chosen))


def fixed_best(cell: dict[str, Any], field: str) -> tuple[int, ...]:
    return min(
        (value[f"{field}_subset"] for value in cell["frontiers"].values()),
        key=lambda subset: (
            a75.energy(cell["model"], subset, cell["reward"]), subset
        ),
    )


def initial_state(
    cell: dict[str, Any], source: str, rng: np.random.Generator
) -> tuple[tuple[int, ...], int, int, int]:
    if source == "cold_random":
        state, attempts, fallback = a75.random_start(
            cell,
            rng,
            int(cell["protocol"]["maximum_initialization_attempts"]),
        )
        return state, attempts, int(fallback), 1
    if source == "decomposed_deterministic":
        return fixed_best(cell, "deterministic"), 0, 0, len(cell["frontiers"])
    raise ValueError(f"unexpected Stage76 audit start source: {source}")


def replay_single(
    cell: dict[str, Any], rng: np.random.Generator, source: str
) -> dict[str, Any]:
    protocol = cell["protocol"]
    budget = int(protocol["proposal_budget"])
    beta_minimum, beta_maximum = (
        float(value) for value in protocol["annealing_beta_range"]
    )
    current, attempts, fallback, candidates = initial_state(cell, source, rng)
    current_energy = a75.energy(cell["model"], current, cell["reward"])
    best, best_energy = current, current_energy
    feasible = evaluations = accepted = 0
    evaluations = 1
    denominator = scale(cell)
    for step in range(budget):
        proposed = neighboring_candidate(cell, current, rng)
        if not a75.valid(cell, proposed):
            continue
        feasible += 1
        evaluations += 1
        proposed_energy = a75.energy(cell["model"], proposed, cell["reward"])
        delta = (proposed_energy - current_energy) / denominator
        fraction = step / max(1, budget - 1)
        beta = beta_minimum * (beta_maximum / beta_minimum) ** fraction
        if delta <= 0.0 or rng.random() < math.exp(-beta * delta):
            current, current_energy = proposed, proposed_energy
            accepted += 1
            if (current_energy, current) < (best_energy, best):
                best, best_energy = current, current_energy
    return {
        "subset": best,
        "proposal_count": budget,
        "feasible_proposal_count": feasible,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "exchange_attempt_count": 0,
        "accepted_exchange_count": 0,
        "initialization_attempt_count": attempts,
        "initialization_fallback_count": fallback,
        "initialization_candidate_count": candidates,
        "initialization_source": source,
    }


def replica_initial_states(
    cell: dict[str, Any], source: str, count: int
) -> list[tuple[int, ...]]:
    states = [
        tuple(value[f"{source}_subset"])
        for value in cell["frontiers"].values()
    ]
    states.sort(
        key=lambda subset: (
            a75.energy(cell["model"], subset, cell["reward"]), subset
        ),
        reverse=True,
    )
    best = states[-1]
    while len(states) < count:
        states.insert(0, best)
    if len(states) > count:
        states = states[-count:]
        states.sort(
            key=lambda subset: (
                a75.energy(cell["model"], subset, cell["reward"]), subset
            ),
            reverse=True,
        )
    return states


def replay_tempering(
    cell: dict[str, Any], rng: np.random.Generator, source: str
) -> dict[str, Any]:
    protocol = cell["protocol"]
    budget = int(protocol["proposal_budget"])
    count = int(protocol["replica_count"])
    interval = int(protocol["exchange_interval_sweeps"])
    beta_low, beta_high = (
        float(value) for value in protocol["parallel_tempering_beta_range"]
    )
    betas = np.geomspace(beta_low, beta_high, count)
    states = replica_initial_states(cell, source, count)
    energies = [
        a75.energy(cell["model"], state, cell["reward"]) for state in states
    ]
    best_index = min(range(count), key=lambda index: (energies[index], states[index]))
    best, best_energy = states[best_index], energies[best_index]
    feasible = accepted = exchanges = accepted_exchanges = 0
    evaluations = count
    denominator = scale(cell)
    sweep = 0
    for step in range(budget):
        replica = step % count
        proposed = neighboring_candidate(cell, states[replica], rng)
        if a75.valid(cell, proposed):
            feasible += 1
            evaluations += 1
            proposed_energy = a75.energy(
                cell["model"], proposed, cell["reward"]
            )
            delta = (proposed_energy - energies[replica]) / denominator
            if delta <= 0.0 or rng.random() < math.exp(-betas[replica] * delta):
                states[replica] = proposed
                energies[replica] = proposed_energy
                accepted += 1
                if (proposed_energy, proposed) < (best_energy, best):
                    best, best_energy = proposed, proposed_energy
        if (step + 1) % count == 0:
            sweep += 1
            if sweep % interval == 0:
                parity = (sweep // interval) % 2
                for left in range(parity, count - 1, 2):
                    right = left + 1
                    exchanges += 1
                    log_probability = (
                        (betas[left] - betas[right])
                        * (energies[left] - energies[right])
                        / denominator
                    )
                    if log_probability >= 0.0 or rng.random() < math.exp(
                        log_probability
                    ):
                        states[left], states[right] = states[right], states[left]
                        energies[left], energies[right] = energies[right], energies[left]
                        accepted_exchanges += 1
    return {
        "subset": best,
        "proposal_count": budget,
        "feasible_proposal_count": feasible,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "exchange_attempt_count": exchanges,
        "accepted_exchange_count": accepted_exchanges,
        "initialization_attempt_count": 0,
        "initialization_fallback_count": 0,
        "initialization_candidate_count": len(cell["frontiers"]),
        "initialization_source": (
            "decomposed_deterministic_fronts"
            if source == "deterministic"
            else "frozen_fixed_k_frontiers"
        ),
    }


def source_solutions(
    cell: dict[str, Any], source_rows: list[dict[str, str]]
) -> list[tuple[int, int, dict[str, Any]]]:
    target = str(cell["model"]["record"]["target_id"])
    fold = int(cell["model"]["record"]["outer_fold"])
    quantile = float(cell["quantile"])
    rows = [
        row
        for row in source_rows
        if row["target_id"] == target
        and int(row["outer_fold"]) == fold
        and math.isclose(float(row["reward_quantile"]), quantile)
        and row["method"] == "constraint_native_variable_annealing"
    ]
    if len(rows) != 8:
        raise ValueError("Stage76 audit source baseline count differs")
    return [
        (
            int(row["repeat"]),
            int(row["seed"]),
            {
                "subset": a75.parse_subset(cell["model"], row["solution_subset"]),
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
            },
        )
        for row in rows
    ]


def references(
    cell: dict[str, Any], source_rows: list[dict[str, str]]
) -> dict[str, Any]:
    frontier = fixed_best(cell, "reference")
    frontier_energy = a75.energy(cell["model"], frontier, cell["reward"])
    target = str(cell["model"]["record"]["target_id"])
    fold = int(cell["model"]["record"]["outer_fold"])
    quantile = float(cell["quantile"])
    candidates = [
        row
        for row in source_rows
        if row["target_id"] == target
        and int(row["outer_fold"]) == fold
        and math.isclose(float(row["reward_quantile"]), quantile)
        and row["method"] == "budgeted_variable_tabu"
    ]
    best = min(
        candidates,
        key=lambda row: (
            float(row["solution_variable_energy"]), row["solution_subset"]
        ),
    )
    joint = a75.parse_subset(cell["model"], best["solution_subset"])
    return {
        "frontier_energy": frontier_energy,
        "joint_energy": a75.energy(cell["model"], joint, cell["reward"]),
        "exact": all(
            value["reference_type"] == "exact_enumeration"
            for value in cell["frontiers"].values()
        ),
    }


def compare_trial(
    observed: dict[str, str],
    cell: dict[str, Any],
    method: str,
    repeat: int,
    seed: int,
    solved: dict[str, Any],
) -> dict[str, Any]:
    if observed["method"] != method or int(observed["repeat"]) != repeat:
        raise ValueError("Stage76 audit trial identity differs")
    if int(observed["seed"]) != seed:
        raise ValueError("Stage76 audit seed differs")
    subset = tuple(solved["subset"])
    if observed["solution_subset"] != a75.label(cell["model"], subset):
        raise ValueError("Stage76 audit trial subset differs")
    for field in (
        "proposal_count",
        "feasible_proposal_count",
        "objective_evaluation_count",
        "accepted_move_count",
        "exchange_attempt_count",
        "accepted_exchange_count",
        "initialization_attempt_count",
        "initialization_fallback_count",
        "initialization_candidate_count",
    ):
        if int(observed[field]) != int(solved[field]):
            raise ValueError(f"Stage76 audit trial {field} differs")
    if observed["initialization_source"] != solved["initialization_source"]:
        raise ValueError("Stage76 audit initialization source differs")
    value = a75.energy(cell["model"], subset, cell["reward"])
    a75.close(observed["solution_variable_energy"], value, "Stage76 trial energy")
    a75.close(
        observed["solution_raw_pair_objective"],
        a75.raw_objective(cell["model"], subset),
        "Stage76 trial raw objective",
    )
    if int(observed["selected_k"]) != len(subset):
        raise ValueError("Stage76 audit selected k differs")
    if int(observed["solution_deficit"]) != a75.deficit(cell["model"], subset):
        raise ValueError("Stage76 audit solution deficit differs")
    if not a75.valid(cell, subset):
        raise ValueError("Stage76 audit solution is infeasible")
    return {
        "target_id": str(cell["model"]["record"]["target_id"]),
        "outer_fold": int(cell["model"]["record"]["outer_fold"]),
        "reward_quantile": float(cell["quantile"]),
        "method": method,
        "repeat": repeat,
        "subset": subset,
        "label": a75.label(cell["model"], subset),
        "energy": value,
        "feasible_proposal_count": int(solved["feasible_proposal_count"]),
        "proposal_count": int(solved["proposal_count"]),
        "accepted_exchange_count": int(solved["accepted_exchange_count"]),
        "exchange_attempt_count": int(solved["exchange_attempt_count"]),
    }


def expected_comparisons(
    trials: list[dict[str, Any]],
    cells: dict[tuple[str, int, float], dict[str, Any]],
    reference_map: dict[tuple[str, int, float], dict[str, Any]],
    tolerance: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        grouped[(row["target_id"], row["outer_fold"], row["reward_quantile"], row["method"])].append(row)
    output = []
    for cell_key, cell in sorted(cells.items()):
        refs = reference_map[cell_key]
        denominator = scale(cell)
        for method in METHODS:
            rows = grouped[cell_key + (method,)]
            best = min(rows, key=lambda row: (row["energy"], row["label"]))
            delta_frontier = (best["energy"] - refs["frontier_energy"]) / denominator
            delta_joint = (best["energy"] - refs["joint_energy"]) / denominator
            output.append(
                {
                    "target_id": cell_key[0],
                    "outer_fold": cell_key[1],
                    "reward_quantile": cell_key[2],
                    "method": method,
                    "trial_count": len(rows),
                    "best_repeat": best["repeat"],
                    "best_selected_k": len(best["subset"]),
                    "best_subset": best["label"],
                    "best_energy": best["energy"],
                    "mean_energy": statistics.fmean(row["energy"] for row in rows),
                    "frozen_frontier_energy": refs["frontier_energy"],
                    "stage75_joint_tabu_energy": refs["joint_energy"],
                    "delta_vs_frontier_normalized": delta_frontier,
                    "delta_vs_joint_normalized": delta_joint,
                    "within_frontier_tolerance": delta_frontier <= tolerance,
                    "within_joint_tolerance": delta_joint <= tolerance,
                    "strict_frontier_improvement": delta_frontier < -tolerance,
                    "exact_frontier_available": refs["exact"],
                    "exact_frontier_match": refs["exact"] and best["energy"] <= refs["frontier_energy"] + TOLERANCE,
                    "unique_solution_count": len({row["label"] for row in rows}),
                    "mean_feasible_proposal_fraction": statistics.fmean(row["feasible_proposal_count"] / max(1, row["proposal_count"]) for row in rows),
                    "mean_exchange_acceptance_fraction": statistics.fmean(row["accepted_exchange_count"] / max(1, row["exchange_attempt_count"]) for row in rows),
                }
            )
    return output


def compare_csv_rows(observed: list[dict[str, str]], expected: list[dict[str, Any]], key_fields: tuple[str, ...]) -> None:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        values = []
        for field in key_fields:
            value: Any = row[field]
            if field in {"outer_fold", "best_repeat"}:
                value = int(value)
            elif field == "reward_quantile":
                value = float(value)
            values.append(value)
        return tuple(values)
    mapping = {key(row): row for row in observed}
    if len(mapping) != len(expected):
        raise ValueError("Stage76 audit CSV row count differs")
    for item in expected:
        row = mapping[key(item)]
        for field, value in item.items():
            if field in key_fields:
                continue
            if isinstance(value, bool):
                if a75.truth(row[field]) != value:
                    raise ValueError(f"Stage76 audit boolean differs: {field}")
            elif isinstance(value, float):
                a75.close(row[field], value, f"Stage76 CSV {field}")
            elif isinstance(value, int):
                if int(row[field]) != value:
                    raise ValueError(f"Stage76 audit integer differs: {field}")
            elif str(row[field]) != str(value):
                raise ValueError(f"Stage76 audit value differs: {field}")


def method_summaries(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact_count = sum(row["exact_frontier_available"] for row in comparisons if row["method"] == SOURCE_METHOD)
    output = []
    for method in METHODS:
        rows = [row for row in comparisons if row["method"] == method]
        exact = [row for row in rows if row["exact_frontier_available"]]
        output.append(
            {
                "method": method,
                "cell_count": len(rows),
                "trial_count": sum(row["trial_count"] for row in rows),
                "exact_frontier_cell_count": exact_count,
                "exact_frontier_match_rate": statistics.fmean(row["exact_frontier_match"] for row in exact),
                "joint_competitive_fraction": statistics.fmean(row["within_joint_tolerance"] for row in rows),
                "frontier_competitive_fraction": statistics.fmean(row["within_frontier_tolerance"] for row in rows),
                "strict_frontier_improvement_cell_count": sum(row["strict_frontier_improvement"] for row in rows),
                "mean_delta_vs_joint_normalized": statistics.fmean(row["delta_vs_joint_normalized"] for row in rows),
                "mean_delta_vs_frontier_normalized": statistics.fmean(row["delta_vs_frontier_normalized"] for row in rows),
                "mean_unique_solution_count": statistics.fmean(row["unique_solution_count"] for row in rows),
                "mean_feasible_proposal_fraction": statistics.fmean(row["mean_feasible_proposal_fraction"] for row in rows),
                "mean_exchange_acceptance_fraction": statistics.fmean(row["mean_exchange_acceptance_fraction"] for row in rows),
            }
        )
    return output


def ablation(comparisons: list[dict[str, Any]], left: str, right: str, tolerance: float) -> dict[str, Any]:
    lookup = {(row["target_id"], row["outer_fold"], row["reward_quantile"], row["method"]): row for row in comparisons}
    changes = []
    wins = losses = ties = 0
    cells = sorted({key[:3] for key in lookup})
    for cell in cells:
        left_row = lookup[cell + (left,)]
        right_row = lookup[cell + (right,)]
        changes.append(right_row["best_energy"] - left_row["best_energy"])
        difference = right_row["delta_vs_joint_normalized"] - left_row["delta_vs_joint_normalized"]
        if difference < -tolerance:
            wins += 1
        elif difference > tolerance:
            losses += 1
        else:
            ties += 1
    return {
        "left_method": left,
        "right_method": right,
        "right_strict_win_cell_count": wins,
        "left_strict_win_cell_count": losses,
        "tie_cell_count": ties,
        "mean_raw_energy_change_right_minus_left": statistics.fmean(changes),
    }


def aggregate(model_checks: list[dict[str, Any]], trials: list[dict[str, Any]], comparisons: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    summaries = method_summaries(comparisons)
    summary = {row["method"]: row for row in summaries}
    tolerance = float(config["benchmark_gate"]["normalized_energy_tolerance"])
    ablations = [ablation(comparisons, left, right, tolerance) for left, right in zip(METHODS, METHODS[1:])]
    gate = config["benchmark_gate"]
    source = summary[METHODS[0]]
    cold = summary[METHODS[1]]
    final = summary[METHODS[-1]]
    encoding = len(model_checks) == int(gate["required_cqm_model_count"]) and all(row["hash"] and row["feasible"] for row in model_checks) and max(row["residual"] for row in model_checks) <= float(gate["maximum_energy_encoding_residual"])
    cold_gate = cold["exact_frontier_match_rate"] >= float(gate["minimum_cold_adjacent_exact_match_rate"]) and cold["joint_competitive_fraction"] >= float(gate["minimum_cold_adjacent_joint_competitive_fraction"])
    warm_gate = final["exact_frontier_match_rate"] >= float(gate["minimum_warm_pt_exact_match_rate"]) and final["joint_competitive_fraction"] >= float(gate["minimum_warm_pt_joint_competitive_fraction"]) and final["frontier_competitive_fraction"] >= float(gate["minimum_warm_pt_frontier_competitive_fraction"]) and final["strict_frontier_improvement_cell_count"] >= int(gate["minimum_warm_pt_frontier_improvement_cell_count"]) and final["joint_competitive_fraction"] - source["joint_competitive_fraction"] >= float(gate["minimum_joint_competitive_gain_over_stage75_cold"])
    return {
        "encoding_summary": {
            "cqm_model_count": len(model_checks),
            "cqm_hash_match_count": sum(row["hash"] for row in model_checks),
            "maximum_energy_identity_residual": max(row["residual"] for row in model_checks),
            "maximum_logical_variable_count": max(row["variables"] for row in model_checks),
            "objective_or_constraint_changes": 0,
        },
        "benchmark_summary": {
            "comparison_cell_count": len(comparisons) // len(METHODS),
            "method_cell_count": len(comparisons),
            "solver_trial_count": len(trials),
            "method_count": len(METHODS),
            "exact_frontier_cell_count": source["exact_frontier_cell_count"],
        },
        "method_summaries": summaries,
        "ablation_summary": ablations,
        "route_gate": {
            "stage75_cqm_identity_preserved": bool(encoding),
            "cold_start_sampler_repair_passed": bool(cold_gate),
            "warm_start_parallel_tempering_fidelity_passed": bool(warm_gate),
        },
        "decision": {
            "objective_redesign_required": False,
            "explicit_variable_k_cqm_remains_frozen": bool(encoding),
            "standalone_cold_sampler_ready": bool(encoding and cold_gate),
            "local_warm_start_hardware_shaped_emulation_authorized": bool(encoding and warm_gate),
            "cloud_cqm_execution_authorized": False,
            "direct_qpu_execution_authorized": False,
            "quantum_scaling_claim_authorized": False,
            "quantum_advantage_claim_authorized": False,
        },
    }


def run(config_path: Path, result_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    result_path = result_path.resolve()
    output_path = output_path.resolve()
    if config_path != (root / "configs/stage76_variable_k_sampler_repair.json").resolve():
        raise ValueError("Stage76 audit config path differs")
    if result_path != (root / "data/stage76_variable_k_sampler_repair_result.json").resolve():
        raise ValueError("Stage76 audit result path differs")
    if output_path != (root / "data/stage76_variable_k_sampler_repair_audit.json").resolve():
        raise ValueError("Stage76 audit output path differs")
    config = a75.read_json(config_path)
    result = a75.read_json(result_path)
    for key, value in config["implementation"].items():
        a75.checked(root, value, f"implementation {key}")
    inputs = {key: a75.checked(root, value, f"input {key}") for key, value in config["inputs"].items()}
    paths = {key: root / str(value) for key, value in config["outputs"].items()}
    observed_identity = a75.read_csv(paths["cqm_identity_csv"])
    observed_trials = a75.read_csv(paths["solver_trials_csv"])
    observed_comparisons = a75.read_csv(paths["cell_method_comparison_csv"])
    observed_summaries = a75.read_csv(paths["method_summary_csv"])
    trial_map = {(row["target_id"], int(row["outer_fold"]), float(row["reward_quantile"]), row["method"], int(row["repeat"])): row for row in observed_trials}
    identity_map = {(row["target_id"], int(row["outer_fold"]), float(row["reward_quantile"])): row for row in observed_identity}
    source = a75.read_json(inputs["stage72_model_record"])
    frozen_models = a75.read_json(inputs["stage75_model_record"])
    frozen_hashes = {(row["target_id"], int(row["outer_fold"]), float(row["reward_quantile"])): row["cqm_sha256"] for row in frozen_models["models"]}
    workloads = a75.read_csv(inputs["stage74_workload_metrics"])
    source_comparisons = a75.read_csv(inputs["stage74_cell_comparison"])
    stage74_trials = a75.read_csv(inputs["stage74_solver_trials"])
    stage75_trials = a75.read_csv(inputs["stage75_solver_trials"])
    models = [a75.rebuild(record) for record in source["models"]]
    expected_trials: list[dict[str, Any]] = []
    cells: dict[tuple[str, int, float], dict[str, Any]] = {}
    reference_map: dict[tuple[str, int, float], dict[str, Any]] = {}
    model_checks = []
    base_seed = int(config["solver_protocol"]["seed_base"])
    repeats = int(config["solver_protocol"]["stochastic_repeats"])
    cell_index = 0
    for model in models:
        frontiers = a75.recover_frontiers(model, workloads, source_comparisons, stage74_trials, config["variable_k_cqm"]["quality_regime"])
        for quantile in config["variable_k_cqm"]["reward_quantiles"]:
            quantile = float(quantile)
            _, reward = a75.reward_record(model, quantile)
            cell = {"model": model, "frontiers": frontiers, "quantile": quantile, "reward": reward, "protocol": config["solver_protocol"]}
            key = (str(model["record"]["target_id"]), int(model["record"]["outer_fold"]), quantile)
            cells[key] = cell
            refs = references(cell, stage75_trials)
            reference_map[key] = refs
            canonical = a75.canonical_sha256(a75.canonical_model(model, frontiers, reward))
            frontier = fixed_best(cell, "reference")
            cqm = a75.make_cqm(model, frontiers, reward)
            sample = a75.sample_for(model, frontiers, frontier)
            residual = abs(float(cqm.objective.energy(sample)) - refs["frontier_energy"])
            observed = identity_map[key]
            if observed["cqm_sha256"] != canonical or a75.truth(observed["cqm_hash_match"]) != (canonical == frozen_hashes[key]):
                raise ValueError("Stage76 audit CQM identity differs")
            a75.close(observed["energy_residual"], residual, "Stage76 CQM residual")
            model_checks.append({"hash": canonical == frozen_hashes[key], "residual": residual, "feasible": bool(cqm.check_feasible(sample)), "variables": int(cqm.num_variables())})
            for repeat, seed, solved in source_solutions(cell, stage75_trials):
                row = trial_map[key + (SOURCE_METHOD, repeat)]
                expected_trials.append(compare_trial(row, cell, SOURCE_METHOD, repeat, seed, solved))
            specs = (
                (METHODS[1], "single", "cold_random"),
                (METHODS[2], "single", "decomposed_deterministic"),
                (METHODS[3], "pt", "deterministic"),
                (METHODS[4], "pt", "reference"),
            )
            for method_index, (method, family, source_name) in enumerate(specs, start=1):
                for repeat in range(repeats):
                    seed = base_seed + cell_index * 100_000 + method_index * 1_000 + repeat
                    rng = np.random.default_rng(seed)
                    solved = replay_single(cell, rng, source_name) if family == "single" else replay_tempering(cell, rng, source_name)
                    row = trial_map[key + (method, repeat)]
                    expected_trials.append(compare_trial(row, cell, method, repeat, seed, solved))
            cell_index += 1
        print(json.dumps({"audit_target": model["record"]["target_id"], "audit_fold": model["record"]["outer_fold"], "trials_replayed": len(expected_trials)}), flush=True)
    if len(trial_map) != len(expected_trials) or len(identity_map) != cell_index:
        raise ValueError("Stage76 audit output counts differ")
    comparisons = expected_comparisons(expected_trials, cells, reference_map, float(config["benchmark_gate"]["normalized_energy_tolerance"]))
    compare_csv_rows(observed_comparisons, comparisons, ("target_id", "outer_fold", "reward_quantile", "method"))
    summaries = method_summaries(comparisons)
    compare_csv_rows(observed_summaries, summaries, ("method",))
    aggregate_value = aggregate(model_checks, expected_trials, comparisons, config)
    for section in ("encoding_summary", "benchmark_summary"):
        for field, value in aggregate_value[section].items():
            observed = result[section][field]
            if isinstance(value, float):
                a75.close(observed, value, f"Stage76 result {section}.{field}")
            elif observed != value:
                raise ValueError(f"Stage76 result {section}.{field} differs")
    if result["method_summaries"] != aggregate_value["method_summaries"] or result["ablation_summary"] != aggregate_value["ablation_summary"]:
        raise ValueError("Stage76 audit aggregate table differs")
    if result["route_gate"] != aggregate_value["route_gate"] or any(result["decision"][key] != value for key, value in aggregate_value["decision"].items()):
        raise ValueError("Stage76 audit decision differs")
    payload = {
        **aggregate_value,
        "cqm_identity_sha256": a75.sha256(paths["cqm_identity_csv"]),
        "solver_trials_sha256": a75.sha256(paths["solver_trials_csv"]),
        "cell_method_comparison_sha256": a75.sha256(paths["cell_method_comparison_csv"]),
        "method_summary_sha256": a75.sha256(paths["method_summary_csv"]),
    }
    if a75.canonical_sha256(payload) != result["analysis_payload_sha256"]:
        raise ValueError("Stage76 audit payload differs")
    boundary = {"historical_development_targets_read": 4, "fresh_validation_rows_read": 0, "locked_test_rows_read": 0, "new_docking_jobs": 0, "cloud_cqm_jobs": 0, "quantum_hardware_jobs": 0}
    if result["data_boundary"] != boundary:
        raise ValueError("Stage76 audit data boundary differs")
    audit = {
        "schema_version": "1.0",
        "status": "stage76_variable_k_sampler_repair_independent_audit_ok",
        "source_result": a75.descriptor(root, result_path),
        "stage75_cqm_models_independently_rebuilt": cell_index,
        "solver_trials_deterministically_replayed": len(expected_trials),
        "cell_method_comparisons_independently_recomputed": len(comparisons),
        "method_summaries_independently_recomputed": len(summaries),
        **aggregate_value["route_gate"],
        **aggregate_value["decision"],
        "data_boundary": boundary,
    }
    a75.write_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage76_variable_k_sampler_repair.json"))
    parser.add_argument("--result", type=Path, default=Path("data/stage76_variable_k_sampler_repair_result.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/stage76_variable_k_sampler_repair_audit.json"))
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
