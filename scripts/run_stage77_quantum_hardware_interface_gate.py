"""Adjudicate stable quantum-hardware interfaces for the frozen variable-k CQM."""

from __future__ import annotations

import argparse
import functools
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import dimod
import numpy as np
from dwave.embedding.zephyr import find_clique_embedding
from dwave.samplers import SimulatedAnnealingSampler

try:
    import scripts.run_stage75_explicit_variable_k_cqm as s75
except ImportError:
    import run_stage75_explicit_variable_k_cqm as s75


TOLERANCE = 1e-12


def source_cells(
    config: dict[str, Any], inputs: dict[str, Path]
) -> list[dict[str, Any]]:
    source = s75.read_json(inputs["stage72_model_record"])
    workloads = s75.read_csv(inputs["stage74_workload_metrics"])
    comparisons = s75.read_csv(inputs["stage74_cell_comparison"])
    trials = s75.read_csv(inputs["stage74_solver_trials"])
    cells: list[dict[str, Any]] = []
    for record in source["models"]:
        model = s75.load_model(record)
        frontiers = s75.source_frontiers(
            model,
            workloads,
            comparisons,
            trials,
            config["frozen_cqm"]["quality_regime"],
        )
        for quantile in config["frozen_cqm"]["reward_quantiles"]:
            reward_record = s75.reward_order_statistic(model, float(quantile))
            reward = float(reward_record["reward"])
            cells.append(
                {
                    "model": model,
                    "frontiers": frontiers,
                    "reward_quantile": float(quantile),
                    "reward": reward,
                    "reward_order_statistic_index": int(
                        reward_record["order_statistic_index"]
                    ),
                    "cqm": s75.build_cqm(model, frontiers, reward),
                    "cqm_sha256": s75.canonical_sha256(
                        s75.cqm_canonical(model, frontiers, reward)
                    ),
                }
            )
    return cells


@functools.lru_cache(maxsize=None)
def zephyr_clique_metrics(variable_count: int, tile_count: int) -> dict[str, int]:
    if variable_count <= 0:
        return {
            "ideal_zephyr_physical_qubit_count": 0,
            "ideal_zephyr_minimum_chain_length": 0,
            "ideal_zephyr_maximum_chain_length": 0,
        }
    embedding = find_clique_embedding(variable_count, tile_count)
    if len(embedding) != variable_count:
        raise ValueError(
            f"ideal Zephyr clique embedding incomplete: {len(embedding)}/{variable_count}"
        )
    lengths = [len(chain) for chain in embedding.values()]
    return {
        "ideal_zephyr_physical_qubit_count": sum(lengths),
        "ideal_zephyr_minimum_chain_length": min(lengths),
        "ideal_zephyr_maximum_chain_length": max(lengths),
    }


def nonzero_biases(bqm: dimod.BinaryQuadraticModel) -> list[float]:
    values = [float(value) for value in bqm.linear.values()]
    values.extend(float(value) for value in bqm.quadratic.values())
    return [abs(value) for value in values if abs(value) > 1e-15]


def minimum_signed_bits(signal_ratio: float) -> int:
    if not signal_ratio > 0.0:
        return 65
    for bits in range(2, 65):
        positive_levels = 2 ** (bits - 1) - 1
        if signal_ratio * positive_levels >= 0.5:
            return bits
    return 65


def direct_encoding_rows(
    cells: list[dict[str, Any]],
    frozen_hashes: dict[tuple[str, int, float], str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    penalty_factor = float(config["direct_cqm_to_bqm"]["penalty_factor"])
    tile_count = int(config["hardware_proxy"]["ideal_zephyr_tile_count"])
    for index, cell in enumerate(cells, start=1):
        model = cell["model"]
        key = (
            str(model["record"]["target_id"]),
            int(model["record"]["outer_fold"]),
            float(cell["reward_quantile"]),
        )
        objective_biases = [
            abs(float(value)) for value in cell["cqm"].objective.quadratic.values()
        ]
        objective_scale = max(objective_biases + [1e-12])
        lagrange_multiplier = penalty_factor * objective_scale
        bqm, _ = dimod.cqm_to_bqm(
            cell["cqm"], lagrange_multiplier=lagrange_multiplier
        )
        biases = nonzero_biases(bqm)
        maximum_bias = max(biases)
        minimum_bias = min(biases)
        original_variables = set(cell["cqm"].variables)
        auxiliary_count = sum(
            variable not in original_variables for variable in bqm.variables
        )
        signal_ratio = objective_scale / maximum_bias
        rows.append(
            {
                "target_id": key[0],
                "outer_fold": key[1],
                "reward_quantile": key[2],
                "candidate_variable_count": model["count"],
                "cqm_variable_count": cell["cqm"].num_variables(),
                "cqm_constraint_count": len(cell["cqm"].constraints),
                "cqm_sha256": cell["cqm_sha256"],
                "cqm_hash_match": cell["cqm_sha256"] == frozen_hashes[key],
                "lagrange_multiplier": lagrange_multiplier,
                "direct_bqm_variable_count": bqm.num_variables,
                "direct_bqm_auxiliary_variable_count": auxiliary_count,
                "direct_bqm_interaction_count": bqm.num_interactions,
                "direct_bqm_density": (
                    2.0
                    * bqm.num_interactions
                    / (bqm.num_variables * (bqm.num_variables - 1))
                ),
                "minimum_absolute_bqm_bias": minimum_bias,
                "maximum_absolute_bqm_bias": maximum_bias,
                "coefficient_dynamic_range": maximum_bias / minimum_bias,
                "maximum_objective_bias": objective_scale,
                "maximum_objective_signal_ratio_after_scaling": signal_ratio,
                "minimum_signed_bits_to_resolve_maximum_objective_bias": minimum_signed_bits(
                    signal_ratio
                ),
                **zephyr_clique_metrics(bqm.num_variables, tile_count),
            }
        )
        if index % 20 == 0:
            print(
                json.dumps(
                    {
                        "stage77_direct_models_completed": index,
                        "stage77_direct_models_total": len(cells),
                    }
                ),
                flush=True,
            )
    return rows


def subset_after_moves(
    warm_subset: tuple[int, ...],
    moves: list[dict[str, Any]],
    selected: Iterable[int],
) -> tuple[tuple[int, ...], bool]:
    selected = tuple(selected)
    removed = [int(moves[index]["removed_index"]) for index in selected]
    added = [int(moves[index]["added_index"]) for index in selected]
    conflict_free = len(removed) == len(set(removed)) and len(added) == len(
        set(added)
    )
    if not conflict_free:
        return warm_subset, False
    subset = tuple(sorted((set(warm_subset) - set(removed)) | set(added)))
    return subset, True


def eligible_moves(
    model: dict[str, Any],
    reward: float,
    warm_subset: tuple[int, ...],
    maximum_move_count: int,
) -> tuple[list[dict[str, Any]], int]:
    chosen = set(warm_subset)
    warm_energy = s75.variable_energy(model, warm_subset, reward)
    warm_deficit = s75.subset_deficit(model, warm_subset)
    moves: list[dict[str, Any]] = []
    for removed in warm_subset:
        for added in range(model["count"]):
            if added in chosen:
                continue
            subset = tuple(sorted((chosen - {removed}) | {added}))
            deficit_delta = s75.subset_deficit(model, subset) - warm_deficit
            if deficit_delta > 0:
                continue
            moves.append(
                {
                    "removed_index": int(removed),
                    "added_index": int(added),
                    "deficit_delta": int(deficit_delta),
                    "energy_delta": float(
                        s75.variable_energy(model, subset, reward) - warm_energy
                    ),
                    "subset": subset,
                }
            )
    moves.sort(
        key=lambda row: (
            row["energy_delta"],
            row["deficit_delta"],
            row["removed_index"],
            row["added_index"],
        )
    )
    eligible_count = len(moves)
    return moves[:maximum_move_count], eligible_count


def build_swap_bqm(
    cell: dict[str, Any], k: int, config: dict[str, Any]
) -> dict[str, Any]:
    model = cell["model"]
    frontier = cell["frontiers"][k]
    warm_subset = tuple(frontier["reference_subset"])
    warm_energy = s75.variable_energy(model, warm_subset, cell["reward"])
    warm_deficit = s75.subset_deficit(model, warm_subset)
    maximum_moves = int(config["local_swap_bqm"]["maximum_move_variable_count"])
    moves, eligible_count = eligible_moves(
        model, cell["reward"], warm_subset, maximum_moves
    )
    bqm = dimod.BinaryQuadraticModel({}, {}, warm_energy, dimod.BINARY)
    variable_names = [f"m{index:03d}" for index in range(len(moves))]
    for name, move in zip(variable_names, moves):
        bqm.add_variable(name, float(move["energy_delta"]))
    conflict_margin = (
        float(config["local_swap_bqm"]["conflict_margin_pair_scale"])
        * float(model["pair_scale"])
    )
    chosen = set(warm_subset)
    conflict_pairs = 0
    nonconflicting_pairs = 0
    for left in range(len(moves)):
        for right in range(left + 1, len(moves)):
            left_move = moves[left]
            right_move = moves[right]
            conflict = (
                left_move["removed_index"] == right_move["removed_index"]
                or left_move["added_index"] == right_move["added_index"]
            )
            if conflict:
                conflict_pairs += 1
                interaction = max(
                    conflict_margin,
                    -float(left_move["energy_delta"])
                    - float(right_move["energy_delta"])
                    + conflict_margin,
                )
            else:
                nonconflicting_pairs += 1
                subset = tuple(
                    sorted(
                        (
                            chosen
                            - {
                                int(left_move["removed_index"]),
                                int(right_move["removed_index"]),
                            }
                        )
                        | {
                            int(left_move["added_index"]),
                            int(right_move["added_index"]),
                        }
                    )
                )
                interaction = (
                    s75.variable_energy(model, subset, cell["reward"])
                    - warm_energy
                    - float(left_move["energy_delta"])
                    - float(right_move["energy_delta"])
                )
            if abs(interaction) > 1e-15:
                bqm.add_interaction(
                    variable_names[left], variable_names[right], interaction
                )
    return {
        "bqm": bqm,
        "moves": moves,
        "variable_names": variable_names,
        "warm_subset": warm_subset,
        "warm_energy": warm_energy,
        "warm_deficit": warm_deficit,
        "quality_threshold": int(frontier["quality_threshold"]),
        "eligible_move_count": eligible_count,
        "conflict_pair_count": conflict_pairs,
        "nonconflicting_pair_count": nonconflicting_pairs,
        "conflict_margin": conflict_margin,
    }


def local_identity_metrics(
    cell: dict[str, Any], local: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    bqm = local["bqm"]
    moves = local["moves"]
    names = local["variable_names"]
    zero = {name: 0 for name in names}
    residuals = [abs(float(bqm.energy(zero)) - local["warm_energy"])]
    conflict_margins: list[float] = []
    for index, move in enumerate(moves):
        sample = dict(zero)
        sample[names[index]] = 1
        residuals.append(
            abs(
                float(bqm.energy(sample))
                - (
                    local["warm_energy"] + float(move["energy_delta"])
                )
            )
        )
    for left in range(len(moves)):
        for right in range(left + 1, len(moves)):
            sample = dict(zero)
            sample[names[left]] = 1
            sample[names[right]] = 1
            subset, conflict_free = subset_after_moves(
                local["warm_subset"], moves, (left, right)
            )
            if conflict_free:
                residuals.append(
                    abs(
                        float(bqm.energy(sample))
                        - s75.variable_energy(
                            cell["model"], subset, cell["reward"]
                        )
                    )
                )
            else:
                conflict_margins.append(
                    float(bqm.energy(sample)) - local["warm_energy"]
                )
    biases = nonzero_biases(bqm)
    full_scale = max(biases + [1e-12])
    bits = int(config["hardware_proxy"]["coefficient_precision_bits"])
    levels = 2 ** (bits - 1) - 1
    retained = statistics.fmean(
        round((float(value) / full_scale) * levels) != 0
        for value in [*bqm.linear.values(), *bqm.quadratic.values()]
        if abs(float(value)) > 1e-15
    )
    best_move_delta = min(
        [0.0] + [float(move["energy_delta"]) for move in moves]
    )
    resolution_margin = float(
        config["hardware_proxy"]["resolvable_improvement_lsb_margin"]
    )
    resolvable = (
        -best_move_delta / full_scale >= resolution_margin / levels
        if best_move_delta < -TOLERANCE
        else False
    )
    return {
        "maximum_objective_identity_residual": max(residuals),
        "minimum_conflicting_pair_energy_margin": min(
            conflict_margins + [local["conflict_margin"]]
        ),
        "minimum_absolute_bqm_bias": min(biases),
        "maximum_absolute_bqm_bias": full_scale,
        "coefficient_dynamic_range": full_scale / min(biases),
        "quantized_bias_retention_fraction": retained,
        "best_single_move_energy_delta": best_move_delta,
        "improving_single_move_available": best_move_delta < -TOLERANCE,
        "hardware_resolvable_single_move_improvement": resolvable,
    }


def local_subproblems(
    cells: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    tile_count = int(config["hardware_proxy"]["ideal_zephyr_tile_count"])
    index = 0
    for cell in cells:
        model = cell["model"]
        for k in cell["frontiers"]:
            local = build_swap_bqm(cell, int(k), config)
            identity = local_identity_metrics(cell, local, config)
            row = {
                "subproblem_index": index,
                "target_id": model["record"]["target_id"],
                "outer_fold": int(model["record"]["outer_fold"]),
                "reward_quantile": float(cell["reward_quantile"]),
                "reward_value": float(cell["reward"]),
                "k": int(k),
                "warm_subset": s75.subset_name(model, local["warm_subset"]),
                "warm_energy": float(local["warm_energy"]),
                "warm_deficit": int(local["warm_deficit"]),
                "quality_threshold": int(local["quality_threshold"]),
                "eligible_quality_nonincreasing_move_count": int(
                    local["eligible_move_count"]
                ),
                "encoded_move_variable_count": len(local["moves"]),
                "improving_single_move_count": sum(
                    float(move["energy_delta"]) < -TOLERANCE
                    for move in local["moves"]
                ),
                "bqm_interaction_count": local["bqm"].num_interactions,
                "bqm_density": (
                    2.0
                    * local["bqm"].num_interactions
                    / (len(local["moves"]) * (len(local["moves"]) - 1))
                    if len(local["moves"]) > 1
                    else 0.0
                ),
                "conflict_pair_count": int(local["conflict_pair_count"]),
                "nonconflicting_pair_count": int(
                    local["nonconflicting_pair_count"]
                ),
                "all_encoded_moves_quality_nonincreasing": all(
                    int(move["deficit_delta"]) <= 0 for move in local["moves"]
                ),
                **identity,
                **zephyr_clique_metrics(len(local["moves"]), tile_count),
            }
            local["cell"] = cell
            local["row"] = row
            records.append(local)
            rows.append(row)
            index += 1
    return records, rows


def hardware_proxy_bqm(
    bqm: dimod.BinaryQuadraticModel,
    condition: dict[str, Any],
    rng: np.random.Generator,
) -> dimod.BinaryQuadraticModel:
    values = nonzero_biases(bqm)
    full_scale = max(values + [1e-12])
    output = bqm.copy()
    noise = float(condition["coefficient_noise_sigma"])
    bits = condition["coefficient_precision_bits"]
    levels = 2 ** (int(bits) - 1) - 1 if bits is not None else None
    for variable in output.variables:
        value = float(output.linear[variable]) / full_scale
        if noise:
            value += float(rng.normal(0.0, noise))
        value = max(-1.0, min(1.0, value))
        if levels is not None:
            value = round(value * levels) / levels
        output.set_linear(variable, value)
    for left, right in list(output.quadratic):
        value = float(output.quadratic[(left, right)]) / full_scale
        if noise:
            value += float(rng.normal(0.0, noise))
        value = max(-1.0, min(1.0, value))
        if levels is not None:
            value = round(value * levels) / levels
        output.set_quadratic(left, right, value)
    output.offset = 0.0
    return output


def emulation_rows(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    protocol = config["warm_start_emulation"]
    sampler = SimulatedAnnealingSampler()
    repeats = int(protocol["repeats"])
    reads = int(protocol["reads_per_repeat"])
    beta_schedule = np.concatenate(
        (
            np.geomspace(
                float(protocol["reverse_beta_start"]),
                float(protocol["reverse_beta_minimum"]),
                int(protocol["reverse_schedule_steps"]),
            ),
            np.geomspace(
                float(protocol["reverse_beta_minimum"]),
                float(protocol["forward_beta_end"]),
                int(protocol["forward_schedule_steps"]),
            ),
        )
    )
    output: list[dict[str, Any]] = []
    base_seed = int(protocol["seed_base"])
    for record_index, record in enumerate(records):
        cell = record["cell"]
        model = cell["model"]
        row = record["row"]
        warm_energy = float(record["warm_energy"])
        initial = {name: 0 for name in record["variable_names"]}
        for condition_index, condition in enumerate(protocol["conditions"]):
            for repeat in range(repeats):
                seed = (
                    base_seed
                    + record_index * 100_000
                    + condition_index * 1_000
                    + repeat
                )
                rng = np.random.default_rng(seed)
                proxy = hardware_proxy_bqm(record["bqm"], condition, rng)
                samples = sampler.sample(
                    proxy,
                    num_reads=reads,
                    num_sweeps=len(beta_schedule),
                    beta_schedule_type="custom",
                    beta_schedule=beta_schedule,
                    initial_states=[initial],
                    initial_states_generator="tile",
                    seed=seed,
                )
                feasible_count = 0
                conflict_count = 0
                unique: set[tuple[int, ...]] = set()
                best_energy = warm_energy
                best_subset = record["warm_subset"]
                for sample in samples.samples():
                    selected = [
                        index
                        for index, name in enumerate(record["variable_names"])
                        if int(sample[name]) == 1
                    ]
                    subset, conflict_free = subset_after_moves(
                        record["warm_subset"], record["moves"], selected
                    )
                    if not conflict_free:
                        conflict_count += 1
                        continue
                    feasible = s75.valid(cell, subset)
                    if not feasible:
                        continue
                    feasible_count += 1
                    unique.add(subset)
                    energy = s75.variable_energy(model, subset, cell["reward"])
                    if (energy, subset) < (best_energy, best_subset):
                        best_energy = energy
                        best_subset = subset
                output.append(
                    {
                        "subproblem_index": row["subproblem_index"],
                        "target_id": row["target_id"],
                        "outer_fold": row["outer_fold"],
                        "reward_quantile": row["reward_quantile"],
                        "k": row["k"],
                        "condition": condition["id"],
                        "coefficient_precision_bits": (
                            "float64"
                            if condition["coefficient_precision_bits"] is None
                            else int(condition["coefficient_precision_bits"])
                        ),
                        "coefficient_noise_sigma": float(
                            condition["coefficient_noise_sigma"]
                        ),
                        "repeat": repeat,
                        "seed": seed,
                        "read_count": reads,
                        "feasible_read_count": feasible_count,
                        "conflict_read_count": conflict_count,
                        "unique_feasible_subset_count": len(unique),
                        "warm_energy": warm_energy,
                        "best_true_energy_with_warm_guard": best_energy,
                        "best_true_energy_gain": warm_energy - best_energy,
                        "best_subset_with_warm_guard": s75.subset_name(
                            model, best_subset
                        ),
                        "strict_improvement_recovered": best_energy
                        < warm_energy - TOLERANCE,
                        "hardware_resolvable_opportunity": bool(
                            row[
                                "hardware_resolvable_single_move_improvement"
                            ]
                        ),
                        "hardware_resolvable_opportunity_recovered": bool(
                            row[
                                "hardware_resolvable_single_move_improvement"
                            ]
                            and best_energy < warm_energy - TOLERANCE
                        ),
                        "warm_guard_nonworse": best_energy
                        <= warm_energy + TOLERANCE,
                    }
                )
        if (record_index + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "stage77_local_subproblems_emulated": record_index + 1,
                        "stage77_local_subproblems_total": len(records),
                        "stage77_emulation_runs": len(output),
                    }
                ),
                flush=True,
            )
    return output


def summarize_emulation(
    rows: list[dict[str, Any]], local_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    output: list[dict[str, Any]] = []
    opportunity_count = sum(
        bool(row["hardware_resolvable_single_move_improvement"])
        for row in local_rows
    )
    for condition, selected in by_condition.items():
        subproblem_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            subproblem_groups[int(row["subproblem_index"])].append(row)
        output.append(
            {
                "condition": condition,
                "subproblem_count": len(subproblem_groups),
                "run_count": len(selected),
                "read_count": sum(int(row["read_count"]) for row in selected),
                "feasible_read_fraction": sum(
                    int(row["feasible_read_count"]) for row in selected
                )
                / sum(int(row["read_count"]) for row in selected),
                "conflict_read_fraction": sum(
                    int(row["conflict_read_count"]) for row in selected
                )
                / sum(int(row["read_count"]) for row in selected),
                "strict_improvement_subproblem_count": sum(
                    any(
                        bool(row["strict_improvement_recovered"])
                        for row in group
                    )
                    for group in subproblem_groups.values()
                ),
                "hardware_resolvable_opportunity_subproblem_count": opportunity_count,
                "hardware_resolvable_opportunity_recovered_count": sum(
                    any(
                        bool(
                            row[
                                "hardware_resolvable_opportunity_recovered"
                            ]
                        )
                        for row in group
                    )
                    for group in subproblem_groups.values()
                ),
                "hardware_resolvable_opportunity_recovery_fraction": (
                    sum(
                        any(
                            bool(
                                row[
                                    "hardware_resolvable_opportunity_recovered"
                                ]
                            )
                            for row in group
                        )
                        for group in subproblem_groups.values()
                    )
                    / opportunity_count
                    if opportunity_count
                    else 1.0
                ),
                "warm_guard_nonworse_run_fraction": statistics.fmean(
                    bool(row["warm_guard_nonworse"]) for row in selected
                ),
            }
        )
    return output


def summarize(
    direct_rows: list[dict[str, Any]],
    local_rows: list[dict[str, Any]],
    emulation_summary: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    direct_gate = config["benchmark_gate"]["direct_full_qpu"]
    local_gate = config["benchmark_gate"]["local_reverse_annealing"]
    summary_by_condition = {
        row["condition"]: row for row in emulation_summary
    }
    clean = summary_by_condition["float64_clean"]
    stress = summary_by_condition["q10_noise_1p00"]
    direct_summary = {
        "cqm_model_count": len(direct_rows),
        "cqm_hash_match_count": sum(
            bool(row["cqm_hash_match"]) for row in direct_rows
        ),
        "maximum_direct_bqm_variable_count": max(
            int(row["direct_bqm_variable_count"]) for row in direct_rows
        ),
        "maximum_direct_bqm_auxiliary_variable_count": max(
            int(row["direct_bqm_auxiliary_variable_count"])
            for row in direct_rows
        ),
        "maximum_direct_bqm_interaction_count": max(
            int(row["direct_bqm_interaction_count"]) for row in direct_rows
        ),
        "maximum_coefficient_dynamic_range": max(
            float(row["coefficient_dynamic_range"]) for row in direct_rows
        ),
        "maximum_required_signed_precision_bits": max(
            int(row["minimum_signed_bits_to_resolve_maximum_objective_bias"])
            for row in direct_rows
        ),
        "maximum_ideal_zephyr_physical_qubit_count": max(
            int(row["ideal_zephyr_physical_qubit_count"])
            for row in direct_rows
        ),
        "maximum_ideal_zephyr_chain_length": max(
            int(row["ideal_zephyr_maximum_chain_length"])
            for row in direct_rows
        ),
    }
    target_counts = Counter(str(row["target_id"]) for row in local_rows)
    unique_improving_instances = {
        (str(row["target_id"]), int(row["outer_fold"]), int(row["k"]))
        for row in local_rows
        if bool(row["improving_single_move_available"])
    }
    unique_resolvable_instances = {
        (str(row["target_id"]), int(row["outer_fold"]), int(row["k"]))
        for row in local_rows
        if bool(row["hardware_resolvable_single_move_improvement"])
    }
    local_summary = {
        "subproblem_count": len(local_rows),
        "subproblem_count_by_target": dict(sorted(target_counts.items())),
        "minimum_eligible_move_count": min(
            int(row["eligible_quality_nonincreasing_move_count"])
            for row in local_rows
        ),
        "mean_encoded_move_variable_count": statistics.fmean(
            int(row["encoded_move_variable_count"]) for row in local_rows
        ),
        "maximum_encoded_move_variable_count": max(
            int(row["encoded_move_variable_count"]) for row in local_rows
        ),
        "maximum_bqm_interaction_count": max(
            int(row["bqm_interaction_count"]) for row in local_rows
        ),
        "maximum_objective_identity_residual": max(
            float(row["maximum_objective_identity_residual"])
            for row in local_rows
        ),
        "minimum_conflicting_pair_energy_margin": min(
            float(row["minimum_conflicting_pair_energy_margin"])
            for row in local_rows
        ),
        "minimum_q10_bias_retention_fraction": min(
            float(row["quantized_bias_retention_fraction"])
            for row in local_rows
        ),
        "mean_q10_bias_retention_fraction": statistics.fmean(
            float(row["quantized_bias_retention_fraction"])
            for row in local_rows
        ),
        "improving_single_move_subproblem_count": sum(
            bool(row["improving_single_move_available"])
            for row in local_rows
        ),
        "unique_improving_fixed_k_instance_count": len(
            unique_improving_instances
        ),
        "hardware_resolvable_improvement_subproblem_count": sum(
            bool(row["hardware_resolvable_single_move_improvement"])
            for row in local_rows
        ),
        "unique_hardware_resolvable_fixed_k_instance_count": len(
            unique_resolvable_instances
        ),
        "maximum_ideal_zephyr_physical_qubit_count": max(
            int(row["ideal_zephyr_physical_qubit_count"])
            for row in local_rows
        ),
        "maximum_ideal_zephyr_chain_length": max(
            int(row["ideal_zephyr_maximum_chain_length"])
            for row in local_rows
        ),
    }
    cqm_identity_passed = (
        direct_summary["cqm_model_count"]
        == int(config["experiment"]["required_cqm_model_count"])
        and direct_summary["cqm_hash_match_count"]
        == direct_summary["cqm_model_count"]
    )
    direct_embedding_passed = (
        direct_summary["maximum_ideal_zephyr_chain_length"]
        <= int(direct_gate["maximum_ideal_zephyr_chain_length"])
        and direct_summary["maximum_ideal_zephyr_physical_qubit_count"]
        <= int(direct_gate["maximum_ideal_zephyr_physical_qubit_count"])
    )
    direct_precision_passed = (
        direct_summary["maximum_required_signed_precision_bits"]
        <= int(direct_gate["maximum_required_signed_precision_bits"])
        and direct_summary["maximum_coefficient_dynamic_range"]
        <= float(direct_gate["maximum_coefficient_dynamic_range"])
    )
    local_identity_passed = (
        len(local_rows)
        == int(config["experiment"]["required_local_subproblem_count"])
        and local_summary["minimum_eligible_move_count"]
        >= int(local_gate["minimum_eligible_move_count"])
        and local_summary["maximum_objective_identity_residual"]
        <= float(local_gate["maximum_objective_identity_residual"])
        and local_summary["minimum_conflicting_pair_energy_margin"] > 0.0
        and all(
            bool(row["all_encoded_moves_quality_nonincreasing"])
            for row in local_rows
        )
    )
    local_topology_passed = (
        local_summary["maximum_encoded_move_variable_count"]
        <= int(local_gate["maximum_logical_variable_count"])
        and local_summary["maximum_ideal_zephyr_chain_length"]
        <= int(local_gate["maximum_ideal_zephyr_chain_length"])
        and local_summary["maximum_ideal_zephyr_physical_qubit_count"]
        <= int(local_gate["maximum_ideal_zephyr_physical_qubit_count"])
    )
    local_precision_passed = (
        local_summary["minimum_q10_bias_retention_fraction"]
        >= float(local_gate["minimum_q10_bias_retention_fraction"])
        and local_summary["hardware_resolvable_improvement_subproblem_count"]
        >= int(local_gate["minimum_hardware_resolvable_opportunity_count"])
        and local_summary[
            "unique_hardware_resolvable_fixed_k_instance_count"
        ]
        >= int(
            local_gate[
                "minimum_unique_hardware_resolvable_fixed_k_instance_count"
            ]
        )
    )
    clean_emulation_passed = (
        float(clean["feasible_read_fraction"])
        >= float(local_gate["minimum_clean_feasible_read_fraction"])
        and int(clean["strict_improvement_subproblem_count"])
        >= int(local_gate["minimum_clean_improvement_subproblem_count"])
    )
    stress_emulation_passed = (
        float(stress["feasible_read_fraction"])
        >= float(local_gate["minimum_stress_feasible_read_fraction"])
        and float(stress["hardware_resolvable_opportunity_recovery_fraction"])
        >= float(local_gate["minimum_stress_resolvable_recovery_fraction"])
        and float(stress["warm_guard_nonworse_run_fraction"])
        >= float(local_gate["minimum_warm_guard_nonworse_run_fraction"])
    )
    local_passed = all(
        (
            cqm_identity_passed,
            local_identity_passed,
            local_topology_passed,
            local_precision_passed,
            clean_emulation_passed,
            stress_emulation_passed,
        )
    )
    return {
        "direct_encoding_summary": direct_summary,
        "local_swap_bqm_summary": local_summary,
        "emulation_summaries": sorted(
            emulation_summary,
            key=lambda row: next(
                index
                for index, condition in enumerate(
                    config["warm_start_emulation"]["conditions"]
                )
                if condition["id"] == row["condition"]
            ),
        ),
        "route_gate": {
            "stage75_cqm_identity_preserved": cqm_identity_passed,
            "direct_full_bqm_ideal_embedding_passed": direct_embedding_passed,
            "direct_full_bqm_precision_passed": direct_precision_passed,
            "local_swap_bqm_identity_and_feasibility_passed": local_identity_passed,
            "local_swap_bqm_ideal_embedding_passed": local_topology_passed,
            "local_swap_bqm_precision_passed": local_precision_passed,
            "local_clean_warm_start_emulation_passed": clean_emulation_passed,
            "local_q10_one_percent_noise_stress_passed": stress_emulation_passed,
            "local_reverse_annealing_poc_gate_passed": local_passed,
        },
        "decision": {
            "frozen_variable_k_cqm_remains_scientific_model": True,
            "leap_hybrid_cqm_application_route_recommended": True,
            "full_direct_qpu_bqm_route_authorized": bool(
                cqm_identity_passed
                and direct_embedding_passed
                and direct_precision_passed
            ),
            "advantage2_local_reverse_annealing_poc_ready_for_budget_request": local_passed,
            "ibm_warm_start_qaoa_full_problem_route_authorized": False,
            "trapped_ion_full_problem_route_authorized": False,
            "neutral_atom_full_problem_route_authorized": False,
            "paid_cloud_execution_authorized": False,
            "paid_qpu_execution_authorized": False,
            "quantum_scaling_claim_authorized": False,
            "quantum_advantage_claim_authorized": False,
        },
    }


def report_text(result: dict[str, Any]) -> str:
    direct = result["direct_encoding_summary"]
    local = result["local_swap_bqm_summary"]
    stress = next(
        row
        for row in result["emulation_summaries"]
        if row["condition"] == "q10_noise_1p00"
    )
    hardware_rows = "\n".join(
        f"| {row['route']} | {row['native_interface']} | {row['status']} | {row['reason']} |"
        for row in result["hardware_route_review"]
    )
    return f"""# Stage77 Quantum-Hardware Interface Gate

## Question

Which current quantum-hardware form can stably accept the frozen dense variable-k receptor-selection problem without turning a local proof of concept into a quantum-advantage claim?

## Direct CQM-to-BQM Gate

- Frozen CQM identities recovered: `{direct['cqm_hash_match_count']}/{direct['cqm_model_count']}`.
- Largest converted BQM: `{direct['maximum_direct_bqm_variable_count']}` variables and `{direct['maximum_direct_bqm_interaction_count']}` interactions.
- Ideal Zephyr upper-bound embedding: `{direct['maximum_ideal_zephyr_physical_qubit_count']}` physical qubits, maximum chain length `{direct['maximum_ideal_zephyr_chain_length']}`.
- Worst coefficient dynamic range: `{direct['maximum_coefficient_dynamic_range']:.3e}`.
- Maximum signed precision required to retain even the largest objective bias: `{direct['maximum_required_signed_precision_bits']}` bits.
- Decision: full direct QPU BQM route authorized = `{result['decision']['full_direct_qpu_bqm_route_authorized']}`. Topology is not the limiting gate; penalty precision is.

## Feasibility-Preserving Local BQM

- Fixed-k frontier subproblems: `{local['subproblem_count']}` across four historical development targets.
- Move variables: mean `{local['mean_encoded_move_variable_count']:.2f}`, maximum `{local['maximum_encoded_move_variable_count']}`.
- Every encoded move is a fixed-k receptor swap with nonpositive quality-deficit change.
- Maximum objective identity residual: `{local['maximum_objective_identity_residual']:.3e}`.
- Ideal Zephyr upper-bound embedding: `{local['maximum_ideal_zephyr_physical_qubit_count']}` physical qubits, maximum chain length `{local['maximum_ideal_zephyr_chain_length']}`.
- Q10 retained-bias fraction: mean `{local['mean_q10_bias_retention_fraction']:.4f}`, minimum `{local['minimum_q10_bias_retention_fraction']:.4f}`.
- Local improving reward-cells: `{local['improving_single_move_subproblem_count']}` from `{local['unique_improving_fixed_k_instance_count']}` unique fixed-k instances; hardware-resolvable reward-cells at four Q10 LSBs: `{local['hardware_resolvable_improvement_subproblem_count']}` from `{local['unique_hardware_resolvable_fixed_k_instance_count']}` unique fixed-k instances. Reward quantiles are repeated evaluation cells, not independent physical BQMs, because a fixed-k reward shift is constant across subsets.
- Q10 plus 1% coefficient-noise proxy: feasible reads `{stress['feasible_read_fraction']:.4f}`, resolvable-opportunity recovery `{stress['hardware_resolvable_opportunity_recovery_fraction']:.4f}`, guarded nonworse runs `{stress['warm_guard_nonworse_run_fraction']:.4f}`.
- Decision: Advantage2 local reverse-annealing PoC ready for a budget request = `{result['decision']['advantage2_local_reverse_annealing_poc_ready_for_budget_request']}`.

## Hardware Route Review

| Route | Native interface | Status | Reason |
| --- | --- | --- | --- |
{hardware_rows}

## Claim Boundary

Stage77 is a local encoding, ideal-topology, quantization, and coefficient-noise proxy study on consumed historical development data. The Gaussian noise model is not a calibration model for a specific QPU. No cloud solver or QPU was contacted. Passing the local gate authorizes only a small, budget-capped reverse-annealing PoC with matched classical controls; it does not establish end-to-end speedup, quantum scaling, biological generalization, or quantum advantage.
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
    stage76_result = s75.read_json(inputs["stage76_result"])
    stage76_audit = s75.read_json(inputs["stage76_audit"])
    if not stage76_result["decision"][
        "local_warm_start_hardware_shaped_emulation_authorized"
    ]:
        raise ValueError("Stage77 requires the Stage76 warm-start route")
    if stage76_audit["status"] != "stage76_variable_k_sampler_repair_independent_audit_ok":
        raise ValueError("Stage77 requires the Stage76 independent audit")
    model_record = s75.read_json(inputs["stage75_model_record"])
    frozen_hashes = {
        (
            str(row["target_id"]),
            int(row["outer_fold"]),
            float(row["reward_quantile"]),
        ): str(row["cqm_sha256"])
        for row in model_record["models"]
    }
    cells = source_cells(config, inputs)
    direct_rows = direct_encoding_rows(cells, frozen_hashes, config)
    records, local_rows = local_subproblems(cells, config)
    trials = emulation_rows(records, config)
    emulation_summary = summarize_emulation(trials, local_rows)
    aggregate = summarize(
        direct_rows, local_rows, emulation_summary, config
    )
    output_paths = {
        key: root / str(value) for key, value in config["outputs"].items()
    }
    s75.write_csv(output_paths["direct_encoding_metrics_csv"], direct_rows)
    s75.write_csv(output_paths["local_swap_bqm_metrics_csv"], local_rows)
    s75.write_csv(output_paths["emulation_trials_csv"], trials)
    s75.write_csv(
        output_paths["emulation_summary_csv"],
        aggregate["emulation_summaries"],
    )
    s75.write_csv(
        output_paths["hardware_route_review_csv"],
        config["hardware_route_review"],
    )
    payload = {
        **aggregate,
        "direct_encoding_metrics_sha256": s75.sha256(
            output_paths["direct_encoding_metrics_csv"]
        ),
        "local_swap_bqm_metrics_sha256": s75.sha256(
            output_paths["local_swap_bqm_metrics_csv"]
        ),
        "emulation_trials_sha256": s75.sha256(
            output_paths["emulation_trials_csv"]
        ),
        "emulation_summary_sha256": s75.sha256(
            output_paths["emulation_summary_csv"]
        ),
        "hardware_route_review_sha256": s75.sha256(
            output_paths["hardware_route_review_csv"]
        ),
    }
    result = {
        "schema_version": "1.0",
        "status": "stage77_quantum_hardware_interface_gate_complete",
        "experiment_class": "post-hoc local hardware-interface and robustness adjudication",
        "config": s75.descriptor(
            root, root / "configs/stage77_quantum_hardware_interface_gate.json"
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
            "dwave_samplers": "1.7.0",
            "ideal_hardware_topology": "Zephyr Z16",
            "physical_qpu_calibration_used": False,
            "wall_clock_used_for_decision": False,
        },
        **aggregate,
        "hardware_route_review": config["hardware_route_review"],
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
            "direct_encoding_metrics_csv",
            "local_swap_bqm_metrics_csv",
            "emulation_trials_csv",
            "emulation_summary_csv",
            "hardware_route_review_csv",
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
    expected = root / "configs/stage77_quantum_hardware_interface_gate.json"
    if config_path != expected.resolve():
        raise ValueError("Stage77 must run from its frozen repository config")
    config = s75.read_json(config_path)
    result_path = root / str(config["outputs"]["result_json"])
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Stage77 result exists: {result_path}")
    result = compute(config, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage77_quantum_hardware_interface_gate.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
