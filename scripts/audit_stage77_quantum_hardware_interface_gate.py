"""Independently rebuild and replay the Stage77 hardware-interface gate."""

from __future__ import annotations

import argparse
import functools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import dimod
import numpy as np
from dwave.embedding.zephyr import find_clique_embedding
from dwave.samplers import SimulatedAnnealingSampler

try:
    import scripts.run_stage75_explicit_variable_k_cqm as s75
except ImportError:
    import run_stage75_explicit_variable_k_cqm as s75


TOLERANCE = 1e-12


def close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
    )


def boolean(value: Any) -> bool:
    return s75.truth(value)


@functools.lru_cache(maxsize=None)
def zephyr_metrics(variable_count: int, tile_count: int) -> tuple[int, int, int]:
    if variable_count <= 0:
        return 0, 0, 0
    embedding = find_clique_embedding(variable_count, tile_count)
    lengths = [len(chain) for chain in embedding.values()]
    if len(lengths) != variable_count:
        raise ValueError("Stage77 audit found an incomplete Zephyr embedding")
    return sum(lengths), min(lengths), max(lengths)


def cells(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    source = s75.read_json(root / config["inputs"]["stage72_model_record"]["path"])
    workloads = s75.read_csv(
        root / config["inputs"]["stage74_workload_metrics"]["path"]
    )
    comparisons = s75.read_csv(
        root / config["inputs"]["stage74_cell_comparison"]["path"]
    )
    trials = s75.read_csv(
        root / config["inputs"]["stage74_solver_trials"]["path"]
    )
    output: list[dict[str, Any]] = []
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
            reward = float(
                s75.reward_order_statistic(model, float(quantile))["reward"]
            )
            output.append(
                {
                    "model": model,
                    "frontiers": frontiers,
                    "reward_quantile": float(quantile),
                    "reward": reward,
                    "cqm": s75.build_cqm(model, frontiers, reward),
                    "cqm_sha256": s75.canonical_sha256(
                        s75.cqm_canonical(model, frontiers, reward)
                    ),
                }
            )
    return output


def audit_direct(
    source_cells: list[dict[str, Any]],
    rows: list[dict[str, str]],
    config: dict[str, Any],
) -> None:
    if len(rows) != len(source_cells):
        raise ValueError("Stage77 direct row count differs")
    factor = float(config["direct_cqm_to_bqm"]["penalty_factor"])
    tile_count = int(config["hardware_proxy"]["ideal_zephyr_tile_count"])
    for cell, row in zip(source_cells, rows):
        objective_scale = max(
            [abs(float(value)) for value in cell["cqm"].objective.quadratic.values()]
            + [1e-12]
        )
        bqm, _ = dimod.cqm_to_bqm(
            cell["cqm"], lagrange_multiplier=factor * objective_scale
        )
        biases = [
            abs(float(value))
            for value in [*bqm.linear.values(), *bqm.quadratic.values()]
            if abs(float(value)) > 1e-15
        ]
        original = set(cell["cqm"].variables)
        physical, minimum_chain, maximum_chain = zephyr_metrics(
            bqm.num_variables, tile_count
        )
        checks = {
            "cqm_sha256": cell["cqm_sha256"],
            "direct_bqm_variable_count": bqm.num_variables,
            "direct_bqm_auxiliary_variable_count": sum(
                variable not in original for variable in bqm.variables
            ),
            "direct_bqm_interaction_count": bqm.num_interactions,
            "ideal_zephyr_physical_qubit_count": physical,
            "ideal_zephyr_minimum_chain_length": minimum_chain,
            "ideal_zephyr_maximum_chain_length": maximum_chain,
        }
        for field, expected in checks.items():
            if str(row[field]) != str(expected):
                raise ValueError(f"Stage77 direct metric differs: {field}")
        numeric = {
            "minimum_absolute_bqm_bias": min(biases),
            "maximum_absolute_bqm_bias": max(biases),
            "coefficient_dynamic_range": max(biases) / min(biases),
            "maximum_objective_bias": objective_scale,
            "maximum_objective_signal_ratio_after_scaling": objective_scale
            / max(biases),
        }
        for field, expected in numeric.items():
            if not close(row[field], expected):
                raise ValueError(f"Stage77 direct numeric metric differs: {field}")


def make_local(
    cell: dict[str, Any], k: int, config: dict[str, Any]
) -> dict[str, Any]:
    model = cell["model"]
    frontier = cell["frontiers"][k]
    warm = tuple(frontier["reference_subset"])
    chosen = set(warm)
    warm_energy = s75.variable_energy(model, warm, cell["reward"])
    warm_deficit = s75.subset_deficit(model, warm)
    moves: list[dict[str, Any]] = []
    for removed in warm:
        for added in range(model["count"]):
            if added in chosen:
                continue
            subset = tuple(sorted((chosen - {removed}) | {added}))
            deficit_delta = s75.subset_deficit(model, subset) - warm_deficit
            if deficit_delta <= 0:
                moves.append(
                    {
                        "removed": int(removed),
                        "added": int(added),
                        "deficit_delta": int(deficit_delta),
                        "energy_delta": float(
                            s75.variable_energy(model, subset, cell["reward"])
                            - warm_energy
                        ),
                    }
                )
    moves.sort(
        key=lambda row: (
            row["energy_delta"],
            row["deficit_delta"],
            row["removed"],
            row["added"],
        )
    )
    eligible_count = len(moves)
    moves = moves[: int(config["local_swap_bqm"]["maximum_move_variable_count"])]
    names = [f"m{index:03d}" for index in range(len(moves))]
    bqm = dimod.BinaryQuadraticModel({}, {}, warm_energy, dimod.BINARY)
    for name, move in zip(names, moves):
        bqm.add_variable(name, float(move["energy_delta"]))
    margin = (
        float(config["local_swap_bqm"]["conflict_margin_pair_scale"])
        * float(model["pair_scale"])
    )
    conflict_pairs = 0
    nonconflicting_pairs = 0
    for left in range(len(moves)):
        for right in range(left + 1, len(moves)):
            left_move = moves[left]
            right_move = moves[right]
            conflict = (
                left_move["removed"] == right_move["removed"]
                or left_move["added"] == right_move["added"]
            )
            if conflict:
                conflict_pairs += 1
                interaction = max(
                    margin,
                    -float(left_move["energy_delta"])
                    - float(right_move["energy_delta"])
                    + margin,
                )
            else:
                nonconflicting_pairs += 1
                subset = tuple(
                    sorted(
                        (
                            chosen
                            - {left_move["removed"], right_move["removed"]}
                        )
                        | {left_move["added"], right_move["added"]}
                    )
                )
                interaction = (
                    s75.variable_energy(model, subset, cell["reward"])
                    - warm_energy
                    - float(left_move["energy_delta"])
                    - float(right_move["energy_delta"])
                )
            if abs(interaction) > 1e-15:
                bqm.add_interaction(names[left], names[right], interaction)
    return {
        "cell": cell,
        "k": k,
        "warm": warm,
        "warm_energy": warm_energy,
        "warm_deficit": warm_deficit,
        "quality_threshold": int(frontier["quality_threshold"]),
        "moves": moves,
        "names": names,
        "bqm": bqm,
        "eligible_count": eligible_count,
        "conflict_pairs": conflict_pairs,
        "nonconflicting_pairs": nonconflicting_pairs,
        "margin": margin,
    }


def decode(local: dict[str, Any], sample: dict[str, int]) -> tuple[tuple[int, ...], bool]:
    selected = [
        index
        for index, name in enumerate(local["names"])
        if int(sample[name]) == 1
    ]
    removed = [local["moves"][index]["removed"] for index in selected]
    added = [local["moves"][index]["added"] for index in selected]
    conflict_free = len(removed) == len(set(removed)) and len(added) == len(
        set(added)
    )
    if not conflict_free:
        return local["warm"], False
    return tuple(sorted((set(local["warm"]) - set(removed)) | set(added))), True


def audit_local(
    source_cells: list[dict[str, Any]],
    rows: list[dict[str, str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for cell in source_cells:
        for k in cell["frontiers"]:
            expected.append(make_local(cell, int(k), config))
    if len(expected) != len(rows):
        raise ValueError("Stage77 local subproblem count differs")
    tile_count = int(config["hardware_proxy"]["ideal_zephyr_tile_count"])
    bits = int(config["hardware_proxy"]["coefficient_precision_bits"])
    levels = 2 ** (bits - 1) - 1
    lsb_margin = float(
        config["hardware_proxy"]["resolvable_improvement_lsb_margin"]
    )
    for index, (local, row) in enumerate(zip(expected, rows)):
        bqm = local["bqm"]
        zero = {name: 0 for name in local["names"]}
        residuals = [abs(float(bqm.energy(zero)) - local["warm_energy"])]
        conflict_margins: list[float] = []
        for left, move in enumerate(local["moves"]):
            sample = dict(zero)
            sample[local["names"][left]] = 1
            residuals.append(
                abs(
                    float(bqm.energy(sample))
                    - local["warm_energy"]
                    - float(move["energy_delta"])
                )
            )
            for right in range(left + 1, len(local["moves"])):
                pair_sample = dict(zero)
                pair_sample[local["names"][left]] = 1
                pair_sample[local["names"][right]] = 1
                pair_subset, conflict_free = decode(local, pair_sample)
                if conflict_free:
                    residuals.append(
                        abs(
                            float(bqm.energy(pair_sample))
                            - s75.variable_energy(
                                local["cell"]["model"],
                                pair_subset,
                                local["cell"]["reward"],
                            )
                        )
                    )
                else:
                    conflict_margins.append(
                        float(bqm.energy(pair_sample)) - local["warm_energy"]
                    )
        biases = [
            abs(float(value))
            for value in [*bqm.linear.values(), *bqm.quadratic.values()]
            if abs(float(value)) > 1e-15
        ]
        maximum_bias = max(biases + [1e-12])
        retained = sum(
            round((float(value) / maximum_bias) * levels) != 0
            for value in [*bqm.linear.values(), *bqm.quadratic.values()]
            if abs(float(value)) > 1e-15
        ) / len(biases)
        best_delta = min(
            [0.0]
            + [float(move["energy_delta"]) for move in local["moves"]]
        )
        resolvable = bool(
            best_delta < -TOLERANCE
            and -best_delta / maximum_bias >= lsb_margin / levels
        )
        physical, minimum_chain, maximum_chain = zephyr_metrics(
            len(local["moves"]), tile_count
        )
        integer_checks = {
            "subproblem_index": index,
            "k": local["k"],
            "eligible_quality_nonincreasing_move_count": local[
                "eligible_count"
            ],
            "encoded_move_variable_count": len(local["moves"]),
            "conflict_pair_count": local["conflict_pairs"],
            "nonconflicting_pair_count": local["nonconflicting_pairs"],
            "ideal_zephyr_physical_qubit_count": physical,
            "ideal_zephyr_minimum_chain_length": minimum_chain,
            "ideal_zephyr_maximum_chain_length": maximum_chain,
        }
        for field, value in integer_checks.items():
            if int(row[field]) != int(value):
                raise ValueError(f"Stage77 local integer metric differs: {field}")
        numeric_checks = {
            "maximum_objective_identity_residual": max(residuals),
            "minimum_conflicting_pair_energy_margin": min(
                conflict_margins + [local["margin"]]
            ),
            "quantized_bias_retention_fraction": retained,
            "best_single_move_energy_delta": best_delta,
        }
        for field, value in numeric_checks.items():
            if not close(row[field], value):
                raise ValueError(f"Stage77 local numeric metric differs: {field}")
        if boolean(row["hardware_resolvable_single_move_improvement"]) != resolvable:
            raise ValueError("Stage77 local resolution label differs")
    return expected


def proxy_bqm(
    bqm: dimod.BinaryQuadraticModel,
    condition: dict[str, Any],
    rng: np.random.Generator,
) -> dimod.BinaryQuadraticModel:
    full_scale = max(
        [
            abs(float(value))
            for value in [*bqm.linear.values(), *bqm.quadratic.values()]
            if abs(float(value)) > 1e-15
        ]
        + [1e-12]
    )
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


def replay_trials(
    locals_: list[dict[str, Any]],
    rows: list[dict[str, str]],
    config: dict[str, Any],
) -> None:
    protocol = config["warm_start_emulation"]
    conditions = protocol["conditions"]
    repeats = int(protocol["repeats"])
    reads = int(protocol["reads_per_repeat"])
    expected_count = len(locals_) * len(conditions) * repeats
    if len(rows) != expected_count:
        raise ValueError("Stage77 emulation trial count differs")
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
    sampler = SimulatedAnnealingSampler()
    base_seed = int(protocol["seed_base"])
    position = 0
    for local_index, local in enumerate(locals_):
        cell = local["cell"]
        model = cell["model"]
        initial = {name: 0 for name in local["names"]}
        for condition_index, condition in enumerate(conditions):
            for repeat in range(repeats):
                row = rows[position]
                position += 1
                seed = (
                    base_seed
                    + local_index * 100_000
                    + condition_index * 1_000
                    + repeat
                )
                proxy = proxy_bqm(
                    local["bqm"], condition, np.random.default_rng(seed)
                )
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
                best_energy = float(local["warm_energy"])
                best_subset = local["warm"]
                for sample in samples.samples():
                    subset, conflict_free = decode(local, sample)
                    if not conflict_free:
                        conflict_count += 1
                        continue
                    if not s75.valid(cell, subset):
                        continue
                    feasible_count += 1
                    unique.add(subset)
                    energy = s75.variable_energy(model, subset, cell["reward"])
                    if (energy, subset) < (best_energy, best_subset):
                        best_energy = energy
                        best_subset = subset
                expected = {
                    "subproblem_index": local_index,
                    "condition": condition["id"],
                    "repeat": repeat,
                    "seed": seed,
                    "read_count": reads,
                    "feasible_read_count": feasible_count,
                    "conflict_read_count": conflict_count,
                    "unique_feasible_subset_count": len(unique),
                    "best_subset_with_warm_guard": s75.subset_name(
                        model, best_subset
                    ),
                }
                for field, value in expected.items():
                    if str(row[field]) != str(value):
                        raise ValueError(
                            f"Stage77 replay differs at trial {position - 1}: {field}"
                        )
                if not close(row["best_true_energy_with_warm_guard"], best_energy):
                    raise ValueError("Stage77 replay best energy differs")


def recompute_summary(
    trials: list[dict[str, str]],
    local_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> None:
    opportunity_count = sum(
        boolean(row["hardware_resolvable_single_move_improvement"])
        for row in local_rows
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trials:
        grouped[row["condition"]].append(row)
    if len(grouped) != len(summary_rows):
        raise ValueError("Stage77 summary condition count differs")
    for summary in summary_rows:
        rows = grouped[summary["condition"]]
        by_subproblem: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_subproblem[int(row["subproblem_index"])].append(row)
        total_reads = sum(int(row["read_count"]) for row in rows)
        checks = {
            "subproblem_count": len(by_subproblem),
            "run_count": len(rows),
            "read_count": total_reads,
            "strict_improvement_subproblem_count": sum(
                any(boolean(row["strict_improvement_recovered"]) for row in group)
                for group in by_subproblem.values()
            ),
            "hardware_resolvable_opportunity_subproblem_count": opportunity_count,
            "hardware_resolvable_opportunity_recovered_count": sum(
                any(
                    boolean(row["hardware_resolvable_opportunity_recovered"])
                    for row in group
                )
                for group in by_subproblem.values()
            ),
        }
        for field, value in checks.items():
            if int(summary[field]) != int(value):
                raise ValueError(f"Stage77 summary count differs: {field}")
        numeric = {
            "feasible_read_fraction": sum(
                int(row["feasible_read_count"]) for row in rows
            )
            / total_reads,
            "conflict_read_fraction": sum(
                int(row["conflict_read_count"]) for row in rows
            )
            / total_reads,
            "warm_guard_nonworse_run_fraction": sum(
                boolean(row["warm_guard_nonworse"]) for row in rows
            )
            / len(rows),
        }
        numeric["hardware_resolvable_opportunity_recovery_fraction"] = (
            checks["hardware_resolvable_opportunity_recovered_count"]
            / opportunity_count
            if opportunity_count
            else 1.0
        )
        for field, value in numeric.items():
            if not close(summary[field], value):
                raise ValueError(f"Stage77 summary numeric differs: {field}")


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = s75.read_json(config_path.resolve())
    for section in ("implementation", "inputs"):
        for label, descriptor in config[section].items():
            s75.verified(root, descriptor, label)
    result_path = root / config["outputs"]["result_json"]
    result = s75.read_json(result_path)
    for descriptor in result["outputs"].values():
        s75.verified(root, descriptor, "Stage77 output")
    direct_rows = s75.read_csv(
        root / config["outputs"]["direct_encoding_metrics_csv"]
    )
    local_rows = s75.read_csv(
        root / config["outputs"]["local_swap_bqm_metrics_csv"]
    )
    trial_rows = s75.read_csv(
        root / config["outputs"]["emulation_trials_csv"]
    )
    summary_rows = s75.read_csv(
        root / config["outputs"]["emulation_summary_csv"]
    )
    source_cells = cells(config, root)
    audit_direct(source_cells, direct_rows, config)
    locals_ = audit_local(source_cells, local_rows, config)
    replay_trials(locals_, trial_rows, config)
    recompute_summary(trial_rows, local_rows, summary_rows)
    expected_decision = {
        "frozen_variable_k_cqm_remains_scientific_model": True,
        "leap_hybrid_cqm_application_route_recommended": True,
        "full_direct_qpu_bqm_route_authorized": False,
        "advantage2_local_reverse_annealing_poc_ready_for_budget_request": True,
        "ibm_warm_start_qaoa_full_problem_route_authorized": False,
        "trapped_ion_full_problem_route_authorized": False,
        "neutral_atom_full_problem_route_authorized": False,
        "paid_cloud_execution_authorized": False,
        "paid_qpu_execution_authorized": False,
        "quantum_scaling_claim_authorized": False,
        "quantum_advantage_claim_authorized": False,
    }
    if result["decision"] != expected_decision:
        raise ValueError("Stage77 audited decision differs")
    value = {
        "schema_version": "1.0",
        "status": "stage77_quantum_hardware_interface_independent_audit_ok",
        "source_result": s75.descriptor(root, result_path),
        "cqm_models_independently_rebuilt": len(source_cells),
        "direct_bqm_metrics_independently_recomputed": len(direct_rows),
        "local_swap_bqms_independently_rebuilt": len(locals_),
        "emulation_runs_deterministically_replayed": len(trial_rows),
        "emulation_summaries_independently_recomputed": len(summary_rows),
        "full_direct_qpu_route_authorized": False,
        "local_reverse_annealing_poc_ready_for_budget_request": True,
        "paid_qpu_execution_authorized": False,
        "quantum_advantage_claim_authorized": False,
    }
    output = root / config["outputs"]["audit_json"]
    s75.write_json(output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage77_quantum_hardware_interface_gate.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    audit(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
