from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage92_bace1_group_robust_hardness_adjudication as common


class ConsensusObjective:
    def __init__(
        self,
        receptor_ids: list[str],
        seed_ids: list[str],
        high_masks: list[list[int]],
        low_masks: list[list[int]],
        group_masks: list[int],
        hit_sets: list[list[frozenset[int]]],
        low_count: int,
    ) -> None:
        self.receptor_ids = receptor_ids
        self.seed_ids = seed_ids
        self.high_masks = high_masks
        self.low_masks = low_masks
        self.group_masks = group_masks
        self.low_count = low_count
        self.pair_overlap = [[0.0] * len(receptor_ids) for _ in receptor_ids]
        for left in range(len(receptor_ids)):
            for right in range(left + 1, len(receptor_ids)):
                values = []
                for seed_index in range(len(seed_ids)):
                    left_set = hit_sets[seed_index][left]
                    right_set = hit_sets[seed_index][right]
                    union = left_set | right_set
                    values.append(len(left_set & right_set) / len(union) if union else 0.0)
                self.pair_overlap[left][right] = statistics.mean(values)

    def consensus_masks(self, selected: tuple[int, ...]) -> tuple[list[int], int]:
        consensus_high = []
        global_low = 0
        for seed_index in range(len(self.seed_ids)):
            high = 0
            low = 0
            for left, right in itertools.combinations(selected, 2):
                high |= (
                    self.high_masks[seed_index][left]
                    & self.high_masks[seed_index][right]
                )
                low |= (
                    self.low_masks[seed_index][left]
                    & self.low_masks[seed_index][right]
                )
            consensus_high.append(high)
            global_low |= low
        return consensus_high, global_low

    def value(self, selected: tuple[int, ...]) -> float:
        high_by_seed, global_low = self.consensus_masks(selected)
        minimum_coverage = math.inf
        coverage_sum = 0.0
        scenario_count = 0
        for high in high_by_seed:
            for group_mask in self.group_masks:
                coverage = (high & group_mask).bit_count() / group_mask.bit_count()
                minimum_coverage = min(minimum_coverage, coverage)
                coverage_sum += coverage
                scenario_count += 1
        pair_count = math.comb(len(selected), 2) if len(selected) >= 2 else 0
        overlap_sum = sum(
            self.pair_overlap[min(left, right)][max(left, right)]
            for left, right in itertools.combinations(selected, 2)
        )
        return (
            0.40 * minimum_coverage
            + 0.30 * coverage_sum / scenario_count
            - 0.20 * global_low.bit_count() / self.low_count
            - 0.10 * (overlap_sum / pair_count if pair_count else 0.0)
        )

    def components(self, selected: tuple[int, ...]) -> dict[str, object]:
        high_by_seed, global_low = self.consensus_masks(selected)
        scenario_coverages = []
        per_seed = {}
        for seed_id, high in zip(self.seed_ids, high_by_seed, strict=True):
            values = [
                (high & group_mask).bit_count() / group_mask.bit_count()
                for group_mask in self.group_masks
            ]
            per_seed[seed_id] = values
            scenario_coverages.extend(values)
        pair_count = math.comb(len(selected), 2) if len(selected) >= 2 else 0
        overlap = (
            sum(
                self.pair_overlap[min(left, right)][max(left, right)]
                for left, right in itertools.combinations(selected, 2)
            )
            / pair_count
            if pair_count
            else 0.0
        )
        low_exposure = global_low.bit_count() / self.low_count
        return {
            "objective": self.value(selected),
            "minimum_scenario_coverage": min(scenario_coverages),
            "mean_scenario_coverage": statistics.mean(scenario_coverages),
            "global_low_consensus_exposure": low_exposure,
            "mean_pair_seed_averaged_jaccard": overlap,
            "scenario_coverages_by_seed": per_seed,
        }


def build_objective(
    scores: list[dict[str, str]], manifest: list[dict[str, str]], config: dict[str, object]
) -> tuple[ConsensusObjective, dict[str, object]]:
    metadata = {row["ligand_id"]: row for row in manifest}
    population_ids = sorted(
        {
            row["ligand_id"]
            for row in scores
            if metadata[row["ligand_id"]]["core_series"].lower() == "true"
            and row["label"] in {"high", "low"}
        }
    )
    high_ids = [ligand_id for ligand_id in population_ids if metadata[ligand_id]["potency_label"] == "high"]
    low_ids = [ligand_id for ligand_id in population_ids if metadata[ligand_id]["potency_label"] == "low"]
    groups = sorted({metadata[ligand_id]["scaffold_group_id"] for ligand_id in high_ids})
    receptor_ids = sorted({row["receptor_id"] for row in scores})
    seed_ids = sorted({row["seed_id"] for row in scores})
    if (len(population_ids), len(high_ids), len(low_ids), len(groups), len(receptor_ids), len(seed_ids)) != (
        222, 195, 27, 6, 34, 3
    ):
        raise ValueError("Stage93 source dimensions differ")
    expected_rows = len(receptor_ids) * len(seed_ids) * len(manifest)
    if len(scores) != expected_rows:
        raise ValueError("Stage93 complete score grid differs")
    hit_count = math.ceil(float(config["objective"]["receptor_hit_fraction_per_seed"]) * len(population_ids))
    if hit_count != 23 or int(config["objective"]["consensus_multiplicity"]) != 2:
        raise ValueError("Stage93 frozen consensus definition differs")

    population_index = {ligand_id: index for index, ligand_id in enumerate(population_ids)}
    high_index = {ligand_id: index for index, ligand_id in enumerate(high_ids)}
    low_index = {ligand_id: index for index, ligand_id in enumerate(low_ids)}
    grid = defaultdict(list)
    for row in scores:
        if row["ligand_id"] in population_index:
            grid[(row["seed_id"], row["receptor_id"])].append(row)
    high_masks: list[list[int]] = []
    low_masks: list[list[int]] = []
    hit_sets: list[list[frozenset[int]]] = []
    for seed_id in seed_ids:
        seed_high = []
        seed_low = []
        seed_sets = []
        for receptor_id in receptor_ids:
            rows = sorted(
                grid[(seed_id, receptor_id)],
                key=lambda row: (float(row["gpu_score"]), row["ligand_id"]),
            )[:hit_count]
            seed_sets.append(frozenset(population_index[row["ligand_id"]] for row in rows))
            seed_high.append(
                sum(1 << high_index[row["ligand_id"]] for row in rows if row["label"] == "high")
            )
            seed_low.append(
                sum(1 << low_index[row["ligand_id"]] for row in rows if row["label"] == "low")
            )
        high_masks.append(seed_high)
        low_masks.append(seed_low)
        hit_sets.append(seed_sets)
    group_masks = [
        sum(
            1 << high_index[ligand_id]
            for ligand_id in high_ids
            if metadata[ligand_id]["scaffold_group_id"] == group
        )
        for group in groups
    ]
    return (
        ConsensusObjective(
            receptor_ids, seed_ids, high_masks, low_masks, group_masks, hit_sets, len(low_ids)
        ),
        {
            "population": len(population_ids),
            "high": len(high_ids),
            "low": len(low_ids),
            "groups": len(groups),
            "seeds": len(seed_ids),
            "scenarios": len(groups) * len(seed_ids),
            "hit_count_per_receptor_seed": hit_count,
            "consensus_multiplicity": 2,
        },
    )


def run(root: Path, config_path: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = common.read_json(config_path)
    inputs = dict(config["inputs"])
    adjudication = common.read_json(common.verified(root, dict(inputs["metadata_adjudication"])))
    scores = common.read_csv(common.verified(root, dict(inputs["amended_seed_scores"])))
    manifest = common.read_csv(common.verified(root, dict(inputs["development_manifest"])))
    if adjudication.get("status") != "stage92a_bace1_target_id_metadata_adjudication_ok":
        raise ValueError("Stage92a metadata adjudication did not pass")
    if {row["target_id"] for row in scores} != {"BACE1"}:
        raise ValueError("Stage93 amended target identity differs")
    objective, population = build_objective(scores, manifest, config)
    k = int(config["objective"]["cardinality_k"])
    records: list[dict[str, object]] = []

    def add(method: str, selected: tuple[int, ...], seconds: float, **extra: object) -> None:
        components = objective.components(selected)
        records.append(
            {
                "method": method,
                "selected_receptor_ids": ";".join(objective.receptor_ids[index] for index in selected),
                "objective": components["objective"],
                "minimum_scenario_coverage": components["minimum_scenario_coverage"],
                "mean_scenario_coverage": components["mean_scenario_coverage"],
                "global_low_consensus_exposure": components["global_low_consensus_exposure"],
                "mean_pair_seed_averaged_jaccard": components["mean_pair_seed_averaged_jaccard"],
                "elapsed_seconds": seconds,
                **extra,
            }
        )

    started = time.perf_counter()
    direct = common.greedy(objective, k)
    add("direct_greedy", direct, time.perf_counter() - started)
    started = time.perf_counter()
    direct_swap, direct_iterations = common.best_one_swap(objective, direct)
    add("greedy_plus_all_one_swaps", direct_swap, time.perf_counter() - started, local_search_iterations=direct_iterations)

    started = time.perf_counter()
    singleton_candidates = []
    singleton_iterations = 0
    for index in range(len(objective.receptor_ids)):
        selected, iterations = common.best_one_swap(objective, common.greedy(objective, k, (index,)))
        singleton_candidates.append(selected)
        singleton_iterations += iterations
    singleton_best = max(
        singleton_candidates,
        key=lambda selected: (objective.value(selected), tuple(-index for index in selected)),
    )
    add(
        "all_singleton_greedy_plus_swaps",
        singleton_best,
        time.perf_counter() - started,
        restart_count=34,
        local_search_iterations=singleton_iterations,
    )

    random_config = dict(config["classical_baselines"]["random_multistart_one_swap"])
    rng = random.Random(int(random_config["seed"]))
    started = time.perf_counter()
    random_candidates = []
    random_iterations = 0
    for _ in range(int(random_config["restart_count"])):
        selected, iterations = common.best_one_swap(
            objective,
            tuple(sorted(rng.sample(range(len(objective.receptor_ids)), k))),
        )
        random_candidates.append(selected)
        random_iterations += iterations
    random_best = max(
        random_candidates,
        key=lambda selected: (objective.value(selected), tuple(-index for index in selected)),
    )
    add(
        "random256_plus_one_swaps",
        random_best,
        time.perf_counter() - started,
        restart_count=256,
        local_search_iterations=random_iterations,
        distinct_local_optimum_count=len(set(random_candidates)),
    )

    tabu_config = dict(config["classical_baselines"]["multistart_tabu"])
    started = time.perf_counter()
    tabu_selected = common.tabu(
        objective,
        k,
        int(tabu_config["restart_count"]),
        int(tabu_config["iteration_count"]),
        int(tabu_config["tabu_tenure"]),
        int(tabu_config["seed"]),
    )
    add("multistart_tabu", tabu_selected, time.perf_counter() - started)

    anneal_config = dict(config["classical_baselines"]["simulated_annealing"])
    started = time.perf_counter()
    anneal_selected = common.anneal(
        objective,
        k,
        int(anneal_config["restart_count"]),
        int(anneal_config["steps_per_restart"]),
        int(anneal_config["seed"]),
    )
    add("simulated_annealing", anneal_selected, time.perf_counter() - started)

    started = time.perf_counter()
    exact_selected, exact_value, second_value, optimal_count = common.exact(objective, k)
    exact_seconds = time.perf_counter() - started
    add(
        "exact_enumeration",
        exact_selected,
        exact_seconds,
        enumerated_state_count=math.comb(len(objective.receptor_ids), k),
        optimal_state_count=optimal_count,
    )
    exact_ids = ";".join(objective.receptor_ids[index] for index in exact_selected)
    for row in records:
        row["objective_gap_to_exact"] = exact_value - float(row["objective"])
        row["matches_exact_selection"] = row["selected_receptor_ids"] == exact_ids

    random_value = objective.value(random_best)
    replacement_distance = len(set(exact_selected) - set(random_best))
    checks = {
        "exact_differs_from_direct_greedy": exact_selected != direct,
        "exact_strictly_improves_greedy_plus_one_swap": exact_value
        > objective.value(direct_swap) + common.TOLERANCE,
        "exact_strictly_improves_best_256_random_restart_one_swap": exact_value
        > random_value + common.TOLERANCE,
        "multi_move_replacement_distance_from_best_one_swap": replacement_distance
        >= int(config["go_gate"]["multi_move_replacement_distance_from_best_one_swap"]),
        "unique_exact_solution": optimal_count == 1,
        "state_count_complete": math.comb(len(objective.receptor_ids), k) == 1344904,
    }
    gate_keys = [
        "exact_differs_from_direct_greedy",
        "exact_strictly_improves_greedy_plus_one_swap",
        "exact_strictly_improves_best_256_random_restart_one_swap",
        "multi_move_replacement_distance_from_best_one_swap",
        "unique_exact_solution",
        "state_count_complete",
    ]
    passed = all(checks[key] for key in gate_keys)
    outputs = dict(config["outputs"])
    baseline_path = root / str(outputs["baseline_csv"])
    common.write_csv(baseline_path, records)
    result = {
        "schema_version": "1.0",
        "status": (
            "stage93_bace1_seed_consensus_hardness_gate_passed"
            if passed
            else "stage93_bace1_seed_consensus_hardness_gate_failed"
        ),
        "experiment_class": config["experiment_class"],
        "population": population,
        "state_count": 1344904,
        "checks": checks,
        "failed_checks": [key for key in gate_keys if not checks[key]],
        "direct_greedy": {
            "selected_receptor_ids": [objective.receptor_ids[index] for index in direct],
            "objective": objective.value(direct),
        },
        "greedy_plus_one_swap": {
            "selected_receptor_ids": [objective.receptor_ids[index] for index in direct_swap],
            "objective": objective.value(direct_swap),
            "iterations": direct_iterations,
        },
        "random256_plus_one_swap": {
            "selected_receptor_ids": [objective.receptor_ids[index] for index in random_best],
            "objective": random_value,
            "replacement_distance_to_exact": replacement_distance,
        },
        "exact": {
            "selected_receptor_ids": [objective.receptor_ids[index] for index in exact_selected],
            "objective": exact_value,
            "components": objective.components(exact_selected),
            "second_best_objective": second_value,
            "gap_to_second_best": exact_value - second_value,
            "optimal_state_count": optimal_count,
            "enumeration_seconds": exact_seconds,
        },
        "authorization": {
            "new_target_preregistration_authorized": passed,
            "new_docking_authorized": False,
            "confirmation_assays_authorized": False,
            "quantum_simulation_or_hardware_authorized": False,
        },
        "data_boundary": {
            "confirmation_scores_read": 0,
            "locked_test_scores_read": 0,
            "new_docking_jobs": 0,
            "quantum_jobs": 0,
        },
        "stop_rule_applies": not passed,
        "outputs": {
            "baseline_csv": {
                "path": str(outputs["baseline_csv"]),
                "sha256": common.sha256(baseline_path),
            }
        },
        "interpretation": (
            "This post-hoc development diagnostic tests whether explicit receptor consensus creates a reproducible multi-move classical trap. Passing would authorize only a prospectively frozen new-target replication, never quantum advantage."
        ),
    }
    result_path = root / str(outputs["result_json"])
    common.write_json(result_path, result)
    report_path = root / str(outputs["report_md"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Stage93 BACE1 seed-consensus hardness diagnostic",
                "",
                f"Status: `{result['status']}`.",
                "",
                "A ligand counted as reliably covered only when at least two selected receptors hit it within the same Uni-Dock seed. Six chemical series crossed with three seeds produced 18 coverage scenarios.",
                "",
                f"Exact objective: {exact_value:.9f}. Best 256-restart one-swap objective: {random_value:.9f}. Gap: {exact_value-random_value:.9f}.",
                "",
                "The stop rule forbids same-matrix tuning if the gate fails. A pass authorizes only a separately preregistered new-target replication.",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage93_bace1_seed_consensus_hardness_diagnostic.json"),
    )
    args = parser.parse_args()
    run(args.root, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
