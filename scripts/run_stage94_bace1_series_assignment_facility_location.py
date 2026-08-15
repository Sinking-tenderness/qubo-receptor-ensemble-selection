from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage92_bace1_group_robust_hardness_adjudication as common


TOLERANCE = 1e-10


def build_pair_catalog(
    scores: list[dict[str, str]], manifest: list[dict[str, str]], config: dict[str, object]
) -> tuple[list[str], list[str], list[tuple[int, int]], list[list[dict[str, float]]], dict[str, object]]:
    metadata = {row["ligand_id"]: row for row in manifest}
    population_ids = sorted(
        {
            row["ligand_id"]
            for row in scores
            if metadata[row["ligand_id"]]["core_series"].lower() == "true"
            and row["label"] in {"high", "low"}
        }
    )
    high_ids = [x for x in population_ids if metadata[x]["potency_label"] == "high"]
    low_ids = [x for x in population_ids if metadata[x]["potency_label"] == "low"]
    group_ids = sorted({metadata[x]["scaffold_group_id"] for x in high_ids})
    receptor_ids = sorted({row["receptor_id"] for row in scores})
    seed_ids = sorted({row["seed_id"] for row in scores})
    if (len(population_ids), len(high_ids), len(low_ids), len(group_ids), len(receptor_ids), len(seed_ids)) != (222, 195, 27, 6, 34, 3):
        raise ValueError("Stage94 source dimensions differ")
    hit_count = math.ceil(float(config["frozen_problem"]["receptor_hit_fraction_per_seed"]) * len(population_ids))
    if hit_count != 23:
        raise ValueError("Stage94 hit count differs")

    population_index = {value: index for index, value in enumerate(population_ids)}
    high_index = {value: index for index, value in enumerate(high_ids)}
    low_index = {value: index for index, value in enumerate(low_ids)}
    group_masks = [
        sum(
            1 << high_index[ligand_id]
            for ligand_id in high_ids
            if metadata[ligand_id]["scaffold_group_id"] == group_id
        )
        for group_id in group_ids
    ]
    grid = defaultdict(list)
    for row in scores:
        if row["ligand_id"] in population_index:
            grid[(row["seed_id"], row["receptor_id"])].append(row)
    hit_sets: list[list[frozenset[int]]] = []
    high_masks: list[list[int]] = []
    low_masks: list[list[int]] = []
    for seed_id in seed_ids:
        seed_sets = []
        seed_high = []
        seed_low = []
        for receptor_id in receptor_ids:
            rows = sorted(
                grid[(seed_id, receptor_id)],
                key=lambda row: (float(row["gpu_score"]), row["ligand_id"]),
            )[:hit_count]
            seed_sets.append(frozenset(population_index[row["ligand_id"]] for row in rows))
            seed_high.append(sum(1 << high_index[row["ligand_id"]] for row in rows if row["label"] == "high"))
            seed_low.append(sum(1 << low_index[row["ligand_id"]] for row in rows if row["label"] == "low"))
        hit_sets.append(seed_sets)
        high_masks.append(seed_high)
        low_masks.append(seed_low)

    pairs = list(itertools.combinations(range(len(receptor_ids)), 2))
    catalog: list[list[dict[str, float]]] = []
    for group_mask in group_masks:
        group_rows = []
        for left, right in pairs:
            coverages = []
            low_exposures = []
            overlaps = []
            for seed_index in range(len(seed_ids)):
                high_joint = high_masks[seed_index][left] & high_masks[seed_index][right]
                low_joint = low_masks[seed_index][left] & low_masks[seed_index][right]
                coverages.append((high_joint & group_mask).bit_count() / group_mask.bit_count())
                low_exposures.append(low_joint.bit_count() / len(low_ids))
                union = hit_sets[seed_index][left] | hit_sets[seed_index][right]
                overlaps.append(
                    len(hit_sets[seed_index][left] & hit_sets[seed_index][right]) / len(union)
                    if union
                    else 0.0
                )
            mean_coverage = statistics.mean(coverages)
            mean_low = statistics.mean(low_exposures)
            mean_overlap = statistics.mean(overlaps)
            group_rows.append(
                {
                    "minimum_coverage": min(coverages),
                    "mean_coverage": mean_coverage,
                    "mean_low_exposure": mean_low,
                    "mean_overlap": mean_overlap,
                    "additive": 0.30 * mean_coverage / len(group_ids)
                    - 0.20 * mean_low / len(group_ids)
                    - 0.10 * mean_overlap / len(group_ids),
                }
            )
        catalog.append(group_rows)
    return receptor_ids, group_ids, pairs, catalog, {
        "population": len(population_ids),
        "high": len(high_ids),
        "low": len(low_ids),
        "groups": len(group_ids),
        "seeds": len(seed_ids),
        "hit_count": hit_count,
        "pair_count": len(pairs),
    }


class ConditionalAssignment:
    def __init__(self, pairs: list[tuple[int, int]], catalog: list[list[dict[str, float]]]) -> None:
        self.pairs = pairs
        self.catalog = catalog
        self.pair_index = {pair: index for index, pair in enumerate(pairs)}

    @functools.lru_cache(maxsize=None)
    def solve(self, opened: tuple[int, ...]) -> tuple[float, tuple[int, ...], float]:
        if len(opened) < 2:
            return -math.inf, (), 0.0
        allowed = [
            self.pair_index[pair]
            for pair in itertools.combinations(opened, 2)
        ]
        thresholds = sorted(
            {self.catalog[group][pair]["minimum_coverage"] for group in range(len(self.catalog)) for pair in allowed},
            reverse=True,
        )
        best_value = -math.inf
        best_assignments: tuple[int, ...] | None = None
        best_threshold = 0.0
        for threshold in thresholds:
            assignments = []
            additive_sum = 0.0
            feasible = True
            for group in range(len(self.catalog)):
                candidates = [
                    pair
                    for pair in allowed
                    if self.catalog[group][pair]["minimum_coverage"] + TOLERANCE >= threshold
                ]
                if not candidates:
                    feasible = False
                    break
                selected = max(
                    candidates,
                    key=lambda pair: (self.catalog[group][pair]["additive"], -pair),
                )
                assignments.append(selected)
                additive_sum += self.catalog[group][selected]["additive"]
            if not feasible:
                continue
            value = 0.40 * threshold + additive_sum
            assignment_tuple = tuple(assignments)
            if value > best_value + TOLERANCE or (
                abs(value - best_value) <= TOLERANCE
                and (best_assignments is None or assignment_tuple < best_assignments)
            ):
                best_value = value
                best_assignments = assignment_tuple
                best_threshold = threshold
        if best_assignments is None:
            raise RuntimeError("conditional assignment has no feasible solution")
        return best_value, best_assignments, best_threshold


def better(value: float, selected: tuple[int, ...], best_value: float, best: tuple[int, ...] | None) -> bool:
    return value > best_value + TOLERANCE or (
        abs(value - best_value) <= TOLERANCE and (best is None or selected < best)
    )


def greedy(assignment: ConditionalAssignment, receptor_count: int, k: int, start: tuple[int, ...] = ()) -> tuple[int, ...]:
    selected = tuple(sorted(start))
    if len(selected) < 2:
        best_pair = max(
            itertools.combinations(range(receptor_count), 2),
            key=lambda pair: (assignment.solve(pair)[0], tuple(-i for i in pair)),
        )
        selected = best_pair
    while len(selected) < k:
        selected_set = set(selected)
        candidates = [tuple(sorted((*selected, index))) for index in range(receptor_count) if index not in selected_set]
        selected = max(
            candidates,
            key=lambda state: (assignment.solve(state)[0], tuple(-i for i in state)),
        )
    return selected


def one_swap(assignment: ConditionalAssignment, receptor_count: int, start: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    selected = tuple(sorted(start))
    iterations = 0
    while True:
        current_value = assignment.solve(selected)[0]
        selected_set = set(selected)
        best = selected
        best_value = current_value
        for removed in selected:
            for added in range(receptor_count):
                if added in selected_set:
                    continue
                candidate = tuple(sorted((selected_set - {removed}) | {added}))
                value = assignment.solve(candidate)[0]
                if better(value, candidate, best_value, best):
                    best, best_value = candidate, value
        if best_value <= current_value + TOLERANCE:
            return selected, iterations
        selected = best
        iterations += 1


def solve_milp(
    receptor_count: int,
    pairs: list[tuple[int, int]],
    catalog: list[list[dict[str, float]]],
    open_count: int,
) -> dict[str, object]:
    group_count = len(catalog)
    pair_count = len(pairs)
    assignment_offset = receptor_count
    t_index = receptor_count + group_count * pair_count
    variable_count = t_index + 1
    c = np.zeros(variable_count)
    for group in range(group_count):
        for pair in range(pair_count):
            c[assignment_offset + group * pair_count + pair] = -catalog[group][pair]["additive"]
    c[t_index] = -0.40
    integrality = np.zeros(variable_count, dtype=int)
    integrality[:t_index] = 1
    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.ones(variable_count)

    row_indices = []
    column_indices = []
    values = []
    lower = []
    upper = []

    def add_constraint(coefficients: dict[int, float], lb: float, ub: float) -> None:
        row = len(lower)
        for column, value in coefficients.items():
            row_indices.append(row)
            column_indices.append(column)
            values.append(value)
        lower.append(lb)
        upper.append(ub)

    add_constraint({index: 1.0 for index in range(receptor_count)}, open_count, open_count)
    for group in range(group_count):
        add_constraint(
            {assignment_offset + group * pair_count + pair: 1.0 for pair in range(pair_count)},
            1.0,
            1.0,
        )
        for pair_index, (left, right) in enumerate(pairs):
            variable = assignment_offset + group * pair_count + pair_index
            add_constraint({variable: 1.0, left: -1.0}, -math.inf, 0.0)
            add_constraint({variable: 1.0, right: -1.0}, -math.inf, 0.0)
        coefficients = {t_index: 1.0}
        for pair in range(pair_count):
            coefficients[assignment_offset + group * pair_count + pair] = -catalog[group][pair]["minimum_coverage"]
        add_constraint(coefficients, -math.inf, 0.0)

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(lower), variable_count),
    ).tocsr()
    started = time.perf_counter()
    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        options={"mip_rel_gap": 0.0},
    )
    elapsed = time.perf_counter() - started
    if not result.success or result.x is None:
        raise RuntimeError(f"Stage94 MILP failed: {result.message}")
    opened = tuple(index for index in range(receptor_count) if result.x[index] > 0.5)
    assignments = []
    for group in range(group_count):
        selected = [
            pair
            for pair in range(pair_count)
            if result.x[assignment_offset + group * pair_count + pair] > 0.5
        ]
        if len(selected) != 1:
            raise ValueError("Stage94 MILP assignment is not one-hot")
        assignments.append(selected[0])
    return {
        "opened": opened,
        "assignments": tuple(assignments),
        "objective": -float(result.fun),
        "worst_coverage": float(result.x[t_index]),
        "elapsed_seconds": elapsed,
        "mip_gap": float(getattr(result, "mip_gap", math.nan)),
        "mip_node_count": int(getattr(result, "mip_node_count", -1)),
        "message": str(result.message),
        "variable_count": variable_count,
        "constraint_count": len(lower),
    }


def tabu(assignment: ConditionalAssignment, receptor_count: int, k: int, config: dict[str, object]) -> tuple[int, ...]:
    rng = random.Random(int(config["seed"]))
    global_best = None
    global_value = -math.inf
    for _ in range(int(config["restart_count"])):
        current = tuple(sorted(rng.sample(range(receptor_count), k)))
        tabu_until = {}
        for step in range(int(config["iteration_count"])):
            current_set = set(current)
            move_best = None
            move_value = -math.inf
            move_pair = None
            for removed in current:
                for added in range(receptor_count):
                    if added in current_set:
                        continue
                    candidate = tuple(sorted((current_set - {removed}) | {added}))
                    value = assignment.solve(candidate)[0]
                    if tabu_until.get((added, removed), -1) > step and value <= global_value + TOLERANCE:
                        continue
                    if better(value, candidate, move_value, move_best):
                        move_best, move_value, move_pair = candidate, value, (removed, added)
            if move_best is None:
                break
            current = move_best
            tabu_until[move_pair] = step + int(config["tabu_tenure"])
            if better(move_value, current, global_value, global_best):
                global_best, global_value = current, move_value
    if global_best is None:
        raise RuntimeError("Stage94 tabu produced no state")
    return global_best


def anneal(assignment: ConditionalAssignment, receptor_count: int, k: int, config: dict[str, object]) -> tuple[int, ...]:
    rng = random.Random(int(config["seed"]))
    best = None
    best_value = -math.inf
    steps = int(config["steps_per_restart"])
    for _ in range(int(config["restart_count"])):
        current = tuple(sorted(rng.sample(range(receptor_count), k)))
        current_value = assignment.solve(current)[0]
        for step in range(steps):
            temperature = 0.02 * (0.0001 / 0.02) ** (step / max(1, steps - 1))
            current_set = set(current)
            removed = rng.choice(current)
            added = rng.choice([index for index in range(receptor_count) if index not in current_set])
            candidate = tuple(sorted((current_set - {removed}) | {added}))
            value = assignment.solve(candidate)[0]
            if value >= current_value or rng.random() < math.exp((value - current_value) / temperature):
                current, current_value = candidate, value
            if better(current_value, current, best_value, best):
                best, best_value = current, current_value
    if best is None:
        raise RuntimeError("Stage94 annealing produced no state")
    return best


def run(root: Path, config_path: Path) -> dict[str, object]:
    root = root.resolve()
    config = common.read_json(config_path.resolve())
    scores = common.read_csv(root / config["inputs"]["amended_seed_scores"])
    manifest = common.read_csv(root / config["inputs"]["development_manifest"])
    adjudication = common.read_json(root / config["inputs"]["metadata_adjudication"])
    if adjudication.get("status") != "stage92a_bace1_target_id_metadata_adjudication_ok":
        raise ValueError("Stage92a did not pass")
    receptor_ids, group_ids, pairs, catalog, dimensions = build_pair_catalog(scores, manifest, config)
    assignment = ConditionalAssignment(pairs, catalog)
    k = int(config["frozen_problem"]["open_receptor_count"])
    exact = solve_milp(len(receptor_ids), pairs, catalog, k)
    conditional_value, conditional_assignments, conditional_threshold = assignment.solve(exact["opened"])
    if abs(conditional_value - exact["objective"]) > 1e-7:
        raise ValueError("MILP and conditional assignment objectives differ")
    if tuple(exact["assignments"]) != conditional_assignments:
        raise ValueError("MILP and conditional assignments differ")

    records = []
    def record(method: str, selected: tuple[int, ...], seconds: float, **extra: object) -> None:
        value, assignments, threshold = assignment.solve(selected)
        records.append(
            {
                "method": method,
                "open_receptor_ids": ";".join(receptor_ids[index] for index in selected),
                "objective": value,
                "worst_series_coverage": threshold,
                "elapsed_seconds": seconds,
                **extra,
            }
        )

    started = time.perf_counter()
    direct = greedy(assignment, len(receptor_ids), k)
    record("direct_greedy_open_set", direct, time.perf_counter() - started)
    started = time.perf_counter()
    direct_swap, direct_iterations = one_swap(assignment, len(receptor_ids), direct)
    record("greedy_plus_one_swap", direct_swap, time.perf_counter() - started, local_search_iterations=direct_iterations)

    started = time.perf_counter()
    pair_candidates = []
    pair_iterations = 0
    for pair in pairs:
        selected, iterations = one_swap(assignment, len(receptor_ids), greedy(assignment, len(receptor_ids), k, pair))
        pair_candidates.append(selected)
        pair_iterations += iterations
    pair_best = max(pair_candidates, key=lambda state: (assignment.solve(state)[0], tuple(-i for i in state)))
    record("all_pair_starts_plus_one_swap", pair_best, time.perf_counter() - started, restart_count=len(pairs), local_search_iterations=pair_iterations)

    random_config = dict(config["classical_baselines"]["random_restart_one_swap"])
    rng = random.Random(int(random_config["seed"]))
    started = time.perf_counter()
    random_candidates = []
    random_iterations = 0
    for _ in range(int(random_config["restart_count"])):
        selected, iterations = one_swap(
            assignment,
            len(receptor_ids),
            tuple(sorted(rng.sample(range(len(receptor_ids)), k))),
        )
        random_candidates.append(selected)
        random_iterations += iterations
    random_best = max(random_candidates, key=lambda state: (assignment.solve(state)[0], tuple(-i for i in state)))
    record("random256_plus_one_swap", random_best, time.perf_counter() - started, restart_count=256, local_search_iterations=random_iterations, distinct_local_optimum_count=len(set(random_candidates)))

    started = time.perf_counter()
    tabu_selected = tabu(assignment, len(receptor_ids), k, dict(config["classical_baselines"]["tabu"]))
    record("multistart_tabu", tabu_selected, time.perf_counter() - started)
    started = time.perf_counter()
    anneal_selected = anneal(assignment, len(receptor_ids), k, dict(config["classical_baselines"]["simulated_annealing"]))
    record("simulated_annealing", anneal_selected, time.perf_counter() - started)
    record("milp_exact", exact["opened"], float(exact["elapsed_seconds"]), mip_gap=exact["mip_gap"], mip_node_count=exact["mip_node_count"])

    exact_value = float(exact["objective"])
    exact_ids = ";".join(receptor_ids[index] for index in exact["opened"])
    for row in records:
        row["objective_gap_to_milp"] = exact_value - float(row["objective"])
        row["matches_milp_open_set"] = row["open_receptor_ids"] == exact_ids
    best_one_swap = max(
        (direct_swap, pair_best, random_best),
        key=lambda state: (assignment.solve(state)[0], tuple(-i for i in state)),
    )
    best_one_swap_value = assignment.solve(best_one_swap)[0]
    replacement_distance = len(set(exact["opened"]) - set(best_one_swap))
    mip_certified = bool(exact["mip_gap"] <= 1e-12)
    checks = {
        "milp_optimum_strictly_improves_greedy_plus_one_swap": exact_value > assignment.solve(direct_swap)[0] + TOLERANCE,
        "milp_optimum_strictly_improves_best_256_random_restart_one_swap": exact_value > assignment.solve(random_best)[0] + TOLERANCE,
        "open_set_replacement_distance_from_best_one_swap": replacement_distance >= int(config["go_gate"]["open_set_replacement_distance_from_best_one_swap_at_least"]),
        "milp_optimality_certificate": mip_certified,
        "routing_policy_has_nonzero_worst_series_coverage": exact["worst_coverage"] > 0,
        "milp_conditional_recomputation_exact": abs(conditional_value - exact_value) <= 1e-7,
    }
    passed = all(checks.values())
    outputs = dict(config["outputs"])
    baseline_path = root / outputs["baseline_csv"]
    common.write_csv(baseline_path, records)
    assignment_rows = []
    for group_index, pair_index in enumerate(exact["assignments"]):
        left, right = pairs[pair_index]
        metrics = catalog[group_index][pair_index]
        assignment_rows.append(
            {
                "series_id": group_ids[group_index],
                "receptor_a": receptor_ids[left],
                "receptor_b": receptor_ids[right],
                **{key: metrics[key] for key in ("minimum_coverage", "mean_coverage", "mean_low_exposure", "mean_overlap")},
            }
        )
    assignment_path = root / outputs["assignment_csv"]
    common.write_csv(assignment_path, assignment_rows)
    result = {
        "schema_version": "1.0",
        "status": "stage94_bace1_series_assignment_gate_passed" if passed else "stage94_bace1_series_assignment_gate_failed",
        "experiment_class": config["experiment_class"],
        "dimensions": dimensions,
        "milp": {
            "open_receptor_ids": [receptor_ids[index] for index in exact["opened"]],
            "objective": exact_value,
            "worst_series_coverage": exact["worst_coverage"],
            "elapsed_seconds": exact["elapsed_seconds"],
            "mip_gap": exact["mip_gap"],
            "mip_node_count": exact["mip_node_count"],
            "variable_count": exact["variable_count"],
            "constraint_count": exact["constraint_count"],
            "message": exact["message"],
        },
        "best_one_swap": {
            "open_receptor_ids": [receptor_ids[index] for index in best_one_swap],
            "objective": best_one_swap_value,
            "replacement_distance_to_milp": replacement_distance,
        },
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "authorization": {
            "prospective_new_target_problem_freeze_authorized": passed,
            "new_docking_authorized": False,
            "confirmation_assays_authorized": False,
            "quantum_simulation_or_hardware_authorized": False,
        },
        "data_boundary": {"confirmation_scores_read": 0, "locked_test_scores_read": 0, "new_docking_jobs": 0, "quantum_jobs": 0},
        "stop_rule_applies": not passed,
        "outputs": {
            "baseline_csv": {"path": outputs["baseline_csv"], "sha256": common.sha256(baseline_path)},
            "assignment_csv": {"path": outputs["assignment_csv"], "sha256": common.sha256(assignment_path)},
        },
        "interpretation": "Stage94 tests a cost-aware joint receptor-opening and chemical-series routing problem. A pass authorizes only a prospective new-target freeze; it does not establish efficacy or quantum advantage.",
    }
    result_path = root / outputs["result_json"]
    common.write_json(result_path, result)
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join([
            "# Stage94 BACE1 series-assignment facility-location diagnostic",
            "",
            f"Status: `{result['status']}`.",
            "",
            f"MILP objective: {exact_value:.9f}; best one-swap objective: {best_one_swap_value:.9f}; gap: {exact_value-best_one_swap_value:.9f}.",
            "",
            f"MILP opened {k} receptors and assigned each of {len(group_ids)} chemical series to one receptor pair. The certificate gap was {exact['mip_gap']}.",
            "",
            "This is a post-hoc feasibility diagnostic. It can authorize only a prospectively frozen replication on a new target.",
        ]) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage94_bace1_series_assignment_facility_location.json"))
    args = parser.parse_args()
    run(args.root.resolve(), args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
