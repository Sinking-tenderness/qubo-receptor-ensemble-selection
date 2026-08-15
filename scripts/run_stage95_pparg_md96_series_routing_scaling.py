from __future__ import annotations

import argparse
import csv
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


def verified(root: Path, descriptor: dict[str, str]) -> Path:
    path = root / descriptor["path"]
    if common.sha256(path) != descriptor["sha256"]:
        raise ValueError(f"input hash differs: {path}")
    return path


def build_receptor_order(frame_rows: list[dict[str, str]]) -> list[str]:
    rows = sorted(
        frame_rows,
        key=lambda row: (
            int(row["temporal_maximin_rank"]),
            int(row["start_index"]),
            row["conformer_id"],
        ),
    )
    receptor_ids = [row["conformer_id"] for row in rows]
    if len(receptor_ids) != 96 or len(set(receptor_ids)) != 96:
        raise ValueError("Stage95 expects 96 unique MD receptors")
    return receptor_ids


def build_hit_sets(
    scores: list[dict[str, str]], ligand_rows: list[dict[str, str]], receptor_ids: list[str], fraction: float
) -> tuple[list[str], list[str], list[list[frozenset[str]]], int]:
    ligand_ids = sorted(row["ligand_id"] for row in ligand_rows)
    active_ids = sorted(row["ligand_id"] for row in ligand_rows if row["label"] == "active")
    decoy_ids = sorted(row["ligand_id"] for row in ligand_rows if row["label"] == "decoy")
    if len(ligand_ids) != 160 or len(active_ids) != 80 or len(decoy_ids) != 80:
        raise ValueError("Stage95 ligand dimensions differ")
    hit_count = math.ceil(fraction * len(ligand_ids))
    grid: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seed_ids = sorted({row["seed_id"] for row in scores})
    for row in scores:
        grid[(row["seed_id"], row["receptor_id"])].append(row)
    if seed_ids != ["seed0", "seed1", "seed2"]:
        raise ValueError("Stage95 seed identities differ")
    hit_sets: list[list[frozenset[str]]] = []
    for seed_id in seed_ids:
        receptor_sets = []
        for receptor_id in receptor_ids:
            rows = sorted(
                grid[(seed_id, receptor_id)],
                key=lambda row: (float(row["gpu_score"]), row["ligand_id"]),
            )
            if len(rows) != 160:
                raise ValueError(f"incomplete score cell: {seed_id}/{receptor_id}")
            receptor_sets.append(frozenset(row["ligand_id"] for row in rows[:hit_count]))
        hit_sets.append(receptor_sets)
    return active_ids, decoy_ids, hit_sets, hit_count


def build_utility(
    receptor_ids: list[str],
    active_ids: list[str],
    decoy_ids: list[str],
    hit_sets: list[list[frozenset[str]]],
    series_rows: list[dict[str, str]],
    series_count: int,
) -> tuple[list[tuple[int, int]], list[str], np.ndarray, np.ndarray]:
    series_column = f"series_{series_count}"
    membership: dict[str, set[str]] = defaultdict(set)
    for row in series_rows:
        membership[row[series_column]].add(row["ligand_id"])
    series_ids = sorted(membership)
    if len(series_ids) != series_count or set().union(*membership.values()) != set(active_ids):
        raise ValueError(f"Stage95 series partition differs at {series_count}")
    pairs = list(itertools.combinations(range(len(receptor_ids)), 2))
    utility = np.zeros((series_count, len(pairs)), dtype=float)
    weights = np.array([len(membership[series_id]) / len(active_ids) for series_id in series_ids])
    decoy_set = set(decoy_ids)
    for pair_index, (left, right) in enumerate(pairs):
        unions = [seed[left] | seed[right] for seed in hit_sets]
        overlaps = [
            len(seed[left] & seed[right]) / len(seed[left] | seed[right])
            for seed in hit_sets
        ]
        low_exposure = statistics.mean(len(union & decoy_set) / len(decoy_ids) for union in unions)
        mean_overlap = statistics.mean(overlaps)
        for group_index, series_id in enumerate(series_ids):
            members = membership[series_id]
            coverages = [len(union & members) / len(members) for union in unions]
            utility[group_index, pair_index] = (
                0.40 * min(coverages)
                + 0.30 * statistics.mean(coverages)
                - 0.20 * low_exposure
                - 0.10 * mean_overlap
            )
    return pairs, series_ids, utility, weights


class ConditionalRouting:
    def __init__(
        self,
        receptor_count: int,
        pairs: list[tuple[int, int]],
        utility: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        self.receptor_count = receptor_count
        self.pairs = pairs
        self.utility = utility
        self.weights = weights
        self.pair_index = np.full((receptor_count, receptor_count), -1, dtype=int)
        for index, (left, right) in enumerate(pairs):
            self.pair_index[left, right] = self.pair_index[right, left] = index

    @functools.lru_cache(maxsize=500000)
    def solve(self, opened: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
        pair_indices = np.array(
            [self.pair_index[left, right] for left, right in itertools.combinations(opened, 2)],
            dtype=int,
        )
        if pair_indices.size == 0:
            return -math.inf, ()
        local = self.utility[:, pair_indices]
        choices = np.argmax(local, axis=1)
        assignments = tuple(int(pair_indices[index]) for index in choices)
        value = float(np.dot(self.weights, local[np.arange(len(self.weights)), choices]))
        return value, assignments

    def best_neighbor(
        self, opened: tuple[int, ...], forbidden: set[tuple[int, int]] | None = None
    ) -> tuple[tuple[int, ...], float, tuple[int, int] | None]:
        current_set = set(opened)
        best = opened
        best_value = self.solve(opened)[0]
        best_move = None
        for removed in opened:
            retained = tuple(index for index in opened if index != removed)
            base_pairs = np.array(
                [self.pair_index[a, b] for a, b in itertools.combinations(retained, 2)],
                dtype=int,
            )
            base = (
                np.max(self.utility[:, base_pairs], axis=1)
                if base_pairs.size
                else np.full(len(self.weights), -math.inf)
            )
            for added in range(self.receptor_count):
                if added in current_set:
                    continue
                move = (removed, added)
                if forbidden is not None and move in forbidden:
                    continue
                new_pairs = np.array([self.pair_index[added, other] for other in retained], dtype=int)
                values = np.maximum(base, np.max(self.utility[:, new_pairs], axis=1))
                value = float(np.dot(self.weights, values))
                candidate = tuple(sorted((*retained, added)))
                if value > best_value + TOLERANCE or (
                    abs(value - best_value) <= TOLERANCE and candidate < best
                ):
                    best, best_value, best_move = candidate, value, move
        return best, best_value, best_move


def greedy(routing: ConditionalRouting, k: int) -> tuple[int, ...]:
    best_pair = max(
        routing.pairs,
        key=lambda pair: (routing.solve(pair)[0], tuple(-index for index in pair)),
    )
    opened = tuple(best_pair)
    while len(opened) < k:
        opened_set = set(opened)
        candidates = [tuple(sorted((*opened, index))) for index in range(routing.receptor_count) if index not in opened_set]
        opened = max(candidates, key=lambda state: (routing.solve(state)[0], tuple(-index for index in state)))
    return opened


def one_swap(routing: ConditionalRouting, start: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    opened = start
    iterations = 0
    while True:
        candidate, value, _ = routing.best_neighbor(opened)
        if value <= routing.solve(opened)[0] + TOLERANCE:
            return opened, iterations
        opened = candidate
        iterations += 1


def random_restart(
    routing: ConditionalRouting, k: int, restart_count: int, seed: int
) -> tuple[tuple[int, ...], int, int]:
    rng = random.Random(seed)
    best = None
    best_value = -math.inf
    total_iterations = 0
    local_optima = set()
    for _ in range(restart_count):
        start = tuple(sorted(rng.sample(range(routing.receptor_count), k)))
        selected, iterations = one_swap(routing, start)
        total_iterations += iterations
        local_optima.add(selected)
        value = routing.solve(selected)[0]
        if value > best_value + TOLERANCE or (
            abs(value - best_value) <= TOLERANCE and (best is None or selected < best)
        ):
            best, best_value = selected, value
    if best is None:
        raise RuntimeError("random restart produced no state")
    return best, total_iterations, len(local_optima)


def tabu(routing: ConditionalRouting, k: int, config: dict[str, object]) -> tuple[int, ...]:
    rng = random.Random(int(config["seed"]))
    global_best = None
    global_value = -math.inf
    for _ in range(int(config["restart_count"])):
        current = tuple(sorted(rng.sample(range(routing.receptor_count), k)))
        tabu_until: dict[tuple[int, int], int] = {}
        for step in range(int(config["iteration_count"])):
            forbidden = {move for move, expiry in tabu_until.items() if expiry > step}
            candidate, value, move = routing.best_neighbor(current, forbidden)
            if move is None:
                break
            current = candidate
            tabu_until[(move[1], move[0])] = step + int(config["tabu_tenure"])
            if value > global_value + TOLERANCE or (
                abs(value - global_value) <= TOLERANCE
                and (global_best is None or current < global_best)
            ):
                global_best, global_value = current, value
    if global_best is None:
        raise RuntimeError("tabu produced no state")
    return global_best


def anneal(routing: ConditionalRouting, k: int, config: dict[str, object]) -> tuple[int, ...]:
    rng = random.Random(int(config["seed"]))
    best = None
    best_value = -math.inf
    steps = int(config["steps_per_restart"])
    for _ in range(int(config["restart_count"])):
        current = tuple(sorted(rng.sample(range(routing.receptor_count), k)))
        current_value = routing.solve(current)[0]
        for step in range(steps):
            temperature = 0.02 * (0.0001 / 0.02) ** (step / max(1, steps - 1))
            current_set = set(current)
            removed = rng.choice(current)
            added = rng.choice([index for index in range(routing.receptor_count) if index not in current_set])
            candidate = tuple(sorted((current_set - {removed}) | {added}))
            value = routing.solve(candidate)[0]
            if value >= current_value or rng.random() < math.exp((value - current_value) / temperature):
                current, current_value = candidate, value
            if current_value > best_value + TOLERANCE or (
                abs(current_value - best_value) <= TOLERANCE
                and (best is None or current < best)
            ):
                best, best_value = current, current_value
    if best is None:
        raise RuntimeError("annealing produced no state")
    return best


def solve_milp(
    receptor_count: int,
    pairs: list[tuple[int, int]],
    utility: np.ndarray,
    weights: np.ndarray,
    open_count: int,
    time_limit: float,
) -> dict[str, object]:
    group_count, pair_count = utility.shape
    assignment_offset = receptor_count
    variable_count = receptor_count + group_count * pair_count
    c = np.zeros(variable_count)
    for group in range(group_count):
        start = assignment_offset + group * pair_count
        c[start : start + pair_count] = -weights[group] * utility[group]
    integrality = np.ones(variable_count, dtype=int)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: dict[int, float], lb: float, ub: float) -> None:
        row = len(lower)
        for column, value in coefficients.items():
            row_indices.append(row)
            column_indices.append(column)
            values.append(value)
        lower.append(lb)
        upper.append(ub)

    add({index: 1.0 for index in range(receptor_count)}, open_count, open_count)
    for group in range(group_count):
        start = assignment_offset + group * pair_count
        add({start + pair: 1.0 for pair in range(pair_count)}, 1.0, 1.0)
        for pair_index, (left, right) in enumerate(pairs):
            variable = start + pair_index
            add({variable: 1.0, left: -1.0}, -math.inf, 0.0)
            add({variable: 1.0, right: -1.0}, -math.inf, 0.0)
    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(lower), variable_count),
    ).tocsr()
    started = time.perf_counter()
    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        options={"mip_rel_gap": 0.0, "time_limit": time_limit},
    )
    elapsed = time.perf_counter() - started
    opened: tuple[int, ...] = ()
    objective = math.nan
    if result.x is not None:
        opened = tuple(index for index in range(receptor_count) if result.x[index] > 0.5)
        objective = -float(result.fun)
    gap_raw = getattr(result, "mip_gap", math.nan)
    gap = float(gap_raw) if gap_raw is not None else math.nan
    optimal = bool(result.success and result.status == 0 and gap <= 1e-12)
    return {
        "opened": opened,
        "raw_incumbent_objective": objective,
        "optimal": optimal,
        "mip_gap": gap,
        "mip_node_count": int(getattr(result, "mip_node_count", -1) or -1),
        "dual_bound": -float(getattr(result, "mip_dual_bound", math.nan)),
        "elapsed_seconds": elapsed,
        "message": str(result.message),
        "variable_count": variable_count,
        "constraint_count": len(lower),
    }


def run(root: Path, config_path: Path) -> dict[str, object]:
    root = root.resolve()
    config = common.read_json(config_path.resolve())
    for key in ("stage43_audit", "scores", "ligand_manifest", "frame_manifest"):
        verified(root, config["inputs"][key])
    audit = common.read_json(root / config["inputs"]["stage43_audit"]["path"])
    if audit.get("status") != "stage43_pparg_md96_unidock_matrix_independent_audit_ok":
        raise ValueError("Stage43 independent audit did not pass")
    series_summary = common.read_json(root / config["inputs"]["series_summary"])
    if series_summary.get("status") != "stage95_pparg_active_series_structure_only_ok":
        raise ValueError("Stage95 structure-only series build did not pass")
    if series_summary["docking_score_rows_read"] != 0 or series_summary["linkage"] != "complete":
        raise ValueError("Stage95 series boundary differs")

    scores = common.read_csv(root / config["inputs"]["scores"]["path"])
    ligand_rows = common.read_csv(root / config["inputs"]["ligand_manifest"]["path"])
    frame_rows = common.read_csv(root / config["inputs"]["frame_manifest"]["path"])
    series_rows = common.read_csv(root / config["inputs"]["series_manifest"])
    full_order = build_receptor_order(frame_rows)
    active_ids, decoy_ids, full_hit_sets, hit_count = build_hit_sets(
        scores,
        ligand_rows,
        full_order,
        float(config["objective"]["receptor_hit_fraction_per_seed"]),
    )

    scale_records: list[dict[str, object]] = []
    solution_records: list[dict[str, object]] = []
    passing_scales = []
    for scale_index, scale in enumerate(config["nested_scale_grid"]):
        scale_id = str(scale["scale_id"])
        receptor_count = int(scale["receptor_count"])
        series_count = int(scale["series_count"])
        k = int(scale["open_receptor_count"])
        receptor_ids = full_order[:receptor_count]
        hit_sets = [seed[:receptor_count] for seed in full_hit_sets]
        pairs, series_ids, utility, weights = build_utility(
            receptor_ids, active_ids, decoy_ids, hit_sets, series_rows, series_count
        )
        routing = ConditionalRouting(receptor_count, pairs, utility, weights)
        methods: dict[str, tuple[tuple[int, ...], float, dict[str, object]]] = {}
        print(f"Stage95 {scale_id}: direct and one-swap baselines", flush=True)

        started = time.perf_counter()
        direct = greedy(routing, k)
        methods["direct_greedy"] = (direct, time.perf_counter() - started, {})
        started = time.perf_counter()
        direct_swap, iterations = one_swap(routing, direct)
        methods["greedy_plus_one_swap"] = (
            direct_swap,
            time.perf_counter() - started,
            {"local_search_iterations": iterations},
        )
        preliminary = config["classical_baselines"]["preliminary"]
        random_config = preliminary["random_restart_one_swap"]
        print(f"Stage95 {scale_id}: preliminary random restarts", flush=True)
        started = time.perf_counter()
        random_best, random_iterations, local_count = random_restart(
            routing,
            k,
            int(random_config["restart_count"]),
            int(random_config["seed"]) + scale_index,
        )
        methods["preliminary_random_plus_one_swap"] = (
            random_best,
            time.perf_counter() - started,
            {
                "local_search_iterations": random_iterations,
                "distinct_local_optimum_count": local_count,
            },
        )
        print(f"Stage95 {scale_id}: preliminary tabu", flush=True)
        started = time.perf_counter()
        tabu_best = tabu(routing, k, preliminary["tabu"])
        methods["preliminary_tabu"] = (tabu_best, time.perf_counter() - started, {})
        print(f"Stage95 {scale_id}: preliminary annealing", flush=True)
        started = time.perf_counter()
        anneal_best = anneal(routing, k, preliminary["simulated_annealing"])
        methods["preliminary_simulated_annealing"] = (anneal_best, time.perf_counter() - started, {})

        strong_name, strong_state = max(
            ((name, state) for name, (state, _, _) in methods.items()),
            key=lambda item: (routing.solve(item[1])[0], tuple(-index for index in item[1])),
        )
        strong_value = routing.solve(strong_state)[0]
        print(f"Stage95 {scale_id}: MILP", flush=True)
        exact = solve_milp(
            receptor_count,
            pairs,
            utility,
            weights,
            k,
            float(config["milp"]["time_limit_seconds_per_scale"]),
        )
        if exact["opened"]:
            recomputed = routing.solve(exact["opened"])[0]
            if exact["optimal"] and abs(recomputed - exact["raw_incumbent_objective"]) > 1e-7:
                raise ValueError(f"MILP conditional objective differs for {scale_id}")
        else:
            recomputed = math.nan
        escalation_threshold = float(
            config["quantum_value_gate"]["minimum_certified_relative_objective_gap_over_best_strong_classical"]
        )
        preliminary_relative_gap = (
            (float(recomputed) - strong_value)
            / max(abs(float(recomputed)), 1e-12)
            if exact["optimal"]
            else math.nan
        )
        classical_confirmation_run = bool(
            exact["optimal"] and preliminary_relative_gap >= escalation_threshold
        )
        if classical_confirmation_run:
            print(f"Stage95 {scale_id}: escalating full classical confirmation", flush=True)
            confirmation = config["classical_baselines"]["confirmation"]
            random_config = confirmation["random_restart_one_swap"]
            started = time.perf_counter()
            state, iterations, local_count = random_restart(
                routing,
                k,
                int(random_config["restart_count"]),
                int(random_config["seed"]) + scale_index,
            )
            methods["confirmation_random256_plus_one_swap"] = (
                state,
                time.perf_counter() - started,
                {
                    "local_search_iterations": iterations,
                    "distinct_local_optimum_count": local_count,
                },
            )
            started = time.perf_counter()
            state = tabu(routing, k, confirmation["tabu"])
            methods["confirmation_multistart_tabu"] = (
                state,
                time.perf_counter() - started,
                {},
            )
            started = time.perf_counter()
            state = anneal(routing, k, confirmation["simulated_annealing"])
            methods["confirmation_simulated_annealing"] = (
                state,
                time.perf_counter() - started,
                {},
            )
            strong_name, strong_state = max(
                ((name, state) for name, (state, _, _) in methods.items()),
                key=lambda item: (routing.solve(item[1])[0], tuple(-index for index in item[1])),
            )
            strong_value = routing.solve(strong_state)[0]
        certified_gap = (
            float(recomputed) - strong_value if exact["optimal"] else math.nan
        )
        relative_gap = certified_gap / max(abs(float(recomputed)), 1e-12) if exact["optimal"] else math.nan
        replacement_distance = (
            len(set(exact["opened"]) - set(strong_state)) if exact["optimal"] else -1
        )
        certified_upper_bound = (
            recomputed if exact["optimal"] else float(exact["dual_bound"])
        )
        maximum_possible_gap = max(0.0, certified_upper_bound - strong_value)
        maximum_possible_relative_gap = maximum_possible_gap / max(
            abs(certified_upper_bound), 1e-12
        )
        scale_passed = bool(
            exact["optimal"]
            and relative_gap
            >= float(config["quantum_value_gate"]["minimum_certified_relative_objective_gap_over_best_strong_classical"])
            and replacement_distance
            >= int(config["quantum_value_gate"]["minimum_open_set_replacement_distance"])
        )
        if scale_passed:
            passing_scales.append(scale_id)
        scale_records.append(
            {
                "scale_id": scale_id,
                "receptor_count": receptor_count,
                "series_count": series_count,
                "open_receptor_count": k,
                "pair_count": len(pairs),
                "route_count": 2 * series_count,
                "fixed_panel_route_count": k * series_count,
                "routing_reduction_factor": k / 2,
                "best_strong_classical_method": strong_name,
                "best_strong_classical_objective": strong_value,
                "classical_confirmation_run": classical_confirmation_run,
                "milp_raw_incumbent_objective": exact["raw_incumbent_objective"],
                "milp_conditionally_rerouted_objective": recomputed,
                "milp_optimal": exact["optimal"],
                "milp_gap": exact["mip_gap"],
                "milp_dual_bound": exact["dual_bound"],
                "milp_node_count": exact["mip_node_count"],
                "milp_variable_count": exact["variable_count"],
                "milp_constraint_count": exact["constraint_count"],
                "milp_elapsed_seconds": exact["elapsed_seconds"],
                "certified_objective_gap": certified_gap,
                "certified_relative_gap": relative_gap,
                "certified_upper_bound": certified_upper_bound,
                "maximum_possible_objective_gap": maximum_possible_gap,
                "maximum_possible_relative_gap": maximum_possible_relative_gap,
                "one_percent_gap_mathematically_excluded": maximum_possible_relative_gap
                < escalation_threshold,
                "replacement_distance": replacement_distance,
                "scale_gate_passed": scale_passed,
            }
        )
        for method, (state, seconds, extras) in methods.items():
            value, assignments = routing.solve(state)
            solution_records.append(
                {
                    "scale_id": scale_id,
                    "method": method,
                    "objective": value,
                    "elapsed_seconds": seconds,
                    "open_receptor_ids": ";".join(receptor_ids[index] for index in state),
                    "assignment_pair_ids": ";".join(
                        f"{receptor_ids[pairs[pair][0]]}+{receptor_ids[pairs[pair][1]]}"
                        for pair in assignments
                    ),
                    **extras,
                }
            )
        if exact["opened"]:
            value, assignments = routing.solve(exact["opened"])
            solution_records.append(
                {
                    "scale_id": scale_id,
                    "method": "milp_incumbent" if not exact["optimal"] else "milp_exact",
                    "objective": value,
                    "elapsed_seconds": exact["elapsed_seconds"],
                    "open_receptor_ids": ";".join(receptor_ids[index] for index in exact["opened"]),
                    "assignment_pair_ids": ";".join(
                        f"{receptor_ids[pairs[pair][0]]}+{receptor_ids[pairs[pair][1]]}"
                        for pair in assignments
                    ),
                }
            )
        checkpoint_dir = root / "results/runs/stage95_pparg_md96_series_routing_scaling/checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        common.write_json(
            checkpoint_dir / f"{scale_id}.json",
            {
                "schema_version": "1.0",
                "status": "stage95_scale_checkpoint_ok",
                "scale": scale_records[-1],
                "solution_rows": [
                    row for row in solution_records if row["scale_id"] == scale_id
                ],
            },
        )
        print(json.dumps(scale_records[-1], sort_keys=True), flush=True)

    minimum_passes = int(config["quantum_value_gate"]["minimum_passing_scale_count"])
    hardness_supported = len(passing_scales) >= minimum_passes
    biological_generalization_available = False
    hardware_authorized = hardness_supported and biological_generalization_available
    outputs = config["outputs"]
    scale_path = root / outputs["scale_csv"]
    solution_path = root / outputs["solution_csv"]
    common.write_csv(scale_path, scale_records)
    common.write_csv(solution_path, solution_records)
    result = {
        "schema_version": "1.0",
        "status": (
            "stage95_pparg_md96_series_routing_hardness_supported"
            if hardness_supported
            else "stage95_pparg_md96_series_routing_hardness_not_supported"
        ),
        "experiment_class": config["experiment_class"],
        "real_data_dimensions": {
            "ligands": 160,
            "actives": 80,
            "decoys": 80,
            "receptors": 96,
            "seeds": 3,
            "score_rows": len(scores),
            "hit_count": hit_count,
            "synthetic_scores": 0,
        },
        "scale_count": len(scale_records),
        "passing_scale_ids": passing_scales,
        "scales_with_one_percent_gap_mathematically_excluded": sum(
            bool(row["one_percent_gap_mathematically_excluded"])
            for row in scale_records
        ),
        "minimum_required_passing_scales": minimum_passes,
        "hardness_supported": hardness_supported,
        "biological_generalization_available": biological_generalization_available,
        "quantum_hardware_authorized": hardware_authorized,
        "same_matrix_objective_retuning_authorized": False,
        "data_boundary": {
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_jobs": 0,
        },
        "outputs": {
            "scale_csv": {"path": outputs["scale_csv"], "sha256": common.sha256(scale_path)},
            "solution_csv": {"path": outputs["solution_csv"], "sha256": common.sha256(solution_path)},
        },
        "interpretation": "Stage95 is a post-hoc solver-scaling diagnostic on 46,080 real PPARG Uni-Dock scores. It cannot establish biological generalization, quantum speedup, or quantum advantage.",
    }
    result_path = root / outputs["result_json"]
    common.write_json(result_path, result)
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Stage95 PPARG MD-96 series-routing scaling",
                "",
                f"Status: `{result['status']}`.",
                "",
                f"Passing scales: {len(passing_scales)} of {len(scale_records)}; required: {minimum_passes}.",
                "",
                "All objective values use real Stage43 scores. No synthetic scores, new docking, protected data, or quantum hardware were used.",
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
        default=Path("configs/stage95_pparg_md96_series_routing_scaling.json"),
    )
    args = parser.parse_args()
    run(args.root, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
