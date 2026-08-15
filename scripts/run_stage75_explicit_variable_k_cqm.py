"""Build and benchmark an explicit variable-k constraint-native CQM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import dimod
import numpy as np


TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
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
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any], label: str) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage75 frozen {label} identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage75 frozen {label} size differs: {path}")
    return path


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def load_model(record: dict[str, Any]) -> dict[str, Any]:
    receptor_ids = tuple(str(value) for value in record["receptor_ids"])
    count = len(receptor_ids)
    pairs = tuple(itertools.combinations(range(count), 2))
    centered = np.asarray(record["centered_pair_coefficients"], dtype=float)
    midpoint = float(record["pair_midpoint_center"])
    raw = centered + midpoint
    if len(raw) != len(pairs):
        raise ValueError("Stage75 pair coefficient count differs")
    matrix = np.zeros((count, count), dtype=float)
    for (left, right), value in zip(pairs, raw):
        matrix[left, right] = value
        matrix[right, left] = value
    deficits = np.asarray(record["integer_deficits"], dtype=int)
    canonical = {
        "variable_order": [f"x{index:03d}" for index in range(count)],
        "objective": {
            "pair_center": midpoint,
            "quadratic_pair_order": [list(pair) for pair in pairs],
            "quadratic_coefficients": [float(value) for value in centered],
            "offset": float(record["objective_offset"]),
        },
        "constraints": {
            "cardinality_exact": {
                "sense": "==",
                "linear_coefficients": [1] * count,
                "rhs": int(record["reference_k"]),
            },
            "quality_floor": {
                "sense": "<=",
                "linear_coefficients": [int(value) for value in deficits],
                "rhs": int(record["maximum_integer_deficit"]),
            },
        },
    }
    if canonical_sha256(canonical) != str(record["cqm_sha256"]).upper():
        raise ValueError("Stage75 failed to reconstruct a Stage72 source CQM")
    pair_scale = float(max(np.max(np.abs(raw)), 1e-12))
    return {
        "record": record,
        "receptor_ids": receptor_ids,
        "id_to_index": {value: index for index, value in enumerate(receptor_ids)},
        "count": count,
        "pairs": pairs,
        "raw_coefficients": raw,
        "matrix": matrix,
        "deficits": deficits,
        "pair_scale": pair_scale,
    }


def parse_subset(model: dict[str, Any], value: str) -> tuple[int, ...]:
    subset = tuple(sorted(model["id_to_index"][item] for item in value.split("+")))
    if not subset:
        raise ValueError("Stage75 parsed an empty subset")
    return subset


def subset_name(model: dict[str, Any], subset: tuple[int, ...]) -> str:
    return "+".join(model["receptor_ids"][index] for index in subset)


def subset_deficit(model: dict[str, Any], subset: tuple[int, ...]) -> int:
    return int(sum(int(model["deficits"][index]) for index in subset))


def raw_objective(model: dict[str, Any], subset: tuple[int, ...]) -> float:
    return float(
        sum(
            model["matrix"][left, right]
            for left, right in itertools.combinations(subset, 2)
        )
    )


def variable_energy(
    model: dict[str, Any], subset: tuple[int, ...], reward: float
) -> float:
    return raw_objective(model, subset) - reward * math.comb(len(subset), 2)


def reward_order_statistic(model: dict[str, Any], quantile: float) -> dict[str, Any]:
    values = np.sort(model["raw_coefficients"])
    position = max(0, math.ceil(quantile * len(values)) - 1)
    return {
        "quantile": quantile,
        "order_statistic_index": position,
        "reward": float(values[position]),
    }


def source_frontiers(
    model: dict[str, Any],
    workloads: list[dict[str, str]],
    comparisons: list[dict[str, str]],
    trials: list[dict[str, str]],
    quality_regime: str,
) -> dict[int, dict[str, Any]]:
    target = str(model["record"]["target_id"])
    fold = int(model["record"]["outer_fold"])
    selected_workloads = [
        row
        for row in workloads
        if row["target_id"] == target
        and int(row["outer_fold"]) == fold
        and row["quality_regime"] == quality_regime
    ]
    output: dict[int, dict[str, Any]] = {}
    for workload in selected_workloads:
        k = int(workload["k"])
        comparison = next(
            row
            for row in comparisons
            if row["target_id"] == target
            and int(row["outer_fold"]) == fold
            and int(row["k"]) == k
            and row["quality_regime"] == quality_regime
        )
        local_trials = [
            row
            for row in trials
            if row["target_id"] == target
            and int(row["outer_fold"]) == fold
            and int(row["k"]) == k
            and row["quality_regime"] == quality_regime
        ]
        reference_candidates = [row for row in local_trials if truth(row["reference_match"])]
        if not reference_candidates:
            raise ValueError("Stage75 cannot recover a Stage74 frontier subset")
        reference_row = min(
            reference_candidates,
            key=lambda row: (
                0 if row["method"] == "exact_enumeration" else 1,
                row["solution_subset"],
            ),
        )
        deterministic_row = next(
            row
            for row in local_trials
            if row["method"] == "deterministic_best_improvement"
            and int(row["repeat"]) == 0
        )
        reference_subset = parse_subset(model, reference_row["solution_subset"])
        deterministic_subset = parse_subset(model, deterministic_row["solution_subset"])
        reference_value = float(comparison["reference_objective"])
        if not math.isclose(
            raw_objective(model, reference_subset),
            reference_value,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError("Stage75 recovered frontier objective differs")
        output[k] = {
            "k": k,
            "quality_threshold": int(workload["quality_threshold"]),
            "reference_type": str(comparison["reference_type"]),
            "reference_objective": reference_value,
            "reference_subset": reference_subset,
            "deterministic_objective": float(deterministic_row["solution_objective"]),
            "deterministic_subset": deterministic_subset,
            "fixed_k_total_state_count": int(workload["total_fixed_k_subset_count"]),
            "fixed_k_feasible_state_count": int(workload["feasible_subset_count"]),
        }
    return dict(sorted(output.items()))


def cqm_canonical(
    model: dict[str, Any], frontiers: dict[int, dict[str, Any]], reward: float
) -> dict[str, Any]:
    x_names = [f"x{index:03d}" for index in range(model["count"])]
    y_names = [f"y_k{k:02d}" for k in frontiers]
    shifted = [float(value - reward) for value in model["raw_coefficients"]]
    return {
        "variable_order": x_names + y_names,
        "objective": {
            "linear_coefficients": [0.0] * (len(x_names) + len(y_names)),
            "quadratic_pair_order": [list(pair) for pair in model["pairs"]],
            "quadratic_coefficients": shifted,
            "offset": 0.0,
        },
        "constraints": {
            "one_hot_cardinality": {
                "sense": "==",
                "coefficients": {name: 1 for name in y_names},
                "rhs": 1,
            },
            "cardinality_link": {
                "sense": "==",
                "x_coefficients": [1] * model["count"],
                "y_coefficients": {
                    f"y_k{k:02d}": -int(k) for k in frontiers
                },
                "rhs": 0,
            },
            "conditional_quality": {
                "sense": "<=",
                "x_coefficients": [int(value) for value in model["deficits"]],
                "y_coefficients": {
                    f"y_k{k:02d}": -int(value["quality_threshold"])
                    for k, value in frontiers.items()
                },
                "rhs": 0,
            },
        },
    }


def build_cqm(
    model: dict[str, Any], frontiers: dict[int, dict[str, Any]], reward: float
) -> dimod.ConstrainedQuadraticModel:
    cqm = dimod.ConstrainedQuadraticModel()
    x = [dimod.Binary(f"x{index:03d}") for index in range(model["count"])]
    y = {k: dimod.Binary(f"y_k{k:02d}") for k in frontiers}
    objective = dimod.quicksum(
        float(value - reward) * x[left] * x[right]
        for (left, right), value in zip(model["pairs"], model["raw_coefficients"])
    )
    cqm.set_objective(objective)
    cqm.add_constraint(dimod.quicksum(y.values()) == 1, label="one_hot_cardinality")
    cqm.add_constraint(
        dimod.quicksum(x)
        - dimod.quicksum(int(k) * value for k, value in y.items())
        == 0,
        label="cardinality_link",
    )
    cqm.add_constraint(
        dimod.quicksum(
            int(model["deficits"][index]) * x[index]
            for index in range(model["count"])
        )
        - dimod.quicksum(
            int(frontiers[k]["quality_threshold"]) * value
            for k, value in y.items()
        )
        <= 0,
        label="conditional_quality",
    )
    return cqm


def assignment(
    model: dict[str, Any], frontiers: dict[int, dict[str, Any]], subset: tuple[int, ...]
) -> dict[str, int]:
    chosen = set(subset)
    sample = {
        f"x{index:03d}": int(index in chosen) for index in range(model["count"])
    }
    sample.update({f"y_k{k:02d}": int(k == len(subset)) for k in frontiers})
    return sample


def valid(cell: dict[str, Any], subset: tuple[int, ...]) -> bool:
    k = len(subset)
    return (
        k in cell["frontiers"]
        and len(set(subset)) == k
        and subset_deficit(cell["model"], subset)
        <= int(cell["frontiers"][k]["quality_threshold"])
    )


def random_feasible_start(
    cell: dict[str, Any], rng: np.random.Generator, maximum_attempts: int
) -> tuple[tuple[int, ...], int, bool]:
    allowed = tuple(cell["frontiers"])
    for attempt in range(1, maximum_attempts + 1):
        k = allowed[int(rng.integers(0, len(allowed)))]
        subset = tuple(
            sorted(
                int(value)
                for value in rng.choice(cell["model"]["count"], k, replace=False)
            )
        )
        if valid(cell, subset):
            return subset, attempt, False
    smallest = min(allowed)
    fallback = min(
        itertools.combinations(range(cell["model"]["count"]), smallest),
        key=lambda subset: (subset_deficit(cell["model"], subset), subset),
    )
    if not valid(cell, fallback):
        raise ValueError("Stage75 deterministic initialization fallback is infeasible")
    return fallback, maximum_attempts, True


def propose(
    cell: dict[str, Any], current: tuple[int, ...], rng: np.random.Generator
) -> tuple[int, ...]:
    swap_probability = float(cell["solver_protocol"]["swap_move_probability"])
    if rng.random() < swap_probability:
        outgoing = current[int(rng.integers(0, len(current)))]
        selected = set(current)
        incoming = int(rng.integers(0, cell["model"]["count"]))
        while incoming in selected:
            incoming = int(rng.integers(0, cell["model"]["count"]))
        return tuple(sorted((selected - {outgoing}) | {incoming}))
    allowed = [value for value in cell["frontiers"] if value != len(current)]
    target_k = allowed[int(rng.integers(0, len(allowed)))]
    return tuple(
        sorted(
            int(value)
            for value in rng.choice(
                cell["model"]["count"], target_k, replace=False
            )
        )
    )


def budgeted_variable_tabu(
    cell: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    protocol = cell["solver_protocol"]
    budget = int(protocol["proposal_budget"])
    batch_size = int(protocol["tabu_candidate_batch_size"])
    tenure = int(protocol["tabu_tenure"])
    current, attempts, fallback = random_feasible_start(
        cell, rng, int(protocol["maximum_initialization_attempts"])
    )
    current_energy = variable_energy(cell["model"], current, cell["reward"])
    best, best_energy = current, current_energy
    proposals = feasible_proposals = 0
    evaluations = 1
    accepted = 0
    restarts = 1
    tabu_queue: deque[tuple[int, ...]] = deque(maxlen=tenure)
    tabu_set: set[tuple[int, ...]] = set()
    while proposals < budget:
        candidates: list[tuple[float, tuple[int, ...]]] = []
        draw = min(batch_size, budget - proposals)
        for _ in range(draw):
            candidate = propose(cell, current, rng)
            proposals += 1
            if not valid(cell, candidate):
                continue
            feasible_proposals += 1
            evaluations += 1
            energy = variable_energy(cell["model"], candidate, cell["reward"])
            if candidate not in tabu_set or energy < best_energy - TOLERANCE:
                candidates.append((energy, candidate))
        if not candidates:
            current, extra, used_fallback = random_feasible_start(
                cell, rng, int(protocol["maximum_initialization_attempts"])
            )
            attempts += extra
            fallback = fallback or used_fallback
            current_energy = variable_energy(cell["model"], current, cell["reward"])
            evaluations += 1
            restarts += 1
            continue
        energy, candidate = min(candidates)
        if len(tabu_queue) == tabu_queue.maxlen:
            removed = tabu_queue.popleft()
            tabu_set.discard(removed)
        tabu_queue.append(current)
        tabu_set.add(current)
        current, current_energy = candidate, energy
        accepted += 1
        if (current_energy, current) < (best_energy, best):
            best, best_energy = current, current_energy
    return {
        "subset": best,
        "proposal_count": proposals,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "restart_count": restarts,
        "initialization_attempt_count": attempts,
        "initialization_fallback_count": int(fallback),
    }


def variable_annealing(
    cell: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    protocol = cell["solver_protocol"]
    budget = int(protocol["proposal_budget"])
    beta_minimum, beta_maximum = [
        float(value) for value in protocol["annealing_beta_range"]
    ]
    current, attempts, fallback = random_feasible_start(
        cell, rng, int(protocol["maximum_initialization_attempts"])
    )
    current_energy = variable_energy(cell["model"], current, cell["reward"])
    best, best_energy = current, current_energy
    feasible_proposals = 0
    evaluations = 1
    accepted = 0
    denominator = cell["model"]["pair_scale"] * math.comb(max(cell["frontiers"]), 2)
    for step in range(budget):
        candidate = propose(cell, current, rng)
        if not valid(cell, candidate):
            continue
        feasible_proposals += 1
        evaluations += 1
        energy = variable_energy(cell["model"], candidate, cell["reward"])
        normalized_delta = (energy - current_energy) / denominator
        fraction = step / max(1, budget - 1)
        beta = beta_minimum * (beta_maximum / beta_minimum) ** fraction
        if normalized_delta <= 0 or rng.random() < math.exp(-beta * normalized_delta):
            current, current_energy = candidate, energy
            accepted += 1
            if (current_energy, current) < (best_energy, best):
                best, best_energy = current, current_energy
    return {
        "subset": best,
        "proposal_count": budget,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "restart_count": 1,
        "initialization_attempt_count": attempts,
        "initialization_fallback_count": int(fallback),
    }


def seed_for(base: int, cell_index: int, method_index: int, repeat: int) -> int:
    return int(base + cell_index * 100_000 + method_index * 1_000 + repeat)


def fixed_candidate(
    cell: dict[str, Any], source_field: str
) -> tuple[tuple[int, ...], float, str]:
    candidates = []
    for k, record in cell["frontiers"].items():
        subset = tuple(record[f"{source_field}_subset"])
        energy = variable_energy(cell["model"], subset, cell["reward"])
        candidates.append((energy, subset, str(record.get("reference_type", ""))))
    energy, subset, reference_type = min(
        candidates, key=lambda value: (value[0], value[1])
    )
    return subset, energy, reference_type


def trial_row(
    cell: dict[str, Any], method: str, repeat: int, seed: int, solved: dict[str, Any]
) -> dict[str, Any]:
    subset = tuple(solved["subset"])
    if not valid(cell, subset):
        raise ValueError("Stage75 solver returned an invalid subset")
    energy = variable_energy(cell["model"], subset, cell["reward"])
    return {
        "target_id": cell["model"]["record"]["target_id"],
        "outer_fold": int(cell["model"]["record"]["outer_fold"]),
        "reward_quantile": float(cell["reward_quantile"]),
        "reward_value": float(cell["reward"]),
        "method": method,
        "repeat": repeat,
        "seed": seed,
        **{key: int(value) for key, value in solved.items() if key != "subset"},
        "selected_k": len(subset),
        "solution_subset": subset_name(cell["model"], subset),
        "solution_deficit": subset_deficit(cell["model"], subset),
        "solution_raw_pair_objective": raw_objective(cell["model"], subset),
        "solution_variable_energy": energy,
    }


def run_cell(
    cell: dict[str, Any], cell_index: int, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_seed = int(config["solver_protocol"]["seed_base"])
    frontier_subset, _, _ = fixed_candidate(cell, "reference")
    frontier_type = (
        "exact_fixed_k_frontier"
        if all(
            value["reference_type"] == "exact_enumeration"
            for value in cell["frontiers"].values()
        )
        else "pooled_best_known_fixed_k_frontier"
    )
    deterministic_subset, _, _ = fixed_candidate(cell, "deterministic")
    fixed_methods = [
        ("fixed_k_frontier_reference", frontier_subset),
        ("decomposed_deterministic_baseline", deterministic_subset),
    ]
    for method_index, (method, subset) in enumerate(fixed_methods):
        solved = {
            "subset": subset,
            "proposal_count": 0,
            "feasible_proposal_count": 0,
            "objective_evaluation_count": len(cell["frontiers"]),
            "accepted_move_count": 0,
            "restart_count": len(cell["frontiers"]),
            "initialization_attempt_count": 0,
            "initialization_fallback_count": 0,
        }
        rows.append(
            trial_row(
                cell,
                method,
                0,
                seed_for(base_seed, cell_index, method_index, 0),
                solved,
            )
        )
    stochastic = [
        ("budgeted_variable_tabu", budgeted_variable_tabu),
        ("constraint_native_variable_annealing", variable_annealing),
    ]
    repeats = int(config["solver_protocol"]["stochastic_repeats"])
    for method_index, (method, solver) in enumerate(stochastic, start=len(fixed_methods)):
        for repeat in range(repeats):
            seed = seed_for(base_seed, cell_index, method_index, repeat)
            rows.append(
                trial_row(
                    cell,
                    method,
                    repeat,
                    seed,
                    solver(cell, np.random.default_rng(seed)),
                )
            )
    frontier_energy = min(
        float(row["solution_variable_energy"])
        for row in rows
        if row["method"] == "fixed_k_frontier_reference"
    )
    pooled_best = min(float(row["solution_variable_energy"]) for row in rows)
    normalization = cell["model"]["pair_scale"] * math.comb(
        max(cell["frontiers"]), 2
    )
    for row in rows:
        row["frontier_reference_type"] = frontier_type
        row["frozen_frontier_energy"] = frontier_energy
        row["delta_vs_frozen_frontier_normalized"] = (
            float(row["solution_variable_energy"]) - frontier_energy
        ) / normalization
        row["pooled_best_energy"] = pooled_best
        row["delta_vs_pooled_best_normalized"] = (
            float(row["solution_variable_energy"]) - pooled_best
        ) / normalization
        row["pooled_best_match"] = (
            float(row["solution_variable_energy"]) <= pooled_best + TOLERANCE
        )
    tolerance = float(config["benchmark_gate"]["normalized_energy_tolerance"])
    method_best = {
        method: min(
            (row for row in rows if row["method"] == method),
            key=lambda row: (
                float(row["solution_variable_energy"]),
                row["solution_subset"],
            ),
        )
        for method in {row["method"] for row in rows}
    }
    joint = method_best["budgeted_variable_tabu"]
    sampler = method_best["constraint_native_variable_annealing"]
    sampler_vs_joint = (
        float(sampler["solution_variable_energy"])
        - float(joint["solution_variable_energy"])
    ) / normalization
    sampler_vs_frontier = (
        float(sampler["solution_variable_energy"]) - frontier_energy
    ) / normalization
    comparison = {
        "target_id": cell["model"]["record"]["target_id"],
        "outer_fold": int(cell["model"]["record"]["outer_fold"]),
        "reward_quantile": float(cell["reward_quantile"]),
        "reward_value": float(cell["reward"]),
        "frontier_reference_type": frontier_type,
        "frozen_frontier_selected_k": int(
            method_best["fixed_k_frontier_reference"]["selected_k"]
        ),
        "frozen_frontier_energy": frontier_energy,
        "decomposed_deterministic_selected_k": int(
            method_best["decomposed_deterministic_baseline"]["selected_k"]
        ),
        "decomposed_deterministic_energy": float(
            method_best["decomposed_deterministic_baseline"][
                "solution_variable_energy"
            ]
        ),
        "joint_classical_best_selected_k": int(joint["selected_k"]),
        "joint_classical_best_energy": float(joint["solution_variable_energy"]),
        "sampler_best_selected_k": int(sampler["selected_k"]),
        "sampler_best_energy": float(sampler["solution_variable_energy"]),
        "sampler_delta_vs_joint_classical_normalized": sampler_vs_joint,
        "sampler_delta_vs_frozen_frontier_normalized": sampler_vs_frontier,
        "sampler_within_joint_classical_tolerance": sampler_vs_joint <= tolerance,
        "sampler_within_frozen_frontier_tolerance": sampler_vs_frontier <= tolerance,
        "sampler_strict_win_vs_joint_classical": sampler_vs_joint < -tolerance,
        "joint_classical_strict_win_vs_sampler": sampler_vs_joint > tolerance,
        "frozen_frontier_refined": pooled_best < frontier_energy - TOLERANCE,
        "pooled_best_energy": pooled_best,
    }
    exact_frontier = frontier_type == "exact_fixed_k_frontier"
    comparison["exact_frontier_available"] = exact_frontier
    comparison["joint_classical_exact_frontier_match"] = (
        exact_frontier
        and float(joint["solution_variable_energy"]) <= frontier_energy + TOLERANCE
    )
    comparison["sampler_exact_frontier_match"] = (
        exact_frontier
        and float(sampler["solution_variable_energy"]) <= frontier_energy + TOLERANCE
    )
    cqm = cell["cqm"]
    sample = assignment(cell["model"], cell["frontiers"], frontier_subset)
    cqm_energy = float(cqm.objective.energy(sample))
    residual = abs(cqm_energy - frontier_energy)
    if not cqm.check_feasible(sample) or residual > 1e-9:
        raise ValueError("Stage75 CQM assignment validation failed")
    model_row = {
        "target_id": cell["model"]["record"]["target_id"],
        "outer_fold": int(cell["model"]["record"]["outer_fold"]),
        "reward_quantile": float(cell["reward_quantile"]),
        "reward_order_statistic_index": int(cell["reward_order_statistic_index"]),
        "reward_value": float(cell["reward"]),
        "allowed_k": "+".join(str(value) for value in cell["frontiers"]),
        "candidate_variable_count": int(cell["model"]["count"]),
        "cardinality_selector_variable_count": len(cell["frontiers"]),
        "total_logical_variable_count": int(cqm.num_variables()),
        "quadratic_coupler_count": len(cell["model"]["pairs"]),
        "explicit_constraint_count": len(cqm.constraints),
        "cqm_sha256": cell["cqm_sha256"],
        "frontier_assignment_feasible": True,
        "frontier_energy_encoding_residual": residual,
        "minimum_shifted_coefficient": float(
            np.min(cell["model"]["raw_coefficients"] - cell["reward"])
        ),
        "maximum_shifted_coefficient": float(
            np.max(cell["model"]["raw_coefficients"] - cell["reward"])
        ),
    }
    return rows, comparison, model_row


def summarize_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for scope in ("ALL", str(row["target_id"])):
            groups[(str(row["method"]), float(row["reward_quantile"]), scope)].append(row)
    output: list[dict[str, Any]] = []
    for (method, quantile, scope), selected in sorted(groups.items()):
        output.append(
            {
                "method": method,
                "reward_quantile": quantile,
                "scope": scope,
                "trial_count": len(selected),
                "cell_count": len(
                    {(row["target_id"], row["outer_fold"]) for row in selected}
                ),
                "pooled_best_match_rate": statistics.fmean(
                    bool(row["pooled_best_match"]) for row in selected
                ),
                "mean_delta_vs_frozen_frontier_normalized": statistics.fmean(
                    float(row["delta_vs_frozen_frontier_normalized"])
                    for row in selected
                ),
                "maximum_delta_vs_frozen_frontier_normalized": max(
                    float(row["delta_vs_frozen_frontier_normalized"])
                    for row in selected
                ),
                "mean_selected_k": statistics.fmean(
                    int(row["selected_k"]) for row in selected
                ),
                "mean_feasible_proposal_count": statistics.fmean(
                    int(row["feasible_proposal_count"]) for row in selected
                ),
                "initialization_fallback_count": sum(
                    int(row["initialization_fallback_count"]) for row in selected
                ),
            }
        )
    return output


def aggregate(
    model_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(
        comparisons,
        key=lambda row: (
            row["target_id"],
            row["outer_fold"],
            row["reward_quantile"],
        ),
    )
    paths: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        paths[(row["target_id"], int(row["outer_fold"]))].append(row)
    monotonic = {
        key: all(
            int(right["frozen_frontier_selected_k"])
            >= int(left["frozen_frontier_selected_k"])
            for left, right in zip(rows, rows[1:])
        )
        for key, rows in paths.items()
    }
    exact = [row for row in comparisons if bool(row["exact_frontier_available"])]
    gate = config["benchmark_gate"]
    sampler_joint = statistics.fmean(
        bool(row["sampler_within_joint_classical_tolerance"])
        for row in comparisons
    )
    sampler_frontier = statistics.fmean(
        bool(row["sampler_within_frozen_frontier_tolerance"])
        for row in comparisons
    )
    joint_exact = statistics.fmean(
        bool(row["joint_classical_exact_frontier_match"]) for row in exact
    )
    sampler_exact = statistics.fmean(
        bool(row["sampler_exact_frontier_match"]) for row in exact
    )
    encoding = {
        "cqm_model_count": len(model_rows),
        "maximum_logical_variable_count": max(
            int(row["total_logical_variable_count"]) for row in model_rows
        ),
        "maximum_quadratic_coupler_count": max(
            int(row["quadratic_coupler_count"]) for row in model_rows
        ),
        "explicit_constraint_count_per_model": 3,
        "maximum_frontier_energy_encoding_residual": max(
            float(row["frontier_energy_encoding_residual"]) for row in model_rows
        ),
        "all_frontier_assignments_feasible": all(
            bool(row["frontier_assignment_feasible"]) for row in model_rows
        ),
        "reward_path_count": len(paths),
        "monotonic_reward_path_count": sum(monotonic.values()),
        "distinct_frontier_selected_k": sorted(
            {int(row["frozen_frontier_selected_k"]) for row in comparisons}
        ),
    }
    performance = {
        "comparison_cell_count": len(comparisons),
        "exact_frontier_cell_count": len(exact),
        "joint_classical_exact_frontier_match_rate": joint_exact,
        "sampler_exact_frontier_match_rate": sampler_exact,
        "sampler_joint_classical_competitive_fraction": sampler_joint,
        "sampler_frozen_frontier_competitive_fraction": sampler_frontier,
        "sampler_strict_win_vs_joint_classical_cell_count": sum(
            bool(row["sampler_strict_win_vs_joint_classical"])
            for row in comparisons
        ),
        "joint_classical_strict_win_vs_sampler_cell_count": sum(
            bool(row["joint_classical_strict_win_vs_sampler"])
            for row in comparisons
        ),
        "joint_sampler_tie_cell_count": sum(
            not bool(row["sampler_strict_win_vs_joint_classical"])
            and not bool(row["joint_classical_strict_win_vs_sampler"])
            for row in comparisons
        ),
        "frozen_frontier_refined_cell_count": sum(
            bool(row["frozen_frontier_refined"]) for row in comparisons
        ),
    }
    encoding_gate = bool(
        encoding["all_frontier_assignments_feasible"]
        and encoding["maximum_frontier_energy_encoding_residual"]
        <= float(gate["maximum_energy_encoding_residual"])
        and encoding["monotonic_reward_path_count"]
        == int(gate["required_monotonic_reward_path_count"])
        and len(encoding["distinct_frontier_selected_k"])
        >= int(gate["minimum_distinct_selected_k"])
    )
    exact_gate = bool(
        joint_exact >= float(gate["minimum_joint_classical_exact_match_rate"])
        and sampler_exact >= float(gate["minimum_sampler_exact_match_rate"])
    )
    sampler_gate = bool(
        sampler_joint >= float(gate["minimum_sampler_joint_competitive_fraction"])
        and sampler_frontier
        >= float(gate["minimum_sampler_frontier_competitive_fraction"])
    )
    return {
        "encoding_summary": encoding,
        "solver_performance": performance,
        "route_gate": {
            "explicit_variable_k_cqm_encoding_passed": encoding_gate,
            "exact_frontier_solver_validation_passed": exact_gate,
            "variable_k_sampler_competitiveness_passed": sampler_gate,
        },
        "decision": {
            "explicit_variable_k_cqm_freeze_authorized": encoding_gate,
            "local_hardware_shaped_emulation_authorized": bool(
                encoding_gate and exact_gate and sampler_gate
            ),
            "cloud_cqm_execution_authorized": False,
            "direct_qpu_execution_authorized": False,
            "quantum_scaling_claim_authorized": False,
            "quantum_advantage_claim_authorized": False,
        },
    }


def report_text(result: dict[str, Any]) -> str:
    encoding = result["encoding_summary"]
    performance = result["solver_performance"]
    return rf"""# Stage75 explicit variable-k constraint-native CQM

## Formulation

For reward level $\rho$, minimize

$$
E_\rho(x)=\sum_{{i<j}}(r_{{ij}}-\rho)x_ix_j
$$

subject to one-hot cardinality selection, $\sum_i x_i=\sum_k k y_k$, and the Stage74 balanced-10% conditional quality constraint $\sum_i d_ix_i\leq\sum_k D_k y_k$. Reward levels are frozen pair-coefficient order statistics at 10%, 25%, 50%, 75%, and 90%.

## Encoding

- CQM models: `{encoding['cqm_model_count']}`.
- Maximum logical variables / quadratic couplers: `{encoding['maximum_logical_variable_count']}` / `{encoding['maximum_quadratic_coupler_count']}`.
- Explicit constraints: `3`.
- Maximum energy-identity residual: `{encoding['maximum_frontier_energy_encoding_residual']:.3e}`.
- Monotonic reward paths: `{encoding['monotonic_reward_path_count']}/{encoding['reward_path_count']}`.
- Distinct selected budgets: `{encoding['distinct_frontier_selected_k']}`.

## Solver comparison

- Exact fixed-$k$ frontier cells: `{performance['exact_frontier_cell_count']}`.
- Joint classical exact-frontier match: `{performance['joint_classical_exact_frontier_match_rate']:.3f}`.
- Variable annealing exact-frontier match: `{performance['sampler_exact_frontier_match_rate']:.3f}`.
- Annealing competitive with joint classical: `{performance['sampler_joint_classical_competitive_fraction']:.3f}`.
- Annealing competitive with frozen fixed-$k$ frontier: `{performance['sampler_frozen_frontier_competitive_fraction']:.3f}`.
- Annealing wins / joint-classical wins / ties: `{performance['sampler_strict_win_vs_joint_classical_cell_count']}` / `{performance['joint_classical_strict_win_vs_sampler_cell_count']}` / `{performance['joint_sampler_tie_cell_count']}`.
- Stage74 pooled frontiers refined: `{performance['frozen_frontier_refined_cell_count']}`.

## Decision

- Explicit variable-$k$ CQM freeze authorized: `{result['decision']['explicit_variable_k_cqm_freeze_authorized']}`.
- Local hardware-shaped emulation authorized: `{result['decision']['local_hardware_shaped_emulation_authorized']}`.
- Cloud CQM / direct QPU / quantum claims authorized: `False / False / False`.

The reward path is an optimization trade-off study, not a biological estimate of the best receptor count. Non-exact Stage74 frontiers remain pooled best-known references and may be improved by Stage75 joint solvers.
"""


def compute(config: dict[str, Any], root: Path) -> dict[str, Any]:
    implementation_paths = {
        key: verified(root, value, key)
        for key, value in config["implementation"].items()
    }
    input_paths = {
        key: verified(root, value, key) for key, value in config["inputs"].items()
    }
    stage74_result = read_json(input_paths["stage74_result"])
    stage74_audit = read_json(input_paths["stage74_audit"])
    if not stage74_result["decision"]["explicit_variable_k_cqm_design_authorized"]:
        raise ValueError("Stage75 requires Stage74 variable-k design authorization")
    if stage74_audit.get("status") != (
        "stage74_larger_k_solver_scaling_independent_audit_ok"
    ):
        raise ValueError("Stage75 requires the Stage74 independent audit")
    source = read_json(input_paths["stage72_model_record"])
    workloads = read_csv(input_paths["stage74_workload_metrics"])
    comparisons_source = read_csv(input_paths["stage74_cell_comparison"])
    source_trials = read_csv(input_paths["stage74_solver_trials"])
    models = [load_model(record) for record in source["models"]]
    quality_regime = str(config["variable_k_cqm"]["quality_regime"])
    trial_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    cell_index = 0
    for model in models:
        frontiers = source_frontiers(
            model, workloads, comparisons_source, source_trials, quality_regime
        )
        for quantile in config["variable_k_cqm"]["reward_quantiles"]:
            reward_record = reward_order_statistic(model, float(quantile))
            reward = float(reward_record["reward"])
            canonical = cqm_canonical(model, frontiers, reward)
            cqm = build_cqm(model, frontiers, reward)
            cell = {
                "model": model,
                "frontiers": frontiers,
                "reward_quantile": float(quantile),
                "reward_order_statistic_index": int(
                    reward_record["order_statistic_index"]
                ),
                "reward": reward,
                "cqm": cqm,
                "cqm_sha256": canonical_sha256(canonical),
                "solver_protocol": config["solver_protocol"],
            }
            local_trials, comparison, model_row = run_cell(
                cell, cell_index, config
            )
            trial_rows.extend(local_trials)
            comparison_rows.append(comparison)
            model_rows.append(model_row)
            model_records.append(
                {
                    "target_id": model["record"]["target_id"],
                    "outer_fold": int(model["record"]["outer_fold"]),
                    "reward_quantile": float(quantile),
                    "reward_order_statistic_index": int(
                        reward_record["order_statistic_index"]
                    ),
                    "reward_value": reward,
                    "allowed_k": list(frontiers),
                    "quality_thresholds": {
                        str(k): int(value["quality_threshold"])
                        for k, value in frontiers.items()
                    },
                    "cqm_sha256": cell["cqm_sha256"],
                    "frontier_selected_k": int(
                        comparison["frozen_frontier_selected_k"]
                    ),
                    "frontier_reference_type": comparison[
                        "frontier_reference_type"
                    ],
                }
            )
            cell_index += 1
        print(
            json.dumps(
                {
                    "target_id": model["record"]["target_id"],
                    "outer_fold": model["record"]["outer_fold"],
                    "cqm_models_completed": len(model_rows),
                    "solver_trials_completed": len(trial_rows),
                }
            ),
            flush=True,
        )
    expected = int(config["benchmark_gate"]["required_cqm_model_count"])
    if len(model_rows) != expected:
        raise ValueError(f"Stage75 CQM count differs: {len(model_rows)} != {expected}")
    summaries = summarize_methods(trial_rows)
    aggregate_value = aggregate(model_rows, comparison_rows, config)
    output_paths = {
        key: root / str(value) for key, value in config["outputs"].items()
    }
    write_csv(output_paths["cqm_metrics_csv"], model_rows)
    write_csv(output_paths["solver_trials_csv"], trial_rows)
    write_csv(output_paths["cell_comparison_csv"], comparison_rows)
    write_csv(output_paths["solver_summary_csv"], summaries)
    model_record = {
        "schema_version": "1.0",
        "algorithm_id": config["variable_k_cqm"]["algorithm_id"],
        "model_count": len(model_records),
        "models": model_records,
    }
    write_json(output_paths["model_record_json"], model_record)
    payload = {
        **aggregate_value,
        "cqm_metrics_sha256": sha256(output_paths["cqm_metrics_csv"]),
        "solver_trials_sha256": sha256(output_paths["solver_trials_csv"]),
        "cell_comparison_sha256": sha256(output_paths["cell_comparison_csv"]),
        "solver_summary_sha256": sha256(output_paths["solver_summary_csv"]),
        "model_record_sha256": sha256(output_paths["model_record_json"]),
    }
    result = {
        "schema_version": "1.0",
        "status": "stage75_explicit_variable_k_cqm_complete",
        "experiment_class": (
            "post-hoc explicit variable-cardinality constraint-native CQM on "
            "frozen historical optimization objectives"
        ),
        "config": descriptor(
            root, root / "configs/stage75_explicit_variable_k_cqm.json"
        ),
        "implementation": {
            key: descriptor(root, path) for key, path in implementation_paths.items()
        },
        "inputs": {key: descriptor(root, path) for key, path in input_paths.items()},
        "runtime": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": np.__version__,
            "dimod": dimod.__version__,
            "wall_clock_used_for_decision": False,
        },
        **aggregate_value,
        "decision": {
            **aggregate_value["decision"],
            "new_target_preregistration_remains_authorized": stage74_result[
                "decision"
            ]["new_target_preregistration_remains_authorized"],
            "next_action": (
                "audit objective precision, coefficient scaling, and hardware-shaped local emulation before any cloud or QPU request"
                if aggregate_value["decision"][
                    "local_hardware_shaped_emulation_authorized"
                ]
                else "retain the explicit CQM formulation but repair variable-cardinality solver fidelity before hardware-shaped emulation"
            ),
        },
        "data_boundary": {
            "historical_development_targets_read": len(
                config["experiment"]["target_order"]
            ),
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "cloud_cqm_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "analysis_payload_sha256": canonical_sha256(payload),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    result["outputs"] = {
        key: descriptor(root, output_paths[key])
        for key in (
            "cqm_metrics_csv",
            "solver_trials_csv",
            "cell_comparison_csv",
            "solver_summary_csv",
            "model_record_json",
        )
    }
    write_json(output_paths["result_json"], result)
    output_paths["report_md"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["report_md"].write_text(
        report_text(result), encoding="utf-8", newline="\n"
    )
    result["outputs"]["report_md"] = descriptor(root, output_paths["report_md"])
    write_json(output_paths["result_json"], result)
    return result


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    expected = root / "configs/stage75_explicit_variable_k_cqm.json"
    if config_path != expected.resolve():
        raise ValueError("Stage75 must run from its frozen repository config")
    config = read_json(config_path)
    result_path = root / str(config["outputs"]["result_json"])
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Stage75 result exists: {result_path}")
    result = compute(config, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage75_explicit_variable_k_cqm.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
