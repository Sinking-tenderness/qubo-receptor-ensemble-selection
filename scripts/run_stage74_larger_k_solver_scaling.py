"""Scale the frozen constraint-native receptor objective beyond k=3."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json, write_json  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()






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
        raise ValueError(f"Stage74 frozen {label} identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage74 frozen {label} size differs: {path}")
    return path


def id_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def load_model(record: dict[str, Any]) -> dict[str, Any]:
    receptor_ids = tuple(str(value) for value in record["receptor_ids"])
    count = len(receptor_ids)
    pairs = tuple(itertools.combinations(range(count), 2))
    centered = np.asarray(record["centered_pair_coefficients"], dtype=float)
    if len(centered) != len(pairs):
        raise ValueError("Stage74 centered pair coefficient count differs")
    midpoint = float(record["pair_midpoint_center"])
    raw = centered + midpoint
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
        raise ValueError("Stage74 failed to rebuild a frozen Stage72 model")
    quality_order = tuple(
        sorted(
            range(count),
            key=lambda index: (
                int(deficits[index]),
                id_hash(receptor_ids[index]),
                receptor_ids[index],
            ),
        )
    )
    pair_scale = float(np.max(np.abs(centered)))
    if pair_scale <= TOLERANCE:
        pair_scale = 1.0
    return {
        "record": record,
        "receptor_ids": receptor_ids,
        "count": count,
        "matrix": matrix,
        "deficits": deficits,
        "quality_order": quality_order,
        "pair_scale": pair_scale,
        "pair_count": len(pairs),
    }


def k_schedule(model: dict[str, Any], config: dict[str, Any]) -> list[int]:
    maximum = min(
        int(config["larger_k_workloads"]["maximum_k"]),
        int(math.floor(model["count"] * config["larger_k_workloads"]["maximum_pool_fraction"])),
    )
    output = [
        int(value)
        for value in config["larger_k_workloads"]["candidate_k_values"]
        if int(value) <= maximum
    ]
    if not output or output[0] != int(model["record"]["reference_k"]):
        raise ValueError("Stage74 k schedule must begin at the frozen reference k")
    return sorted(set(output))


def deficit_distributions(
    deficits: np.ndarray, maximum_k: int
) -> list[dict[int, int]]:
    distributions: list[dict[int, int]] = [defaultdict(int) for _ in range(maximum_k + 1)]
    distributions[0][0] = 1
    for raw_value in deficits:
        value = int(raw_value)
        for cardinality in range(maximum_k, 0, -1):
            for total, count in list(distributions[cardinality - 1].items()):
                distributions[cardinality][total + value] += count
    return [dict(value) for value in distributions]


def quality_thresholds(
    distribution: dict[int, int], total_count: int, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for regime in config["larger_k_workloads"]["quality_regimes"]:
        name = str(regime["name"])
        density = float(regime["minimum_feasible_density"])
        required = max(1, math.ceil(density * total_count))
        cumulative = 0
        threshold = 0
        if density >= 1.0 - TOLERANCE:
            threshold = max(distribution)
            cumulative = total_count
        else:
            for value in sorted(distribution):
                cumulative += int(distribution[value])
                if cumulative >= required:
                    threshold = int(value)
                    break
        output[name] = {
            "threshold": threshold,
            "feasible_count": cumulative,
            "feasible_fraction": cumulative / total_count,
            "requested_density": density,
        }
    return output


def subset_deficit(model: dict[str, Any], subset: tuple[int, ...]) -> int:
    return int(sum(int(model["deficits"][index]) for index in subset))


def subset_objective(model: dict[str, Any], subset: tuple[int, ...]) -> float:
    return float(
        sum(
            model["matrix"][left, right]
            for left, right in itertools.combinations(subset, 2)
        )
    )


def mean_pair_objective(objective: float, k: int) -> float:
    return float(objective / math.comb(k, 2))


def exact_oracles(
    model: dict[str, Any], k: int, thresholds: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    best = {
        regime: {"objective": math.inf, "subset": None, "degeneracy": 0}
        for regime in thresholds
    }
    for subset in itertools.combinations(range(model["count"]), k):
        deficit = subset_deficit(model, subset)
        objective = subset_objective(model, subset)
        for regime, metrics in thresholds.items():
            if deficit > int(metrics["threshold"]):
                continue
            current = best[regime]
            if objective < float(current["objective"]) - TOLERANCE:
                current.update(
                    {"objective": objective, "subset": subset, "degeneracy": 1}
                )
            elif abs(objective - float(current["objective"])) <= TOLERANCE:
                current["degeneracy"] = int(current["degeneracy"]) + 1
                if current["subset"] is None or subset < current["subset"]:
                    current["subset"] = subset
    for regime, value in best.items():
        if value["subset"] is None:
            raise ValueError(f"Stage74 exact oracle found no feasible state: {regime}")
    return best


def state(
    model: dict[str, Any],
    k: int,
    regime: str,
    quality: dict[str, Any],
    exact: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "k": k,
        "regime": regime,
        "quality_threshold": int(quality["threshold"]),
        "feasible_count": int(quality["feasible_count"]),
        "feasible_fraction": float(quality["feasible_fraction"]),
        "requested_density": float(quality["requested_density"]),
        "total_count": math.comb(model["count"], k),
        "exact": exact,
    }


def objective_delta(
    cell: dict[str, Any], subset: tuple[int, ...], outgoing: int, incoming: int
) -> float:
    matrix = cell["model"]["matrix"]
    remaining = [value for value in subset if value != outgoing]
    return float(
        sum(matrix[incoming, value] - matrix[outgoing, value] for value in remaining)
    )


def swap_subset(
    subset: tuple[int, ...], outgoing: int, incoming: int
) -> tuple[int, ...]:
    return tuple(sorted((set(subset) - {outgoing}) | {incoming}))


def feasible(cell: dict[str, Any], subset: tuple[int, ...]) -> bool:
    return subset_deficit(cell["model"], subset) <= int(cell["quality_threshold"])


def deterministic_start(cell: dict[str, Any]) -> tuple[int, ...]:
    subset = tuple(sorted(cell["model"]["quality_order"][: cell["k"]]))
    if not feasible(cell, subset):
        raise ValueError("Stage74 deterministic quality start is infeasible")
    return subset


def random_feasible_start(
    cell: dict[str, Any], rng: np.random.Generator, maximum_attempts: int
) -> tuple[tuple[int, ...], int, bool]:
    for attempt in range(1, maximum_attempts + 1):
        subset = tuple(
            sorted(
                int(value)
                for value in rng.choice(
                    cell["model"]["count"], size=cell["k"], replace=False
                )
            )
        )
        if feasible(cell, subset):
            return subset, attempt, False
    return deterministic_start(cell), maximum_attempts, True


def all_swaps(cell: dict[str, Any], subset: tuple[int, ...]) -> list[tuple[int, int]]:
    selected = set(subset)
    return [
        (outgoing, incoming)
        for outgoing in subset
        for incoming in range(cell["model"]["count"])
        if incoming not in selected
    ]


def deterministic_best_improvement(cell: dict[str, Any]) -> dict[str, Any]:
    current = deterministic_start(cell)
    current_objective = subset_objective(cell["model"], current)
    best = current
    best_objective = current_objective
    proposals = feasible_proposals = evaluations = accepted = 0
    while True:
        local_subset = current
        local_objective = current_objective
        for outgoing, incoming in all_swaps(cell, current):
            proposals += 1
            candidate = swap_subset(current, outgoing, incoming)
            if not feasible(cell, candidate):
                continue
            feasible_proposals += 1
            evaluations += 1
            value = current_objective + objective_delta(
                cell, current, outgoing, incoming
            )
            if (value, candidate) < (local_objective - TOLERANCE, local_subset):
                local_subset, local_objective = candidate, value
            if (value, candidate) < (best_objective, best):
                best, best_objective = candidate, value
        if local_objective >= current_objective - TOLERANCE:
            break
        current, current_objective = local_subset, local_objective
        accepted += 1
    return {
        "subset": best,
        "proposal_count": proposals,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations + 1,
        "accepted_move_count": accepted,
        "restart_count": 1,
        "initialization_attempt_count": 1,
        "initialization_fallback_count": 0,
    }


def budgeted_multistart_greedy(
    cell: dict[str, Any], budget: int, rng: np.random.Generator, maximum_attempts: int
) -> dict[str, Any]:
    proposals = feasible_proposals = evaluations = accepted = restarts = 0
    initialization_attempts = fallbacks = 0
    best: tuple[int, ...] | None = None
    best_objective = math.inf
    while proposals < budget:
        restarts += 1
        current, attempts, fallback = random_feasible_start(cell, rng, maximum_attempts)
        initialization_attempts += attempts
        fallbacks += int(fallback)
        current_objective = subset_objective(cell["model"], current)
        evaluations += 1
        if best is None or (current_objective, current) < (best_objective, best):
            best, best_objective = current, current_objective
        while proposals < budget:
            moves = all_swaps(cell, current)
            order = rng.permutation(len(moves))
            local_subset, local_objective = current, current_objective
            for position in order:
                if proposals >= budget:
                    break
                outgoing, incoming = moves[int(position)]
                proposals += 1
                candidate = swap_subset(current, outgoing, incoming)
                if not feasible(cell, candidate):
                    continue
                feasible_proposals += 1
                evaluations += 1
                value = current_objective + objective_delta(
                    cell, current, outgoing, incoming
                )
                if (value, candidate) < (
                    local_objective - TOLERANCE,
                    local_subset,
                ):
                    local_subset, local_objective = candidate, value
                if (value, candidate) < (best_objective, best or candidate):
                    best, best_objective = candidate, value
            if local_objective >= current_objective - TOLERANCE:
                break
            current, current_objective = local_subset, local_objective
            accepted += 1
    if best is None:
        raise ValueError("Stage74 multistart greedy produced no solution")
    return {
        "subset": best,
        "proposal_count": proposals,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "restart_count": restarts,
        "initialization_attempt_count": initialization_attempts,
        "initialization_fallback_count": fallbacks,
    }


def budgeted_tabu_search(
    cell: dict[str, Any],
    budget: int,
    batch_size: int,
    tenure: int,
    rng: np.random.Generator,
    maximum_attempts: int,
) -> dict[str, Any]:
    current, attempts, fallback = random_feasible_start(cell, rng, maximum_attempts)
    current_objective = subset_objective(cell["model"], current)
    best, best_objective = current, current_objective
    proposals = feasible_proposals = 0
    evaluations = 1
    accepted = 0
    iteration = 0
    tabu_until: dict[int, int] = {}
    while proposals < budget:
        moves = all_swaps(cell, current)
        draw = min(batch_size, len(moves), budget - proposals)
        positions = rng.choice(len(moves), size=draw, replace=False)
        candidates: list[tuple[float, tuple[int, ...], int, int]] = []
        for position in positions:
            outgoing, incoming = moves[int(position)]
            proposals += 1
            candidate = swap_subset(current, outgoing, incoming)
            if not feasible(cell, candidate):
                continue
            feasible_proposals += 1
            evaluations += 1
            value = current_objective + objective_delta(
                cell, current, outgoing, incoming
            )
            tabu = (
                tabu_until.get(outgoing, -1) > iteration
                or tabu_until.get(incoming, -1) > iteration
            )
            if not tabu or value < best_objective - TOLERANCE:
                candidates.append((value, candidate, outgoing, incoming))
        if not candidates:
            current, extra_attempts, extra_fallback = random_feasible_start(
                cell, rng, maximum_attempts
            )
            attempts += extra_attempts
            fallback = fallback or extra_fallback
            current_objective = subset_objective(cell["model"], current)
            evaluations += 1
            iteration += 1
            continue
        value, candidate, outgoing, incoming = min(candidates)
        current, current_objective = candidate, value
        tabu_until[outgoing] = iteration + tenure
        tabu_until[incoming] = iteration + tenure
        accepted += 1
        if (current_objective, current) < (best_objective, best):
            best, best_objective = current, current_objective
        iteration += 1
    return {
        "subset": best,
        "proposal_count": proposals,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "restart_count": 1,
        "initialization_attempt_count": attempts,
        "initialization_fallback_count": int(fallback),
    }


def constraint_preserving_annealing(
    cell: dict[str, Any],
    budget: int,
    beta_minimum: float,
    beta_maximum: float,
    rng: np.random.Generator,
    maximum_attempts: int,
) -> dict[str, Any]:
    current, attempts, fallback = random_feasible_start(cell, rng, maximum_attempts)
    current_objective = subset_objective(cell["model"], current)
    best, best_objective = current, current_objective
    feasible_proposals = 0
    evaluations = 1
    accepted = 0
    for step in range(budget):
        outgoing = current[int(rng.integers(0, len(current)))]
        selected = set(current)
        available = [
            value for value in range(cell["model"]["count"]) if value not in selected
        ]
        incoming = available[int(rng.integers(0, len(available)))]
        candidate = swap_subset(current, outgoing, incoming)
        if not feasible(cell, candidate):
            continue
        feasible_proposals += 1
        evaluations += 1
        delta = objective_delta(cell, current, outgoing, incoming)
        normalized_delta = delta / (
            max(1, cell["k"] - 1) * float(cell["model"]["pair_scale"])
        )
        fraction = step / max(1, budget - 1)
        beta = beta_minimum * (beta_maximum / beta_minimum) ** fraction
        if normalized_delta <= 0 or rng.random() < math.exp(-beta * normalized_delta):
            current = candidate
            current_objective += delta
            accepted += 1
            if (current_objective, current) < (best_objective, best):
                best, best_objective = current, current_objective
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


def subset_name(model: dict[str, Any], subset: tuple[int, ...]) -> str:
    return "+".join(model["receptor_ids"][index] for index in subset)


def trial(
    cell: dict[str, Any],
    method: str,
    repeat: int,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    solver = config["solver_protocol"]
    budget = int(solver["swap_proposal_budget"])
    if method == "exact_enumeration":
        if cell["exact"] is None:
            raise ValueError("Stage74 exact solver called outside the oracle envelope")
        solved = {
            "subset": cell["exact"]["subset"],
            "proposal_count": cell["total_count"],
            "feasible_proposal_count": cell["feasible_count"],
            "objective_evaluation_count": cell["feasible_count"],
            "accepted_move_count": 0,
            "restart_count": 1,
            "initialization_attempt_count": 0,
            "initialization_fallback_count": 0,
        }
    elif method == "deterministic_best_improvement":
        solved = deterministic_best_improvement(cell)
    else:
        rng = np.random.default_rng(seed)
        maximum_attempts = int(solver["maximum_initialization_attempts"])
        if method == "budgeted_multistart_greedy":
            solved = budgeted_multistart_greedy(
                cell, budget, rng, maximum_attempts
            )
        elif method == "budgeted_tabu_search":
            solved = budgeted_tabu_search(
                cell,
                budget,
                int(solver["tabu_candidate_batch_size"]),
                int(solver["tabu_tenure"]),
                rng,
                maximum_attempts,
            )
        elif method == "constraint_preserving_annealing":
            solved = constraint_preserving_annealing(
                cell,
                budget,
                float(solver["annealing_beta_range"][0]),
                float(solver["annealing_beta_range"][1]),
                rng,
                maximum_attempts,
            )
        else:
            raise ValueError(f"unknown Stage74 method: {method}")
    subset = tuple(solved["subset"])
    if len(subset) != cell["k"] or not feasible(cell, subset):
        raise ValueError("Stage74 solver returned an invalid subset")
    objective = subset_objective(cell["model"], subset)
    exact_match: bool | None = None
    exact_regret: float | None = None
    if cell["exact"] is not None:
        exact_regret = objective - float(cell["exact"]["objective"])
        exact_match = exact_regret <= TOLERANCE
    return {
        "target_id": cell["model"]["record"]["target_id"],
        "outer_fold": int(cell["model"]["record"]["outer_fold"]),
        "candidate_count": int(cell["model"]["count"]),
        "k": int(cell["k"]),
        "quality_regime": cell["regime"],
        "quality_threshold": int(cell["quality_threshold"]),
        "total_fixed_k_subset_count": int(cell["total_count"]),
        "feasible_subset_count": int(cell["feasible_count"]),
        "exact_oracle_available": cell["exact"] is not None,
        "method": method,
        "repeat": repeat,
        "seed": seed,
        "configured_swap_proposal_budget": (
            int(cell["total_count"])
            if method == "exact_enumeration"
            else int(solved["proposal_count"])
            if method == "deterministic_best_improvement"
            else budget
        ),
        **{key: int(value) for key, value in solved.items() if key != "subset"},
        "solution_subset": subset_name(cell["model"], subset),
        "solution_deficit": subset_deficit(cell["model"], subset),
        "solution_objective": objective,
        "solution_mean_pair_redundancy": mean_pair_objective(objective, cell["k"]),
        "exact_optimum_match": exact_match if exact_match is not None else "",
        "exact_objective_regret": exact_regret if exact_regret is not None else "",
        "exact_mean_pair_regret": (
            mean_pair_objective(exact_regret, cell["k"])
            if exact_regret is not None
            else ""
        ),
    }


def workload_row(cell: dict[str, Any]) -> dict[str, Any]:
    exact = cell["exact"]
    return {
        "target_id": cell["model"]["record"]["target_id"],
        "outer_fold": int(cell["model"]["record"]["outer_fold"]),
        "candidate_count": int(cell["model"]["count"]),
        "k": int(cell["k"]),
        "quality_regime": cell["regime"],
        "quality_threshold": int(cell["quality_threshold"]),
        "requested_feasible_density": float(cell["requested_density"]),
        "total_fixed_k_subset_count": int(cell["total_count"]),
        "log10_total_fixed_k_subset_count": math.log10(cell["total_count"]),
        "feasible_subset_count": int(cell["feasible_count"]),
        "feasible_subset_fraction": float(
            cell["feasible_count"] / cell["total_count"]
        ),
        "logical_variable_count": int(cell["model"]["count"]),
        "quadratic_coupler_count": int(cell["model"]["pair_count"]),
        "explicit_constraint_count": 2,
        "exact_oracle_available": exact is not None,
        "exact_optimum_subset": (
            subset_name(cell["model"], exact["subset"]) if exact is not None else ""
        ),
        "exact_optimum_objective": exact["objective"] if exact is not None else "",
        "exact_optimum_mean_pair_redundancy": (
            mean_pair_objective(exact["objective"], cell["k"])
            if exact is not None
            else ""
        ),
        "exact_optimum_degeneracy": exact["degeneracy"] if exact is not None else "",
    }


def attach_references(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["target_id"]),
            int(row["outer_fold"]),
            int(row["k"]),
            str(row["quality_regime"]),
        )
        grouped[key].append(row)
    cell_rows: list[dict[str, Any]] = []
    classical_methods = set(config["solver_protocol"]["strong_classical_methods"])
    sampler_method = str(config["solver_protocol"]["qubo_sampler_analog_method"])
    tolerance = float(config["benchmark_gate"]["mean_pair_objective_tolerance"])
    for key, selected in sorted(grouped.items()):
        exact_rows = [row for row in selected if row["method"] == "exact_enumeration"]
        candidates = [row for row in selected if row["method"] != "exact_enumeration"]
        reference = (
            float(exact_rows[0]["solution_objective"])
            if exact_rows
            else min(float(row["solution_objective"]) for row in candidates)
        )
        reference_type = "exact_enumeration" if exact_rows else "pooled_best_known"
        for row in selected:
            regret = float(row["solution_objective"]) - reference
            row["reference_type"] = reference_type
            row["reference_objective"] = reference
            row["reference_mean_pair_regret"] = mean_pair_objective(
                regret, int(row["k"])
            )
            row["reference_match"] = regret <= TOLERANCE
        method_best: dict[str, dict[str, Any]] = {}
        for method in sorted({str(row["method"]) for row in candidates}):
            method_best[method] = min(
                (row for row in candidates if row["method"] == method),
                key=lambda row: (
                    float(row["solution_objective"]),
                    str(row["solution_subset"]),
                ),
            )
        classical_best = min(
            (row for row in candidates if row["method"] in classical_methods),
            key=lambda row: (
                float(row["solution_objective"]),
                str(row["solution_subset"]),
            ),
        )
        sampler_best = method_best[sampler_method]
        classical_value = float(classical_best["solution_objective"])
        sampler_value = float(sampler_best["solution_objective"])
        per_pair_delta = mean_pair_objective(
            sampler_value - classical_value, int(key[2])
        )
        method_values = {
            method: float(row["solution_mean_pair_redundancy"])
            for method, row in method_best.items()
        }
        origin = sorted(
            method
            for method, row in method_best.items()
            if float(row["solution_objective"]) <= reference + TOLERANCE
        )
        cell_rows.append(
            {
                "target_id": key[0],
                "outer_fold": key[1],
                "k": key[2],
                "quality_regime": key[3],
                "exact_oracle_available": bool(exact_rows),
                "reference_type": reference_type,
                "reference_objective": reference,
                "reference_mean_pair_redundancy": mean_pair_objective(
                    reference, key[2]
                ),
                "reference_origin_methods": "+".join(origin),
                "strong_classical_best_method": classical_best["method"],
                "strong_classical_best_objective": classical_value,
                "strong_classical_reference_match": classical_value
                <= reference + TOLERANCE,
                "sampler_best_objective": sampler_value,
                "sampler_reference_match": sampler_value <= reference + TOLERANCE,
                "sampler_delta_vs_strong_classical_per_pair": per_pair_delta,
                "sampler_within_tolerance_of_strong_classical": per_pair_delta
                <= tolerance,
                "sampler_strict_win_vs_strong_classical": per_pair_delta
                < -tolerance,
                "strong_classical_strict_win_vs_sampler": per_pair_delta
                > tolerance,
                "solver_best_mean_pair_spread": max(method_values.values())
                - min(method_values.values()),
                "solver_disagreement": max(method_values.values())
                - min(method_values.values())
                > tolerance,
            }
        )
    return rows, cell_rows


def method_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for scope in ("ALL", str(row["target_id"])):
            key = (
                str(row["method"]),
                str(row["quality_regime"]),
                int(row["k"]),
                scope,
            )
            groups[key].append(row)
    output: list[dict[str, Any]] = []
    for (method, regime, k, scope), selected in sorted(groups.items()):
        exact = [row for row in selected if bool(row["exact_oracle_available"])]
        output.append(
            {
                "method": method,
                "quality_regime": regime,
                "k": k,
                "scope": scope,
                "trial_count": len(selected),
                "workload_cell_count": len(
                    {(row["target_id"], row["outer_fold"]) for row in selected}
                ),
                "exact_oracle_trial_count": len(exact),
                "exact_optimum_success_rate": (
                    statistics.fmean(bool(row["exact_optimum_match"]) for row in exact)
                    if exact
                    else ""
                ),
                "reference_match_rate": statistics.fmean(
                    bool(row["reference_match"]) for row in selected
                ),
                "mean_reference_regret_per_pair": statistics.fmean(
                    float(row["reference_mean_pair_regret"]) for row in selected
                ),
                "maximum_reference_regret_per_pair": max(
                    float(row["reference_mean_pair_regret"]) for row in selected
                ),
                "mean_initialization_attempt_count": statistics.fmean(
                    int(row["initialization_attempt_count"]) for row in selected
                ),
                "initialization_fallback_count": sum(
                    int(row["initialization_fallback_count"]) for row in selected
                ),
                "mean_swap_proposal_count": statistics.fmean(
                    int(row["proposal_count"]) for row in selected
                ),
            }
        )
    return output


def aggregate_result(
    workloads: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    exact_cells = [row for row in cells if bool(row["exact_oracle_available"])]
    nonexact = [row for row in cells if not bool(row["exact_oracle_available"])]
    gate = config["benchmark_gate"]
    scaling = {
        "model_count": len(
            {(row["target_id"], int(row["outer_fold"])) for row in workloads}
        ),
        "model_k_count": len(
            {
                (row["target_id"], int(row["outer_fold"]), int(row["k"]))
                for row in workloads
            }
        ),
        "workload_cell_count": len(workloads),
        "solver_trial_count": len(trials),
        "exact_oracle_cell_count": len(exact_cells),
        "maximum_candidate_count": max(int(row["candidate_count"]) for row in workloads),
        "maximum_k": max(int(row["k"]) for row in workloads),
        "maximum_total_fixed_k_subset_count": max(
            int(row["total_fixed_k_subset_count"]) for row in workloads
        ),
        "maximum_log10_total_fixed_k_subset_count": max(
            float(row["log10_total_fixed_k_subset_count"]) for row in workloads
        ),
        "maximum_quadratic_coupler_count": max(
            int(row["quadratic_coupler_count"]) for row in workloads
        ),
        "exact_oracle_unique_state_checks": sum(
            int(row["total_fixed_k_subset_count"])
            for row in workloads
            if bool(row["exact_oracle_available"])
            and row["quality_regime"]
            == config["larger_k_workloads"]["quality_regimes"][0]["name"]
        ),
    }
    strong_exact_rate = statistics.fmean(
        bool(row["strong_classical_reference_match"]) for row in exact_cells
    )
    sampler_exact_rate = statistics.fmean(
        bool(row["sampler_reference_match"]) for row in exact_cells
    )
    disagreement_fraction = statistics.fmean(
        bool(row["solver_disagreement"]) for row in nonexact
    )
    sampler_competitive_fraction = statistics.fmean(
        bool(row["sampler_within_tolerance_of_strong_classical"])
        for row in nonexact
    )
    hardness = {
        "nonexact_workload_cell_count": len(nonexact),
        "nonexact_solver_disagreement_cell_count": sum(
            bool(row["solver_disagreement"]) for row in nonexact
        ),
        "nonexact_solver_disagreement_fraction": disagreement_fraction,
        "sampler_strict_win_cell_count": sum(
            bool(row["sampler_strict_win_vs_strong_classical"]) for row in nonexact
        ),
        "strong_classical_strict_win_cell_count": sum(
            bool(row["strong_classical_strict_win_vs_sampler"]) for row in nonexact
        ),
        "sampler_tie_within_tolerance_cell_count": sum(
            not bool(row["sampler_strict_win_vs_strong_classical"])
            and not bool(row["strong_classical_strict_win_vs_sampler"])
            for row in nonexact
        ),
        "maximum_solver_best_mean_pair_spread": max(
            float(row["solver_best_mean_pair_spread"]) for row in nonexact
        ),
    }
    validation = {
        "strong_classical_exact_cell_success_rate": strong_exact_rate,
        "sampler_exact_cell_success_rate": sampler_exact_rate,
        "sampler_nonexact_competitive_fraction": sampler_competitive_fraction,
    }
    scale_gate = (
        scaling["maximum_total_fixed_k_subset_count"]
        >= int(gate["minimum_large_scale_state_count"])
    )
    exact_gate = strong_exact_rate >= float(
        gate["minimum_strong_classical_exact_cell_success_rate"]
    )
    hardness_gate = disagreement_fraction >= float(
        gate["minimum_nonexact_solver_disagreement_fraction"]
    )
    sampler_gate = sampler_competitive_fraction >= float(
        gate["minimum_sampler_nonexact_competitive_fraction"]
    )
    return {
        "scaling_summary": scaling,
        "exact_validation": validation,
        "hardness_summary": hardness,
        "route_gate": {
            "large_scale_state_space_reached": bool(scale_gate),
            "strong_classical_exact_validation_passed": bool(exact_gate),
            "nonexact_solver_hardness_observed": bool(hardness_gate),
            "qubo_sampler_analog_competitiveness_passed": bool(sampler_gate),
        },
        "decision": {
            "explicit_variable_k_cqm_design_authorized": bool(
                scale_gate and exact_gate and hardness_gate
            ),
            "hardware_shaped_sampler_poc_authorized": bool(
                scale_gate and exact_gate and hardness_gate and sampler_gate
            ),
            "direct_qpu_execution_authorized": False,
            "quantum_scaling_claim_authorized": False,
            "quantum_advantage_claim_authorized": False,
        },
    }


def report_text(result: dict[str, Any]) -> str:
    scale = result["scaling_summary"]
    validation = result["exact_validation"]
    hardness = result["hardness_summary"]
    gate = result["route_gate"]
    return rf"""# Stage74 larger-k constraint-native solver scaling

## Question

Does the frozen pair-redundancy CQM become computationally nontrivial when receptor cardinality grows beyond $k=3$, while retaining deterministic quality constraints?

## Protocol

The complete receptor pools and frozen Stage72 pair coefficients are reused. Cardinalities are drawn from $k\in\{{3,4,6,8,10,12,16\}}$ subject to pool size. Quality thresholds contain at least 1%, 10%, or 100% of fixed-$k$ subsets and are computed exactly by integer subset-sum dynamic programming. Exact enumeration is used only when $\binom{{n}}{{k}}\leq 200,000$; larger cells use an explicitly labelled pooled best-known reference.

## Scale

- Models / model-$k$ pairs / workload cells: `{scale['model_count']}` / `{scale['model_k_count']}` / `{scale['workload_cell_count']}`.
- Solver trials: `{scale['solver_trial_count']}`.
- Largest state space: `{scale['maximum_total_fixed_k_subset_count']}` ($10^{{{scale['maximum_log10_total_fixed_k_subset_count']:.2f}}}$).
- Exact-oracle workload cells: `{scale['exact_oracle_cell_count']}`.
- Largest logical model: `{scale['maximum_candidate_count']}` variables and `{scale['maximum_quadratic_coupler_count']}` pair couplers.

## Solver validation and hardness

- Strong classical exact-cell success: `{validation['strong_classical_exact_cell_success_rate']:.3f}`.
- Annealing exact-cell success: `{validation['sampler_exact_cell_success_rate']:.3f}`.
- Non-exact solver disagreement: `{hardness['nonexact_solver_disagreement_cell_count']}/{hardness['nonexact_workload_cell_count']}` (`{hardness['nonexact_solver_disagreement_fraction']:.3f}`).
- Annealing strict wins / classical strict wins / ties: `{hardness['sampler_strict_win_cell_count']}` / `{hardness['strong_classical_strict_win_cell_count']}` / `{hardness['sampler_tie_within_tolerance_cell_count']}`.

## Decision

- Larger-scale state-space gate: `{gate['large_scale_state_space_reached']}`.
- Exact classical validation gate: `{gate['strong_classical_exact_validation_passed']}`.
- Solver-hardness gate: `{gate['nonexact_solver_hardness_observed']}`.
- Explicit variable-$k$ CQM design authorized: `{result['decision']['explicit_variable_k_cqm_design_authorized']}`.
- Hardware-shaped sampler PoC authorized: `{result['decision']['hardware_shaped_sampler_poc_authorized']}`.
- Direct QPU / quantum-scaling / quantum-advantage claims: `False / False / False`.

Large cells do not have certified global optima. A pooled best-known solution is not an exact oracle, and solver disagreement is evidence of optimization difficulty rather than evidence of quantum advantage or biological benefit.
"""


def compute(config: dict[str, Any], root: Path) -> dict[str, Any]:
    implementation_paths = {
        key: verified(root, value, key)
        for key, value in config["implementation"].items()
    }
    input_paths = {
        key: verified(root, value, key) for key, value in config["inputs"].items()
    }
    stage73_result = read_json(input_paths["stage73_result"])
    stage73_audit = read_json(input_paths["stage73_audit"])
    if not stage73_result["decision"]["larger_k_scaling_study_authorized"]:
        raise ValueError("Stage74 requires Stage73 larger-k authorization")
    if stage73_audit.get("status") != (
        "stage73_constraint_native_solver_scaling_independent_audit_ok"
    ):
        raise ValueError("Stage74 requires the Stage73 independent audit")
    source = read_json(input_paths["stage72_model_record"])
    if int(source["model_count"]) != int(config["experiment"]["required_model_count"]):
        raise ValueError("Stage74 source model count differs")
    models = [load_model(record) for record in source["models"]]
    workloads: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    methods = [str(value) for value in config["solver_protocol"]["method_order"]]
    stochastic = set(config["solver_protocol"]["stochastic_methods"])
    repeats = int(config["solver_protocol"]["stochastic_repeats"])
    exact_limit = int(config["larger_k_workloads"]["exact_enumeration_state_limit"])
    base_seed = int(config["solver_protocol"]["seed_base"])
    cell_index = 0
    for model in models:
        schedule = k_schedule(model, config)
        distributions = deficit_distributions(model["deficits"], max(schedule))
        for k in schedule:
            total = math.comb(model["count"], k)
            quality = quality_thresholds(distributions[k], total, config)
            oracle = exact_oracles(model, k, quality) if total <= exact_limit else None
            for regime in [
                str(value["name"])
                for value in config["larger_k_workloads"]["quality_regimes"]
            ]:
                cell = state(
                    model,
                    k,
                    regime,
                    quality[regime],
                    oracle[regime] if oracle is not None else None,
                )
                workloads.append(workload_row(cell))
                for method_index, method in enumerate(methods):
                    if method == "exact_enumeration" and cell["exact"] is None:
                        continue
                    method_repeats = repeats if method in stochastic else 1
                    for repeat in range(method_repeats):
                        seed = seed_for(base_seed, cell_index, method_index, repeat)
                        trials.append(trial(cell, method, repeat, seed, config))
                cell_index += 1
        print(
            json.dumps(
                {
                    "target_id": model["record"]["target_id"],
                    "outer_fold": model["record"]["outer_fold"],
                    "workload_cells_completed": len(workloads),
                    "solver_trials_completed": len(trials),
                }
            ),
            flush=True,
        )
    expected_cells = int(config["benchmark_gate"]["required_workload_cell_count"])
    if len(workloads) != expected_cells:
        raise ValueError(
            f"Stage74 workload count differs: {len(workloads)} != {expected_cells}"
        )
    trials, cell_rows = attach_references(trials, config)
    summaries = method_summary(trials)
    aggregate = aggregate_result(workloads, trials, cell_rows, config)
    output_paths = {
        key: root / str(value) for key, value in config["outputs"].items()
    }
    write_csv(output_paths["workload_metrics_csv"], workloads)
    write_csv(output_paths["solver_trials_csv"], trials)
    write_csv(output_paths["cell_comparison_csv"], cell_rows)
    write_csv(output_paths["solver_summary_csv"], summaries)
    payload = {
        **aggregate,
        "workload_metrics_sha256": sha256(output_paths["workload_metrics_csv"]),
        "solver_trials_sha256": sha256(output_paths["solver_trials_csv"]),
        "cell_comparison_sha256": sha256(output_paths["cell_comparison_csv"]),
        "solver_summary_sha256": sha256(output_paths["solver_summary_csv"]),
    }
    result = {
        "schema_version": "1.0",
        "status": "stage74_larger_k_solver_scaling_complete",
        "experiment_class": (
            "post-hoc larger-k deterministic-work-unit scaling on frozen historical "
            "constraint-native objectives"
        ),
        "config": descriptor(
            root, root / "configs/stage74_larger_k_solver_scaling.json"
        ),
        "implementation": {
            key: descriptor(root, path) for key, path in implementation_paths.items()
        },
        "inputs": {key: descriptor(root, path) for key, path in input_paths.items()},
        "runtime": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": np.__version__,
            "wall_clock_used_for_decision": False,
        },
        **aggregate,
        "decision": {
            **aggregate["decision"],
            "new_target_preregistration_remains_authorized": stage73_result[
                "decision"
            ]["new_target_preregistration_remains_authorized"],
            "next_action": (
                "freeze an explicit variable-k constraint-native CQM and compare it with the larger-k strong classical frontier"
                if aggregate["decision"]["explicit_variable_k_cqm_design_authorized"]
                else "do not build a variable-k CQM until the larger-k solver-hardness gate is repaired"
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
            "workload_metrics_csv",
            "solver_trials_csv",
            "cell_comparison_csv",
            "solver_summary_csv",
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
    expected = root / "configs/stage74_larger_k_solver_scaling.json"
    if config_path != expected.resolve():
        raise ValueError("Stage74 must run from its frozen repository config")
    config = read_json(config_path)
    result_path = root / str(config["outputs"]["result_json"])
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Stage74 result exists: {result_path}")
    result = compute(config, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage74_larger_k_solver_scaling.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
