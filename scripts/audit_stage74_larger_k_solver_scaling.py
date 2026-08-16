"""Independently audit the Stage74 larger-k solver-scaling benchmark."""

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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest().upper()




def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))




def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def checked(root: Path, value: dict[str, Any], label: str) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage74 audit {label} identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage74 audit {label} size differs: {path}")
    return path


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean: {value!r}")


def close(observed: Any, expected: Any, label: str) -> None:
    if not math.isclose(
        float(observed), float(expected), rel_tol=TOLERANCE, abs_tol=TOLERANCE
    ):
        raise ValueError(f"Stage74 audit {label} differs: {observed} != {expected}")


def rebuild(record: dict[str, Any]) -> dict[str, Any]:
    ids = tuple(str(value) for value in record["receptor_ids"])
    count = len(ids)
    pairs = tuple(itertools.combinations(range(count), 2))
    centered = tuple(float(value) for value in record["centered_pair_coefficients"])
    midpoint = float(record["pair_midpoint_center"])
    raw = tuple(value + midpoint for value in centered)
    matrix = np.zeros((count, count), dtype=float)
    for pair, value in zip(pairs, raw):
        matrix[pair] = value
        matrix[pair[::-1]] = value
    deficits = tuple(int(value) for value in record["integer_deficits"])
    canonical = {
        "variable_order": [f"x{index:03d}" for index in range(count)],
        "objective": {
            "pair_center": midpoint,
            "quadratic_pair_order": [list(pair) for pair in pairs],
            "quadratic_coefficients": list(centered),
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
                "linear_coefficients": list(deficits),
                "rhs": int(record["maximum_integer_deficit"]),
            },
        },
    }
    if canonical_sha256(canonical) != str(record["cqm_sha256"]).upper():
        raise ValueError("Stage74 audit failed to reconstruct Stage72 CQM")
    order = tuple(
        sorted(
            range(count),
            key=lambda index: (
                deficits[index],
                hashlib.sha256(ids[index].encode("utf-8")).hexdigest().upper(),
                ids[index],
            ),
        )
    )
    scale = max(abs(value) for value in centered)
    return {
        "record": record,
        "ids": ids,
        "count": count,
        "matrix": matrix,
        "deficits": deficits,
        "order": order,
        "scale": scale if scale > 1e-12 else 1.0,
        "couplers": len(pairs),
    }


def schedule(model: dict[str, Any], config: dict[str, Any]) -> list[int]:
    section = config["larger_k_workloads"]
    maximum = min(
        int(section["maximum_k"]),
        math.floor(model["count"] * float(section["maximum_pool_fraction"])),
    )
    return sorted(
        {
            int(value)
            for value in section["candidate_k_values"]
            if int(value) <= maximum
        }
    )


def distributions(deficits: tuple[int, ...], maximum_k: int) -> list[dict[int, int]]:
    values: list[dict[int, int]] = [defaultdict(int) for _ in range(maximum_k + 1)]
    values[0][0] = 1
    for deficit in deficits:
        for k in range(maximum_k, 0, -1):
            for total, count in list(values[k - 1].items()):
                values[k][total + deficit] += count
    return [dict(value) for value in values]


def threshold_grid(
    distribution: dict[int, int], total: int, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in config["larger_k_workloads"]["quality_regimes"]:
        name = str(item["name"])
        density = float(item["minimum_feasible_density"])
        if density >= 1.0 - 1e-12:
            threshold = max(distribution)
            cumulative = total
        else:
            required = max(1, math.ceil(density * total))
            cumulative = 0
            threshold = 0
            for value in sorted(distribution):
                cumulative += int(distribution[value])
                threshold = int(value)
                if cumulative >= required:
                    break
        output[name] = {
            "threshold": threshold,
            "count": cumulative,
            "fraction": cumulative / total,
            "requested": density,
        }
    return output


def deficit(model: dict[str, Any], subset: tuple[int, ...]) -> int:
    return sum(model["deficits"][index] for index in subset)


def objective(model: dict[str, Any], subset: tuple[int, ...]) -> float:
    return float(
        sum(
            model["matrix"][left, right]
            for left, right in itertools.combinations(subset, 2)
        )
    )


def mean_pair(value: float, k: int) -> float:
    return value / math.comb(k, 2)


def oracle_grid(
    model: dict[str, Any], k: int, thresholds: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    output = {
        name: {"objective": math.inf, "subset": None, "degeneracy": 0}
        for name in thresholds
    }
    for subset in itertools.combinations(range(model["count"]), k):
        quality = deficit(model, subset)
        value = objective(model, subset)
        for name, threshold in thresholds.items():
            if quality > threshold["threshold"]:
                continue
            current = output[name]
            if value < current["objective"] - 1e-12:
                current.update({"objective": value, "subset": subset, "degeneracy": 1})
            elif abs(value - current["objective"]) <= 1e-12:
                current["degeneracy"] += 1
                if current["subset"] is None or subset < current["subset"]:
                    current["subset"] = subset
    if any(value["subset"] is None for value in output.values()):
        raise ValueError("Stage74 audit exact oracle is empty")
    return output


def make_cell(
    model: dict[str, Any],
    k: int,
    name: str,
    quality: dict[str, Any],
    oracle: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "k": k,
        "name": name,
        "threshold": int(quality["threshold"]),
        "count": int(quality["count"]),
        "fraction": float(quality["fraction"]),
        "requested": float(quality["requested"]),
        "total": math.comb(model["count"], k),
        "oracle": oracle,
    }


def is_feasible(cell: dict[str, Any], subset: tuple[int, ...]) -> bool:
    return deficit(cell["model"], subset) <= cell["threshold"]


def quality_start(cell: dict[str, Any]) -> tuple[int, ...]:
    return tuple(sorted(cell["model"]["order"][: cell["k"]]))


def random_start(
    cell: dict[str, Any], rng: np.random.Generator, maximum_attempts: int
) -> tuple[tuple[int, ...], int, bool]:
    for attempt in range(1, maximum_attempts + 1):
        subset = tuple(
            sorted(
                int(value)
                for value in rng.choice(
                    cell["model"]["count"], cell["k"], replace=False
                )
            )
        )
        if is_feasible(cell, subset):
            return subset, attempt, False
    return quality_start(cell), maximum_attempts, True


def moves(cell: dict[str, Any], subset: tuple[int, ...]) -> list[tuple[int, int]]:
    selected = set(subset)
    return [
        (outgoing, incoming)
        for outgoing in subset
        for incoming in range(cell["model"]["count"])
        if incoming not in selected
    ]


def swapped(subset: tuple[int, ...], outgoing: int, incoming: int) -> tuple[int, ...]:
    return tuple(sorted((set(subset) - {outgoing}) | {incoming}))


def delta(
    cell: dict[str, Any], subset: tuple[int, ...], outgoing: int, incoming: int
) -> float:
    matrix = cell["model"]["matrix"]
    return float(
        sum(
            matrix[incoming, value] - matrix[outgoing, value]
            for value in subset
            if value != outgoing
        )
    )


def replay_deterministic(cell: dict[str, Any]) -> dict[str, Any]:
    current = quality_start(cell)
    current_value = objective(cell["model"], current)
    best, best_value = current, current_value
    proposals = feasible_proposals = evaluations = accepted = 0
    while True:
        local, local_value = current, current_value
        for outgoing, incoming in moves(cell, current):
            proposals += 1
            candidate = swapped(current, outgoing, incoming)
            if not is_feasible(cell, candidate):
                continue
            feasible_proposals += 1
            evaluations += 1
            value = current_value + delta(cell, current, outgoing, incoming)
            if (value, candidate) < (local_value - 1e-12, local):
                local, local_value = candidate, value
            if (value, candidate) < (best_value, best):
                best, best_value = candidate, value
        if local_value >= current_value - 1e-12:
            break
        current, current_value = local, local_value
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


def replay_multistart(
    cell: dict[str, Any], budget: int, rng: np.random.Generator, maximum_attempts: int
) -> dict[str, Any]:
    proposals = feasible_proposals = evaluations = accepted = restarts = 0
    attempts = fallbacks = 0
    best: tuple[int, ...] | None = None
    best_value = math.inf
    while proposals < budget:
        restarts += 1
        current, count, fallback = random_start(cell, rng, maximum_attempts)
        attempts += count
        fallbacks += int(fallback)
        current_value = objective(cell["model"], current)
        evaluations += 1
        if best is None or (current_value, current) < (best_value, best):
            best, best_value = current, current_value
        while proposals < budget:
            candidates = moves(cell, current)
            order = rng.permutation(len(candidates))
            local, local_value = current, current_value
            for position in order:
                if proposals >= budget:
                    break
                outgoing, incoming = candidates[int(position)]
                proposals += 1
                candidate = swapped(current, outgoing, incoming)
                if not is_feasible(cell, candidate):
                    continue
                feasible_proposals += 1
                evaluations += 1
                value = current_value + delta(cell, current, outgoing, incoming)
                if (value, candidate) < (local_value - 1e-12, local):
                    local, local_value = candidate, value
                if best is None or (value, candidate) < (best_value, best):
                    best, best_value = candidate, value
            if local_value >= current_value - 1e-12:
                break
            current, current_value = local, local_value
            accepted += 1
    if best is None:
        raise ValueError("Stage74 audit multistart replay is empty")
    return {
        "subset": best,
        "proposal_count": proposals,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "restart_count": restarts,
        "initialization_attempt_count": attempts,
        "initialization_fallback_count": fallbacks,
    }


def replay_tabu(
    cell: dict[str, Any],
    budget: int,
    batch_size: int,
    tenure: int,
    rng: np.random.Generator,
    maximum_attempts: int,
) -> dict[str, Any]:
    current, attempts, fallback = random_start(cell, rng, maximum_attempts)
    current_value = objective(cell["model"], current)
    best, best_value = current, current_value
    proposals = feasible_proposals = 0
    evaluations = 1
    accepted = iteration = 0
    tabu_until: dict[int, int] = {}
    while proposals < budget:
        available_moves = moves(cell, current)
        draw = min(batch_size, len(available_moves), budget - proposals)
        positions = rng.choice(len(available_moves), draw, replace=False)
        candidates: list[tuple[float, tuple[int, ...], int, int]] = []
        for position in positions:
            outgoing, incoming = available_moves[int(position)]
            proposals += 1
            candidate = swapped(current, outgoing, incoming)
            if not is_feasible(cell, candidate):
                continue
            feasible_proposals += 1
            evaluations += 1
            value = current_value + delta(cell, current, outgoing, incoming)
            tabu = (
                tabu_until.get(outgoing, -1) > iteration
                or tabu_until.get(incoming, -1) > iteration
            )
            if not tabu or value < best_value - 1e-12:
                candidates.append((value, candidate, outgoing, incoming))
        if not candidates:
            current, extra, used_fallback = random_start(cell, rng, maximum_attempts)
            attempts += extra
            fallback = fallback or used_fallback
            current_value = objective(cell["model"], current)
            evaluations += 1
            iteration += 1
            continue
        value, candidate, outgoing, incoming = min(candidates)
        current, current_value = candidate, value
        tabu_until[outgoing] = iteration + tenure
        tabu_until[incoming] = iteration + tenure
        accepted += 1
        if (current_value, current) < (best_value, best):
            best, best_value = current, current_value
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


def replay_annealing(
    cell: dict[str, Any],
    budget: int,
    beta_minimum: float,
    beta_maximum: float,
    rng: np.random.Generator,
    maximum_attempts: int,
) -> dict[str, Any]:
    current, attempts, fallback = random_start(cell, rng, maximum_attempts)
    current_value = objective(cell["model"], current)
    best, best_value = current, current_value
    feasible_proposals = 0
    evaluations = 1
    accepted = 0
    for step in range(budget):
        outgoing = current[int(rng.integers(0, len(current)))]
        selected = set(current)
        available = [value for value in range(cell["model"]["count"]) if value not in selected]
        incoming = available[int(rng.integers(0, len(available)))]
        candidate = swapped(current, outgoing, incoming)
        if not is_feasible(cell, candidate):
            continue
        feasible_proposals += 1
        evaluations += 1
        change = delta(cell, current, outgoing, incoming)
        normalized = change / (max(1, cell["k"] - 1) * cell["model"]["scale"])
        fraction = step / max(1, budget - 1)
        beta = beta_minimum * (beta_maximum / beta_minimum) ** fraction
        if normalized <= 0 or rng.random() < math.exp(-beta * normalized):
            current = candidate
            current_value += change
            accepted += 1
            if (current_value, current) < (best_value, best):
                best, best_value = current, current_value
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


def expected_solution(
    cell: dict[str, Any], method: str, seed: int, config: dict[str, Any]
) -> dict[str, Any]:
    solver = config["solver_protocol"]
    budget = int(solver["swap_proposal_budget"])
    attempts = int(solver["maximum_initialization_attempts"])
    if method == "exact_enumeration":
        oracle = cell["oracle"]
        return {
            "subset": oracle["subset"],
            "proposal_count": cell["total"],
            "feasible_proposal_count": cell["count"],
            "objective_evaluation_count": cell["count"],
            "accepted_move_count": 0,
            "restart_count": 1,
            "initialization_attempt_count": 0,
            "initialization_fallback_count": 0,
        }
    if method == "deterministic_best_improvement":
        return replay_deterministic(cell)
    rng = np.random.default_rng(seed)
    if method == "budgeted_multistart_greedy":
        return replay_multistart(cell, budget, rng, attempts)
    if method == "budgeted_tabu_search":
        return replay_tabu(
            cell,
            budget,
            int(solver["tabu_candidate_batch_size"]),
            int(solver["tabu_tenure"]),
            rng,
            attempts,
        )
    if method == "constraint_preserving_annealing":
        return replay_annealing(
            cell,
            budget,
            float(solver["annealing_beta_range"][0]),
            float(solver["annealing_beta_range"][1]),
            rng,
            attempts,
        )
    raise ValueError(f"Stage74 audit unknown method: {method}")


def subset_label(model: dict[str, Any], subset: tuple[int, ...]) -> str:
    return "+".join(model["ids"][index] for index in subset)


def compare_workload(row: dict[str, str], cell: dict[str, Any]) -> None:
    model = cell["model"]
    exact = cell["oracle"]
    expected_text = {
        "target_id": str(model["record"]["target_id"]),
        "quality_regime": cell["name"],
        "exact_optimum_subset": subset_label(model, exact["subset"]) if exact else "",
    }
    for field, value in expected_text.items():
        if row[field] != value:
            raise ValueError(f"Stage74 audit workload {field} differs")
    integers = {
        "outer_fold": model["record"]["outer_fold"],
        "candidate_count": model["count"],
        "k": cell["k"],
        "quality_threshold": cell["threshold"],
        "total_fixed_k_subset_count": cell["total"],
        "feasible_subset_count": cell["count"],
        "logical_variable_count": model["count"],
        "quadratic_coupler_count": model["couplers"],
        "explicit_constraint_count": 2,
    }
    for field, value in integers.items():
        if int(row[field]) != int(value):
            raise ValueError(f"Stage74 audit workload {field} differs")
    if truth(row["exact_oracle_available"]) != bool(exact):
        raise ValueError("Stage74 audit workload oracle flag differs")
    close(row["requested_feasible_density"], cell["requested"], "requested density")
    close(row["feasible_subset_fraction"], cell["count"] / cell["total"], "feasible fraction")
    close(row["log10_total_fixed_k_subset_count"], math.log10(cell["total"]), "state-space log")
    if exact:
        close(row["exact_optimum_objective"], exact["objective"], "exact objective")
        close(row["exact_optimum_mean_pair_redundancy"], mean_pair(exact["objective"], cell["k"]), "exact mean pair")
        if int(row["exact_optimum_degeneracy"]) != int(exact["degeneracy"]):
            raise ValueError("Stage74 audit exact degeneracy differs")


def compare_trial(
    row: dict[str, str],
    cell: dict[str, Any],
    method: str,
    repeat: int,
    seed: int,
    solved: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if row["method"] != method or int(row["repeat"]) != repeat or int(row["seed"]) != seed:
        raise ValueError("Stage74 audit trial identity differs")
    subset = tuple(solved["subset"])
    if not is_feasible(cell, subset) or len(subset) != cell["k"]:
        raise ValueError("Stage74 audit replay produced an invalid state")
    if row["solution_subset"] != subset_label(cell["model"], subset):
        raise ValueError("Stage74 audit trial subset differs")
    for field in (
        "proposal_count",
        "feasible_proposal_count",
        "objective_evaluation_count",
        "accepted_move_count",
        "restart_count",
        "initialization_attempt_count",
        "initialization_fallback_count",
    ):
        if int(row[field]) != int(solved[field]):
            raise ValueError(f"Stage74 audit trial {field} differs")
    budget = int(config["solver_protocol"]["swap_proposal_budget"])
    configured = (
        cell["total"]
        if method == "exact_enumeration"
        else solved["proposal_count"]
        if method == "deterministic_best_improvement"
        else budget
    )
    if int(row["configured_swap_proposal_budget"]) != configured:
        raise ValueError("Stage74 audit trial configured budget differs")
    value = objective(cell["model"], subset)
    close(row["solution_objective"], value, "trial objective")
    close(row["solution_mean_pair_redundancy"], mean_pair(value, cell["k"]), "trial mean pair")
    if int(row["solution_deficit"]) != deficit(cell["model"], subset):
        raise ValueError("Stage74 audit trial deficit differs")
    exact_match: bool | None = None
    if cell["oracle"]:
        regret = value - cell["oracle"]["objective"]
        exact_match = regret <= 1e-12
        if truth(row["exact_optimum_match"]) != exact_match:
            raise ValueError("Stage74 audit exact-match flag differs")
        close(row["exact_objective_regret"], regret, "exact regret")
        close(row["exact_mean_pair_regret"], mean_pair(regret, cell["k"]), "exact mean regret")
    else:
        if row["exact_optimum_match"] or row["exact_objective_regret"] or row["exact_mean_pair_regret"]:
            raise ValueError("Stage74 audit non-oracle exact fields are populated")
    return {
        "target_id": str(cell["model"]["record"]["target_id"]),
        "outer_fold": int(cell["model"]["record"]["outer_fold"]),
        "candidate_count": int(cell["model"]["count"]),
        "k": int(cell["k"]),
        "quality_regime": cell["name"],
        "exact_oracle_available": bool(cell["oracle"]),
        "method": method,
        "repeat": repeat,
        "subset": subset,
        "solution_subset": subset_label(cell["model"], subset),
        "objective": value,
        "mean_pair": mean_pair(value, cell["k"]),
        "exact_match": exact_match,
        "initialization_attempt_count": int(solved["initialization_attempt_count"]),
        "initialization_fallback_count": int(solved["initialization_fallback_count"]),
        "proposal_count": int(solved["proposal_count"]),
    }


def attach_expected_references(
    trials: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        groups[(row["target_id"], row["outer_fold"], row["k"], row["quality_regime"])].append(row)
    classical = set(config["solver_protocol"]["strong_classical_methods"])
    sampler = str(config["solver_protocol"]["qubo_sampler_analog_method"])
    tolerance = float(config["benchmark_gate"]["mean_pair_objective_tolerance"])
    cells: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        exact = [row for row in rows if row["method"] == "exact_enumeration"]
        candidates = [row for row in rows if row["method"] != "exact_enumeration"]
        reference = exact[0]["objective"] if exact else min(row["objective"] for row in candidates)
        reference_type = "exact_enumeration" if exact else "pooled_best_known"
        for row in rows:
            row["reference_type"] = reference_type
            row["reference_objective"] = reference
            row["reference_regret"] = mean_pair(row["objective"] - reference, row["k"])
            row["reference_match"] = row["objective"] <= reference + 1e-12
        method_best = {
            method: min(
                (row for row in candidates if row["method"] == method),
                key=lambda row: (row["objective"], row["solution_subset"]),
            )
            for method in {row["method"] for row in candidates}
        }
        classical_best = min(
            (row for row in candidates if row["method"] in classical),
            key=lambda row: (row["objective"], row["solution_subset"]),
        )
        sampler_best = method_best[sampler]
        pair_delta = mean_pair(sampler_best["objective"] - classical_best["objective"], key[2])
        method_values = {method: row["mean_pair"] for method, row in method_best.items()}
        origin = sorted(
            method
            for method, row in method_best.items()
            if row["objective"] <= reference + 1e-12
        )
        cells.append(
            {
                "target_id": key[0],
                "outer_fold": key[1],
                "k": key[2],
                "quality_regime": key[3],
                "exact_oracle_available": bool(exact),
                "reference_type": reference_type,
                "reference_objective": reference,
                "reference_mean_pair_redundancy": mean_pair(reference, key[2]),
                "reference_origin_methods": "+".join(origin),
                "strong_classical_best_method": classical_best["method"],
                "strong_classical_best_objective": classical_best["objective"],
                "strong_classical_reference_match": classical_best["objective"] <= reference + 1e-12,
                "sampler_best_objective": sampler_best["objective"],
                "sampler_reference_match": sampler_best["objective"] <= reference + 1e-12,
                "sampler_delta_vs_strong_classical_per_pair": pair_delta,
                "sampler_within_tolerance_of_strong_classical": pair_delta <= tolerance,
                "sampler_strict_win_vs_strong_classical": pair_delta < -tolerance,
                "strong_classical_strict_win_vs_sampler": pair_delta > tolerance,
                "solver_best_mean_pair_spread": max(method_values.values()) - min(method_values.values()),
                "solver_disagreement": max(method_values.values()) - min(method_values.values()) > tolerance,
            }
        )
    return cells


def compare_cell_rows(observed: list[dict[str, str]], expected: list[dict[str, Any]]) -> None:
    fields_bool = (
        "exact_oracle_available",
        "strong_classical_reference_match",
        "sampler_reference_match",
        "sampler_within_tolerance_of_strong_classical",
        "sampler_strict_win_vs_strong_classical",
        "strong_classical_strict_win_vs_sampler",
        "solver_disagreement",
    )
    fields_float = (
        "reference_objective",
        "reference_mean_pair_redundancy",
        "strong_classical_best_objective",
        "sampler_best_objective",
        "sampler_delta_vs_strong_classical_per_pair",
        "solver_best_mean_pair_spread",
    )
    mapping = {
        (row["target_id"], int(row["outer_fold"]), int(row["k"]), row["quality_regime"]): row
        for row in observed
    }
    if len(mapping) != len(observed) or len(mapping) != len(expected):
        raise ValueError("Stage74 audit cell-comparison grid differs")
    for item in expected:
        key = (item["target_id"], item["outer_fold"], item["k"], item["quality_regime"])
        row = mapping[key]
        for field in (
            "reference_type",
            "reference_origin_methods",
            "strong_classical_best_method",
        ):
            if row[field] != str(item[field]):
                raise ValueError(f"Stage74 audit cell {field} differs: {key}")
        for field in fields_bool:
            if truth(row[field]) != bool(item[field]):
                raise ValueError(f"Stage74 audit cell {field} differs: {key}")
        for field in fields_float:
            close(row[field], item[field], f"cell {field} {key}")


def expected_method_summaries(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        for scope in ("ALL", row["target_id"]):
            groups[(row["method"], row["quality_regime"], row["k"], scope)].append(row)
    output: list[dict[str, Any]] = []
    for (method, regime, k, scope), rows in sorted(groups.items()):
        exact = [row for row in rows if row["exact_oracle_available"]]
        output.append(
            {
                "method": method,
                "quality_regime": regime,
                "k": k,
                "scope": scope,
                "trial_count": len(rows),
                "workload_cell_count": len({(row["target_id"], row["outer_fold"]) for row in rows}),
                "exact_oracle_trial_count": len(exact),
                "exact_optimum_success_rate": statistics.fmean(bool(row["exact_match"]) for row in exact) if exact else "",
                "reference_match_rate": statistics.fmean(bool(row["reference_match"]) for row in rows),
                "mean_reference_regret_per_pair": statistics.fmean(row["reference_regret"] for row in rows),
                "maximum_reference_regret_per_pair": max(row["reference_regret"] for row in rows),
                "mean_initialization_attempt_count": statistics.fmean(row["initialization_attempt_count"] for row in rows),
                "initialization_fallback_count": sum(row["initialization_fallback_count"] for row in rows),
                "mean_swap_proposal_count": statistics.fmean(row["proposal_count"] for row in rows),
            }
        )
    return output


def compare_summaries(observed: list[dict[str, str]], expected: list[dict[str, Any]]) -> None:
    mapping = {
        (row["method"], row["quality_regime"], int(row["k"]), row["scope"]): row
        for row in observed
    }
    if len(mapping) != len(observed) or len(mapping) != len(expected):
        raise ValueError("Stage74 audit method-summary grid differs")
    integer_fields = (
        "trial_count",
        "workload_cell_count",
        "exact_oracle_trial_count",
        "initialization_fallback_count",
    )
    float_fields = (
        "reference_match_rate",
        "mean_reference_regret_per_pair",
        "maximum_reference_regret_per_pair",
        "mean_initialization_attempt_count",
        "mean_swap_proposal_count",
    )
    for item in expected:
        key = (item["method"], item["quality_regime"], item["k"], item["scope"])
        row = mapping[key]
        for field in integer_fields:
            if int(row[field]) != int(item[field]):
                raise ValueError(f"Stage74 audit summary {field} differs: {key}")
        for field in float_fields:
            close(row[field], item[field], f"summary {field} {key}")
        if item["exact_optimum_success_rate"] == "":
            if row["exact_optimum_success_rate"]:
                raise ValueError("Stage74 audit empty exact summary differs")
        else:
            close(row["exact_optimum_success_rate"], item["exact_optimum_success_rate"], f"summary exact rate {key}")


def aggregate(
    workloads: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    exact = [row for row in cells if row["exact_oracle_available"]]
    nonexact = [row for row in cells if not row["exact_oracle_available"]]
    first_regime = config["larger_k_workloads"]["quality_regimes"][0]["name"]
    scaling = {
        "model_count": len({(row["target_id"], row["outer_fold"]) for row in workloads}),
        "model_k_count": len({(row["target_id"], row["outer_fold"], row["k"]) for row in workloads}),
        "workload_cell_count": len(workloads),
        "solver_trial_count": len(trials),
        "exact_oracle_cell_count": len(exact),
        "maximum_candidate_count": max(row["candidate_count"] for row in workloads),
        "maximum_k": max(row["k"] for row in workloads),
        "maximum_total_fixed_k_subset_count": max(row["total"] for row in workloads),
        "maximum_log10_total_fixed_k_subset_count": max(math.log10(row["total"]) for row in workloads),
        "maximum_quadratic_coupler_count": max(row["couplers"] for row in workloads),
        "exact_oracle_unique_state_checks": sum(
            row["total"]
            for row in workloads
            if row["oracle"] is not None and row["name"] == first_regime
        ),
    }
    strong_rate = statistics.fmean(row["strong_classical_reference_match"] for row in exact)
    sampler_rate = statistics.fmean(row["sampler_reference_match"] for row in exact)
    disagreement = statistics.fmean(row["solver_disagreement"] for row in nonexact)
    competitive = statistics.fmean(row["sampler_within_tolerance_of_strong_classical"] for row in nonexact)
    hardness = {
        "nonexact_workload_cell_count": len(nonexact),
        "nonexact_solver_disagreement_cell_count": sum(row["solver_disagreement"] for row in nonexact),
        "nonexact_solver_disagreement_fraction": disagreement,
        "sampler_strict_win_cell_count": sum(row["sampler_strict_win_vs_strong_classical"] for row in nonexact),
        "strong_classical_strict_win_cell_count": sum(row["strong_classical_strict_win_vs_sampler"] for row in nonexact),
        "sampler_tie_within_tolerance_cell_count": sum(
            not row["sampler_strict_win_vs_strong_classical"] and not row["strong_classical_strict_win_vs_sampler"]
            for row in nonexact
        ),
        "maximum_solver_best_mean_pair_spread": max(row["solver_best_mean_pair_spread"] for row in nonexact),
    }
    validation = {
        "strong_classical_exact_cell_success_rate": strong_rate,
        "sampler_exact_cell_success_rate": sampler_rate,
        "sampler_nonexact_competitive_fraction": competitive,
    }
    gate = config["benchmark_gate"]
    routes = {
        "large_scale_state_space_reached": scaling["maximum_total_fixed_k_subset_count"] >= int(gate["minimum_large_scale_state_count"]),
        "strong_classical_exact_validation_passed": strong_rate >= float(gate["minimum_strong_classical_exact_cell_success_rate"]),
        "nonexact_solver_hardness_observed": disagreement >= float(gate["minimum_nonexact_solver_disagreement_fraction"]),
        "qubo_sampler_analog_competitiveness_passed": competitive >= float(gate["minimum_sampler_nonexact_competitive_fraction"]),
    }
    decision = {
        "explicit_variable_k_cqm_design_authorized": routes["large_scale_state_space_reached"] and routes["strong_classical_exact_validation_passed"] and routes["nonexact_solver_hardness_observed"],
        "hardware_shaped_sampler_poc_authorized": all(routes.values()),
        "direct_qpu_execution_authorized": False,
        "quantum_scaling_claim_authorized": False,
        "quantum_advantage_claim_authorized": False,
    }
    return {
        "scaling_summary": scaling,
        "exact_validation": validation,
        "hardness_summary": hardness,
        "route_gate": routes,
        "decision": decision,
    }


def run(
    config_path: Path, result_path: Path, root: Path, output_path: Path
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    result_path = result_path.resolve()
    output_path = output_path.resolve()
    if config_path != (root / "configs/stage74_larger_k_solver_scaling.json").resolve():
        raise ValueError("Stage74 audit config path differs")
    if result_path != (root / "data/stage74_larger_k_solver_scaling_result.json").resolve():
        raise ValueError("Stage74 audit result path differs")
    if output_path != (root / "data/stage74_larger_k_solver_scaling_audit.json").resolve():
        raise ValueError("Stage74 audit output path differs")
    config = read_json(config_path)
    result = read_json(result_path)
    for key, value in config["implementation"].items():
        checked(root, value, f"implementation {key}")
    inputs = {
        key: checked(root, value, f"input {key}")
        for key, value in config["inputs"].items()
    }
    paths = {key: root / str(value) for key, value in config["outputs"].items()}
    observed_workloads = read_csv(paths["workload_metrics_csv"])
    observed_trials = read_csv(paths["solver_trials_csv"])
    observed_cells = read_csv(paths["cell_comparison_csv"])
    observed_summaries = read_csv(paths["solver_summary_csv"])
    workload_map = {
        (row["target_id"], int(row["outer_fold"]), int(row["k"]), row["quality_regime"]): row
        for row in observed_workloads
    }
    trial_map = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            int(row["k"]),
            row["quality_regime"],
            row["method"],
            int(row["repeat"]),
        ): row
        for row in observed_trials
    }
    if len(workload_map) != len(observed_workloads) or len(trial_map) != len(observed_trials):
        raise ValueError("Stage74 audit found duplicate output rows")
    source = read_json(inputs["stage72_model_record"])
    models = [rebuild(record) for record in source["models"]]
    methods = [str(value) for value in config["solver_protocol"]["method_order"]]
    stochastic = set(config["solver_protocol"]["stochastic_methods"])
    repeats = int(config["solver_protocol"]["stochastic_repeats"])
    exact_limit = int(config["larger_k_workloads"]["exact_enumeration_state_limit"])
    base_seed = int(config["solver_protocol"]["seed_base"])
    regimes = [str(item["name"]) for item in config["larger_k_workloads"]["quality_regimes"]]
    expected_trials: list[dict[str, Any]] = []
    workload_records: list[dict[str, Any]] = []
    cell_index = 0
    for model in models:
        ks = schedule(model, config)
        values = distributions(model["deficits"], max(ks))
        for k in ks:
            total = math.comb(model["count"], k)
            thresholds = threshold_grid(values[k], total, config)
            oracle = oracle_grid(model, k, thresholds) if total <= exact_limit else None
            for name in regimes:
                cell = make_cell(model, k, name, thresholds[name], oracle[name] if oracle else None)
                key = (str(model["record"]["target_id"]), int(model["record"]["outer_fold"]), k, name)
                observed = workload_map.get(key)
                if observed is None:
                    raise ValueError(f"Stage74 audit missing workload: {key}")
                compare_workload(observed, cell)
                workload_records.append(
                    {
                        "target_id": key[0],
                        "outer_fold": key[1],
                        "candidate_count": model["count"],
                        "k": k,
                        "name": name,
                        "total": total,
                        "couplers": model["couplers"],
                        "oracle": cell["oracle"],
                    }
                )
                for method_index, method in enumerate(methods):
                    if method == "exact_enumeration" and cell["oracle"] is None:
                        continue
                    count = repeats if method in stochastic else 1
                    for repeat in range(count):
                        seed = base_seed + cell_index * 100_000 + method_index * 1_000 + repeat
                        trial_key = key + (method, repeat)
                        row = trial_map.get(trial_key)
                        if row is None:
                            raise ValueError(f"Stage74 audit missing trial: {trial_key}")
                        solved = expected_solution(cell, method, seed, config)
                        expected_trials.append(compare_trial(row, cell, method, repeat, seed, solved, config))
                cell_index += 1
        print(json.dumps({"audit_target": model["record"]["target_id"], "audit_fold": model["record"]["outer_fold"], "trials_replayed": len(expected_trials)}), flush=True)
    if len(observed_workloads) != cell_index or len(observed_trials) != len(expected_trials):
        raise ValueError("Stage74 audit output cardinality differs")
    expected_cells = attach_expected_references(expected_trials, config)
    compare_cell_rows(observed_cells, expected_cells)
    compare_summaries(observed_summaries, expected_method_summaries(expected_trials))
    aggregate_value = aggregate(workload_records, expected_trials, expected_cells, config)
    for section in ("scaling_summary", "exact_validation", "hardness_summary", "route_gate"):
        observed = result[section]
        expected = aggregate_value[section]
        if section in {"scaling_summary", "hardness_summary"}:
            for key, value in expected.items():
                if isinstance(value, float):
                    close(observed[key], value, f"result {section}.{key}")
                elif observed[key] != value:
                    raise ValueError(f"Stage74 audit result {section}.{key} differs")
        elif section == "exact_validation":
            for key, value in expected.items():
                close(observed[key], value, f"result {section}.{key}")
        elif observed != expected:
            raise ValueError(f"Stage74 audit result {section} differs")
    for key, value in aggregate_value["decision"].items():
        if bool(result["decision"][key]) != bool(value):
            raise ValueError(f"Stage74 audit decision differs: {key}")
    payload = {
        **aggregate_value,
        "workload_metrics_sha256": sha256(paths["workload_metrics_csv"]),
        "solver_trials_sha256": sha256(paths["solver_trials_csv"]),
        "cell_comparison_sha256": sha256(paths["cell_comparison_csv"]),
        "solver_summary_sha256": sha256(paths["solver_summary_csv"]),
    }
    if canonical_sha256(payload) != result["analysis_payload_sha256"]:
        raise ValueError("Stage74 audit analysis payload hash differs")
    expected_boundary = {
        "historical_development_targets_read": 4,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "cloud_cqm_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
    if result["data_boundary"] != expected_boundary:
        raise ValueError("Stage74 audit data boundary differs")
    audit = {
        "schema_version": "1.0",
        "status": "stage74_larger_k_solver_scaling_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "stage72_models_independently_rebuilt": len(models),
        "workload_cells_independently_recomputed": cell_index,
        "solver_trials_deterministically_replayed": len(expected_trials),
        "cell_comparisons_independently_recomputed": len(expected_cells),
        "solver_summaries_independently_recomputed": len(observed_summaries),
        **aggregate_value["route_gate"],
        **aggregate_value["decision"],
        "data_boundary": expected_boundary,
    }
    write_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage74_larger_k_solver_scaling.json"))
    parser.add_argument("--result", type=Path, default=Path("data/stage74_larger_k_solver_scaling_result.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/stage74_larger_k_solver_scaling_audit.json"))
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
