"""Independently audit the Stage73 constraint-native solver benchmark."""

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
from pathlib import Path
from typing import Any

import numpy as np


TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()




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
    if not path.is_file():
        raise FileNotFoundError(f"Stage73 audit missing {label}: {path}")
    if sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage73 audit {label} hash differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage73 audit {label} size differs: {path}")
    return path


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def close(observed: Any, expected: Any, label: str) -> None:
    if not math.isclose(
        float(observed), float(expected), rel_tol=TOLERANCE, abs_tol=TOLERANCE
    ):
        raise ValueError(f"Stage73 audit {label} differs: {observed} != {expected}")


def receptor_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def rebuild_model(record: dict[str, Any]) -> dict[str, Any]:
    receptor_ids = tuple(str(value) for value in record["receptor_ids"])
    count = len(receptor_ids)
    pair_order = tuple(itertools.combinations(range(count), 2))
    centered = tuple(float(value) for value in record["centered_pair_coefficients"])
    if len(centered) != len(pair_order):
        raise ValueError("Stage73 audit centered coefficient count differs")
    midpoint = float(record["pair_midpoint_center"])
    pair_values = {
        pair: coefficient + midpoint
        for pair, coefficient in zip(pair_order, centered)
    }
    deficits = tuple(int(value) for value in record["integer_deficits"])
    reference_k = int(record["reference_k"])
    quality_limit = int(record["maximum_integer_deficit"])
    canonical = {
        "variable_order": [f"x{index:03d}" for index in range(count)],
        "objective": {
            "pair_center": midpoint,
            "quadratic_pair_order": [list(pair) for pair in pair_order],
            "quadratic_coefficients": list(centered),
            "offset": float(record["objective_offset"]),
        },
        "constraints": {
            "cardinality_exact": {
                "sense": "==",
                "linear_coefficients": [1] * count,
                "rhs": reference_k,
            },
            "quality_floor": {
                "sense": "<=",
                "linear_coefficients": list(deficits),
                "rhs": quality_limit,
            },
        },
    }
    if canonical_sha256(canonical) != str(record["cqm_sha256"]).upper():
        raise ValueError("Stage73 audit failed to reconstruct a Stage72 CQM")
    order = tuple(
        sorted(
            range(count),
            key=lambda index: (
                deficits[index],
                receptor_hash(receptor_ids[index]),
                receptor_ids[index],
            ),
        )
    )
    return {
        "record": record,
        "receptor_ids": receptor_ids,
        "count": count,
        "pair_values": pair_values,
        "deficits": deficits,
        "reference_k": reference_k,
        "quality_limit": quality_limit,
        "pool_order": order,
    }


def objective(model: dict[str, Any], subset: tuple[int, ...]) -> float:
    return float(
        sum(
            model["pair_values"][tuple(sorted(pair))]
            for pair in itertools.combinations(subset, 2)
        )
    )


def deficit(model: dict[str, Any], subset: tuple[int, ...]) -> int:
    return sum(model["deficits"][index] for index in subset)


def scheduled_sizes(model: dict[str, Any], config: dict[str, Any]) -> list[int]:
    maximum = int(model["count"])
    sizes = {
        int(value)
        for value in config["scaling_workloads"]["candidate_pool_sizes"]
        if int(value) <= maximum
    }
    if truth(config["scaling_workloads"]["always_include_full_pool"]):
        sizes.add(maximum)
    output = sorted(sizes)
    if not output or min(output) < int(model["reference_k"]):
        raise ValueError("Stage73 audit found an invalid pool schedule")
    return output


def build_pool(
    model: dict[str, Any], size: int, config: dict[str, Any]
) -> dict[str, Any]:
    pool = tuple(sorted(model["pool_order"][:size]))
    subsets = tuple(itertools.combinations(pool, int(model["reference_k"])))
    deficits = tuple(deficit(model, subset) for subset in subsets)
    objectives = tuple(objective(model, subset) for subset in subsets)
    ordered_deficits = sorted(deficits)
    fraction = float(config["scaling_workloads"]["relaxed_feasible_quantile"])
    position = max(0, math.ceil(fraction * len(ordered_deficits)) - 1)
    thresholds = {
        "frozen_quality_floor": int(model["quality_limit"]),
        "relaxed_10pct_quality_floor": max(
            int(model["quality_limit"]), int(ordered_deficits[position])
        ),
        "no_quality_floor": max(deficits),
    }
    pair_values = [
        model["pair_values"][pair] for pair in itertools.combinations(pool, 2)
    ]
    midpoint = (min(pair_values) + max(pair_values)) / 2.0
    scale = max(abs(value - midpoint) for value in pair_values)
    return {
        "pool": pool,
        "subsets": subsets,
        "deficits": deficits,
        "objectives": objectives,
        "thresholds": thresholds,
        "scale": scale if scale > 1e-12 else 1.0,
    }


def build_cell(
    model: dict[str, Any], pool: dict[str, Any], regime: str
) -> dict[str, Any]:
    threshold = int(pool["thresholds"][regime])
    feasible = tuple(
        subset
        for subset, value in zip(pool["subsets"], pool["deficits"])
        if value <= threshold
    )
    if not feasible:
        raise ValueError("Stage73 audit found an empty feasible set")
    values = tuple(objective(model, subset) for subset in feasible)
    optimum = min(values)
    optima = tuple(
        subset
        for subset, value in zip(feasible, values)
        if value <= optimum + 1e-12
    )
    return {
        "model": model,
        "pool": pool["pool"],
        "pool_size": len(pool["pool"]),
        "regime": regime,
        "threshold": threshold,
        "total_count": len(pool["subsets"]),
        "feasible": feasible,
        "feasible_count": len(feasible),
        "feasible_set": set(feasible),
        "scale": float(pool["scale"]),
        "optimum": float(optimum),
        "canonical_optimum": min(optima),
        "degeneracy": len(optima),
        "full_pool": len(pool["pool"]) == int(model["count"]),
    }


def subset_label(model: dict[str, Any], subset: tuple[int, ...]) -> str:
    return "+".join(model["receptor_ids"][index] for index in subset)


def swaps(cell: dict[str, Any], subset: tuple[int, ...]) -> list[tuple[int, ...]]:
    selected = set(subset)
    return sorted(
        {
            tuple(sorted((selected - {outgoing}) | {incoming}))
            for outgoing in subset
            for incoming in cell["pool"]
            if incoming not in selected
        }
    )


def replay_single_greedy(cell: dict[str, Any]) -> dict[str, Any]:
    model = cell["model"]
    current = min(cell["feasible"], key=lambda state: (deficit(model, state), state))
    current_value = objective(model, current)
    proposals = 0
    feasible_proposals = 0
    evaluations = 1
    accepted = 0
    while True:
        candidate_best = current
        value_best = current_value
        for candidate in swaps(cell, current):
            proposals += 1
            if deficit(model, candidate) > cell["threshold"]:
                continue
            feasible_proposals += 1
            evaluations += 1
            value = objective(model, candidate)
            if (value, candidate) < (value_best - 1e-12, candidate_best):
                candidate_best = candidate
                value_best = value
        if value_best >= current_value - 1e-12:
            break
        current = candidate_best
        current_value = value_best
        accepted += 1
    return {
        "subset": current,
        "proposal_count": proposals,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "restart_count": 1,
    }


def replay_random(
    cell: dict[str, Any], budget: int, rng: np.random.Generator
) -> dict[str, Any]:
    drawn = rng.integers(0, cell["feasible_count"], size=budget)
    candidates = [cell["feasible"][int(index)] for index in drawn]
    best = min(candidates, key=lambda state: (objective(cell["model"], state), state))
    return {
        "subset": best,
        "proposal_count": budget,
        "feasible_proposal_count": budget,
        "objective_evaluation_count": budget,
        "accepted_move_count": 0,
        "restart_count": budget,
    }


def replay_multistart(
    cell: dict[str, Any], budget: int, rng: np.random.Generator
) -> dict[str, Any]:
    model = cell["model"]
    proposals = feasible_proposals = evaluations = accepted = restarts = 0
    best: tuple[int, ...] | None = None
    best_value = math.inf
    while proposals < budget:
        restarts += 1
        current = cell["feasible"][int(rng.integers(0, cell["feasible_count"]))]
        current_value = objective(model, current)
        evaluations += 1
        if best is None or (current_value, current) < (best_value, best):
            best, best_value = current, current_value
        while proposals < budget:
            candidates = swaps(cell, current)
            order = rng.permutation(len(candidates))
            local_best, local_value = current, current_value
            for index in order:
                if proposals >= budget:
                    break
                candidate = candidates[int(index)]
                proposals += 1
                if deficit(model, candidate) > cell["threshold"]:
                    continue
                feasible_proposals += 1
                evaluations += 1
                value = objective(model, candidate)
                if (value, candidate) < (local_value - 1e-12, local_best):
                    local_best, local_value = candidate, value
                if best is None or (value, candidate) < (best_value, best):
                    best, best_value = candidate, value
            if local_value >= current_value - 1e-12:
                break
            current, current_value = local_best, local_value
            accepted += 1
    if best is None:
        raise ValueError("Stage73 audit multistart replay produced no solution")
    return {
        "subset": best,
        "proposal_count": proposals,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "restart_count": restarts,
    }


def replay_annealing(
    cell: dict[str, Any],
    budget: int,
    beta_minimum: float,
    beta_maximum: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    model = cell["model"]
    current = cell["feasible"][int(rng.integers(0, cell["feasible_count"]))]
    current_value = objective(model, current)
    best, best_value = current, current_value
    feasible_proposals = 0
    evaluations = 1
    accepted = 0
    for step in range(budget):
        selected = set(current)
        outgoing = current[int(rng.integers(0, len(current)))]
        available = [value for value in cell["pool"] if value not in selected]
        incoming = available[int(rng.integers(0, len(available)))]
        candidate = tuple(sorted((selected - {outgoing}) | {incoming}))
        if deficit(model, candidate) > cell["threshold"]:
            continue
        feasible_proposals += 1
        evaluations += 1
        candidate_value = objective(model, candidate)
        delta = (candidate_value - current_value) / cell["scale"]
        fraction = step / max(1, budget - 1)
        beta = beta_minimum * (beta_maximum / beta_minimum) ** fraction
        if delta <= 0 or rng.random() < math.exp(-beta * delta):
            current, current_value = candidate, candidate_value
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
    }


def expected_solution(
    cell: dict[str, Any], method: str, seed: int, config: dict[str, Any]
) -> dict[str, Any]:
    budget = int(config["solver_protocol"]["proposal_budget"])
    if method == "exact_enumeration":
        return {
            "subset": cell["canonical_optimum"],
            "proposal_count": cell["total_count"],
            "feasible_proposal_count": cell["feasible_count"],
            "objective_evaluation_count": cell["feasible_count"],
            "accepted_move_count": 0,
            "restart_count": 1,
        }
    if method == "single_start_best_improvement":
        return replay_single_greedy(cell)
    rng = np.random.default_rng(seed)
    if method == "budgeted_random_feasible":
        return replay_random(cell, budget, rng)
    if method == "budgeted_multistart_greedy":
        return replay_multistart(cell, budget, rng)
    if method == "constraint_preserving_annealing":
        beta_minimum, beta_maximum = config["solver_protocol"][
            "annealing_beta_range"
        ]
        return replay_annealing(
            cell, budget, float(beta_minimum), float(beta_maximum), rng
        )
    raise ValueError(f"Stage73 audit unknown method: {method}")


def compare_workload(row: dict[str, str], cell: dict[str, Any]) -> None:
    model = cell["model"]
    expected_text = {
        "target_id": str(model["record"]["target_id"]),
        "quality_regime": cell["regime"],
        "exact_optimum_subset": subset_label(model, cell["canonical_optimum"]),
        "pool_receptor_ids_sha256": canonical_sha256(
            [model["receptor_ids"][index] for index in cell["pool"]]
        ),
    }
    for field, expected in expected_text.items():
        if row[field] != expected:
            raise ValueError(f"Stage73 audit workload {field} differs")
    expected_integer = {
        "outer_fold": model["record"]["outer_fold"],
        "pool_size": cell["pool_size"],
        "quality_threshold": cell["threshold"],
        "total_fixed_k_subset_count": cell["total_count"],
        "feasible_subset_count": cell["feasible_count"],
        "exact_optimum_degeneracy": cell["degeneracy"],
    }
    for field, expected in expected_integer.items():
        if int(row[field]) != int(expected):
            raise ValueError(f"Stage73 audit workload {field} differs")
    if truth(row["is_full_pool"]) != bool(cell["full_pool"]):
        raise ValueError("Stage73 audit workload full-pool flag differs")
    close(row["feasible_subset_fraction"], cell["feasible_count"] / cell["total_count"], "workload feasible fraction")
    close(row["exact_optimum_objective"], cell["optimum"], "workload optimum")
    close(row["objective_normalization_scale"], cell["scale"], "workload scale")


def compare_trial(
    row: dict[str, str],
    cell: dict[str, Any],
    method: str,
    repeat: int,
    seed: int,
    solved: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    model = cell["model"]
    if row["method"] != method or int(row["repeat"]) != repeat:
        raise ValueError("Stage73 audit trial identity differs")
    if int(row["seed"]) != seed:
        raise ValueError("Stage73 audit trial seed differs")
    if tuple(solved["subset"]) not in cell["feasible_set"]:
        raise ValueError("Stage73 replay produced an infeasible solution")
    expected_text = subset_label(model, solved["subset"])
    if row["solution_subset"] != expected_text:
        raise ValueError("Stage73 audit trial solution subset differs")
    for field in (
        "proposal_count",
        "feasible_proposal_count",
        "objective_evaluation_count",
        "accepted_move_count",
        "restart_count",
    ):
        if int(row[field]) != int(solved[field]):
            raise ValueError(f"Stage73 audit trial {field} differs")
    effective_budget = (
        cell["total_count"]
        if method == "exact_enumeration"
        else solved["proposal_count"]
        if method == "single_start_best_improvement"
        else int(config["solver_protocol"]["proposal_budget"])
    )
    if int(row["configured_proposal_budget"]) != int(effective_budget):
        raise ValueError("Stage73 audit configured proposal budget differs")
    value = objective(model, solved["subset"])
    regret = value - cell["optimum"]
    expected_exact = regret <= 1e-12
    expected_canonical = solved["subset"] == cell["canonical_optimum"]
    exact_set = set(cell["canonical_optimum"])
    solution_set = set(solved["subset"])
    jaccard = len(exact_set & solution_set) / len(exact_set | solution_set)
    close(row["solution_objective"], value, "trial objective")
    close(row["absolute_objective_regret"], regret, "trial regret")
    close(row["normalized_objective_regret"], regret / cell["scale"], "trial normalized regret")
    close(row["canonical_optimum_jaccard"], jaccard, "trial Jaccard")
    if truth(row["exact_optimum_match"]) != expected_exact:
        raise ValueError("Stage73 audit exact-optimum flag differs")
    if truth(row["canonical_optimum_subset_match"]) != expected_canonical:
        raise ValueError("Stage73 audit canonical-optimum flag differs")
    return {
        "target_id": str(model["record"]["target_id"]),
        "outer_fold": int(model["record"]["outer_fold"]),
        "pool_size": int(cell["pool_size"]),
        "is_full_pool": bool(cell["full_pool"]),
        "quality_regime": cell["regime"],
        "method": method,
        "exact": expected_exact,
        "canonical": expected_canonical,
        "regret": regret,
        "normalized_regret": regret / cell["scale"],
        "jaccard": jaccard,
        "proposal_count": int(solved["proposal_count"]),
        "evaluation_count": int(solved["objective_evaluation_count"]),
    }


def summary_rows(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in trials:
        for scope in ("ALL", row["target_id"]):
            key = (row["method"], row["quality_regime"], row["pool_size"], scope)
            groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (method, regime, size, scope), rows in sorted(groups.items()):
        output.append(
            {
                "method": method,
                "quality_regime": regime,
                "pool_size": size,
                "scope": scope,
                "trial_count": len(rows),
                "workload_cell_count": len(
                    {(row["target_id"], row["outer_fold"]) for row in rows}
                ),
                "exact_optimum_success_rate": statistics.fmean(
                    int(row["exact"]) for row in rows
                ),
                "canonical_subset_success_rate": statistics.fmean(
                    int(row["canonical"]) for row in rows
                ),
                "mean_absolute_objective_regret": statistics.fmean(
                    row["regret"] for row in rows
                ),
                "maximum_absolute_objective_regret": max(row["regret"] for row in rows),
                "mean_normalized_objective_regret": statistics.fmean(
                    row["normalized_regret"] for row in rows
                ),
                "mean_canonical_optimum_jaccard": statistics.fmean(
                    row["jaccard"] for row in rows
                ),
                "mean_proposal_count": statistics.fmean(
                    row["proposal_count"] for row in rows
                ),
                "mean_objective_evaluation_count": statistics.fmean(
                    row["evaluation_count"] for row in rows
                ),
            }
        )
    return output


def compare_summaries(
    observed: list[dict[str, str]], expected: list[dict[str, Any]]
) -> None:
    key_fields = ("method", "quality_regime", "pool_size", "scope")
    observed_map = {
        (row["method"], row["quality_regime"], int(row["pool_size"]), row["scope"]): row
        for row in observed
    }
    if len(observed_map) != len(observed) or len(observed_map) != len(expected):
        raise ValueError("Stage73 audit solver-summary grid differs")
    integer_fields = ("trial_count", "workload_cell_count")
    float_fields = (
        "exact_optimum_success_rate",
        "canonical_subset_success_rate",
        "mean_absolute_objective_regret",
        "maximum_absolute_objective_regret",
        "mean_normalized_objective_regret",
        "mean_canonical_optimum_jaccard",
        "mean_proposal_count",
        "mean_objective_evaluation_count",
    )
    for item in expected:
        key = tuple(item[field] for field in key_fields)
        row = observed_map.get(key)
        if row is None:
            raise ValueError(f"Stage73 audit missing solver summary: {key}")
        for field in integer_fields:
            if int(row[field]) != int(item[field]):
                raise ValueError(f"Stage73 audit summary {field} differs: {key}")
        for field in float_fields:
            close(row[field], item[field], f"summary {field} {key}")


def performance(
    trials: list[dict[str, Any]], method: str, full_frozen_only: bool = True
) -> dict[str, Any]:
    selected = [
        row
        for row in trials
        if row["method"] == method
        and (
            not full_frozen_only
            or (row["is_full_pool"] and row["quality_regime"] == "frozen_quality_floor")
        )
    ]
    return {
        "trial_count": len(selected),
        "workload_cell_count": len(
            {
                (row["target_id"], row["outer_fold"], row["pool_size"], row["quality_regime"])
                for row in selected
            }
        ),
        "exact_optimum_success_rate": statistics.fmean(
            int(row["exact"]) for row in selected
        ),
        "mean_normalized_objective_regret": statistics.fmean(
            row["normalized_regret"] for row in selected
        ),
        "maximum_normalized_objective_regret": max(
            row["normalized_regret"] for row in selected
        ),
    }


def run(
    config_path: Path, result_path: Path, root: Path, output_path: Path
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    result_path = result_path.resolve()
    output_path = output_path.resolve()
    expected_config = root / "configs/stage73_constraint_native_solver_scaling.json"
    expected_result = root / "data/stage73_constraint_native_solver_scaling_result.json"
    expected_output = root / "data/stage73_constraint_native_solver_scaling_audit.json"
    if config_path != expected_config.resolve() or result_path != expected_result.resolve():
        raise ValueError("Stage73 audit must use the frozen repository inputs")
    if output_path != expected_output.resolve():
        raise ValueError("Stage73 audit output path differs")
    config = read_json(config_path)
    result = read_json(result_path)
    for key, value in config["implementation"].items():
        checked(root, value, f"implementation {key}")
    input_paths = {
        key: checked(root, value, f"input {key}")
        for key, value in config["inputs"].items()
    }
    output_paths = {
        key: root / str(value) for key, value in config["outputs"].items()
    }
    workload_observed = read_csv(output_paths["workload_metrics_csv"])
    trial_observed = read_csv(output_paths["solver_trials_csv"])
    summary_observed = read_csv(output_paths["solver_summary_csv"])
    workload_map = {
        (row["target_id"], int(row["outer_fold"]), int(row["pool_size"]), row["quality_regime"]): row
        for row in workload_observed
    }
    trial_map = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            int(row["pool_size"]),
            row["quality_regime"],
            row["method"],
            int(row["repeat"]),
        ): row
        for row in trial_observed
    }
    if len(workload_map) != len(workload_observed):
        raise ValueError("Stage73 audit found duplicate workload rows")
    if len(trial_map) != len(trial_observed):
        raise ValueError("Stage73 audit found duplicate solver trials")
    source = read_json(input_paths["stage72_model_record"])
    models = [rebuild_model(record) for record in source["models"]]
    methods = [str(value) for value in config["solver_protocol"]["method_order"]]
    stochastic = set(config["solver_protocol"]["stochastic_methods"])
    repeats = int(config["solver_protocol"]["stochastic_repeats"])
    seed_base = int(config["solver_protocol"]["seed_base"])
    regimes = [str(value) for value in config["scaling_workloads"]["quality_regimes"]]
    expected_trial_records: list[dict[str, Any]] = []
    exact_state_checks = 0
    maximum_pool = maximum_total = maximum_feasible = maximum_full_frozen = 0
    cell_index = 0
    for model in models:
        for size in scheduled_sizes(model, config):
            pool = build_pool(model, size, config)
            for regime in regimes:
                cell = build_cell(model, pool, regime)
                key = (
                    str(model["record"]["target_id"]),
                    int(model["record"]["outer_fold"]),
                    size,
                    regime,
                )
                row = workload_map.get(key)
                if row is None:
                    raise ValueError(f"Stage73 audit missing workload row: {key}")
                compare_workload(row, cell)
                exact_state_checks += cell["total_count"]
                maximum_pool = max(maximum_pool, size)
                maximum_total = max(maximum_total, cell["total_count"])
                maximum_feasible = max(maximum_feasible, cell["feasible_count"])
                if cell["full_pool"] and regime == "frozen_quality_floor":
                    maximum_full_frozen = max(maximum_full_frozen, cell["feasible_count"])
                for method_index, method in enumerate(methods):
                    method_repeats = repeats if method in stochastic else 1
                    for repeat in range(method_repeats):
                        seed = seed_base + cell_index * 100_000 + method_index * 1_000 + repeat
                        trial_key = key + (method, repeat)
                        observed = trial_map.get(trial_key)
                        if observed is None:
                            raise ValueError(f"Stage73 audit missing solver trial: {trial_key}")
                        solved = expected_solution(cell, method, seed, config)
                        expected_trial_records.append(
                            compare_trial(observed, cell, method, repeat, seed, solved, config)
                        )
                cell_index += 1
    if len(workload_observed) != cell_index:
        raise ValueError("Stage73 audit workload count differs")
    if len(trial_observed) != len(expected_trial_records):
        raise ValueError("Stage73 audit solver-trial count differs")
    compare_summaries(summary_observed, summary_rows(expected_trial_records))
    scaling = {
        "model_count": len(models),
        "workload_cell_count": cell_index,
        "solver_trial_count": len(expected_trial_records),
        "maximum_pool_size": maximum_pool,
        "maximum_total_fixed_k_subset_count": maximum_total,
        "maximum_feasible_subset_count": maximum_feasible,
        "maximum_full_pool_frozen_feasible_subset_count": maximum_full_frozen,
        "total_exact_enumeration_state_checks": exact_state_checks,
    }
    if result["scaling_summary"] != scaling:
        raise ValueError("Stage73 audit scaling summary differs")
    full_frozen = {method: performance(expected_trial_records, method) for method in methods}
    for method, expected in full_frozen.items():
        observed = result["full_pool_frozen_performance"][method]
        for field in ("trial_count", "workload_cell_count"):
            if int(observed[field]) != int(expected[field]):
                raise ValueError(f"Stage73 audit full-pool {method} {field} differs")
        for field in (
            "exact_optimum_success_rate",
            "mean_normalized_objective_regret",
            "maximum_normalized_objective_regret",
        ):
            close(observed[field], expected[field], f"full-pool {method} {field}")
    nonoracle = [row for row in expected_trial_records if row["method"] != "exact_enumeration"]
    hardness = {
        "single_start_greedy_failure_cell_count": len(
            {
                (row["target_id"], row["outer_fold"], row["pool_size"], row["quality_regime"])
                for row in nonoracle
                if row["method"] == "single_start_best_improvement" and not row["exact"]
            }
        ),
        "budgeted_random_feasible_failure_trial_count": sum(
            row["method"] == "budgeted_random_feasible" and not row["exact"]
            for row in nonoracle
        ),
        "budgeted_multistart_greedy_failure_trial_count": sum(
            row["method"] == "budgeted_multistart_greedy" and not row["exact"]
            for row in nonoracle
        ),
        "constraint_preserving_annealing_failure_trial_count": sum(
            row["method"] == "constraint_preserving_annealing" and not row["exact"]
            for row in nonoracle
        ),
    }
    if result["hardness_summary"] != hardness:
        raise ValueError("Stage73 audit hardness summary differs")
    gate = config["benchmark_gate"]
    tractable = (
        maximum_total <= int(gate["maximum_current_exact_enumeration_state_count"])
        and maximum_full_frozen <= int(gate["maximum_full_pool_frozen_feasible_subset_count"])
    )
    solver_gate = (
        full_frozen["budgeted_multistart_greedy"]["exact_optimum_success_rate"]
        >= float(gate["minimum_full_frozen_multistart_success_rate"])
        and full_frozen["constraint_preserving_annealing"]["exact_optimum_success_rate"]
        >= float(gate["minimum_full_frozen_annealing_success_rate"])
    )
    expected_route = {
        "current_k3_exact_enumeration_tractable": bool(tractable),
        "constraint_native_solver_gate_passed": bool(solver_gate),
    }
    if result["route_gate"] != expected_route:
        raise ValueError("Stage73 audit route gate differs")
    expected_decision = {
        "larger_k_scaling_study_authorized": bool(tractable and solver_gate),
        "direct_qpu_execution_authorized": False,
        "quantum_scaling_claim_authorized": False,
        "quantum_advantage_claim_authorized": False,
    }
    for field, expected in expected_decision.items():
        if bool(result["decision"][field]) != expected:
            raise ValueError(f"Stage73 audit decision differs: {field}")
    payload = {
        "scaling_summary": scaling,
        "full_pool_frozen_performance": full_frozen,
        "hardness_summary": hardness,
        "current_k3_exact_enumeration_tractable": bool(tractable),
        "constraint_native_solver_gate_passed": bool(solver_gate),
    }
    if canonical_sha256(payload) != result["analysis_payload_sha256"]:
        raise ValueError("Stage73 audit analysis payload hash differs")
    if result["data_boundary"] != {
        "historical_development_targets_read": 4,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "cloud_cqm_jobs": 0,
        "quantum_hardware_jobs": 0,
    }:
        raise ValueError("Stage73 audit data boundary differs")
    audit = {
        "schema_version": "1.0",
        "status": "stage73_constraint_native_solver_scaling_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "stage72_models_independently_rebuilt": len(models),
        "workload_cells_independently_enumerated": cell_index,
        "solver_trials_deterministically_replayed": len(expected_trial_records),
        "solver_summaries_independently_recomputed": len(summary_observed),
        "current_k3_exact_enumeration_tractable": bool(tractable),
        "constraint_native_solver_gate_passed": bool(solver_gate),
        "larger_k_scaling_study_authorized": bool(tractable and solver_gate),
        "direct_qpu_execution_authorized": False,
        "quantum_scaling_claim_authorized": False,
        "quantum_advantage_claim_authorized": False,
        "data_boundary": result["data_boundary"],
    }
    write_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage73_constraint_native_solver_scaling.json"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("data/stage73_constraint_native_solver_scaling_result.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage73_constraint_native_solver_scaling_audit.json"),
    )
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
