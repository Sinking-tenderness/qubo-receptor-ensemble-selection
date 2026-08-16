"""Benchmark classical solver routes for the frozen Stage72 constrained objective."""

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
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any], label: str) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage73 frozen {label} identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage73 frozen {label} size differs: {path}")
    return path


def quality_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def load_model(record: dict[str, Any]) -> dict[str, Any]:
    receptor_ids = [str(value) for value in record["receptor_ids"]]
    count = len(receptor_ids)
    pairs = list(itertools.combinations(range(count), 2))
    centered = np.asarray(record["centered_pair_coefficients"], dtype=float)
    if len(centered) != len(pairs):
        raise ValueError("Stage73 centered pair coefficient count differs")
    center = float(record["pair_midpoint_center"])
    raw = centered + center
    redundancy = np.zeros((count, count), dtype=float)
    for (left, right), value in zip(pairs, raw):
        redundancy[left, right] = value
        redundancy[right, left] = value
    deficits = np.asarray(record["integer_deficits"], dtype=int)
    subset_size = int(record["reference_k"])
    maximum = int(record["maximum_integer_deficit"])
    canonical = {
        "variable_order": [f"x{index:03d}" for index in range(count)],
        "objective": {
            "pair_center": center,
            "quadratic_pair_order": [list(pair) for pair in pairs],
            "quadratic_coefficients": [float(value) for value in centered],
            "offset": float(record["objective_offset"]),
        },
        "constraints": {
            "cardinality_exact": {
                "sense": "==",
                "linear_coefficients": [1] * count,
                "rhs": subset_size,
            },
            "quality_floor": {
                "sense": "<=",
                "linear_coefficients": [int(value) for value in deficits],
                "rhs": maximum,
            },
        },
    }
    if canonical_sha256(canonical) != str(record["cqm_sha256"]).upper():
        raise ValueError("Stage73 source Stage72 CQM hash differs")
    order = sorted(
        range(count),
        key=lambda index: (
            int(deficits[index]),
            quality_hash(receptor_ids[index]),
            receptor_ids[index],
        ),
    )
    return {
        "record": record,
        "receptor_ids": receptor_ids,
        "count": count,
        "deficits": deficits,
        "subset_size": subset_size,
        "frozen_maximum": maximum,
        "redundancy": redundancy,
        "quality_order": order,
        "source_cqm_sha256": record["cqm_sha256"],
    }


def pool_sizes(model: dict[str, Any], config: dict[str, Any]) -> list[int]:
    values = [
        int(value)
        for value in config["scaling_workloads"]["candidate_pool_sizes"]
        if int(value) <= int(model["count"])
    ]
    if config["scaling_workloads"]["always_include_full_pool"]:
        values.append(int(model["count"]))
    output = sorted(set(values))
    if not output or output[0] < int(model["subset_size"]):
        raise ValueError("Stage73 pool-size schedule is invalid")
    return output


def subset_objective(model: dict[str, Any], subset: tuple[int, ...]) -> float:
    return float(
        sum(
            model["redundancy"][left, right]
            for left, right in itertools.combinations(subset, 2)
        )
    )


def subset_deficit(model: dict[str, Any], subset: tuple[int, ...]) -> int:
    return int(sum(int(model["deficits"][index]) for index in subset))


def enumerate_pool(
    model: dict[str, Any], size: int, config: dict[str, Any]
) -> dict[str, Any]:
    selected_pool = tuple(sorted(model["quality_order"][:size]))
    subsets = sorted(
        tuple(sorted(value))
        for value in itertools.combinations(selected_pool, model["subset_size"])
    )
    deficits = np.asarray(
        [subset_deficit(model, subset) for subset in subsets], dtype=int
    )
    objectives = np.asarray(
        [subset_objective(model, subset) for subset in subsets], dtype=float
    )
    sorted_deficits = np.sort(deficits)
    quantile = float(config["scaling_workloads"]["relaxed_feasible_quantile"])
    quantile_index = max(0, math.ceil(quantile * len(sorted_deficits)) - 1)
    relaxed = max(
        int(model["frozen_maximum"]), int(sorted_deficits[quantile_index])
    )
    thresholds = {
        "frozen_quality_floor": int(model["frozen_maximum"]),
        "relaxed_10pct_quality_floor": relaxed,
        "no_quality_floor": int(np.max(deficits)),
    }
    pair_values = np.asarray(
        [
            model["redundancy"][left, right]
            for left, right in itertools.combinations(selected_pool, 2)
        ],
        dtype=float,
    )
    midpoint = (float(np.min(pair_values)) + float(np.max(pair_values))) / 2.0
    scale = float(np.max(np.abs(pair_values - midpoint)))
    if scale <= TOLERANCE:
        scale = 1.0
    return {
        "pool_indices": selected_pool,
        "subsets": subsets,
        "deficits": deficits,
        "objectives": objectives,
        "thresholds": thresholds,
        "normalization_scale": scale,
        "total_subset_count": len(subsets),
    }


def make_cell(
    model: dict[str, Any], pool: dict[str, Any], regime: str
) -> dict[str, Any]:
    threshold = int(pool["thresholds"][regime])
    indices = [
        index for index, value in enumerate(pool["deficits"]) if int(value) <= threshold
    ]
    if not indices:
        raise ValueError("Stage73 workload has no feasible state")
    feasible_subsets = [pool["subsets"][index] for index in indices]
    feasible_objectives = np.asarray(
        [pool["objectives"][index] for index in indices], dtype=float
    )
    order = sorted(
        range(len(feasible_subsets)),
        key=lambda index: (feasible_objectives[index], feasible_subsets[index]),
    )
    optimum = float(feasible_objectives[order[0]])
    optimal_indices = [
        index
        for index in order
        if float(feasible_objectives[index]) <= optimum + TOLERANCE
    ]
    canonical_index = min(
        optimal_indices, key=lambda index: feasible_subsets[index]
    )
    return {
        "model": model,
        "pool_indices": pool["pool_indices"],
        "pool_size": len(pool["pool_indices"]),
        "quality_regime": regime,
        "quality_threshold": threshold,
        "total_subset_count": int(pool["total_subset_count"]),
        "feasible_subsets": feasible_subsets,
        "feasible_objectives": feasible_objectives,
        "feasible_lookup": {
            subset: index for index, subset in enumerate(feasible_subsets)
        },
        "feasible_subset_count": len(feasible_subsets),
        "normalization_scale": float(pool["normalization_scale"]),
        "exact_objective": optimum,
        "exact_subset": feasible_subsets[canonical_index],
        "exact_optimum_degeneracy": len(optimal_indices),
        "is_full_pool": len(pool["pool_indices"]) == int(model["count"]),
    }


def subset_name(model: dict[str, Any], subset: tuple[int, ...]) -> str:
    return "+".join(model["receptor_ids"][index] for index in subset)


def solution_metrics(cell: dict[str, Any], subset: tuple[int, ...]) -> dict[str, Any]:
    model = cell["model"]
    objective = subset_objective(model, subset)
    regret = objective - float(cell["exact_objective"])
    exact_set = set(cell["exact_subset"])
    selected_set = set(subset)
    return {
        "solution_subset": subset_name(model, subset),
        "solution_objective": objective,
        "absolute_objective_regret": regret,
        "normalized_objective_regret": regret / float(cell["normalization_scale"]),
        "exact_optimum_match": regret <= TOLERANCE,
        "canonical_optimum_subset_match": subset == cell["exact_subset"],
        "canonical_optimum_jaccard": len(exact_set & selected_set)
        / len(exact_set | selected_set),
    }


def candidate_swaps(
    cell: dict[str, Any], subset: tuple[int, ...]
) -> list[tuple[int, ...]]:
    selected = set(subset)
    output = {
        tuple(sorted((selected - {outgoing}) | {incoming}))
        for outgoing in subset
        for incoming in cell["pool_indices"]
        if incoming not in selected
    }
    return sorted(output)


def single_start_best_improvement(cell: dict[str, Any]) -> dict[str, Any]:
    model = cell["model"]
    current = min(
        cell["feasible_subsets"],
        key=lambda subset: (subset_deficit(model, subset), subset),
    )
    current_objective = subset_objective(model, current)
    proposals = 0
    feasible_proposals = 0
    evaluations = 1
    accepted = 0
    while True:
        best_subset = current
        best_objective = current_objective
        for candidate in candidate_swaps(cell, current):
            proposals += 1
            if subset_deficit(model, candidate) > int(cell["quality_threshold"]):
                continue
            feasible_proposals += 1
            evaluations += 1
            value = subset_objective(model, candidate)
            if (value, candidate) < (best_objective - TOLERANCE, best_subset):
                best_subset = candidate
                best_objective = value
        if best_objective >= current_objective - TOLERANCE:
            break
        current = best_subset
        current_objective = best_objective
        accepted += 1
    return {
        "subset": current,
        "proposal_count": proposals,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "restart_count": 1,
    }


def random_feasible_sampling(
    cell: dict[str, Any], budget: int, rng: np.random.Generator
) -> dict[str, Any]:
    indices = rng.integers(0, len(cell["feasible_subsets"]), size=budget)
    best_index = min(
        (int(index) for index in indices),
        key=lambda index: (
            float(cell["feasible_objectives"][index]),
            cell["feasible_subsets"][index],
        ),
    )
    return {
        "subset": cell["feasible_subsets"][best_index],
        "proposal_count": budget,
        "feasible_proposal_count": budget,
        "objective_evaluation_count": budget,
        "accepted_move_count": 0,
        "restart_count": budget,
    }


def budgeted_multistart_greedy(
    cell: dict[str, Any], budget: int, rng: np.random.Generator
) -> dict[str, Any]:
    model = cell["model"]
    proposals = 0
    feasible_proposals = 0
    evaluations = 0
    accepted = 0
    restarts = 0
    best_subset: tuple[int, ...] | None = None
    best_objective = math.inf
    while proposals < budget:
        restarts += 1
        current = cell["feasible_subsets"][
            int(rng.integers(0, len(cell["feasible_subsets"])))
        ]
        current_objective = subset_objective(model, current)
        evaluations += 1
        if (current_objective, current) < (best_objective, best_subset or current):
            best_subset = current
            best_objective = current_objective
        while proposals < budget:
            candidates = candidate_swaps(cell, current)
            order = rng.permutation(len(candidates))
            local_best = current
            local_objective = current_objective
            for position in order:
                if proposals >= budget:
                    break
                candidate = candidates[int(position)]
                proposals += 1
                if subset_deficit(model, candidate) > int(cell["quality_threshold"]):
                    continue
                feasible_proposals += 1
                evaluations += 1
                value = subset_objective(model, candidate)
                if (value, candidate) < (
                    local_objective - TOLERANCE,
                    local_best,
                ):
                    local_best = candidate
                    local_objective = value
                if (value, candidate) < (best_objective, best_subset or candidate):
                    best_subset = candidate
                    best_objective = value
            if local_objective >= current_objective - TOLERANCE:
                break
            current = local_best
            current_objective = local_objective
            accepted += 1
    if best_subset is None:
        raise ValueError("Stage73 multistart greedy produced no state")
    return {
        "subset": best_subset,
        "proposal_count": proposals,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "restart_count": restarts,
    }


def constraint_preserving_annealing(
    cell: dict[str, Any],
    budget: int,
    beta_minimum: float,
    beta_maximum: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    model = cell["model"]
    current = cell["feasible_subsets"][
        int(rng.integers(0, len(cell["feasible_subsets"])))
    ]
    current_objective = subset_objective(model, current)
    best_subset = current
    best_objective = current_objective
    feasible_proposals = 0
    evaluations = 1
    accepted = 0
    pool = tuple(cell["pool_indices"])
    for step in range(budget):
        selected = set(current)
        outgoing = current[int(rng.integers(0, len(current)))]
        available = [value for value in pool if value not in selected]
        incoming = available[int(rng.integers(0, len(available)))]
        candidate = tuple(sorted((selected - {outgoing}) | {incoming}))
        if subset_deficit(model, candidate) > int(cell["quality_threshold"]):
            continue
        feasible_proposals += 1
        evaluations += 1
        candidate_objective = subset_objective(model, candidate)
        normalized_delta = (
            candidate_objective - current_objective
        ) / float(cell["normalization_scale"])
        fraction = step / max(1, budget - 1)
        beta = beta_minimum * (beta_maximum / beta_minimum) ** fraction
        if normalized_delta <= 0 or rng.random() < math.exp(-beta * normalized_delta):
            current = candidate
            current_objective = candidate_objective
            accepted += 1
            if (current_objective, current) < (best_objective, best_subset):
                best_subset = current
                best_objective = current_objective
    return {
        "subset": best_subset,
        "proposal_count": budget,
        "feasible_proposal_count": feasible_proposals,
        "objective_evaluation_count": evaluations,
        "accepted_move_count": accepted,
        "restart_count": 1,
    }


def trial_seed(base: int, cell_index: int, method_index: int, repeat: int) -> int:
    return int(base + cell_index * 100_000 + method_index * 1_000 + repeat)


def workload_row(cell: dict[str, Any]) -> dict[str, Any]:
    model = cell["model"]
    return {
        "target_id": model["record"]["target_id"],
        "outer_fold": int(model["record"]["outer_fold"]),
        "pool_size": int(cell["pool_size"]),
        "is_full_pool": bool(cell["is_full_pool"]),
        "quality_regime": cell["quality_regime"],
        "quality_threshold": int(cell["quality_threshold"]),
        "total_fixed_k_subset_count": int(cell["total_subset_count"]),
        "feasible_subset_count": int(cell["feasible_subset_count"]),
        "feasible_subset_fraction": float(
            cell["feasible_subset_count"] / cell["total_subset_count"]
        ),
        "exact_optimum_subset": subset_name(model, cell["exact_subset"]),
        "exact_optimum_objective": float(cell["exact_objective"]),
        "exact_optimum_degeneracy": int(cell["exact_optimum_degeneracy"]),
        "objective_normalization_scale": float(cell["normalization_scale"]),
        "pool_receptor_ids_sha256": canonical_sha256(
            [model["receptor_ids"][index] for index in cell["pool_indices"]]
        ),
    }


def solver_trial_row(
    cell: dict[str, Any],
    method: str,
    repeat: int,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    budget = int(config["solver_protocol"]["proposal_budget"])
    if method == "exact_enumeration":
        solved = {
            "subset": cell["exact_subset"],
            "proposal_count": int(cell["total_subset_count"]),
            "feasible_proposal_count": int(cell["feasible_subset_count"]),
            "objective_evaluation_count": int(cell["feasible_subset_count"]),
            "accepted_move_count": 0,
            "restart_count": 1,
        }
        effective_budget = int(cell["total_subset_count"])
    elif method == "single_start_best_improvement":
        solved = single_start_best_improvement(cell)
        effective_budget = int(solved["proposal_count"])
    else:
        rng = np.random.default_rng(seed)
        if method == "budgeted_random_feasible":
            solved = random_feasible_sampling(cell, budget, rng)
        elif method == "budgeted_multistart_greedy":
            solved = budgeted_multistart_greedy(cell, budget, rng)
        elif method == "constraint_preserving_annealing":
            solved = constraint_preserving_annealing(
                cell,
                budget,
                float(config["solver_protocol"]["annealing_beta_range"][0]),
                float(config["solver_protocol"]["annealing_beta_range"][1]),
                rng,
            )
        else:
            raise ValueError(f"unknown Stage73 solver method: {method}")
        effective_budget = budget
    model = cell["model"]
    return {
        "target_id": model["record"]["target_id"],
        "outer_fold": int(model["record"]["outer_fold"]),
        "pool_size": int(cell["pool_size"]),
        "is_full_pool": bool(cell["is_full_pool"]),
        "quality_regime": cell["quality_regime"],
        "quality_threshold": int(cell["quality_threshold"]),
        "total_fixed_k_subset_count": int(cell["total_subset_count"]),
        "feasible_subset_count": int(cell["feasible_subset_count"]),
        "method": method,
        "repeat": repeat,
        "seed": seed,
        "configured_proposal_budget": effective_budget,
        **{key: int(value) for key, value in solved.items() if key != "subset"},
        **solution_metrics(cell, solved["subset"]),
    }


def summarize_trials(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        for scope in ("ALL", str(row["target_id"])):
            key = (
                str(row["method"]),
                str(row["quality_regime"]),
                int(row["pool_size"]),
                scope,
            )
            groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (method, regime, size, scope), selected in sorted(groups.items()):
        output.append(
            {
                "method": method,
                "quality_regime": regime,
                "pool_size": size,
                "scope": scope,
                "trial_count": len(selected),
                "workload_cell_count": len(
                    {
                        (row["target_id"], int(row["outer_fold"]))
                        for row in selected
                    }
                ),
                "exact_optimum_success_rate": statistics.fmean(
                    int(bool(row["exact_optimum_match"])) for row in selected
                ),
                "canonical_subset_success_rate": statistics.fmean(
                    int(bool(row["canonical_optimum_subset_match"]))
                    for row in selected
                ),
                "mean_absolute_objective_regret": statistics.fmean(
                    float(row["absolute_objective_regret"]) for row in selected
                ),
                "maximum_absolute_objective_regret": max(
                    float(row["absolute_objective_regret"]) for row in selected
                ),
                "mean_normalized_objective_regret": statistics.fmean(
                    float(row["normalized_objective_regret"]) for row in selected
                ),
                "mean_canonical_optimum_jaccard": statistics.fmean(
                    float(row["canonical_optimum_jaccard"]) for row in selected
                ),
                "mean_proposal_count": statistics.fmean(
                    int(row["proposal_count"]) for row in selected
                ),
                "mean_objective_evaluation_count": statistics.fmean(
                    int(row["objective_evaluation_count"]) for row in selected
                ),
            }
        )
    return output


def method_performance(
    rows: list[dict[str, Any]], method: str, selector: Any
) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method and selector(row)]
    if not selected:
        raise ValueError(f"Stage73 performance selection is empty: {method}")
    return {
        "trial_count": len(selected),
        "workload_cell_count": len(
            {
                (
                    row["target_id"],
                    int(row["outer_fold"]),
                    int(row["pool_size"]),
                    row["quality_regime"],
                )
                for row in selected
            }
        ),
        "exact_optimum_success_rate": statistics.fmean(
            int(bool(row["exact_optimum_match"])) for row in selected
        ),
        "mean_normalized_objective_regret": statistics.fmean(
            float(row["normalized_objective_regret"]) for row in selected
        ),
        "maximum_normalized_objective_regret": max(
            float(row["normalized_objective_regret"]) for row in selected
        ),
    }


def report_text(result: dict[str, Any]) -> str:
    scale = result["scaling_summary"]
    frozen = result["full_pool_frozen_performance"]
    hardness = result["hardness_summary"]
    return rf"""# Stage73 constraint-native solver scaling

## Question

Do the current $k=3$ constraint-native receptor-selection instances require a global quantum or hybrid solver, and when do budget-matched classical methods begin to fail as the pool and feasible region grow?

## Protocol

Nested pools are ordered only by integer quality deficit and a receptor-ID hash. Each pool is tested under the frozen quality floor, a deterministic 10% feasible-density floor, and no quality floor. Exact enumeration supplies the oracle. Single-start greedy, budgeted random feasible sampling, budgeted multistart greedy, and constraint-preserving simulated annealing are compared using deterministic work-unit accounting; cloud CQM and quantum hardware are not executed.

## Scale

- Workload cells: `{scale['workload_cell_count']}`.
- Largest candidate pool: `{scale['maximum_pool_size']}`.
- Largest fixed-$k$ search space: `{scale['maximum_total_fixed_k_subset_count']}`.
- Largest full-pool frozen feasible set: `{scale['maximum_full_pool_frozen_feasible_subset_count']}`.
- Current exact-enumeration tractability gate: `{result['route_gate']['current_k3_exact_enumeration_tractable']}`.

## Full-pool frozen task

- Multistart greedy success: `{frozen['budgeted_multistart_greedy']['exact_optimum_success_rate']:.3f}`.
- Constraint-preserving annealing success: `{frozen['constraint_preserving_annealing']['exact_optimum_success_rate']:.3f}`.
- Random feasible success: `{frozen['budgeted_random_feasible']['exact_optimum_success_rate']:.3f}`.

## Hardness

- Single-start greedy failed workload cells: `{hardness['single_start_greedy_failure_cell_count']}`.
- Budgeted multistart greedy failed trials: `{hardness['budgeted_multistart_greedy_failure_trial_count']}`.
- Constraint-preserving annealing failed trials: `{hardness['constraint_preserving_annealing_failure_trial_count']}`.

## Decision

- Larger-$k$ scaling study authorized: `{result['decision']['larger_k_scaling_study_authorized']}`.
- Direct QPU execution authorized: `{result['decision']['direct_qpu_execution_authorized']}`.
- Quantum scaling/advantage claim authorized: `{result['decision']['quantum_scaling_claim_authorized']}` / `{result['decision']['quantum_advantage_claim_authorized']}`.

The current $k=3$ instances are a logical formulation proof, not yet a quantum-scale workload. Stage73 uses oracle-feasible initialization for the budgeted stochastic methods and reports operation counts rather than claiming end-to-end solver speedup.
"""


def compute(config: dict[str, Any], root: Path) -> dict[str, Any]:
    implementation_paths = {
        key: verified(root, value, key)
        for key, value in config["implementation"].items()
    }
    input_paths = {
        key: verified(root, value, key) for key, value in config["inputs"].items()
    }
    stage72_result = read_json(input_paths["stage72_result"])
    stage72_audit = read_json(input_paths["stage72_audit"])
    if not stage72_result["decision"]["solver_scaling_benchmark_authorized"]:
        raise ValueError("Stage73 requires Stage72 solver-scaling authorization")
    if stage72_audit.get("status") != (
        "stage72_constraint_native_cqm_independent_audit_ok"
    ):
        raise ValueError("Stage73 requires the Stage72 independent audit")
    source_record = read_json(input_paths["stage72_model_record"])
    if int(source_record["model_count"]) != int(
        config["experiment"]["required_model_count"]
    ):
        raise ValueError("Stage73 source model count differs")
    models = [load_model(record) for record in source_record["models"]]
    regimes = [str(value) for value in config["scaling_workloads"]["quality_regimes"]]
    workload_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    cell_index = 0
    method_order = [str(value) for value in config["solver_protocol"]["method_order"]]
    stochastic = set(config["solver_protocol"]["stochastic_methods"])
    repeats = int(config["solver_protocol"]["stochastic_repeats"])
    seed_base = int(config["solver_protocol"]["seed_base"])
    for model in models:
        for size in pool_sizes(model, config):
            pool = enumerate_pool(model, size, config)
            for regime in regimes:
                cell = make_cell(model, pool, regime)
                workload_rows.append(workload_row(cell))
                for method_index, method in enumerate(method_order):
                    method_repeats = repeats if method in stochastic else 1
                    for repeat in range(method_repeats):
                        seed = trial_seed(
                            seed_base, cell_index, method_index, repeat
                        )
                        solver_rows.append(
                            solver_trial_row(
                                cell, method, repeat, seed, config
                            )
                        )
                cell_index += 1
        print(
            json.dumps(
                {
                    "target_id": model["record"]["target_id"],
                    "outer_fold": model["record"]["outer_fold"],
                    "workload_cells_completed": len(workload_rows),
                    "solver_trials_completed": len(solver_rows),
                }
            )
        )
    summaries = summarize_trials(solver_rows)
    expected_cells = int(config["benchmark_gate"]["required_workload_cell_count"])
    if len(workload_rows) != expected_cells:
        raise ValueError(
            f"Stage73 workload count differs: {len(workload_rows)} != {expected_cells}"
        )
    scaling_summary = {
        "model_count": len(models),
        "workload_cell_count": len(workload_rows),
        "solver_trial_count": len(solver_rows),
        "maximum_pool_size": max(int(row["pool_size"]) for row in workload_rows),
        "maximum_total_fixed_k_subset_count": max(
            int(row["total_fixed_k_subset_count"]) for row in workload_rows
        ),
        "maximum_feasible_subset_count": max(
            int(row["feasible_subset_count"]) for row in workload_rows
        ),
        "maximum_full_pool_frozen_feasible_subset_count": max(
            int(row["feasible_subset_count"])
            for row in workload_rows
            if bool(row["is_full_pool"])
            and row["quality_regime"] == "frozen_quality_floor"
        ),
        "total_exact_enumeration_state_checks": sum(
            int(row["total_fixed_k_subset_count"]) for row in workload_rows
        ),
    }
    full_frozen_selector = lambda row: bool(row["is_full_pool"]) and row[
        "quality_regime"
    ] == "frozen_quality_floor"
    full_pool_frozen = {
        method: method_performance(solver_rows, method, full_frozen_selector)
        for method in method_order
    }
    nonoracle = [
        row for row in solver_rows if row["method"] != "exact_enumeration"
    ]
    single_failures = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            int(row["pool_size"]),
            row["quality_regime"],
        )
        for row in nonoracle
        if row["method"] == "single_start_best_improvement"
        and not bool(row["exact_optimum_match"])
    }
    hardness = {
        "single_start_greedy_failure_cell_count": len(single_failures),
        "budgeted_random_feasible_failure_trial_count": sum(
            row["method"] == "budgeted_random_feasible"
            and not bool(row["exact_optimum_match"])
            for row in nonoracle
        ),
        "budgeted_multistart_greedy_failure_trial_count": sum(
            row["method"] == "budgeted_multistart_greedy"
            and not bool(row["exact_optimum_match"])
            for row in nonoracle
        ),
        "constraint_preserving_annealing_failure_trial_count": sum(
            row["method"] == "constraint_preserving_annealing"
            and not bool(row["exact_optimum_match"])
            for row in nonoracle
        ),
    }
    gate = config["benchmark_gate"]
    tractable = bool(
        scaling_summary["maximum_total_fixed_k_subset_count"]
        <= int(gate["maximum_current_exact_enumeration_state_count"])
        and scaling_summary["maximum_full_pool_frozen_feasible_subset_count"]
        <= int(gate["maximum_full_pool_frozen_feasible_subset_count"])
    )
    solver_gate_passed = bool(
        full_pool_frozen["budgeted_multistart_greedy"][
            "exact_optimum_success_rate"
        ]
        >= float(gate["minimum_full_frozen_multistart_success_rate"])
        and full_pool_frozen["constraint_preserving_annealing"][
            "exact_optimum_success_rate"
        ]
        >= float(gate["minimum_full_frozen_annealing_success_rate"])
    )
    payload = {
        "scaling_summary": scaling_summary,
        "full_pool_frozen_performance": full_pool_frozen,
        "hardness_summary": hardness,
        "current_k3_exact_enumeration_tractable": tractable,
        "constraint_native_solver_gate_passed": solver_gate_passed,
    }
    output_paths = {
        key: root / str(value) for key, value in config["outputs"].items()
    }
    write_csv(output_paths["workload_metrics_csv"], workload_rows)
    write_csv(output_paths["solver_trials_csv"], solver_rows)
    write_csv(output_paths["solver_summary_csv"], summaries)
    result = {
        "schema_version": "1.0",
        "status": "stage73_constraint_native_solver_scaling_complete",
        "experiment_class": (
            "post-hoc deterministic-work-unit solver scaling on frozen historical "
            "constraint-native objectives"
        ),
        "config": descriptor(
            root, root / "configs/stage73_constraint_native_solver_scaling.json"
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
        "scaling_summary": scaling_summary,
        "full_pool_frozen_performance": full_pool_frozen,
        "hardness_summary": hardness,
        "route_gate": {
            "current_k3_exact_enumeration_tractable": tractable,
            "constraint_native_solver_gate_passed": solver_gate_passed,
        },
        "decision": {
            "larger_k_scaling_study_authorized": bool(
                tractable and solver_gate_passed
            ),
            "direct_qpu_execution_authorized": False,
            "quantum_scaling_claim_authorized": False,
            "quantum_advantage_claim_authorized": False,
            "new_target_preregistration_remains_authorized": stage72_result[
                "decision"
            ]["new_target_preregistration_remains_authorized"],
            "next_action": (
                "construct a preregistered larger-k and variable-k constraint-native scaling benchmark before considering cloud or hardware execution"
                if tractable and solver_gate_passed
                else "repair the local constraint-native solver protocol before increasing k"
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
    expected = root / "configs/stage73_constraint_native_solver_scaling.json"
    if config_path != expected.resolve():
        raise ValueError("Stage73 must run from its frozen repository config")
    config = read_json(config_path)
    result_path = root / str(config["outputs"]["result_json"])
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Stage73 result exists: {result_path}")
    result = compute(config, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage73_constraint_native_solver_scaling.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
