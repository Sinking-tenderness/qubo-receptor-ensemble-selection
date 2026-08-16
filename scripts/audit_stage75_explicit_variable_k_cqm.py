"""Independently audit the Stage75 explicit variable-k CQM benchmark."""

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
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import dimod
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
        raise ValueError(f"Stage75 audit {label} identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage75 audit {label} size differs: {path}")
    return path


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"cannot parse boolean: {value!r}")


def close(observed: Any, expected: Any, label: str) -> None:
    if not math.isclose(
        float(observed), float(expected), rel_tol=TOLERANCE, abs_tol=TOLERANCE
    ):
        raise ValueError(f"Stage75 audit {label} differs: {observed} != {expected}")


def rebuild(record: dict[str, Any]) -> dict[str, Any]:
    ids = tuple(str(value) for value in record["receptor_ids"])
    count = len(ids)
    pairs = tuple(itertools.combinations(range(count), 2))
    centered = tuple(float(value) for value in record["centered_pair_coefficients"])
    midpoint = float(record["pair_midpoint_center"])
    raw = tuple(value + midpoint for value in centered)
    matrix = np.zeros((count, count), dtype=float)
    for (left, right), value in zip(pairs, raw):
        matrix[left, right] = value
        matrix[right, left] = value
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
        raise ValueError("Stage75 audit failed to rebuild Stage72 CQM")
    return {
        "record": record,
        "ids": ids,
        "index": {value: index for index, value in enumerate(ids)},
        "count": count,
        "pairs": pairs,
        "raw": raw,
        "matrix": matrix,
        "deficits": deficits,
        "scale": max(max(abs(value) for value in raw), 1e-12),
    }


def parse_subset(model: dict[str, Any], label: str) -> tuple[int, ...]:
    return tuple(sorted(model["index"][value] for value in label.split("+")))


def label(model: dict[str, Any], subset: tuple[int, ...]) -> str:
    return "+".join(model["ids"][index] for index in subset)


def deficit(model: dict[str, Any], subset: tuple[int, ...]) -> int:
    return sum(model["deficits"][index] for index in subset)


def raw_objective(model: dict[str, Any], subset: tuple[int, ...]) -> float:
    return float(
        sum(
            model["matrix"][left, right]
            for left, right in itertools.combinations(subset, 2)
        )
    )


def energy(model: dict[str, Any], subset: tuple[int, ...], reward: float) -> float:
    return raw_objective(model, subset) - reward * math.comb(len(subset), 2)


def recover_frontiers(
    model: dict[str, Any],
    workloads: list[dict[str, str]],
    comparisons: list[dict[str, str]],
    trials: list[dict[str, str]],
    regime: str,
) -> dict[int, dict[str, Any]]:
    target = str(model["record"]["target_id"])
    fold = int(model["record"]["outer_fold"])
    output: dict[int, dict[str, Any]] = {}
    for workload in workloads:
        if (
            workload["target_id"] != target
            or int(workload["outer_fold"]) != fold
            or workload["quality_regime"] != regime
        ):
            continue
        k = int(workload["k"])
        comparison = next(
            row
            for row in comparisons
            if row["target_id"] == target
            and int(row["outer_fold"]) == fold
            and int(row["k"]) == k
            and row["quality_regime"] == regime
        )
        local = [
            row
            for row in trials
            if row["target_id"] == target
            and int(row["outer_fold"]) == fold
            and int(row["k"]) == k
            and row["quality_regime"] == regime
        ]
        candidates = [row for row in local if truth(row["reference_match"])]
        reference = min(
            candidates,
            key=lambda row: (
                0 if row["method"] == "exact_enumeration" else 1,
                row["solution_subset"],
            ),
        )
        deterministic = next(
            row
            for row in local
            if row["method"] == "deterministic_best_improvement"
            and int(row["repeat"]) == 0
        )
        reference_subset = parse_subset(model, reference["solution_subset"])
        deterministic_subset = parse_subset(model, deterministic["solution_subset"])
        close(
            raw_objective(model, reference_subset),
            comparison["reference_objective"],
            "source frontier objective",
        )
        output[k] = {
            "threshold": int(workload["quality_threshold"]),
            "reference_type": str(comparison["reference_type"]),
            "reference_subset": reference_subset,
            "deterministic_subset": deterministic_subset,
        }
    return dict(sorted(output.items()))


def reward_record(model: dict[str, Any], quantile: float) -> tuple[int, float]:
    ordered = sorted(model["raw"])
    position = max(0, math.ceil(quantile * len(ordered)) - 1)
    return position, float(ordered[position])


def canonical_model(
    model: dict[str, Any], frontiers: dict[int, dict[str, Any]], reward: float
) -> dict[str, Any]:
    x_names = [f"x{index:03d}" for index in range(model["count"])]
    y_names = [f"y_k{k:02d}" for k in frontiers]
    return {
        "variable_order": x_names + y_names,
        "objective": {
            "linear_coefficients": [0.0] * (len(x_names) + len(y_names)),
            "quadratic_pair_order": [list(pair) for pair in model["pairs"]],
            "quadratic_coefficients": [value - reward for value in model["raw"]],
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
                "y_coefficients": {f"y_k{k:02d}": -k for k in frontiers},
                "rhs": 0,
            },
            "conditional_quality": {
                "sense": "<=",
                "x_coefficients": list(model["deficits"]),
                "y_coefficients": {
                    f"y_k{k:02d}": -int(value["threshold"])
                    for k, value in frontiers.items()
                },
                "rhs": 0,
            },
        },
    }


def make_cqm(
    model: dict[str, Any], frontiers: dict[int, dict[str, Any]], reward: float
) -> dimod.ConstrainedQuadraticModel:
    cqm = dimod.ConstrainedQuadraticModel()
    x = [dimod.Binary(f"x{index:03d}") for index in range(model["count"])]
    y = {k: dimod.Binary(f"y_k{k:02d}") for k in frontiers}
    cqm.set_objective(
        dimod.quicksum(
            (value - reward) * x[left] * x[right]
            for (left, right), value in zip(model["pairs"], model["raw"])
        )
    )
    cqm.add_constraint(dimod.quicksum(y.values()) == 1, label="one_hot_cardinality")
    cqm.add_constraint(
        dimod.quicksum(x) - dimod.quicksum(k * y[k] for k in frontiers) == 0,
        label="cardinality_link",
    )
    cqm.add_constraint(
        dimod.quicksum(
            model["deficits"][index] * x[index] for index in range(model["count"])
        )
        - dimod.quicksum(frontiers[k]["threshold"] * y[k] for k in frontiers)
        <= 0,
        label="conditional_quality",
    )
    return cqm


def sample_for(
    model: dict[str, Any], frontiers: dict[int, dict[str, Any]], subset: tuple[int, ...]
) -> dict[str, int]:
    selected = set(subset)
    output = {
        f"x{index:03d}": int(index in selected) for index in range(model["count"])
    }
    output.update({f"y_k{k:02d}": int(k == len(subset)) for k in frontiers})
    return output


def valid(cell: dict[str, Any], subset: tuple[int, ...]) -> bool:
    k = len(subset)
    return (
        k in cell["frontiers"]
        and len(set(subset)) == k
        and deficit(cell["model"], subset) <= cell["frontiers"][k]["threshold"]
    )


def random_start(
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
        key=lambda subset: (deficit(cell["model"], subset), subset),
    )
    return fallback, maximum_attempts, True


def proposal(
    cell: dict[str, Any], current: tuple[int, ...], rng: np.random.Generator
) -> tuple[int, ...]:
    if rng.random() < float(cell["protocol"]["swap_move_probability"]):
        outgoing = current[int(rng.integers(0, len(current)))]
        selected = set(current)
        incoming = int(rng.integers(0, cell["model"]["count"]))
        while incoming in selected:
            incoming = int(rng.integers(0, cell["model"]["count"]))
        return tuple(sorted((selected - {outgoing}) | {incoming}))
    choices = [value for value in cell["frontiers"] if value != len(current)]
    k = choices[int(rng.integers(0, len(choices)))]
    return tuple(
        sorted(
            int(value)
            for value in rng.choice(cell["model"]["count"], k, replace=False)
        )
    )


def replay_tabu(cell: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    protocol = cell["protocol"]
    budget = int(protocol["proposal_budget"])
    batch = int(protocol["tabu_candidate_batch_size"])
    tenure = int(protocol["tabu_tenure"])
    current, attempts, fallback = random_start(
        cell, rng, int(protocol["maximum_initialization_attempts"])
    )
    current_energy = energy(cell["model"], current, cell["reward"])
    best, best_energy = current, current_energy
    proposals = feasible_proposals = 0
    evaluations = 1
    accepted = 0
    restarts = 1
    queue: deque[tuple[int, ...]] = deque(maxlen=tenure)
    tabu: set[tuple[int, ...]] = set()
    while proposals < budget:
        candidates: list[tuple[float, tuple[int, ...]]] = []
        for _ in range(min(batch, budget - proposals)):
            candidate = proposal(cell, current, rng)
            proposals += 1
            if not valid(cell, candidate):
                continue
            feasible_proposals += 1
            evaluations += 1
            value = energy(cell["model"], candidate, cell["reward"])
            if candidate not in tabu or value < best_energy - 1e-12:
                candidates.append((value, candidate))
        if not candidates:
            current, extra, used_fallback = random_start(
                cell, rng, int(protocol["maximum_initialization_attempts"])
            )
            attempts += extra
            fallback = fallback or used_fallback
            current_energy = energy(cell["model"], current, cell["reward"])
            evaluations += 1
            restarts += 1
            continue
        value, candidate = min(candidates)
        if len(queue) == queue.maxlen:
            tabu.discard(queue.popleft())
        queue.append(current)
        tabu.add(current)
        current, current_energy = candidate, value
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


def replay_annealing(cell: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    protocol = cell["protocol"]
    budget = int(protocol["proposal_budget"])
    beta_minimum, beta_maximum = [float(value) for value in protocol["annealing_beta_range"]]
    current, attempts, fallback = random_start(
        cell, rng, int(protocol["maximum_initialization_attempts"])
    )
    current_energy = energy(cell["model"], current, cell["reward"])
    best, best_energy = current, current_energy
    feasible_proposals = 0
    evaluations = 1
    accepted = 0
    denominator = cell["model"]["scale"] * math.comb(max(cell["frontiers"]), 2)
    for step in range(budget):
        candidate = proposal(cell, current, rng)
        if not valid(cell, candidate):
            continue
        feasible_proposals += 1
        evaluations += 1
        value = energy(cell["model"], candidate, cell["reward"])
        normalized = (value - current_energy) / denominator
        fraction = step / max(1, budget - 1)
        beta = beta_minimum * (beta_maximum / beta_minimum) ** fraction
        if normalized <= 0 or rng.random() < math.exp(-beta * normalized):
            current, current_energy = candidate, value
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


def fixed_best(
    cell: dict[str, Any], field: str
) -> tuple[tuple[int, ...], float]:
    candidates = []
    for record in cell["frontiers"].values():
        subset = tuple(record[field])
        candidates.append((energy(cell["model"], subset, cell["reward"]), subset))
    value, subset = min(candidates, key=lambda item: (item[0], item[1]))
    return subset, value


def expected_solution(
    cell: dict[str, Any], method: str, seed: int
) -> dict[str, Any]:
    if method == "fixed_k_frontier_reference":
        subset, _ = fixed_best(cell, "reference_subset")
        return {
            "subset": subset,
            "proposal_count": 0,
            "feasible_proposal_count": 0,
            "objective_evaluation_count": len(cell["frontiers"]),
            "accepted_move_count": 0,
            "restart_count": len(cell["frontiers"]),
            "initialization_attempt_count": 0,
            "initialization_fallback_count": 0,
        }
    if method == "decomposed_deterministic_baseline":
        subset, _ = fixed_best(cell, "deterministic_subset")
        return {
            "subset": subset,
            "proposal_count": 0,
            "feasible_proposal_count": 0,
            "objective_evaluation_count": len(cell["frontiers"]),
            "accepted_move_count": 0,
            "restart_count": len(cell["frontiers"]),
            "initialization_attempt_count": 0,
            "initialization_fallback_count": 0,
        }
    rng = np.random.default_rng(seed)
    if method == "budgeted_variable_tabu":
        return replay_tabu(cell, rng)
    if method == "constraint_native_variable_annealing":
        return replay_annealing(cell, rng)
    raise ValueError(f"Stage75 audit unknown method: {method}")


def compare_trial(
    observed: dict[str, str],
    cell: dict[str, Any],
    method: str,
    repeat: int,
    seed: int,
    solved: dict[str, Any],
) -> dict[str, Any]:
    if observed["method"] != method or int(observed["repeat"]) != repeat or int(observed["seed"]) != seed:
        raise ValueError("Stage75 audit trial identity differs")
    subset = tuple(solved["subset"])
    if not valid(cell, subset) or observed["solution_subset"] != label(cell["model"], subset):
        raise ValueError("Stage75 audit trial subset differs")
    for field in (
        "proposal_count",
        "feasible_proposal_count",
        "objective_evaluation_count",
        "accepted_move_count",
        "restart_count",
        "initialization_attempt_count",
        "initialization_fallback_count",
    ):
        if int(observed[field]) != int(solved[field]):
            raise ValueError(f"Stage75 audit trial {field} differs")
    value = energy(cell["model"], subset, cell["reward"])
    close(observed["solution_variable_energy"], value, "trial energy")
    close(observed["solution_raw_pair_objective"], raw_objective(cell["model"], subset), "trial raw objective")
    if int(observed["selected_k"]) != len(subset) or int(observed["solution_deficit"]) != deficit(cell["model"], subset):
        raise ValueError("Stage75 audit trial state metrics differ")
    return {
        "target_id": str(cell["model"]["record"]["target_id"]),
        "outer_fold": int(cell["model"]["record"]["outer_fold"]),
        "reward_quantile": float(cell["quantile"]),
        "method": method,
        "repeat": repeat,
        "subset": subset,
        "solution_subset": label(cell["model"], subset),
        "selected_k": len(subset),
        "energy": value,
        "feasible_proposal_count": int(solved["feasible_proposal_count"]),
        "initialization_fallback_count": int(solved["initialization_fallback_count"]),
    }


def attach_expected(
    trials: list[dict[str, Any]], cells: dict[tuple[str, int, float], dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        grouped[(row["target_id"], row["outer_fold"], row["reward_quantile"])].append(row)
    output: list[dict[str, Any]] = []
    tolerance = float(config["benchmark_gate"]["normalized_energy_tolerance"])
    for key, rows in sorted(grouped.items()):
        cell = cells[key]
        frontier_row = next(
            row for row in rows if row["method"] == "fixed_k_frontier_reference"
        )
        frontier = frontier_row["energy"]
        pooled = min(row["energy"] for row in rows)
        denominator = cell["model"]["scale"] * math.comb(max(cell["frontiers"]), 2)
        for row in rows:
            row["frontier"] = frontier
            row["pooled"] = pooled
            row["delta_frontier"] = (row["energy"] - frontier) / denominator
            row["delta_pooled"] = (row["energy"] - pooled) / denominator
            row["pooled_match"] = row["energy"] <= pooled + 1e-12
        best = {
            method: min(
                (row for row in rows if row["method"] == method),
                key=lambda row: (row["energy"], row["solution_subset"]),
            )
            for method in {row["method"] for row in rows}
        }
        joint = best["budgeted_variable_tabu"]
        sampler = best["constraint_native_variable_annealing"]
        delta_joint = (sampler["energy"] - joint["energy"]) / denominator
        delta_frontier = (sampler["energy"] - frontier) / denominator
        exact = all(
            value["reference_type"] == "exact_enumeration"
            for value in cell["frontiers"].values()
        )
        output.append(
            {
                "target_id": key[0],
                "outer_fold": key[1],
                "reward_quantile": key[2],
                "reward_value": cell["reward"],
                "frontier_reference_type": "exact_fixed_k_frontier" if exact else "pooled_best_known_fixed_k_frontier",
                "frozen_frontier_selected_k": best["fixed_k_frontier_reference"]["selected_k"],
                "frozen_frontier_energy": frontier,
                "decomposed_deterministic_selected_k": best["decomposed_deterministic_baseline"]["selected_k"],
                "decomposed_deterministic_energy": best["decomposed_deterministic_baseline"]["energy"],
                "joint_classical_best_selected_k": joint["selected_k"],
                "joint_classical_best_energy": joint["energy"],
                "sampler_best_selected_k": sampler["selected_k"],
                "sampler_best_energy": sampler["energy"],
                "sampler_delta_vs_joint_classical_normalized": delta_joint,
                "sampler_delta_vs_frozen_frontier_normalized": delta_frontier,
                "sampler_within_joint_classical_tolerance": delta_joint <= tolerance,
                "sampler_within_frozen_frontier_tolerance": delta_frontier <= tolerance,
                "sampler_strict_win_vs_joint_classical": delta_joint < -tolerance,
                "joint_classical_strict_win_vs_sampler": delta_joint > tolerance,
                "frozen_frontier_refined": pooled < frontier - 1e-12,
                "pooled_best_energy": pooled,
                "exact_frontier_available": exact,
                "joint_classical_exact_frontier_match": exact and joint["energy"] <= frontier + 1e-12,
                "sampler_exact_frontier_match": exact and sampler["energy"] <= frontier + 1e-12,
            }
        )
    return output


def compare_comparisons(observed: list[dict[str, str]], expected: list[dict[str, Any]]) -> None:
    mapping = {
        (row["target_id"], int(row["outer_fold"]), float(row["reward_quantile"])): row
        for row in observed
    }
    if len(mapping) != len(observed) or len(mapping) != len(expected):
        raise ValueError("Stage75 audit comparison grid differs")
    text_fields = ("frontier_reference_type",)
    integer_fields = (
        "frozen_frontier_selected_k",
        "decomposed_deterministic_selected_k",
        "joint_classical_best_selected_k",
        "sampler_best_selected_k",
    )
    float_fields = (
        "reward_value",
        "frozen_frontier_energy",
        "decomposed_deterministic_energy",
        "joint_classical_best_energy",
        "sampler_best_energy",
        "sampler_delta_vs_joint_classical_normalized",
        "sampler_delta_vs_frozen_frontier_normalized",
        "pooled_best_energy",
    )
    bool_fields = (
        "sampler_within_joint_classical_tolerance",
        "sampler_within_frozen_frontier_tolerance",
        "sampler_strict_win_vs_joint_classical",
        "joint_classical_strict_win_vs_sampler",
        "frozen_frontier_refined",
        "exact_frontier_available",
        "joint_classical_exact_frontier_match",
        "sampler_exact_frontier_match",
    )
    for item in expected:
        key = (item["target_id"], item["outer_fold"], item["reward_quantile"])
        row = mapping[key]
        for field in text_fields:
            if row[field] != item[field]:
                raise ValueError(f"Stage75 audit comparison {field} differs: {key}")
        for field in integer_fields:
            if int(row[field]) != int(item[field]):
                raise ValueError(f"Stage75 audit comparison {field} differs: {key}")
        for field in float_fields:
            close(row[field], item[field], f"comparison {field} {key}")
        for field in bool_fields:
            if truth(row[field]) != bool(item[field]):
                raise ValueError(f"Stage75 audit comparison {field} differs: {key}")


def expected_summaries(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        for scope in ("ALL", row["target_id"]):
            groups[(row["method"], row["reward_quantile"], scope)].append(row)
    output: list[dict[str, Any]] = []
    for (method, quantile, scope), rows in sorted(groups.items()):
        output.append(
            {
                "method": method,
                "reward_quantile": quantile,
                "scope": scope,
                "trial_count": len(rows),
                "cell_count": len({(row["target_id"], row["outer_fold"]) for row in rows}),
                "pooled_best_match_rate": statistics.fmean(row["pooled_match"] for row in rows),
                "mean_delta_vs_frozen_frontier_normalized": statistics.fmean(row["delta_frontier"] for row in rows),
                "maximum_delta_vs_frozen_frontier_normalized": max(row["delta_frontier"] for row in rows),
                "mean_selected_k": statistics.fmean(row["selected_k"] for row in rows),
                "mean_feasible_proposal_count": statistics.fmean(row["feasible_proposal_count"] for row in rows),
                "initialization_fallback_count": sum(row["initialization_fallback_count"] for row in rows),
            }
        )
    return output


def compare_summaries(observed: list[dict[str, str]], expected: list[dict[str, Any]]) -> None:
    mapping = {
        (row["method"], float(row["reward_quantile"]), row["scope"]): row
        for row in observed
    }
    if len(mapping) != len(observed) or len(mapping) != len(expected):
        raise ValueError("Stage75 audit summary grid differs")
    for item in expected:
        key = (item["method"], item["reward_quantile"], item["scope"])
        row = mapping[key]
        for field in ("trial_count", "cell_count", "initialization_fallback_count"):
            if int(row[field]) != int(item[field]):
                raise ValueError(f"Stage75 audit summary {field} differs: {key}")
        for field in (
            "pooled_best_match_rate",
            "mean_delta_vs_frozen_frontier_normalized",
            "maximum_delta_vs_frozen_frontier_normalized",
            "mean_selected_k",
            "mean_feasible_proposal_count",
        ):
            close(row[field], item[field], f"summary {field} {key}")


def aggregate(
    model_rows: list[dict[str, Any]], comparisons: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    paths: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        paths[(row["target_id"], row["outer_fold"])].append(row)
    for rows in paths.values():
        rows.sort(key=lambda row: row["reward_quantile"])
    exact = [row for row in comparisons if row["exact_frontier_available"]]
    encoding = {
        "cqm_model_count": len(model_rows),
        "maximum_logical_variable_count": max(row["variables"] for row in model_rows),
        "maximum_quadratic_coupler_count": max(row["couplers"] for row in model_rows),
        "explicit_constraint_count_per_model": 3,
        "maximum_frontier_energy_encoding_residual": max(row["residual"] for row in model_rows),
        "all_frontier_assignments_feasible": all(row["feasible"] for row in model_rows),
        "reward_path_count": len(paths),
        "monotonic_reward_path_count": sum(
            all(right["frozen_frontier_selected_k"] >= left["frozen_frontier_selected_k"] for left, right in zip(rows, rows[1:]))
            for rows in paths.values()
        ),
        "distinct_frontier_selected_k": sorted({row["frozen_frontier_selected_k"] for row in comparisons}),
    }
    performance = {
        "comparison_cell_count": len(comparisons),
        "exact_frontier_cell_count": len(exact),
        "joint_classical_exact_frontier_match_rate": statistics.fmean(row["joint_classical_exact_frontier_match"] for row in exact),
        "sampler_exact_frontier_match_rate": statistics.fmean(row["sampler_exact_frontier_match"] for row in exact),
        "sampler_joint_classical_competitive_fraction": statistics.fmean(row["sampler_within_joint_classical_tolerance"] for row in comparisons),
        "sampler_frozen_frontier_competitive_fraction": statistics.fmean(row["sampler_within_frozen_frontier_tolerance"] for row in comparisons),
        "sampler_strict_win_vs_joint_classical_cell_count": sum(row["sampler_strict_win_vs_joint_classical"] for row in comparisons),
        "joint_classical_strict_win_vs_sampler_cell_count": sum(row["joint_classical_strict_win_vs_sampler"] for row in comparisons),
        "joint_sampler_tie_cell_count": sum(not row["sampler_strict_win_vs_joint_classical"] and not row["joint_classical_strict_win_vs_sampler"] for row in comparisons),
        "frozen_frontier_refined_cell_count": sum(row["frozen_frontier_refined"] for row in comparisons),
    }
    gate = config["benchmark_gate"]
    encoding_gate = (
        encoding["all_frontier_assignments_feasible"]
        and encoding["maximum_frontier_energy_encoding_residual"] <= float(gate["maximum_energy_encoding_residual"])
        and encoding["monotonic_reward_path_count"] == int(gate["required_monotonic_reward_path_count"])
        and len(encoding["distinct_frontier_selected_k"]) >= int(gate["minimum_distinct_selected_k"])
    )
    exact_gate = (
        performance["joint_classical_exact_frontier_match_rate"] >= float(gate["minimum_joint_classical_exact_match_rate"])
        and performance["sampler_exact_frontier_match_rate"] >= float(gate["minimum_sampler_exact_match_rate"])
    )
    sampler_gate = (
        performance["sampler_joint_classical_competitive_fraction"] >= float(gate["minimum_sampler_joint_competitive_fraction"])
        and performance["sampler_frozen_frontier_competitive_fraction"] >= float(gate["minimum_sampler_frontier_competitive_fraction"])
    )
    return {
        "encoding_summary": encoding,
        "solver_performance": performance,
        "route_gate": {
            "explicit_variable_k_cqm_encoding_passed": bool(encoding_gate),
            "exact_frontier_solver_validation_passed": bool(exact_gate),
            "variable_k_sampler_competitiveness_passed": bool(sampler_gate),
        },
        "decision": {
            "explicit_variable_k_cqm_freeze_authorized": bool(encoding_gate),
            "local_hardware_shaped_emulation_authorized": bool(encoding_gate and exact_gate and sampler_gate),
            "cloud_cqm_execution_authorized": False,
            "direct_qpu_execution_authorized": False,
            "quantum_scaling_claim_authorized": False,
            "quantum_advantage_claim_authorized": False,
        },
    }


def run(
    config_path: Path, result_path: Path, root: Path, output_path: Path
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    result_path = result_path.resolve()
    output_path = output_path.resolve()
    if config_path != (root / "configs/stage75_explicit_variable_k_cqm.json").resolve():
        raise ValueError("Stage75 audit config path differs")
    if result_path != (root / "data/stage75_explicit_variable_k_cqm_result.json").resolve():
        raise ValueError("Stage75 audit result path differs")
    if output_path != (root / "data/stage75_explicit_variable_k_cqm_audit.json").resolve():
        raise ValueError("Stage75 audit output path differs")
    config = read_json(config_path)
    result = read_json(result_path)
    for key, value in config["implementation"].items():
        checked(root, value, f"implementation {key}")
    inputs = {
        key: checked(root, value, f"input {key}")
        for key, value in config["inputs"].items()
    }
    paths = {key: root / str(value) for key, value in config["outputs"].items()}
    observed_metrics = read_csv(paths["cqm_metrics_csv"])
    observed_trials = read_csv(paths["solver_trials_csv"])
    observed_comparisons = read_csv(paths["cell_comparison_csv"])
    observed_summaries = read_csv(paths["solver_summary_csv"])
    model_record = read_json(paths["model_record_json"])
    metric_map = {
        (row["target_id"], int(row["outer_fold"]), float(row["reward_quantile"])): row
        for row in observed_metrics
    }
    trial_map = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            float(row["reward_quantile"]),
            row["method"],
            int(row["repeat"]),
        ): row
        for row in observed_trials
    }
    source = read_json(inputs["stage72_model_record"])
    workloads = read_csv(inputs["stage74_workload_metrics"])
    source_comparisons = read_csv(inputs["stage74_cell_comparison"])
    source_trials = read_csv(inputs["stage74_solver_trials"])
    models = [rebuild(record) for record in source["models"]]
    expected_trials: list[dict[str, Any]] = []
    cells: dict[tuple[str, int, float], dict[str, Any]] = {}
    model_rows: list[dict[str, Any]] = []
    base_seed = int(config["solver_protocol"]["seed_base"])
    repeats = int(config["solver_protocol"]["stochastic_repeats"])
    regime = str(config["variable_k_cqm"]["quality_regime"])
    cell_index = 0
    for model in models:
        frontiers = recover_frontiers(
            model, workloads, source_comparisons, source_trials, regime
        )
        for quantile in config["variable_k_cqm"]["reward_quantiles"]:
            quantile = float(quantile)
            position, reward = reward_record(model, quantile)
            canonical = canonical_model(model, frontiers, reward)
            cqm = make_cqm(model, frontiers, reward)
            cell = {
                "model": model,
                "frontiers": frontiers,
                "quantile": quantile,
                "reward": reward,
                "protocol": config["solver_protocol"],
            }
            key = (str(model["record"]["target_id"]), int(model["record"]["outer_fold"]), quantile)
            cells[key] = cell
            metric = metric_map[key]
            if int(metric["reward_order_statistic_index"]) != position:
                raise ValueError("Stage75 audit reward order statistic differs")
            if metric["cqm_sha256"] != canonical_sha256(canonical):
                raise ValueError("Stage75 audit CQM hash differs")
            frontier_subset, frontier_energy = fixed_best(cell, "reference_subset")
            sample = sample_for(model, frontiers, frontier_subset)
            residual = abs(float(cqm.objective.energy(sample)) - frontier_energy)
            if not cqm.check_feasible(sample):
                raise ValueError("Stage75 audit CQM frontier sample is infeasible")
            close(metric["frontier_energy_encoding_residual"], residual, "CQM residual")
            if int(metric["total_logical_variable_count"]) != cqm.num_variables() or int(metric["explicit_constraint_count"]) != len(cqm.constraints):
                raise ValueError("Stage75 audit CQM size differs")
            model_rows.append(
                {
                    "variables": int(cqm.num_variables()),
                    "couplers": len(model["pairs"]),
                    "residual": residual,
                    "feasible": True,
                }
            )
            methods = [
                ("fixed_k_frontier_reference", 0, 0),
                ("decomposed_deterministic_baseline", 1, 0),
            ]
            methods.extend(
                ("budgeted_variable_tabu", 2, repeat) for repeat in range(repeats)
            )
            methods.extend(
                ("constraint_native_variable_annealing", 3, repeat)
                for repeat in range(repeats)
            )
            for method, method_index, repeat in methods:
                seed = base_seed + cell_index * 100_000 + method_index * 1_000 + repeat
                observed = trial_map[(key[0], key[1], key[2], method, repeat)]
                solved = expected_solution(cell, method, seed)
                expected_trials.append(
                    compare_trial(observed, cell, method, repeat, seed, solved)
                )
            cell_index += 1
        print(json.dumps({"audit_target": model["record"]["target_id"], "audit_fold": model["record"]["outer_fold"], "trials_replayed": len(expected_trials)}), flush=True)
    if len(metric_map) != cell_index or len(trial_map) != len(expected_trials):
        raise ValueError("Stage75 audit output counts differ")
    expected_comparisons = attach_expected(expected_trials, cells, config)
    compare_comparisons(observed_comparisons, expected_comparisons)
    compare_summaries(observed_summaries, expected_summaries(expected_trials))
    if model_record["model_count"] != cell_index:
        raise ValueError("Stage75 audit model record count differs")
    aggregate_value = aggregate(model_rows, expected_comparisons, config)
    for section in ("encoding_summary", "solver_performance"):
        for key, value in aggregate_value[section].items():
            observed = result[section][key]
            if isinstance(value, float):
                close(observed, value, f"result {section}.{key}")
            elif observed != value:
                raise ValueError(f"Stage75 audit result {section}.{key} differs")
    if result["route_gate"] != aggregate_value["route_gate"]:
        raise ValueError("Stage75 audit route gate differs")
    for key, value in aggregate_value["decision"].items():
        if bool(result["decision"][key]) != bool(value):
            raise ValueError(f"Stage75 audit decision differs: {key}")
    payload = {
        **aggregate_value,
        "cqm_metrics_sha256": sha256(paths["cqm_metrics_csv"]),
        "solver_trials_sha256": sha256(paths["solver_trials_csv"]),
        "cell_comparison_sha256": sha256(paths["cell_comparison_csv"]),
        "solver_summary_sha256": sha256(paths["solver_summary_csv"]),
        "model_record_sha256": sha256(paths["model_record_json"]),
    }
    if canonical_sha256(payload) != result["analysis_payload_sha256"]:
        raise ValueError("Stage75 audit analysis payload differs")
    expected_boundary = {
        "historical_development_targets_read": 4,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "cloud_cqm_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
    if result["data_boundary"] != expected_boundary:
        raise ValueError("Stage75 audit data boundary differs")
    audit = {
        "schema_version": "1.0",
        "status": "stage75_explicit_variable_k_cqm_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "stage72_models_independently_rebuilt": len(models),
        "cqm_models_independently_rebuilt": cell_index,
        "solver_trials_deterministically_replayed": len(expected_trials),
        "cell_comparisons_independently_recomputed": len(expected_comparisons),
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
    parser.add_argument("--config", type=Path, default=Path("configs/stage75_explicit_variable_k_cqm.json"))
    parser.add_argument("--result", type=Path, default=Path("data/stage75_explicit_variable_k_cqm_result.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/stage75_explicit_variable_k_cqm_audit.json"))
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
