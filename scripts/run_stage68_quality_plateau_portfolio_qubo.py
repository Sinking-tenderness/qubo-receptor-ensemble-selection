"""Develop a quality-preserving functional-diversity portfolio QUBO."""

from __future__ import annotations

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
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

from scripts.run_stage42d_bace1_large_pool_qubo_screen import (
    bedroc_metrics,
    rank_cube,
)
from scripts.run_stage64_cross_target_uncertainty_shrunk_qubo import (
    TOLERANCE,
    jackknife_pair_statistics,
    load_target,
    pairwise_jaccard,
)


SOLVER_PAIR_OFF = "pair_off_baseline"
SOLVER_EXACT = "continuous_milp_certificate"
SOLVER_GREEDY = "same_constraint_direct_greedy"
SOLVER_SWAP = "same_constraint_greedy_swap"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        raise ValueError(f"Stage68 frozen {label} identity differs: {path}")
    return path


def subset_name(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(receptor_ids[index] for index in subset)


def subset_jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    return len(left_set & right_set) / len(left_set | right_set)


def stable_redundancy(ranks: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Penalize only rank-profile correlation that is positive in every seed."""
    matrices = np.stack(
        [
            np.corrcoef(ranks[seed, train_mask, :], rowvar=False)
            for seed in range(ranks.shape[0])
        ]
    )
    if not np.isfinite(matrices).all():
        raise ValueError("Stage68 rank-profile correlation is non-finite")
    redundancy = np.maximum(0.0, np.min(matrices, axis=0))
    redundancy = np.clip((redundancy + redundancy.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(redundancy, 0.0)
    return redundancy


def redundancy_sum(subset: tuple[int, ...], redundancy: np.ndarray) -> float:
    return float(
        sum(redundancy[left, right] for left, right in itertools.combinations(subset, 2))
    )


def redundancy_mean(subset: tuple[int, ...], redundancy: np.ndarray) -> float:
    pair_count = math.comb(len(subset), 2)
    return redundancy_sum(subset, redundancy) / pair_count if pair_count else 0.0


def pair_off_subset(utility: np.ndarray, subset_size: int) -> tuple[int, ...]:
    order = np.argsort(-utility, kind="stable")[:subset_size]
    return tuple(sorted(int(value) for value in order))


def quality_plateau(
    utility: np.ndarray,
    spread: np.ndarray,
    subset_size: int,
    uncertainty_multiplier: float,
) -> dict[str, Any]:
    baseline = pair_off_subset(utility, subset_size)
    baseline_quality = float(np.mean(utility[list(baseline)]))
    robust_uncertainty = float(
        np.sqrt(np.sum(np.square(spread[list(baseline)]))) / subset_size
    )
    quality_floor = baseline_quality - uncertainty_multiplier * robust_uncertainty
    return {
        "baseline_subset": baseline,
        "baseline_quality": baseline_quality,
        "robust_uncertainty": robust_uncertainty,
        "quality_floor": quality_floor,
    }


class PortfolioMilp:
    """Exact MILP certificate for a fixed-cardinality quadratic portfolio."""

    def __init__(self, redundancy: np.ndarray):
        self.redundancy = redundancy
        self.receptor_count = redundancy.shape[0]
        self.pairs = list(itertools.combinations(range(self.receptor_count), 2))
        self.edge_count = len(self.pairs)
        self.variable_count = self.receptor_count + self.edge_count
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for edge, (left, right) in enumerate(self.pairs):
            rows.extend((edge, edge, edge))
            columns.extend((left, right, self.receptor_count + edge))
            values.extend((-1.0, -1.0, 1.0))
        self.edge_constraints = coo_matrix(
            (values, (rows, columns)),
            shape=(self.edge_count, self.variable_count),
        ).tocsr()
        self.cardinality = np.zeros(self.variable_count, dtype=float)
        self.cardinality[: self.receptor_count] = 1.0
        self.objective = np.zeros(self.variable_count, dtype=float)
        self.objective[self.receptor_count :] = np.asarray(
            [redundancy[left, right] for left, right in self.pairs], dtype=float
        )
        self.bounds = Bounds(
            np.zeros(self.variable_count), np.ones(self.variable_count)
        )
        self.integrality = np.ones(self.variable_count)

    def solve_lower_quality(
        self,
        utility: np.ndarray,
        subset_size: int,
        quality_floor_sum: float,
        time_limit_seconds: float,
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        quality = np.zeros(self.variable_count, dtype=float)
        quality[: self.receptor_count] = utility
        matrix = vstack(
            [
                self.edge_constraints,
                coo_matrix(self.cardinality),
                coo_matrix(quality),
            ]
        ).tocsr()
        lower = np.concatenate(
            [
                np.full(self.edge_count, -1.0),
                np.asarray([subset_size, quality_floor_sum]),
            ]
        )
        upper = np.concatenate(
            [
                np.full(self.edge_count, np.inf),
                np.asarray([subset_size, np.inf]),
            ]
        )
        return self._solve(matrix, lower, upper, utility, time_limit_seconds)

    def solve_upper_deficit(
        self,
        deficits: np.ndarray,
        subset_size: int,
        maximum_deficit: int,
        tie_utility: np.ndarray,
        time_limit_seconds: float,
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        deficit = np.zeros(self.variable_count, dtype=float)
        deficit[: self.receptor_count] = deficits
        matrix = vstack(
            [
                self.edge_constraints,
                coo_matrix(self.cardinality),
                coo_matrix(deficit),
            ]
        ).tocsr()
        lower = np.concatenate(
            [
                np.full(self.edge_count, -1.0),
                np.asarray([subset_size, -np.inf]),
            ]
        )
        upper = np.concatenate(
            [
                np.full(self.edge_count, np.inf),
                np.asarray([subset_size, maximum_deficit]),
            ]
        )
        return self._solve(matrix, lower, upper, tie_utility, time_limit_seconds)

    def _solve(
        self,
        matrix: Any,
        lower: np.ndarray,
        upper: np.ndarray,
        tie_utility: np.ndarray,
        time_limit_seconds: float,
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        objective = self.objective.copy()
        value_range = max(float(np.ptp(tie_utility)), TOLERANCE)
        objective[: self.receptor_count] = (
            -1e-7 * (tie_utility - float(np.min(tie_utility))) / value_range
            + 1e-11 * np.arange(self.receptor_count)
        )
        result = milp(
            objective,
            integrality=self.integrality,
            bounds=self.bounds,
            constraints=LinearConstraint(matrix, lower, upper),
            options={
                "time_limit": float(time_limit_seconds),
                "mip_rel_gap": 0.0,
                "presolve": True,
            },
        )
        if not bool(result.success) or result.x is None:
            raise RuntimeError(f"Stage68 MILP failed: {result.message}")
        subset = tuple(
            int(value)
            for value in np.flatnonzero(
                result.x[: self.receptor_count] > 0.5
            )
        )
        record = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "mip_gap": float(getattr(result, "mip_gap", 0.0) or 0.0),
            "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
            "objective": float(result.fun),
        }
        return subset, record


def can_complete(
    selected: tuple[int, ...],
    candidate: int,
    utility: np.ndarray,
    subset_size: int,
    quality_floor_sum: float,
) -> bool:
    proposed = tuple(sorted((*selected, candidate)))
    remaining_count = subset_size - len(proposed)
    remaining = [
        index
        for index in range(len(utility))
        if index not in proposed
    ]
    best_remaining = sorted(
        (float(utility[index]) for index in remaining), reverse=True
    )[:remaining_count]
    if len(best_remaining) != remaining_count:
        return False
    return (
        float(np.sum(utility[list(proposed)])) + sum(best_remaining)
        >= quality_floor_sum - TOLERANCE
    )


def direct_greedy(
    utility: np.ndarray,
    redundancy: np.ndarray,
    subset_size: int,
    quality_floor: float,
) -> tuple[int, ...]:
    selected: tuple[int, ...] = ()
    floor_sum = subset_size * quality_floor
    while len(selected) < subset_size:
        candidates = [
            index
            for index in range(len(utility))
            if index not in selected
            and can_complete(
                selected, index, utility, subset_size, floor_sum
            )
        ]
        if not candidates:
            raise ValueError("Stage68 greedy search lost quality-floor feasibility")

        def key(index: int) -> tuple[Any, ...]:
            proposed = tuple(sorted((*selected, index)))
            return (
                redundancy_sum(proposed, redundancy),
                -float(np.mean(utility[list(proposed)])),
                proposed,
            )

        selected = tuple(sorted((*selected, min(candidates, key=key))))
    if float(np.mean(utility[list(selected)])) < quality_floor - TOLERANCE:
        raise ValueError("Stage68 greedy subset violates its quality floor")
    return selected


def local_swap(
    initial: tuple[int, ...],
    utility: np.ndarray,
    redundancy: np.ndarray,
    quality_floor: float,
) -> tuple[int, ...]:
    current = initial
    while True:
        current_key = (
            redundancy_sum(current, redundancy),
            -float(np.mean(utility[list(current)])),
            current,
        )
        selected = set(current)
        neighbors = {
            tuple(sorted((selected - {removed}) | {added}))
            for removed in current
            for added in range(len(utility))
            if added not in selected
        }
        feasible = [
            subset
            for subset in neighbors
            if float(np.mean(utility[list(subset)]))
            >= quality_floor - TOLERANCE
        ]
        if not feasible:
            return current

        def key(subset: tuple[int, ...]) -> tuple[Any, ...]:
            return (
                redundancy_sum(subset, redundancy),
                -float(np.mean(utility[list(subset)])),
                subset,
            )

        best = min(feasible, key=key)
        if key(best) >= current_key:
            return current
        current = best


def integerize_quality(
    utility: np.ndarray,
    subset_size: int,
    quality_floor: float,
    scale: int,
) -> dict[str, Any]:
    utility_range = float(np.ptp(utility))
    if utility_range <= TOLERANCE:
        raise ValueError("Stage68 utility range is degenerate")
    normalized_deficit = (float(np.max(utility)) - utility) / utility_range
    deficits = np.ceil(scale * normalized_deficit - TOLERANCE).astype(int)
    maximum_deficit = math.floor(
        scale
        * subset_size
        * (float(np.max(utility)) - quality_floor)
        / utility_range
        + TOLERANCE
    )
    baseline = pair_off_subset(utility, subset_size)
    if int(np.sum(deficits[list(baseline)])) > maximum_deficit:
        raise ValueError("Stage68 conservative integerization excludes pair-off")
    slack_bit_count = max(1, math.ceil(math.log2(maximum_deficit + 1)))
    slack_weights = [1 << bit for bit in range(slack_bit_count)]
    return {
        "deficits": deficits,
        "maximum_deficit": maximum_deficit,
        "slack_weights": slack_weights,
        "utility_range": utility_range,
        "maximum_continuous_floor_relaxation": utility_range
        * subset_size
        / scale,
    }


def expanded_qubo_summary(
    redundancy: np.ndarray,
    deficits: np.ndarray,
    maximum_deficit: int,
    slack_weights: list[int],
    subset_size: int,
    cardinality_penalty: float,
    quality_penalty: float,
) -> dict[str, Any]:
    receptor_count = len(deficits)
    weights = [int(value) for value in deficits] + slack_weights
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}
    constant = cardinality_penalty * subset_size**2
    for index in range(receptor_count):
        linear[index] = cardinality_penalty * (1 - 2 * subset_size)
    for left, right in itertools.combinations(range(receptor_count), 2):
        quadratic[(left, right)] = (
            2 * cardinality_penalty + float(redundancy[left, right])
        )
    constant += quality_penalty * maximum_deficit**2
    for index, weight in enumerate(weights):
        variable = index
        linear[variable] = linear.get(variable, 0.0) + quality_penalty * (
            weight**2 - 2 * maximum_deficit * weight
        )
    for left, right in itertools.combinations(range(len(weights)), 2):
        value = 2 * quality_penalty * weights[left] * weights[right]
        quadratic[(left, right)] = quadratic.get((left, right), 0.0) + value
    coefficients = [
        abs(value)
        for value in [*linear.values(), *quadratic.values()]
        if abs(value) > TOLERANCE
    ]
    return {
        "logical_variable_count": receptor_count + len(slack_weights),
        "receptor_variable_count": receptor_count,
        "slack_variable_count": len(slack_weights),
        "linear_coefficient_count": len(linear),
        "quadratic_coefficient_count": len(quadratic),
        "constant": constant,
        "minimum_absolute_nonzero_coefficient": min(coefficients),
        "maximum_absolute_coefficient": max(coefficients),
        "coefficient_dynamic_range": max(coefficients) / min(coefficients),
    }


def factorized_energy(
    subset: tuple[int, ...],
    slack_value: int,
    redundancy: np.ndarray,
    deficits: np.ndarray,
    maximum_deficit: int,
    subset_size: int,
    cardinality_penalty: float,
    quality_penalty: float,
) -> float:
    return float(
        redundancy_sum(subset, redundancy)
        + cardinality_penalty * (len(subset) - subset_size) ** 2
        + quality_penalty
        * (
            int(np.sum(deficits[list(subset)]))
            + slack_value
            - maximum_deficit
        )
        ** 2
    )


def metrics_row(
    target_id: str,
    outer_fold: int,
    candidate_id: str,
    uncertainty_multiplier: float,
    solver_id: str,
    subset_size: int,
    subset: tuple[int, ...],
    receptor_ids: list[str],
    utility: np.ndarray,
    quality_floor: float,
    robust_uncertainty: float,
    redundancy: np.ndarray,
    ranks: np.ndarray,
    labels: np.ndarray,
    holdout_mask: np.ndarray,
    alpha: float,
    solver_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    holdout = bedroc_metrics(
        ranks[:, holdout_mask, :], labels[holdout_mask], subset, alpha
    )
    quality = float(np.mean(utility[list(subset)]))
    row = {
        "target_id": target_id,
        "outer_fold": outer_fold,
        "candidate_id": candidate_id,
        "uncertainty_multiplier": uncertainty_multiplier,
        "solver_id": solver_id,
        "subset_size": subset_size,
        "selected_subset": subset_name(subset, receptor_ids),
        "train_mean_singleton_utility": quality,
        "train_quality_floor": quality_floor,
        "train_quality_margin": quality - quality_floor,
        "robust_quality_uncertainty": robust_uncertainty,
        "stable_redundancy_sum": redundancy_sum(subset, redundancy),
        "stable_redundancy_mean": redundancy_mean(subset, redundancy),
        "holdout_primary_bedroc": holdout["primary_bedroc"],
        "holdout_mean_seed_bedroc": holdout["mean_seed_bedroc"],
        "holdout_worst_seed_bedroc": holdout["worst_seed_bedroc"],
        "holdout_robust_bedroc": holdout["robust_bedroc_composite"],
    }
    if solver_record is not None:
        row.update(
            {
                "milp_status": solver_record["status"],
                "milp_gap": solver_record["mip_gap"],
                "milp_node_count": solver_record["mip_node_count"],
            }
        )
    return row


def summarize_candidates(
    rows: list[dict[str, Any]],
    multipliers: list[float],
    targets: list[str],
    gates: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pair_off = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in rows
        if row["solver_id"] == SOLVER_PAIR_OFF
    }
    target_rows: list[dict[str, Any]] = []
    for multiplier in multipliers:
        candidate_id = f"uncertainty_{str(multiplier).replace('.', 'p')}x"
        for target_id in targets:
            selected = [
                row
                for row in rows
                if row["target_id"] == target_id
                and row["candidate_id"] == candidate_id
                and row["solver_id"] == SOLVER_EXACT
            ]
            gains = [
                float(row["holdout_robust_bedroc"])
                - float(
                    pair_off[
                        (
                            target_id,
                            int(row["outer_fold"]),
                            int(row["subset_size"]),
                        )
                    ]["holdout_robust_bedroc"]
                )
                for row in selected
            ]
            redundancy_reductions = [
                float(
                    pair_off[
                        (
                            target_id,
                            int(row["outer_fold"]),
                            int(row["subset_size"]),
                        )
                    ]["stable_redundancy_mean"]
                )
                - float(row["stable_redundancy_mean"])
                for row in selected
            ]
            target_rows.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "uncertainty_multiplier": multiplier,
                    "fixed_k_cell_count": len(selected),
                    "mean_holdout_robust_bedroc": statistics.fmean(
                        float(row["holdout_robust_bedroc"]) for row in selected
                    ),
                    "mean_gain_over_pair_off": statistics.fmean(gains),
                    "minimum_fold_k_gain_over_pair_off": min(gains),
                    "noninferior_fold_k_count_at_0p01": sum(
                        value >= -0.01 - TOLERANCE for value in gains
                    ),
                    "mean_stable_redundancy_reduction": statistics.fmean(
                        redundancy_reductions
                    ),
                    "minimum_stable_redundancy_reduction": min(
                        redundancy_reductions
                    ),
                    "selection_difference_count_vs_pair_off": sum(
                        row["selected_subset"]
                        != pair_off[
                            (
                                target_id,
                                int(row["outer_fold"]),
                                int(row["subset_size"]),
                            )
                        ]["selected_subset"]
                        for row in selected
                    ),
                    "mean_fixed_k_selection_jaccard": statistics.fmean(
                        pairwise_jaccard(
                            [
                                row["selected_subset"]
                                for row in selected
                                if int(row["subset_size"]) == subset_size
                            ]
                        )
                        for subset_size in range(2, 7)
                    ),
                }
            )
    lookup = {
        (row["target_id"], row["candidate_id"]): row for row in target_rows
    }
    global_rows: list[dict[str, Any]] = []
    for multiplier in multipliers:
        candidate_id = f"uncertainty_{str(multiplier).replace('.', 'p')}x"
        selected = [lookup[(target, candidate_id)] for target in targets]
        gains = [float(row["mean_gain_over_pair_off"]) for row in selected]
        reductions = [
            float(row["mean_stable_redundancy_reduction"]) for row in selected
        ]
        global_rows.append(
            {
                "candidate_id": candidate_id,
                "uncertainty_multiplier": multiplier,
                "mean_target_gain_over_pair_off": statistics.fmean(gains),
                "worst_target_gain_over_pair_off": min(gains),
                "target_count_within_0p01_of_pair_off": sum(
                    value >= -0.01 - TOLERANCE for value in gains
                ),
                "positive_target_count_over_pair_off": sum(
                    value > TOLERANCE for value in gains
                ),
                "mean_target_stable_redundancy_reduction": statistics.fmean(
                    reductions
                ),
                "target_count_with_nonnegative_redundancy_reduction": sum(
                    value >= -TOLERANCE for value in reductions
                ),
                "selection_difference_count_vs_pair_off": sum(
                    int(row["selection_difference_count_vs_pair_off"])
                    for row in selected
                ),
                "mean_target_selection_jaccard": statistics.fmean(
                    float(row["mean_fixed_k_selection_jaccard"])
                    for row in selected
                ),
            }
        )
    eligible = [
        row
        for row in global_rows
        if float(row["mean_target_gain_over_pair_off"])
        >= float(gates["minimum_mean_target_gain_over_pair_off"])
        and float(row["worst_target_gain_over_pair_off"])
        >= float(gates["minimum_worst_target_gain_over_pair_off"])
        and int(row["target_count_within_0p01_of_pair_off"])
        >= int(gates["minimum_target_count_within_0p01"])
        and float(row["mean_target_stable_redundancy_reduction"])
        >= float(gates["minimum_mean_redundancy_reduction"])
        and int(row["target_count_with_nonnegative_redundancy_reduction"])
        >= int(gates["minimum_target_count_with_nonnegative_redundancy_reduction"])
    ]
    selected = (
        min(eligible, key=lambda row: float(row["uncertainty_multiplier"]))
        if eligible
        else {}
    )
    return target_rows, global_rows, selected


def loto_summary(
    target_rows: list[dict[str, Any]],
    multipliers: list[float],
    targets: list[str],
    gates: dict[str, Any],
) -> list[dict[str, Any]]:
    lookup = {
        (row["target_id"], row["candidate_id"]): row for row in target_rows
    }
    output: list[dict[str, Any]] = []
    for held in targets:
        development_targets = [target for target in targets if target != held]
        selected: dict[str, Any] | None = None
        for multiplier in sorted(multipliers):
            candidate_id = f"uncertainty_{str(multiplier).replace('.', 'p')}x"
            rows = [lookup[(target, candidate_id)] for target in development_targets]
            gains = [float(row["mean_gain_over_pair_off"]) for row in rows]
            reductions = [
                float(row["mean_stable_redundancy_reduction"]) for row in rows
            ]
            if (
                statistics.fmean(gains)
                >= float(gates["minimum_mean_target_gain_over_pair_off"])
                and min(gains)
                >= float(gates["minimum_worst_target_gain_over_pair_off"])
                and sum(value >= -0.01 - TOLERANCE for value in gains)
                == len(development_targets)
                and statistics.fmean(reductions)
                >= float(gates["minimum_mean_redundancy_reduction"])
                and all(value >= -TOLERANCE for value in reductions)
            ):
                selected = {
                    "candidate_id": candidate_id,
                    "uncertainty_multiplier": multiplier,
                    "development_mean_gain": statistics.fmean(gains),
                    "development_worst_gain": min(gains),
                    "development_mean_redundancy_reduction": statistics.fmean(
                        reductions
                    ),
                }
                break
        if selected is None:
            output.append(
                {
                    "held_target_id": held,
                    "selected_candidate_id": "pair_off_fallback",
                    "selected_uncertainty_multiplier": 0.0,
                    "development_mean_gain": 0.0,
                    "development_worst_gain": 0.0,
                    "development_mean_redundancy_reduction": 0.0,
                    "held_target_gain_over_pair_off": 0.0,
                    "held_target_redundancy_reduction": 0.0,
                }
            )
            continue
        held_row = lookup[(held, selected["candidate_id"])]
        output.append(
            {
                "held_target_id": held,
                "selected_candidate_id": selected["candidate_id"],
                "selected_uncertainty_multiplier": selected[
                    "uncertainty_multiplier"
                ],
                "development_mean_gain": selected["development_mean_gain"],
                "development_worst_gain": selected["development_worst_gain"],
                "development_mean_redundancy_reduction": selected[
                    "development_mean_redundancy_reduction"
                ],
                "held_target_gain_over_pair_off": float(
                    held_row["mean_gain_over_pair_off"]
                ),
                "held_target_redundancy_reduction": float(
                    held_row["mean_stable_redundancy_reduction"]
                ),
            }
        )
    return output


def report_text(result: dict[str, Any]) -> str:
    selected = result["selected_candidate"]
    quantized = result["qubo_fidelity"]
    loto = result["loto_gate"]
    return rf"""# Stage68 quality-plateau portfolio QUBO

## Question

Can a compact QUBO reduce functionally redundant receptor choices while retaining the strong additive pair-off screening baseline within a frozen training-uncertainty floor?

## Frozen formulation

For fixed $k$, the pair-off top-$k$ mean singleton utility defines $U_k^*$. The quality constraint is

$$
\frac{{1}}{{k}}\sum_i u_i x_i \ge U_k^* - m\sigma_k,
$$

where $m\in\{{0.25,0.5,1.0\}}$ and $\sigma_k$ is the root-sum-square jackknife spread of the pair-off top-$k$ set divided by $k$. Within that feasible plateau, Stage68 minimizes

$$
\sum_{{i<j}} R_{{ij}}x_ix_j,
\qquad
R_{{ij}}=\max\left(0,\min_s \mathrm{{corr}}(r_{{s,i}},r_{{s,j}})\right).
$$

Only receptor rank correlations positive in all three docking seeds are penalized.

## Development result

- Selected multiplier: `{selected.get('uncertainty_multiplier', 'none')}x`.
- Mean target BEDROC20 gain versus pair-off: `{selected.get('mean_target_gain_over_pair_off', float('nan')):+.6f}`.
- Worst target mean gain: `{selected.get('worst_target_gain_over_pair_off', float('nan')):+.6f}`.
- Mean stable-redundancy reduction: `{selected.get('mean_target_stable_redundancy_reduction', float('nan')):.6f}`.
- Targets within 0.01 BEDROC of pair-off: `{selected.get('target_count_within_0p01_of_pair_off', 0)}/4`.

## Transfer and QUBO fidelity

- Leave-one-target-out mean held-target gain: `{loto['mean_held_target_gain_over_pair_off']:+.6f}`.
- Leave-one-target-out worst held-target gain: `{loto['worst_held_target_gain_over_pair_off']:+.6f}`.
- QUBO/continuous mean subset Jaccard: `{quantized['mean_subset_jaccard_vs_continuous']:.6f}`.
- QUBO mean holdout gap versus continuous certificate: `{quantized['mean_holdout_bedroc_gap_vs_continuous']:+.6f}`.
- Maximum logical variables: `{quantized['maximum_logical_variable_count']}`.
- Maximum coefficient dynamic range: `{quantized['maximum_coefficient_dynamic_range']:.6g}`.

## Decision boundary

The objective may be frozen only for preregistration on a genuinely new target if every Stage68 route gate passes. Exact MILP is retained as the strongest classical reference. This stage does not establish independent efficacy, solver speedup, quantum execution, or quantum advantage. Hardware execution remains blocked until coefficient compression and embedding are audited.

An unfrozen alternate jackknife partition changed the route decision from pass to fail. Therefore this is a protocol-specific development freeze, not evidence of partition robustness; the alternate probe is retained under `analysis/stage68_unfrozen_partition_probe_20260806`.
"""


def compute(config: dict[str, Any], root: Path) -> dict[str, Any]:
    implementation_paths = {
        key: verified(root, value, key)
        for key, value in config["implementation"].items()
    }
    input_paths = {
        key: verified(root, value, key) for key, value in config["inputs"].items()
    }
    stage67 = read_json(input_paths["stage67_result"])
    if stage67.get("status") != "stage67_bedroc_rankbin_qubo_complete":
        raise ValueError("Stage68 requires completed Stage67 adjudication")
    if stage67["decision"]["same_target_retuning_authorized"]:
        raise ValueError("Stage67 boundary unexpectedly authorizes retuning")
    stage64_config = read_json(input_paths["stage64_config"])
    development = config["development"]
    targets = [str(value) for value in development["target_order"]]
    multipliers = [float(value) for value in development["uncertainty_multipliers"]]
    subset_sizes = [int(value) for value in development["candidate_k_values"]]
    alpha = float(development["bedroc_alpha"])
    loaded = {
        target: load_target(root, target, stage64_config["targets"][target])
        for target in targets
    }
    rows: list[dict[str, Any]] = []
    fold_cache: dict[tuple[str, int], dict[str, Any]] = {}
    milp_count = 0
    for target_id in targets:
        target = loaded[target_id]
        ligand_ids = target["ligand_ids"]
        labels = target["labels"]
        receptor_ids = target["receptor_ids"]
        for outer_fold in range(int(development["outer_fold_count"])):
            train_mask = np.asarray(
                [target["outer"][ligand_id] != outer_fold for ligand_id in ligand_ids]
            )
            holdout_mask = ~train_mask
            ranks = rank_cube(target["scores"], train_mask)
            train_rows = [
                row for row, keep in zip(target["ligands"], train_mask) if keep
            ]
            statistics_ = jackknife_pair_statistics(
                target["scores"][:, train_mask, :],
                labels[train_mask],
                train_rows,
                alpha,
                int(development["jackknife_block_count"]),
                int(development["jackknife_seed_base"])
                + outer_fold,
            )
            utility = statistics_["full_singleton"]
            spread = statistics_["singleton_spread"]
            redundancy = stable_redundancy(ranks, train_mask)
            workspace = PortfolioMilp(redundancy)
            fold_cache[(target_id, outer_fold)] = {
                "target": target,
                "labels": labels,
                "receptor_ids": receptor_ids,
                "train_mask": train_mask,
                "holdout_mask": holdout_mask,
                "ranks": ranks,
                "utility": utility,
                "spread": spread,
                "redundancy": redundancy,
                "workspace": workspace,
            }
            for subset_size in subset_sizes:
                baseline_plateau = quality_plateau(
                    utility, spread, subset_size, 0.0
                )
                baseline = baseline_plateau["baseline_subset"]
                rows.append(
                    metrics_row(
                        target_id,
                        outer_fold,
                        "pair_off",
                        0.0,
                        SOLVER_PAIR_OFF,
                        subset_size,
                        baseline,
                        receptor_ids,
                        utility,
                        baseline_plateau["baseline_quality"],
                        baseline_plateau["robust_uncertainty"],
                        redundancy,
                        ranks,
                        labels,
                        holdout_mask,
                        alpha,
                    )
                )
                for multiplier in multipliers:
                    candidate_id = (
                        f"uncertainty_{str(multiplier).replace('.', 'p')}x"
                    )
                    plateau = quality_plateau(
                        utility, spread, subset_size, multiplier
                    )
                    exact, solver_record = workspace.solve_lower_quality(
                        utility,
                        subset_size,
                        subset_size * plateau["quality_floor"],
                        float(development["milp_time_limit_seconds"]),
                    )
                    milp_count += 1
                    greedy = direct_greedy(
                        utility,
                        redundancy,
                        subset_size,
                        plateau["quality_floor"],
                    )
                    swapped = local_swap(
                        greedy, utility, redundancy, plateau["quality_floor"]
                    )
                    for solver_id, subset, record in (
                        (SOLVER_EXACT, exact, solver_record),
                        (SOLVER_GREEDY, greedy, None),
                        (SOLVER_SWAP, swapped, None),
                    ):
                        rows.append(
                            metrics_row(
                                target_id,
                                outer_fold,
                                candidate_id,
                                multiplier,
                                solver_id,
                                subset_size,
                                subset,
                                receptor_ids,
                                utility,
                                plateau["quality_floor"],
                                plateau["robust_uncertainty"],
                                redundancy,
                                ranks,
                                labels,
                                holdout_mask,
                                alpha,
                                record,
                            )
                        )
            print(
                json.dumps(
                    {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "candidate_count": len(multipliers),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    target_rows, global_rows, selected = summarize_candidates(
        rows, multipliers, targets, config["route_gate"]
    )
    if not selected:
        selected_multiplier = float(multipliers[0])
        selected_candidate_id = "none"
    else:
        selected_multiplier = float(selected["uncertainty_multiplier"])
        selected_candidate_id = str(selected["candidate_id"])
    loto_rows = loto_summary(
        target_rows, multipliers, targets, config["route_gate"]
    )
    loto_gains = [
        float(row["held_target_gain_over_pair_off"]) for row in loto_rows
    ]
    loto_gate = {
        "mean_held_target_gain_over_pair_off": statistics.fmean(loto_gains),
        "worst_held_target_gain_over_pair_off": min(loto_gains),
        "held_target_count_within_0p01": sum(
            value >= -0.01 - TOLERANCE for value in loto_gains
        ),
    }

    qubo_rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    maximum_energy_residual = 0.0
    qubo = config["qubo_encoding"]
    for target_id in targets:
        for outer_fold in range(int(development["outer_fold_count"])):
            cached = fold_cache[(target_id, outer_fold)]
            utility = cached["utility"]
            spread = cached["spread"]
            redundancy = cached["redundancy"]
            workspace = cached["workspace"]
            receptor_ids = cached["receptor_ids"]
            for subset_size in subset_sizes:
                plateau = quality_plateau(
                    utility, spread, subset_size, selected_multiplier
                )
                continuous, _ = workspace.solve_lower_quality(
                    utility,
                    subset_size,
                    subset_size * plateau["quality_floor"],
                    float(development["milp_time_limit_seconds"]),
                )
                milp_count += 1
                integerized = integerize_quality(
                    utility,
                    subset_size,
                    plateau["quality_floor"],
                    int(qubo["quality_integer_scale"]),
                )
                quantized, quantized_record = workspace.solve_upper_deficit(
                    integerized["deficits"],
                    subset_size,
                    integerized["maximum_deficit"],
                    utility,
                    float(development["milp_time_limit_seconds"]),
                )
                milp_count += 1
                continuous_metrics = bedroc_metrics(
                    cached["ranks"][:, cached["holdout_mask"], :],
                    cached["labels"][cached["holdout_mask"]],
                    continuous,
                    alpha,
                )
                quantized_metrics = bedroc_metrics(
                    cached["ranks"][:, cached["holdout_mask"], :],
                    cached["labels"][cached["holdout_mask"]],
                    quantized,
                    alpha,
                )
                actual_quality = float(np.mean(utility[list(quantized)]))
                qubo_rows.append(
                    {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "candidate_id": selected_candidate_id,
                        "subset_size": subset_size,
                        "continuous_subset": subset_name(
                            continuous, receptor_ids
                        ),
                        "quantized_qubo_subset": subset_name(
                            quantized, receptor_ids
                        ),
                        "subset_jaccard": subset_jaccard(
                            continuous, quantized
                        ),
                        "continuous_stable_redundancy_mean": redundancy_mean(
                            continuous, redundancy
                        ),
                        "quantized_stable_redundancy_mean": redundancy_mean(
                            quantized, redundancy
                        ),
                        "quantized_train_quality": actual_quality,
                        "continuous_quality_floor": plateau["quality_floor"],
                        "actual_quality_floor_margin": actual_quality
                        - plateau["quality_floor"],
                        "integer_deficit": int(
                            np.sum(integerized["deficits"][list(quantized)])
                        ),
                        "maximum_integer_deficit": integerized[
                            "maximum_deficit"
                        ],
                        "continuous_holdout_robust_bedroc": continuous_metrics[
                            "robust_bedroc_composite"
                        ],
                        "quantized_holdout_robust_bedroc": quantized_metrics[
                            "robust_bedroc_composite"
                        ],
                        "quantized_minus_continuous_holdout_bedroc": quantized_metrics[
                            "robust_bedroc_composite"
                        ]
                        - continuous_metrics["robust_bedroc_composite"],
                        "milp_gap": quantized_record["mip_gap"],
                    }
                )
                if subset_size == int(qubo["reference_model_k"]):
                    slack_value = integerized["maximum_deficit"] - int(
                        np.sum(integerized["deficits"][list(quantized)])
                    )
                    energy = factorized_energy(
                        quantized,
                        slack_value,
                        redundancy,
                        integerized["deficits"],
                        integerized["maximum_deficit"],
                        subset_size,
                        float(qubo["cardinality_penalty"]),
                        float(qubo["quality_penalty"]),
                    )
                    residual = abs(
                        energy - redundancy_sum(quantized, redundancy)
                    )
                    maximum_energy_residual = max(
                        maximum_energy_residual, residual
                    )
                    expanded = expanded_qubo_summary(
                        redundancy,
                        integerized["deficits"],
                        integerized["maximum_deficit"],
                        integerized["slack_weights"],
                        subset_size,
                        float(qubo["cardinality_penalty"]),
                        float(qubo["quality_penalty"]),
                    )
                    model_records.append(
                        {
                            "target_id": target_id,
                            "outer_fold": outer_fold,
                            "candidate_id": selected_candidate_id,
                            "reference_k": subset_size,
                            "receptor_ids": receptor_ids,
                            "selected_subset": subset_name(
                                quantized, receptor_ids
                            ),
                            "quality_integer_scale": int(
                                qubo["quality_integer_scale"]
                            ),
                            "integer_deficits": [
                                int(value)
                                for value in integerized["deficits"]
                            ],
                            "maximum_integer_deficit": integerized[
                                "maximum_deficit"
                            ],
                            "slack_weights": integerized["slack_weights"],
                            "selected_slack_value": slack_value,
                            "selected_factorized_energy": energy,
                            "selected_redundancy_sum": redundancy_sum(
                                quantized, redundancy
                            ),
                            "energy_residual": residual,
                            "stable_redundancy_upper_triangle": [
                                float(redundancy[left, right])
                                for left, right in itertools.combinations(
                                    range(len(receptor_ids)), 2
                                )
                            ],
                            "qubo_scale": expanded,
                        }
                    )
    fidelity = {
        "cell_count": len(qubo_rows),
        "mean_subset_jaccard_vs_continuous": statistics.fmean(
            float(row["subset_jaccard"]) for row in qubo_rows
        ),
        "minimum_subset_jaccard_vs_continuous": min(
            float(row["subset_jaccard"]) for row in qubo_rows
        ),
        "mean_holdout_bedroc_gap_vs_continuous": statistics.fmean(
            float(row["quantized_minus_continuous_holdout_bedroc"])
            for row in qubo_rows
        ),
        "maximum_absolute_holdout_bedroc_gap_vs_continuous": max(
            abs(float(row["quantized_minus_continuous_holdout_bedroc"]))
            for row in qubo_rows
        ),
        "minimum_actual_quality_floor_margin": min(
            float(row["actual_quality_floor_margin"]) for row in qubo_rows
        ),
        "maximum_factorized_energy_residual": maximum_energy_residual,
        "maximum_logical_variable_count": max(
            int(row["qubo_scale"]["logical_variable_count"])
            for row in model_records
        ),
        "maximum_coefficient_dynamic_range": max(
            float(row["qubo_scale"]["coefficient_dynamic_range"])
            for row in model_records
        ),
    }
    gates = config["route_gate"]
    checks = {
        "selected_candidate_exists": bool(selected),
        "selected_minimum_mean_target_gain": bool(selected)
        and float(selected["mean_target_gain_over_pair_off"])
        >= float(gates["minimum_mean_target_gain_over_pair_off"]),
        "selected_minimum_worst_target_gain": bool(selected)
        and float(selected["worst_target_gain_over_pair_off"])
        >= float(gates["minimum_worst_target_gain_over_pair_off"]),
        "selected_all_targets_within_0p01": bool(selected)
        and int(selected["target_count_within_0p01_of_pair_off"])
        >= int(gates["minimum_target_count_within_0p01"]),
        "selected_minimum_redundancy_reduction": bool(selected)
        and float(selected["mean_target_stable_redundancy_reduction"])
        >= float(gates["minimum_mean_redundancy_reduction"]),
        "loto_minimum_mean_gain": loto_gate[
            "mean_held_target_gain_over_pair_off"
        ]
        >= float(gates["minimum_loto_mean_gain"]),
        "loto_minimum_worst_gain": loto_gate[
            "worst_held_target_gain_over_pair_off"
        ]
        >= float(gates["minimum_loto_worst_gain"]),
        "loto_all_targets_within_0p01": loto_gate[
            "held_target_count_within_0p01"
        ]
        >= int(gates["minimum_loto_target_count_within_0p01"]),
        "qubo_minimum_mean_jaccard": fidelity[
            "mean_subset_jaccard_vs_continuous"
        ]
        >= float(gates["minimum_qubo_mean_subset_jaccard"]),
        "qubo_quality_floor_preserved": fidelity[
            "minimum_actual_quality_floor_margin"
        ]
        >= -float(gates["maximum_quality_floor_violation"]),
        "qubo_energy_equivalent": fidelity[
            "maximum_factorized_energy_residual"
        ]
        <= float(gates["maximum_factorized_energy_residual"]),
        "qubo_variable_count_within_limit": fidelity[
            "maximum_logical_variable_count"
        ]
        <= int(gates["maximum_logical_variable_count"]),
    }
    freeze_authorized = all(checks.values())
    output_paths = {
        key: root / str(value) for key, value in config["outputs"].items()
    }
    write_csv(output_paths["fixed_k_metrics_csv"], rows)
    write_csv(output_paths["target_summary_csv"], target_rows)
    write_csv(output_paths["global_summary_csv"], global_rows)
    write_csv(output_paths["loto_summary_csv"], loto_rows)
    write_csv(output_paths["qubo_fidelity_csv"], qubo_rows)
    model = {
        "schema_version": "1.0",
        "algorithm_id": "quality-floor-stable-redundancy-portfolio-qubo-v1",
        "candidate_id": selected_candidate_id,
        "quality_integer_scale": int(qubo["quality_integer_scale"]),
        "cardinality_penalty": float(qubo["cardinality_penalty"]),
        "quality_penalty": float(qubo["quality_penalty"]),
        "model_count": len(model_records),
        "models": model_records,
    }
    write_json(output_paths["model_record_json"], model)
    payload = {
        "selected_candidate": selected,
        "loto_gate": loto_gate,
        "qubo_fidelity": fidelity,
        "route_gate_checks": checks,
    }
    result = {
        "schema_version": "1.0",
        "status": "stage68_quality_plateau_portfolio_qubo_complete",
        "experiment_class": "post-hoc cross-target constrained-objective development",
        "config": descriptor(root, root / "configs/stage68_quality_plateau_portfolio_qubo.json"),
        "implementation": {
            key: descriptor(root, path) for key, path in implementation_paths.items()
        },
        "inputs": {key: descriptor(root, path) for key, path in input_paths.items()},
        "target_input_audits": {
            target_id: {
                "ligand_count": len(loaded[target_id]["ligand_ids"]),
                "receptor_count": len(loaded[target_id]["receptor_ids"]),
                "score_row_count": 3
                * len(loaded[target_id]["ligand_ids"])
                * len(loaded[target_id]["receptor_ids"]),
                "input_descriptors": loaded[target_id]["input_descriptors"],
            }
            for target_id in targets
        },
        "candidate_count": len(multipliers),
        "fixed_k_metric_count": len(rows),
        "milp_certificate_count": milp_count,
        "selected_candidate": selected,
        "loto_gate": loto_gate,
        "qubo_fidelity": fidelity,
        "route_gate": {
            "checks": checks,
            "quality_plateau_qubo_freeze_authorized": freeze_authorized,
        },
        "decision": {
            "same_target_retuning_authorized": False,
            "future_new_target_preregistration_authorized": freeze_authorized,
            "robustness_claim_authorized": False,
            "alternate_partition_probe_passed": False,
            "new_docking_authorized_in_stage68": False,
            "quantum_hardware_authorized": False,
            "next_action": (
                "freeze Stage68 for one genuinely new target and compress coefficients before hardware"
                if freeze_authorized
                else "do not freeze Stage68; retain pair-off as the performance baseline"
            ),
        },
        "data_boundary": {
            "historical_development_targets_read": len(targets),
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "analysis_payload_sha256": canonical_sha256(payload),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    result["outputs"] = {
        key: descriptor(root, path)
        for key, path in output_paths.items()
        if key not in {"result_json", "audit_json", "report_md"}
    }
    write_json(output_paths["result_json"], result)
    output_paths["report_md"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["report_md"].write_text(
        report_text(result), encoding="utf-8", newline="\n"
    )
    result["outputs"].update(
        {
            "report_md": descriptor(root, output_paths["report_md"]),
        }
    )
    write_json(output_paths["result_json"], result)
    return result


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    expected = root / "configs/stage68_quality_plateau_portfolio_qubo.json"
    if config_path != expected.resolve():
        raise ValueError("Stage68 must run from its frozen repository config")
    result_path = root / str(config["outputs"]["result_json"])
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Stage68 result exists: {result_path}")
    result = compute(config, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage68_quality_plateau_portfolio_qubo.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
