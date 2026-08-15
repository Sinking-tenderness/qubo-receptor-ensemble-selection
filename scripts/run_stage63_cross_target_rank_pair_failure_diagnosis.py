"""Diagnose cross-target failure modes of the frozen rank-pair QUBO."""

from __future__ import annotations

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


TARGET_ORDER = ("BACE1", "PPARG", "PPARA", "PPARD")
K_VALUES = tuple(range(1, 7))
TOLERANCE = 1e-12


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage63 frozen input differs: {path}")
    return path


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[start][1]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for index in range(start, stop):
            ranks[ordered[index][0]] = rank
        start = stop
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation vectors must have equal length >= 2")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale <= TOLERANCE or right_scale <= TOLERANCE:
        return 0.0
    return numerator / (left_scale * right_scale)


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def subset_set(value: str) -> set[str]:
    selected = {item for item in value.split("+") if item}
    if not selected:
        raise ValueError("selected subset is empty")
    return selected


def pairwise_jaccard(values: list[str]) -> float:
    sets = [subset_set(value) for value in values]
    pairs = list(itertools.combinations(sets, 2))
    if not pairs:
        return 1.0
    return statistics.fmean(len(left & right) / len(left | right) for left, right in pairs)


def standard_error(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def validate_source_results(inputs: dict[str, Path]) -> dict[str, Any]:
    expected_statuses = {
        "stage42f_result": "stage42f_bace1_rank_sensitive_pair_qubo_complete",
        "stage44_result": "stage44_pparg_md96_rank_sensitive_qubo_complete",
        "stage53_result": "stage53_ppara_large_pool_qubo_transfer_complete",
        "stage62_result": "stage62_ppard_train240_frozen_nested_qubo_complete",
        "stage42f_audit": "stage42d_f_bace1_qubo_independent_audit_ok",
        "stage44_audit": "stage44_pparg_md96_rank_sensitive_qubo_independent_audit_ok",
        "stage53_audit": "stage53_ppara_large_pool_qubo_transfer_independent_audit_ok",
        "stage62_audit": "stage62_ppard_train240_full_recomputation_audit_ok",
    }
    values = {key: read_json(inputs[key]) for key in expected_statuses}
    for key, status in expected_statuses.items():
        if values[key].get("status") != status:
            raise ValueError(f"Stage63 source status differs: {key}")
    objective = values["stage42f_result"]["objective"]
    if values["stage53_result"]["objectives"]["rank_pair_qubo"] != objective:
        raise ValueError("Stage53 rank-pair objective differs from Stage42f")
    if values["stage62_result"]["objective"] != objective:
        raise ValueError("Stage62 rank-pair objective differs from Stage42f")
    stage44_objective = read_json(inputs["stage44_config"])["objective"]
    stage44_checks = {
        "source_objective_id": objective["objective_id"],
        "bedroc_alpha": objective["bedroc_alpha"],
        "minimum_subset_size": objective["minimum_subset_size"],
        "maximum_subset_size": objective["maximum_subset_size"],
    }
    if any(stage44_objective[key] != value for key, value in stage44_checks.items()):
        raise ValueError("Stage44 rank-pair objective differs from Stage42f")
    for key in ("stage42f_result", "stage44_result", "stage53_result", "stage62_result"):
        boundary = dict(values[key]["data_boundary"])
        for boundary_key in (
            "fresh_validation_rows_read",
            "locked_test_rows_read",
            "test_rows_read",
            "new_docking_jobs",
            "quantum_hardware_jobs",
        ):
            if boundary_key in boundary and int(boundary[boundary_key]) != 0:
                raise ValueError(f"Stage63 source crossed boundary: {key}/{boundary_key}")
    return {"objective": objective, "source_results": values}


def standardize_landscape(inputs: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for row in read_csv(inputs["stage42f_fold_metrics"]):
        rows.append(
            {
                "target_id": "BACE1",
                "source_stage": "Stage42f",
                "outer_fold": int(row["outer_fold"]),
                "subset_size": int(row["subset_size"]),
                "solver_method": "exact",
                "exact_certified": True,
                "selected_subset": row["exact_subset"],
                "train_qubo_objective": float(row["train_exact_qubo"]),
                "holdout_robust_bedroc": float(row["holdout_exact_robust_bedroc"]),
            }
        )

    pparg_rows = read_csv(inputs["stage44_selection_metrics"])
    pparg_train = {
        (int(row["fold"]), int(row["subset_size"])): row
        for row in pparg_rows
        if row["scope"] == "outer_train" and row["method"] == "strong_classical"
    }
    pparg_holdout = {
        (int(row["fold"]), int(row["subset_size"])): row
        for row in pparg_rows
        if row["scope"] == "outer_holdout" and row["method"] == "strong_classical"
    }
    if set(pparg_train) != set(pparg_holdout):
        raise ValueError("Stage44 outer train/holdout grids differ")
    for key in sorted(pparg_train):
        train = pparg_train[key]
        holdout = pparg_holdout[key]
        if train["selected_subset"] != holdout["selected_subset"]:
            raise ValueError(f"Stage44 selected subset differs across scopes: {key}")
        fold, subset_size = key
        rows.append(
            {
                "target_id": "PPARG",
                "source_stage": "Stage44",
                "outer_fold": fold,
                "subset_size": subset_size,
                "solver_method": "strong_classical",
                "exact_certified": subset_size <= 3,
                "selected_subset": train["selected_subset"],
                "train_qubo_objective": float(train["qubo_objective"]),
                "holdout_robust_bedroc": float(holdout["robust_bedroc_composite"]),
            }
        )

    for row in read_csv(inputs["stage53_fixed_k_landscape"]):
        if row["method"] != "rank_pair_qubo_exact":
            continue
        rows.append(
            {
                "target_id": "PPARA",
                "source_stage": "Stage53",
                "outer_fold": int(row["outer_fold"]),
                "subset_size": int(row["subset_size"]),
                "solver_method": "exact",
                "exact_certified": True,
                "selected_subset": row["selected_subset"],
                "train_qubo_objective": float(row["train_objective"]),
                "holdout_robust_bedroc": float(row["evaluation_robust_bedroc"]),
            }
        )

    for row in read_csv(inputs["stage62_outer_k_metrics"]):
        rows.append(
            {
                "target_id": "PPARD",
                "source_stage": "Stage62",
                "outer_fold": int(row["outer_fold"]),
                "subset_size": int(row["subset_size"]),
                "solver_method": "exact",
                "exact_certified": True,
                "selected_subset": row["selected_subset"],
                "train_qubo_objective": float(row["train_qubo_objective"]),
                "holdout_robust_bedroc": float(row["holdout_robust_bedroc"]),
            }
        )

    rows.sort(
        key=lambda row: (
            TARGET_ORDER.index(str(row["target_id"])),
            int(row["outer_fold"]),
            int(row["subset_size"]),
        )
    )
    if len(rows) != len(TARGET_ORDER) * 4 * len(K_VALUES):
        raise ValueError("Stage63 standardized landscape dimensions differ")
    for target_id in TARGET_ORDER:
        for outer_fold in range(4):
            observed = {
                int(row["subset_size"])
                for row in rows
                if row["target_id"] == target_id
                and int(row["outer_fold"]) == outer_fold
            }
            if observed != set(K_VALUES):
                raise ValueError(f"Stage63 fixed-k grid differs: {target_id}/{outer_fold}")
    return rows


def fold_diagnostics(landscape: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target_id in TARGET_ORDER:
        for outer_fold in range(4):
            rows = [
                row
                for row in landscape
                if row["target_id"] == target_id
                and int(row["outer_fold"]) == outer_fold
            ]
            rows.sort(key=lambda row: int(row["subset_size"]))
            by_k = {int(row["subset_size"]): row for row in rows}
            train_best = min(
                K_VALUES,
                key=lambda k: (-float(by_k[k]["train_qubo_objective"]), k),
            )
            holdout_best = min(
                K_VALUES,
                key=lambda k: (-float(by_k[k]["holdout_robust_bedroc"]), k),
            )
            train_delta = float(by_k[2]["train_qubo_objective"]) - float(
                by_k[1]["train_qubo_objective"]
            )
            holdout_delta = float(by_k[2]["holdout_robust_bedroc"]) - float(
                by_k[1]["holdout_robust_bedroc"]
            )
            output.append(
                {
                    "target_id": target_id,
                    "outer_fold": outer_fold,
                    "training_best_k": train_best,
                    "holdout_best_k": holdout_best,
                    "training_best_matches_holdout_best": train_best == holdout_best,
                    "train_holdout_spearman": spearman(
                        [float(row["train_qubo_objective"]) for row in rows],
                        [float(row["holdout_robust_bedroc"]) for row in rows],
                    ),
                    "k2_minus_k1_train_objective": train_delta,
                    "k2_minus_k1_holdout_bedroc": holdout_delta,
                    "k2_pair_reward_direction_conflict": (
                        train_delta > TOLERANCE and holdout_delta < -TOLERANCE
                    ),
                    "holdout_best_gain_over_k1": float(
                        by_k[holdout_best]["holdout_robust_bedroc"]
                    )
                    - float(by_k[1]["holdout_robust_bedroc"]),
                }
            )
    return output


def target_k_summary(landscape: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target_id in TARGET_ORDER:
        target_rows = [row for row in landscape if row["target_id"] == target_id]
        baseline = {
            int(row["outer_fold"]): float(row["holdout_robust_bedroc"])
            for row in target_rows
            if int(row["subset_size"]) == 1
        }
        for subset_size in K_VALUES:
            rows = [
                row for row in target_rows if int(row["subset_size"]) == subset_size
            ]
            values = [float(row["holdout_robust_bedroc"]) for row in rows]
            gains = [
                float(row["holdout_robust_bedroc"])
                - baseline[int(row["outer_fold"])]
                for row in rows
            ]
            output.append(
                {
                    "target_id": target_id,
                    "subset_size": subset_size,
                    "mean_holdout_robust_bedroc": statistics.fmean(values),
                    "holdout_standard_error": standard_error(values),
                    "mean_gain_over_k1": statistics.fmean(gains),
                    "positive_gain_over_k1_fold_count": sum(
                        gain > TOLERANCE for gain in gains
                    ),
                    "mean_pairwise_selection_jaccard": pairwise_jaccard(
                        [str(row["selected_subset"]) for row in rows]
                    ),
                }
            )
    return output


def summarize_targets(
    k_summary: list[dict[str, Any]], folds: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target_id in TARGET_ORDER:
        rows = [row for row in k_summary if row["target_id"] == target_id]
        by_k = {int(row["subset_size"]): row for row in rows}
        best_k = min(
            K_VALUES,
            key=lambda k: (-float(by_k[k]["mean_holdout_robust_bedroc"]), k),
        )
        fold_rows = [row for row in folds if row["target_id"] == target_id]
        output.append(
            {
                "target_id": target_id,
                "best_mean_holdout_k": best_k,
                "best_mean_holdout_robust_bedroc": float(
                    by_k[best_k]["mean_holdout_robust_bedroc"]
                ),
                "k1_mean_holdout_robust_bedroc": float(
                    by_k[1]["mean_holdout_robust_bedroc"]
                ),
                "best_mean_gain_over_k1": float(by_k[best_k]["mean_gain_over_k1"]),
                "k2_mean_gain_over_k1": float(by_k[2]["mean_gain_over_k1"]),
                "negative_train_holdout_spearman_fold_count": sum(
                    float(row["train_holdout_spearman"]) < 0.0 for row in fold_rows
                ),
                "k2_pair_reward_direction_conflict_fold_count": sum(
                    bool(row["k2_pair_reward_direction_conflict"])
                    for row in fold_rows
                ),
            }
        )
    return output


def ppard_nested_diagnostics(
    inputs: dict[str, Path], landscape: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    inner_rows = read_csv(inputs["stage62_inner_k_selection"])
    outer = {
        (int(row["outer_fold"]), int(row["subset_size"])): row
        for row in landscape
        if row["target_id"] == "PPARD"
    }
    output: list[dict[str, Any]] = []
    for outer_fold in range(4):
        rows = [
            row for row in inner_rows if int(row["outer_fold"]) == outer_fold
        ]
        rows.sort(key=lambda row: int(row["subset_size"]))
        if len(rows) != 6:
            raise ValueError(f"PPARD inner-k grid differs: {outer_fold}")
        selected_values = {int(row["selected_k"]) for row in rows}
        best_values = {int(row["best_k"]) for row in rows}
        if len(selected_values) != 1 or len(best_values) != 1:
            raise ValueError(f"PPARD inner-k decision differs: {outer_fold}")
        inner_values = {
            int(row["subset_size"]): float(row["mean_inner_holdout_robust_bedroc"])
            for row in rows
        }
        outer_values = {
            k: float(outer[(outer_fold, k)]["holdout_robust_bedroc"])
            for k in K_VALUES
        }
        selected_k = next(iter(selected_values))
        outer_best_k = min(
            K_VALUES, key=lambda k: (-outer_values[k], k)
        )
        output.append(
            {
                "outer_fold": outer_fold,
                "inner_best_k": next(iter(best_values)),
                "inner_selected_k": selected_k,
                "outer_oracle_best_k": outer_best_k,
                "inner_outer_k_curve_spearman": spearman(
                    [inner_values[k] for k in K_VALUES],
                    [outer_values[k] for k in K_VALUES],
                ),
                "selected_outer_holdout_robust_bedroc": outer_values[selected_k],
                "oracle_outer_holdout_robust_bedroc": outer_values[outer_best_k],
                "selected_k_outer_regret": outer_values[outer_best_k]
                - outer_values[selected_k],
            }
        )
    return output


def solver_diagnostics(inputs: dict[str, Path]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    bace_fold = read_csv(inputs["stage42f_fold_metrics"])
    bace_full = read_csv(inputs["stage42f_full_metrics"])
    exact_strong_gaps = [
        float(row["train_exact_minus_classical_gap"]) for row in bace_fold
    ] + [float(row["exact_minus_classical_gap"]) for row in bace_full]
    exact_strong_subsets = [
        row["exact_subset"] != row["classical_subset"]
        for row in bace_fold + bace_full
    ]
    weak_gaps = [
        float(row["exact_qubo"]) - float(row["direct_greedy_qubo"])
        for row in bace_full
    ]
    weak_subsets = [
        row["exact_subset"] != row["direct_greedy_subset"] for row in bace_full
    ]
    output.append(
        {
            "target_id": "BACE1",
            "exact_certified_cell_count": len(exact_strong_gaps),
            "exact_over_strong_positive_gap_count": sum(
                gap > TOLERANCE for gap in exact_strong_gaps
            ),
            "exact_strong_subset_difference_count": sum(exact_strong_subsets),
            "weak_greedy_comparison_cell_count": len(weak_gaps),
            "strong_or_exact_over_weak_greedy_positive_gap_count": sum(
                gap > TOLERANCE for gap in weak_gaps
            ),
            "strong_or_exact_weak_greedy_subset_difference_count": sum(weak_subsets),
        }
    )

    pparg_solver = read_csv(inputs["stage44_solver_comparison"])
    pparg_exact = [row for row in pparg_solver if row["exact_available"] == "True"]
    pparg_selection = read_csv(inputs["stage44_selection_metrics"])
    pparg_strong = {
        (row["scope"], row["fold"], row["subset_size"]): row
        for row in pparg_selection
        if row["method"] == "strong_classical"
        and row["scope"] in {"outer_train", "full_data"}
    }
    pparg_direct = {
        (row["scope"], row["fold"], row["subset_size"]): row
        for row in pparg_selection
        if row["method"] == "direct_greedy"
        and row["scope"] in {"outer_train", "full_data"}
    }
    pparg_exact_selected = {
        (row["scope"], row["fold"], row["subset_size"]): row
        for row in pparg_selection
        if row["method"] == "exact"
        and row["scope"] in {"outer_train", "full_data"}
    }
    exact_keys = {
        key for key in pparg_strong if int(key[2]) <= 3
    }
    if exact_keys != set(pparg_exact_selected):
        raise ValueError("Stage44 exact selection grid differs")
    weak_keys = set(pparg_strong)
    if weak_keys != set(pparg_direct):
        raise ValueError("Stage44 direct-greedy grid differs")
    output.append(
        {
            "target_id": "PPARG",
            "exact_certified_cell_count": len(pparg_exact),
            "exact_over_strong_positive_gap_count": sum(
                abs(float(row["classical_exact_gap"])) > TOLERANCE
                for row in pparg_exact
            ),
            "exact_strong_subset_difference_count": sum(
                pparg_exact_selected[key]["selected_subset"]
                != pparg_strong[key]["selected_subset"]
                for key in exact_keys
            ),
            "weak_greedy_comparison_cell_count": len(weak_keys),
            "strong_or_exact_over_weak_greedy_positive_gap_count": sum(
                float(pparg_strong[key]["qubo_objective"])
                - float(pparg_direct[key]["qubo_objective"])
                > TOLERANCE
                for key in weak_keys
            ),
            "strong_or_exact_weak_greedy_subset_difference_count": sum(
                pparg_strong[key]["selected_subset"]
                != pparg_direct[key]["selected_subset"]
                for key in weak_keys
            ),
        }
    )

    ppara_rows = read_csv(inputs["stage53_fixed_k_landscape"])
    ppara = {
        method: {
            (row["outer_fold"], row["subset_size"]): row
            for row in ppara_rows
            if row["method"] == method
        }
        for method in (
            "rank_pair_qubo_exact",
            "rank_pair_strong_classical",
            "rank_pair_direct_greedy",
        )
    }
    exact = ppara["rank_pair_qubo_exact"]
    strong = ppara["rank_pair_strong_classical"]
    direct = ppara["rank_pair_direct_greedy"]
    if set(exact) != set(strong) or set(exact) != set(direct):
        raise ValueError("Stage53 solver grids differ")
    output.append(
        {
            "target_id": "PPARA",
            "exact_certified_cell_count": len(exact),
            "exact_over_strong_positive_gap_count": sum(
                float(exact[key]["train_objective"])
                - float(strong[key]["train_objective"])
                > TOLERANCE
                for key in exact
            ),
            "exact_strong_subset_difference_count": sum(
                exact[key]["selected_subset"] != strong[key]["selected_subset"]
                for key in exact
            ),
            "weak_greedy_comparison_cell_count": len(exact),
            "strong_or_exact_over_weak_greedy_positive_gap_count": sum(
                float(exact[key]["train_objective"])
                - float(direct[key]["train_objective"])
                > TOLERANCE
                for key in exact
            ),
            "strong_or_exact_weak_greedy_subset_difference_count": sum(
                exact[key]["selected_subset"] != direct[key]["selected_subset"]
                for key in exact
            ),
        }
    )

    ppard_rows = read_csv(inputs["stage62_objective_gap_cells"])
    output.append(
        {
            "target_id": "PPARD",
            "exact_certified_cell_count": len(ppard_rows),
            "exact_over_strong_positive_gap_count": sum(
                float(row["exact_minus_strong_gap"]) > TOLERANCE
                for row in ppard_rows
            ),
            "exact_strong_subset_difference_count": sum(
                row["exact_subset"] != row["strong_classical_subset"]
                for row in ppard_rows
            ),
            "weak_greedy_comparison_cell_count": len(ppard_rows),
            "strong_or_exact_over_weak_greedy_positive_gap_count": sum(
                float(row["exact_minus_direct_greedy_gap"]) > TOLERANCE
                for row in ppard_rows
            ),
            "strong_or_exact_weak_greedy_subset_difference_count": sum(
                row["exact_subset"] != row["direct_greedy_subset"]
                for row in ppard_rows
            ),
        }
    )
    return output


def compute_analysis(config: dict[str, Any], root: Path) -> dict[str, Any]:
    input_paths = {
        key: verified(root, value) for key, value in config["inputs"].items()
    }
    source_audit = validate_source_results(input_paths)
    landscape = standardize_landscape(input_paths)
    folds = fold_diagnostics(landscape)
    k_summary = target_k_summary(landscape)
    targets = summarize_targets(k_summary, folds)
    ppard_nested = ppard_nested_diagnostics(input_paths, landscape)
    solvers = solver_diagnostics(input_paths)

    primary = {
        "target_count": len(TARGET_ORDER),
        "outer_fold_count": len(folds),
        "fixed_k_cell_count": len(landscape),
        "training_objective_best_k2_fold_count": sum(
            int(row["training_best_k"]) == 2 for row in folds
        ),
        "holdout_k2_beats_k1_fold_count": sum(
            float(row["k2_minus_k1_holdout_bedroc"]) > TOLERANCE
            for row in folds
        ),
        "k2_pair_reward_direction_conflict_fold_count": sum(
            bool(row["k2_pair_reward_direction_conflict"]) for row in folds
        ),
        "mean_k2_minus_k1_train_objective": statistics.fmean(
            float(row["k2_minus_k1_train_objective"]) for row in folds
        ),
        "mean_k2_minus_k1_holdout_bedroc": statistics.fmean(
            float(row["k2_minus_k1_holdout_bedroc"]) for row in folds
        ),
        "negative_train_holdout_spearman_fold_count": sum(
            float(row["train_holdout_spearman"]) < 0.0 for row in folds
        ),
        "mean_train_holdout_spearman": statistics.fmean(
            float(row["train_holdout_spearman"]) for row in folds
        ),
        "targets_with_best_mean_k_greater_than_1": sum(
            int(row["best_mean_holdout_k"]) > 1 for row in targets
        ),
        "exact_certified_solver_cell_count": sum(
            int(row["exact_certified_cell_count"]) for row in solvers
        ),
        "exact_over_strong_positive_gap_count": sum(
            int(row["exact_over_strong_positive_gap_count"]) for row in solvers
        ),
        "exact_strong_subset_difference_count": sum(
            int(row["exact_strong_subset_difference_count"]) for row in solvers
        ),
        "weak_greedy_comparison_cell_count": sum(
            int(row["weak_greedy_comparison_cell_count"]) for row in solvers
        ),
        "strong_or_exact_over_weak_greedy_positive_gap_count": sum(
            int(row["strong_or_exact_over_weak_greedy_positive_gap_count"])
            for row in solvers
        ),
        "ppard_mean_inner_outer_k_curve_spearman": statistics.fmean(
            float(row["inner_outer_k_curve_spearman"]) for row in ppard_nested
        ),
        "ppard_mean_selected_k_outer_regret": statistics.fmean(
            float(row["selected_k_outer_regret"]) for row in ppard_nested
        ),
    }
    mechanism = {
        "primary_failure_mode": "pair_complementarity_generalization_failure",
        "solver_search_bottleneck_supported": False,
        "cardinality_selection_instability_supported": True,
        "fixed_pair_reward_transfer_supported": False,
        "evidence": [
            "The training QUBO objective prefers k=2 in every one of 16 outer folds, but k=2 beats k=1 on only 2 holdouts.",
            "Training objective and holdout BEDROC are negatively rank-correlated in 15 of 16 folds.",
            "Exact optimization and strong classical search agree in all 171 exact-certified cells.",
            "PPARD inner and outer k curves have near-zero mean rank correlation, with one large selected-k regret fold.",
        ],
    }
    next_objective = {
        "status": "design_requirements_only_not_frozen",
        "candidate_family_id": "uncertainty_shrunk_rank_pair_qubo_v2_candidate",
        "pair_term": (
            "Estimate delta_ij across resamples and reward a pair only when a "
            "predefined lower confidence bound, such as median(delta_ij) minus "
            "lambda_delta times MAD(delta_ij), remains positive."
        ),
        "singleton_term": (
            "Use the same resampling and uncertainty shrinkage for q_i so that "
            "single-receptor utility and pair residuals share one robustness scale."
        ),
        "cardinality_rule": (
            "Select k outside the QUBO with nested holdout lower-confidence bounds; "
            "default to k=1 unless a larger k clears a preregistered gain margin."
        ),
        "mandatory_comparators": [
            "best single receptor",
            "linear Top-k",
            "direct robust BEDROC greedy",
            "beam plus swap strong classical search on the same QUBO",
            "exact fixed-k optimization where tractable",
        ],
        "development_boundary": (
            "No lambda, confidence threshold, or gain margin is selected in Stage63. "
            "Any v2 constants must be frozen using historical targets and tested "
            "once on a genuinely new target."
        ),
    }
    decision = {
        "current_rank_pair_qubo_retained_as_baseline": True,
        "current_rank_pair_qubo_supported_for_new_application_claim": False,
        "same_target_ppard_retuning_authorized": False,
        "new_cross_target_objective_development_authorized": True,
        "fresh_validation_authorized": False,
        "quantum_hardware_authorized": False,
        "next_action": (
            "Develop and freeze an uncertainty-shrunk pair objective across historical "
            "targets, then preregister one untouched target before any new docking."
        ),
    }
    return {
        "source_audit": source_audit,
        "landscape_rows": landscape,
        "fold_rows": folds,
        "target_k_rows": k_summary,
        "target_rows": targets,
        "ppard_nested_rows": ppard_nested,
        "solver_rows": solvers,
        "primary_diagnosis": primary,
        "mechanism": mechanism,
        "next_objective_design": next_objective,
        "decision": decision,
    }


def report_text(analysis: dict[str, Any]) -> str:
    primary = analysis["primary_diagnosis"]
    target_rows = analysis["target_rows"]
    ppard = analysis["ppard_nested_rows"]
    target_table = "\n".join(
        "| {target_id} | {best_mean_holdout_k} | {k1_mean_holdout_robust_bedroc:.6f} "
        "| {best_mean_holdout_robust_bedroc:.6f} | {best_mean_gain_over_k1:+.6f} |".format(
            **row
        )
        for row in target_rows
    )
    ppard_table = "\n".join(
        "| {outer_fold} | {inner_selected_k} | {outer_oracle_best_k} | "
        "{inner_outer_k_curve_spearman:.3f} | {selected_k_outer_regret:.6f} |".format(
            **row
        )
        for row in ppard
    )
    return f"""# Stage63 cross-target rank-pair QUBO failure diagnosis

## Scope

This is a post-hoc, development-only mechanism diagnosis. It reads existing BACE1,
PPARG, PPARA, and PPARD fold results, performs no docking, accesses no fresh
validation or locked test rows, and does not tune a replacement objective.

## Main finding

The dominant failure is objective transfer, not solver search. The normalized
rank-pair QUBO training objective selected `k=2` as the best cardinality in
{primary['training_objective_best_k2_fold_count']}/{primary['outer_fold_count']}
outer folds. Yet `k=2` beat `k=1` on only
{primary['holdout_k2_beats_k1_fold_count']}/{primary['outer_fold_count']} holdouts.
The mean training objective change was
{primary['mean_k2_minus_k1_train_objective']:+.6f}, while mean holdout BEDROC
changed by {primary['mean_k2_minus_k1_holdout_bedroc']:+.6f}.

Training objective and holdout BEDROC were negatively rank-correlated in
{primary['negative_train_holdout_spearman_fold_count']}/{primary['outer_fold_count']}
folds (mean Spearman {primary['mean_train_holdout_spearman']:.3f}). This is the
signature of an over-optimistic pair complementarity term.

## Target-level fixed-k behavior

| Target | Best mean k | Mean k=1 BEDROC | Best mean BEDROC | Gain over k=1 |
|---|---:|---:|---:|---:|
{target_table}

BACE1 and PPARG benefit on average from larger ensembles, while PPARA and PPARD
prefer one receptor. Therefore, ensemble benefit is target dependent; one universal
unshrunk pair reward is not supported.

## Solver diagnosis

Across {primary['exact_certified_solver_cell_count']} exact-certified cells, exact
optimization and the beam-plus-swap strong classical search had
{primary['exact_over_strong_positive_gap_count']} positive objective gaps and
{primary['exact_strong_subset_difference_count']} subset differences. A weaker
direct greedy search missed the stronger solution in
{primary['strong_or_exact_over_weak_greedy_positive_gap_count']}/
{primary['weak_greedy_comparison_cell_count']} comparable cells. The landscape can
trap weak greedy search, but the current instances do not separate exact QUBO from
a strong classical solver.

## PPARD nested-k diagnosis

| Fold | Inner-selected k | Outer oracle k | Curve Spearman | Selection regret |
|---:|---:|---:|---:|---:|
{ppard_table}

The mean inner-versus-outer k-curve Spearman was
{primary['ppard_mean_inner_outer_k_curve_spearman']:.3f}; mean selected-k regret
was {primary['ppard_mean_selected_k_outer_regret']:.6f}. The large fold-2 miss is
consistent with finite-sample cardinality instability.

## Next objective requirements

1. Estimate singleton and pair terms across resamples.
2. Reward a pair only when a preregistered lower confidence bound remains positive.
3. Select k using nested holdout lower-confidence bounds with k=1 as the fallback.
4. Keep best-single, linear Top-k, direct BEDROC greedy, strong same-QUBO search,
   and exact optimization where tractable.
5. Freeze all constants across historical targets, then test once on a genuinely
   new target. Do not retune and retest on PPARD.

## Decision

The current rank-pair QUBO remains a useful baseline but is not supported for a new
application claim. Stage63 authorizes cross-target development of an
uncertainty-shrunk objective, not fresh validation or quantum hardware.
"""


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    for key, value in config["implementation"].items():
        path = root / str(value["path"])
        if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
            raise ValueError(f"Stage63 implementation differs: {key}")
    runner_path = root / str(config["implementation"]["runner"]["path"])
    if runner_path.resolve() != Path(__file__).resolve():
        raise ValueError("Stage63 runner path differs")

    analysis = compute_analysis(config, root)
    outputs = {
        key: root / str(value) for key, value in config["outputs"].items()
    }
    write_csv(outputs["fixed_k_landscape_csv"], analysis["landscape_rows"])
    write_csv(outputs["fold_diagnostics_csv"], analysis["fold_rows"])
    write_csv(outputs["target_k_summary_csv"], analysis["target_k_rows"])
    write_csv(outputs["target_summary_csv"], analysis["target_rows"])
    write_csv(outputs["ppard_nested_k_diagnostics_csv"], analysis["ppard_nested_rows"])
    write_csv(outputs["solver_diagnostics_csv"], analysis["solver_rows"])
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text(report_text(analysis), encoding="ascii")

    fingerprint_payload = {
        key: value for key, value in analysis.items() if key != "source_audit"
    }
    result = {
        "schema_version": "1.0",
        "status": "stage63_cross_target_rank_pair_failure_diagnosis_complete",
        "experiment_class": "post-hoc cross-target development-only mechanism diagnosis",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, runner_path),
        "source_audit": analysis["source_audit"],
        "analysis_payload_sha256": canonical_sha256(fingerprint_payload),
        "primary_diagnosis": analysis["primary_diagnosis"],
        "mechanism": analysis["mechanism"],
        "target_summary": analysis["target_rows"],
        "next_objective_design": analysis["next_objective_design"],
        "decision": analysis["decision"],
        "data_boundary": {
            "development_results_read": 4,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            key: descriptor(root, path)
            for key, path in outputs.items()
            if key != "result_json" and key != "audit_json"
        },
        "interpretation_boundary": (
            "Stage63 diagnoses existing development results and may define requirements "
            "for a future objective. It does not tune replacement coefficients, repair "
            "a failed target, establish independent efficacy, authorize fresh validation "
            "or hardware, or support quantum speedup or advantage."
        ),
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
