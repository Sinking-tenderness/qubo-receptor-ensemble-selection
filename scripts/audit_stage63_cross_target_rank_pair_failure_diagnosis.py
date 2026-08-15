"""Independently audit the Stage63 cross-target mechanism diagnosis outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checked(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage63 output identity differs: {path}")
    if path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage63 output size differs: {path}")
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


def spearman(left: list[float], right: list[float]) -> float:
    x = average_ranks(left)
    y = average_ranks(right)
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    scale_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    scale_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    return numerator / (scale_x * scale_y)


def run(
    config_path: Path,
    result_path: Path,
    root: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    result_path = result_path.resolve()
    config = read_json(config_path)
    result = read_json(result_path)
    if result.get("status") != "stage63_cross_target_rank_pair_failure_diagnosis_complete":
        raise ValueError("Stage63 source analysis did not complete")
    if checked(root, result["config"]).resolve() != config_path:
        raise ValueError("Stage63 result config differs")
    auditor = dict(config["implementation"])["independent_auditor"]
    auditor_path = root / str(auditor["path"])
    if auditor_path.resolve() != Path(__file__).resolve() or sha256(
        auditor_path
    ) != str(auditor["sha256"]).upper():
        raise ValueError("Stage63 auditor identity differs")
    output_paths = {
        key: checked(root, value) for key, value in result["outputs"].items()
    }

    landscape = read_csv(output_paths["fixed_k_landscape_csv"])
    folds = read_csv(output_paths["fold_diagnostics_csv"])
    target_k = read_csv(output_paths["target_k_summary_csv"])
    targets = read_csv(output_paths["target_summary_csv"])
    nested = read_csv(output_paths["ppard_nested_k_diagnostics_csv"])
    solvers = read_csv(output_paths["solver_diagnostics_csv"])
    if len(landscape) != 96 or len(folds) != 16 or len(target_k) != 24:
        raise ValueError("Stage63 primary CSV dimensions differ")
    if len(targets) != 4 or len(nested) != 4 or len(solvers) != 4:
        raise ValueError("Stage63 summary CSV dimensions differ")

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in landscape:
        grouped[(row["target_id"], int(row["outer_fold"]))].append(row)
    if len(grouped) != 16:
        raise ValueError("Stage63 target-fold grid differs")
    recomputed_fold_rows: list[dict[str, Any]] = []
    for (target_id, outer_fold), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["subset_size"]))
        if [int(row["subset_size"]) for row in rows] != list(range(1, 7)):
            raise ValueError("Stage63 fixed-k cell differs")
        train = [float(row["train_qubo_objective"]) for row in rows]
        holdout = [float(row["holdout_robust_bedroc"]) for row in rows]
        training_best_k = min(range(1, 7), key=lambda k: (-train[k - 1], k))
        holdout_best_k = min(range(1, 7), key=lambda k: (-holdout[k - 1], k))
        train_delta = train[1] - train[0]
        holdout_delta = holdout[1] - holdout[0]
        recomputed_fold_rows.append(
            {
                "target_id": target_id,
                "outer_fold": outer_fold,
                "training_best_k": training_best_k,
                "holdout_best_k": holdout_best_k,
                "train_holdout_spearman": spearman(train, holdout),
                "train_delta": train_delta,
                "holdout_delta": holdout_delta,
                "conflict": train_delta > TOLERANCE and holdout_delta < -TOLERANCE,
            }
        )
    observed_folds = {
        (row["target_id"], int(row["outer_fold"])): row for row in folds
    }
    for row in recomputed_fold_rows:
        observed = observed_folds[(row["target_id"], row["outer_fold"])]
        if int(observed["training_best_k"]) != row["training_best_k"]:
            raise ValueError("Stage63 training-best k differs")
        if int(observed["holdout_best_k"]) != row["holdout_best_k"]:
            raise ValueError("Stage63 holdout-best k differs")
        if abs(
            float(observed["train_holdout_spearman"])
            - row["train_holdout_spearman"]
        ) > TOLERANCE:
            raise ValueError("Stage63 fold Spearman differs")
        if abs(
            float(observed["k2_minus_k1_train_objective"])
            - row["train_delta"]
        ) > TOLERANCE:
            raise ValueError("Stage63 k2 train delta differs")
        if abs(
            float(observed["k2_minus_k1_holdout_bedroc"])
            - row["holdout_delta"]
        ) > TOLERANCE:
            raise ValueError("Stage63 k2 holdout delta differs")

    primary = {
        "target_count": len({row["target_id"] for row in landscape}),
        "outer_fold_count": len(recomputed_fold_rows),
        "fixed_k_cell_count": len(landscape),
        "training_objective_best_k2_fold_count": sum(
            row["training_best_k"] == 2 for row in recomputed_fold_rows
        ),
        "holdout_k2_beats_k1_fold_count": sum(
            row["holdout_delta"] > TOLERANCE for row in recomputed_fold_rows
        ),
        "k2_pair_reward_direction_conflict_fold_count": sum(
            row["conflict"] for row in recomputed_fold_rows
        ),
        "mean_k2_minus_k1_train_objective": statistics.fmean(
            row["train_delta"] for row in recomputed_fold_rows
        ),
        "mean_k2_minus_k1_holdout_bedroc": statistics.fmean(
            row["holdout_delta"] for row in recomputed_fold_rows
        ),
        "negative_train_holdout_spearman_fold_count": sum(
            row["train_holdout_spearman"] < 0.0 for row in recomputed_fold_rows
        ),
        "mean_train_holdout_spearman": statistics.fmean(
            row["train_holdout_spearman"] for row in recomputed_fold_rows
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
            float(row["inner_outer_k_curve_spearman"]) for row in nested
        ),
        "ppard_mean_selected_k_outer_regret": statistics.fmean(
            float(row["selected_k_outer_regret"]) for row in nested
        ),
    }
    reported = result["primary_diagnosis"]
    for key, expected in primary.items():
        observed = reported[key]
        if isinstance(expected, float):
            if abs(float(observed) - expected) > TOLERANCE:
                raise ValueError(f"Stage63 primary metric differs: {key}")
        elif int(observed) != expected:
            raise ValueError(f"Stage63 primary count differs: {key}")

    exact_constants = {
        "training_objective_best_k2_fold_count": 16,
        "holdout_k2_beats_k1_fold_count": 2,
        "k2_pair_reward_direction_conflict_fold_count": 14,
        "negative_train_holdout_spearman_fold_count": 15,
        "exact_certified_solver_cell_count": 171,
        "exact_over_strong_positive_gap_count": 0,
        "exact_strong_subset_difference_count": 0,
        "weak_greedy_comparison_cell_count": 162,
        "strong_or_exact_over_weak_greedy_positive_gap_count": 48,
    }
    for key, value in exact_constants.items():
        if int(primary[key]) != value:
            raise ValueError(f"Stage63 frozen diagnostic constant differs: {key}")
    if result["decision"]["same_target_ppard_retuning_authorized"] is not False:
        raise ValueError("Stage63 improperly authorizes PPARD retuning")
    if result["decision"]["fresh_validation_authorized"] is not False:
        raise ValueError("Stage63 improperly authorizes fresh validation")
    if result["decision"]["quantum_hardware_authorized"] is not False:
        raise ValueError("Stage63 improperly authorizes hardware")
    if result["next_objective_design"]["status"] != (
        "design_requirements_only_not_frozen"
    ):
        raise ValueError("Stage63 replacement objective boundary differs")

    audit = {
        "schema_version": "1.0",
        "status": "stage63_cross_target_rank_pair_failure_diagnosis_independent_audit_ok",
        "source_result": {
            "path": result_path.relative_to(root).as_posix(),
            "sha256": sha256(result_path),
            "size_bytes": result_path.stat().st_size,
        },
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256(config_path),
            "size_bytes": config_path.stat().st_size,
        },
        "row_counts": {
            "fixed_k_landscape": len(landscape),
            "fold_diagnostics": len(folds),
            "target_k_summary": len(target_k),
            "target_summary": len(targets),
            "ppard_nested_k_diagnostics": len(nested),
            "solver_diagnostics": len(solvers),
        },
        "all_output_hashes_exact": True,
        "primary_diagnosis_independently_recomputed": True,
        "decision_boundary_exact": True,
        "data_boundary": result["data_boundary"],
        "interpretation_boundary": result["interpretation_boundary"],
    }
    output_path = output_path if output_path.is_absolute() else root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
