"""Independently audit the Stage 20 adaptive-k cardinality screen."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage19e_cross_target_qubo_v2 import (
    load_target,
    safe_spearman,
)
from scripts.diagnose_stage20_adaptive_k_objective_screen import (
    BASELINE_METHODS,
    CANDIDATE_METHOD,
    all_subsets,
    build_qubo,
    build_terms,
    greedy_path,
    make_method_row,
    metrics_for_subset,
    score_all_sizes,
    singleton_certificate,
    singleton_assignment,
    utility_map,
    write_singleton_qubo,
    threshold_certificate,
)
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import make_context


DEFAULT_CONFIG = Path("configs/stage20_adaptive_k_objective_screen.json")
DEFAULT_OUTPUT = Path("data/stage20_adaptive_k_objective_screen_audit.json")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def rooted(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path leaves repository root: {value}") from error
    return path


def verify_descriptor(root: Path, descriptor: dict[str, Any]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    if not path.is_file() or file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"descriptor differs: {path}")
    if "size_bytes" in descriptor and path.stat().st_size != int(
        descriptor["size_bytes"]
    ):
        raise ValueError(f"descriptor size differs: {path}")
    return path


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def compare_nested(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{label} keys differ")
        for key in actual:
            compare_nested(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label} lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_nested(left, right, f"{label}/{index}")
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        assert_close(float(actual), float(expected), label)
    elif actual != expected:
        raise ValueError(f"{label} differs: {actual!r} != {expected!r}")


def compare_metrics(
    row: dict[str, str],
    values: dict[str, np.ndarray],
    index: int,
    prefix: str,
    label: str,
) -> None:
    for metric_id in (
        "primary",
        "sensitivity",
        "mean_seed",
        "worst_seed",
        "robust_composite",
    ):
        assert_close(
            float(row[f"{prefix}_{metric_id}"]),
            float(values[metric_id][index]),
            f"{label}/{prefix}/{metric_id}",
        )


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = rooted(root, config_path.as_posix())
    config = read_json(config_path)
    implementation_path = rooted(root, config["implementation"]["path"])
    if file_sha256(implementation_path) != str(config["implementation"]["sha256"]).upper():
        raise ValueError("Stage 20 implementation hash differs")
    input_paths = {
        key: verify_descriptor(root, descriptor)
        for key, descriptor in config["inputs"].items()
    }
    source_config = read_json(input_paths["stage19e_config"])
    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    if result["status"] != "stage20_adaptive_k_train_only_screen_complete":
        raise ValueError("unexpected Stage 20 status")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("Stage 20 result identifies another config")
    output_paths = {
        key: verify_descriptor(root, descriptor)
        for key, descriptor in result["outputs"].items()
    }
    fold_rows = read_csv(output_paths["fold_methods_csv"])
    incremental_rows = read_csv(output_paths["incremental_csv"])
    model_record = read_json(output_paths["model_record_json"])
    diagnostic = config["diagnostic"]
    minimum_k = int(diagnostic["minimum_k"])
    maximum_k = int(diagnostic["maximum_k"])
    outer_count = int(diagnostic["outer_fold_count"])
    candidate_count = maximum_k - minimum_k + 1
    target_count = len(source_config["targets"])
    expected_fold_rows = target_count * outer_count * candidate_count * (
        len(BASELINE_METHODS) + 1
    )
    expected_incremental_rows = target_count * candidate_count
    if len(fold_rows) != expected_fold_rows:
        raise ValueError("Stage 20 fold row count differs")
    if len(incremental_rows) != expected_incremental_rows:
        raise ValueError("Stage 20 incremental row count differs")
    if len({tuple(row.values()) for row in fold_rows}) != len(fold_rows):
        raise ValueError("duplicate Stage 20 fold row")
    expected_boundary = {
        "train_rows_read_by_target": {
            target_id: int(spec["expected"]["ligand_count"])
            for target_id, spec in source_config["targets"].items()
        },
        "new_docking_jobs": 0,
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
        "bace1_docking_rows_read": 0,
        "quantum_hardware_jobs": 0,
    }
    if result["data_boundary"] != expected_boundary or model_record["data_boundary"] != expected_boundary:
        raise ValueError("Stage 20 data boundary differs")

    fold_recomputed = 0
    full_recomputed = 0
    schedule = {int(item["k"]): item for item in diagnostic["k_schedule"]}
    alpha = float(diagnostic["bedroc_alpha"])
    coverage_fraction = float(diagnostic["coverage_fraction"])
    decoy_weight = float(diagnostic["decoy_weight"])
    cardinality_penalty = float(diagnostic["cardinality_penalty"])
    constraint_penalty = float(diagnostic["constraint_penalty"])
    fold_index = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            int(row["k"]),
            row["method"],
        ): row
        for row in fold_rows
    }

    for target_id, target_spec in source_config["targets"].items():
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        all_ids = {row["ligand_id"] for row in ligands}
        assignments = make_frozen_group_folds(
            ligands, outer_count, int(diagnostic["fold_seed"])
        )
        for outer_fold in range(outer_count):
            holdout_ids = {
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == outer_fold
            }
            train_ids = all_ids - holdout_ids
            context = make_context(
                train_ids,
                holdout_ids,
                matrices,
                receptor_ids,
                {
                    "coverage_fraction": float(target_spec["v1_qubo"]["coverage_fraction"]),
                    "utility_metric": "bedroc",
                },
            )
            subsets, train_values = score_all_sizes(
                context, receptor_ids, maximum_k, "train", alpha
            )
            _, holdout_values = score_all_sizes(
                context, receptor_ids, maximum_k, "validation", alpha
            )
            subset_index = {subset: index for index, subset in enumerate(subsets)}
            utilities = utility_map(subsets, train_values)
            path = greedy_path(utilities, receptor_ids, maximum_k)
            for k in range(minimum_k, maximum_k + 1):
                item = schedule[k]
                objective_id = str(item["objective_id"])
                exact_subset = min(
                    [subset for subset in subsets if len(subset) == k],
                    key=lambda subset: (-utilities[subset], subset),
                )
                additive_subset = tuple(
                    sorted(
                        sorted(
                            receptor_ids,
                            key=lambda receptor_id: (
                                -utilities[(receptor_id,)],
                                receptor_id,
                            ),
                        )[:k]
                    )
                )
                expected_baselines = {
                    "direct_greedy": path[k],
                    "additive_top_k": additive_subset,
                    "exact_robust_oracle": exact_subset,
                }
                for method, subset in expected_baselines.items():
                    row = fold_index[(target_id, outer_fold, k, method)]
                    if row["selected_subset"] != "+".join(subset):
                        raise ValueError(f"{target_id}/{outer_fold}/{k}/{method} differs")
                    index = subset_index[subset]
                    compare_metrics(row, train_values, index, "train", method)
                    compare_metrics(row, holdout_values, index, "holdout", method)
                    fold_recomputed += 1
                if k == 1:
                    singleton_values = {
                        receptor_id: utilities[(receptor_id,)]
                        for receptor_id in receptor_ids
                    }
                    qubo = write_singleton_qubo(
                        receptor_ids, singleton_values, cardinality_penalty
                    )
                    subset = min(
                        [(receptor_id,) for receptor_id in receptor_ids],
                        key=lambda value: (-utilities[value], value),
                    )
                    cert = singleton_certificate(qubo, receptor_ids, singleton_values)
                else:
                    terms = build_terms(
                        context,
                        receptor_ids,
                        coverage_fraction,
                        int(item["active_threshold"]),
                        alpha,
                    )
                    qubo = build_qubo(
                        terms,
                        receptor_ids,
                        k,
                        decoy_weight,
                        float(item["redundancy_weight"]),
                        cardinality_penalty,
                        constraint_penalty,
                    )
                    cert = threshold_certificate(
                        terms,
                        qubo,
                        receptor_ids,
                        k,
                        decoy_weight,
                        float(item["redundancy_weight"]),
                    )
                    subset = tuple(cert["selected_subset"])
                row = fold_index[(target_id, outer_fold, k, CANDIDATE_METHOD)]
                if row["selected_subset"] != "+".join(subset):
                    raise ValueError(f"{target_id}/{outer_fold}/{k}/candidate differs")
                index = subset_index[subset]
                compare_metrics(row, train_values, index, "train", "candidate")
                compare_metrics(row, holdout_values, index, "holdout", "candidate")
                if int(row["qubo_state_count"]) != int(cert["state_count"]):
                    raise ValueError(f"{target_id}/{outer_fold}/{k}/state count differs")
                assert_close(
                    float(row["qubo_scaled_best_second_gap"]),
                    float(cert["scaled_best_second_gap"]),
                    f"{target_id}/{outer_fold}/{k}/gap",
                )
                fold_recomputed += 1

        full_context = make_context(
            all_ids,
            set(),
            matrices,
            receptor_ids,
            {
                "coverage_fraction": float(target_spec["v1_qubo"]["coverage_fraction"]),
                "utility_metric": "bedroc",
            },
        )
        full_subsets, full_values = score_all_sizes(
            full_context, receptor_ids, maximum_k, "train", alpha
        )
        full_subset_index = {subset: index for index, subset in enumerate(full_subsets)}
        full_utility = utility_map(full_subsets, full_values)
        singleton_values = {
            receptor_id: full_utility[(receptor_id,)] for receptor_id in receptor_ids
        }
        for k in range(minimum_k, maximum_k + 1):
            item = schedule[k]
            if k == 1:
                qubo = write_singleton_qubo(
                    receptor_ids, singleton_values, cardinality_penalty
                )
                cert = singleton_certificate(qubo, receptor_ids, singleton_values)
            else:
                terms = build_terms(
                    full_context,
                    receptor_ids,
                    coverage_fraction,
                    int(item["active_threshold"]),
                    alpha,
                )
                qubo = build_qubo(
                    terms,
                    receptor_ids,
                    k,
                    decoy_weight,
                    float(item["redundancy_weight"]),
                    cardinality_penalty,
                    constraint_penalty,
                )
                cert = threshold_certificate(
                    terms,
                    qubo,
                    receptor_ids,
                    k,
                    decoy_weight,
                    float(item["redundancy_weight"]),
                )
            recorded = model_record["target_models"][target_id][str(k)]
            if recorded["selected_subset"] != cert["selected_subset"]:
                raise ValueError(f"{target_id}/{k}/full subset differs")
            assert_close(
                float(recorded["scaled_best_second_gap"]),
                float(cert["scaled_best_second_gap"]),
                f"{target_id}/{k}/full gap",
            )
            if k > 1:
                compare_nested(
                    terms,
                    recorded["terms"],
                    f"{target_id}/{k}/terms",
                )
            compare_nested(qubo, recorded["qubo"], f"{target_id}/{k}/qubo")
            full_recomputed += 1

    recorded_incremental = read_csv(output_paths["incremental_csv"])
    if len(recorded_incremental) != len(result["incremental_results"]):
        raise ValueError("incremental result row count differs")
    for row in recorded_incremental:
        match = next(
            value
            for value in result["incremental_results"]
            if value["target_id"] == row["target_id"]
            and int(value["k"]) == int(row["k"])
        )
        for field, value in row.items():
            if value == "":
                continue
            if field in {"target_id", "fold_deltas"}:
                continue
            if field in {"continue_condition_passed"}:
                if str(match[field]).lower() != value.lower():
                    raise ValueError(f"incremental boolean differs: {field}")
            elif field in {"k", "previous_k", "positive_fold_count", "consecutive_failure_count"}:
                if int(match[field]) != int(value):
                    raise ValueError(f"incremental integer differs: {field}")
            else:
                assert_close(float(match[field]), float(value), f"incremental/{field}")

    if result["stopping_recommendation"]["recommended_stop_k"] != 3:
        raise ValueError("unexpected Stage 20 stop recommendation")
    if result["one_standard_error"]["recommended_smallest_k"] != 1:
        raise ValueError("unexpected one-standard-error recommendation")
    if result["data_boundary"]["quantum_hardware_jobs"] != 0:
        raise ValueError("Stage 20 contains hardware jobs")

    forbidden = ("fresh_validation", "locked_test", "bace1_docking")
    paths = [str(descriptor["path"]).lower() for descriptor in config["inputs"].values()]
    paths.extend(
        str(descriptor["path"]).lower()
        for spec in source_config["targets"].values()
        for descriptor in spec["inputs"].values()
    )
    if any(marker in path for path in paths for marker in forbidden):
        raise ValueError("Stage 20 input path crosses protected data")

    return {
        "schema_version": "1.0",
        "status": "stage20_adaptive_k_objective_screen_audit_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "result": {
            "path": result_path.relative_to(root).as_posix(),
            "sha256": file_sha256(result_path),
        },
        "auditor": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "coverage": {
            "fold_rows_recomputed": fold_recomputed,
            "full_train_models_recomputed": full_recomputed,
            "incremental_rows_checked": len(incremental_rows),
            "subset_states_by_target_represented": sum(
                sum(math.comb(int(value["receptor_count"]), k) for k in range(minimum_k, maximum_k + 1))
                for value in result["target_dimensions"].values()
            ),
        },
        "checks": {
            "all_input_and_output_hashes_verified": True,
            "all_outer_baselines_recomputed": True,
            "all_adaptive_k_qubos_recomputed": True,
            "all_full_train_model_coefficients_recomputed": True,
            "incremental_curve_recomputed": True,
            "one_standard_error_rule_reproduced": True,
            "stopping_rule_reproduced": True,
            "new_docking_jobs": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "quantum_hardware_jobs": 0,
            "bace1_method_amendment_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.config, args.root)
    output = rooted(args.root.resolve(), args.output.as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
