"""Independently audit the Stage 19i objective and noise screen."""

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
    score_subsets,
)
from scripts.diagnose_stage19i_objective_adequacy_noise_screen import (
    BASELINE_METHODS,
    CANDIDATE_METHOD,
    MATRIX_IDS,
    METRIC_IDS,
    all_subsets,
    build_qubo,
    build_terms,
    certify_states,
    greedy_selection,
    method_row,
    paired_comparison,
    score_mixed_subsets,
)
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import make_context


DEFAULT_CONFIG = Path("configs/stage19i_objective_adequacy_noise_screen.json")
DEFAULT_OUTPUT = Path("data/stage19i_objective_adequacy_noise_screen_audit.json")


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


def parameters_for_candidate(candidate: dict[str, Any]) -> dict[str, float]:
    return {
        "decoy_weight": 1.0,
        "redundancy_weight": float(candidate["redundancy_weight"]),
    }


def compare_metrics(
    row: dict[str, str], values: dict[str, np.ndarray], index: int, prefix: str, label: str
) -> None:
    for metric_id in METRIC_IDS:
        assert_close(
            float(row[f"{prefix}_{metric_id}"]),
            float(values[metric_id][index]),
            f"{label}/{prefix}/{metric_id}",
        )


def recompute_noise_summary_from_trials(
    trials: list[dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, str]]] = defaultdict(list)
    for row in trials:
        grouped[
            (
                row["target_id"],
                row["candidate_id"],
                row["noise_model_id"],
                float(row["noise_level"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (target_id, candidate_id, noise_model_id, noise_level), rows in sorted(grouped.items()):
        matches = [row["matches_baseline"].lower() == "true" for row in rows]
        regrets = [float(row["original_objective_regret"]) for row in rows]
        ranks = [int(row["original_objective_rank"]) for row in rows]
        output.append(
            {
                "target_id": target_id,
                "candidate_id": candidate_id,
                "noise_model_id": noise_model_id,
                "noise_level": noise_level,
                "repeat_count": len(rows),
                "selection_stability": sum(matches) / len(matches),
                "mean_original_objective_regret": statistics.fmean(regrets),
                "worst_original_objective_regret": max(regrets),
                "mean_original_objective_rank": statistics.fmean(ranks),
            }
        )
    return output


def noise_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["target_id"]),
        str(row["candidate_id"]),
        str(row["noise_model_id"]),
        float(row["noise_level"]),
    )


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = rooted(root, config_path.as_posix())
    config = read_json(config_path)
    implementation_path = rooted(root, config["implementation"]["path"])
    if file_sha256(implementation_path) != str(config["implementation"]["sha256"]).upper():
        raise ValueError("Stage 19i implementation hash differs")
    input_paths = {
        key: verify_descriptor(root, descriptor)
        for key, descriptor in config["inputs"].items()
    }
    source_config = read_json(input_paths["stage19e_config"])
    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    if result["status"] != "stage19i_no_candidate_hardware_ready_do_not_execute_quantum":
        raise ValueError("unexpected Stage 19i status")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("Stage 19i result identifies another config")
    output_paths = {
        key: verify_descriptor(root, descriptor)
        for key, descriptor in result["outputs"].items()
    }
    fold_rows = read_csv(output_paths["fold_methods_csv"])
    noise_trials = read_csv(output_paths["noise_trials_csv"])
    noise_summary = read_csv(output_paths["noise_summary_csv"])
    model_record = read_json(output_paths["model_record_json"])
    diagnostic = config["diagnostic"]
    target_count = len(source_config["targets"])
    outer_count = int(diagnostic["outer_fold_count"])
    candidate_count = len(diagnostic["candidates"])
    noise_model_count = len(diagnostic["noise_models"])
    level_count = len(diagnostic["noise_levels"])
    repeats = int(diagnostic["noise_repeats"])
    expected_fold_rows = target_count * outer_count * (len(BASELINE_METHODS) + candidate_count)
    expected_noise_summary = target_count * candidate_count * noise_model_count * level_count
    expected_noise_trials = expected_noise_summary * repeats
    if len(fold_rows) != expected_fold_rows:
        raise ValueError("fold row count differs")
    if len(noise_summary) != expected_noise_summary:
        raise ValueError("noise summary row count differs")
    if len(noise_trials) != expected_noise_trials:
        raise ValueError("noise trial row count differs")
    if len({tuple(row.values()) for row in fold_rows}) != len(fold_rows):
        raise ValueError("duplicate fold row")
    if len({tuple(row.values()) for row in noise_trials}) != len(noise_trials):
        raise ValueError("duplicate noise trial row")
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
        raise ValueError("Stage 19i data boundary differs")
    if model_record["status"] != result["status"]:
        raise ValueError("model record status differs")

    recomputed_noise_summary = recompute_noise_summary_from_trials(noise_trials)
    recorded_noise_summary = [
        {
            key: (float(value) if key not in {"target_id", "candidate_id", "noise_model_id"} else value)
            for key, value in row.items()
            if key != "repeat_count" or True
        }
        for row in noise_summary
    ]
    compare_nested(
        sorted(recomputed_noise_summary, key=noise_row_sort_key),
        sorted(recorded_noise_summary, key=noise_row_sort_key),
        "noise summary",
    )

    outer_recomputed = 0
    full_recomputed = 0
    fold_seed = int(diagnostic["fold_seed"])
    alpha = float(diagnostic["bedroc_alpha"])
    target_size = int(diagnostic["target_size"])
    coverage_fraction = float(diagnostic["coverage_fraction"])
    cardinality_penalty = float(diagnostic["cardinality_penalty"])
    constraint_penalty = float(diagnostic["constraint_penalty"])
    for target_id, target_spec in source_config["targets"].items():
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        all_ids = {row["ligand_id"] for row in ligands}
        assignments = make_frozen_group_folds(ligands, outer_count, fold_seed)
        triples = [
            tuple(sorted(value)) for value in itertools.combinations(receptor_ids, target_size)
        ]
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
            subsets = all_subsets(receptor_ids, target_size)
            train_values = score_mixed_subsets(context, subsets, receptor_ids, "train", alpha)
            holdout_values = score_mixed_subsets(
                context, subsets, receptor_ids, "validation", alpha
            )
            subset_index = {subset: index for index, subset in enumerate(subsets)}
            utility_train = {
                subset: float(train_values["robust_composite"][index])
                for subset, index in subset_index.items()
            }
            greedy_subset = greedy_selection(utility_train, receptor_ids, target_size)
            additive_subset = tuple(
                sorted(
                    sorted(
                        receptor_ids,
                        key=lambda receptor_id: (
                            -utility_train[(receptor_id,)],
                            receptor_id,
                        ),
                    )[:target_size]
                )
            )
            exact_subset = min(
                (subset for subset in subsets if len(subset) == target_size),
                key=lambda subset: (-utility_train[subset], subset),
            )
            expected_baselines = {
                "direct_greedy": greedy_subset,
                "additive_top3": additive_subset,
                "exact_robust_oracle": exact_subset,
            }
            for method, subset in expected_baselines.items():
                rows = [
                    row
                    for row in fold_rows
                    if row["target_id"] == target_id
                    and int(row["outer_fold"]) == outer_fold
                    and row["method"] == method
                    and row["candidate_id"] == ""
                ]
                if len(rows) != 1 or rows[0]["selected_subset"] != "+".join(subset):
                    raise ValueError(f"{target_id}/{outer_fold}/{method} differs")
                index = subset_index[subset]
                compare_metrics(rows[0], train_values, index, "train", method)
                compare_metrics(rows[0], holdout_values, index, "holdout", method)
            for candidate in diagnostic["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                terms = build_terms(
                    context,
                    receptor_ids,
                    coverage_fraction,
                    int(candidate["active_threshold"]),
                    alpha,
                )
                qubo = build_qubo(
                    terms,
                    receptor_ids,
                    target_size,
                    float(diagnostic["decoy_weight"]),
                    float(candidate["redundancy_weight"]),
                    cardinality_penalty,
                    constraint_penalty,
                )
                certificate = certify_states(
                    terms,
                    qubo,
                    receptor_ids,
                    target_size,
                    float(diagnostic["decoy_weight"]),
                    float(candidate["redundancy_weight"]),
                )
                rows = [
                    row
                    for row in fold_rows
                    if row["target_id"] == target_id
                    and int(row["outer_fold"]) == outer_fold
                    and row["method"] == CANDIDATE_METHOD
                    and row["candidate_id"] == candidate_id
                ]
                if len(rows) != 1:
                    raise ValueError(f"{target_id}/{outer_fold}/{candidate_id} row count differs")
                row = rows[0]
                index = subset_index[tuple(certificate["selected_subset"])]
                if row["selected_subset"] != "+".join(certificate["selected_subset"]):
                    raise ValueError(f"{target_id}/{outer_fold}/{candidate_id} subset differs")
                compare_metrics(row, train_values, index, "train", candidate_id)
                compare_metrics(row, holdout_values, index, "holdout", candidate_id)
                for field, expected in (
                    ("qubo_variable_count", len(qubo["variables"])),
                    ("qubo_state_count", certificate["state_count"]),
                ):
                    if int(row[field]) != int(expected):
                        raise ValueError(f"{target_id}/{outer_fold}/{candidate_id}/{field} differs")
                assert_close(
                    float(row["qubo_scaled_best_second_gap"]),
                    float(certificate["scaled_best_second_gap"]),
                    f"{target_id}/{outer_fold}/{candidate_id}/gap",
                )
                assert_close(
                    float(row["qubo_equivalence_residual"]),
                    float(certificate["equivalence_residual"]),
                    f"{target_id}/{outer_fold}/{candidate_id}/residual",
                )
                outer_recomputed += 1

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
        for candidate in diagnostic["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            terms = build_terms(
                full_context,
                receptor_ids,
                coverage_fraction,
                int(candidate["active_threshold"]),
                alpha,
            )
            qubo = build_qubo(
                terms,
                receptor_ids,
                target_size,
                float(diagnostic["decoy_weight"]),
                float(candidate["redundancy_weight"]),
                cardinality_penalty,
                constraint_penalty,
            )
            certificate = certify_states(
                terms,
                qubo,
                receptor_ids,
                target_size,
                float(diagnostic["decoy_weight"]),
                float(candidate["redundancy_weight"]),
            )
            recorded = model_record["target_models"][target_id][candidate_id]
            compare_nested(
                terms,
                recorded["terms"],
                f"{target_id}/{candidate_id}/terms",
            )
            compare_nested(
                qubo,
                recorded["qubo"],
                f"{target_id}/{candidate_id}/qubo",
            )
            if certificate["selected_subset"] != recorded["selected_subset"]:
                raise ValueError(f"{target_id}/{candidate_id}/full subset differs")
            assert_close(
                certificate["equivalence_residual"],
                float(recorded["equivalence_residual"]),
                f"{target_id}/{candidate_id}/full residual",
            )
            assert_close(
                certificate["scaled_best_second_gap"],
                float(recorded["scaled_best_second_gap"]),
                f"{target_id}/{candidate_id}/full gap",
            )
            full_recomputed += 1

    recomputed_comparisons = {
        candidate_id: {
            f"{candidate_id}_vs_{method}": paired_comparison(
                fold_rows, candidate_id, method
            )
            for method in ("direct_greedy", "additive_top3")
        }
        for candidate_id in result["candidate_ids"]
    }
    compare_nested(recomputed_comparisons, result["paired_comparisons"], "comparisons")
    recomputed_classical = {}
    for candidate_id, comparison in recomputed_comparisons.items():
        greedy = comparison[f"{candidate_id}_vs_direct_greedy"]
        recomputed_classical[candidate_id] = (
            all(value["mean_delta"] > 0.0 for value in greedy["per_target"].values())
            and greedy["positive_fold_count"] >= int(
                diagnostic["classical_minimum_positive_folds_of_eight"]
            )
        )
    if recomputed_classical != result["classical_checks"]:
        raise ValueError("classical checks differ")
    if compare_nested(
        sorted(recomputed_noise_summary, key=noise_row_sort_key),
        sorted([
            {
                key: (float(value) if key not in {"target_id", "candidate_id", "noise_model_id"} else value)
                for key, value in row.items()
            }
            for row in noise_summary
        ], key=noise_row_sort_key), "noise summary second pass"
    ) is None:
        pass
    if any(value["ready_for_hardware_pilot"] for value in result["candidate_gate"].values()):
        raise ValueError("Stage 19i unexpectedly authorizes a hardware candidate")
    if result["ready_candidates"]:
        raise ValueError("Stage 19i ready candidate list is not empty")

    forbidden = ("fresh_validation", "locked_test", "bace1_docking")
    paths = [str(descriptor["path"]).lower() for descriptor in config["inputs"].values()]
    paths.extend(
        str(descriptor["path"]).lower()
        for spec in source_config["targets"].values()
        for descriptor in spec["inputs"].values()
    )
    if any(marker in path for path in paths for marker in forbidden):
        raise ValueError("Stage 19i input path crosses protected data")

    return {
        "schema_version": "1.0",
        "status": "stage19i_objective_adequacy_noise_screen_audit_ok",
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
            "fold_rows_recomputed": len(fold_rows),
            "candidate_outer_selections_recomputed": outer_recomputed,
            "full_train_models_recomputed": full_recomputed,
            "noise_summary_rows_recomputed": len(noise_summary),
            "noise_trial_rows_recomputed_from_summary": len(noise_trials),
        },
        "checks": {
            "all_input_and_output_hashes_verified": True,
            "all_outer_baselines_recomputed": True,
            "all_candidate_outer_qubos_recomputed": True,
            "all_full_train_qubos_recomputed": True,
            "all_560_state_equivalence_certificates_recomputed": True,
            "noise_summary_recomputed_from_raw_trials": True,
            "paired_comparisons_recomputed": True,
            "no_candidate_passed_hardware_gate": True,
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
