"""Run repeated grouped-scaffold CV for one frozen MAPK14 QUBO candidate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .prepare_receptor import file_sha256
    from .run_stage05_mk14_uncertainty_qubo_gate import (
        MATRIX_IDS,
        SEED_IDS,
        audit_inputs,
        checked_input_paths,
        exact_fixed_subset_distribution,
        fit_method,
        fit_qubo,
        make_context,
        percentile,
        score_records,
        write_csv,
        write_json,
    )
    from .run_stage05_mk14_uncertainty_qubo_gate import (
        load_config as load_original_config,
    )
    from .run_stage05_mk14_method_gate import make_frozen_group_folds
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from prepare_receptor import file_sha256
    from run_stage05_mk14_uncertainty_qubo_gate import (
        MATRIX_IDS,
        SEED_IDS,
        audit_inputs,
        checked_input_paths,
        exact_fixed_subset_distribution,
        fit_method,
        fit_qubo,
        make_context,
        percentile,
        score_records,
        write_csv,
        write_json,
    )
    from run_stage05_mk14_uncertainty_qubo_gate import (
        load_config as load_original_config,
    )
    from run_stage05_mk14_method_gate import make_frozen_group_folds


METHOD_IDS = ("fixed_qubo", "matched_linear_top_k", "single_best")


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def load_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    required = {
        "schema_version",
        "authorization_id",
        "created_on",
        "target_id",
        "purpose",
        "development_history",
        "fixed_candidate",
        "repeated_cross_validation",
        "comparators",
        "acceptance",
        "outputs",
        "interpretation_boundary",
    }
    if set(config) != required:
        raise ValueError("fixed-candidate preregistration keys differ")
    history = config["development_history"]
    candidate = config["fixed_candidate"]
    repeated = config["repeated_cross_validation"]
    acceptance = config["acceptance"]
    assert isinstance(history, dict)
    assert isinstance(candidate, dict)
    assert isinstance(repeated, dict)
    assert isinstance(acceptance, dict)
    for key in (
        "original_matrix_and_method_preregistration",
        "candidate_diagnostic",
        "failed_delta_aware_nested_gate",
    ):
        record = history[key]
        assert isinstance(record, dict)
        source = Path(str(record["path"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        if file_sha256(source) != str(record["sha256"]).upper():
            raise ValueError(f"fixed-candidate history SHA-256 differs: {key}")
    diagnostic = read_json(Path(str(history["candidate_diagnostic"]["path"])))
    best = diagnostic[history["candidate_diagnostic"]["selection_rule"]]
    if (
        candidate["family"] != best["family"]
        or int(candidate["target_size"]) != int(best["target_size"])
        or candidate["aggregation"] != best["aggregation"]
        or candidate["weights"] != json.loads(str(best["weights"]))
    ):
        raise ValueError("the fixed candidate differs from its frozen provenance")
    failed = read_json(
        Path(str(history["failed_delta_aware_nested_gate"]["path"]))
    )
    if (
        failed.get("status")
        != history["failed_delta_aware_nested_gate"]["required_status"]
    ):
        raise ValueError("the failed delta-aware gate status changed")
    seeds = [int(value) for value in repeated["fold_seeds"]]
    if (
        len(seeds) != int(repeated["repeat_count"])
        or len(seeds) != len(set(seeds))
        or int(repeated["fold_count"]) != 4
    ):
        raise ValueError("repeated-CV seeds or fold count differ")
    if (
        int(repeated["validation_rows_available"]) != 0
        or int(repeated["test_rows_available"]) != 0
    ):
        raise ValueError("validation and test rows must remain unavailable")
    if candidate.get("new_candidate_search_permitted") is not False:
        raise ValueError("fixed-candidate execution cannot search candidates")
    if acceptance.get("all_checks_required") is not True:
        raise ValueError("every fixed-candidate check must be required")
    return config


def nearest_rank(values: list[float], fraction: float) -> float:
    if not values or not 0.0 < fraction <= 1.0:
        raise ValueError("nearest-rank inputs are invalid")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def add_records(
    destination: dict[str, dict[str, dict[str, object]]],
    method: str,
    matrix_id: str,
    records: dict[str, dict[str, object]],
) -> None:
    overlap = set(destination[method][matrix_id]) & set(records)
    if overlap:
        raise ValueError(f"duplicate repeated-CV record: {method}/{matrix_id}")
    destination[method][matrix_id].update(records)


def metrics_from_records(
    records: dict[str, dict[str, dict[str, object]]]
) -> dict[str, dict[str, dict[str, object]]]:
    return {
        method: {
            matrix_id: ranked_metrics_with_ids(matrix_records)
            for matrix_id, matrix_records in by_matrix.items()
        }
        for method, by_matrix in records.items()
    }


def repeat_deltas(
    metrics: dict[str, dict[str, dict[str, object]]]
) -> dict[str, float]:
    q = metrics["fixed_qubo"]
    linear = metrics["matched_linear_top_k"]
    single = metrics["single_best"]
    linear_seed = {
        seed: float(q[seed]["bedroc_alpha_20"])
        - float(linear[seed]["bedroc_alpha_20"])
        for seed in SEED_IDS
    }
    single_seed = {
        seed: float(q[seed]["bedroc_alpha_20"])
        - float(single[seed]["bedroc_alpha_20"])
        for seed in SEED_IDS
    }
    return {
        "primary_vs_linear": float(q["primary"]["bedroc_alpha_20"])
        - float(linear["primary"]["bedroc_alpha_20"]),
        "mean_seed_vs_linear": statistics.fmean(linear_seed.values()),
        "worst_seed_vs_linear": min(linear_seed.values()),
        "primary_vs_single_best": float(q["primary"]["bedroc_alpha_20"])
        - float(single["primary"]["bedroc_alpha_20"]),
        "worst_seed_vs_single_best": min(single_seed.values()),
        **{f"{seed}_vs_linear": value for seed, value in linear_seed.items()},
        **{
            f"{seed}_vs_single_best": value
            for seed, value in single_seed.items()
        },
    }


def run_repeat(
    repeat_index: int,
    fold_seed: int,
    manifest_rows: list[dict[str, str]],
    matrices_by_id: dict[str, dict[str, dict[str, object]]],
    receptor_ids: list[str],
    model: dict[str, object],
    candidate: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    assignments = make_frozen_group_folds(manifest_rows, 4, fold_seed)
    ligand_ids = set(assignments)
    records = {
        method: {matrix_id: {} for matrix_id in MATRIX_IDS}
        for method in METHOD_IDS
    }
    contexts: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    fold_jaccards: list[float] = []
    subset_difference_count = 0
    for fold in range(4):
        validation_ids = {
            ligand_id
            for ligand_id, assigned in assignments.items()
            if assigned == fold
        }
        context = make_context(
            ligand_ids - validation_ids,
            validation_ids,
            matrices_by_id,
            receptor_ids,
            model,
        )
        contexts.append(context)
        q_subset, q_details = fit_qubo(
            candidate, context, receptor_ids, model
        )
        linear_subset = tuple(q_details["matched_linear_subset"])
        single_config = {
            "family": "single_best",
            "target_size": 1,
            "aggregation": "min_score",
        }
        single_subset, _ = fit_method(
            single_config, context, receptor_ids, model
        )
        fitted = {
            "fixed_qubo": (
                q_subset,
                str(candidate["aggregation"]),
            ),
            "matched_linear_top_k": (
                linear_subset,
                str(candidate["aggregation"]),
            ),
            "single_best": (single_subset, "min_score"),
        }
        fold_jaccards.append(
            float(q_details["seed_pairwise_jaccard"]["mean"])
        )
        subset_difference_count += q_subset != linear_subset
        fold_metrics: dict[str, dict[str, dict[str, object]]] = {}
        for method, (subset, aggregation) in fitted.items():
            fold_metrics[method] = {}
            for matrix_id in MATRIX_IDS:
                rows = context["matrices"][matrix_id]["validation"]
                method_records = score_records(
                    rows, subset, aggregation
                )
                add_records(records, method, matrix_id, method_records)
                fold_metrics[method][matrix_id] = ranked_metrics_with_ids(
                    method_records
                )
        fold_rows.append(
            {
                "repeat_index": repeat_index,
                "fold_seed": fold_seed,
                "fold": fold,
                "validation_ligand_count": len(validation_ids),
                "fixed_qubo_subset": "+".join(q_subset),
                "matched_linear_subset": "+".join(linear_subset),
                "single_best_subset": "+".join(single_subset),
                "qubo_differs_from_linear": q_subset != linear_subset,
                "seed_fit_pairwise_jaccard": q_details[
                    "seed_pairwise_jaccard"
                ]["mean"],
                "primary_bedroc_delta_vs_linear": float(
                    fold_metrics["fixed_qubo"]["primary"][
                        "bedroc_alpha_20"
                    ]
                )
                - float(
                    fold_metrics["matched_linear_top_k"]["primary"][
                        "bedroc_alpha_20"
                    ]
                ),
                "worst_seed_bedroc_delta_vs_linear": min(
                    float(
                        fold_metrics["fixed_qubo"][seed][
                            "bedroc_alpha_20"
                        ]
                    )
                    - float(
                        fold_metrics["matched_linear_top_k"][seed][
                            "bedroc_alpha_20"
                        ]
                    )
                    for seed in SEED_IDS
                ),
            }
        )
    metrics = metrics_from_records(records)
    deltas = repeat_deltas(metrics)
    distribution = exact_fixed_subset_distribution(
        contexts,
        receptor_ids,
        int(candidate["target_size"]),
        str(candidate["aggregation"]),
    )
    q_primary = float(
        metrics["fixed_qubo"]["primary"]["bedroc_alpha_20"]
    )
    q_mean_seed = statistics.fmean(
        float(metrics["fixed_qubo"][seed]["bedroc_alpha_20"])
        for seed in SEED_IDS
    )
    repeat_row = {
        "repeat_index": repeat_index,
        "fold_seed": fold_seed,
        "fold_seed_fit_mean_pairwise_jaccard": statistics.fmean(
            fold_jaccards
        ),
        "fold_subset_difference_count": subset_difference_count,
        **deltas,
        **{
            f"fixed_qubo_{matrix_id}_bedroc": metrics["fixed_qubo"][
                matrix_id
            ]["bedroc_alpha_20"]
            for matrix_id in MATRIX_IDS
        },
        **{
            f"matched_linear_{matrix_id}_bedroc": metrics[
                "matched_linear_top_k"
            ][matrix_id]["bedroc_alpha_20"]
            for matrix_id in MATRIX_IDS
        },
        **{
            f"single_best_{matrix_id}_bedroc": metrics["single_best"][
                matrix_id
            ]["bedroc_alpha_20"]
            for matrix_id in MATRIX_IDS
        },
        "primary_fixed_subset_percentile": percentile(
            [float(row["primary_bedroc"]) for row in distribution],
            q_primary,
        ),
        "mean_seed_fixed_subset_percentile": percentile(
            [float(row["mean_seed_bedroc"]) for row in distribution],
            q_mean_seed,
        ),
    }
    return repeat_row, fold_rows


def gate_decision(
    repeat_rows: list[dict[str, object]],
    final_subset: tuple[str, ...],
    final_linear_subset: tuple[str, ...],
    final_details: dict[str, object],
    acceptance: dict[str, object],
) -> tuple[dict[str, object], dict[str, bool], bool]:
    primary_linear = [
        float(row["primary_vs_linear"]) for row in repeat_rows
    ]
    mean_seed_linear = [
        float(row["mean_seed_vs_linear"]) for row in repeat_rows
    ]
    worst_seed_linear = [
        float(row["worst_seed_vs_linear"]) for row in repeat_rows
    ]
    primary_single = [
        float(row["primary_vs_single_best"]) for row in repeat_rows
    ]
    worst_seed_single = [
        float(row["worst_seed_vs_single_best"]) for row in repeat_rows
    ]
    every_seed_nonworse = [
        all(float(row[f"{seed}_vs_linear"]) >= 0.0 for seed in SEED_IDS)
        for row in repeat_rows
    ]
    stability = {
        "repeat_count": len(repeat_rows),
        "median_primary_bedroc_delta_vs_linear": statistics.median(
            primary_linear
        ),
        "median_mean_seed_bedroc_delta_vs_linear": statistics.median(
            mean_seed_linear
        ),
        "q25_worst_seed_bedroc_delta_vs_linear": nearest_rank(
            worst_seed_linear, 0.25
        ),
        "primary_nonworse_repeat_fraction_vs_linear": sum(
            value >= 0.0 for value in primary_linear
        )
        / len(primary_linear),
        "every_seed_nonworse_repeat_fraction_vs_linear": sum(
            every_seed_nonworse
        )
        / len(every_seed_nonworse),
        "median_primary_bedroc_delta_vs_single_best": statistics.median(
            primary_single
        ),
        "q25_worst_seed_bedroc_delta_vs_single_best": nearest_rank(
            worst_seed_single, 0.25
        ),
        "median_fold_seed_fit_pairwise_jaccard": statistics.median(
            float(row["fold_seed_fit_mean_pairwise_jaccard"])
            for row in repeat_rows
        ),
        "median_primary_fixed_subset_percentile": statistics.median(
            float(row["primary_fixed_subset_percentile"])
            for row in repeat_rows
        ),
        "median_mean_seed_fixed_subset_percentile": statistics.median(
            float(row["mean_seed_fixed_subset_percentile"])
            for row in repeat_rows
        ),
        "quartile_method": "nearest_rank",
    }
    quadratic = final_details["noncardinality_quadratic"]
    checks = {
        "minimum_selected_subset_size": len(final_subset)
        >= int(acceptance["minimum_selected_subset_size"]),
        "nonconstant_noncardinality_quadratic_terms": (
            float(quadratic["maximum_absolute"]) > 1e-12
            and float(quadratic["range"]) > 1e-12
        ),
        "full_train_subset_differs_from_matched_linear": final_subset
        != final_linear_subset,
        "median_primary_bedroc_delta_vs_linear": float(
            stability["median_primary_bedroc_delta_vs_linear"]
        )
        >= float(acceptance["minimum_median_primary_bedroc_delta_vs_linear"]),
        "median_mean_seed_bedroc_delta_vs_linear": float(
            stability["median_mean_seed_bedroc_delta_vs_linear"]
        )
        >= float(
            acceptance["minimum_median_mean_seed_bedroc_delta_vs_linear"]
        ),
        "q25_worst_seed_bedroc_delta_vs_linear": float(
            stability["q25_worst_seed_bedroc_delta_vs_linear"]
        )
        >= float(
            acceptance["minimum_q25_worst_seed_bedroc_delta_vs_linear"]
        ),
        "primary_nonworse_repeat_fraction_vs_linear": float(
            stability["primary_nonworse_repeat_fraction_vs_linear"]
        )
        >= float(
            acceptance["minimum_primary_nonworse_repeat_fraction_vs_linear"]
        ),
        "every_seed_nonworse_repeat_fraction_vs_linear": float(
            stability["every_seed_nonworse_repeat_fraction_vs_linear"]
        )
        >= float(
            acceptance[
                "minimum_every_seed_nonworse_repeat_fraction_vs_linear"
            ]
        ),
        "median_primary_bedroc_delta_vs_single_best": float(
            stability["median_primary_bedroc_delta_vs_single_best"]
        )
        >= float(
            acceptance["minimum_median_primary_bedroc_delta_vs_single_best"]
        ),
        "q25_worst_seed_bedroc_delta_vs_single_best": float(
            stability["q25_worst_seed_bedroc_delta_vs_single_best"]
        )
        >= float(
            acceptance["minimum_q25_worst_seed_bedroc_delta_vs_single_best"]
        ),
        "median_fold_seed_fit_pairwise_jaccard": float(
            stability["median_fold_seed_fit_pairwise_jaccard"]
        )
        >= float(
            acceptance["minimum_median_fold_seed_fit_pairwise_jaccard"]
        ),
        "full_train_seed_fit_pairwise_jaccard": float(
            final_details["seed_pairwise_jaccard"]["mean"]
        )
        >= float(
            acceptance["minimum_full_train_seed_fit_pairwise_jaccard"]
        ),
        "median_primary_fixed_subset_percentile": float(
            stability["median_primary_fixed_subset_percentile"]
        )
        >= float(
            acceptance["minimum_median_primary_fixed_subset_percentile"]
        ),
        "median_mean_seed_fixed_subset_percentile": float(
            stability["median_mean_seed_fixed_subset_percentile"]
        )
        >= float(
            acceptance["minimum_median_mean_seed_fixed_subset_percentile"]
        ),
    }
    return stability, checks, all(checks.values())


def compact_result(summary: dict[str, object]) -> dict[str, object]:
    keys = (
        "schema_version",
        "authorization_id",
        "status",
        "preregistration",
        "implementation",
        "data_boundary",
        "fixed_candidate",
        "full_train_refit",
        "repeat_stability",
        "acceptance_checks",
        "gate_passed",
        "validation_status",
        "test_status",
        "next_action",
        "interpretation_note",
    )
    return {key: summary[key] for key in keys}


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    history = config["development_history"]
    candidate = config["fixed_candidate"]
    repeated = config["repeated_cross_validation"]
    acceptance = config["acceptance"]
    outputs = config["outputs"]
    assert isinstance(history, dict)
    assert isinstance(candidate, dict)
    assert isinstance(repeated, dict)
    assert isinstance(acceptance, dict)
    assert isinstance(outputs, dict)
    original_path = Path(
        str(history["original_matrix_and_method_preregistration"]["path"])
    )
    original = load_original_config(original_path)
    input_paths = checked_input_paths(original)
    audited = audit_inputs(original, input_paths)
    receptor_ids = [str(value) for value in original["receptor_ids"]]
    model = original["model"]
    assert isinstance(model, dict)
    manifest_rows = audited["manifest_rows"]
    matrices = audited["matrices"]
    assert isinstance(manifest_rows, list)
    assert isinstance(matrices, dict)
    matrices_by_id = {
        matrix_id: {str(row["ligand_id"]): row for row in rows}
        for matrix_id, rows in matrices.items()
    }
    fixed_candidate = {
        "family": candidate["family"],
        "target_size": int(candidate["target_size"]),
        "aggregation": candidate["aggregation"],
        "weights": {
            key: float(value) for key, value in dict(candidate["weights"]).items()
        },
    }
    repeat_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for repeat_index, fold_seed in enumerate(repeated["fold_seeds"]):
        repeat_row, current_folds = run_repeat(
            repeat_index,
            int(fold_seed),
            manifest_rows,
            matrices_by_id,
            receptor_ids,
            model,
            fixed_candidate,
        )
        repeat_rows.append(repeat_row)
        fold_rows.extend(current_folds)

    ligand_ids = {row["ligand_id"] for row in manifest_rows}
    full_context = make_context(
        ligand_ids, set(), matrices_by_id, receptor_ids, model
    )
    final_subset, final_details = fit_qubo(
        fixed_candidate, full_context, receptor_ids, model
    )
    final_linear_subset = tuple(final_details["matched_linear_subset"])
    stability, checks, passed = gate_decision(
        repeat_rows,
        final_subset,
        final_linear_subset,
        final_details,
        acceptance,
    )
    status = (
        "fixed_qubo_repeated_cv_passed_validation_still_unavailable"
        if passed
        else "fixed_qubo_repeated_cv_failed_validation_unavailable"
    )
    next_action = (
        "Freeze the fixed train-only candidate and prepare a separate preregistration and remote docking bundle for one development-validation experiment."
        if passed
        else "Do not tune the 160-ligand matrix again; expand the development-train ligand matrix before further QUBO selection."
    )
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    existing = [
        path
        for key, path in output_paths.items()
        if key != "run_directory" and path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError("fixed repeated-CV outputs exist; use --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()
    output_paths["run_directory"].mkdir(parents=True, exist_ok=True)
    write_csv(output_paths["repeat_metrics_csv"], repeat_rows)
    write_csv(output_paths["fold_results_csv"], fold_rows)
    implementation = {
        "path": f"scripts/{Path(__file__).name}",
        "sha256": file_sha256(Path(__file__)),
    }
    summary = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": status,
        "preregistration": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "implementation": implementation,
        "development_history": history,
        "data_boundary": {
            "development_train_ligands": len(ligand_ids),
            "active": sum(row["label"] == "active" for row in manifest_rows),
            "decoy": sum(row["label"] == "decoy" for row in manifest_rows),
            "receptors": len(receptor_ids),
            "e32_seed_matrices": len(SEED_IDS),
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "e32_cells_replaced": 0,
            "e64_scores_used_in_matrix": 0,
        },
        "fixed_candidate": fixed_candidate,
        "full_train_refit": {
            "subset": list(final_subset),
            "matched_linear_subset": list(final_linear_subset),
            "seed_specific_subsets": {
                seed: list(value)
                for seed, value in final_details[
                    "seed_specific_subsets"
                ].items()
            },
            "seed_pairwise_jaccard": final_details[
                "seed_pairwise_jaccard"
            ],
            "noncardinality_quadratic": final_details[
                "noncardinality_quadratic"
            ],
        },
        "repeat_stability": stability,
        "acceptance_checks": checks,
        "gate_passed": passed,
        "validation_status": "unavailable_not_evaluated",
        "test_status": "locked_unreleased",
        "next_action": next_action,
        "outputs": {
            "repeat_metrics_csv": {
                "path": output_paths["repeat_metrics_csv"].as_posix(),
                "sha256": file_sha256(output_paths["repeat_metrics_csv"]),
            },
            "fold_results_csv": {
                "path": output_paths["fold_results_csv"].as_posix(),
                "sha256": file_sha256(output_paths["fold_results_csv"]),
            },
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    write_json(output_paths["summary_json"], summary)
    write_json(output_paths["tracked_result_json"], compact_result(summary))
    print(
        json.dumps(
            {
                "status": status,
                "full_train_subset": list(final_subset),
                "matched_linear_subset": list(final_linear_subset),
                "repeat_stability": stability,
                "acceptance_checks": checks,
                "validation_rows_read": 0,
                "test_rows_read": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
