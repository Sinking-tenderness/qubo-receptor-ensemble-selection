"""Run the preregistered nested MAPK14 delta-aware QUBO gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

try:
    from .prepare_receptor import file_sha256
    from .run_stage05_mk14_uncertainty_qubo_gate import (
        MATRIX_IDS,
        METHOD_IDS,
        SEED_IDS,
        add_oof_records,
        audit_inputs,
        checked_input_paths,
        classical_candidate_configs,
        exact_fixed_subset_distribution,
        fit_method,
        fit_qubo,
        label_counts,
        make_context,
        metrics_for_context,
        oof_metrics,
        percentile,
        qubo_candidate_configs,
        robust_metric_summary,
        score_records,
        tune_candidates,
        write_csv,
        write_json,
    )
    from .run_stage05_mk14_uncertainty_qubo_gate import (
        load_config as load_original_config,
    )
    from .run_stage05_mk14_method_gate import make_frozen_group_folds
except ImportError:
    from prepare_receptor import file_sha256
    from run_stage05_mk14_uncertainty_qubo_gate import (
        MATRIX_IDS,
        METHOD_IDS,
        SEED_IDS,
        add_oof_records,
        audit_inputs,
        checked_input_paths,
        classical_candidate_configs,
        exact_fixed_subset_distribution,
        fit_method,
        fit_qubo,
        label_counts,
        make_context,
        metrics_for_context,
        oof_metrics,
        percentile,
        qubo_candidate_configs,
        robust_metric_summary,
        score_records,
        tune_candidates,
        write_csv,
        write_json,
    )
    from run_stage05_mk14_uncertainty_qubo_gate import (
        load_config as load_original_config,
    )
    from run_stage05_mk14_method_gate import make_frozen_group_folds


DELTA_METHOD_IDS = (
    "single_best",
    "matched_linear_top_k",
    "exhaustive",
    "greedy",
    "all_receptors",
    "delta_aware_qubo",
)
METRIC_KEYS = (
    "roc_auc",
    "pr_auc_average_precision",
    "bedroc_alpha_20",
    "EF1%",
    "EF5%",
    "EF10%",
)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def load_delta_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    required = {
        "schema_version",
        "authorization_id",
        "created_on",
        "target_id",
        "purpose",
        "development_history",
        "frozen_inputs",
        "cross_validation",
        "candidate_space",
        "inner_selection_order",
        "baselines",
        "acceptance",
        "outputs",
        "interpretation_boundary",
    }
    if set(config) != required:
        raise ValueError("delta-aware preregistration keys differ")
    history = config["development_history"]
    cv = config["cross_validation"]
    space = config["candidate_space"]
    acceptance = config["acceptance"]
    assert isinstance(history, dict)
    assert isinstance(cv, dict)
    assert isinstance(space, dict)
    assert isinstance(acceptance, dict)
    for key in (
        "original_preregistration",
        "failed_gate_result",
        "posthoc_candidate_diagnostic",
    ):
        record = history[key]
        assert isinstance(record, dict)
        evidence_path = Path(str(record["path"]))
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)
        if file_sha256(evidence_path) != str(record["sha256"]).upper():
            raise ValueError(f"development-history SHA-256 differs: {key}")
    failed = read_json(Path(str(history["failed_gate_result"]["path"])))
    diagnostic = read_json(
        Path(str(history["posthoc_candidate_diagnostic"]["path"]))
    )
    if failed.get("status") != history["failed_gate_result"]["required_status"]:
        raise ValueError("the original failed gate status changed")
    if (
        diagnostic.get("status")
        != history["posthoc_candidate_diagnostic"]["required_status"]
    ):
        raise ValueError("the post hoc diagnostic status changed")
    if (
        int(cv["validation_rows_available"]) != 0
        or int(cv["test_rows_available"]) != 0
    ):
        raise ValueError("validation and test rows must remain unavailable")
    if space.get("inherit_original_576_candidates") is not True:
        raise ValueError("the original candidate grid must be inherited")
    if space.get("new_weights_or_subsets_added") is not False:
        raise ValueError("the delta gate cannot add candidates")
    if acceptance.get("all_checks_required") is not True:
        raise ValueError("every delta-aware acceptance check must be required")
    if acceptance.get("validation_remains_unavailable_after_pass") is not True:
        raise ValueError("validation must remain unavailable after this gate")
    if acceptance.get("test_remains_locked_after_pass") is not True:
        raise ValueError("test must remain locked after this gate")
    return config


def average_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average an empty metric list")
    return {
        key: statistics.fmean(float(row[key]) for row in rows)
        for key in METRIC_KEYS
    }


def delta_trial(
    candidate: dict[str, object],
    contexts: list[dict[str, object]],
    receptor_ids: list[str],
    model: dict[str, object],
) -> dict[str, object]:
    qubo_metric_rows = {matrix_id: [] for matrix_id in MATRIX_IDS}
    linear_metric_rows = {matrix_id: [] for matrix_id in MATRIX_IDS}
    subsets: list[list[str]] = []
    linear_subsets: list[list[str]] = []
    jaccards: list[float] = []
    for context in contexts:
        subset, details = fit_qubo(candidate, context, receptor_ids, model)
        linear_subset = tuple(details["matched_linear_subset"])
        subsets.append(list(subset))
        linear_subsets.append(list(linear_subset))
        jaccards.append(float(details["seed_pairwise_jaccard"]["mean"]))
        qubo_metrics = metrics_for_context(
            context,
            subset,
            str(candidate["aggregation"]),
            "validation",
        )
        linear_metrics = metrics_for_context(
            context,
            linear_subset,
            str(candidate["aggregation"]),
            "validation",
        )
        for matrix_id in MATRIX_IDS:
            qubo_metric_rows[matrix_id].append(qubo_metrics[matrix_id])
            linear_metric_rows[matrix_id].append(linear_metrics[matrix_id])
    qubo_mean = {
        matrix_id: average_metrics(rows)
        for matrix_id, rows in qubo_metric_rows.items()
    }
    linear_mean = {
        matrix_id: average_metrics(rows)
        for matrix_id, rows in linear_metric_rows.items()
    }
    seed_deltas = {
        seed: float(qubo_mean[seed]["bedroc_alpha_20"])
        - float(linear_mean[seed]["bedroc_alpha_20"])
        for seed in SEED_IDS
    }
    q_robust = robust_metric_summary(qubo_mean)
    return {
        "config": candidate,
        "qubo_mean_validation_metrics": qubo_mean,
        "linear_mean_validation_metrics": linear_mean,
        "primary_bedroc_delta": float(
            qubo_mean["primary"]["bedroc_alpha_20"]
        )
        - float(linear_mean["primary"]["bedroc_alpha_20"]),
        "mean_seed_bedroc_delta": statistics.fmean(seed_deltas.values()),
        "worst_seed_bedroc_delta": min(seed_deltas.values()),
        "seed_bedroc_deltas": seed_deltas,
        "qubo_robust_metrics": q_robust,
        "subsets": subsets,
        "matched_linear_subsets": linear_subsets,
        "linear_difference_count": sum(
            subset != linear
            for subset, linear in zip(subsets, linear_subsets)
        ),
        "mean_seed_pairwise_jaccard": statistics.fmean(jaccards),
    }


def delta_trial_key(trial: dict[str, object]) -> tuple[object, ...]:
    config = trial["config"]
    robust = trial["qubo_robust_metrics"]
    assert isinstance(config, dict)
    assert isinstance(robust, dict)
    weights = dict(config["weights"])
    family_order = {"coverage_qubo": 0, "discriminative_qubo": 1}
    aggregation_order = {"min_score": 0, "mean_score": 1}
    return (
        -float(trial["worst_seed_bedroc_delta"]),
        -float(trial["primary_bedroc_delta"]),
        -float(trial["mean_seed_bedroc_delta"]),
        -float(robust["worst_seed_bedroc"]),
        -float(robust["primary_bedroc"]),
        -float(robust["primary_pr_auc"]),
        -float(robust["primary_roc_auc"]),
        -float(trial["mean_seed_pairwise_jaccard"]),
        int(config["target_size"]),
        sum(float(value) for value in weights.values()),
        family_order[str(config["family"])],
        aggregation_order[str(config["aggregation"])],
        json.dumps(config, sort_keys=True),
    )


def tune_delta_candidates(
    candidates: list[dict[str, object]],
    contexts: list[dict[str, object]],
    receptor_ids: list[str],
    model: dict[str, object],
    eligibility: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    trials = [
        delta_trial(candidate, contexts, receptor_ids, model)
        for candidate in candidates
    ]
    eligible = [
        trial
        for trial in trials
        if int(trial["linear_difference_count"])
        >= int(
            eligibility[
                "minimum_inner_contexts_where_qubo_subset_differs_from_matched_linear"
            ]
        )
        and float(trial["mean_seed_pairwise_jaccard"])
        >= float(
            eligibility[
                "minimum_mean_seed_specific_subset_pairwise_jaccard"
            ]
        )
    ]
    if not eligible:
        raise ValueError("no QUBO candidate satisfies inner eligibility")
    return min(eligible, key=delta_trial_key), trials


def flatten_delta_trial(trial: dict[str, object]) -> dict[str, object]:
    config = trial["config"]
    robust = trial["qubo_robust_metrics"]
    assert isinstance(config, dict)
    assert isinstance(robust, dict)
    return {
        "family": config["family"],
        "target_size": config["target_size"],
        "aggregation": config["aggregation"],
        "weights": json.dumps(config["weights"], sort_keys=True),
        "primary_bedroc_delta": trial["primary_bedroc_delta"],
        "mean_seed_bedroc_delta": trial["mean_seed_bedroc_delta"],
        "worst_seed_bedroc_delta": trial["worst_seed_bedroc_delta"],
        **{
            f"{seed}_bedroc_delta": trial["seed_bedroc_deltas"][seed]
            for seed in SEED_IDS
        },
        **{f"qubo_{key}": value for key, value in robust.items()},
        "linear_difference_count": trial["linear_difference_count"],
        "mean_seed_pairwise_jaccard": trial["mean_seed_pairwise_jaccard"],
        "subsets": json.dumps(trial["subsets"]),
        "matched_linear_subsets": json.dumps(
            trial["matched_linear_subsets"]
        ),
    }


def gate_decision(
    metrics: dict[str, dict[str, dict[str, object]]],
    final_subset: tuple[str, ...],
    final_linear_subset: tuple[str, ...],
    final_details: dict[str, object],
    outer_jaccard: float,
    random_context: dict[str, object],
    acceptance: dict[str, object],
) -> tuple[dict[str, float], dict[str, bool], bool]:
    candidate = metrics["delta_aware_qubo"]
    linear = metrics["matched_linear_top_k"]
    single = metrics["single_best"]
    linear_seed_deltas = {
        seed: float(candidate[seed]["bedroc_alpha_20"])
        - float(linear[seed]["bedroc_alpha_20"])
        for seed in SEED_IDS
    }
    single_seed_deltas = {
        seed: float(candidate[seed]["bedroc_alpha_20"])
        - float(single[seed]["bedroc_alpha_20"])
        for seed in SEED_IDS
    }
    deltas = {
        "primary_vs_linear": float(
            candidate["primary"]["bedroc_alpha_20"]
        )
        - float(linear["primary"]["bedroc_alpha_20"]),
        "mean_seed_vs_linear": statistics.fmean(linear_seed_deltas.values()),
        "worst_seed_vs_linear": min(linear_seed_deltas.values()),
        "primary_vs_single_best": float(
            candidate["primary"]["bedroc_alpha_20"]
        )
        - float(single["primary"]["bedroc_alpha_20"]),
        "worst_seed_vs_single_best": min(single_seed_deltas.values()),
        **{
            f"{seed}_vs_linear": value
            for seed, value in linear_seed_deltas.items()
        },
        **{
            f"{seed}_vs_single_best": value
            for seed, value in single_seed_deltas.items()
        },
    }
    quadratic = final_details["noncardinality_quadratic"]
    checks = {
        "minimum_selected_subset_size": len(final_subset)
        >= int(acceptance["minimum_selected_subset_size"]),
        "nonconstant_noncardinality_quadratic_terms": (
            float(quadratic["maximum_absolute"]) > 1e-12
            and float(quadratic["range"]) > 1e-12
        ),
        "final_subset_differs_from_matched_linear": final_subset
        != final_linear_subset,
        "primary_bedroc_vs_linear": deltas["primary_vs_linear"]
        >= float(acceptance["minimum_primary_bedroc_delta_vs_matched_linear"]),
        "mean_seed_bedroc_vs_linear": deltas["mean_seed_vs_linear"]
        >= float(acceptance["minimum_mean_seed_bedroc_delta_vs_matched_linear"]),
        "every_seed_bedroc_vs_linear": deltas["worst_seed_vs_linear"]
        >= float(acceptance["minimum_every_seed_bedroc_delta_vs_matched_linear"]),
        "primary_bedroc_vs_single_best": deltas["primary_vs_single_best"]
        >= float(acceptance["minimum_primary_bedroc_delta_vs_single_best"]),
        "every_seed_bedroc_vs_single_best": deltas[
            "worst_seed_vs_single_best"
        ]
        >= float(acceptance["minimum_every_seed_bedroc_delta_vs_single_best"]),
        "outer_seed_fit_mean_pairwise_jaccard": outer_jaccard
        >= float(acceptance["minimum_outer_seed_fit_mean_pairwise_jaccard"]),
        "final_seed_fit_mean_pairwise_jaccard": float(
            final_details["seed_pairwise_jaccard"]["mean"]
        )
        >= float(acceptance["minimum_final_seed_fit_mean_pairwise_jaccard"]),
        "primary_exact_fixed_subset_percentile": float(
            random_context["primary_bedroc_percentile"]
        )
        >= float(acceptance["minimum_primary_exact_fixed_subset_percentile"]),
        "mean_seed_exact_fixed_subset_percentile": float(
            random_context["mean_seed_bedroc_percentile"]
        )
        >= float(acceptance["minimum_mean_seed_exact_fixed_subset_percentile"]),
    }
    return deltas, checks, all(checks.values())


def compact_tracked_result(summary: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "authorization_id": summary["authorization_id"],
        "status": summary["status"],
        "preregistration": summary["preregistration"],
        "implementation": summary["implementation"],
        "data_boundary": summary["data_boundary"],
        "selected_qubo": summary["selected_qubo"],
        "method_oof_metrics": summary["method_oof_metrics"],
        "comparison": summary["comparison"],
        "exact_fixed_subset_context": summary["exact_fixed_subset_context"],
        "validation_status": summary["validation_status"],
        "test_status": summary["test_status"],
        "next_action": summary["next_action"],
        "interpretation_note": summary["interpretation_note"],
    }


def run_gate(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = load_delta_config(config_path)
    history = config["development_history"]
    cv = config["cross_validation"]
    space = config["candidate_space"]
    acceptance = config["acceptance"]
    outputs = config["outputs"]
    assert isinstance(history, dict)
    assert isinstance(cv, dict)
    assert isinstance(space, dict)
    assert isinstance(acceptance, dict)
    assert isinstance(outputs, dict)
    original_path = Path(str(history["original_preregistration"]["path"]))
    original = load_original_config(original_path)
    paths = checked_input_paths(original)
    audited = audit_inputs(original, paths)
    receptor_ids = [str(value) for value in original["receptor_ids"]]
    model = original["model"]
    expected = original["expected"]
    assert isinstance(model, dict)
    assert isinstance(expected, dict)
    manifest_rows = audited["manifest_rows"]
    matrices = audited["matrices"]
    assert isinstance(manifest_rows, list)
    assert isinstance(matrices, dict)
    matrices_by_id = {
        matrix_id: {str(row["ligand_id"]): row for row in rows}
        for matrix_id, rows in matrices.items()
    }
    ligand_ids = {row["ligand_id"] for row in manifest_rows}
    assignments = make_frozen_group_folds(
        manifest_rows,
        int(cv["outer_fold_count"]),
        int(cv["fold_seed"]),
    )
    fold_rows = [
        {
            "ligand_id": row["ligand_id"],
            "label": row["label"],
            "split_group_id": row["split_group_id"],
            "scaffold_smiles": row["scaffold_smiles"],
            "outer_fold": assignments[row["ligand_id"]],
        }
        for row in sorted(manifest_rows, key=lambda value: value["ligand_id"])
    ]
    qubo_candidates = qubo_candidate_configs(model)
    if len(qubo_candidates) != 576:
        raise ValueError("the inherited QUBO candidate count changed")
    classical = classical_candidate_configs(model, len(receptor_ids))
    eligibility = space["eligible_inner_candidate_checks"]
    assert isinstance(eligibility, dict)

    oof = {
        method: {matrix_id: {} for matrix_id in MATRIX_IDS}
        for method in DELTA_METHOD_IDS
    }
    outer_contexts: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    outer_details: list[dict[str, object]] = []
    outer_jaccards: list[float] = []
    for outer_fold in range(int(cv["outer_fold_count"])):
        outer_ids = {
            ligand_id
            for ligand_id, fold in assignments.items()
            if fold == outer_fold
        }
        outer_train = ligand_ids - outer_ids
        inner_contexts = []
        for inner_fold in range(int(cv["outer_fold_count"])):
            if inner_fold == outer_fold:
                continue
            inner_ids = {
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == inner_fold
            }
            inner_contexts.append(
                make_context(
                    outer_train - inner_ids,
                    inner_ids,
                    matrices_by_id,
                    receptor_ids,
                    model,
                )
            )
        outer_context = make_context(
            outer_train,
            outer_ids,
            matrices_by_id,
            receptor_ids,
            model,
        )
        outer_contexts.append(outer_context)
        selected_delta, _ = tune_delta_candidates(
            qubo_candidates,
            inner_contexts,
            receptor_ids,
            model,
            eligibility,
        )
        q_config = selected_delta["config"]
        q_subset, q_details = fit_qubo(
            q_config, outer_context, receptor_ids, model
        )
        linear_subset = tuple(q_details["matched_linear_subset"])
        outer_jaccards.append(
            float(q_details["seed_pairwise_jaccard"]["mean"])
        )
        fitted: dict[str, tuple[tuple[str, ...], dict[str, object]]] = {
            "delta_aware_qubo": (q_subset, q_config),
            "matched_linear_top_k": (
                linear_subset,
                {
                    "family": "matched_linear_top_k",
                    "target_size": q_config["target_size"],
                    "aggregation": q_config["aggregation"],
                    "source_qubo_config": q_config,
                },
            ),
        }
        for method, candidates in classical.items():
            selected, _ = tune_candidates(
                candidates, inner_contexts, receptor_ids, model
            )
            subset, _ = fit_method(
                selected["config"], outer_context, receptor_ids, model
            )
            fitted[method] = (subset, selected["config"])

        fold_detail = {
            "outer_fold": outer_fold,
            "train_ligand_count": len(outer_train),
            "validation_ligand_count": len(outer_ids),
            "delta_inner_selection": {
                "config": q_config,
                "primary_bedroc_delta": selected_delta[
                    "primary_bedroc_delta"
                ],
                "mean_seed_bedroc_delta": selected_delta[
                    "mean_seed_bedroc_delta"
                ],
                "worst_seed_bedroc_delta": selected_delta[
                    "worst_seed_bedroc_delta"
                ],
                "linear_difference_count": selected_delta[
                    "linear_difference_count"
                ],
            },
            "methods": {},
        }
        for method in DELTA_METHOD_IDS:
            subset, method_config = fitted[method]
            aggregation = str(method_config["aggregation"])
            fold_metrics = metrics_for_context(
                outer_context, subset, aggregation, "validation"
            )
            robust = robust_metric_summary(fold_metrics)
            for matrix_id in MATRIX_IDS:
                add_oof_records(
                    oof,
                    method,
                    matrix_id,
                    score_records(
                        outer_context["matrices"][matrix_id]["validation"],
                        subset,
                        aggregation,
                    ),
                )
            outer_rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": method,
                    "subset": "+".join(subset),
                    "target_size": len(subset),
                    "aggregation": aggregation,
                    "selected_config": json.dumps(method_config, sort_keys=True),
                    **robust,
                    "seed_fit_mean_pairwise_jaccard": (
                        q_details["seed_pairwise_jaccard"]["mean"]
                        if method == "delta_aware_qubo"
                        else ""
                    ),
                }
            )
            fold_detail["methods"][method] = {
                "config": method_config,
                "subset": list(subset),
                "metrics": fold_metrics,
            }
        fold_detail["methods"]["delta_aware_qubo"][
            "seed_specific_subsets"
        ] = {
            seed: list(value)
            for seed, value in q_details["seed_specific_subsets"].items()
        }
        fold_detail["methods"]["delta_aware_qubo"][
            "seed_pairwise_jaccard"
        ] = q_details["seed_pairwise_jaccard"]
        outer_details.append(fold_detail)

    metrics = oof_metrics(oof)
    if any(
        len(oof[method][matrix_id]) != len(ligand_ids)
        for method in DELTA_METHOD_IDS
        for matrix_id in MATRIX_IDS
    ):
        raise ValueError("delta-aware OOF predictions are incomplete")

    final_contexts = []
    for fold in range(int(cv["outer_fold_count"])):
        validation_ids = {
            ligand_id
            for ligand_id, assigned in assignments.items()
            if assigned == fold
        }
        final_contexts.append(
            make_context(
                ligand_ids - validation_ids,
                validation_ids,
                matrices_by_id,
                receptor_ids,
                model,
            )
        )
    final_selected, final_trials = tune_delta_candidates(
        qubo_candidates,
        final_contexts,
        receptor_ids,
        model,
        eligibility,
    )
    full_context = make_context(
        ligand_ids, set(), matrices_by_id, receptor_ids, model
    )
    final_config = final_selected["config"]
    final_subset, final_details = fit_qubo(
        final_config, full_context, receptor_ids, model
    )
    final_linear_subset = tuple(final_details["matched_linear_subset"])

    fixed_distribution = exact_fixed_subset_distribution(
        outer_contexts,
        receptor_ids,
        int(final_config["target_size"]),
        str(final_config["aggregation"]),
    )
    q_robust = robust_metric_summary(metrics["delta_aware_qubo"])
    random_context = {
        "subset_count": len(fixed_distribution),
        "primary_bedroc_percentile": percentile(
            [float(row["primary_bedroc"]) for row in fixed_distribution],
            q_robust["primary_bedroc"],
        ),
        "mean_seed_bedroc_percentile": percentile(
            [float(row["mean_seed_bedroc"]) for row in fixed_distribution],
            q_robust["mean_seed_bedroc"],
        ),
        "worst_seed_bedroc_percentile": percentile(
            [float(row["worst_seed_bedroc"]) for row in fixed_distribution],
            q_robust["worst_seed_bedroc"],
        ),
        "random_numbers_used": False,
    }
    deltas, checks, passed = gate_decision(
        metrics,
        final_subset,
        final_linear_subset,
        final_details,
        statistics.fmean(outer_jaccards),
        random_context,
        acceptance,
    )
    status = (
        "train_delta_aware_qubo_gate_passed_validation_still_unavailable"
        if passed
        else "train_delta_aware_qubo_gate_failed_validation_unavailable"
    )
    next_action = (
        "Freeze this train-only candidate and prepare a separate preregistration and remote docking plan for one development-validation experiment; do not read validation or test scores yet."
        if passed
        else "Keep validation unavailable and redesign the paired selector or QUBO objective before another train-only gate."
    )

    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    existing = [
        path
        for key, path in output_paths.items()
        if key != "run_directory" and path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError("delta-aware outputs exist; use --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()
    output_paths["run_directory"].mkdir(parents=True, exist_ok=True)
    write_csv(output_paths["fold_assignments_csv"], fold_rows)
    write_csv(output_paths["outer_fold_results_csv"], outer_rows)
    oof_rows = [
        {
            "method": method,
            "matrix": matrix_id,
            "ligand_id": ligand_id,
            "label": record["label"],
            "outer_fold": assignments[ligand_id],
            "normalized_ensemble_score": record["score"],
        }
        for method in DELTA_METHOD_IDS
        for matrix_id in MATRIX_IDS
        for ligand_id, record in sorted(oof[method][matrix_id].items())
    ]
    write_csv(output_paths["oof_scores_csv"], oof_rows)
    metric_rows = [
        {
            "method": method,
            "matrix": matrix_id,
            **{key: value for key, value in metric.items() if key != "top10_ligand_ids"},
        }
        for method in DELTA_METHOD_IDS
        for matrix_id, metric in metrics[method].items()
    ]
    write_csv(output_paths["method_metrics_csv"], metric_rows)
    write_csv(
        output_paths["final_tuning_trials_csv"],
        [flatten_delta_trial(trial) for trial in final_trials],
    )
    write_csv(
        output_paths["exact_fixed_subset_distribution_csv"],
        fixed_distribution,
    )

    implementation = {
        "path": f"scripts/{Path(__file__).name}",
        "sha256": file_sha256(Path(__file__)),
    }
    selected_qubo = {
        "config": final_config,
        "subset": list(final_subset),
        "matched_linear_subset": list(final_linear_subset),
        "seed_specific_subsets": {
            seed: list(value)
            for seed, value in final_details["seed_specific_subsets"].items()
        },
        "seed_pairwise_jaccard": final_details["seed_pairwise_jaccard"],
        "noncardinality_quadratic": final_details[
            "noncardinality_quadratic"
        ],
        "final_tuning": {
            "primary_bedroc_delta": final_selected[
                "primary_bedroc_delta"
            ],
            "mean_seed_bedroc_delta": final_selected[
                "mean_seed_bedroc_delta"
            ],
            "worst_seed_bedroc_delta": final_selected[
                "worst_seed_bedroc_delta"
            ],
            "linear_difference_count": final_selected[
                "linear_difference_count"
            ],
        },
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
            "label_counts": label_counts(manifest_rows),
            "receptors": len(receptor_ids),
            "e32_seed_matrices": len(SEED_IDS),
            "e32_cells_replaced": 0,
            "e64_scores_used_in_matrix": 0,
            "validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "candidate_count": len(qubo_candidates),
        "outer_fold_details": outer_details,
        "selected_qubo": selected_qubo,
        "method_oof_metrics": metrics,
        "comparison": {
            "bedroc_deltas": deltas,
            "acceptance_checks": checks,
            "gate_passed": passed,
        },
        "exact_fixed_subset_context": random_context,
        "validation_status": "unavailable_not_evaluated",
        "test_status": "locked_unreleased",
        "next_action": next_action,
        "interpretation_note": config["interpretation_boundary"],
    }
    candidate_protocol = {
        key: value
        for key, value in summary.items()
        if key not in {"outer_fold_details", "method_oof_metrics"}
    }
    candidate_protocol["method_oof_metrics"] = metrics
    write_json(output_paths["candidate_protocol_json"], candidate_protocol)
    output_records = {
        key: {"path": path.as_posix(), "sha256": file_sha256(path)}
        for key, path in output_paths.items()
        if key
        not in {
            "run_directory",
            "summary_json",
            "tracked_result_json",
        }
    }
    summary["outputs"] = output_records
    write_json(output_paths["summary_json"], summary)
    write_json(
        output_paths["tracked_result_json"], compact_tracked_result(summary)
    )
    print(
        json.dumps(
            {
                "status": status,
                "selected_subset": list(final_subset),
                "matched_linear_subset": list(final_linear_subset),
                "bedroc_deltas": deltas,
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
    run_gate(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
