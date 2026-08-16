"""Compare frozen QUBO and classical methods on PPARG Train-668 only."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json, write_json  # noqa: F401 (deduped)
import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy
import sklearn
import xgboost

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_stage05_mk14_literature_baselines import (
    consensus_ranking_scores,
    fit_supervised_method,
    paired_group_bootstrap_delta,
    predict_supervised_method,
    select_hantz_top_k,
)
from scripts.fit_enopt_xgboost_baseline import (
    MATRIX_IDS,
    SEED_IDS,
    fit_primary_model,
    metrics_for_probabilities,
    minmax_bounds,
    predict_matrix,
    robust_bedroc,
)
from scripts.normalized_receptor_qubo import build_coefficients, coefficient_energy
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import (
    choose_exhaustive,
    choose_greedy,
    make_context,
    matched_linear_top_k,
    noncardinality_quadratic_summary,
    pair_synergy_terms_for_aggregation,
    score_records,
)
from scripts.screen_stage10_mk14_expanded16_qubo_greedy import (
    build_matrices,
    fixed_cardinality_exact,
    fixed_cardinality_greedy,
)


SUBSET_METHODS = (
    "qubo_exact_top3",
    "qubo_greedy_top3",
    "matched_linear_top3",
    "direct_greedy_top3",
    "direct_exact_top3",
    "single_best",
)
CONSENSUS_METHODS = (
    "all16_min",
    "all16_mean",
    "hantz_auc_top3_min",
    "hantz_auc_top3_mean",
)
PRIMARY_ONLY_GEOMETRIC_METHODS = (
    "all16_geometric",
    "hantz_auc_top3_geometric",
)
SUPERVISED_METHODS = (
    "ricci_lr_all16",
    "ricci_gbt_all16",
    "ricci_gbt_rfe3",
    "edock_rf_all16",
    "edock_rf_rfe3",
)
XGBOOST_METHOD = "enopt_xgboost_all16"
METHODS = SUBSET_METHODS + CONSENSUS_METHODS + SUPERVISED_METHODS + (
    XGBOOST_METHOD,
)
LITERATURE_METHODS = CONSENSUS_METHODS + SUPERVISED_METHODS + (
    XGBOOST_METHOD,
)




def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)




def rooted(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path leaves repository root: {value}") from error
    return path


def verified(root: Path, descriptor: dict[str, object]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    if not path.is_file() or file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"input identity differs: {path}")
    return path


def output_descriptor(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def robust_key(metrics: dict[str, object], method: str) -> tuple[object, ...]:
    robust = robust_bedroc(dict(metrics))
    primary = dict(metrics["primary"])
    return (
        -float(robust["worst_seed"]),
        -float(robust["primary"]),
        -float(robust["mean_seed"]),
        -float(primary["pr_auc_average_precision"]),
        -float(primary["roc_auc"]),
        method,
    )


def subset_predictions(
    context: dict[str, object],
    subset: tuple[str, ...],
    aggregation: str,
    split: str,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for matrix_id in MATRIX_IDS:
        records = score_records(
            context["matrices"][matrix_id][split], subset, aggregation
        )
        output[matrix_id] = {
            ligand_id: -float(record["score"])
            for ligand_id, record in records.items()
        }
    return output


def select_qubo_subsets(
    context: dict[str, object],
    receptor_ids: list[str],
    config: dict[str, object],
) -> tuple[dict[str, tuple[str, ...]], dict[str, object]]:
    frozen = dict(config["frozen_qubo"])
    target_size = int(frozen["target_size"])
    aggregation = str(frozen["aggregation"])
    terms = pair_synergy_terms_for_aggregation(context["terms"], aggregation)
    coefficients = build_coefficients(
        terms,
        receptor_ids,
        target_size,
        {key: float(value) for key, value in dict(frozen["weights"]).items()},
        float(frozen["size_penalty"]),
    )
    exact, exact_energy = fixed_cardinality_exact(
        coefficients, receptor_ids, target_size
    )
    qubo_greedy, greedy_energy, greedy_path = fixed_cardinality_greedy(
        coefficients, receptor_ids, target_size
    )
    subsets = {
        "qubo_exact_top3": exact,
        "qubo_greedy_top3": qubo_greedy,
        "matched_linear_top3": matched_linear_top_k(coefficients, target_size),
        "direct_greedy_top3": choose_greedy(
            context, receptor_ids, target_size, aggregation
        ),
        "direct_exact_top3": choose_exhaustive(
            context, receptor_ids, target_size, aggregation
        ),
        "single_best": choose_exhaustive(context, receptor_ids, 1, aggregation),
    }
    evidence = {
        "coefficients": coefficients,
        "exact_energy": exact_energy,
        "qubo_greedy_energy": greedy_energy,
        "qubo_greedy_regret": greedy_energy - exact_energy,
        "qubo_greedy_path": greedy_path,
        "noncardinality_quadratic": noncardinality_quadratic_summary(
            coefficients, float(frozen["size_penalty"])
        ),
    }
    return subsets, evidence


def add_predictions(
    destination: dict[str, dict[str, dict[str, float]]],
    method: str,
    predictions: dict[str, dict[str, float]],
) -> None:
    for matrix_id, values in predictions.items():
        overlap = set(destination[method][matrix_id]) & set(values)
        if overlap:
            raise ValueError(f"duplicate OOF predictions: {method}/{matrix_id}")
        destination[method][matrix_id].update(values)


def supervised_predictions(
    method: str,
    model: Any,
    subset: tuple[str, ...],
    matrices: dict[str, dict[str, dict[str, object]]],
    train_ids: list[str],
    validation_ids: list[str],
) -> dict[str, dict[str, float]]:
    return {
        matrix_id: predict_supervised_method(
            model,
            matrices[matrix_id],
            matrices[matrix_id],
            train_ids,
            validation_ids,
            subset,
        )
        for matrix_id in MATRIX_IDS
    }


def consensus_predictions(
    method: str,
    matrices: dict[str, dict[str, dict[str, object]]],
    ligand_ids: list[str],
    receptor_ids: list[str],
    hantz_subset: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], tuple[str, ...]]:
    subset = (
        tuple(receptor_ids) if method.startswith("all16_") else hantz_subset
    )
    strategy = method.rsplit("_", 1)[-1]
    return (
        {
            matrix_id: consensus_ranking_scores(
                matrices[matrix_id], ligand_ids, subset, strategy
            )
            for matrix_id in MATRIX_IDS
        },
        subset,
    )


def records_for_bootstrap(
    matrix: dict[str, dict[str, object]], predictions: dict[str, float]
) -> dict[str, dict[str, object]]:
    return {
        ligand_id: {
            "label": matrix[ligand_id]["label"],
            "score": -float(prediction),
        }
        for ligand_id, prediction in predictions.items()
    }


def validate_inputs(
    config: dict[str, object], paths: dict[str, Path]
) -> tuple[
    list[dict[str, str]],
    list[str],
    dict[str, dict[str, dict[str, object]]],
]:
    summary = read_json(paths["stage19c_summary"])
    audit = read_json(paths["stage19c_audit"])
    amendment = read_json(paths["target_id_amendment"])
    preregistration = read_json(paths["pparg_preregistration"])
    if summary.get("status") != "stage19c_pparg_train668_unidock_matrix_ok":
        raise ValueError("Stage 19c source matrix did not pass")
    if (
        audit.get("status")
        != "independent_stage19c_pparg_train668_unidock_matrix_audit_ok"
    ):
        raise ValueError("Stage 19c independent audit did not pass")
    if amendment.get("status") != "stage19c_pparg_target_id_amendment01_ok":
        raise ValueError("Stage 19c target metadata amendment did not pass")
    if summary.get("stage18e_confirmatory_gate") != "closed_failed_14_of_24":
        raise ValueError("Stage 18e failure boundary differs")
    if any(int(value) != 0 for value in dict(audit["data_boundary"]).values()):
        raise ValueError("Stage 19c audit crossed a data boundary")

    frozen = dict(config["frozen_qubo"])
    registered = dict(preregistration["qubo"])
    for key in ("family", "target_size", "aggregation", "size_penalty"):
        if registered[key] != frozen[key]:
            raise ValueError(f"PPARG preregistered QUBO differs: {key}")
    if {
        key: float(value) for key, value in dict(registered["weights"]).items()
    } != {key: float(value) for key, value in dict(frozen["weights"]).items()}:
        raise ValueError("PPARG preregistered QUBO weights differ")

    ligands = read_csv(paths["ligand_manifest"])
    receptors = read_csv(paths["receptor_manifest"])
    scores = read_csv(paths["corrected_scores"])
    expected = dict(config["expected"])
    receptor_ids = [row["conformer_id"] for row in receptors]
    if (
        len(ligands) != int(expected["ligand_count"])
        or len(receptor_ids) != int(expected["receptor_count"])
        or Counter(row["label"] for row in ligands)
        != Counter(
            {key: int(value) for key, value in dict(expected["label_counts"]).items()}
        )
        or {row["split"] for row in ligands} != {expected["allowed_split"]}
        or {row["selection_role"] for row in ligands}
        != {expected["allowed_selection_role"]}
        or {row["target_id"] for row in ligands} != {config["target_id"]}
        or {row["target_id"] for row in scores} != {config["target_id"]}
    ):
        raise ValueError("Stage 19d input dimensions or boundaries differ")
    if len(scores) != int(expected["pair_count"]):
        raise ValueError("Stage 19d corrected score count differs")

    matrices = build_matrices(
        read_csv(paths["primary_matrix"]),
        read_csv(paths["sensitivity_matrix"]),
        scores,
        ligands,
        receptor_ids,
    )
    return ligands, receptor_ids, matrices


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, dict(config["implementation"]))
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 19d implementation path differs")
    paths = {
        key: verified(root, dict(value))
        for key, value in dict(config["inputs"]).items()
    }
    ligands, receptor_ids, matrices = validate_inputs(config, paths)
    ligand_ids = sorted(row["ligand_id"] for row in ligands)
    assignments = make_frozen_group_folds(
        ligands,
        int(config["cross_validation"]["fold_count"]),
        int(config["cross_validation"]["fold_seed"]),
    )
    oof = {
        method: {matrix_id: {} for matrix_id in MATRIX_IDS}
        for method in METHODS
    }
    primary_only_oof = {
        method: {} for method in PRIMARY_ONLY_GEOMETRIC_METHODS
    }
    selection_rows: list[dict[str, object]] = []
    fold_qubo: dict[str, dict[str, object]] = {}
    model_config = dict(config["models"])
    xgboost_config = dict(config["xgboost"])
    fold_count = int(config["cross_validation"]["fold_count"])
    for outer_fold in range(fold_count):
        validation_ids = sorted(
            ligand_id
            for ligand_id, fold in assignments.items()
            if fold == outer_fold
        )
        train_ids = sorted(set(ligand_ids) - set(validation_ids))
        context = make_context(
            set(train_ids),
            set(validation_ids),
            matrices,
            receptor_ids,
            dict(config["qubo_model"]),
        )
        subsets, qubo_evidence = select_qubo_subsets(
            context, receptor_ids, config
        )
        fold_qubo[str(outer_fold)] = qubo_evidence
        for method, subset in subsets.items():
            add_predictions(
                oof,
                method,
                subset_predictions(
                    context,
                    subset,
                    str(config["frozen_qubo"]["aggregation"]),
                    "validation",
                ),
            )
            selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": method,
                    "subset": "+".join(subset),
                    "receptor_count": len(subset),
                    "train_ligand_count": len(train_ids),
                    "validation_ligand_count": len(validation_ids),
                }
            )

        hantz_subset, _ = select_hantz_top_k(
            matrices["primary"],
            train_ids,
            receptor_ids,
            int(model_config["budget_matched_feature_count"]),
        )
        for method in CONSENSUS_METHODS:
            predictions, subset = consensus_predictions(
                method, matrices, validation_ids, receptor_ids, hantz_subset
            )
            add_predictions(oof, method, predictions)
            selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": method,
                    "subset": "+".join(subset),
                    "receptor_count": len(subset),
                    "train_ligand_count": len(train_ids),
                    "validation_ligand_count": len(validation_ids),
                }
            )
        for method in PRIMARY_ONLY_GEOMETRIC_METHODS:
            subset = (
                tuple(receptor_ids)
                if method.startswith("all16_")
                else hantz_subset
            )
            predictions = consensus_ranking_scores(
                matrices["primary"],
                validation_ids,
                subset,
                "geometric",
            )
            overlap = set(primary_only_oof[method]) & set(predictions)
            if overlap:
                raise ValueError(f"duplicate primary-only predictions: {method}")
            primary_only_oof[method].update(predictions)
            selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": method,
                    "subset": "+".join(subset),
                    "receptor_count": len(subset),
                    "train_ligand_count": len(train_ids),
                    "validation_ligand_count": len(validation_ids),
                    "matrix_scope": "primary_only",
                }
            )

        for method in SUPERVISED_METHODS:
            model, subset = fit_supervised_method(
                method,
                matrices["primary"],
                train_ids,
                receptor_ids,
                model_config,
            )
            add_predictions(
                oof,
                method,
                supervised_predictions(
                    method,
                    model,
                    subset,
                    matrices,
                    train_ids,
                    validation_ids,
                ),
            )
            selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": method,
                    "subset": "+".join(subset),
                    "receptor_count": len(subset),
                    "train_ligand_count": len(train_ids),
                    "validation_ligand_count": len(validation_ids),
                }
            )

        xgb_model, _ = fit_primary_model(
            matrices["primary"],
            train_ids,
            tuple(receptor_ids),
            dict(xgboost_config["parameters"]),
            int(xgboost_config["model_seed"]),
        )
        add_predictions(
            oof,
            XGBOOST_METHOD,
            {
                matrix_id: predict_matrix(
                    xgb_model,
                    matrices[matrix_id],
                    train_ids,
                    validation_ids,
                    tuple(receptor_ids),
                )
                for matrix_id in MATRIX_IDS
            },
        )
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "method": XGBOOST_METHOD,
                "subset": "+".join(receptor_ids),
                "receptor_count": len(receptor_ids),
                "train_ligand_count": len(train_ids),
                "validation_ligand_count": len(validation_ids),
            }
        )

    if any(
        set(oof[method][matrix_id]) != set(ligand_ids)
        for method in METHODS
        for matrix_id in MATRIX_IDS
    ):
        raise ValueError("Stage 19d OOF prediction coverage is incomplete")
    if any(
        set(predictions) != set(ligand_ids)
        for predictions in primary_only_oof.values()
    ):
        raise ValueError("Stage 19d primary-only prediction coverage is incomplete")
    metrics = {
        method: {
            matrix_id: metrics_for_probabilities(
                matrices[matrix_id], predictions
            )
            for matrix_id, predictions in by_matrix.items()
        }
        for method, by_matrix in oof.items()
    }
    robust = {method: robust_bedroc(by_matrix) for method, by_matrix in metrics.items()}
    primary_only_metrics = {
        method: metrics_for_probabilities(matrices["primary"], predictions)
        for method, predictions in primary_only_oof.items()
    }
    ranking = sorted(METHODS, key=lambda method: robust_key(metrics[method], method))
    strongest_literature = min(
        LITERATURE_METHODS,
        key=lambda method: robust_key(metrics[method], method),
    )

    full_context = make_context(
        set(ligand_ids),
        set(),
        matrices,
        receptor_ids,
        dict(config["qubo_model"]),
    )
    full_subsets, full_qubo = select_qubo_subsets(
        full_context, receptor_ids, config
    )
    full_hantz, singleton_auc = select_hantz_top_k(
        matrices["primary"],
        ligand_ids,
        receptor_ids,
        int(model_config["budget_matched_feature_count"]),
    )
    outputs = dict(config["outputs"])
    run_directory = rooted(root, str(outputs["run_directory"]))
    model_directory = run_directory / "models"
    output_paths = {
        key: rooted(root, str(value))
        for key, value in outputs.items()
        if key != "run_directory"
    }
    protected = list(output_paths.values())
    if not overwrite and (run_directory.exists() or any(path.exists() for path in protected)):
        raise FileExistsError("Stage 19d outputs exist; pass --overwrite")
    model_directory.mkdir(parents=True, exist_ok=True)

    model_evidence: dict[str, object] = {}
    for method in SUPERVISED_METHODS:
        model, subset = fit_supervised_method(
            method,
            matrices["primary"],
            ligand_ids,
            receptor_ids,
            model_config,
        )
        model_path = model_directory / f"{method}.joblib"
        joblib.dump(model, model_path, compress=3)
        model_evidence[method] = {
            "subset": list(subset),
            "parameters": dict(
                model_config[
                    "logistic_regression"
                    if method.startswith("ricci_lr_")
                    else "gradient_boosting"
                    if method.startswith("ricci_gbt_")
                    else "random_forest"
                ]["parameters"]
            ),
            "model": output_descriptor(root, model_path),
            "normalization_bounds": {
                matrix_id: minmax_bounds(
                    matrices[matrix_id], ligand_ids, subset
                )
                for matrix_id in MATRIX_IDS
            },
        }
    final_xgb, _ = fit_primary_model(
        matrices["primary"],
        ligand_ids,
        tuple(receptor_ids),
        dict(xgboost_config["parameters"]),
        int(xgboost_config["model_seed"]),
    )
    xgb_path = model_directory / f"{XGBOOST_METHOD}.json"
    final_xgb.save_model(xgb_path)
    model_evidence[XGBOOST_METHOD] = {
        "subset": receptor_ids,
        "parameters": dict(xgboost_config["parameters"]),
        "model": output_descriptor(root, xgb_path),
        "normalization_bounds": {
            matrix_id: minmax_bounds(
                matrices[matrix_id], ligand_ids, tuple(receptor_ids)
            )
            for matrix_id in MATRIX_IDS
        },
    }

    fold_rows = [
        {
            "ligand_id": row["ligand_id"],
            "label": row["label"],
            "split_group_id": row["split_group_id"],
            "scaffold_smiles": row["scaffold_smiles"],
            "outer_fold": assignments[row["ligand_id"]],
        }
        for row in sorted(ligands, key=lambda value: value["ligand_id"])
    ]
    prediction_rows = [
        {
            "method": method,
            "matrix": matrix_id,
            "ligand_id": ligand_id,
            "label": matrices[matrix_id][ligand_id]["label"],
            "outer_fold": assignments[ligand_id],
            "ranking_score": prediction,
        }
        for method in METHODS
        for matrix_id in MATRIX_IDS
        for ligand_id, prediction in sorted(oof[method][matrix_id].items())
    ] + [
        {
            "method": method,
            "matrix": "primary",
            "ligand_id": ligand_id,
            "label": matrices["primary"][ligand_id]["label"],
            "outer_fold": assignments[ligand_id],
            "ranking_score": prediction,
        }
        for method in PRIMARY_ONLY_GEOMETRIC_METHODS
        for ligand_id, prediction in sorted(primary_only_oof[method].items())
    ]
    metric_rows = [
        {
            "method": method,
            "matrix": matrix_id,
            **{
                key: value
                for key, value in metric.items()
                if key != "top10_ligand_ids"
            },
        }
        for method in METHODS
        for matrix_id, metric in metrics[method].items()
    ] + [
        {
            "method": method,
            "matrix": "primary",
            "scope": "primary_only_due_to_mixed_sign_seed_scores",
            **{
                key: value
                for key, value in metric.items()
                if key != "top10_ligand_ids"
            },
        }
        for method, metric in primary_only_metrics.items()
    ]
    full_selection_rows = [
        {
            "method": method,
            "subset": "+".join(subset),
            "receptor_count": len(subset),
        }
        for method, subset in full_subsets.items()
    ] + [
        {
            "method": method,
            "subset": "+".join(
                receptor_ids if method.startswith("all16_") else full_hantz
            ),
            "receptor_count": (
                len(receptor_ids) if method.startswith("all16_") else len(full_hantz)
            ),
        }
        for method in CONSENSUS_METHODS + PRIMARY_ONLY_GEOMETRIC_METHODS
    ] + [
        {
            "method": method,
            "subset": "+".join(dict(evidence)["subset"]),
            "receptor_count": len(dict(evidence)["subset"]),
        }
        for method, evidence in model_evidence.items()
    ]

    write_csv(output_paths["fold_assignments_csv"], fold_rows)
    write_csv(output_paths["fold_method_selections_csv"], selection_rows)
    write_csv(output_paths["oof_predictions_csv"], prediction_rows)
    write_csv(output_paths["oof_metrics_csv"], metric_rows)
    write_csv(output_paths["full_train_selections_csv"], full_selection_rows)
    write_json(
        output_paths["full_train_qubo_json"],
        {
            "schema_version": "1.0",
            "target_id": config["target_id"],
            "subsets": {key: list(value) for key, value in full_subsets.items()},
            **full_qubo,
        },
    )
    write_json(
        output_paths["frozen_methods_json"],
        {
            "schema_version": "1.0",
            "status": "stage19d_pparg_train_only_methods_frozen",
            "subsets": {key: list(value) for key, value in full_subsets.items()},
            "hantz_auc_top3_subset": list(full_hantz),
            "singleton_roc_auc": singleton_auc,
            "models": model_evidence,
            "versions": {
                "joblib": joblib.__version__,
                "numpy": numpy.__version__,
                "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
            "data_boundary": {"validation_rows_read": 0, "test_rows_read": 0},
        },
    )

    group_by_ligand = {
        row["ligand_id"]: row["split_group_id"] for row in ligands
    }
    bootstrap_config = dict(config["bootstrap"])
    qubo_records = records_for_bootstrap(
        matrices["primary"], oof["qubo_exact_top3"]["primary"]
    )
    primary_bootstrap = paired_group_bootstrap_delta(
        qubo_records,
        records_for_bootstrap(
            matrices["primary"], oof["direct_greedy_top3"]["primary"]
        ),
        group_by_ligand,
        int(bootstrap_config["replicates"]),
        int(bootstrap_config["seed"]),
    )
    literature_bootstrap = paired_group_bootstrap_delta(
        qubo_records,
        records_for_bootstrap(
            matrices["primary"], oof[strongest_literature]["primary"]
        ),
        group_by_ligand,
        int(bootstrap_config["replicates"]),
        int(bootstrap_config["seed"]) + 1,
    )

    primary_delta = (
        float(robust["qubo_exact_top3"]["primary"])
        - float(robust["direct_greedy_top3"]["primary"])
    )
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage19d_pparg_train_only_comparison_complete",
        "experiment_class": "posthoc_exploratory_train_only",
        "stage18e_confirmatory_gate": "closed_failed_14_of_24",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "input_dimensions": {
            "ligand_count": len(ligands),
            "receptor_count": len(receptor_ids),
            "seed_count": len(SEED_IDS),
            "outer_fold_count": fold_count,
        },
        "method_count": len(METHODS),
        "primary_only_geometric_method_count": len(
            PRIMARY_ONLY_GEOMETRIC_METHODS
        ),
        "method_ranking_by_oof_robust_key": ranking,
        "oof_robust_bedroc": robust,
        "primary_only_geometric_metrics": primary_only_metrics,
        "geometric_consensus_boundary": (
            "Raw signed geometric consensus is undefined for the seed0 and "
            "seed1 matrices because each contains a finite positive-energy "
            "outlier among 16 scores. Geometric consensus is reported on "
            "the all-negative primary median matrix only and is excluded "
            "from seed-robust ranking."
        ),
        "strongest_literature_method": strongest_literature,
        "primary_oof_comparison": {
            "direction": "QUBO exact Top-3 minus direct BEDROC greedy Top-3",
            "bedroc_alpha_20_delta": primary_delta,
            "qubo_exact_better": primary_delta > 0.0,
            "bootstrap": primary_bootstrap,
        },
        "secondary_oof_comparison": {
            "direction": f"QUBO exact Top-3 minus {strongest_literature}",
            "bedroc_alpha_20_delta": (
                float(robust["qubo_exact_top3"]["primary"])
                - float(robust[strongest_literature]["primary"])
            ),
            "bootstrap": literature_bootstrap,
        },
        "full_train_subsets": {
            key: list(value) for key, value in full_subsets.items()
        },
        "full_train_qubo_diagnostics": {
            "exact_energy": full_qubo["exact_energy"],
            "qubo_greedy_energy": full_qubo["qubo_greedy_energy"],
            "qubo_greedy_regret": full_qubo["qubo_greedy_regret"],
            "noncardinality_quadratic": full_qubo[
                "noncardinality_quadratic"
            ],
            "exact_differs_from_qubo_greedy": (
                full_subsets["qubo_exact_top3"]
                != full_subsets["qubo_greedy_top3"]
            ),
            "exact_differs_from_matched_linear": (
                full_subsets["qubo_exact_top3"]
                != full_subsets["matched_linear_top3"]
            ),
        },
        "fold_qubo_diagnostics": {
            fold: {
                "qubo_greedy_regret": value["qubo_greedy_regret"],
                "noncardinality_quadratic": value[
                    "noncardinality_quadratic"
                ],
            }
            for fold, value in fold_qubo.items()
        },
        "data_boundary": {"validation_rows_read": 0, "test_rows_read": 0},
        "outputs": {
            key: output_descriptor(root, path)
            for key, path in output_paths.items()
            if key not in {"result_json", "report_md"}
        },
        "next_gate": "review train-only OOF evidence, then freeze PPARG fresh-validation execution without changing the failed Stage 18e technical record",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(output_paths["result_json"], result)
    write_report(output_paths["report_md"], result, metrics)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def write_report(
    path: Path,
    result: dict[str, object],
    metrics: dict[str, dict[str, dict[str, object]]],
) -> None:
    robust = dict(result["oof_robust_bedroc"])
    lines = [
        "# Stage 19d PPARG Train-668 comparator analysis",
        "",
        "## Scope",
        "",
        "This is scaffold-grouped four-fold out-of-fold development evidence.",
        "The PPARG pair-synergy QUBO v1 weights were transferred without retuning.",
        "No fresh-validation or locked-test row was read.",
        "",
        "## OOF results",
        "",
        "| Method | BEDROC20 | Mean seed | Worst seed | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in result["method_ranking_by_oof_robust_key"]:
        row = dict(robust[method])
        primary = dict(metrics[method]["primary"])
        lines.append(
            f"| {method} | {float(row['primary']):.4f} | "
            f"{float(row['mean_seed']):.4f} | {float(row['worst_seed']):.4f} | "
            f"{float(primary['roc_auc']):.4f} | "
            f"{float(primary['pr_auc_average_precision']):.4f} |"
        )
    for method, primary_value in dict(
        result["primary_only_geometric_metrics"]
    ).items():
        primary = dict(primary_value)
        lines.append(
            f"| {method} (primary only) | "
            f"{float(primary['bedroc_alpha_20']):.4f} | NA | NA | "
            f"{float(primary['roc_auc']):.4f} | "
            f"{float(primary['pr_auc_average_precision']):.4f} |"
        )
    comparison = dict(result["primary_oof_comparison"])
    lines.extend(
        [
            "",
            "## Primary comparison",
            "",
            f"QUBO exact minus direct greedy BEDROC20: {float(comparison['bedroc_alpha_20_delta']):+.4f}.",
            "",
            "## Interpretation",
            "",
            str(result["interpretation_boundary"]),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
