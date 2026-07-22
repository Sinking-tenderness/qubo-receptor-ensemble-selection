"""Fit frozen EnOpt-style XGBoost baselines using Train-696 only."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import xgboost
from xgboost import XGBClassifier

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .prepare_receptor import file_sha256
    from .run_stage05_mk14_method_gate import make_frozen_group_folds
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from prepare_receptor import file_sha256
    from run_stage05_mk14_method_gate import make_frozen_group_folds


MATRIX_IDS = ("primary", "sensitivity", "seed0", "seed1", "seed2")
SEED_IDS = ("seed0", "seed1", "seed2")
METRIC_IDS = (
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
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


def rows_by_id(
    rows: list[dict[str, str]], label: str
) -> dict[str, dict[str, str]]:
    output = {row["ligand_id"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{label} contains duplicate ligand IDs")
    return output


def load_train_matrices(
    primary_rows: list[dict[str, str]],
    sensitivity_rows: list[dict[str, str]],
    long_rows: list[dict[str, str]],
    receptor_ids: list[str],
) -> dict[str, dict[str, dict[str, object]]]:
    primary = rows_by_id(primary_rows, "primary matrix")
    sensitivity = rows_by_id(sensitivity_rows, "sensitivity matrix")
    if set(primary) != set(sensitivity):
        raise ValueError("primary and sensitivity matrix IDs differ")
    matrices: dict[str, dict[str, dict[str, object]]] = {
        "primary": {key: dict(value) for key, value in primary.items()},
        "sensitivity": {
            key: dict(value) for key, value in sensitivity.items()
        },
    }
    allowed = set(receptor_ids)
    long_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for row in long_rows:
        if row["receptor_id"] not in allowed:
            continue
        key = (row["ligand_id"], row["receptor_id"])
        if key in long_by_pair:
            raise ValueError(f"duplicate train pair: {key}")
        long_by_pair[key] = row
    expected_pairs = len(primary) * len(receptor_ids)
    if len(long_by_pair) != expected_pairs:
        raise ValueError(
            f"train long matrix has {len(long_by_pair)} required pairs; "
            f"expected {expected_pairs}"
        )
    columns = {
        "seed0": "seed0_representative_score",
        "seed1": "seed1_representative_score",
        "seed2": "seed2_representative_score",
    }
    for matrix_id, column in columns.items():
        matrix: dict[str, dict[str, object]] = {}
        for ligand_id, source in primary.items():
            row: dict[str, object] = {
                "ligand_id": ligand_id,
                "label": source["label"],
                "selection_role": source["selection_role"],
            }
            for receptor_id in receptor_ids:
                row[receptor_id] = float(
                    long_by_pair[(ligand_id, receptor_id)][column]
                )
            matrix[ligand_id] = row
        matrices[matrix_id] = matrix
    return matrices


def input_paths(config: dict[str, object]) -> dict[str, Path]:
    return {
        key: Path(str(value["path"]))
        for key, value in dict(config["inputs"]).items()
    }


def output_paths(config: dict[str, object]) -> dict[str, Path]:
    return {
        key: Path(str(value))
        for key, value in dict(config["outputs"]).items()
    }


def verify_input_hashes(
    config: dict[str, object], paths: dict[str, Path]
) -> None:
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = str(dict(config["inputs"])[key]["sha256"]).upper()
        if file_sha256(path) != expected:
            raise ValueError(f"input SHA-256 differs: {key}")


def audit_inputs(
    config: dict[str, object],
    matrices: dict[str, dict[str, dict[str, object]]],
    manifest_rows: list[dict[str, str]],
    decision: dict[str, object],
) -> None:
    expected = dict(config["expected"])
    receptors = [str(value) for value in config["receptor_pool"]]
    manifest = rows_by_id(manifest_rows, "ligand manifest")
    expected_ids = set(manifest)
    if len(manifest) != int(expected["ligand_count"]):
        raise ValueError("train ligand count differs")
    if Counter(row["label"] for row in manifest_rows) != Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    ):
        raise ValueError("train label counts differ")
    if any(
        row["split"] != expected["allowed_split"]
        or row["selection_role"] != expected["allowed_selection_role"]
        for row in manifest_rows
    ):
        raise ValueError("train split or selection role differs")
    if any(not row.get(str(config["cross_validation"]["group_column"])) for row in manifest_rows):
        raise ValueError("a train row lacks the frozen group column")
    for matrix_id, matrix in matrices.items():
        if set(matrix) != expected_ids:
            raise ValueError(f"{matrix_id} and manifest ligand IDs differ")
        for ligand_id, row in matrix.items():
            if row["label"] != manifest[ligand_id]["label"]:
                raise ValueError(f"label differs: {matrix_id}/{ligand_id}")
            for receptor_id in receptors:
                value = float(row[receptor_id])
                if not math.isfinite(value):
                    raise ValueError(
                        f"non-finite score: {matrix_id}/{ligand_id}/{receptor_id}"
                    )
    required_status = str(config["data_boundary"]["required_train_gate_status"])
    if decision.get("status") != required_status:
        raise ValueError("Train-696 decision status does not authorize fitting")
    if int(decision.get("validation_rows_read", -1)) != 0:
        raise ValueError("source Train-696 gate read validation rows")
    if int(decision.get("test_rows_read", -1)) != 0:
        raise ValueError("source Train-696 gate read test rows")


def minmax_bounds(
    matrix: dict[str, dict[str, object]],
    ligand_ids: Iterable[str],
    receptor_ids: Iterable[str],
) -> dict[str, dict[str, float]]:
    ids = list(ligand_ids)
    if not ids:
        raise ValueError("cannot fit normalization on zero ligands")
    output: dict[str, dict[str, float]] = {}
    for receptor_id in receptor_ids:
        values = [float(matrix[ligand_id][receptor_id]) for ligand_id in ids]
        output[receptor_id] = {
            "minimum": min(values),
            "maximum": max(values),
        }
    return output


def normalized_features(
    matrix: dict[str, dict[str, object]],
    ligand_ids: list[str],
    receptor_ids: tuple[str, ...],
    bounds: dict[str, dict[str, float]],
) -> np.ndarray:
    rows: list[list[float]] = []
    for ligand_id in ligand_ids:
        features: list[float] = []
        for receptor_id in receptor_ids:
            lower = float(bounds[receptor_id]["minimum"])
            upper = float(bounds[receptor_id]["maximum"])
            value = float(matrix[ligand_id][receptor_id])
            features.append(
                0.0 if upper == lower else (value - lower) / (upper - lower)
            )
        rows.append(features)
    return np.asarray(rows, dtype=np.float64)


def labels_for(
    matrix: dict[str, dict[str, object]], ligand_ids: list[str]
) -> np.ndarray:
    return np.asarray(
        [int(matrix[ligand_id]["label"] == "active") for ligand_id in ligand_ids],
        dtype=np.int32,
    )


def parameter_grid(model_config: dict[str, object]) -> list[dict[str, object]]:
    variable = dict(model_config["hyperparameter_grid"])
    keys = list(variable)
    output: list[dict[str, object]] = []
    for values in itertools.product(*(variable[key] for key in keys)):
        params = dict(model_config["fixed_parameters"])
        params.update(dict(zip(keys, values)))
        output.append(params)
    return output


def feature_subsets(
    method: str, receptor_ids: list[str], budget: int
) -> list[tuple[str, ...]]:
    if method == "xgboost_all5":
        return [tuple(receptor_ids)]
    if method == "xgboost_budget3":
        return list(itertools.combinations(receptor_ids, budget))
    raise ValueError(f"unknown model method: {method}")


def build_model(params: dict[str, object], model_seed: int) -> XGBClassifier:
    return XGBClassifier(
        **params,
        random_state=model_seed,
        n_jobs=1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
    )


def fit_primary_model(
    primary: dict[str, dict[str, object]],
    train_ids: list[str],
    receptor_ids: tuple[str, ...],
    params: dict[str, object],
    model_seed: int,
) -> tuple[XGBClassifier, dict[str, dict[str, float]]]:
    bounds = minmax_bounds(primary, train_ids, receptor_ids)
    model = build_model(params, model_seed)
    model.fit(
        normalized_features(primary, train_ids, receptor_ids, bounds),
        labels_for(primary, train_ids),
        verbose=False,
    )
    return model, bounds


def predict_matrix(
    model: XGBClassifier,
    matrix: dict[str, dict[str, object]],
    train_ids: list[str],
    validation_ids: list[str],
    receptor_ids: tuple[str, ...],
) -> dict[str, float]:
    bounds = minmax_bounds(matrix, train_ids, receptor_ids)
    probabilities = model.predict_proba(
        normalized_features(matrix, validation_ids, receptor_ids, bounds)
    )[:, 1]
    return {
        ligand_id: float(probability)
        for ligand_id, probability in zip(validation_ids, probabilities)
    }


def metrics_for_probabilities(
    matrix: dict[str, dict[str, object]], probabilities: dict[str, float]
) -> dict[str, object]:
    records = {
        ligand_id: {
            "label": matrix[ligand_id]["label"],
            "score": -probability,
        }
        for ligand_id, probability in probabilities.items()
    }
    return ranked_metrics_with_ids(records)


def average_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        metric: statistics.fmean(float(row[metric]) for row in rows)
        for metric in METRIC_IDS
    }


def candidate_id(
    method: str, subset: tuple[str, ...], params: dict[str, object]
) -> str:
    payload = json.dumps(
        {"method": method, "subset": subset, "params": params},
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16].upper()


def candidate_trials(
    method: str,
    candidates: list[tuple[tuple[str, ...], dict[str, object]]],
    contexts: list[tuple[list[str], list[str]]],
    primary: dict[str, dict[str, object]],
    model_seed: int,
) -> list[dict[str, object]]:
    trials: list[dict[str, object]] = []
    for subset, params in candidates:
        fold_metrics: list[dict[str, object]] = []
        for train_ids, validation_ids in contexts:
            model, _ = fit_primary_model(
                primary, train_ids, subset, params, model_seed
            )
            probabilities = predict_matrix(
                model,
                primary,
                train_ids,
                validation_ids,
                subset,
            )
            fold_metrics.append(
                metrics_for_probabilities(primary, probabilities)
            )
        mean = average_metrics(fold_metrics)
        trials.append(
            {
                "candidate_id": candidate_id(method, subset, params),
                "method": method,
                "subset": list(subset),
                "parameters": params,
                "mean_validation_metrics": mean,
                "bedroc_population_std": statistics.pstdev(
                    float(row["bedroc_alpha_20"]) for row in fold_metrics
                ),
                "fold_count": len(contexts),
            }
        )
    return trials


def trial_selection_key(trial: dict[str, object]) -> tuple[object, ...]:
    metrics = dict(trial["mean_validation_metrics"])
    params = dict(trial["parameters"])
    return (
        -float(metrics["bedroc_alpha_20"]),
        -float(metrics["pr_auc_average_precision"]),
        -float(metrics["roc_auc"]),
        float(trial["bedroc_population_std"]),
        int(params["max_depth"]),
        int(params["n_estimators"]),
        float(params["learning_rate"]),
        -float(params["min_child_weight"]),
        tuple(trial["subset"]),
        str(trial["candidate_id"]),
    )


def select_trial(trials: list[dict[str, object]]) -> dict[str, object]:
    if not trials:
        raise ValueError("cannot select from zero XGBoost trials")
    return min(trials, key=trial_selection_key)


def flatten_trial(
    scope: str, outer_fold: int | str, trial: dict[str, object], selected: bool
) -> dict[str, object]:
    metrics = dict(trial["mean_validation_metrics"])
    params = dict(trial["parameters"])
    return {
        "scope": scope,
        "outer_fold": outer_fold,
        "method": trial["method"],
        "candidate_id": trial["candidate_id"],
        "selected": selected,
        "subset": "+".join(str(value) for value in trial["subset"]),
        **{f"param_{key}": value for key, value in params.items()},
        **{f"mean_{key}": value for key, value in metrics.items()},
        "bedroc_population_std": trial["bedroc_population_std"],
        "fold_count": trial["fold_count"],
    }


def robust_bedroc(
    metrics: dict[str, dict[str, object]]
) -> dict[str, float]:
    seed_values = [
        float(metrics[matrix_id]["bedroc_alpha_20"])
        for matrix_id in SEED_IDS
    ]
    return {
        "primary": float(metrics["primary"]["bedroc_alpha_20"]),
        "sensitivity": float(metrics["sensitivity"]["bedroc_alpha_20"]),
        "mean_seed": statistics.fmean(seed_values),
        "worst_seed": min(seed_values),
    }


def model_candidates(
    method: str,
    receptor_ids: list[str],
    budget: int,
    params: list[dict[str, object]],
) -> list[tuple[tuple[str, ...], dict[str, object]]]:
    return [
        (subset, values)
        for subset in feature_subsets(method, receptor_ids, budget)
        for values in params
    ]


def ensure_output_boundary(
    outputs: dict[str, Path], overwrite: bool
) -> None:
    files = [path for key, path in outputs.items() if key != "run_directory"]
    existing = [path for path in files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("baseline outputs exist; use --overwrite")
    if overwrite:
        for path in existing:
            if path.is_file():
                path.unlink()
    outputs["run_directory"].mkdir(parents=True, exist_ok=True)


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = read_json(config_path)
    required_xgboost = str(config["xgboost"]["required_version"])
    if xgboost.__version__ != required_xgboost:
        raise ValueError(
            f"XGBoost version {xgboost.__version__} differs from "
            f"required {required_xgboost}"
        )
    paths = input_paths(config)
    outputs = output_paths(config)
    verify_input_hashes(config, paths)
    ensure_output_boundary(outputs, overwrite)

    receptors = [str(value) for value in config["receptor_pool"]]
    matrices = load_train_matrices(
        read_csv(paths["primary_train_matrix"]),
        read_csv(paths["sensitivity_train_matrix"]),
        read_csv(paths["aggregated_seed_scores"]),
        receptors,
    )
    manifest_rows = read_csv(paths["train_ligand_manifest"])
    decision = read_json(paths["train_qubo_gate_result"])
    audit_inputs(config, matrices, manifest_rows, decision)

    cv = dict(config["cross_validation"])
    fold_count = int(cv["outer_fold_count"])
    assignments = make_frozen_group_folds(
        manifest_rows, fold_count, int(cv["fold_seed"])
    )
    ligand_ids = sorted(assignments)
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

    model_config = dict(config["xgboost"])
    params = parameter_grid(model_config)
    model_seed = int(model_config["model_seed"])
    budget = int(model_config["budget_matched_feature_count"])
    methods = ("xgboost_all5", "xgboost_budget3")
    candidates = {
        method: model_candidates(method, receptors, budget, params)
        for method in methods
    }
    oof: dict[str, dict[str, dict[str, float]]] = {
        method: {matrix_id: {} for matrix_id in MATRIX_IDS}
        for method in methods
    }
    trial_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []

    for outer_fold in range(fold_count):
        outer_validation = sorted(
            ligand_id
            for ligand_id, fold in assignments.items()
            if fold == outer_fold
        )
        outer_train = sorted(set(ligand_ids) - set(outer_validation))
        inner_contexts: list[tuple[list[str], list[str]]] = []
        for inner_fold in range(fold_count):
            if inner_fold == outer_fold:
                continue
            inner_validation = sorted(
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == inner_fold
            )
            inner_train = sorted(set(outer_train) - set(inner_validation))
            inner_contexts.append((inner_train, inner_validation))

        for method in methods:
            trials = candidate_trials(
                method,
                candidates[method],
                inner_contexts,
                matrices["primary"],
                model_seed,
            )
            selected = select_trial(trials)
            trial_rows.extend(
                flatten_trial(
                    "nested_outer",
                    outer_fold,
                    trial,
                    trial["candidate_id"] == selected["candidate_id"],
                )
                for trial in trials
            )
            subset = tuple(str(value) for value in selected["subset"])
            selected_params = dict(selected["parameters"])
            fitted, _ = fit_primary_model(
                matrices["primary"],
                outer_train,
                subset,
                selected_params,
                model_seed,
            )
            matrix_metrics: dict[str, dict[str, object]] = {}
            for matrix_id in MATRIX_IDS:
                probabilities = predict_matrix(
                    fitted,
                    matrices[matrix_id],
                    outer_train,
                    outer_validation,
                    subset,
                )
                overlap = set(oof[method][matrix_id]) & set(probabilities)
                if overlap:
                    raise ValueError(
                        f"duplicate OOF predictions: {method}/{matrix_id}"
                    )
                oof[method][matrix_id].update(probabilities)
                matrix_metrics[matrix_id] = metrics_for_probabilities(
                    matrices[matrix_id], probabilities
                )
            outer_rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": method,
                    "train_ligand_count": len(outer_train),
                    "validation_ligand_count": len(outer_validation),
                    "selected_candidate_id": selected["candidate_id"],
                    "selected_subset": "+".join(subset),
                    "selected_parameters": json.dumps(
                        selected_params, sort_keys=True, separators=(",", ":")
                    ),
                    **{
                        f"primary_{metric}": matrix_metrics["primary"][metric]
                        for metric in METRIC_IDS
                    },
                }
            )

    if any(
        set(oof[method][matrix_id]) != set(ligand_ids)
        for method in methods
        for matrix_id in MATRIX_IDS
    ):
        raise ValueError("OOF prediction coverage is incomplete")
    oof_metrics = {
        method: {
            matrix_id: metrics_for_probabilities(
                matrices[matrix_id], probabilities
            )
            for matrix_id, probabilities in by_matrix.items()
        }
        for method, by_matrix in oof.items()
    }
    oof_robust = {
        method: robust_bedroc(metrics)
        for method, metrics in oof_metrics.items()
    }

    final_contexts = []
    for validation_fold in range(fold_count):
        validation_ids = sorted(
            ligand_id
            for ligand_id, fold in assignments.items()
            if fold == validation_fold
        )
        train_ids = sorted(set(ligand_ids) - set(validation_ids))
        final_contexts.append((train_ids, validation_ids))

    final_models: dict[str, dict[str, object]] = {}
    for method in methods:
        trials = candidate_trials(
            method,
            candidates[method],
            final_contexts,
            matrices["primary"],
            model_seed,
        )
        selected = select_trial(trials)
        trial_rows.extend(
            flatten_trial(
                "full_train_selection",
                "all",
                trial,
                trial["candidate_id"] == selected["candidate_id"],
            )
            for trial in trials
        )
        subset = tuple(str(value) for value in selected["subset"])
        selected_params = dict(selected["parameters"])
        fitted, primary_bounds = fit_primary_model(
            matrices["primary"],
            ligand_ids,
            subset,
            selected_params,
            model_seed,
        )
        model_path = outputs[f"{method}_model_json"]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        fitted.save_model(str(model_path))
        all_bounds = {
            matrix_id: minmax_bounds(
                matrices[matrix_id], ligand_ids, subset
            )
            for matrix_id in MATRIX_IDS
        }
        if all_bounds["primary"] != primary_bounds:
            raise RuntimeError("primary normalization bounds changed")
        final_models[method] = {
            "candidate_id": selected["candidate_id"],
            "feature_order": list(subset),
            "feature_count": len(subset),
            "parameters": selected_params,
            "selection_metrics": selected["mean_validation_metrics"],
            "selection_bedroc_population_std": selected[
                "bedroc_population_std"
            ],
            "normalization_bounds": all_bounds,
            "model_path": model_path.as_posix(),
            "model_sha256": file_sha256(model_path),
            "feature_importance_gain_fraction": {
                receptor_id: float(value)
                for receptor_id, value in zip(
                    subset, fitted.feature_importances_.tolist()
                )
            },
        }

    oof_rows = [
        {
            "method": method,
            "matrix": matrix_id,
            "ligand_id": ligand_id,
            "label": matrices[matrix_id][ligand_id]["label"],
            "outer_fold": assignments[ligand_id],
            "active_probability": probability,
            "ranking_score": probability,
        }
        for method in methods
        for matrix_id in MATRIX_IDS
        for ligand_id, probability in sorted(oof[method][matrix_id].items())
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
        for method, values in oof_metrics.items()
        for matrix_id, metric in values.items()
    ]
    write_csv(outputs["fold_assignments_csv"], fold_rows)
    write_csv(outputs["candidate_trials_csv"], trial_rows)
    write_csv(outputs["outer_fold_results_csv"], outer_rows)
    write_csv(outputs["oof_predictions_csv"], oof_rows)
    write_csv(outputs["oof_metrics_csv"], metric_rows)

    output_evidence = {
        key: {
            "path": path.as_posix(),
            "sha256": file_sha256(path),
        }
        for key, path in outputs.items()
        if key
        in {
            "fold_assignments_csv",
            "candidate_trials_csv",
            "outer_fold_results_csv",
            "oof_predictions_csv",
            "oof_metrics_csv",
            "xgboost_all5_model_json",
            "xgboost_budget3_model_json",
        }
    }
    summary = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": "train696_enopt_xgboost_fit_ok_validation_scores_unavailable",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "implementation": {
            "path": Path(__file__).as_posix(),
            "sha256": file_sha256(Path(__file__)),
        },
        "versions": {
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
        },
        "data_boundary": {
            "train_ligand_count": len(ligand_ids),
            "train_label_counts": dict(
                sorted(Counter(row["label"] for row in manifest_rows).items())
            ),
            "validation_rows_read": 0,
            "validation_scores_read": 0,
            "test_rows_read": 0,
            "test_scores_read": 0,
        },
        "cross_validation": {
            "outer_fold_count": fold_count,
            "inner_fold_count_per_outer_split": fold_count - 1,
            "fold_seed": cv["fold_seed"],
            "group_column": cv["group_column"],
            "selection_metric": cv["selection_metric"],
            "hyperparameter_candidate_count": len(params),
            "all5_candidate_count": len(candidates["xgboost_all5"]),
            "budget3_candidate_count": len(candidates["xgboost_budget3"]),
        },
        "oof_metrics": oof_metrics,
        "oof_robust_bedroc": oof_robust,
        "final_models": final_models,
        "outputs": output_evidence,
        "interpretation_note": config["interpretation_boundary"],
    }
    outputs["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    artifact = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": "train696_enopt_xgboost_models_frozen_validation_scores_unavailable",
        "config": summary["config"],
        "implementation": summary["implementation"],
        "versions": summary["versions"],
        "training_inputs": {
            key: {
                "path": path.as_posix(),
                "sha256": file_sha256(path),
            }
            for key, path in paths.items()
        },
        "data_boundary": summary["data_boundary"],
        "model_selection": {
            "selection_scope": "Train-696 grouped nested CV only",
            "primary_metric": cv["selection_metric"],
            "tie_breakers": cv["selection_tie_breakers"],
            "future_refitting_or_retuning_permitted": False,
        },
        "models": final_models,
        "train_oof_metrics": oof_metrics,
        "train_oof_robust_bedroc": oof_robust,
        "summary": {
            "path": outputs["summary_json"].as_posix(),
            "sha256": file_sha256(outputs["summary_json"]),
        },
        "future_validation": config["future_validation"],
        "interpretation_note": config["interpretation_boundary"],
    }
    outputs["frozen_artifact_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["frozen_artifact_json"].write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "oof_robust_bedroc": oof_robust,
                "frozen_models": {
                    method: {
                        "features": model["feature_order"],
                        "selection_metrics": model["selection_metrics"],
                    }
                    for method, model in final_models.items()
                },
                "validation_scores_read": 0,
                "test_scores_read": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
