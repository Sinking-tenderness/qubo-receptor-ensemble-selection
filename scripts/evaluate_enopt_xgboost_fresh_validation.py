"""Evaluate frozen EnOpt-style models after the primary validation gate."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import xgboost
from xgboost import XGBClassifier

try:
    from .fit_enopt_xgboost_baseline import (
        MATRIX_IDS,
        SEED_IDS,
        load_train_matrices,
        metrics_for_probabilities,
        normalized_features,
        read_csv,
        read_json,
        robust_bedroc,
        rows_by_id,
        write_csv,
    )
    from .prepare_receptor import file_sha256
except ImportError:
    from fit_enopt_xgboost_baseline import (
        MATRIX_IDS,
        SEED_IDS,
        load_train_matrices,
        metrics_for_probabilities,
        normalized_features,
        read_csv,
        read_json,
        robust_bedroc,
        rows_by_id,
        write_csv,
    )
    from prepare_receptor import file_sha256


def verify_static_inputs(config: dict[str, object]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, value in dict(config["static_inputs"]).items():
        path = Path(str(value["path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(value["sha256"]).upper():
            raise ValueError(f"static input SHA-256 differs: {key}")
        paths[key] = path
    return paths


def load_fresh_matrices(
    aggregate_directory: Path, receptor_ids: list[str]
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, object]]:
    summary_path = aggregate_directory / "summary.json"
    primary_path = aggregate_directory / "primary_median_score_matrix.csv"
    sensitivity_path = aggregate_directory / "sensitivity_minimum_score_matrix.csv"
    long_path = aggregate_directory / "aggregated_seed_scores.csv"
    for path in (summary_path, primary_path, sensitivity_path, long_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = read_json(summary_path)
    outputs = dict(summary["outputs"])
    for key, path in (
        ("primary_median_matrix_csv", primary_path),
        ("sensitivity_minimum_matrix_csv", sensitivity_path),
        ("aggregated_long_csv", long_path),
    ):
        if file_sha256(path) != str(outputs[key]["sha256"]).upper():
            raise ValueError(f"aggregate output SHA-256 differs: {key}")
    matrices = load_train_matrices(
        read_csv(primary_path),
        read_csv(sensitivity_path),
        read_csv(long_path),
        receptor_ids,
    )
    return matrices, summary


def predict_frozen_model(
    model_path: Path,
    feature_order: tuple[str, ...],
    normalization_bounds: dict[str, object],
    matrices: dict[str, dict[str, dict[str, object]]],
) -> dict[str, dict[str, float]]:
    model = XGBClassifier()
    model.load_model(str(model_path))
    output: dict[str, dict[str, float]] = {}
    for matrix_id in MATRIX_IDS:
        ligand_ids = sorted(matrices[matrix_id])
        matrix_bounds = dict(normalization_bounds[matrix_id])
        probabilities = model.predict_proba(
            normalized_features(
                matrices[matrix_id],
                ligand_ids,
                feature_order,
                matrix_bounds,
            )
        )[:, 1]
        output[matrix_id] = {
            ligand_id: float(probability)
            for ligand_id, probability in zip(ligand_ids, probabilities)
        }
    return output


def robust_bedroc_delta(
    left: dict[str, float], right: dict[str, float]
) -> dict[str, float]:
    return {
        key: float(left[key]) - float(right[key])
        for key in ("primary", "sensitivity", "mean_seed", "worst_seed")
    }


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if Path(str(implementation["path"])).resolve() != Path(__file__).resolve():
        raise ValueError("supplementary evaluator path differs from preregistration")
    if file_sha256(Path(__file__)) != str(implementation["sha256"]).upper():
        raise ValueError("supplementary evaluator SHA-256 differs")
    paths = verify_static_inputs(config)
    result_path = Path(str(config["outputs"]["result_json"]))
    prediction_path = Path(str(config["outputs"]["predictions_csv"]))
    metric_path = Path(str(config["outputs"]["metrics_csv"]))
    existing = [
        path for path in (result_path, prediction_path, metric_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError("supplementary validation outputs exist; use --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()

    artifact = read_json(paths["frozen_xgboost_artifact"])
    if artifact.get("status") != (
        "train696_enopt_xgboost_models_frozen_validation_scores_unavailable"
    ):
        raise ValueError("XGBoost models were not frozen before validation")
    required_version = str(artifact["versions"]["xgboost"])
    if xgboost.__version__ != required_version:
        raise ValueError(
            f"XGBoost version {xgboost.__version__} differs from frozen "
            f"version {required_version}"
        )
    if int(artifact["data_boundary"]["validation_scores_read"]) != 0:
        raise ValueError("frozen artifact reports prior validation score access")
    if bool(artifact["model_selection"]["future_refitting_or_retuning_permitted"]):
        raise ValueError("frozen artifact unexpectedly permits model changes")

    main_result_path = Path(str(config["primary_gate"]["result_path"]))
    if not main_result_path.is_file():
        raise FileNotFoundError(
            "primary fresh-validation result must exist before the supplementary evaluator"
        )
    main_result = read_json(main_result_path)
    allowed_statuses = set(config["primary_gate"]["allowed_statuses"])
    if main_result.get("status") not in allowed_statuses:
        raise ValueError("primary fresh-validation result status is not final")
    if str(main_result["config"]["sha256"]) != file_sha256(
        paths["primary_validation_preregistration"]
    ):
        raise ValueError("primary result was not produced from the frozen gate")
    if str(main_result["frozen_model"]["sha256"]) != file_sha256(
        paths["primary_frozen_model"]
    ):
        raise ValueError("primary result used a different frozen model")

    panel_rows = read_csv(paths["fresh_validation_panel"])
    panel = rows_by_id(panel_rows, "fresh validation panel")
    receptor_ids = [str(value) for value in config["receptor_pool"]]
    matrices, aggregate = load_fresh_matrices(
        Path(str(config["aggregate_directory"])), receptor_ids
    )
    expected = dict(config["expected"])
    if (
        aggregate.get("status") != "ok"
        or int(aggregate["ligand_count"]) != int(expected["ligand_count"])
        or int(aggregate["receptor_count"]) != len(receptor_ids)
        or int(aggregate["seed_count"]) != len(SEED_IDS)
        or int(aggregate["aggregated_pair_count"])
        != int(expected["ligand_count"]) * len(receptor_ids)
        or int(aggregate.get("locked_test_manifest_rows", -1)) != 0
    ):
        raise ValueError("fresh-validation aggregate did not pass admission")
    if Counter(row["label"] for row in panel_rows) != Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    ):
        raise ValueError("fresh-validation label counts differ")
    if any(
        row["split"] != "validation"
        or row["selection_role"] != "fresh_validation_preregistered"
        for row in panel_rows
    ):
        raise ValueError("fresh-validation panel boundary differs")
    for matrix_id, matrix in matrices.items():
        if set(matrix) != set(panel):
            raise ValueError(f"panel and {matrix_id} ligand IDs differ")
        for ligand_id, row in matrix.items():
            if row["label"] != panel[ligand_id]["label"]:
                raise ValueError(f"label differs: {matrix_id}/{ligand_id}")

    predictions: dict[str, dict[str, dict[str, float]]] = {}
    metrics: dict[str, dict[str, dict[str, object]]] = {}
    model_details: dict[str, dict[str, object]] = {}
    for method in config["methods"]:
        details = dict(artifact["models"][method])
        model_path = Path(str(details["model_path"]))
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if file_sha256(model_path) != str(details["model_sha256"]).upper():
            raise ValueError(f"frozen model SHA-256 differs: {method}")
        features = tuple(str(value) for value in details["feature_order"])
        if not set(features).issubset(receptor_ids):
            raise ValueError(f"model uses an unauthorized receptor: {method}")
        predictions[method] = predict_frozen_model(
            model_path,
            features,
            dict(details["normalization_bounds"]),
            matrices,
        )
        metrics[method] = {
            matrix_id: metrics_for_probabilities(
                matrices[matrix_id], matrix_predictions
            )
            for matrix_id, matrix_predictions in predictions[method].items()
        }
        model_details[method] = {
            "model_path": model_path.as_posix(),
            "model_sha256": file_sha256(model_path),
            "feature_order": list(features),
            "parameters": details["parameters"],
        }

    robust = {
        method: robust_bedroc(method_metrics)
        for method, method_metrics in metrics.items()
    }
    primary_qubo_robust = {
        key: float(value)
        for key, value in dict(main_result["robust_bedroc"])[
            "pair_synergy_qubo"
        ].items()
    }
    comparisons = {
        method: {
            "xgboost_minus_qubo": robust_bedroc_delta(
                method_robust, primary_qubo_robust
            ),
            "qubo_minus_xgboost": robust_bedroc_delta(
                primary_qubo_robust, method_robust
            ),
        }
        for method, method_robust in robust.items()
    }

    prediction_rows = [
        {
            "method": method,
            "matrix": matrix_id,
            "ligand_id": ligand_id,
            "label": panel[ligand_id]["label"],
            "split_group_id": panel[ligand_id]["split_group_id"],
            "active_probability": probability,
            "ranking_score": probability,
        }
        for method in config["methods"]
        for matrix_id in MATRIX_IDS
        for ligand_id, probability in sorted(
            predictions[method][matrix_id].items()
        )
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
        for method, by_matrix in metrics.items()
        for matrix_id, metric in by_matrix.items()
    ]
    write_csv(prediction_path, prediction_rows)
    write_csv(metric_path, metric_rows)
    result = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": "supplementary_xgboost_validation_evaluated_primary_gate_unchanged",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "frozen_artifact": {
            "path": paths["frozen_xgboost_artifact"].as_posix(),
            "sha256": file_sha256(paths["frozen_xgboost_artifact"]),
        },
        "primary_gate": {
            "path": main_result_path.as_posix(),
            "sha256": file_sha256(main_result_path),
            "status": main_result["status"],
            "all_checks_passed": main_result["all_checks_passed"],
            "unchanged_by_this_supplementary_evaluation": True,
        },
        "data_boundary": {
            "fresh_validation_ligands": len(panel),
            "fresh_validation_label_counts": dict(
                sorted(Counter(row["label"] for row in panel_rows).items())
            ),
            "model_refits": 0,
            "hyperparameter_changes": 0,
            "receptor_reselections": 0,
            "test_rows_read": 0,
            "test_scores_read": 0,
        },
        "models": model_details,
        "metrics": metrics,
        "robust_bedroc": robust,
        "primary_qubo_robust_bedroc": primary_qubo_robust,
        "bedroc_comparisons": comparisons,
        "outputs": {
            "predictions_csv": {
                "path": prediction_path.as_posix(),
                "sha256": file_sha256(prediction_path),
            },
            "metrics_csv": {
                "path": metric_path.as_posix(),
                "sha256": file_sha256(metric_path),
            },
        },
        "test_status": "locked_unreleased",
        "interpretation_note": config["interpretation_boundary"],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_gate_status": main_result["status"],
                "robust_bedroc": robust,
                "bedroc_comparisons": comparisons,
                "model_refits": 0,
                "test_status": "locked_unreleased",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
