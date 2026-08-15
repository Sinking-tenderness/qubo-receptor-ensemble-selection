"""Fit train-only literature baselines and evaluate fresh MK14 validation."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    from .evaluate_enopt_xgboost_fresh_validation import load_fresh_matrices
    from .evaluate_stage05_mk14_fresh_validation import quantile, sampled_bedroc
    from .fit_enopt_xgboost_baseline import (
        MATRIX_IDS,
        METRIC_IDS,
        SEED_IDS,
        audit_inputs,
        labels_for,
        load_train_matrices,
        metrics_for_probabilities,
        minmax_bounds,
        normalized_features,
        read_csv,
        read_json,
        robust_bedroc,
        rows_by_id,
        write_csv,
    )
    from .prepare_receptor import file_sha256
    from .run_stage05_mk14_method_gate import make_frozen_group_folds
except ImportError:
    from evaluate_enopt_xgboost_fresh_validation import load_fresh_matrices
    from evaluate_stage05_mk14_fresh_validation import quantile, sampled_bedroc
    from fit_enopt_xgboost_baseline import (
        MATRIX_IDS,
        METRIC_IDS,
        SEED_IDS,
        audit_inputs,
        labels_for,
        load_train_matrices,
        metrics_for_probabilities,
        minmax_bounds,
        normalized_features,
        read_csv,
        read_json,
        robust_bedroc,
        rows_by_id,
        write_csv,
    )
    from prepare_receptor import file_sha256
    from run_stage05_mk14_method_gate import make_frozen_group_folds


SUPERVISED_METHODS = (
    "ricci_lr_all5",
    "ricci_gbt_all5",
    "ricci_gbt_rfe3",
    "edock_rf_all5",
    "edock_rf_rfe3",
)
ALL5_CONSENSUS_METHODS = (
    "consensus_min_all5",
    "consensus_mean_all5",
    "consensus_geometric_all5",
)
HANTZ_METHODS = (
    "hantz_auc_top3_min",
    "hantz_auc_top3_mean",
    "hantz_auc_top3_geometric",
)
METHODS = SUPERVISED_METHODS + ALL5_CONSENSUS_METHODS + HANTZ_METHODS


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


def verify_file_hashes(
    config: dict[str, object], paths: dict[str, Path]
) -> None:
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = str(dict(config["inputs"])[key]["sha256"]).upper()
        if file_sha256(path) != expected:
            raise ValueError(f"input SHA-256 differs: {key}")


def ensure_output_boundary(
    outputs: dict[str, Path], methods: Iterable[str], overwrite: bool
) -> None:
    run_directory = outputs["run_directory"]
    model_directory = run_directory / "models"
    files = [
        path
        for key, path in outputs.items()
        if key != "run_directory"
    ] + [model_directory / f"{method}.joblib" for method in methods]
    existing = [path for path in files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("literature baseline outputs exist; use --overwrite")
    if overwrite:
        for path in existing:
            if path.is_file():
                path.unlink()
    model_directory.mkdir(parents=True, exist_ok=True)


def signed_geometric_mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    if not numbers or any(not math.isfinite(value) for value in numbers):
        raise ValueError("geometric mean requires finite values")
    if any(value == 0.0 for value in numbers):
        return 0.0
    negative_count = sum(value < 0.0 for value in numbers)
    if negative_count == len(numbers):
        return -math.exp(
            statistics.fmean(math.log(-value) for value in numbers)
        )
    if negative_count == 0:
        return math.exp(statistics.fmean(math.log(value) for value in numbers))
    if 0 < negative_count < len(numbers) and len(numbers) % 2 == 0:
        raise ValueError("mixed-sign geometric mean requires an odd ensemble")
    sign = -1.0 if negative_count % 2 else 1.0
    return sign * math.exp(
        statistics.fmean(math.log(abs(value)) for value in numbers)
    )


def consensus_strategy(method: str) -> str:
    if method.endswith("_min") or "_min_" in method:
        return "min"
    if method.endswith("_mean") or "_mean_" in method:
        return "mean"
    if method.endswith("_geometric") or "_geometric_" in method:
        return "geometric"
    raise ValueError(f"cannot infer consensus strategy: {method}")


def consensus_ranking_scores(
    matrix: dict[str, dict[str, object]],
    ligand_ids: list[str],
    receptor_ids: tuple[str, ...],
    strategy: str,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for ligand_id in ligand_ids:
        values = [float(matrix[ligand_id][receptor]) for receptor in receptor_ids]
        if strategy == "min":
            docking_score = min(values)
        elif strategy == "mean":
            docking_score = statistics.fmean(values)
        elif strategy == "geometric":
            docking_score = signed_geometric_mean(values)
        else:
            raise ValueError(f"unknown consensus strategy: {strategy}")
        output[ligand_id] = -docking_score
    return output


def singleton_auc_values(
    primary: dict[str, dict[str, object]],
    ligand_ids: list[str],
    receptor_ids: list[str],
) -> dict[str, float]:
    labels = labels_for(primary, ligand_ids)
    return {
        receptor: float(
            roc_auc_score(
                labels,
                [-float(primary[ligand_id][receptor]) for ligand_id in ligand_ids],
            )
        )
        for receptor in receptor_ids
    }


def select_hantz_top_k(
    primary: dict[str, dict[str, object]],
    ligand_ids: list[str],
    receptor_ids: list[str],
    budget: int,
) -> tuple[tuple[str, ...], dict[str, float]]:
    auc_values = singleton_auc_values(primary, ligand_ids, receptor_ids)
    ordered = sorted(
        receptor_ids,
        key=lambda receptor: (-auc_values[receptor], receptor),
    )
    return tuple(sorted(ordered[:budget])), auc_values


def model_family(method: str) -> str:
    if method.startswith("ricci_lr_"):
        return "logistic_regression"
    if method.startswith("ricci_gbt_"):
        return "gradient_boosting"
    if method.startswith("edock_rf_"):
        return "random_forest"
    raise ValueError(f"unknown supervised method: {method}")


def build_classifier(
    method: str, model_config: dict[str, object], seed: int
) -> Any:
    family = model_family(method)
    parameters = dict(dict(model_config[family])["parameters"])
    if family == "logistic_regression":
        return LogisticRegression(**parameters, random_state=seed)
    if family == "gradient_boosting":
        return GradientBoostingClassifier(**parameters, random_state=seed)
    return RandomForestClassifier(**parameters, random_state=seed, n_jobs=1)


def fit_supervised_method(
    method: str,
    primary: dict[str, dict[str, object]],
    train_ids: list[str],
    receptor_ids: list[str],
    model_config: dict[str, object],
) -> tuple[Any, tuple[str, ...]]:
    all_features = tuple(receptor_ids)
    seed = int(model_config["model_seed"])
    budget = int(model_config["budget_matched_feature_count"])
    labels = labels_for(primary, train_ids)
    if method.endswith("_rfe3"):
        all_bounds = minmax_bounds(primary, train_ids, all_features)
        selector = RFE(
            estimator=build_classifier(method, model_config, seed),
            n_features_to_select=budget,
            step=1,
        )
        selector.fit(
            normalized_features(primary, train_ids, all_features, all_bounds),
            labels,
        )
        subset = tuple(
            receptor
            for receptor, keep in zip(all_features, selector.support_)
            if bool(keep)
        )
    else:
        subset = all_features
    if len(subset) not in {budget, len(all_features)}:
        raise RuntimeError(f"unexpected feature count for {method}: {len(subset)}")
    bounds = minmax_bounds(primary, train_ids, subset)
    model = build_classifier(method, model_config, seed)
    model.fit(normalized_features(primary, train_ids, subset, bounds), labels)
    return model, subset


def predict_supervised_method(
    model: Any,
    train_matrix: dict[str, dict[str, object]],
    evaluation_matrix: dict[str, dict[str, object]],
    train_ids: list[str],
    evaluation_ids: list[str],
    subset: tuple[str, ...],
) -> dict[str, float]:
    bounds = minmax_bounds(train_matrix, train_ids, subset)
    probabilities = model.predict_proba(
        normalized_features(evaluation_matrix, evaluation_ids, subset, bounds)
    )[:, 1]
    return {
        ligand_id: float(probability)
        for ligand_id, probability in zip(evaluation_ids, probabilities)
    }


def method_predictions(
    method: str,
    matrix: dict[str, dict[str, object]],
    ligand_ids: list[str],
    receptor_ids: list[str],
    hantz_subset: tuple[str, ...],
    supervised_model: Any | None = None,
    supervised_subset: tuple[str, ...] | None = None,
    train_matrix: dict[str, dict[str, object]] | None = None,
    train_ids: list[str] | None = None,
) -> tuple[dict[str, float], tuple[str, ...]]:
    if method in SUPERVISED_METHODS:
        if (
            supervised_model is None
            or supervised_subset is None
            or train_matrix is None
            or train_ids is None
        ):
            raise ValueError(f"supervised inputs missing: {method}")
        return (
            predict_supervised_method(
                supervised_model,
                train_matrix,
                matrix,
                train_ids,
                ligand_ids,
                supervised_subset,
            ),
            supervised_subset,
        )
    subset = tuple(receptor_ids) if method in ALL5_CONSENSUS_METHODS else hantz_subset
    return (
        consensus_ranking_scores(
            matrix,
            ligand_ids,
            subset,
            consensus_strategy(method),
        ),
        subset,
    )


def average_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        metric: statistics.fmean(float(row[metric]) for row in rows)
        for metric in METRIC_IDS
    }


def paired_group_bootstrap_delta(
    left_records: dict[str, dict[str, object]],
    right_records: dict[str, dict[str, object]],
    group_by_ligand: dict[str, str],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    ligand_ids = set(left_records)
    if set(right_records) != ligand_ids or set(group_by_ligand) != ligand_ids:
        raise ValueError("bootstrap inputs contain different ligand IDs")
    grouped_ids: dict[str, list[str]] = defaultdict(list)
    for ligand_id, group_id in group_by_ligand.items():
        grouped_ids[group_id].append(ligand_id)
    for values in grouped_ids.values():
        values.sort()
    group_ids = sorted(grouped_ids)
    rng = random.Random(seed)
    deltas: list[float] = []
    attempts = 0
    while len(deltas) < replicates:
        attempts += 1
        if attempts > replicates * 2:
            raise ValueError("too many bootstrap samples lacked both labels")
        sampled = rng.choices(group_ids, k=len(group_ids))
        sampled_labels = {
            str(left_records[ligand_id]["label"])
            for group_id in sampled
            for ligand_id in grouped_ids[group_id]
        }
        if sampled_labels != {"active", "decoy"}:
            continue
        left = sampled_bedroc(left_records, grouped_ids, sampled)
        right = sampled_bedroc(right_records, grouped_ids, sampled)
        if math.isfinite(left) and math.isfinite(right):
            deltas.append(left - right)
    return {
        "unit": "split_group_id block",
        "seed": seed,
        "valid_replicates": len(deltas),
        "attempts": attempts,
        "confidence_level": 0.95,
        "direction": "left BEDROC20 minus right BEDROC20",
        "mean": statistics.fmean(deltas),
        "lower_95pct": quantile(deltas, 0.025),
        "upper_95pct": quantile(deltas, 0.975),
        "positive_fraction": sum(value > 0.0 for value in deltas) / len(deltas),
    }


def run_train_oof(
    matrices: dict[str, dict[str, dict[str, object]]],
    manifest_rows: list[dict[str, str]],
    receptor_ids: list[str],
    config: dict[str, object],
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, dict[str, object]]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    cv = dict(config["cross_validation"])
    fold_count = int(cv["fold_count"])
    assignments = make_frozen_group_folds(
        manifest_rows, fold_count, int(cv["fold_seed"])
    )
    ligand_ids = sorted(assignments)
    model_config = dict(config["models"])
    budget = int(model_config["budget_matched_feature_count"])
    oof: dict[str, dict[str, dict[str, float]]] = {
        method: {matrix_id: {} for matrix_id in MATRIX_IDS}
        for method in METHODS
    }
    outer_rows: list[dict[str, object]] = []
    for outer_fold in range(fold_count):
        validation_ids = sorted(
            ligand_id
            for ligand_id, fold in assignments.items()
            if fold == outer_fold
        )
        train_ids = sorted(set(ligand_ids) - set(validation_ids))
        hantz_subset, _ = select_hantz_top_k(
            matrices["primary"], train_ids, receptor_ids, budget
        )
        fitted = {
            method: fit_supervised_method(
                method,
                matrices["primary"],
                train_ids,
                receptor_ids,
                model_config,
            )
            for method in SUPERVISED_METHODS
        }
        for method in METHODS:
            fold_primary_metrics: dict[str, object] | None = None
            selected_subset: tuple[str, ...] | None = None
            for matrix_id in MATRIX_IDS:
                model, model_subset = fitted.get(method, (None, None))
                predictions, selected_subset = method_predictions(
                    method,
                    matrices[matrix_id],
                    validation_ids,
                    receptor_ids,
                    hantz_subset,
                    model,
                    model_subset,
                    matrices[matrix_id],
                    train_ids,
                )
                overlap = set(oof[method][matrix_id]) & set(predictions)
                if overlap:
                    raise ValueError(f"duplicate OOF predictions: {method}/{matrix_id}")
                oof[method][matrix_id].update(predictions)
                if matrix_id == "primary":
                    fold_primary_metrics = metrics_for_probabilities(
                        matrices[matrix_id], predictions
                    )
            assert fold_primary_metrics is not None and selected_subset is not None
            outer_rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": method,
                    "train_ligand_count": len(train_ids),
                    "validation_ligand_count": len(validation_ids),
                    "selected_subset": "+".join(selected_subset),
                    **{
                        f"primary_{metric}": fold_primary_metrics[metric]
                        for metric in METRIC_IDS
                    },
                }
            )
    if any(
        set(oof[method][matrix_id]) != set(ligand_ids)
        for method in METHODS
        for matrix_id in MATRIX_IDS
    ):
        raise ValueError("OOF prediction coverage is incomplete")
    metrics = {
        method: {
            matrix_id: metrics_for_probabilities(
                matrices[matrix_id], predictions
            )
            for matrix_id, predictions in by_matrix.items()
        }
        for method, by_matrix in oof.items()
    }
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
    return oof, metrics, fold_rows, outer_rows


def fit_final_methods(
    matrices: dict[str, dict[str, dict[str, object]]],
    receptor_ids: list[str],
    config: dict[str, object],
    model_directory: Path,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[str, ...]],
    dict[str, object],
]:
    train_ids = sorted(matrices["primary"])
    model_config = dict(config["models"])
    budget = int(model_config["budget_matched_feature_count"])
    hantz_subset, singleton_auc = select_hantz_top_k(
        matrices["primary"], train_ids, receptor_ids, budget
    )
    models: dict[str, Any] = {}
    subsets: dict[str, tuple[str, ...]] = {}
    evidence: dict[str, object] = {}
    for method in SUPERVISED_METHODS:
        model, subset = fit_supervised_method(
            method,
            matrices["primary"],
            train_ids,
            receptor_ids,
            model_config,
        )
        path = model_directory / f"{method}.joblib"
        joblib.dump(model, path, compress=3)
        models[method] = joblib.load(path)
        subsets[method] = subset
        evidence[method] = {
            "family": model_family(method),
            "subset": list(subset),
            "receptor_count": len(subset),
            "parameters": dict(
                dict(model_config[model_family(method)])["parameters"]
            ),
            "model_path": path.as_posix(),
            "model_sha256": file_sha256(path),
        }
    for method in ALL5_CONSENSUS_METHODS:
        subsets[method] = tuple(receptor_ids)
        evidence[method] = {
            "family": "ricci_consensus",
            "strategy": consensus_strategy(method),
            "subset": receptor_ids,
            "receptor_count": len(receptor_ids),
        }
    for method in HANTZ_METHODS:
        subsets[method] = hantz_subset
        evidence[method] = {
            "family": "hantz_auc_top3",
            "strategy": consensus_strategy(method),
            "subset": list(hantz_subset),
            "receptor_count": len(hantz_subset),
            "train_singleton_roc_auc": singleton_auc,
        }
    return models, subsets, evidence


def audit_fresh_validation(
    matrices: dict[str, dict[str, dict[str, object]]],
    aggregate: dict[str, object],
    panel_rows: list[dict[str, str]],
    receptor_ids: list[str],
    expected: dict[str, object],
) -> dict[str, dict[str, str]]:
    panel = rows_by_id(panel_rows, "fresh validation panel")
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
        if any(matrix[ligand_id]["label"] != panel[ligand_id]["label"] for ligand_id in panel):
            raise ValueError(f"label differs in {matrix_id}")
    return panel


def flatten_metric_rows(
    metrics: dict[str, dict[str, dict[str, object]]],
    method_evidence: dict[str, object],
) -> list[dict[str, object]]:
    return [
        {
            "method": method,
            "family": dict(method_evidence[method])["family"],
            "matrix": matrix_id,
            "receptor_count": dict(method_evidence[method])["receptor_count"],
            "subset": "+".join(dict(method_evidence[method])["subset"]),
            **{
                key: value
                for key, value in metric.items()
                if key != "top10_ligand_ids"
            },
        }
        for method, by_matrix in metrics.items()
        for matrix_id, metric in by_matrix.items()
    ]


def reference_primary_rows(
    primary_result: dict[str, object],
    xgboost_result: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    frozen = dict(primary_result["frozen_methods"])
    for method, by_matrix in dict(primary_result["method_metrics"]).items():
        metric = dict(dict(by_matrix)["primary"])
        subset = [str(value) for value in frozen[method]]
        rows.append(
            {
                "method": method,
                "evidence_timing": "preregistered_primary",
                "family": "primary_receptor_selection",
                "receptor_count": len(subset),
                "subset": "+".join(subset),
                **{metric_id: metric[metric_id] for metric_id in METRIC_IDS},
            }
        )
    models = dict(xgboost_result["models"])
    for method, by_matrix in dict(xgboost_result["metrics"]).items():
        metric = dict(dict(by_matrix)["primary"])
        subset = [str(value) for value in dict(models[method])["feature_order"]]
        rows.append(
            {
                "method": method,
                "evidence_timing": "preregistered_supplementary",
                "family": "enopt_xgboost",
                "receptor_count": len(subset),
                "subset": "+".join(subset),
                **{metric_id: metric[metric_id] for metric_id in METRIC_IDS},
            }
        )
    return rows


def write_report(
    path: Path,
    combined_rows: list[dict[str, object]],
    new_methods: set[str],
    config: dict[str, object],
    exploratory_bootstrap: dict[str, object],
) -> None:
    qubo = next(row for row in combined_rows if row["method"] == "pair_synergy_qubo")
    best_new = min(
        (row for row in combined_rows if row["method"] in new_methods),
        key=lambda row: -float(row["bedroc_alpha_20"]),
    )
    lines = [
        "# MAPK14 Post Hoc Literature Baseline Record",
        "",
        "## Scope and evidence timing",
        "",
        "These literature-family baselines were specified after the fresh-validation",
        "results were already available. The implementation fits and selects using",
        "Train-696 only, but the comparison is post hoc and cannot modify the frozen",
        "primary gate or support a new confirmatory significance claim. The locked test",
        "was not read.",
        "",
        "## Primary fresh-validation comparison",
        "",
        "| Rank | Method | Timing | Receptors | BEDROC20 | ROC-AUC | PR-AUC | Delta vs QUBO |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(combined_rows, 1):
        lines.append(
            "| {rank} | {method} | {timing} | {count} | {bedroc:.4f} | "
            "{roc:.4f} | {pr:.4f} | {delta:+.4f} |".format(
                rank=rank,
                method=row["method"],
                timing=row["evidence_timing"],
                count=row["receptor_count"],
                bedroc=float(row["bedroc_alpha_20"]),
                roc=float(row["roc_auc"]),
                pr=float(row["pr_auc_average_precision"]),
                delta=float(row["bedroc_minus_qubo"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The best newly added baseline was `{best_new['method']}` with "
            f"BEDROC20={float(best_new['bedroc_alpha_20']):.4f}. The frozen QUBO/greedy "
            f"ranking remained at BEDROC20={float(qubo['bedroc_alpha_20']):.4f}, a "
            f"difference of {float(best_new['bedroc_minus_qubo']):+.4f} for the new method "
            "minus QUBO.",
            "",
            "An exploratory, post hoc split-group bootstrap for QUBO minus the best",
            "new RF baseline gave a 95% interval of "
            f"[{float(exploratory_bootstrap['lower_95pct']):.4f}, "
            f"{float(exploratory_bootstrap['upper_95pct']):.4f}]. Because this interval",
            "crosses zero and the comparator was inspected post hoc, it does not support",
            "a claim that QUBO is statistically superior to RF.",
            "",
            "The result tests whether standard linear, tree, feature-selection, and",
            "consensus families explain the validation performance. It does not establish",
            "quantum advantage, and method choice must not be revised using this validation",
            "table before the locked-test protocol is frozen.",
            "",
            "## Fixed literature mapping",
            "",
            "- Ricci-Lopez et al. (JCIM 2021): LR, GBT, RFE, MIN, AVG, and GEO families.",
            "- Chandak et al. / EDock-ML: random-forest score-matrix classifier family.",
            "- Hantz and Lindert (JCIM 2022): top-three receptors by train-only singleton ROC-AUC.",
            "- Swift et al. (JCIM 2016): already represented by the frozen exhaustive, greedy, and linear top-k rows.",
            "",
            "Configuration: `" + str(config["authorization_id"]) + "`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if Path(str(implementation["path"])).resolve() != Path(__file__).resolve():
        raise ValueError("implementation path differs from config")
    if file_sha256(Path(__file__)) != str(implementation["sha256"]).upper():
        raise ValueError("implementation SHA-256 differs from config")
    paths = input_paths(config)
    outputs = output_paths(config)
    verify_file_hashes(config, paths)
    ensure_output_boundary(outputs, METHODS, overwrite)

    receptor_ids = [str(value) for value in config["receptor_pool"]]
    train_matrices = load_train_matrices(
        read_csv(paths["primary_train_matrix"]),
        read_csv(paths["sensitivity_train_matrix"]),
        read_csv(paths["aggregated_train_seed_scores"]),
        receptor_ids,
    )
    manifest_rows = read_csv(paths["train_ligand_manifest"])
    train_decision = read_json(paths["train_qubo_gate_result"])
    audit_inputs(config, train_matrices, manifest_rows, train_decision)

    oof, oof_metrics, fold_rows, outer_rows = run_train_oof(
        train_matrices, manifest_rows, receptor_ids, config
    )
    train_ids = sorted(train_matrices["primary"])
    model_directory = outputs["run_directory"] / "models"
    models, subsets, method_evidence = fit_final_methods(
        train_matrices,
        receptor_ids,
        config,
        model_directory,
    )
    train_fit_artifact = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": "posthoc_train_only_fit_complete_before_validation_load",
        "implementation": {
            "path": Path(__file__).as_posix(),
            "sha256": file_sha256(Path(__file__)),
        },
        "versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "data_boundary": {
            "train_ligand_count": len(train_ids),
            "validation_rows_used_for_fit_or_selection": 0,
            "test_rows_read": 0,
            "historical_fresh_validation_already_available": True,
            "evidence_class": "posthoc_supplementary",
        },
        "methods": method_evidence,
        "train_oof_robust_bedroc": {
            method: robust_bedroc(metric) for method, metric in oof_metrics.items()
        },
    }
    outputs["train_fit_artifact_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["train_fit_artifact_json"].write_text(
        json.dumps(train_fit_artifact, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    primary_result = read_json(paths["primary_validation_result"])
    if primary_result.get("status") not in {
        "fresh_validation_passed_test_locked",
        "fresh_validation_failed_test_locked",
    }:
        raise ValueError("primary validation result is not final")
    if primary_result.get("test_status") != "locked_unreleased":
        raise ValueError("locked test status differs")
    xgboost_result = read_json(paths["xgboost_validation_result"])
    if xgboost_result.get("test_status") != "locked_unreleased":
        raise ValueError("XGBoost result does not preserve the locked test")
    fresh_matrices, aggregate = load_fresh_matrices(
        Path(str(config["fresh_validation_aggregate_directory"])),
        receptor_ids,
    )
    panel_rows = read_csv(paths["fresh_validation_panel"])
    panel = audit_fresh_validation(
        fresh_matrices,
        aggregate,
        panel_rows,
        receptor_ids,
        dict(config["fresh_validation_expected"]),
    )
    validation_ids = sorted(panel)
    predictions: dict[str, dict[str, dict[str, float]]] = {
        method: {} for method in METHODS
    }
    metrics: dict[str, dict[str, dict[str, object]]] = {
        method: {} for method in METHODS
    }
    hantz_subset = subsets[HANTZ_METHODS[0]]
    for method in METHODS:
        for matrix_id in MATRIX_IDS:
            predictions[method][matrix_id], observed_subset = method_predictions(
                method,
                fresh_matrices[matrix_id],
                validation_ids,
                receptor_ids,
                hantz_subset,
                models.get(method),
                subsets.get(method),
                train_matrices[matrix_id],
                train_ids,
            )
            if observed_subset != subsets[method]:
                raise RuntimeError(f"evaluation subset changed: {method}")
            metrics[method][matrix_id] = metrics_for_probabilities(
                fresh_matrices[matrix_id], predictions[method][matrix_id]
            )

    qubo_robust = {
        key: float(value)
        for key, value in dict(primary_result["robust_bedroc"])[
            "pair_synergy_qubo"
        ].items()
    }
    robust = {method: robust_bedroc(value) for method, value in metrics.items()}
    deltas = {
        method: {
            key: float(value[key]) - qubo_robust[key]
            for key in ("primary", "sensitivity", "mean_seed", "worst_seed")
        }
        for method, value in robust.items()
    }
    qubo_score_rows = [
        row
        for row in read_csv(paths["normalized_primary_method_scores"])
        if row["matrix"] == "primary"
        and row["method"] == "pair_synergy_qubo"
    ]
    if len(qubo_score_rows) != len(validation_ids):
        raise ValueError("primary QUBO score coverage differs")
    qubo_records = {
        row["ligand_id"]: {
            "label": row["label"],
            "score": float(row["normalized_ensemble_score"]),
        }
        for row in qubo_score_rows
    }
    rf_records = {
        ligand_id: {
            "label": panel[ligand_id]["label"],
            "score": -score,
        }
        for ligand_id, score in predictions["edock_rf_all5"]["primary"].items()
    }
    bootstrap_config = dict(config["exploratory_bootstrap"])
    exploratory_bootstrap = paired_group_bootstrap_delta(
        qubo_records,
        rf_records,
        {
            ligand_id: panel[ligand_id]["split_group_id"]
            for ligand_id in validation_ids
        },
        int(bootstrap_config["replicates"]),
        int(bootstrap_config["seed"]),
    )
    exploratory_bootstrap.update(
        {
            "left_method": "pair_synergy_qubo",
            "right_method": "edock_rf_all5",
            "point_delta": qubo_robust["primary"]
            - robust["edock_rf_all5"]["primary"],
            "evidence_class": "exploratory_posthoc",
        }
    )

    prediction_rows = [
        {
            "method": method,
            "matrix": matrix_id,
            "ligand_id": ligand_id,
            "label": panel[ligand_id]["label"],
            "split_group_id": panel[ligand_id]["split_group_id"],
            "ranking_score": score,
        }
        for method in METHODS
        for matrix_id in MATRIX_IDS
        for ligand_id, score in sorted(predictions[method][matrix_id].items())
    ]
    oof_rows = [
        {
            "method": method,
            "matrix": matrix_id,
            "ligand_id": ligand_id,
            "label": train_matrices[matrix_id][ligand_id]["label"],
            "ranking_score": score,
        }
        for method in METHODS
        for matrix_id in MATRIX_IDS
        for ligand_id, score in sorted(oof[method][matrix_id].items())
    ]
    write_csv(outputs["fold_assignments_csv"], fold_rows)
    write_csv(outputs["outer_fold_results_csv"], outer_rows)
    write_csv(outputs["oof_predictions_csv"], oof_rows)
    write_csv(
        outputs["oof_metrics_csv"],
        flatten_metric_rows(oof_metrics, method_evidence),
    )
    write_csv(outputs["validation_predictions_csv"], prediction_rows)
    write_csv(
        outputs["validation_metrics_csv"],
        flatten_metric_rows(metrics, method_evidence),
    )

    reference_rows = reference_primary_rows(primary_result, xgboost_result)
    new_rows = [
        {
            "method": method,
            "evidence_timing": "posthoc_train_only_fit",
            "family": dict(method_evidence[method])["family"],
            "receptor_count": dict(method_evidence[method])["receptor_count"],
            "subset": "+".join(dict(method_evidence[method])["subset"]),
            **{
                metric_id: metrics[method]["primary"][metric_id]
                for metric_id in METRIC_IDS
            },
        }
        for method in METHODS
    ]
    qubo_bedroc = next(
        float(row["bedroc_alpha_20"])
        for row in reference_rows
        if row["method"] == "pair_synergy_qubo"
    )
    combined_rows = sorted(
        [
            {
                **row,
                "bedroc_minus_qubo": float(row["bedroc_alpha_20"])
                - qubo_bedroc,
            }
            for row in reference_rows + new_rows
        ],
        key=lambda row: (
            -float(row["bedroc_alpha_20"]),
            -float(row["pr_auc_average_precision"]),
            str(row["method"]),
        ),
    )
    write_csv(outputs["combined_primary_metrics_csv"], combined_rows)

    result = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": "posthoc_literature_baselines_evaluated_test_locked",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "implementation": train_fit_artifact["implementation"],
        "versions": train_fit_artifact["versions"],
        "data_boundary": {
            "train_ligand_count": len(train_ids),
            "fresh_validation_ligand_count": len(validation_ids),
            "validation_rows_used_for_fit_or_selection": 0,
            "historical_fresh_validation_already_available": True,
            "evidence_class": "posthoc_supplementary",
            "test_rows_read": 0,
            "test_status": "locked_unreleased",
        },
        "methods": method_evidence,
        "train_oof_metrics": oof_metrics,
        "train_oof_robust_bedroc": train_fit_artifact[
            "train_oof_robust_bedroc"
        ],
        "fresh_validation_metrics": metrics,
        "fresh_validation_robust_bedroc": robust,
        "fresh_validation_bedroc_delta_vs_qubo": deltas,
        "exploratory_paired_group_bootstrap": exploratory_bootstrap,
        "combined_primary_ranking": combined_rows,
        "outputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in outputs.items()
            if key
            in {
                "train_fit_artifact_json",
                "fold_assignments_csv",
                "outer_fold_results_csv",
                "oof_predictions_csv",
                "oof_metrics_csv",
                "validation_predictions_csv",
                "validation_metrics_csv",
                "combined_primary_metrics_csv",
            }
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    outputs["result_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["result_json"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    write_report(
        outputs["report_md"],
        combined_rows,
        set(METHODS),
        config,
        exploratory_bootstrap,
    )
    result["outputs"]["report_md"] = {
        "path": outputs["report_md"].as_posix(),
        "sha256": file_sha256(outputs["report_md"]),
    }
    result["outputs"]["result_json"] = {
        "path": outputs["result_json"].as_posix(),
        "sha256_before_self_reference": file_sha256(outputs["result_json"]),
    }
    outputs["result_json"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "new_primary_metrics": {
                    method: {
                        metric: metrics[method]["primary"][metric]
                        for metric in (
                            "bedroc_alpha_20",
                            "roc_auc",
                            "pr_auc_average_precision",
                        )
                    }
                    for method in METHODS
                },
                "bedroc_delta_vs_qubo": {
                    method: values["primary"] for method, values in deltas.items()
                },
                "exploratory_qubo_minus_rf_bootstrap": exploratory_bootstrap,
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
