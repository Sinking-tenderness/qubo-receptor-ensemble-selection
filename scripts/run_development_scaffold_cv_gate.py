"""Run nested scaffold CV on development ligands while keeping final test locked."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .cross_validate_ensemble_mvp import paired_bootstrap_delta
    from .normalized_receptor_qubo import (
        build_normalized_terms,
        exact_select,
    )
    from .prepare_receptor import file_sha256
    from .run_receptor_selection_validation_gate import (
        choose_exhaustive_train_best,
        choose_greedy_train,
        compact_metrics,
        normalize_from_train,
        read_csv,
        subset_metrics,
        validate_dataset,
        write_csv,
    )
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from cross_validate_ensemble_mvp import paired_bootstrap_delta
    from normalized_receptor_qubo import build_normalized_terms, exact_select
    from prepare_receptor import file_sha256
    from run_receptor_selection_validation_gate import (
        choose_exhaustive_train_best,
        choose_greedy_train,
        compact_metrics,
        normalize_from_train,
        read_csv,
        subset_metrics,
        validate_dataset,
        write_csv,
    )


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "inputs",
    "input_sha256",
    "receptor_ids",
    "expected",
    "cross_validation",
    "model",
    "acceptance",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_INPUT_KEYS = {
    "primary_matrix",
    "sensitivity_matrix",
    "warning_table",
    "aggregate_summary",
    "split_manifest",
    "split_summary",
}
REQUIRED_OUTPUT_KEYS = {
    "run_directory",
    "fold_assignments_csv",
    "outer_fold_results_csv",
    "method_metrics_csv",
    "oof_scores_csv",
    "final_tuning_trials_csv",
    "candidate_protocol_json",
    "summary_json",
}
METRIC_KEYS = {
    "roc_auc",
    "pr_auc_average_precision",
    "bedroc_alpha_20",
    "EF1%",
    "EF5%",
    "EF10%",
}


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("development CV config must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"development CV config is missing keys: {', '.join(missing)}")
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    receptors = config["receptor_ids"]
    expected = config["expected"]
    cv = config["cross_validation"]
    model = config["model"]
    acceptance = config["acceptance"]
    outputs = config["outputs"]
    if not isinstance(inputs, dict) or not REQUIRED_INPUT_KEYS.issubset(inputs):
        raise ValueError("inputs is missing one or more required paths")
    if not isinstance(hashes, dict) or not REQUIRED_INPUT_KEYS.issubset(hashes):
        raise ValueError("input_sha256 is missing one or more required hashes")
    if (
        not isinstance(receptors, list)
        or len(receptors) < 2
        or len(receptors) != len(set(str(value) for value in receptors))
    ):
        raise ValueError("receptor_ids must contain unique values")
    if not isinstance(expected, dict):
        raise ValueError("expected must be an object")
    for key in (
        "ligand_count",
        "development_ligand_count",
        "locked_test_ligand_count",
    ):
        if int(expected.get(key, 0)) <= 0:
            raise ValueError(f"expected {key} must be positive")
    if (
        int(expected["development_ligand_count"])
        + int(expected["locked_test_ligand_count"])
        != int(expected["ligand_count"])
    ):
        raise ValueError("development and test counts do not sum to ligand_count")
    if not isinstance(cv, dict):
        raise ValueError("cross_validation must be an object")
    if cv.get("development_splits") != ["train", "validation"]:
        raise ValueError("development_splits must be train and validation")
    if cv.get("locked_split") != "test" or cv.get("evaluate_locked_test") is not False:
        raise ValueError("the final test split must remain locked")
    if not isinstance(cv.get("matrices_exclude_locked_split", False), bool):
        raise ValueError("matrices_exclude_locked_split must be boolean")
    if int(cv.get("fold_count", 0)) < 3:
        raise ValueError("fold_count must be at least three")
    if int(cv.get("fold_seed", 0)) <= 0:
        raise ValueError("fold_seed must be positive")
    if cv.get("score_normalization") != "train_minmax":
        raise ValueError("score_normalization must be train_minmax")
    if cv.get("inner_selection_metric") not in METRIC_KEYS:
        raise ValueError("inner_selection_metric is unsupported")
    tie_breakers = cv.get("inner_tie_breakers")
    if (
        not isinstance(tie_breakers, list)
        or any(str(value) not in METRIC_KEYS for value in tie_breakers)
    ):
        raise ValueError("inner_tie_breakers contains an unsupported metric")
    if int(cv.get("bootstrap_iterations", 0)) <= 0 or int(
        cv.get("bootstrap_seed", 0)
    ) <= 0:
        raise ValueError("bootstrap settings must be positive")
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    if model.get("utility_metric") not in {"roc_auc", "bedroc", "ef5"}:
        raise ValueError("utility_metric is unsupported")
    if not 0.0 < float(model.get("coverage_fraction", 0.0)) <= 1.0:
        raise ValueError("coverage_fraction must be in (0, 1]")
    subset_sizes = model.get("subset_sizes")
    if (
        not isinstance(subset_sizes, list)
        or not subset_sizes
        or any(not 1 <= int(value) < len(receptors) for value in subset_sizes)
    ):
        raise ValueError("subset_sizes is invalid")
    aggregations = model.get("aggregation_methods")
    if (
        not isinstance(aggregations, list)
        or not aggregations
        or not set(str(value) for value in aggregations).issubset(
            {"min_score", "mean_score"}
        )
    ):
        raise ValueError("aggregation_methods is invalid")
    if float(model.get("size_penalty", 0.0)) <= 0.0:
        raise ValueError("size_penalty must be positive")
    if model.get("families") != ["coverage_qubo", "discriminative_qubo"]:
        raise ValueError("both preregistered QUBO families are required")
    grids = model.get("weight_grids")
    if not isinstance(grids, dict):
        raise ValueError("weight_grids must be an object")
    for key in (
        "active_coverage",
        "decoy_exposure_discriminative",
        "active_overlap",
        "redundancy",
    ):
        values = grids.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(float(value) < 0.0 for value in values)
        ):
            raise ValueError(f"weight grid {key} is invalid")
    if not isinstance(acceptance, dict):
        raise ValueError("acceptance must be an object")
    if acceptance.get("comparison_method") != "single_best":
        raise ValueError("comparison_method must be single_best")
    for key in (
        "minimum_primary_bedroc_delta",
        "minimum_primary_roc_auc_delta",
        "minimum_primary_pr_auc_delta",
        "minimum_primary_bedroc_bootstrap_ci95_low",
        "minimum_sensitivity_bedroc_delta",
    ):
        float(acceptance[key])
    if acceptance.get("test_release_requires_manual_review") is not True:
        raise ValueError("test release must require manual review")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUT_KEYS.issubset(outputs):
        raise ValueError("outputs is missing one or more required paths")
    return config


def make_scaffold_folds(
    rows: list[dict[str, str]], fold_count: int, seed: int
) -> dict[str, int]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scaffold = row.get("scaffold_smiles", "")
        if not scaffold:
            raise ValueError(f"missing scaffold for {row['ligand_id']}")
        groups[scaffold].append(row)
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    labels = ("active", "decoy")
    totals = {
        label: sum(row["label"] == label for row in rows) for label in labels
    }
    targets = {label: totals[label] / fold_count for label in labels}
    counts = [{label: 0 for label in labels} for _ in range(fold_count)]
    assignment: dict[str, int] = {}
    for _, group in items:
        group_counts = {
            label: sum(row["label"] == label for row in group)
            for label in labels
        }

        def cost(fold: int) -> tuple[float, float, int]:
            projected = [dict(value) for value in counts]
            for label in labels:
                projected[fold][label] += group_counts[label]
            label_distance = sum(
                (
                    (projected[index][label] - targets[label])
                    / max(1.0, targets[label])
                )
                ** 2
                for index in range(fold_count)
                for label in labels
            )
            size_target = len(rows) / fold_count
            size_distance = sum(
                (
                    (sum(projected[index].values()) - size_target)
                    / max(1.0, size_target)
                )
                ** 2
                for index in range(fold_count)
            )
            return label_distance, size_distance, fold

        fold = min(range(fold_count), key=cost)
        for row in group:
            assignment[row["ligand_id"]] = fold
        for label in labels:
            counts[fold][label] += group_counts[label]
    if len(assignment) != len(rows):
        raise ValueError("scaffold fold assignment is incomplete")
    for scaffold, group in groups.items():
        observed = {assignment[row["ligand_id"]] for row in group}
        if len(observed) != 1:
            raise ValueError(f"scaffold crosses folds: {scaffold}")
    for fold in range(fold_count):
        fold_labels = {
            row["label"]
            for row in rows
            if assignment[row["ligand_id"]] == fold
        }
        if fold_labels != set(labels):
            raise ValueError(f"fold {fold} does not contain both labels")
    return assignment


def candidate_configs(model: dict[str, object], family: str) -> list[dict[str, object]]:
    grids = model["weight_grids"]
    assert isinstance(grids, dict)
    output: list[dict[str, object]] = []
    for size in [int(value) for value in model["subset_sizes"]]:
        overlap_values = [0.0] if size == 1 else [
            float(value) for value in grids["active_overlap"]
        ]
        redundancy_values = [0.0] if size == 1 else [
            float(value) for value in grids["redundancy"]
        ]
        decoy_values = (
            [0.0]
            if family == "coverage_qubo"
            else [
                float(value)
                for value in grids["decoy_exposure_discriminative"]
            ]
        )
        aggregations = (
            ["min_score"]
            if size == 1
            else [str(value) for value in model["aggregation_methods"]]
        )
        for active, decoy, overlap, redundancy in itertools.product(
            [float(value) for value in grids["active_coverage"]],
            decoy_values,
            overlap_values,
            redundancy_values,
        ):
            weights = {
                "active_coverage": active,
                "decoy_exposure": decoy,
                "active_overlap": overlap,
                "redundancy": redundancy,
            }
            for aggregation in aggregations:
                output.append(
                    {
                        "family": family,
                        "target_size": size,
                        "aggregation": aggregation,
                        "weights": weights,
                    }
                )
    return output


def make_context(
    train_ids: set[str],
    validation_ids: set[str],
    primary_by_id: dict[str, dict[str, str]],
    sensitivity_by_id: dict[str, dict[str, str]],
    receptor_ids: list[str],
    model: dict[str, object],
) -> dict[str, object]:
    primary_train_raw = [primary_by_id[key] for key in sorted(train_ids)]
    primary_validation_raw = [
        primary_by_id[key] for key in sorted(validation_ids)
    ]
    sensitivity_train_raw = [
        sensitivity_by_id[key] for key in sorted(train_ids)
    ]
    sensitivity_validation_raw = [
        sensitivity_by_id[key] for key in sorted(validation_ids)
    ]
    primary_train, primary_validation, primary_bounds = normalize_from_train(
        primary_train_raw, primary_validation_raw, receptor_ids
    )
    sensitivity_train, sensitivity_validation, sensitivity_bounds = (
        normalize_from_train(
            sensitivity_train_raw,
            sensitivity_validation_raw,
            receptor_ids,
        )
    )
    terms = build_normalized_terms(
        primary_train,
        receptor_ids,
        float(model["coverage_fraction"]),
        str(model["utility_metric"]),
    )
    return {
        "train_ids": sorted(train_ids),
        "validation_ids": sorted(validation_ids),
        "primary_train": primary_train,
        "primary_validation": primary_validation,
        "sensitivity_train": sensitivity_train,
        "sensitivity_validation": sensitivity_validation,
        "primary_bounds": primary_bounds,
        "sensitivity_bounds": sensitivity_bounds,
        "terms": terms,
    }


def fit_config(
    config: dict[str, object],
    context: dict[str, object],
    receptor_ids: list[str],
    model: dict[str, object],
) -> tuple[tuple[str, ...], dict[str, object]]:
    kind = str(config["family"])
    if kind in {"coverage_qubo", "discriminative_qubo"}:
        subset, energy, coefficients = exact_select(
            context["terms"],
            receptor_ids,
            int(config["target_size"]),
            dict(config["weights"]),
            float(model["size_penalty"]),
        )
        return subset, {"energy": energy, "coefficients": coefficients}
    if kind == "exhaustive":
        subset, _ = choose_exhaustive_train_best(
            context["primary_train"],
            receptor_ids,
            int(config["target_size"]),
            str(config["aggregation"]),
            "bedroc_alpha_20",
        )
        return subset, {}
    if kind == "greedy":
        subset, _ = choose_greedy_train(
            context["primary_train"],
            receptor_ids,
            int(config["target_size"]),
            str(config["aggregation"]),
            "bedroc_alpha_20",
        )
        return subset, {}
    if kind == "all_receptors":
        return tuple(receptor_ids), {}
    if kind == "single_best":
        subset, _ = choose_exhaustive_train_best(
            context["primary_train"],
            receptor_ids,
            1,
            "min_score",
            "bedroc_alpha_20",
        )
        return subset, {}
    raise ValueError(f"unsupported candidate family: {kind}")


def mean_metric_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    keys = (
        "roc_auc",
        "pr_auc_average_precision",
        "bedroc_alpha_20",
        "EF1%",
        "EF5%",
        "EF10%",
    )
    return {
        key: statistics.fmean(float(row[key]) for row in rows) for key in keys
    }


def tune_configs(
    configs: list[dict[str, object]],
    contexts: list[dict[str, object]],
    receptor_ids: list[str],
    model: dict[str, object],
    cv: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    trials: list[dict[str, object]] = []
    for config in configs:
        metrics_rows: list[dict[str, object]] = []
        subsets: list[list[str]] = []
        for context in contexts:
            subset, _ = fit_config(config, context, receptor_ids, model)
            subsets.append(list(subset))
            metrics_rows.append(
                subset_metrics(
                    context["primary_validation"],
                    subset,
                    str(config["aggregation"]),
                )
            )
        mean_metrics = mean_metric_rows(metrics_rows)
        selection_values = [
            float(row[str(cv["inner_selection_metric"])])
            for row in metrics_rows
        ]
        trials.append(
            {
                "config": config,
                "mean_validation_metrics": mean_metrics,
                "selection_metric_std": statistics.pstdev(selection_values),
                "inner_subsets": subsets,
            }
        )

    aggregation_order = {"min_score": 0, "mean_score": 1}
    family_order = {
        "single_best": 0,
        "exhaustive": 1,
        "greedy": 2,
        "all_receptors": 3,
        "coverage_qubo": 4,
        "discriminative_qubo": 5,
    }
    metric_order = [
        str(cv["inner_selection_metric"]),
        *[str(value) for value in cv["inner_tie_breakers"]],
    ]

    def key(row: dict[str, object]) -> tuple[object, ...]:
        config = row["config"]
        metrics = row["mean_validation_metrics"]
        assert isinstance(config, dict)
        assert isinstance(metrics, dict)
        weights = config.get("weights", {})
        assert isinstance(weights, dict)
        return (
            *[-float(metrics[metric]) for metric in metric_order],
            float(row["selection_metric_std"]),
            int(config["target_size"]),
            sum(float(value) for value in weights.values()),
            family_order[str(config["family"])],
            aggregation_order[str(config["aggregation"])],
            json.dumps(config, sort_keys=True),
        )

    return min(trials, key=key), trials


def collect_scores(
    rows: list[dict[str, object]],
    subset: tuple[str, ...],
    aggregation: str,
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        scores = [float(row[receptor_id]) for receptor_id in subset]
        score = min(scores) if aggregation == "min_score" else statistics.fmean(scores)
        output[str(row["ligand_id"])] = {
            "label": row["label"],
            "score": score,
        }
    return output


def method_configs(
    model: dict[str, object], receptor_count: int
) -> dict[str, list[dict[str, object]]]:
    if receptor_count < 1:
        raise ValueError("receptor_count must be positive")
    subset_sizes = [int(value) for value in model["subset_sizes"]]
    aggregations = [str(value) for value in model["aggregation_methods"]]
    classical = {
        family: [
            {
                "family": family,
                "target_size": size,
                "aggregation": aggregation,
            }
            for size in subset_sizes
            for aggregation in (["min_score"] if size == 1 else aggregations)
        ]
        for family in ("exhaustive", "greedy")
    }
    return {
        "single_best": [
            {
                "family": "single_best",
                "target_size": 1,
                "aggregation": "min_score",
            }
        ],
        **classical,
        "all_receptors": [
            {
                "family": "all_receptors",
                "target_size": receptor_count,
                "aggregation": aggregation,
            }
            for aggregation in aggregations
        ],
        "coverage_qubo": candidate_configs(model, "coverage_qubo"),
        "discriminative_qubo": candidate_configs(
            model, "discriminative_qubo"
        ),
    }


def flatten_outer_result(row: dict[str, object]) -> dict[str, object]:
    output = {
        "outer_fold": row["outer_fold"],
        "method": row["method"],
        "family": row["selected_config"]["family"],
        "subset": "+".join(row["subset"]),
        "target_size": len(row["subset"]),
        "aggregation": row["selected_config"]["aggregation"],
        "selected_config": json.dumps(row["selected_config"], sort_keys=True),
    }
    for matrix_name in ("primary", "sensitivity"):
        for key, value in row[f"{matrix_name}_outer_metrics"].items():
            output[f"{matrix_name}_{key}"] = value
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    input_hashes = config["input_sha256"]
    expected = config["expected"]
    cv = config["cross_validation"]
    model = config["model"]
    acceptance = config["acceptance"]
    outputs = config["outputs"]
    receptor_ids = [str(value) for value in config["receptor_ids"]]
    assert isinstance(inputs, dict)
    assert isinstance(input_hashes, dict)
    assert isinstance(expected, dict)
    assert isinstance(cv, dict)
    assert isinstance(model, dict)
    assert isinstance(acceptance, dict)
    assert isinstance(outputs, dict)

    input_paths = {key: Path(str(value)) for key, value in inputs.items()}
    for key, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(input_hashes[key]).upper():
            raise ValueError(f"input SHA-256 differs for {key}")

    aggregate_summary = json.loads(
        input_paths["aggregate_summary"].read_text(encoding="ascii")
    )
    split_summary = json.loads(
        input_paths["split_summary"].read_text(encoding="ascii")
    )
    matrices_exclude_locked = bool(cv.get("matrices_exclude_locked_split", False))
    matrix_ligand_count = int(
        expected[
            "development_ligand_count" if matrices_exclude_locked else "ligand_count"
        ]
    )
    if int(aggregate_summary.get("receptor_ligand_pair_count", 0)) != (
        matrix_ligand_count * len(receptor_ids)
    ):
        raise ValueError("aggregate receptor-ligand pair count differs")
    if split_summary.get("scaffold_disjoint") is not True:
        raise ValueError("input split is not scaffold-disjoint")

    primary_rows = read_csv(input_paths["primary_matrix"])
    sensitivity_rows = read_csv(input_paths["sensitivity_matrix"])
    split_rows = read_csv(input_paths["split_manifest"])
    warning_rows = read_csv(input_paths["warning_table"])
    development_splits = set(str(value) for value in cv["development_splits"])
    audit = validate_dataset(
        primary_rows,
        sensitivity_rows,
        split_rows,
        warning_rows,
        receptor_ids,
        expected,
        development_splits if matrices_exclude_locked else None,
    )
    development_manifest = [
        row for row in split_rows if row["split"] in development_splits
    ]
    if len(development_manifest) != int(expected["development_ligand_count"]):
        raise ValueError("development ligand count differs")
    locked_rows = [row for row in split_rows if row["split"] == cv["locked_split"]]
    if len(locked_rows) != int(expected["locked_test_ligand_count"]):
        raise ValueError("locked test ligand count differs")

    assignments = make_scaffold_folds(
        development_manifest,
        int(cv["fold_count"]),
        int(cv["fold_seed"]),
    )
    fold_assignment_rows = [
        {
            "ligand_id": row["ligand_id"],
            "label": row["label"],
            "scaffold_smiles": row["scaffold_smiles"],
            "development_fold": assignments[row["ligand_id"]],
        }
        for row in sorted(development_manifest, key=lambda item: item["ligand_id"])
    ]
    fold_counts = {
        str(fold): {
            label: sum(
                row["label"] == label
                and assignments[row["ligand_id"]] == fold
                for row in development_manifest
            )
            for label in ("active", "decoy")
        }
        for fold in range(int(cv["fold_count"]))
    }

    primary_by_id = {
        row["ligand_id"]: row
        for split in ("train", "validation")
        for row in audit["primary_visible"][split]
    }
    sensitivity_by_id = {
        row["ligand_id"]: row
        for split in ("train", "validation")
        for row in audit["sensitivity_visible"][split]
    }
    development_ids = set(primary_by_id)
    if development_ids != set(assignments):
        raise ValueError("development matrix and fold IDs differ")

    configurations = method_configs(model, len(receptor_ids))
    methods = list(configurations)
    primary_oof = {method: {} for method in methods}
    sensitivity_oof = {method: {} for method in methods}
    outer_results: list[dict[str, object]] = []
    for outer_fold in range(int(cv["fold_count"])):
        outer_ids = {
            ligand_id
            for ligand_id, fold in assignments.items()
            if fold == outer_fold
        }
        outer_train_ids = development_ids - outer_ids
        remaining_folds = [
            fold for fold in range(int(cv["fold_count"])) if fold != outer_fold
        ]
        inner_contexts = []
        for inner_validation_fold in remaining_folds:
            inner_validation_ids = {
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == inner_validation_fold
            }
            inner_train_ids = outer_train_ids - inner_validation_ids
            inner_contexts.append(
                make_context(
                    inner_train_ids,
                    inner_validation_ids,
                    primary_by_id,
                    sensitivity_by_id,
                    receptor_ids,
                    model,
                )
            )
        outer_context = make_context(
            outer_train_ids,
            outer_ids,
            primary_by_id,
            sensitivity_by_id,
            receptor_ids,
            model,
        )
        for method in methods:
            selected_trial, _ = tune_configs(
                configurations[method],
                inner_contexts,
                receptor_ids,
                model,
                cv,
            )
            selected_config = selected_trial["config"]
            subset, fit_details = fit_config(
                selected_config, outer_context, receptor_ids, model
            )
            aggregation = str(selected_config["aggregation"])
            primary_metrics = subset_metrics(
                outer_context["primary_validation"], subset, aggregation
            )
            sensitivity_metrics = subset_metrics(
                outer_context["sensitivity_validation"], subset, aggregation
            )
            primary_oof[method].update(
                collect_scores(
                    outer_context["primary_validation"], subset, aggregation
                )
            )
            sensitivity_oof[method].update(
                collect_scores(
                    outer_context["sensitivity_validation"], subset, aggregation
                )
            )
            outer_results.append(
                {
                    "outer_fold": outer_fold,
                    "method": method,
                    "selected_config": selected_config,
                    "inner_mean_validation_metrics": selected_trial[
                        "mean_validation_metrics"
                    ],
                    "inner_selection_metric_std": selected_trial[
                        "selection_metric_std"
                    ],
                    "subset": list(subset),
                    "fit_details": fit_details,
                    "primary_outer_metrics": primary_metrics,
                    "sensitivity_outer_metrics": sensitivity_metrics,
                }
            )

    primary_metrics = {
        method: compact_metrics(ranked_metrics_with_ids(records))
        for method, records in primary_oof.items()
    }
    sensitivity_metrics = {
        method: compact_metrics(ranked_metrics_with_ids(records))
        for method, records in sensitivity_oof.items()
    }
    family_order = {"coverage_qubo": 0, "discriminative_qubo": 1}
    selected_family = min(
        ("coverage_qubo", "discriminative_qubo"),
        key=lambda family: (
            -float(primary_metrics[family]["bedroc_alpha_20"]),
            -float(primary_metrics[family]["pr_auc_average_precision"]),
            -float(primary_metrics[family]["roc_auc"]),
            family_order[family],
        ),
    )
    bootstrap = paired_bootstrap_delta(
        primary_oof["single_best"],
        primary_oof[selected_family],
        int(cv["bootstrap_iterations"]),
        int(cv["bootstrap_seed"]),
    )
    deltas = {
        "primary_bedroc": float(
            primary_metrics[selected_family]["bedroc_alpha_20"]
        )
        - float(primary_metrics["single_best"]["bedroc_alpha_20"]),
        "primary_roc_auc": float(primary_metrics[selected_family]["roc_auc"])
        - float(primary_metrics["single_best"]["roc_auc"]),
        "primary_pr_auc": float(
            primary_metrics[selected_family]["pr_auc_average_precision"]
        )
        - float(primary_metrics["single_best"]["pr_auc_average_precision"]),
        "sensitivity_bedroc": float(
            sensitivity_metrics[selected_family]["bedroc_alpha_20"]
        )
        - float(sensitivity_metrics["single_best"]["bedroc_alpha_20"]),
    }
    checks = {
        "primary_bedroc_delta": deltas["primary_bedroc"]
        >= float(acceptance["minimum_primary_bedroc_delta"]),
        "primary_roc_auc_delta": deltas["primary_roc_auc"]
        >= float(acceptance["minimum_primary_roc_auc_delta"]),
        "primary_pr_auc_delta": deltas["primary_pr_auc"]
        >= float(acceptance["minimum_primary_pr_auc_delta"]),
        "primary_bedroc_bootstrap_ci95_low": float(
            bootstrap["bedroc_alpha_20"]["ci95_low"]
        )
        >= float(acceptance["minimum_primary_bedroc_bootstrap_ci95_low"]),
        "sensitivity_bedroc_delta": deltas["sensitivity_bedroc"]
        >= float(acceptance["minimum_sensitivity_bedroc_delta"]),
    }
    gate_passed = all(checks.values())

    final_contexts = []
    for validation_fold in range(int(cv["fold_count"])):
        validation_ids = {
            ligand_id
            for ligand_id, fold in assignments.items()
            if fold == validation_fold
        }
        train_ids = development_ids - validation_ids
        final_contexts.append(
            make_context(
                train_ids,
                validation_ids,
                primary_by_id,
                sensitivity_by_id,
                receptor_ids,
                model,
            )
        )
    final_selected_trial, final_trials = tune_configs(
        configurations[selected_family],
        final_contexts,
        receptor_ids,
        model,
        cv,
    )
    full_context = make_context(
        development_ids,
        set(),
        primary_by_id,
        sensitivity_by_id,
        receptor_ids,
        model,
    )
    final_config = final_selected_trial["config"]
    final_subset, final_fit_details = fit_config(
        final_config, full_context, receptor_ids, model
    )

    implementation_path = Path(__file__)
    implementation_record = {
        "path": f"scripts/{implementation_path.name}",
        "sha256": file_sha256(implementation_path),
    }
    dependency_records = [
        {
            "path": f"scripts/{name}",
            "sha256": file_sha256(implementation_path.with_name(name)),
        }
        for name in (
            "normalized_receptor_qubo.py",
            "run_receptor_selection_validation_gate.py",
            "cross_validate_ensemble_mvp.py",
            "compare_receptor_screening.py",
        )
    ]
    candidate_protocol = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "development_gate_passed_manual_review_required"
            if gate_passed
            else "development_gate_rejected_test_locked"
        ),
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "implementation": implementation_record,
        "implementation_dependencies": dependency_records,
        "input_sha256": {
            key: file_sha256(path) for key, path in input_paths.items()
        },
        "selected_family": selected_family,
        "development_oof_metrics": {
            "primary": primary_metrics[selected_family],
            "sensitivity": sensitivity_metrics[selected_family],
        },
        "comparison_to_single": {
            "deltas": deltas,
            "paired_bootstrap": bootstrap,
            "acceptance_checks": checks,
            "gate_passed": gate_passed,
        },
        "final_development_cv_tuning": final_selected_trial,
        "final_refit": {
            "config": final_config,
            "subset": list(final_subset),
            "fit_details": final_fit_details,
            "development_ligand_count": len(development_ids),
        },
        "test_evaluated": False,
        "test_release_requires_manual_review": True,
    }

    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    core_outputs = [
        path for key, path in output_paths.items() if key != "run_directory"
    ]
    existing = [path for path in core_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("development CV outputs exist; review before overwrite")
    if args.overwrite:
        for path in existing:
            path.unlink()
    output_paths["run_directory"].mkdir(parents=True, exist_ok=True)
    write_csv(output_paths["fold_assignments_csv"], fold_assignment_rows)
    write_csv(
        output_paths["outer_fold_results_csv"],
        [flatten_outer_result(row) for row in outer_results],
    )
    method_metric_rows = []
    for matrix_name, metrics_by_method in (
        ("primary", primary_metrics),
        ("sensitivity", sensitivity_metrics),
    ):
        for method, metrics in metrics_by_method.items():
            method_metric_rows.append(
                {"matrix": matrix_name, "method": method, **metrics}
            )
    write_csv(output_paths["method_metrics_csv"], method_metric_rows)
    oof_score_rows = []
    for matrix_name, scores_by_method in (
        ("primary", primary_oof),
        ("sensitivity", sensitivity_oof),
    ):
        for method, records in scores_by_method.items():
            if set(records) != development_ids:
                raise ValueError(
                    f"OOF ligand IDs are incomplete for {matrix_name} / {method}"
                )
            for ligand_id, record in sorted(records.items()):
                oof_score_rows.append(
                    {
                        "matrix": matrix_name,
                        "method": method,
                        "ligand_id": ligand_id,
                        "label": record["label"],
                        "development_fold": assignments[ligand_id],
                        "normalized_ensemble_score": record["score"],
                    }
                )
    write_csv(output_paths["oof_scores_csv"], oof_score_rows)
    write_csv(
        output_paths["final_tuning_trials_csv"],
        [
            {
                "family": row["config"]["family"],
                "target_size": row["config"]["target_size"],
                "aggregation": row["config"]["aggregation"],
                "weights": json.dumps(row["config"].get("weights", {}), sort_keys=True),
                "selection_metric_std": row["selection_metric_std"],
                **{
                    f"mean_validation_{key}": value
                    for key, value in row["mean_validation_metrics"].items()
                },
                "inner_subsets": json.dumps(row["inner_subsets"]),
            }
            for row in final_trials
        ],
    )
    output_paths["candidate_protocol_json"].write_text(
        json.dumps(candidate_protocol, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": candidate_protocol["status"],
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "implementation": implementation_record,
        "implementation_dependencies": dependency_records,
        "inputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in input_paths.items()
        },
        "audit": {
            "development_ligand_count": len(development_ids),
            "locked_test_ligand_count": len(locked_rows),
            "locked_test_scores_evaluated": False,
            "fold_label_counts": fold_counts,
            "scaffold_disjoint_folds": True,
            "seed_warning_pair_count_by_original_split": audit[
                "warning_by_split"
            ],
        },
        "method_oof_metrics": {
            "primary": primary_metrics,
            "sensitivity": sensitivity_metrics,
        },
        "outer_fold_results": outer_results,
        "selected_qubo_family": selected_family,
        "comparison_to_single": candidate_protocol["comparison_to_single"],
        "final_candidate": candidate_protocol["final_refit"],
        "test_lock": {
            "split": "test",
            "ligand_count": len(locked_rows),
            "scores_evaluated": False,
            "metrics_computed": False,
            "release_requires_manual_review": True,
        },
        "outputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in output_paths.items()
            if key not in {"run_directory", "summary_json"}
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    output_paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "selected_qubo_family": selected_family,
                "acceptance_checks": checks,
                "final_subset": list(final_subset),
                "test_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
