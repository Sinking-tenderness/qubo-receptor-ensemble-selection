"""Select a receptor-subset candidate while keeping the test split locked."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

from scipy.stats import spearmanr

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .prepare_receptor import file_sha256
    from .select_receptor_baselines import metrics_for_subset
    from .solve_coverage_qubo import (
        combined_coefficients,
        coverage_objective,
        coverage_terms,
    )
    from .solve_qubo_receptor_subset import build_qubo
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from prepare_receptor import file_sha256
    from select_receptor_baselines import metrics_for_subset
    from solve_coverage_qubo import (
        combined_coefficients,
        coverage_objective,
        coverage_terms,
    )
    from solve_qubo_receptor_subset import build_qubo


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "inputs",
    "input_sha256",
    "receptor_ids",
    "expected",
    "selection",
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
    "receptor_metrics_csv",
    "baseline_comparison_csv",
    "qubo_trials_csv",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
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


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("validation-gate config must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"validation-gate config is missing keys: {', '.join(missing)}")

    inputs = config["inputs"]
    hashes = config["input_sha256"]
    receptors = config["receptor_ids"]
    expected = config["expected"]
    selection = config["selection"]
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
        raise ValueError("receptor_ids must contain unique receptor names")
    if not isinstance(expected, dict):
        raise ValueError("expected must be a JSON object")
    if int(expected.get("ligand_count", 0)) <= 0:
        raise ValueError("expected ligand_count must be positive")
    if int(expected.get("seed_warning_count", -1)) < 0:
        raise ValueError("expected seed_warning_count must be nonnegative")
    if not isinstance(expected.get("label_counts"), dict):
        raise ValueError("expected label_counts must be an object")
    if not isinstance(expected.get("split_label_counts"), dict):
        raise ValueError("expected split_label_counts must be an object")
    if not isinstance(selection, dict):
        raise ValueError("selection must be a JSON object")
    if selection.get("selection_split") != "train":
        raise ValueError("selection_split must be train")
    if selection.get("tuning_split") != "validation":
        raise ValueError("tuning_split must be validation")
    if selection.get("locked_split") != "test":
        raise ValueError("locked_split must be test")
    if selection.get("evaluate_locked_test") is not False:
        raise ValueError("evaluate_locked_test must be false for this gate")
    if selection.get("score_normalization") != "train_minmax":
        raise ValueError("score_normalization must be train_minmax")
    subset_sizes = selection.get("subset_sizes")
    if (
        not isinstance(subset_sizes, list)
        or not subset_sizes
        or any(not 1 <= int(size) < len(receptors) for size in subset_sizes)
        or len(subset_sizes) != len(set(int(size) for size in subset_sizes))
    ):
        raise ValueError("subset_sizes must be unique and smaller than receptor count")
    aggregations = selection.get("aggregation_methods")
    if (
        not isinstance(aggregations, list)
        or not aggregations
        or not set(str(value) for value in aggregations).issubset(
            {"min_score", "mean_score"}
        )
    ):
        raise ValueError("aggregation_methods contains an unsupported method")
    if selection.get("utility_metric") not in {"roc_auc", "bedroc", "ef5"}:
        raise ValueError("utility_metric is unsupported")
    if selection.get("train_selection_metric") not in METRIC_KEYS:
        raise ValueError("train_selection_metric is unsupported")
    if selection.get("validation_metric") not in METRIC_KEYS:
        raise ValueError("validation_metric is unsupported")
    tie_breakers = selection.get("validation_tie_breakers")
    if (
        not isinstance(tie_breakers, list)
        or any(str(metric) not in METRIC_KEYS for metric in tie_breakers)
    ):
        raise ValueError("validation_tie_breakers contains an unsupported metric")
    if not 0.0 < float(selection.get("coverage_fraction", 0.0)) <= 1.0:
        raise ValueError("coverage_fraction must be in (0, 1]")
    if float(selection.get("size_penalty", 0.0)) <= 0.0:
        raise ValueError("size_penalty must be positive")
    grids = selection.get("weight_grids")
    if not isinstance(grids, dict):
        raise ValueError("weight_grids must be an object")
    for name in ("coverage", "overlap", "redundancy"):
        values = grids.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(float(value) < 0.0 for value in values)
        ):
            raise ValueError(f"weight grid {name} must contain nonnegative values")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUT_KEYS.issubset(outputs):
        raise ValueError("outputs is missing one or more required paths")
    return config


def validate_dataset(
    primary_rows: list[dict[str, str]],
    sensitivity_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
    warning_rows: list[dict[str, str]],
    receptor_ids: list[str],
    expected: dict[str, object],
    matrix_splits: set[str] | None = None,
) -> dict[str, object]:
    def unique_lookup(
        rows: list[dict[str, str]], name: str
    ) -> dict[str, dict[str, str]]:
        ids = [row["ligand_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} contains duplicate ligand IDs")
        return {row["ligand_id"]: row for row in rows}

    primary = unique_lookup(primary_rows, "primary matrix")
    sensitivity = unique_lookup(sensitivity_rows, "sensitivity matrix")
    splits = unique_lookup(split_rows, "split manifest")
    if len(splits) != int(expected["ligand_count"]):
        raise ValueError("split manifest ligand count differs")
    expected_matrix_ids = (
        set(splits)
        if matrix_splits is None
        else {
            ligand_id
            for ligand_id, row in splits.items()
            if row["split"] in matrix_splits
        }
    )
    if (
        set(primary) != set(sensitivity)
        or set(primary) != expected_matrix_ids
    ):
        raise ValueError("matrix and split ligand ID sets differ")
    if matrix_splits is None:
        expected_labels = {
            str(key): int(value)
            for key, value in dict(expected["label_counts"]).items()
        }
    else:
        expected_labels: dict[str, int] = {}
        expected_split_counts = dict(expected["split_label_counts"])
        for split in matrix_splits:
            for label, count in dict(expected_split_counts[split]).items():
                expected_labels[str(label)] = expected_labels.get(str(label), 0) + int(count)
    observed_labels: dict[str, int] = {}
    for ligand_id, row in primary.items():
        label = row["label"]
        if sensitivity[ligand_id]["label"] != label or splits[ligand_id]["label"] != label:
            raise ValueError(f"label differs across inputs: {ligand_id}")
        observed_labels[label] = observed_labels.get(label, 0) + 1
        for receptor_id in receptor_ids:
            for matrix_name, matrix_row in (
                ("primary", row),
                ("sensitivity", sensitivity[ligand_id]),
            ):
                if matrix_row.get(receptor_id, "") == "":
                    raise ValueError(
                        f"missing {matrix_name} score: {ligand_id} / {receptor_id}"
                    )
                if not math.isfinite(float(matrix_row[receptor_id])):
                    raise ValueError(
                        f"nonfinite {matrix_name} score: {ligand_id} / {receptor_id}"
                    )
    if observed_labels != expected_labels:
        raise ValueError(
            f"label counts differ: expected {expected_labels}, got {observed_labels}"
        )

    split_label_counts: dict[str, dict[str, int]] = {}
    for row in split_rows:
        split = row["split"]
        label = row["label"]
        split_label_counts.setdefault(split, {})
        split_label_counts[split][label] = (
            split_label_counts[split].get(label, 0) + 1
        )
    normalized_expected_splits = {
        str(split): {
            str(label): int(count)
            for label, count in dict(counts).items()
        }
        for split, counts in dict(expected["split_label_counts"]).items()
    }
    if split_label_counts != normalized_expected_splits:
        raise ValueError(
            "split label counts differ: "
            f"expected {normalized_expected_splits}, got {split_label_counts}"
        )
    if len(warning_rows) != int(expected["seed_warning_count"]):
        raise ValueError("seed warning count differs")
    warning_keys: set[tuple[str, str]] = set()
    warning_by_split: dict[str, int] = {}
    for row in warning_rows:
        key = (row["ligand_id"], row["receptor_id"])
        if key in warning_keys:
            raise ValueError(f"duplicate seed warning pair: {key}")
        warning_keys.add(key)
        if row["ligand_id"] not in splits or row["receptor_id"] not in receptor_ids:
            raise ValueError(f"seed warning is absent from matrix inputs: {key}")
        split = splits[row["ligand_id"]]["split"]
        warning_by_split[split] = warning_by_split.get(split, 0) + 1

    split_by_id = {ligand_id: row["split"] for ligand_id, row in splits.items()}
    primary_visible = {
        split: [
            row
            for row in primary_rows
            if split_by_id[row["ligand_id"]] == split
        ]
        for split in ("train", "validation")
    }
    sensitivity_visible = {
        split: [
            row
            for row in sensitivity_rows
            if split_by_id[row["ligand_id"]] == split
        ]
        for split in ("train", "validation")
    }
    return {
        "primary_visible": primary_visible,
        "sensitivity_visible": sensitivity_visible,
        "split_label_counts": split_label_counts,
        "warning_by_split": warning_by_split,
    }


def normalize_from_train(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    receptor_ids: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, float]]]:
    bounds = {
        receptor_id: {
            "minimum": min(float(row[receptor_id]) for row in train_rows),
            "maximum": max(float(row[receptor_id]) for row in train_rows),
        }
        for receptor_id in receptor_ids
    }

    def transform(rows: list[dict[str, str]]) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        for row in rows:
            normalized: dict[str, object] = dict(row)
            for receptor_id in receptor_ids:
                lower = bounds[receptor_id]["minimum"]
                upper = bounds[receptor_id]["maximum"]
                value = float(row[receptor_id])
                normalized[receptor_id] = (
                    0.0 if upper == lower else (value - lower) / (upper - lower)
                )
            output.append(normalized)
        return output

    return transform(train_rows), transform(validation_rows), bounds


def compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    keys = (
        "ligand_count",
        "active_count",
        "roc_auc",
        "pr_auc_average_precision",
        "bedroc_alpha_20",
        "EF1%",
        "EF5%",
        "EF10%",
        "top10_active_count",
    )
    return {key: metrics[key] for key in keys}


def subset_metrics(
    rows: list[dict[str, object]], subset: tuple[str, ...], aggregation: str
) -> dict[str, object]:
    return compact_metrics(metrics_for_subset(rows, subset, aggregation))


def candidate_sort_key(
    metrics: dict[str, object],
    selection_metric: str,
    subset: tuple[str, ...],
) -> tuple[object, ...]:
    return (
        -float(metrics[selection_metric]),
        -float(metrics["pr_auc_average_precision"]),
        -float(metrics["roc_auc"]),
        subset,
    )


def choose_exhaustive_train_best(
    train_rows: list[dict[str, object]],
    receptor_ids: list[str],
    size: int,
    aggregation: str,
    selection_metric: str,
) -> tuple[tuple[str, ...], dict[str, object]]:
    candidates = []
    for subset in itertools.combinations(receptor_ids, size):
        metrics = subset_metrics(train_rows, subset, aggregation)
        candidates.append((subset, metrics))
    return min(
        candidates,
        key=lambda item: candidate_sort_key(item[1], selection_metric, item[0]),
    )


def choose_greedy_train(
    train_rows: list[dict[str, object]],
    receptor_ids: list[str],
    size: int,
    aggregation: str,
    selection_metric: str,
) -> tuple[tuple[str, ...], dict[str, object]]:
    selected: tuple[str, ...] = ()
    while len(selected) < size:
        candidates = []
        for receptor_id in receptor_ids:
            if receptor_id in selected:
                continue
            subset = tuple(sorted((*selected, receptor_id)))
            metrics = subset_metrics(train_rows, subset, aggregation)
            candidates.append((subset, metrics))
        selected, selected_metrics = min(
            candidates,
            key=lambda item: candidate_sort_key(
                item[1], selection_metric, item[0]
            ),
        )
    return selected, selected_metrics


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def random_subset_distribution(
    rows: list[dict[str, object]],
    receptor_ids: list[str],
    size: int,
    aggregation: str,
) -> dict[str, object]:
    entries = [
        subset_metrics(rows, subset, aggregation)
        for subset in itertools.combinations(receptor_ids, size)
    ]
    summary: dict[str, object] = {
        "interpretation": "exact uniform distribution over every fixed-size subset",
        "subset_count": len(entries),
    }
    for metric in (
        "roc_auc",
        "pr_auc_average_precision",
        "bedroc_alpha_20",
        "EF5%",
    ):
        values = [float(entry[metric]) for entry in entries]
        summary[metric] = {
            "minimum": min(values),
            "q05": percentile(values, 0.05),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "q95": percentile(values, 0.95),
            "maximum": max(values),
        }
    return summary


def powerset(receptor_ids: list[str]) -> list[tuple[str, ...]]:
    return [
        subset
        for size in range(len(receptor_ids) + 1)
        for subset in itertools.combinations(receptor_ids, size)
    ]


def solve_qubo_grid(
    train_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    receptor_ids: list[str],
    selection: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    grids = selection["weight_grids"]
    assert isinstance(grids, dict)
    all_subsets = powerset(receptor_ids)
    trials: list[dict[str, object]] = []
    for target_size in [int(value) for value in selection["subset_sizes"]]:
        for coverage_weight, overlap_weight, redundancy_weight in itertools.product(
            [float(value) for value in grids["coverage"]],
            [float(value) for value in grids["overlap"]],
            [float(value) for value in grids["redundancy"]],
        ):
            base = build_qubo(
                train_rows,
                receptor_ids,
                target_size,
                redundancy_weight,
                0.0,
                float(selection["size_penalty"]),
                str(selection["utility_metric"]),
                "minmax",
            )
            _, rewards, overlaps = coverage_terms(
                train_rows,
                receptor_ids,
                float(selection["coverage_fraction"]),
            )
            scored = [
                (
                    coverage_objective(
                        subset,
                        base,
                        rewards,
                        overlaps,
                        coverage_weight,
                        overlap_weight,
                    ),
                    subset,
                )
                for subset in all_subsets
            ]
            objective_value, subset = min(scored, key=lambda item: (item[0], item[1]))
            if len(subset) != target_size:
                raise ValueError(
                    "QUBO size penalty did not enforce the configured target size"
                )
            for aggregation in [
                str(value) for value in selection["aggregation_methods"]
            ]:
                trials.append(
                    {
                        "target_size": target_size,
                        "coverage_weight": coverage_weight,
                        "overlap_weight": overlap_weight,
                        "redundancy_weight": redundancy_weight,
                        "size_penalty": float(selection["size_penalty"]),
                        "aggregation": aggregation,
                        "subset": list(subset),
                        "objective_train": float(objective_value),
                        "train_metrics": subset_metrics(
                            train_rows, subset, aggregation
                        ),
                        "validation_metrics": subset_metrics(
                            validation_rows, subset, aggregation
                        ),
                    }
                )

    aggregation_order = {
        str(value): index
        for index, value in enumerate(selection["aggregation_methods"])
    }
    metrics = [
        str(selection["validation_metric"]),
        *[str(value) for value in selection["validation_tie_breakers"]],
    ]

    def tuning_key(row: dict[str, object]) -> tuple[object, ...]:
        validation = row["validation_metrics"]
        assert isinstance(validation, dict)
        weights = (
            float(row["coverage_weight"]),
            float(row["overlap_weight"]),
            float(row["redundancy_weight"]),
        )
        return (
            *[-float(validation[metric]) for metric in metrics],
            int(row["target_size"]),
            sum(weights),
            weights,
            aggregation_order[str(row["aggregation"])],
            tuple(row["subset"]),
        )

    chosen = min(trials, key=tuning_key)
    return trials, chosen


def rebuild_chosen_qubo(
    train_rows: list[dict[str, object]],
    receptor_ids: list[str],
    selection: dict[str, object],
    chosen: dict[str, object],
) -> dict[str, object]:
    base = build_qubo(
        train_rows,
        receptor_ids,
        int(chosen["target_size"]),
        float(chosen["redundancy_weight"]),
        0.0,
        float(chosen["size_penalty"]),
        str(selection["utility_metric"]),
        "minmax",
    )
    coverage_sets, rewards, overlaps = coverage_terms(
        train_rows,
        receptor_ids,
        float(selection["coverage_fraction"]),
    )
    coefficients = combined_coefficients(
        base,
        rewards,
        overlaps,
        float(chosen["coverage_weight"]),
        float(chosen["overlap_weight"]),
    )
    return {
        "base_qubo": base,
        "coverage_active_ids_train": {
            receptor_id: sorted(values)
            for receptor_id, values in coverage_sets.items()
        },
        "coverage_rewards_train": rewards,
        "coverage_overlaps_train": overlaps,
        "qubo_coefficients": coefficients,
    }


def baseline_record(
    method: str,
    subset: tuple[str, ...],
    aggregation: str,
    train_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "method": method,
        "subset": list(subset),
        "target_size": len(subset),
        "aggregation": aggregation,
        "train_metrics": subset_metrics(train_rows, subset, aggregation),
        "validation_metrics": subset_metrics(
            validation_rows, subset, aggregation
        ),
    }


def flatten_metrics_record(record: dict[str, object]) -> dict[str, object]:
    output = {
        "method": record["method"],
        "subset": "+".join(record["subset"]),
        "target_size": record["target_size"],
        "aggregation": record["aggregation"],
    }
    for matrix_name in ("primary", "sensitivity"):
        for split in ("train", "validation"):
            metrics = record[f"{matrix_name}_{split}_metrics"]
            for key, value in metrics.items():
                output[f"{matrix_name}_{split}_{key}"] = value
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
    selection = config["selection"]
    outputs = config["outputs"]
    receptor_ids = [str(value) for value in config["receptor_ids"]]
    assert isinstance(inputs, dict)
    assert isinstance(input_hashes, dict)
    assert isinstance(expected, dict)
    assert isinstance(selection, dict)
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
    if int(aggregate_summary.get("receptor_ligand_pair_count", 0)) != (
        int(expected["ligand_count"]) * len(receptor_ids)
    ):
        raise ValueError("aggregate receptor-ligand pair count differs")
    if int(aggregate_summary.get("seed_stability_warning_count", -1)) != int(
        expected["seed_warning_count"]
    ):
        raise ValueError("aggregate seed warning count differs")
    if split_summary.get("scaffold_disjoint") is not True:
        raise ValueError("split summary is not scaffold-disjoint")

    primary_rows = read_csv(input_paths["primary_matrix"])
    sensitivity_rows = read_csv(input_paths["sensitivity_matrix"])
    split_rows = read_csv(input_paths["split_manifest"])
    warning_rows = read_csv(input_paths["warning_table"])
    audit = validate_dataset(
        primary_rows,
        sensitivity_rows,
        split_rows,
        warning_rows,
        receptor_ids,
        expected,
    )
    primary_visible = audit["primary_visible"]
    sensitivity_visible = audit["sensitivity_visible"]
    assert isinstance(primary_visible, dict)
    assert isinstance(sensitivity_visible, dict)

    primary_train, primary_validation, primary_bounds = normalize_from_train(
        primary_visible["train"], primary_visible["validation"], receptor_ids
    )
    sensitivity_train, sensitivity_validation, sensitivity_bounds = normalize_from_train(
        sensitivity_visible["train"],
        sensitivity_visible["validation"],
        receptor_ids,
    )

    receptor_metric_records: list[dict[str, object]] = []
    for receptor_id in receptor_ids:
        for matrix_name, train_rows, validation_rows in (
            ("primary", primary_train, primary_validation),
            ("sensitivity", sensitivity_train, sensitivity_validation),
        ):
            for split_name, rows in (
                ("train", train_rows),
                ("validation", validation_rows),
            ):
                receptor_metric_records.append(
                    {
                        "matrix": matrix_name,
                        "split": split_name,
                        "receptor_id": receptor_id,
                        **subset_metrics(rows, (receptor_id,), "min_score"),
                    }
                )

    train_correlations: dict[str, float] = {}
    for first, second in itertools.combinations(receptor_ids, 2):
        value = float(
            spearmanr(
                [float(row[first]) for row in primary_train],
                [float(row[second]) for row in primary_train],
            ).statistic
        )
        train_correlations[f"{first}__{second}"] = value
    coverage_sets, coverage_rewards, coverage_overlaps = coverage_terms(
        primary_train,
        receptor_ids,
        float(selection["coverage_fraction"]),
    )

    baseline_records: list[dict[str, object]] = []
    single_subset, _ = choose_exhaustive_train_best(
        primary_train,
        receptor_ids,
        1,
        "min_score",
        str(selection["train_selection_metric"]),
    )
    baseline_records.append(
        baseline_record(
            "single_best_train",
            single_subset,
            "min_score",
            primary_train,
            primary_validation,
        )
    )
    for size in [int(value) for value in selection["subset_sizes"]]:
        for aggregation in [
            str(value) for value in selection["aggregation_methods"]
        ]:
            exhaustive_subset, _ = choose_exhaustive_train_best(
                primary_train,
                receptor_ids,
                size,
                aggregation,
                str(selection["train_selection_metric"]),
            )
            baseline_records.append(
                baseline_record(
                    "exhaustive_train_best",
                    exhaustive_subset,
                    aggregation,
                    primary_train,
                    primary_validation,
                )
            )
            greedy_subset, _ = choose_greedy_train(
                primary_train,
                receptor_ids,
                size,
                aggregation,
                str(selection["train_selection_metric"]),
            )
            baseline_records.append(
                baseline_record(
                    "greedy_train",
                    greedy_subset,
                    aggregation,
                    primary_train,
                    primary_validation,
                )
            )
    for aggregation in [
        str(value) for value in selection["aggregation_methods"]
    ]:
        baseline_records.append(
            baseline_record(
                "all_receptors",
                tuple(receptor_ids),
                aggregation,
                primary_train,
                primary_validation,
            )
        )

    qubo_trials, chosen = solve_qubo_grid(
        primary_train,
        primary_validation,
        receptor_ids,
        selection,
    )
    chosen_subset = tuple(str(value) for value in chosen["subset"])
    chosen_aggregation = str(chosen["aggregation"])
    baseline_records.append(
        baseline_record(
            "qubo_validation_tuned",
            chosen_subset,
            chosen_aggregation,
            primary_train,
            primary_validation,
        )
    )

    for record in baseline_records:
        subset = tuple(str(value) for value in record["subset"])
        aggregation = str(record["aggregation"])
        record["primary_train_metrics"] = record.pop("train_metrics")
        record["primary_validation_metrics"] = record.pop("validation_metrics")
        record["sensitivity_train_metrics"] = subset_metrics(
            sensitivity_train, subset, aggregation
        )
        record["sensitivity_validation_metrics"] = subset_metrics(
            sensitivity_validation, subset, aggregation
        )

    random_distributions = {
        f"size_{size}_{aggregation}": random_subset_distribution(
            primary_validation,
            receptor_ids,
            size,
            aggregation,
        )
        for size in [int(value) for value in selection["subset_sizes"]]
        for aggregation in [
            str(value) for value in selection["aggregation_methods"]
        ]
    }

    chosen_model = rebuild_chosen_qubo(
        primary_train, receptor_ids, selection, chosen
    )
    implementation_path = Path(__file__)
    implementation_record = {
        "path": f"scripts/{implementation_path.name}",
        "sha256": file_sha256(implementation_path),
    }
    candidate_protocol = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "candidate_selected_test_locked",
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "implementation": implementation_record,
        "input_sha256": {
            key: file_sha256(path) for key, path in input_paths.items()
        },
        "selection_contract": selection,
        "selected_candidate": chosen,
        "selected_qubo_model": chosen_model,
        "test_evaluated": False,
        "test_release_rule": (
            "Run the separate final-test gate only after this candidate protocol "
            "is reviewed and its SHA-256 is frozen."
        ),
    }

    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    core_outputs = [
        path for key, path in output_paths.items() if key != "run_directory"
    ]
    existing = [path for path in core_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("validation-gate outputs exist; use --overwrite after review")
    if args.overwrite:
        for path in existing:
            path.unlink()
    output_paths["run_directory"].mkdir(parents=True, exist_ok=True)

    write_csv(output_paths["receptor_metrics_csv"], receptor_metric_records)
    write_csv(
        output_paths["baseline_comparison_csv"],
        [flatten_metrics_record(record) for record in baseline_records],
    )
    write_csv(
        output_paths["qubo_trials_csv"],
        [
            {
                "target_size": row["target_size"],
                "coverage_weight": row["coverage_weight"],
                "overlap_weight": row["overlap_weight"],
                "redundancy_weight": row["redundancy_weight"],
                "size_penalty": row["size_penalty"],
                "aggregation": row["aggregation"],
                "subset": "+".join(row["subset"]),
                "objective_train": row["objective_train"],
                **{
                    f"train_{key}": value
                    for key, value in row["train_metrics"].items()
                },
                **{
                    f"validation_{key}": value
                    for key, value in row["validation_metrics"].items()
                },
            }
            for row in qubo_trials
        ],
    )
    output_paths["candidate_protocol_json"].write_text(
        json.dumps(candidate_protocol, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    selected_baseline_record = next(
        record
        for record in baseline_records
        if record["method"] == "qubo_validation_tuned"
    )
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "validation_candidate_selected_test_locked",
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "implementation": implementation_record,
        "inputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in input_paths.items()
        },
        "audit": {
            "ligand_count": int(expected["ligand_count"]),
            "receptor_count": len(receptor_ids),
            "split_label_counts": audit["split_label_counts"],
            "seed_warning_count": len(warning_rows),
            "seed_warning_pair_count_by_split": audit["warning_by_split"],
            "scaffold_disjoint": True,
        },
        "test_lock": {
            "split": "test",
            "scores_evaluated": False,
            "metrics_computed": False,
            "label_counts": audit["split_label_counts"]["test"],
        },
        "normalization": {
            "method": "per-receptor min-max fitted on train only",
            "primary_train_bounds": primary_bounds,
            "sensitivity_train_bounds": sensitivity_bounds,
        },
        "train_complementarity": {
            "score_spearman": train_correlations,
            "top_fraction": float(selection["coverage_fraction"]),
            "active_ids_by_receptor": {
                receptor_id: sorted(values)
                for receptor_id, values in coverage_sets.items()
            },
            "active_coverage_reward": coverage_rewards,
            "active_overlap": coverage_overlaps,
            "active_union_count": len(set().union(*coverage_sets.values())),
        },
        "qubo_grid": {
            "trial_count": len(qubo_trials),
            "unique_subset_aggregation_count": len(
                {
                    (tuple(row["subset"]), str(row["aggregation"]))
                    for row in qubo_trials
                }
            ),
            "selected_candidate": chosen,
        },
        "selected_candidate_primary_and_sensitivity": selected_baseline_record,
        "classical_baselines": baseline_records,
        "uniform_random_subset_distributions_on_validation": random_distributions,
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
                "selected_candidate": chosen,
                "candidate_protocol": output_paths[
                    "candidate_protocol_json"
                ].as_posix(),
                "test_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
