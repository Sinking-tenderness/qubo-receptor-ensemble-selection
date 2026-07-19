"""Run the preregistered MAPK14 train-only seed-uncertainty QUBO gate."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .normalized_receptor_qubo import (
        build_normalized_terms,
        exact_select,
        minmax_terms,
    )
    from .prepare_receptor import file_sha256
    from .run_receptor_selection_validation_gate import (
        normalize_from_train,
        read_csv,
        subset_metrics,
        write_csv,
    )
    from .run_stage05_mk14_method_gate import make_frozen_group_folds
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from normalized_receptor_qubo import (
        build_normalized_terms,
        exact_select,
        minmax_terms,
    )
    from prepare_receptor import file_sha256
    from run_receptor_selection_validation_gate import (
        normalize_from_train,
        read_csv,
        subset_metrics,
        write_csv,
    )
    from run_stage05_mk14_method_gate import make_frozen_group_folds


SEED_IDS = ("seed0", "seed1", "seed2")
MATRIX_IDS = ("primary", "sensitivity", *SEED_IDS)
METHOD_IDS = (
    "single_best",
    "matched_linear_top_k",
    "exhaustive",
    "greedy",
    "all_receptors",
    "uncertainty_qubo",
)
METRIC_KEYS = (
    "roc_auc",
    "pr_auc_average_precision",
    "bedroc_alpha_20",
    "EF1%",
    "EF5%",
    "EF10%",
)
REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "authorization_id",
    "target_id",
    "protocol_amendment",
    "inputs",
    "input_sha256",
    "receptor_ids",
    "expected",
    "cross_validation",
    "model",
    "baselines",
    "acceptance",
    "outputs",
    "interpretation_boundary",
}


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def load_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"uncertainty gate config is missing: {missing}")
    receptors = [str(value) for value in config["receptor_ids"]]
    expected = config["expected"]
    cv = config["cross_validation"]
    model = config["model"]
    acceptance = config["acceptance"]
    assert isinstance(expected, dict)
    assert isinstance(cv, dict)
    assert isinstance(model, dict)
    assert isinstance(acceptance, dict)
    if len(receptors) != 8 or len(receptors) != len(set(receptors)):
        raise ValueError("the preregistered receptor pool must contain eight IDs")
    if tuple(expected["seed_ids"]) != SEED_IDS:
        raise ValueError("the seed IDs differ from the frozen three-seed design")
    if int(expected["validation_rows"]) or int(expected["test_rows"]):
        raise ValueError("validation and test rows must remain unavailable")
    if int(cv["outer_fold_count"]) != 4:
        raise ValueError("the outer fold count must remain four")
    sizes = [int(value) for value in model["subset_sizes"]]
    if not sizes or any(size < 2 or size >= len(receptors) for size in sizes):
        raise ValueError("QUBO subset sizes must be between two and seven")
    if model["qubo_families"] != ["coverage_qubo", "discriminative_qubo"]:
        raise ValueError("the frozen QUBO family set changed")
    grids = model["weight_grids"]
    assert isinstance(grids, dict)
    for key in (
        "active_coverage",
        "decoy_exposure_discriminative",
        "active_overlap",
        "redundancy",
        "seed_stability",
    ):
        values = grids.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"weight grid is empty: {key}")
    if acceptance.get("all_checks_required") is not True:
        raise ValueError("every uncertainty gate check must be required")
    if acceptance.get("validation_remains_unavailable_after_pass") is not True:
        raise ValueError("validation must remain unavailable after this gate")
    if acceptance.get("test_remains_locked_after_pass") is not True:
        raise ValueError("test must remain locked after this gate")
    return config


def checked_input_paths(config: dict[str, object]) -> dict[str, Path]:
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    assert isinstance(inputs, dict)
    assert isinstance(hashes, dict)
    if set(inputs) != set(hashes):
        raise ValueError("input paths and hashes differ")
    paths = {key: Path(str(value)) for key, value in inputs.items()}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(hashes[key]).upper():
            raise ValueError(f"input SHA-256 differs: {key}")
    return paths


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


def rows_by_ligand(
    rows: list[dict[str, str]], name: str
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        ligand_id = row["ligand_id"]
        if ligand_id in output:
            raise ValueError(f"duplicate ligand in {name}: {ligand_id}")
        output[ligand_id] = row
    return output


def audit_inputs(
    config: dict[str, object], paths: dict[str, Path]
) -> dict[str, object]:
    expected = config["expected"]
    receptor_ids = [str(value) for value in config["receptor_ids"]]
    assert isinstance(expected, dict)
    aggregate = read_json(paths["aggregate_summary"])
    e32 = read_json(paths["e32_admission_summary"])
    e64 = read_json(paths["e64_consensus_audit"])
    if (
        aggregate.get("status") != "ok"
        or int(aggregate.get("ligand_count", 0)) != int(expected["ligand_count"])
        or int(aggregate.get("receptor_count", 0)) != len(receptor_ids)
        or int(aggregate.get("seed_count", 0)) != len(SEED_IDS)
        or int(aggregate.get("aggregated_pair_count", 0))
        != int(expected["receptor_ligand_pair_count"])
        or int(aggregate.get("locked_test_manifest_rows", -1)) != 0
    ):
        raise ValueError("aggregate summary differs from the frozen train matrix")
    if (
        e32.get("status") != "e32_matrix_admission_rejected"
        or e32.get("qubo_fitted") is not False
        or int(e32.get("validation_rows_read", -1)) != 0
        or int(e32.get("test_rows_read", -1)) != 0
    ):
        raise ValueError("the original e32 rejection boundary changed")
    if (
        e64.get("status")
        != "independent_e64_audit_ok_uniform_e64_not_supported"
        or e64.get("complete_uniform_e64_recomputation_supported") is not False
        or int(e64.get("e32_matrix_cells_replaced", -1)) != 0
        or int(e64.get("validation_rows_read", -1)) != 0
        or int(e64.get("test_rows_read", -1)) != 0
    ):
        raise ValueError("e64 evidence does not authorize this amendment")

    manifest_rows = read_csv(paths["ligand_manifest"])
    if len(manifest_rows) != int(expected["ligand_count"]):
        raise ValueError("ligand manifest count differs")
    if label_counts(manifest_rows) != {
        str(key): int(value)
        for key, value in dict(expected["label_counts"]).items()
    }:
        raise ValueError("ligand label counts differ")
    if {row["selection_role"] for row in manifest_rows} != set(
        str(value) for value in expected["allowed_selection_roles"]
    ):
        raise ValueError("a non-train selection role is visible")
    if {row["split"] for row in manifest_rows} != set(
        str(value) for value in expected["allowed_source_splits"]
    ):
        raise ValueError("a validation or test source split is visible")
    if any(not row.get("split_group_id") for row in manifest_rows):
        raise ValueError("a train ligand lacks split_group_id")
    manifest_by_id = rows_by_ligand(manifest_rows, "ligand manifest")

    receptor_rows = read_csv(paths["receptor_manifest"])
    observed_receptors = {row["conformer_id"] for row in receptor_rows}
    if observed_receptors != set(receptor_ids) or any(
        row["status"] != "ok" for row in receptor_rows
    ):
        raise ValueError("receptor manifest differs from the frozen pool")

    primary_rows = read_csv(paths["primary_median_matrix"])
    sensitivity_rows = read_csv(paths["sensitivity_minimum_matrix"])
    primary_by_id = rows_by_ligand(primary_rows, "primary matrix")
    sensitivity_by_id = rows_by_ligand(sensitivity_rows, "sensitivity matrix")
    ligand_ids = set(manifest_by_id)
    if set(primary_by_id) != ligand_ids or set(sensitivity_by_id) != ligand_ids:
        raise ValueError("matrix and train-manifest ligand IDs differ")

    long_rows = read_csv(paths["aggregated_seed_scores"])
    if len(long_rows) != int(expected["receptor_ligand_pair_count"]):
        raise ValueError("aggregated seed pair count differs")
    long_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    seed_columns = dict(expected["seed_score_columns"])
    for row in long_rows:
        key = (row["ligand_id"], row["receptor_id"])
        if key in long_by_pair:
            raise ValueError(f"duplicate aggregated pair: {key}")
        if (
            row["ligand_id"] not in ligand_ids
            or row["receptor_id"] not in receptor_ids
            or row["status"] != "ok"
            or int(row["seed_count"]) != len(SEED_IDS)
            or row["selection_role"] != "development_train"
        ):
            raise ValueError(f"invalid aggregated pair: {key}")
        values = [float(row[str(seed_columns[seed])]) for seed in SEED_IDS]
        if any(not math.isfinite(value) or value >= 0.0 for value in values):
            raise ValueError(f"invalid docking score: {key}")
        primary = float(primary_by_id[key[0]][key[1]])
        sensitivity = float(sensitivity_by_id[key[0]][key[1]])
        if not math.isclose(
            primary,
            float(row["median_representative_score"]),
            abs_tol=1e-9,
        ) or not math.isclose(
            sensitivity,
            float(row["minimum_representative_score"]),
            abs_tol=1e-9,
        ):
            raise ValueError(f"aggregated and matrix scores differ: {key}")
        long_by_pair[key] = row

    matrices: dict[str, list[dict[str, object]]] = {
        "primary": [dict(row) for row in primary_rows],
        "sensitivity": [dict(row) for row in sensitivity_rows],
    }
    for seed in SEED_IDS:
        column = str(seed_columns[seed])
        seed_rows: list[dict[str, object]] = []
        for ligand_id in sorted(ligand_ids):
            source = primary_by_id[ligand_id]
            row: dict[str, object] = {
                "target_id": source["target_id"],
                "ligand_id": ligand_id,
                "label": source["label"],
                "selection_role": source["selection_role"],
            }
            for receptor_id in receptor_ids:
                row[receptor_id] = float(
                    long_by_pair[(ligand_id, receptor_id)][column]
                )
            seed_rows.append(row)
        matrices[seed] = seed_rows

    for matrix_id, rows in matrices.items():
        for row in rows:
            ligand_id = str(row["ligand_id"])
            if row["label"] != manifest_by_id[ligand_id]["label"]:
                raise ValueError(f"label differs in {matrix_id}: {ligand_id}")
            for receptor_id in receptor_ids:
                value = float(row[receptor_id])
                if not math.isfinite(value) or value >= 0.0:
                    raise ValueError(
                        f"invalid {matrix_id} score: {ligand_id}/{receptor_id}"
                    )
    return {
        "manifest_rows": manifest_rows,
        "manifest_by_id": manifest_by_id,
        "matrices": matrices,
        "long_rows": long_rows,
        "validation_rows_read": 0,
        "test_rows_read": 0,
    }


def stability_scores(
    seed_terms: dict[str, dict[str, object]], receptor_ids: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    dispersion = {
        receptor_id: statistics.pstdev(
            float(seed_terms[seed]["normalized"]["utility"][receptor_id])
            for seed in SEED_IDS
        )
        for receptor_id in receptor_ids
    }
    raw_stability = {
        receptor_id: -dispersion[receptor_id] for receptor_id in receptor_ids
    }
    return raw_stability, minmax_terms(raw_stability)


def make_context(
    train_ids: set[str],
    validation_ids: set[str],
    matrices_by_id: dict[str, dict[str, dict[str, object]]],
    receptor_ids: list[str],
    model: dict[str, object],
) -> dict[str, object]:
    matrices: dict[str, dict[str, object]] = {}
    for matrix_id in MATRIX_IDS:
        source = matrices_by_id[matrix_id]
        train_raw = [source[ligand_id] for ligand_id in sorted(train_ids)]
        validation_raw = [
            source[ligand_id] for ligand_id in sorted(validation_ids)
        ]
        train, validation, bounds = normalize_from_train(
            train_raw, validation_raw, receptor_ids
        )
        matrices[matrix_id] = {
            "train": train,
            "validation": validation,
            "bounds": bounds,
        }
    central_terms = build_normalized_terms(
        matrices["primary"]["train"],
        receptor_ids,
        float(model["coverage_fraction"]),
        str(model["utility_metric"]),
    )
    seed_terms = {
        seed: build_normalized_terms(
            matrices[seed]["train"],
            receptor_ids,
            float(model["coverage_fraction"]),
            str(model["utility_metric"]),
        )
        for seed in SEED_IDS
    }
    raw_stability, normalized_stability = stability_scores(
        seed_terms, receptor_ids
    )
    central_terms["raw"]["stability"] = raw_stability
    central_terms["normalized"]["stability"] = normalized_stability
    for seed in SEED_IDS:
        seed_terms[seed]["raw"]["stability"] = raw_stability
        seed_terms[seed]["normalized"]["stability"] = normalized_stability
    return {
        "train_ids": sorted(train_ids),
        "validation_ids": sorted(validation_ids),
        "matrices": matrices,
        "terms": central_terms,
        "seed_terms": seed_terms,
    }


def jaccard(first: tuple[str, ...], second: tuple[str, ...]) -> float:
    left = set(first)
    right = set(second)
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def pairwise_jaccard(subsets: list[tuple[str, ...]]) -> dict[str, float]:
    values = [
        jaccard(first, second)
        for first, second in itertools.combinations(subsets, 2)
    ]
    if not values:
        return {"mean": 1.0, "minimum": 1.0, "maximum": 1.0}
    return {
        "mean": statistics.fmean(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def qubo_candidate_configs(model: dict[str, object]) -> list[dict[str, object]]:
    grids = model["weight_grids"]
    assert isinstance(grids, dict)
    candidates: list[dict[str, object]] = []
    for family in model["qubo_families"]:
        decoy_values = (
            [0.0]
            if family == "coverage_qubo"
            else [float(value) for value in grids["decoy_exposure_discriminative"]]
        )
        for (
            target_size,
            aggregation,
            active,
            decoy,
            overlap,
            redundancy,
            stability,
        ) in itertools.product(
            [int(value) for value in model["subset_sizes"]],
            [str(value) for value in model["aggregation_methods"]],
            [float(value) for value in grids["active_coverage"]],
            decoy_values,
            [float(value) for value in grids["active_overlap"]],
            [float(value) for value in grids["redundancy"]],
            [float(value) for value in grids["seed_stability"]],
        ):
            if overlap == 0.0 and redundancy == 0.0:
                continue
            candidates.append(
                {
                    "family": family,
                    "target_size": target_size,
                    "aggregation": aggregation,
                    "weights": {
                        "active_coverage": active,
                        "decoy_exposure": decoy,
                        "active_overlap": overlap,
                        "redundancy": redundancy,
                        "stability": stability,
                    },
                }
            )
    if not candidates:
        raise ValueError("the frozen QUBO grid produced no candidates")
    return candidates


def classical_candidate_configs(
    model: dict[str, object], receptor_count: int
) -> dict[str, list[dict[str, object]]]:
    sizes = [int(value) for value in model["subset_sizes"]]
    aggregations = [str(value) for value in model["aggregation_methods"]]
    return {
        "single_best": [
            {
                "family": "single_best",
                "target_size": 1,
                "aggregation": "min_score",
            }
        ],
        "exhaustive": [
            {
                "family": "exhaustive",
                "target_size": size,
                "aggregation": aggregation,
            }
            for size in sizes
            for aggregation in aggregations
        ],
        "greedy": [
            {
                "family": "greedy",
                "target_size": size,
                "aggregation": aggregation,
            }
            for size in sizes
            for aggregation in aggregations
        ],
        "all_receptors": [
            {
                "family": "all_receptors",
                "target_size": receptor_count,
                "aggregation": aggregation,
            }
            for aggregation in aggregations
        ],
    }


def metrics_for_context(
    context: dict[str, object],
    subset: tuple[str, ...],
    aggregation: str,
    split: str,
) -> dict[str, dict[str, object]]:
    return {
        matrix_id: subset_metrics(
            context["matrices"][matrix_id][split], subset, aggregation
        )
        for matrix_id in MATRIX_IDS
    }


def robust_metric_summary(
    metrics_by_matrix: dict[str, dict[str, object]]
) -> dict[str, float]:
    seed_bedroc = [
        float(metrics_by_matrix[seed]["bedroc_alpha_20"]) for seed in SEED_IDS
    ]
    return {
        "primary_bedroc": float(
            metrics_by_matrix["primary"]["bedroc_alpha_20"]
        ),
        "primary_pr_auc": float(
            metrics_by_matrix["primary"]["pr_auc_average_precision"]
        ),
        "primary_roc_auc": float(metrics_by_matrix["primary"]["roc_auc"]),
        "sensitivity_bedroc": float(
            metrics_by_matrix["sensitivity"]["bedroc_alpha_20"]
        ),
        "mean_seed_bedroc": statistics.fmean(seed_bedroc),
        "worst_seed_bedroc": min(seed_bedroc),
    }


def robust_subset_key(
    context: dict[str, object],
    subset: tuple[str, ...],
    aggregation: str,
) -> tuple[object, ...]:
    metrics = metrics_for_context(context, subset, aggregation, "train")
    robust = robust_metric_summary(metrics)
    return (
        -robust["worst_seed_bedroc"],
        -robust["primary_bedroc"],
        -robust["mean_seed_bedroc"],
        -robust["primary_pr_auc"],
        -robust["primary_roc_auc"],
        subset,
    )


def choose_exhaustive(
    context: dict[str, object],
    receptor_ids: list[str],
    size: int,
    aggregation: str,
) -> tuple[str, ...]:
    return min(
        itertools.combinations(receptor_ids, size),
        key=lambda subset: robust_subset_key(context, subset, aggregation),
    )


def choose_greedy(
    context: dict[str, object],
    receptor_ids: list[str],
    size: int,
    aggregation: str,
) -> tuple[str, ...]:
    selected: tuple[str, ...] = ()
    while len(selected) < size:
        candidates = [
            tuple(sorted((*selected, receptor_id)))
            for receptor_id in receptor_ids
            if receptor_id not in selected
        ]
        selected = min(
            candidates,
            key=lambda subset: robust_subset_key(context, subset, aggregation),
        )
    return selected


def matched_linear_top_k(
    coefficients: dict[str, object], target_size: int
) -> tuple[str, ...]:
    linear = coefficients["linear"]
    assert isinstance(linear, dict)
    ordered = sorted(linear, key=lambda receptor_id: (float(linear[receptor_id]), receptor_id))
    return tuple(sorted(ordered[:target_size]))


def noncardinality_quadratic_summary(
    coefficients: dict[str, object], size_penalty: float
) -> dict[str, object]:
    quadratic = coefficients["quadratic"]
    assert isinstance(quadratic, dict)
    values = {
        key: float(value) - 2.0 * size_penalty
        for key, value in quadratic.items()
    }
    numeric = list(values.values())
    return {
        "terms": values,
        "term_count": len(values),
        "maximum_absolute": max(abs(value) for value in numeric),
        "range": max(numeric) - min(numeric),
    }


def fit_qubo(
    candidate: dict[str, object],
    context: dict[str, object],
    receptor_ids: list[str],
    model: dict[str, object],
) -> tuple[tuple[str, ...], dict[str, object]]:
    size = int(candidate["target_size"])
    weights = {key: float(value) for key, value in dict(candidate["weights"]).items()}
    subset, energy, coefficients = exact_select(
        context["terms"],
        receptor_ids,
        size,
        weights,
        float(model["size_penalty"]),
    )
    linear_subset = matched_linear_top_k(coefficients, size)
    seed_subsets: dict[str, tuple[str, ...]] = {}
    for seed in SEED_IDS:
        seed_subset, _, _ = exact_select(
            context["seed_terms"][seed],
            receptor_ids,
            size,
            weights,
            float(model["size_penalty"]),
        )
        seed_subsets[seed] = seed_subset
    stability = pairwise_jaccard(list(seed_subsets.values()))
    return subset, {
        "energy": energy,
        "coefficients": coefficients,
        "matched_linear_subset": linear_subset,
        "seed_specific_subsets": seed_subsets,
        "seed_pairwise_jaccard": stability,
        "noncardinality_quadratic": noncardinality_quadratic_summary(
            coefficients, float(model["size_penalty"])
        ),
    }


def fit_method(
    candidate: dict[str, object],
    context: dict[str, object],
    receptor_ids: list[str],
    model: dict[str, object],
) -> tuple[tuple[str, ...], dict[str, object]]:
    family = str(candidate["family"])
    size = int(candidate["target_size"])
    aggregation = str(candidate["aggregation"])
    if family in {"coverage_qubo", "discriminative_qubo"}:
        return fit_qubo(candidate, context, receptor_ids, model)
    if family == "single_best":
        return choose_exhaustive(context, receptor_ids, 1, "min_score"), {}
    if family == "exhaustive":
        return choose_exhaustive(context, receptor_ids, size, aggregation), {}
    if family == "greedy":
        return choose_greedy(context, receptor_ids, size, aggregation), {}
    if family == "all_receptors":
        return tuple(receptor_ids), {}
    raise ValueError(f"unsupported method family: {family}")


def average_metrics(
    rows: list[dict[str, object]]
) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average an empty metric list")
    return {
        key: statistics.fmean(float(row[key]) for row in rows)
        for key in METRIC_KEYS
    }


def trial_key(trial: dict[str, object]) -> tuple[object, ...]:
    robust = trial["robust_metrics"]
    config = trial["config"]
    assert isinstance(robust, dict)
    assert isinstance(config, dict)
    weights = dict(config.get("weights", {}))
    family_order = {
        "single_best": 0,
        "exhaustive": 1,
        "greedy": 2,
        "all_receptors": 3,
        "coverage_qubo": 4,
        "discriminative_qubo": 5,
    }
    aggregation_order = {"min_score": 0, "mean_score": 1}
    return (
        -float(robust["worst_seed_bedroc"]),
        -float(robust["primary_bedroc"]),
        -float(robust["mean_seed_bedroc"]),
        -float(robust["primary_pr_auc"]),
        -float(robust["primary_roc_auc"]),
        -float(trial["mean_seed_pairwise_jaccard"]),
        float(trial["primary_bedroc_population_std"]),
        int(config["target_size"]),
        sum(float(value) for value in weights.values()),
        family_order[str(config["family"])],
        aggregation_order[str(config["aggregation"])],
        json.dumps(config, sort_keys=True),
    )


def tune_candidates(
    candidates: list[dict[str, object]],
    contexts: list[dict[str, object]],
    receptor_ids: list[str],
    model: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    trials: list[dict[str, object]] = []
    for candidate in candidates:
        metrics_rows = {matrix_id: [] for matrix_id in MATRIX_IDS}
        subsets: list[list[str]] = []
        linear_subsets: list[list[str]] = []
        jaccards: list[float] = []
        for context in contexts:
            subset, details = fit_method(
                candidate, context, receptor_ids, model
            )
            subsets.append(list(subset))
            if "matched_linear_subset" in details:
                linear_subsets.append(list(details["matched_linear_subset"]))
                jaccards.append(
                    float(details["seed_pairwise_jaccard"]["mean"])
                )
            fold_metrics = metrics_for_context(
                context,
                subset,
                str(candidate["aggregation"]),
                "validation",
            )
            for matrix_id in MATRIX_IDS:
                metrics_rows[matrix_id].append(fold_metrics[matrix_id])
        mean_metrics = {
            matrix_id: average_metrics(rows)
            for matrix_id, rows in metrics_rows.items()
        }
        robust = robust_metric_summary(mean_metrics)
        trials.append(
            {
                "config": candidate,
                "mean_validation_metrics": mean_metrics,
                "robust_metrics": robust,
                "primary_bedroc_population_std": statistics.pstdev(
                    float(row["bedroc_alpha_20"])
                    for row in metrics_rows["primary"]
                ),
                "subsets": subsets,
                "matched_linear_subsets": linear_subsets,
                "qubo_linear_difference_count": sum(
                    subset != linear
                    for subset, linear in zip(subsets, linear_subsets)
                ),
                "mean_seed_pairwise_jaccard": (
                    statistics.fmean(jaccards) if jaccards else 1.0
                ),
            }
        )
    return min(trials, key=trial_key), trials


def score_records(
    rows: list[dict[str, object]],
    subset: tuple[str, ...],
    aggregation: str,
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        values = [float(row[receptor_id]) for receptor_id in subset]
        score = (
            min(values)
            if aggregation == "min_score"
            else statistics.fmean(values)
        )
        output[str(row["ligand_id"])] = {
            "label": row["label"],
            "score": score,
        }
    return output


def add_oof_records(
    destination: dict[str, dict[str, dict[str, dict[str, object]]]],
    method: str,
    matrix_id: str,
    records: dict[str, dict[str, object]],
) -> None:
    current = destination[method][matrix_id]
    overlap = set(current) & set(records)
    if overlap:
        raise ValueError(f"duplicate OOF ligand IDs: {method}/{matrix_id}")
    current.update(records)


def oof_metrics(
    records: dict[str, dict[str, dict[str, dict[str, object]]]]
) -> dict[str, dict[str, dict[str, object]]]:
    return {
        method: {
            matrix_id: ranked_metrics_with_ids(matrix_records)
            for matrix_id, matrix_records in by_matrix.items()
        }
        for method, by_matrix in records.items()
    }


def percentile(values: list[float], target: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    return sum(value <= target for value in values) / len(values)


def uncertainty_tables(
    long_rows: list[dict[str, str]],
    matrices: dict[str, list[dict[str, object]]],
    receptor_ids: list[str],
    seed_columns: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cell_rows: list[dict[str, object]] = []
    for row in long_rows:
        values = [
            float(row[str(seed_columns[seed])]) for seed in SEED_IDS
        ]
        minimum = min(values)
        cell_rows.append(
            {
                "ligand_id": row["ligand_id"],
                "label": row["label"],
                "receptor_id": row["receptor_id"],
                **{seed: value for seed, value in zip(SEED_IDS, values)},
                "median_score": statistics.median(values),
                "minimum_score": minimum,
                "maximum_score": max(values),
                "population_std": statistics.pstdev(values),
                "score_range": max(values) - minimum,
                "replicates_within_0_5_of_minimum": sum(
                    value <= minimum + 0.5 for value in values
                ),
            }
        )
    matrix_by_id = {
        matrix_id: rows_by_ligand(
            [{key: str(value) for key, value in row.items()} for row in rows],
            matrix_id,
        )
        for matrix_id, rows in matrices.items()
    }
    receptor_rows: list[dict[str, object]] = []
    for receptor_id in receptor_ids:
        receptor_cells = [
            row for row in cell_rows if row["receptor_id"] == receptor_id
        ]
        ranges = sorted(float(row["score_range"]) for row in receptor_cells)
        bedroc_by_matrix: dict[str, float] = {}
        for matrix_id in ("primary", *SEED_IDS):
            data = {
                ligand_id: {
                    "label": row["label"],
                    "score": float(row[receptor_id]),
                }
                for ligand_id, row in matrix_by_id[matrix_id].items()
            }
            bedroc_by_matrix[matrix_id] = float(
                ranked_metrics_with_ids(data)["bedroc_alpha_20"]
            )
        seed_bedrocs = [bedroc_by_matrix[seed] for seed in SEED_IDS]
        receptor_rows.append(
            {
                "receptor_id": receptor_id,
                "pair_count": len(receptor_cells),
                "mean_seed_score_range": statistics.fmean(ranges),
                "median_seed_score_range": statistics.median(ranges),
                "p95_seed_score_range": ranges[
                    max(0, math.ceil(0.95 * len(ranges)) - 1)
                ],
                "maximum_seed_score_range": max(ranges),
                "range_above_0_5_count": sum(value > 0.5 for value in ranges),
                "range_above_1_0_count": sum(value > 1.0 for value in ranges),
                "primary_singleton_bedroc": bedroc_by_matrix["primary"],
                "seed0_singleton_bedroc": bedroc_by_matrix["seed0"],
                "seed1_singleton_bedroc": bedroc_by_matrix["seed1"],
                "seed2_singleton_bedroc": bedroc_by_matrix["seed2"],
                "mean_seed_singleton_bedroc": statistics.fmean(seed_bedrocs),
                "minimum_seed_singleton_bedroc": min(seed_bedrocs),
                "seed_singleton_bedroc_population_std": statistics.pstdev(
                    seed_bedrocs
                ),
            }
        )
    return cell_rows, receptor_rows


def exact_fixed_subset_distribution(
    outer_contexts: list[dict[str, object]],
    receptor_ids: list[str],
    size: int,
    aggregation: str,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for subset in itertools.combinations(receptor_ids, size):
        records = {matrix_id: {} for matrix_id in MATRIX_IDS}
        for context in outer_contexts:
            for matrix_id in MATRIX_IDS:
                fold = score_records(
                    context["matrices"][matrix_id]["validation"],
                    subset,
                    aggregation,
                )
                if set(records[matrix_id]) & set(fold):
                    raise ValueError("duplicate fixed-subset OOF ligand")
                records[matrix_id].update(fold)
        metrics = {
            matrix_id: ranked_metrics_with_ids(matrix_records)
            for matrix_id, matrix_records in records.items()
        }
        robust = robust_metric_summary(metrics)
        output.append(
            {
                "subset": "+".join(subset),
                "target_size": size,
                "aggregation": aggregation,
                **robust,
            }
        )
    return output


def gate_decision(
    candidate_metrics: dict[str, dict[str, object]],
    linear_metrics: dict[str, dict[str, object]],
    final_subset: tuple[str, ...],
    linear_subset: tuple[str, ...],
    quadratic: dict[str, object],
    outer_jaccard_mean: float,
    final_jaccard_mean: float,
    acceptance: dict[str, object],
) -> tuple[dict[str, float], dict[str, bool], bool]:
    seed_deltas = {
        seed: float(candidate_metrics[seed]["bedroc_alpha_20"])
        - float(linear_metrics[seed]["bedroc_alpha_20"])
        for seed in SEED_IDS
    }
    deltas = {
        "primary_median_bedroc": float(
            candidate_metrics["primary"]["bedroc_alpha_20"]
        )
        - float(linear_metrics["primary"]["bedroc_alpha_20"]),
        "mean_seed_bedroc": statistics.fmean(seed_deltas.values()),
        "worst_seed_bedroc": min(seed_deltas.values()),
        **{f"{seed}_bedroc": value for seed, value in seed_deltas.items()},
    }
    checks = {
        "minimum_selected_subset_size": len(final_subset)
        >= int(acceptance["minimum_selected_subset_size"]),
        "noncardinality_quadratic_terms": (
            float(quadratic["maximum_absolute"]) > 1e-12
            and float(quadratic["range"]) > 1e-12
        ),
        "selected_subset_differs_from_matched_linear": final_subset
        != linear_subset,
        "primary_median_bedroc_delta": deltas["primary_median_bedroc"]
        >= float(
            acceptance[
                "minimum_primary_median_bedroc_delta_vs_matched_linear"
            ]
        ),
        "mean_seed_bedroc_delta": deltas["mean_seed_bedroc"]
        >= float(acceptance["minimum_mean_seed_bedroc_delta_vs_matched_linear"]),
        "worst_seed_bedroc_delta": deltas["worst_seed_bedroc"]
        >= float(acceptance["minimum_worst_seed_bedroc_delta_vs_matched_linear"]),
        "outer_seed_fit_mean_pairwise_jaccard": outer_jaccard_mean
        >= float(acceptance["minimum_outer_seed_fit_mean_pairwise_jaccard"]),
        "final_seed_fit_mean_pairwise_jaccard": final_jaccard_mean
        >= float(acceptance["minimum_final_seed_fit_mean_pairwise_jaccard"]),
    }
    return deltas, checks, all(checks.values())


def flatten_trial(trial: dict[str, object]) -> dict[str, object]:
    config = trial["config"]
    robust = trial["robust_metrics"]
    assert isinstance(config, dict)
    assert isinstance(robust, dict)
    return {
        "family": config["family"],
        "target_size": config["target_size"],
        "aggregation": config["aggregation"],
        "weights": json.dumps(config.get("weights", {}), sort_keys=True),
        **robust,
        "primary_bedroc_population_std": trial[
            "primary_bedroc_population_std"
        ],
        "mean_seed_pairwise_jaccard": trial[
            "mean_seed_pairwise_jaccard"
        ],
        "qubo_linear_difference_count": trial[
            "qubo_linear_difference_count"
        ],
        "subsets": json.dumps(trial["subsets"]),
        "matched_linear_subsets": json.dumps(
            trial["matched_linear_subsets"]
        ),
    }


def run_gate(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    paths = checked_input_paths(config)
    audited = audit_inputs(config, paths)
    receptor_ids = [str(value) for value in config["receptor_ids"]]
    expected = config["expected"]
    cv = config["cross_validation"]
    model = config["model"]
    acceptance = config["acceptance"]
    outputs = config["outputs"]
    assert isinstance(expected, dict)
    assert isinstance(cv, dict)
    assert isinstance(model, dict)
    assert isinstance(acceptance, dict)
    assert isinstance(outputs, dict)

    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    core_outputs = [
        path for key, path in output_paths.items() if key != "run_directory"
    ]
    existing = [path for path in core_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("uncertainty-gate outputs exist; use --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()
    output_paths["run_directory"].mkdir(parents=True, exist_ok=True)

    manifest_rows = audited["manifest_rows"]
    matrices = audited["matrices"]
    assert isinstance(manifest_rows, list)
    assert isinstance(matrices, dict)
    matrices_by_id = {
        matrix_id: {
            str(row["ligand_id"]): row for row in matrix_rows
        }
        for matrix_id, matrix_rows in matrices.items()
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
    fold_label_counts = {
        str(fold): label_counts(
            [
                row
                for row in manifest_rows
                if assignments[row["ligand_id"]] == fold
            ]
        )
        for fold in range(int(cv["outer_fold_count"]))
    }

    qubo_candidates = qubo_candidate_configs(model)
    classical = classical_candidate_configs(model, len(receptor_ids))
    oof: dict[str, dict[str, dict[str, dict[str, object]]]] = {
        method: {matrix_id: {} for matrix_id in MATRIX_IDS}
        for method in METHOD_IDS
    }
    outer_rows: list[dict[str, object]] = []
    outer_details: list[dict[str, object]] = []
    outer_contexts: list[dict[str, object]] = []
    outer_qubo_jaccards: list[float] = []

    for outer_fold in range(int(cv["outer_fold_count"])):
        outer_ids = {
            ligand_id
            for ligand_id, fold in assignments.items()
            if fold == outer_fold
        }
        outer_train_ids = ligand_ids - outer_ids
        inner_contexts = []
        for inner_fold in range(int(cv["outer_fold_count"])):
            if inner_fold == outer_fold:
                continue
            inner_validation_ids = {
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == inner_fold
            }
            inner_contexts.append(
                make_context(
                    outer_train_ids - inner_validation_ids,
                    inner_validation_ids,
                    matrices_by_id,
                    receptor_ids,
                    model,
                )
            )
        outer_context = make_context(
            outer_train_ids,
            outer_ids,
            matrices_by_id,
            receptor_ids,
            model,
        )
        outer_contexts.append(outer_context)
        selected_trials: dict[str, dict[str, object]] = {}
        for method, candidates in classical.items():
            selected_trials[method], _ = tune_candidates(
                candidates, inner_contexts, receptor_ids, model
            )
        selected_qubo, _ = tune_candidates(
            qubo_candidates, inner_contexts, receptor_ids, model
        )
        selected_trials["uncertainty_qubo"] = selected_qubo

        fitted: dict[str, tuple[tuple[str, ...], dict[str, object], dict[str, object]]] = {}
        for method in (
            "single_best",
            "exhaustive",
            "greedy",
            "all_receptors",
            "uncertainty_qubo",
        ):
            candidate = selected_trials[method]["config"]
            subset, details = fit_method(
                candidate, outer_context, receptor_ids, model
            )
            fitted[method] = (subset, details, candidate)
        qubo_subset, qubo_details, qubo_config = fitted["uncertainty_qubo"]
        linear_subset = tuple(qubo_details["matched_linear_subset"])
        fitted["matched_linear_top_k"] = (
            linear_subset,
            {},
            {
                "family": "matched_linear_top_k",
                "target_size": qubo_config["target_size"],
                "aggregation": qubo_config["aggregation"],
                "source_qubo_config": qubo_config,
            },
        )
        outer_qubo_jaccards.append(
            float(qubo_details["seed_pairwise_jaccard"]["mean"])
        )

        fold_detail = {
            "outer_fold": outer_fold,
            "train_ligand_count": len(outer_train_ids),
            "validation_ligand_count": len(outer_ids),
            "methods": {},
        }
        for method in METHOD_IDS:
            subset, details, candidate = fitted[method]
            aggregation = str(candidate["aggregation"])
            fold_metrics = metrics_for_context(
                outer_context, subset, aggregation, "validation"
            )
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
            robust = robust_metric_summary(fold_metrics)
            outer_rows.append(
                {
                    "outer_fold": outer_fold,
                    "method": method,
                    "subset": "+".join(subset),
                    "target_size": len(subset),
                    "aggregation": aggregation,
                    "selected_config": json.dumps(candidate, sort_keys=True),
                    **robust,
                    "seed_fit_mean_pairwise_jaccard": (
                        details.get("seed_pairwise_jaccard", {}).get("mean", "")
                        if isinstance(details, dict)
                        else ""
                    ),
                }
            )
            fold_detail["methods"][method] = {
                "config": candidate,
                "subset": list(subset),
                "metrics": fold_metrics,
                "seed_specific_subsets": {
                    seed: list(value)
                    for seed, value in details.get(
                        "seed_specific_subsets", {}
                    ).items()
                },
                "seed_pairwise_jaccard": details.get(
                    "seed_pairwise_jaccard"
                ),
                "matched_linear_subset": list(
                    details.get("matched_linear_subset", ())
                ),
            }
        outer_details.append(fold_detail)

    metrics = oof_metrics(oof)
    if any(
        len(oof[method][matrix_id]) != int(expected["ligand_count"])
        for method in METHOD_IDS
        for matrix_id in MATRIX_IDS
    ):
        raise ValueError("OOF predictions do not cover all train ligands")

    final_contexts = []
    for validation_fold in range(int(cv["outer_fold_count"])):
        validation_ids = {
            ligand_id
            for ligand_id, fold in assignments.items()
            if fold == validation_fold
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
    final_selected_qubo, final_trials = tune_candidates(
        qubo_candidates, final_contexts, receptor_ids, model
    )
    full_context = make_context(
        ligand_ids, set(), matrices_by_id, receptor_ids, model
    )
    final_config = final_selected_qubo["config"]
    final_subset, final_details = fit_qubo(
        final_config, full_context, receptor_ids, model
    )
    final_linear_subset = tuple(final_details["matched_linear_subset"])
    final_baselines: dict[str, dict[str, object]] = {}
    for method, candidates in classical.items():
        selected, _ = tune_candidates(
            candidates, final_contexts, receptor_ids, model
        )
        subset, _ = fit_method(
            selected["config"], full_context, receptor_ids, model
        )
        final_baselines[method] = {
            "config": selected["config"],
            "subset": list(subset),
        }

    deltas, checks, passed = gate_decision(
        metrics["uncertainty_qubo"],
        metrics["matched_linear_top_k"],
        final_subset,
        final_linear_subset,
        final_details["noncardinality_quadratic"],
        statistics.fmean(outer_qubo_jaccards),
        float(final_details["seed_pairwise_jaccard"]["mean"]),
        acceptance,
    )
    status = (
        "train_uncertainty_qubo_gate_passed_validation_unavailable"
        if passed
        else "train_uncertainty_qubo_gate_failed_validation_unavailable"
    )

    fixed_distribution = exact_fixed_subset_distribution(
        outer_contexts,
        receptor_ids,
        int(final_config["target_size"]),
        str(final_config["aggregation"]),
    )
    candidate_robust = robust_metric_summary(metrics["uncertainty_qubo"])
    random_context = {
        "subset_count": len(fixed_distribution),
        "primary_bedroc_percentile": percentile(
            [float(row["primary_bedroc"]) for row in fixed_distribution],
            candidate_robust["primary_bedroc"],
        ),
        "mean_seed_bedroc_percentile": percentile(
            [float(row["mean_seed_bedroc"]) for row in fixed_distribution],
            candidate_robust["mean_seed_bedroc"],
        ),
        "worst_seed_bedroc_percentile": percentile(
            [float(row["worst_seed_bedroc"]) for row in fixed_distribution],
            candidate_robust["worst_seed_bedroc"],
        ),
        "random_numbers_used": False,
    }

    cell_uncertainty, receptor_uncertainty = uncertainty_tables(
        audited["long_rows"],
        matrices,
        receptor_ids,
        dict(expected["seed_score_columns"]),
    )
    write_csv(output_paths["cell_uncertainty_csv"], cell_uncertainty)
    write_csv(output_paths["receptor_uncertainty_csv"], receptor_uncertainty)
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
        for method in METHOD_IDS
        for matrix_id in MATRIX_IDS
        for ligand_id, record in sorted(oof[method][matrix_id].items())
    ]
    write_csv(output_paths["oof_scores_csv"], oof_rows)
    method_metric_rows = [
        {
            "method": method,
            "matrix": matrix_id,
            **{key: value for key, value in metric.items() if key != "top10_ligand_ids"},
        }
        for method in METHOD_IDS
        for matrix_id, metric in metrics[method].items()
    ]
    write_csv(output_paths["method_metrics_csv"], method_metric_rows)
    write_csv(
        output_paths["final_tuning_trials_csv"],
        [flatten_trial(trial) for trial in final_trials],
    )
    write_csv(output_paths["exact_random_subsets_csv"], fixed_distribution)

    implementation = {
        "path": f"scripts/{Path(__file__).name}",
        "sha256": file_sha256(Path(__file__)),
    }
    input_records = {
        key: {"path": path.as_posix(), "sha256": file_sha256(path)}
        for key, path in paths.items()
    }
    candidate_protocol = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": status,
        "preregistration": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "implementation": implementation,
        "inputs": input_records,
        "data_boundary": {
            "development_train_ligands": len(ligand_ids),
            "validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "selected_qubo": {
            "config": final_config,
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
        "oof_metrics": metrics,
        "comparison_to_matched_linear": {
            "bedroc_deltas": deltas,
            "acceptance_checks": checks,
            "gate_passed": passed,
        },
        "final_baselines": final_baselines,
        "exact_fixed_subset_context": random_context,
        "validation_status": "unavailable_not_evaluated",
        "test_status": "locked_unreleased",
        "interpretation_note": config["interpretation_boundary"],
    }
    write_json(output_paths["candidate_protocol_json"], candidate_protocol)

    output_records = {
        key: {"path": path.as_posix(), "sha256": file_sha256(path)}
        for key, path in output_paths.items()
        if key not in {"run_directory", "summary_json"}
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
        "inputs": input_records,
        "audit": {
            "ligand_count": len(ligand_ids),
            "label_counts": label_counts(manifest_rows),
            "receptor_count": len(receptor_ids),
            "seed_count": len(SEED_IDS),
            "fold_label_counts": fold_label_counts,
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "e32_cells_replaced": 0,
            "e64_scores_used_in_matrix": 0,
        },
        "candidate_counts": {
            "uncertainty_qubo": len(qubo_candidates),
            **{key: len(value) for key, value in classical.items()},
        },
        "method_oof_metrics": metrics,
        "outer_fold_details": outer_details,
        "selected_qubo": candidate_protocol["selected_qubo"],
        "comparison_to_matched_linear": candidate_protocol[
            "comparison_to_matched_linear"
        ],
        "exact_fixed_subset_context": random_context,
        "final_baselines": final_baselines,
        "validation_status": "unavailable_not_evaluated",
        "test_status": "locked_unreleased",
        "outputs": output_records,
        "interpretation_note": config["interpretation_boundary"],
    }
    write_json(output_paths["summary_json"], summary)
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
