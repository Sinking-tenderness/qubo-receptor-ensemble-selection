"""Run the preregistered MAPK14 train-only method and validation gate."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy
import scipy

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .cross_validate_ensemble_mvp import paired_bootstrap_delta
    from .prepare_receptor import file_sha256
    from .run_development_scaffold_cv_gate import (
        collect_scores,
        fit_config,
        make_context,
        make_scaffold_folds,
        method_configs,
        tune_configs,
    )
    from .run_receptor_selection_validation_gate import (
        compact_metrics,
        percentile,
        read_csv,
        subset_metrics,
        write_csv,
    )
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from cross_validate_ensemble_mvp import paired_bootstrap_delta
    from prepare_receptor import file_sha256
    from run_development_scaffold_cv_gate import (
        collect_scores,
        fit_config,
        make_context,
        make_scaffold_folds,
        method_configs,
        tune_configs,
    )
    from run_receptor_selection_validation_gate import (
        compact_metrics,
        percentile,
        read_csv,
        subset_metrics,
        write_csv,
    )


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "runtime",
    "preregistration",
    "inputs",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_INPUT_KEYS = {
    "receptor_manifest",
    "ligand_manifest",
    "seed_aggregation_config",
    "aggregate_summary",
    "matrix_rescue_summary",
    "primary_matrix",
    "sensitivity_matrix",
}
REQUIRED_OUTPUT_KEYS = {
    "run_directory",
    "fold_assignments_csv",
    "inner_cv_trials_csv",
    "selected_methods_csv",
    "validation_metrics_csv",
    "validation_scores_csv",
    "exact_random_subsets_csv",
    "exact_random_summary_csv",
    "candidate_protocol_json",
    "summary_json",
}
RANKING_METRICS = (
    "roc_auc",
    "pr_auc_average_precision",
    "bedroc_alpha_20",
    "EF1%",
    "EF5%",
    "EF10%",
)
RANDOM_SUMMARY_METRICS = (
    "roc_auc",
    "pr_auc_average_precision",
    "bedroc_alpha_20",
    "EF5%",
)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def load_execution_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"execution config is missing keys: {', '.join(missing)}")
    preregistration = config["preregistration"]
    runtime = config["runtime"]
    inputs = config["inputs"]
    outputs = config["outputs"]
    if not isinstance(preregistration, dict) or set(preregistration) != {
        "path",
        "sha256",
    }:
        raise ValueError("preregistration must contain path and sha256")
    if not isinstance(runtime, dict) or set(runtime) != {
        "conda_environment",
        "python_version",
        "numpy_version",
        "scipy_version",
    }:
        raise ValueError("runtime lock is incomplete")
    if not isinstance(inputs, dict) or set(inputs) != REQUIRED_INPUT_KEYS:
        raise ValueError("inputs do not match the required method-gate inputs")
    for key, record in inputs.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"input record is invalid: {key}")
    if not isinstance(outputs, dict) or set(outputs) != REQUIRED_OUTPUT_KEYS:
        raise ValueError("outputs do not match the required method-gate outputs")
    return config


def check_runtime(config: dict[str, object]) -> dict[str, str]:
    expected = config["runtime"]
    assert isinstance(expected, dict)
    actual = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "python_executable": sys.executable,
    }
    for key in (
        "conda_environment",
        "python_version",
        "numpy_version",
        "scipy_version",
    ):
        if actual[key] != str(expected[key]):
            raise RuntimeError(
                f"runtime differs for {key}: {actual[key]} != {expected[key]}"
            )
    return actual


def validate_preregistration(preregistration: dict[str, object]) -> None:
    if preregistration.get("target_id") != "MK14":
        raise ValueError("preregistration target must be MK14")
    roles = preregistration.get("data_roles")
    if not isinstance(roles, dict):
        raise ValueError("preregistration data_roles are missing")
    expected_roles = {
        "train": (80, 80, "development_train"),
        "validation": (40, 40, "development_validation"),
    }
    for role, (active, decoy, selection_role) in expected_roles.items():
        record = roles.get(role)
        if not isinstance(record, dict):
            raise ValueError(f"preregistered {role} role is missing")
        if (
            int(record.get("active", -1)) != active
            or int(record.get("decoy", -1)) != decoy
            or record.get("selection_role") != selection_role
        ):
            raise ValueError(f"preregistered {role} counts or role changed")
    test = roles.get("test")
    if not isinstance(test, dict) or test.get("status") != "locked_unreleased":
        raise ValueError("test must remain locked_unreleased")
    inner = preregistration.get("inner_selection")
    if not isinstance(inner, dict):
        raise ValueError("inner_selection is missing")
    if (
        inner.get("data") != "development_train only"
        or int(inner.get("fold_count", 0)) != 4
        or inner.get("score_normalization")
        != "fit per-receptor min-max bounds on each inner-training fold only"
        or inner.get("selection_metric") != "bedroc_alpha_20"
    ):
        raise ValueError("train-only inner-selection protocol changed")
    if inner.get("qubo_families") != [
        "coverage_qubo",
        "discriminative_qubo",
    ]:
        raise ValueError("QUBO family set changed")
    gate = preregistration.get("validation_gate")
    if not isinstance(gate, dict) or gate.get("all_checks_required") is not True:
        raise ValueError("validation gate must require every check")
    if gate.get("comparison_method") != "single_best":
        raise ValueError("validation comparator must be single_best")
    release = preregistration.get("test_release")
    if (
        not isinstance(release, dict)
        or release.get("automatic_release") is not False
        or release.get("manual_review_required") is not True
    ):
        raise ValueError("test release boundary changed")


def model_from_preregistration(
    preregistration: dict[str, object],
) -> dict[str, object]:
    inner = preregistration["inner_selection"]
    assert isinstance(inner, dict)
    return {
        "utility_metric": "bedroc",
        "coverage_fraction": float(inner["coverage_fraction"]),
        "subset_sizes": [int(value) for value in inner["subset_sizes"]],
        "aggregation_methods": [
            str(value) for value in inner["aggregation_methods"]
        ],
        "size_penalty": float(inner["size_penalty"]),
        "families": [str(value) for value in inner["qubo_families"]],
        "weight_grids": {
            key: [float(value) for value in values]
            for key, values in dict(inner["weight_grids"]).items()
        },
    }


def cv_from_preregistration(
    preregistration: dict[str, object],
) -> dict[str, object]:
    inner = preregistration["inner_selection"]
    assert isinstance(inner, dict)
    return {
        "inner_selection_metric": str(inner["selection_metric"]),
        "inner_tie_breakers": [
            str(value) for value in inner["tie_breakers"]
        ],
    }


def checked_input_paths(
    config: dict[str, object],
) -> tuple[Path, dict[str, Path]]:
    preregistration = config["preregistration"]
    inputs = config["inputs"]
    assert isinstance(preregistration, dict)
    assert isinstance(inputs, dict)
    prereg_path = Path(str(preregistration["path"]))
    if not prereg_path.is_file():
        raise FileNotFoundError(prereg_path)
    if file_sha256(prereg_path) != str(preregistration["sha256"]).upper():
        raise ValueError("preregistration SHA-256 differs")
    paths: dict[str, Path] = {}
    for key, record in inputs.items():
        assert isinstance(record, dict)
        path = Path(str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(record["sha256"]).upper():
            raise ValueError(f"input SHA-256 differs: {key}")
        paths[key] = path
    return prereg_path, paths


def count_by_role_and_label(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[row["selection_role"]][row["label"]] += 1
    return {
        role: dict(sorted(values.items()))
        for role, values in sorted(counts.items())
    }


def audit_matrix_rows(
    matrix_name: str,
    rows: list[dict[str, str]],
    manifest_by_id: dict[str, dict[str, str]],
    receptor_ids: list[str],
) -> int:
    ids = [row["ligand_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{matrix_name} contains duplicate ligand IDs")
    if set(ids) != set(manifest_by_id):
        raise ValueError(f"{matrix_name} and ligand manifest IDs differ")
    nonnegative = 0
    for row in rows:
        manifest = manifest_by_id[row["ligand_id"]]
        for key in ("target_id", "label", "selection_role"):
            if row.get(key) != manifest.get(key):
                raise ValueError(
                    f"{matrix_name} metadata differs for {row['ligand_id']}: {key}"
                )
        for receptor_id in receptor_ids:
            try:
                value = float(row[receptor_id])
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"invalid {matrix_name} score for {row['ligand_id']} / "
                    f"{receptor_id}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"non-finite score in {matrix_name}")
            nonnegative += int(value >= 0.0)
    return nonnegative


def audit_inputs(
    preregistration: dict[str, object],
    paths: dict[str, Path],
) -> dict[str, object]:
    frozen = preregistration.get("frozen_inputs")
    if not isinstance(frozen, dict):
        raise ValueError("preregistration frozen_inputs are missing")
    for key in ("receptor_manifest", "ligand_manifest", "seed_aggregation_config"):
        record = frozen.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"frozen input is missing: {key}")
        if (
            str(record.get("path")) != paths[key].as_posix()
            or str(record.get("sha256")).upper() != file_sha256(paths[key])
        ):
            raise ValueError(f"execution input differs from preregistration: {key}")

    aggregate = read_json(paths["aggregate_summary"])
    if (
        aggregate.get("status") != "ok"
        or int(aggregate.get("ligand_count", 0)) != 240
        or int(aggregate.get("receptor_count", 0)) != 4
        or int(aggregate.get("seed_count", 0)) != 3
        or int(aggregate.get("aggregated_pair_count", 0)) != 960
        or int(aggregate.get("locked_test_manifest_rows", -1)) != 0
    ):
        raise ValueError("seed-aggregation summary is not admissible")
    aggregate_outputs = aggregate.get("outputs")
    if not isinstance(aggregate_outputs, dict):
        raise ValueError("aggregate output records are missing")
    for matrix_key, aggregate_key in (
        ("primary_matrix", "primary_median_matrix_csv"),
        ("sensitivity_matrix", "sensitivity_minimum_matrix_csv"),
    ):
        record = aggregate_outputs.get(aggregate_key)
        if (
            not isinstance(record, dict)
            or str(record.get("path")) != paths[matrix_key].as_posix()
            or str(record.get("sha256")).upper() != file_sha256(paths[matrix_key])
        ):
            raise ValueError(f"aggregate output record differs: {matrix_key}")

    rescue = read_json(paths["matrix_rescue_summary"])
    if (
        rescue.get("status") != "matrix_admission_rescued"
        or rescue.get("all_cases_passed") is not True
        or rescue.get("primary_matrix_authorized") is not True
        or rescue.get("sensitivity_matrix_authorized") is not True
        or int(rescue.get("original_matrix_cells_replaced", -1)) != 0
        or rescue.get("test_evaluated") is not False
    ):
        raise ValueError("matrix rescue did not authorize unchanged matrices")

    receptor_ids = [str(value) for value in preregistration["receptor_ids"]]
    receptor_rows = read_csv(paths["receptor_manifest"])
    observed_receptors = [row["conformer_id"] for row in receptor_rows]
    if set(observed_receptors) != set(receptor_ids):
        raise ValueError("receptor manifest and preregistration IDs differ")

    manifest_rows = read_csv(paths["ligand_manifest"])
    manifest_ids = [row["ligand_id"] for row in manifest_rows]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("ligand manifest contains duplicate IDs")
    roles = count_by_role_and_label(manifest_rows)
    expected_roles = {
        "development_train": {"active": 80, "decoy": 80},
        "development_validation": {"active": 40, "decoy": 40},
    }
    if roles != expected_roles:
        raise ValueError("development role counts differ or test rows are present")
    if {row["split"] for row in manifest_rows} != {"train", "validation"}:
        raise ValueError("ligand manifest includes a non-development split")

    train_rows = [
        row for row in manifest_rows if row["selection_role"] == "development_train"
    ]
    validation_rows = [
        row
        for row in manifest_rows
        if row["selection_role"] == "development_validation"
    ]
    train_groups = {row["split_group_id"] for row in train_rows}
    validation_groups = {row["split_group_id"] for row in validation_rows}
    if "" in train_groups | validation_groups:
        raise ValueError("a frozen split_group_id is missing")
    if train_groups & validation_groups:
        raise ValueError("a frozen split group crosses train and validation")
    train_scaffolds = {row["scaffold_smiles"] for row in train_rows}
    validation_scaffolds = {row["scaffold_smiles"] for row in validation_rows}
    if "" in train_scaffolds | validation_scaffolds:
        raise ValueError("a scaffold is missing")
    if train_scaffolds & validation_scaffolds:
        raise ValueError("a scaffold crosses train and validation")

    manifest_by_id = {row["ligand_id"]: row for row in manifest_rows}
    primary_rows = read_csv(paths["primary_matrix"])
    sensitivity_rows = read_csv(paths["sensitivity_matrix"])
    nonnegative = {
        "primary": audit_matrix_rows(
            "primary", primary_rows, manifest_by_id, receptor_ids
        ),
        "sensitivity": audit_matrix_rows(
            "sensitivity", sensitivity_rows, manifest_by_id, receptor_ids
        ),
    }
    allowed_nonnegative = int(
        dict(preregistration["matrix_admission"])[
            "maximum_allowed_nonnegative_score_pairs"
        ]
    )
    if any(value > allowed_nonnegative for value in nonnegative.values()):
        raise ValueError("a matrix contains a nonnegative docking score")

    return {
        "receptor_ids": receptor_ids,
        "manifest_rows": manifest_rows,
        "train_manifest_rows": train_rows,
        "validation_manifest_rows": validation_rows,
        "primary_rows": primary_rows,
        "sensitivity_rows": sensitivity_rows,
        "role_label_counts": roles,
        "train_group_count": len(train_groups),
        "validation_group_count": len(validation_groups),
        "train_validation_group_overlap": 0,
        "train_validation_scaffold_overlap": 0,
        "nonnegative_score_counts": nonnegative,
    }


def make_frozen_group_folds(
    rows: list[dict[str, str]], fold_count: int, seed: int
) -> dict[str, int]:
    fold_rows = [
        {**row, "scaffold_smiles": row["split_group_id"]} for row in rows
    ]
    assignments = make_scaffold_folds(fold_rows, fold_count, seed)
    scaffold_folds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        scaffold_folds[row["scaffold_smiles"]].add(assignments[row["ligand_id"]])
    if any(len(folds) != 1 for folds in scaffold_folds.values()):
        raise ValueError("a non-chiral scaffold crosses inner folds")
    return assignments


def trial_selection_key(
    trial: dict[str, object],
    cv: dict[str, object],
    family_order: dict[str, int],
) -> tuple[object, ...]:
    config = trial["config"]
    metrics = trial["mean_validation_metrics"]
    assert isinstance(config, dict)
    assert isinstance(metrics, dict)
    weights = config.get("weights", {})
    assert isinstance(weights, dict)
    metric_order = [
        str(cv["inner_selection_metric"]),
        *[str(value) for value in cv["inner_tie_breakers"]],
    ]
    return (
        *[-float(metrics[metric]) for metric in metric_order],
        float(trial["selection_metric_std"]),
        int(config["target_size"]),
        sum(float(value) for value in weights.values()),
        family_order[str(config["family"])],
        str(config["aggregation"]),
        json.dumps(config, sort_keys=True),
    )


def flatten_trial(method: str, trial: dict[str, object]) -> dict[str, object]:
    config = trial["config"]
    metrics = trial["mean_validation_metrics"]
    assert isinstance(config, dict)
    assert isinstance(metrics, dict)
    return {
        "method": method,
        "family": config["family"],
        "target_size": config["target_size"],
        "aggregation": config["aggregation"],
        "mean_inner_roc_auc": metrics["roc_auc"],
        "mean_inner_pr_auc": metrics["pr_auc_average_precision"],
        "mean_inner_bedroc_alpha_20": metrics["bedroc_alpha_20"],
        "selection_metric_std": trial["selection_metric_std"],
        "inner_subsets": json.dumps(trial["inner_subsets"]),
        "config": json.dumps(config, sort_keys=True),
    }


def metric_distribution(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "q05": percentile(values, 0.05),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "q95": percentile(values, 0.95),
        "maximum": max(values),
    }


def exact_random_subset_tables(
    matrix_rows: dict[str, list[dict[str, object]]],
    receptor_ids: list[str],
    subset_sizes: list[int],
    aggregation_methods: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for matrix_name, rows in matrix_rows.items():
        for size in subset_sizes:
            aggregations = ["min_score"] if size == 1 else aggregation_methods
            for aggregation in aggregations:
                group: list[dict[str, object]] = []
                for subset in itertools.combinations(receptor_ids, size):
                    metrics = subset_metrics(rows, subset, aggregation)
                    record = {
                        "matrix": matrix_name,
                        "target_size": size,
                        "aggregation": aggregation,
                        "subset": "+".join(subset),
                        **metrics,
                    }
                    detail_rows.append(record)
                    group.append(record)
                summary: dict[str, object] = {
                    "matrix": matrix_name,
                    "target_size": size,
                    "aggregation": aggregation,
                    "subset_count": len(group),
                    "interpretation": (
                        "exact uniform distribution over every fixed-size subset"
                    ),
                }
                for metric in RANDOM_SUMMARY_METRICS:
                    distribution = metric_distribution(
                        [float(row[metric]) for row in group]
                    )
                    for statistic, value in distribution.items():
                        summary[f"{metric}_{statistic}"] = value
                summary_rows.append(summary)
    return detail_rows, summary_rows


def gate_decision(
    selected_qubo_metrics: dict[str, dict[str, object]],
    single_best_metrics: dict[str, dict[str, object]],
    bootstrap: dict[str, dict[str, float | int]],
    gate: dict[str, object],
) -> tuple[dict[str, float], dict[str, bool], bool]:
    deltas = {
        "primary_bedroc": float(
            selected_qubo_metrics["primary"]["bedroc_alpha_20"]
        )
        - float(single_best_metrics["primary"]["bedroc_alpha_20"]),
        "primary_roc_auc": float(selected_qubo_metrics["primary"]["roc_auc"])
        - float(single_best_metrics["primary"]["roc_auc"]),
        "primary_pr_auc": float(
            selected_qubo_metrics["primary"]["pr_auc_average_precision"]
        )
        - float(single_best_metrics["primary"]["pr_auc_average_precision"]),
        "sensitivity_bedroc": float(
            selected_qubo_metrics["sensitivity"]["bedroc_alpha_20"]
        )
        - float(single_best_metrics["sensitivity"]["bedroc_alpha_20"]),
    }
    checks = {
        "primary_bedroc_delta": deltas["primary_bedroc"]
        >= float(gate["minimum_primary_bedroc_delta"]),
        "primary_roc_auc_delta": deltas["primary_roc_auc"]
        >= float(gate["minimum_primary_roc_auc_delta"]),
        "primary_pr_auc_delta": deltas["primary_pr_auc"]
        >= float(gate["minimum_primary_pr_auc_delta"]),
        "sensitivity_bedroc_delta": deltas["sensitivity_bedroc"]
        >= float(gate["minimum_sensitivity_bedroc_delta"]),
        "primary_bedroc_bootstrap_ci95_low": float(
            bootstrap["bedroc_alpha_20"]["ci95_low"]
        )
        >= float(gate["minimum_primary_bedroc_bootstrap_ci95_low"]),
    }
    return deltas, checks, all(checks.values())


def run_method_gate(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = load_execution_config(config_path)
    runtime = check_runtime(config)
    prereg_path, input_paths = checked_input_paths(config)
    preregistration = read_json(prereg_path)
    validate_preregistration(preregistration)
    audited = audit_inputs(preregistration, input_paths)
    receptor_ids = audited["receptor_ids"]
    train_manifest = audited["train_manifest_rows"]
    validation_manifest = audited["validation_manifest_rows"]
    primary_rows = audited["primary_rows"]
    sensitivity_rows = audited["sensitivity_rows"]
    assert isinstance(receptor_ids, list)
    assert isinstance(train_manifest, list)
    assert isinstance(validation_manifest, list)
    assert isinstance(primary_rows, list)
    assert isinstance(sensitivity_rows, list)

    outputs = config["outputs"]
    assert isinstance(outputs, dict)
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    core_outputs = [
        path for key, path in output_paths.items() if key != "run_directory"
    ]
    existing = [path for path in core_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("method-gate outputs exist; review before overwrite")
    if overwrite:
        for path in existing:
            path.unlink()
    output_paths["run_directory"].mkdir(parents=True, exist_ok=True)

    inner = preregistration["inner_selection"]
    gate = preregistration["validation_gate"]
    assert isinstance(inner, dict)
    assert isinstance(gate, dict)
    model = model_from_preregistration(preregistration)
    cv = cv_from_preregistration(preregistration)
    fold_count = int(inner["fold_count"])
    assignments = make_frozen_group_folds(
        train_manifest, fold_count, int(inner["fold_seed"])
    )
    fold_rows = [
        {
            "ligand_id": row["ligand_id"],
            "label": row["label"],
            "split_group_id": row["split_group_id"],
            "scaffold_smiles": row["scaffold_smiles"],
            "train_fold": assignments[row["ligand_id"]],
        }
        for row in sorted(train_manifest, key=lambda value: value["ligand_id"])
    ]
    fold_label_counts = {
        str(fold): {
            label: sum(
                row["label"] == label
                and assignments[row["ligand_id"]] == fold
                for row in train_manifest
            )
            for label in ("active", "decoy")
        }
        for fold in range(fold_count)
    }

    primary_by_id = {row["ligand_id"]: row for row in primary_rows}
    sensitivity_by_id = {row["ligand_id"]: row for row in sensitivity_rows}
    train_ids = {row["ligand_id"] for row in train_manifest}
    validation_ids = {row["ligand_id"] for row in validation_manifest}
    if train_ids & validation_ids:
        raise ValueError("train and validation ligand IDs overlap")
    inner_contexts: list[dict[str, object]] = []
    for validation_fold in range(fold_count):
        fold_validation_ids = {
            ligand_id
            for ligand_id, fold in assignments.items()
            if fold == validation_fold
        }
        inner_contexts.append(
            make_context(
                train_ids - fold_validation_ids,
                fold_validation_ids,
                primary_by_id,
                sensitivity_by_id,
                receptor_ids,
                model,
            )
        )

    configurations = method_configs(model, len(receptor_ids))
    selected_trials: dict[str, dict[str, object]] = {}
    all_trials: dict[str, list[dict[str, object]]] = {}
    for method, candidates in configurations.items():
        selected, trials = tune_configs(
            candidates, inner_contexts, receptor_ids, model, cv
        )
        selected_trials[method] = selected
        all_trials[method] = trials

    qubo_families = [str(value) for value in inner["qubo_families"]]
    family_order = {family: index for index, family in enumerate(qubo_families)}
    selected_qubo_family = min(
        qubo_families,
        key=lambda family: trial_selection_key(
            selected_trials[family], cv, family_order
        ),
    )

    final_context = make_context(
        train_ids,
        validation_ids,
        primary_by_id,
        sensitivity_by_id,
        receptor_ids,
        model,
    )
    fitted_methods: dict[str, dict[str, object]] = {}
    validation_metrics: dict[str, dict[str, dict[str, object]]] = {
        "primary": {},
        "sensitivity": {},
    }
    validation_scores: dict[
        str, dict[str, dict[str, dict[str, object]]]
    ] = {"primary": {}, "sensitivity": {}}
    selected_method_rows: list[dict[str, object]] = []
    for method, selected_trial in selected_trials.items():
        selected_config = dict(selected_trial["config"])
        subset, fit_details = fit_config(
            selected_config, final_context, receptor_ids, model
        )
        aggregation = str(selected_config["aggregation"])
        fitted_methods[method] = {
            "config": selected_config,
            "subset": list(subset),
            "fit_details": fit_details,
            "inner_cv": {
                "mean_validation_metrics": selected_trial[
                    "mean_validation_metrics"
                ],
                "selection_metric_std": selected_trial[
                    "selection_metric_std"
                ],
                "inner_subsets": selected_trial["inner_subsets"],
            },
        }
        selected_method_rows.append(
            {
                "method": method,
                "family": selected_config["family"],
                "target_size": len(subset),
                "aggregation": aggregation,
                "subset": "+".join(subset),
                "selected_for_qubo_gate": method == selected_qubo_family,
                "inner_cv_mean_roc_auc": selected_trial[
                    "mean_validation_metrics"
                ]["roc_auc"],
                "inner_cv_mean_pr_auc": selected_trial[
                    "mean_validation_metrics"
                ]["pr_auc_average_precision"],
                "inner_cv_mean_bedroc_alpha_20": selected_trial[
                    "mean_validation_metrics"
                ]["bedroc_alpha_20"],
                "inner_cv_selection_metric_std": selected_trial[
                    "selection_metric_std"
                ],
                "config": json.dumps(selected_config, sort_keys=True),
                "fit_details": json.dumps(fit_details, sort_keys=True),
            }
        )
        for matrix_name, rows in (
            ("primary", final_context["primary_validation"]),
            ("sensitivity", final_context["sensitivity_validation"]),
        ):
            assert isinstance(rows, list)
            validation_metrics[matrix_name][method] = subset_metrics(
                rows, subset, aggregation
            )
            validation_scores[matrix_name][method] = collect_scores(
                rows, subset, aggregation
            )

    bootstrap = paired_bootstrap_delta(
        validation_scores["primary"]["single_best"],
        validation_scores["primary"][selected_qubo_family],
        int(gate["bootstrap_iterations"]),
        int(gate["bootstrap_seed"]),
    )
    selected_qubo_metrics = {
        matrix: validation_metrics[matrix][selected_qubo_family]
        for matrix in ("primary", "sensitivity")
    }
    single_best_metrics = {
        matrix: validation_metrics[matrix]["single_best"]
        for matrix in ("primary", "sensitivity")
    }
    deltas, checks, gate_passed = gate_decision(
        selected_qubo_metrics, single_best_metrics, bootstrap, gate
    )
    status = (
        "development_gate_passed_test_locked"
        if gate_passed
        else "development_gate_failed_test_locked"
    )

    random_detail, random_summary = exact_random_subset_tables(
        {
            "primary": final_context["primary_validation"],
            "sensitivity": final_context["sensitivity_validation"],
        },
        receptor_ids,
        [int(value) for value in inner["subset_sizes"]],
        [str(value) for value in inner["aggregation_methods"]],
    )

    inner_trial_rows = [
        flatten_trial(method, trial)
        for method in configurations
        for trial in all_trials[method]
    ]
    validation_metric_rows = [
        {"matrix": matrix, "method": method, **metrics}
        for matrix, methods in validation_metrics.items()
        for method, metrics in methods.items()
    ]
    validation_score_rows = [
        {
            "matrix": matrix,
            "method": method,
            "ligand_id": ligand_id,
            "label": record["label"],
            "selection_role": "development_validation",
            "normalized_ensemble_score": record["score"],
        }
        for matrix, methods in validation_scores.items()
        for method, records in methods.items()
        for ligand_id, record in sorted(records.items())
    ]

    write_csv(output_paths["fold_assignments_csv"], fold_rows)
    write_csv(output_paths["inner_cv_trials_csv"], inner_trial_rows)
    write_csv(output_paths["selected_methods_csv"], selected_method_rows)
    write_csv(output_paths["validation_metrics_csv"], validation_metric_rows)
    write_csv(output_paths["validation_scores_csv"], validation_score_rows)
    write_csv(output_paths["exact_random_subsets_csv"], random_detail)
    write_csv(output_paths["exact_random_summary_csv"], random_summary)

    implementation_path = Path(__file__)
    dependency_names = (
        "run_development_scaffold_cv_gate.py",
        "normalized_receptor_qubo.py",
        "run_receptor_selection_validation_gate.py",
        "cross_validate_ensemble_mvp.py",
        "compare_receptor_screening.py",
    )
    candidate_protocol = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "preregistration": {
            "path": prereg_path.as_posix(),
            "sha256": file_sha256(prereg_path),
        },
        "runtime": runtime,
        "implementation": {
            "path": f"scripts/{implementation_path.name}",
            "sha256": file_sha256(implementation_path),
        },
        "implementation_dependencies": [
            {
                "path": f"scripts/{name}",
                "sha256": file_sha256(implementation_path.with_name(name)),
            }
            for name in dependency_names
        ],
        "data_boundary": {
            "train_ligand_count": len(train_ids),
            "validation_ligand_count": len(validation_ids),
            "test_status": "locked_unreleased",
            "test_evaluated": False,
            "validation_used_for_tuning": False,
            "validation_evaluations_per_fixed_method": 1,
        },
        "train_selected_methods": fitted_methods,
        "selected_qubo_family": selected_qubo_family,
        "validation_metrics": validation_metrics,
        "validation_gate": {
            "comparison_method": "single_best",
            "deltas": deltas,
            "paired_bootstrap": bootstrap,
            "acceptance_checks": checks,
            "all_checks_required": True,
            "gate_passed": gate_passed,
        },
        "test_release": {
            "automatic_release": False,
            "manual_review_required": True,
            "released": False,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(output_paths["candidate_protocol_json"], candidate_protocol)

    output_records = {
        key: {"path": path.as_posix(), "sha256": file_sha256(path)}
        for key, path in output_paths.items()
        if key not in {"run_directory", "summary_json"}
    }
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "preregistration": {
            "path": prereg_path.as_posix(),
            "sha256": file_sha256(prereg_path),
        },
        "runtime": runtime,
        "input_sha256": {
            key: file_sha256(path) for key, path in input_paths.items()
        },
        "input_audit": {
            "matrix_admission": "rescued_unchanged_e16_matrices",
            "receptor_count": len(receptor_ids),
            "ligand_count": len(audited["manifest_rows"]),
            "role_label_counts": audited["role_label_counts"],
            "train_validation_ligand_id_overlap": 0,
            "train_validation_group_overlap": audited[
                "train_validation_group_overlap"
            ],
            "train_validation_scaffold_overlap": audited[
                "train_validation_scaffold_overlap"
            ],
            "nonnegative_score_counts": audited["nonnegative_score_counts"],
            "test_rows_read": 0,
        },
        "inner_cross_validation": {
            "data_role": "development_train only",
            "fold_count": fold_count,
            "fold_seed": int(inner["fold_seed"]),
            "fold_label_counts": fold_label_counts,
            "normalization": inner["score_normalization"],
            "candidate_count_by_method": {
                method: len(trials) for method, trials in all_trials.items()
            },
        },
        "train_selected_methods": {
            method: {
                "subset": value["subset"],
                "aggregation": value["config"]["aggregation"],
                "inner_cv_mean_metrics": value["inner_cv"][
                    "mean_validation_metrics"
                ],
            }
            for method, value in fitted_methods.items()
        },
        "selected_qubo_family": selected_qubo_family,
        "validation_metrics": validation_metrics,
        "validation_gate": candidate_protocol["validation_gate"],
        "exact_random_subset_distribution": {
            "detail_row_count": len(random_detail),
            "summary_group_count": len(random_summary),
            "random_numbers_used": False,
        },
        "test_evaluated": False,
        "test_status": "locked_unreleased",
        "outputs": output_records,
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(output_paths["summary_json"], summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_method_gate(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
