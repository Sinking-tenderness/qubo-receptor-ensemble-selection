"""Independently audit the MAPK14 development method-gate outputs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .cross_validate_ensemble_mvp import paired_bootstrap_delta
    from .prepare_receptor import file_sha256
    from .run_receptor_selection_validation_gate import (
        compact_metrics,
        normalize_from_train,
        percentile,
        read_csv,
        subset_metrics,
    )
    from .run_stage05_mk14_method_gate import check_runtime, gate_decision
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from cross_validate_ensemble_mvp import paired_bootstrap_delta
    from prepare_receptor import file_sha256
    from run_receptor_selection_validation_gate import (
        compact_metrics,
        normalize_from_train,
        percentile,
        read_csv,
        subset_metrics,
    )
    from run_stage05_mk14_method_gate import check_runtime, gate_decision


REQUIRED_INPUTS = {
    "execution_config",
    "summary",
    "candidate_protocol",
    "validation_metrics",
    "validation_scores",
    "exact_random_subsets",
    "exact_random_summary",
}
METRIC_KEYS = (
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
RANDOM_METRICS = (
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


def load_audit_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != REQUIRED_INPUTS:
        raise ValueError("audit inputs do not match the required evidence set")
    for key, record in inputs.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"invalid audit input record: {key}")
    tolerance = float(config.get("numeric_tolerance", 0.0))
    if tolerance <= 0.0:
        raise ValueError("numeric_tolerance must be positive")
    if not config.get("output_json"):
        raise ValueError("output_json is missing")
    return config


def checked_paths(config: dict[str, object]) -> dict[str, Path]:
    inputs = config["inputs"]
    assert isinstance(inputs, dict)
    paths: dict[str, Path] = {}
    for key, record in inputs.items():
        assert isinstance(record, dict)
        path = Path(str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(record["sha256"]).upper():
            raise ValueError(f"audit input SHA-256 differs: {key}")
        paths[key] = path
    return paths


def assert_close(actual: object, expected: object, tolerance: float, name: str) -> None:
    actual_value = float(actual)
    expected_value = float(expected)
    if not math.isclose(
        actual_value,
        expected_value,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            f"numeric audit differs for {name}: {actual_value} != {expected_value}"
        )


def compare_metric_dicts(
    actual: dict[str, object],
    expected: dict[str, object],
    tolerance: float,
    prefix: str,
) -> None:
    for key in METRIC_KEYS:
        assert_close(actual[key], expected[key], tolerance, f"{prefix}.{key}")


def validation_score_groups(
    rows: list[dict[str, str]],
    expected_validation_ids: set[str],
    expected_methods: set[str],
) -> dict[tuple[str, str], dict[str, dict[str, object]]]:
    groups: dict[
        tuple[str, str], dict[str, dict[str, object]]
    ] = defaultdict(dict)
    for row in rows:
        if row.get("selection_role") != "development_validation":
            raise ValueError("validation score table contains a non-validation role")
        matrix = row["matrix"]
        method = row["method"]
        if matrix not in {"primary", "sensitivity"}:
            raise ValueError("validation score table contains an unknown matrix")
        if method not in expected_methods:
            raise ValueError("validation score table contains an unknown method")
        ligand_id = row["ligand_id"]
        if ligand_id in groups[(matrix, method)]:
            raise ValueError("validation score table contains a duplicate ligand")
        groups[(matrix, method)][ligand_id] = {
            "label": row["label"],
            "score": float(row["normalized_ensemble_score"]),
        }
    expected_keys = {
        (matrix, method)
        for matrix in ("primary", "sensitivity")
        for method in expected_methods
    }
    if set(groups) != expected_keys:
        raise ValueError("validation score matrix/method groups are incomplete")
    for key, records in groups.items():
        if set(records) != expected_validation_ids:
            raise ValueError(f"validation ligand IDs differ for {key}")
        if Counter(record["label"] for record in records.values()) != {
            "active": 40,
            "decoy": 40,
        }:
            raise ValueError(f"validation labels differ for {key}")
    return groups


def recompute_random_rows(
    matrix_rows: dict[str, list[dict[str, object]]],
    receptor_ids: list[str],
    subset_sizes: list[int],
    aggregations: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    details: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for matrix, rows in matrix_rows.items():
        for size in subset_sizes:
            methods = ["min_score"] if size == 1 else aggregations
            for aggregation in methods:
                group: list[dict[str, object]] = []
                for subset in itertools.combinations(receptor_ids, size):
                    record = {
                        "matrix": matrix,
                        "target_size": size,
                        "aggregation": aggregation,
                        "subset": "+".join(subset),
                        **subset_metrics(rows, subset, aggregation),
                    }
                    details.append(record)
                    group.append(record)
                summary: dict[str, object] = {
                    "matrix": matrix,
                    "target_size": size,
                    "aggregation": aggregation,
                    "subset_count": len(group),
                }
                for metric in RANDOM_METRICS:
                    values = [float(row[metric]) for row in group]
                    statistics_by_name = {
                        "minimum": min(values),
                        "q05": percentile(values, 0.05),
                        "median": statistics.median(values),
                        "mean": statistics.fmean(values),
                        "population_std": statistics.pstdev(values),
                        "q95": percentile(values, 0.95),
                        "maximum": max(values),
                    }
                    for name, value in statistics_by_name.items():
                        summary[f"{metric}_{name}"] = value
                summaries.append(summary)
    return details, summaries


def compare_random_tables(
    observed_details: list[dict[str, str]],
    observed_summaries: list[dict[str, str]],
    expected_details: list[dict[str, object]],
    expected_summaries: list[dict[str, object]],
    tolerance: float,
) -> None:
    detail_key = lambda row: (
        row["matrix"],
        int(row["target_size"]),
        row["aggregation"],
        row["subset"],
    )
    observed_detail_map = {detail_key(row): row for row in observed_details}
    expected_detail_map = {detail_key(row): row for row in expected_details}
    if set(observed_detail_map) != set(expected_detail_map):
        raise ValueError("exact random-subset detail keys differ")
    for key, expected in expected_detail_map.items():
        observed = observed_detail_map[key]
        for metric in METRIC_KEYS:
            assert_close(
                observed[metric],
                expected[metric],
                tolerance,
                f"random_detail.{key}.{metric}",
            )

    summary_key = lambda row: (
        row["matrix"],
        int(row["target_size"]),
        row["aggregation"],
    )
    observed_summary_map = {summary_key(row): row for row in observed_summaries}
    expected_summary_map = {summary_key(row): row for row in expected_summaries}
    if set(observed_summary_map) != set(expected_summary_map):
        raise ValueError("exact random-subset summary keys differ")
    for key, expected in expected_summary_map.items():
        observed = observed_summary_map[key]
        if int(observed["subset_count"]) != int(expected["subset_count"]):
            raise ValueError(f"random subset count differs for {key}")
        for metric in RANDOM_METRICS:
            for name in (
                "minimum",
                "q05",
                "median",
                "mean",
                "population_std",
                "q95",
                "maximum",
            ):
                column = f"{metric}_{name}"
                assert_close(
                    observed[column],
                    expected[column],
                    tolerance,
                    f"random_summary.{key}.{column}",
                )


def run_audit(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = load_audit_config(config_path)
    paths = checked_paths(config)
    tolerance = float(config["numeric_tolerance"])
    output_path = Path(str(config["output_json"]))
    if output_path.exists() and not overwrite:
        raise FileExistsError("independent audit output exists; review before overwrite")

    execution = read_json(paths["execution_config"])
    runtime = check_runtime(execution)
    summary = read_json(paths["summary"])
    candidate = read_json(paths["candidate_protocol"])
    if summary.get("status") != "development_gate_failed_test_locked":
        raise ValueError("unexpected method-gate summary status")
    if candidate.get("status") != summary.get("status"):
        raise ValueError("candidate and summary statuses differ")
    if summary.get("test_evaluated") is not False:
        raise ValueError("summary reports that test was evaluated")
    boundary = candidate.get("data_boundary")
    if (
        not isinstance(boundary, dict)
        or boundary.get("test_evaluated") is not False
        or boundary.get("validation_used_for_tuning") is not False
    ):
        raise ValueError("candidate data boundary is invalid")

    summary_outputs = summary.get("outputs")
    if not isinstance(summary_outputs, dict):
        raise ValueError("summary output hashes are missing")
    for record in summary_outputs.values():
        if not isinstance(record, dict):
            raise ValueError("invalid summary output record")
        path = Path(str(record["path"]))
        if not path.is_file() or file_sha256(path) != str(record["sha256"]).upper():
            raise ValueError(f"summary output hash differs: {path}")

    prereg_record = execution.get("preregistration")
    inputs = execution.get("inputs")
    if not isinstance(prereg_record, dict) or not isinstance(inputs, dict):
        raise ValueError("execution config input records are missing")
    prereg_path = Path(str(prereg_record["path"]))
    if file_sha256(prereg_path) != str(prereg_record["sha256"]).upper():
        raise ValueError("preregistration hash differs")
    preregistration = read_json(prereg_path)
    ligand_path = Path(str(inputs["ligand_manifest"]["path"]))
    primary_path = Path(str(inputs["primary_matrix"]["path"]))
    sensitivity_path = Path(str(inputs["sensitivity_matrix"]["path"]))
    for key, path in (
        ("ligand_manifest", ligand_path),
        ("primary_matrix", primary_path),
        ("sensitivity_matrix", sensitivity_path),
    ):
        if file_sha256(path) != str(inputs[key]["sha256"]).upper():
            raise ValueError(f"execution input hash differs: {key}")

    manifest_rows = read_csv(ligand_path)
    train_ids = {
        row["ligand_id"]
        for row in manifest_rows
        if row["selection_role"] == "development_train"
    }
    validation_ids = {
        row["ligand_id"]
        for row in manifest_rows
        if row["selection_role"] == "development_validation"
    }
    if len(train_ids) != 160 or len(validation_ids) != 80 or train_ids & validation_ids:
        raise ValueError("development train/validation IDs are invalid")
    if any(
        row["selection_role"]
        not in {"development_train", "development_validation"}
        for row in manifest_rows
    ):
        raise ValueError("ligand manifest contains a test or unknown role")

    expected_methods = set(candidate["train_selected_methods"])
    score_rows = read_csv(paths["validation_scores"])
    score_groups = validation_score_groups(
        score_rows, validation_ids, expected_methods
    )
    metric_rows = read_csv(paths["validation_metrics"])
    metric_map = {
        (row["matrix"], row["method"]): row for row in metric_rows
    }
    if len(metric_map) != len(metric_rows):
        raise ValueError("validation metric rows are duplicated")
    recomputed_metrics: dict[str, dict[str, dict[str, object]]] = {
        "primary": {},
        "sensitivity": {},
    }
    for key, records in score_groups.items():
        matrix, method = key
        metrics = compact_metrics(ranked_metrics_with_ids(records))
        recomputed_metrics[matrix][method] = metrics
        compare_metric_dicts(metrics, metric_map[key], tolerance, f"metrics.{key}")
        compare_metric_dicts(
            metrics,
            candidate["validation_metrics"][matrix][method],
            tolerance,
            f"candidate_metrics.{key}",
        )

    gate = preregistration["validation_gate"]
    assert isinstance(gate, dict)
    selected_qubo = str(candidate["selected_qubo_family"])
    bootstrap = paired_bootstrap_delta(
        score_groups[("primary", "single_best")],
        score_groups[("primary", selected_qubo)],
        int(gate["bootstrap_iterations"]),
        int(gate["bootstrap_seed"]),
    )
    recorded_gate = candidate["validation_gate"]
    assert isinstance(recorded_gate, dict)
    recorded_bootstrap = recorded_gate["paired_bootstrap"]
    assert isinstance(recorded_bootstrap, dict)
    for metric, values in bootstrap.items():
        for key, value in values.items():
            assert_close(
                value,
                recorded_bootstrap[metric][key],
                tolerance,
                f"bootstrap.{metric}.{key}",
            )
    selected_metrics = {
        matrix: recomputed_metrics[matrix][selected_qubo]
        for matrix in ("primary", "sensitivity")
    }
    baseline_metrics = {
        matrix: recomputed_metrics[matrix]["single_best"]
        for matrix in ("primary", "sensitivity")
    }
    deltas, checks, passed = gate_decision(
        selected_metrics, baseline_metrics, bootstrap, gate
    )
    for key, value in deltas.items():
        assert_close(
            value,
            recorded_gate["deltas"][key],
            tolerance,
            f"gate_delta.{key}",
        )
    if checks != recorded_gate["acceptance_checks"] or passed is not False:
        raise ValueError("recomputed gate decision differs")

    receptor_ids = [str(value) for value in preregistration["receptor_ids"]]
    primary_rows = read_csv(primary_path)
    sensitivity_rows = read_csv(sensitivity_path)
    primary_by_id = {row["ligand_id"]: row for row in primary_rows}
    sensitivity_by_id = {row["ligand_id"]: row for row in sensitivity_rows}
    primary_train, primary_validation, _ = normalize_from_train(
        [primary_by_id[key] for key in sorted(train_ids)],
        [primary_by_id[key] for key in sorted(validation_ids)],
        receptor_ids,
    )
    sensitivity_train, sensitivity_validation, _ = normalize_from_train(
        [sensitivity_by_id[key] for key in sorted(train_ids)],
        [sensitivity_by_id[key] for key in sorted(validation_ids)],
        receptor_ids,
    )
    if len(primary_train) != 160 or len(sensitivity_train) != 160:
        raise ValueError("normalization train rows differ")
    inner = preregistration["inner_selection"]
    assert isinstance(inner, dict)
    expected_random_details, expected_random_summaries = recompute_random_rows(
        {
            "primary": primary_validation,
            "sensitivity": sensitivity_validation,
        },
        receptor_ids,
        [int(value) for value in inner["subset_sizes"]],
        [str(value) for value in inner["aggregation_methods"]],
    )
    compare_random_tables(
        read_csv(paths["exact_random_subsets"]),
        read_csv(paths["exact_random_summary"]),
        expected_random_details,
        expected_random_summaries,
        tolerance,
    )

    audit = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "independent_audit_ok",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "implementation": {
            "path": f"scripts/{Path(__file__).name}",
            "sha256": file_sha256(Path(__file__)),
        },
        "input_sha256": {key: file_sha256(path) for key, path in paths.items()},
        "runtime": runtime,
        "checks": {
            "all_input_hashes_match": True,
            "all_summary_output_hashes_match": True,
            "validation_score_group_count": len(score_groups),
            "validation_rows_per_group": 80,
            "validation_label_counts_per_group": {"active": 40, "decoy": 40},
            "test_rows_observed": 0,
            "all_validation_metrics_reproduced": True,
            "paired_bootstrap_reproduced": True,
            "exact_random_detail_rows_reproduced": len(expected_random_details),
            "exact_random_summary_groups_reproduced": len(
                expected_random_summaries
            ),
            "gate_decision_reproduced": True,
        },
        "recomputed_gate": {
            "selected_qubo_family": selected_qubo,
            "deltas": deltas,
            "paired_bootstrap": bootstrap,
            "acceptance_checks": checks,
            "gate_passed": passed,
            "status": "development_gate_failed_test_locked",
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(output_path, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_audit(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
