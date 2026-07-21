"""Evaluate the frozen MAPK14 receptor subsets once on fresh validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .evaluate_virtual_screening import bedroc
    from .prepare_receptor import file_sha256
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from evaluate_virtual_screening import bedroc
    from prepare_receptor import file_sha256


SEED_IDS = ("seed0", "seed1", "seed2")
MATRIX_IDS = ("primary", "sensitivity", *SEED_IDS)
AGGREGATE_DIRECTORY = Path(
    "results/runs/stage05_mk14_fresh_validation_e32_aggregated"
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


def validation_matrices(
    primary_rows: list[dict[str, str]],
    sensitivity_rows: list[dict[str, str]],
    long_rows: list[dict[str, str]],
    receptor_ids: list[str],
) -> dict[str, dict[str, dict[str, object]]]:
    primary = rows_by_id(primary_rows, "validation primary matrix")
    sensitivity = rows_by_id(
        sensitivity_rows, "validation sensitivity matrix"
    )
    if set(primary) != set(sensitivity):
        raise ValueError("validation primary and sensitivity IDs differ")
    matrices: dict[str, dict[str, dict[str, object]]] = {
        "primary": {key: dict(value) for key, value in primary.items()},
        "sensitivity": {
            key: dict(value) for key, value in sensitivity.items()
        },
    }
    long_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for row in long_rows:
        key = (row["ligand_id"], row["receptor_id"])
        if key in long_by_pair:
            raise ValueError(f"duplicate validation pair: {key}")
        long_by_pair[key] = row
    if len(long_by_pair) != len(primary) * len(receptor_ids):
        raise ValueError("validation long matrix pair count differs")
    columns = {
        "seed0": "seed0_representative_score",
        "seed1": "seed1_representative_score",
        "seed2": "seed2_representative_score",
    }
    for seed, column in columns.items():
        seed_rows: dict[str, dict[str, object]] = {}
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
            seed_rows[ligand_id] = row
        matrices[seed] = seed_rows
    return matrices


def normalize_matrices(
    matrices: dict[str, dict[str, dict[str, object]]],
    bounds: dict[str, object],
    receptor_ids: list[str],
) -> dict[str, dict[str, dict[str, object]]]:
    output: dict[str, dict[str, dict[str, object]]] = {}
    for matrix_id in MATRIX_IDS:
        matrix_bounds = bounds[matrix_id]
        output[matrix_id] = {}
        for ligand_id, row in matrices[matrix_id].items():
            normalized = dict(row)
            for receptor_id in receptor_ids:
                lower = float(matrix_bounds[receptor_id]["minimum"])
                upper = float(matrix_bounds[receptor_id]["maximum"])
                value = float(row[receptor_id])
                normalized[receptor_id] = (
                    0.0
                    if upper == lower
                    else (value - lower) / (upper - lower)
                )
            output[matrix_id][ligand_id] = normalized
    return output


def method_records(
    rows: dict[str, dict[str, object]], subset: list[str]
) -> dict[str, dict[str, object]]:
    return {
        ligand_id: {
            "label": row["label"],
            "score": min(float(row[receptor_id]) for receptor_id in subset),
        }
        for ligand_id, row in rows.items()
    }


def robust_bedroc(
    metrics: dict[str, dict[str, object]]
) -> dict[str, float]:
    seed_values = [
        float(metrics[seed]["bedroc_alpha_20"]) for seed in SEED_IDS
    ]
    return {
        "primary": float(metrics["primary"]["bedroc_alpha_20"]),
        "sensitivity": float(
            metrics["sensitivity"]["bedroc_alpha_20"]
        ),
        "mean_seed": statistics.fmean(seed_values),
        "worst_seed": min(seed_values),
    }


def quantile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile input is invalid")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sampled_bedroc(
    records: dict[str, dict[str, object]],
    grouped_ids: dict[str, list[str]],
    sampled_groups: list[str],
) -> float:
    ranked_values: list[tuple[float, int, str, int]] = []
    for draw_index, group_id in enumerate(sampled_groups):
        for ligand_id in grouped_ids[group_id]:
            record = records[ligand_id]
            ranked_values.append(
                (
                    float(record["score"]),
                    draw_index,
                    ligand_id,
                    int(record["label"] == "active"),
                )
            )
    ranked_values.sort(key=lambda value: (value[0], value[1], value[2]))
    ranked = [
        {"binary_label": binary_label}
        for _, _, _, binary_label in ranked_values
    ]
    return float(bedroc(ranked, 20.0))


def paired_group_bootstrap(
    records_by_method: dict[str, dict[str, dict[str, object]]],
    group_by_ligand: dict[str, str],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    method_ids = list(records_by_method)
    ligand_ids = set(records_by_method[method_ids[0]])
    if any(set(records_by_method[method]) != ligand_ids for method in method_ids):
        raise ValueError("bootstrap methods contain different ligand IDs")
    if set(group_by_ligand) != ligand_ids:
        raise ValueError("bootstrap group map differs from score records")
    grouped_ids: dict[str, list[str]] = defaultdict(list)
    for ligand_id, group_id in group_by_ligand.items():
        grouped_ids[group_id].append(ligand_id)
    for ligand_group in grouped_ids.values():
        ligand_group.sort()
    group_ids = sorted(grouped_ids)
    rng = random.Random(seed)
    deltas = {
        "matched_linear_top_k": [],
        "nested_exhaustive_final": [],
    }
    valid = 0
    attempts = 0
    while valid < replicates:
        attempts += 1
        if attempts > replicates * 2:
            raise ValueError("too many bootstrap samples lacked both labels")
        sampled = rng.choices(group_ids, k=len(group_ids))
        q_value = sampled_bedroc(
            records_by_method["pair_synergy_qubo"], grouped_ids, sampled
        )
        if not math.isfinite(q_value):
            continue
        values: dict[str, float] = {}
        invalid = False
        for comparator in deltas:
            value = sampled_bedroc(
                records_by_method[comparator], grouped_ids, sampled
            )
            if not math.isfinite(value):
                invalid = True
                break
            values[comparator] = q_value - value
        if invalid:
            continue
        for comparator, value in values.items():
            deltas[comparator].append(value)
        valid += 1
    return {
        "unit": "split_group_id block",
        "seed": seed,
        "valid_replicates": valid,
        "attempts": attempts,
        "confidence_level": 0.95,
        "deltas": {
            comparator: {
                "mean": statistics.fmean(values),
                "lower_95pct": quantile(values, 0.025),
                "upper_95pct": quantile(values, 0.975),
            }
            for comparator, values in deltas.items()
        },
    }


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = read_json(config_path)
    outputs = config["outputs"]
    panel_path = Path(str(outputs["fresh_panel_csv"]))
    model_path = Path(str(outputs["frozen_model_artifact_json"]))
    result_path = Path(str(outputs["validation_result_json"]))
    aggregate_summary_path = AGGREGATE_DIRECTORY / "summary.json"
    primary_path = AGGREGATE_DIRECTORY / "primary_median_score_matrix.csv"
    sensitivity_path = (
        AGGREGATE_DIRECTORY / "sensitivity_minimum_score_matrix.csv"
    )
    long_path = AGGREGATE_DIRECTORY / "aggregated_seed_scores.csv"
    for path in (
        panel_path,
        model_path,
        aggregate_summary_path,
        primary_path,
        sensitivity_path,
        long_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if result_path.exists() and not overwrite:
        raise FileExistsError("validation result exists; use --overwrite")

    model = read_json(model_path)
    if model.get("status") != "fresh_validation_model_frozen_scores_unavailable":
        raise ValueError("fresh validation model was not frozen")
    if str(model["config"]["sha256"]) != file_sha256(config_path):
        raise ValueError("frozen model config hash differs")
    aggregate = read_json(aggregate_summary_path)
    panel_rows = read_csv(panel_path)
    panel_by_id = rows_by_id(panel_rows, "fresh validation panel")
    receptor_ids = [
        str(value) for value in config["frozen_methods"]["fixed_receptor_union"]
    ]
    if (
        aggregate.get("status") != "ok"
        or int(aggregate["ligand_count"]) != len(panel_rows)
        or int(aggregate["receptor_count"]) != len(receptor_ids)
        or int(aggregate["seed_count"]) != len(SEED_IDS)
        or int(aggregate["aggregated_pair_count"])
        != len(panel_rows) * len(receptor_ids)
        or int(aggregate.get("locked_test_manifest_rows", -1)) != 0
    ):
        raise ValueError("fresh validation aggregate did not pass")
    aggregate_outputs = aggregate["outputs"]
    for key, path in (
        ("aggregated_long_csv", long_path),
        ("primary_median_matrix_csv", primary_path),
        ("sensitivity_minimum_matrix_csv", sensitivity_path),
    ):
        if file_sha256(path) != str(aggregate_outputs[key]["sha256"]).upper():
            raise ValueError(f"aggregate output hash differs: {key}")

    matrices = validation_matrices(
        read_csv(primary_path),
        read_csv(sensitivity_path),
        read_csv(long_path),
        receptor_ids,
    )
    if any(set(matrix) != set(panel_by_id) for matrix in matrices.values()):
        raise ValueError("fresh validation matrices and panel IDs differ")
    if any(
        row["split"] != "validation"
        or row["selection_role"] != "fresh_validation_preregistered"
        for row in panel_rows
    ):
        raise ValueError("fresh validation panel role or split differs")
    for matrix_id, matrix in matrices.items():
        for ligand_id, row in matrix.items():
            source = panel_by_id[ligand_id]
            if (
                row["label"] != source["label"]
                or row["selection_role"] != source["selection_role"]
            ):
                raise ValueError(
                    f"fresh validation metadata differs: {matrix_id}/{ligand_id}"
                )
    normalized = normalize_matrices(
        matrices, model["normalization_bounds"], receptor_ids
    )
    methods = {
        key: [str(value) for value in values]
        for key, values in dict(config["frozen_methods"]).items()
        if key
        in {
            "pair_synergy_qubo",
            "matched_linear_top_k",
            "nested_exhaustive_final",
            "single_best",
            "nested_greedy_final",
        }
    }
    records = {
        method: {
            matrix_id: method_records(normalized[matrix_id], subset)
            for matrix_id in MATRIX_IDS
        }
        for method, subset in methods.items()
    }
    if any(
        records["pair_synergy_qubo"][matrix_id]
        != records["nested_greedy_final"][matrix_id]
        for matrix_id in MATRIX_IDS
    ):
        raise ValueError("frozen QUBO and greedy predictions unexpectedly differ")
    metrics = {
        method: {
            matrix_id: ranked_metrics_with_ids(matrix_records)
            for matrix_id, matrix_records in method_records_by_matrix.items()
        }
        for method, method_records_by_matrix in records.items()
    }
    robust = {method: robust_bedroc(value) for method, value in metrics.items()}
    q = robust["pair_synergy_qubo"]

    def deltas(comparator: str) -> dict[str, float]:
        baseline = robust[comparator]
        return {
            key: q[key] - baseline[key]
            for key in ("primary", "sensitivity", "mean_seed", "worst_seed")
        }

    comparison = {
        comparator: deltas(comparator)
        for comparator in (
            "matched_linear_top_k",
            "nested_exhaustive_final",
            "single_best",
        )
    }
    bootstrap_config = config["evaluation"]["paired_bootstrap"]
    bootstrap = paired_group_bootstrap(
        {
            method: records[method]["primary"]
            for method in (
                "pair_synergy_qubo",
                "matched_linear_top_k",
                "nested_exhaustive_final",
            )
        },
        {
            ligand_id: panel_by_id[ligand_id]["split_group_id"]
            for ligand_id in panel_by_id
        },
        int(bootstrap_config["replicates"]),
        int(bootstrap_config["seed"]),
    )
    acceptance = config["acceptance"]
    checks = {
        "primary_vs_matched_linear": comparison["matched_linear_top_k"][
            "primary"
        ]
        >= float(acceptance["minimum_primary_bedroc_delta_vs_matched_linear"]),
        "mean_seed_vs_matched_linear": comparison["matched_linear_top_k"][
            "mean_seed"
        ]
        >= float(acceptance["minimum_mean_seed_bedroc_delta_vs_matched_linear"]),
        "worst_seed_vs_matched_linear": comparison["matched_linear_top_k"][
            "worst_seed"
        ]
        >= float(acceptance["minimum_worst_seed_bedroc_delta_vs_matched_linear"]),
        "bootstrap_vs_matched_linear": bootstrap["deltas"][
            "matched_linear_top_k"
        ]["lower_95pct"]
        >= float(
            acceptance[
                "minimum_group_bootstrap_95pct_lower_bedroc_delta_vs_matched_linear"
            ]
        ),
        "primary_vs_exhaustive": comparison["nested_exhaustive_final"][
            "primary"
        ]
        >= float(acceptance["minimum_primary_bedroc_delta_vs_exhaustive"]),
        "mean_seed_vs_exhaustive": comparison["nested_exhaustive_final"][
            "mean_seed"
        ]
        >= float(acceptance["minimum_mean_seed_bedroc_delta_vs_exhaustive"]),
        "worst_seed_vs_exhaustive": comparison["nested_exhaustive_final"][
            "worst_seed"
        ]
        >= float(acceptance["minimum_worst_seed_bedroc_delta_vs_exhaustive"]),
        "bootstrap_vs_exhaustive": bootstrap["deltas"][
            "nested_exhaustive_final"
        ]["lower_95pct"]
        >= float(
            acceptance[
                "minimum_group_bootstrap_95pct_lower_bedroc_delta_vs_exhaustive"
            ]
        ),
        "primary_vs_single_best": comparison["single_best"]["primary"]
        >= float(acceptance["minimum_primary_bedroc_delta_vs_single_best"]),
        "complete_three_seed_matrix": True,
        "zero_failed_receptor_ligand_pairs": True,
        "qubo_equals_frozen_greedy": True,
    }
    passed = all(checks.values())
    status = (
        "fresh_validation_passed_test_locked"
        if passed
        else "fresh_validation_failed_test_locked"
    )

    normalized_score_path = AGGREGATE_DIRECTORY / "normalized_method_scores.csv"
    metric_path = AGGREGATE_DIRECTORY / "frozen_method_metrics.csv"
    normalized_rows = [
        {
            "matrix": matrix_id,
            "method": method,
            "ligand_id": ligand_id,
            "label": record["label"],
            "normalized_ensemble_score": record["score"],
        }
        for matrix_id in MATRIX_IDS
        for method in methods
        for ligand_id, record in sorted(records[method][matrix_id].items())
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
        for method, values in metrics.items()
        for matrix_id, metric in values.items()
    ]
    write_csv(normalized_score_path, normalized_rows)
    write_csv(metric_path, metric_rows)
    result = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": status,
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "frozen_model": {
            "path": model_path.as_posix(),
            "sha256": file_sha256(model_path),
        },
        "aggregate_summary": {
            "path": aggregate_summary_path.as_posix(),
            "sha256": file_sha256(aggregate_summary_path),
        },
        "data_boundary": {
            "fresh_validation_ligands": len(panel_rows),
            "fresh_validation_label_counts": dict(
                sorted(Counter(row["label"] for row in panel_rows).items())
            ),
            "validation_scores_read_once": True,
            "test_rows_read": 0,
            "test_scores_read": 0,
        },
        "frozen_methods": methods,
        "method_metrics": metrics,
        "robust_bedroc": robust,
        "qubo_bedroc_deltas": comparison,
        "paired_group_bootstrap": bootstrap,
        "acceptance_checks": checks,
        "all_checks_passed": passed,
        "greedy_interpretation": config["frozen_methods"][
            "greedy_interpretation"
        ],
        "outputs": {
            "normalized_method_scores_csv": {
                "path": normalized_score_path.as_posix(),
                "sha256": file_sha256(normalized_score_path),
            },
            "frozen_method_metrics_csv": {
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
                "status": status,
                "robust_bedroc": robust,
                "qubo_bedroc_deltas": comparison,
                "bootstrap": bootstrap,
                "acceptance_checks": checks,
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
