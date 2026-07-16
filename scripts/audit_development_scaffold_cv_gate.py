"""Independently audit development-only scaffold-CV gate outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score


CORE_METRICS = (
    "roc_auc",
    "pr_auc_average_precision",
    "bedroc_alpha_20",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def independent_metrics(
    records: list[dict[str, object]], include_sklearn: bool = True
) -> dict[str, float]:
    ranked = sorted(
        records,
        key=lambda row: (float(row["score"]), str(row.get("ligand_id", ""))),
    )
    labels = [int(row["label"] == "active") for row in ranked]
    if not labels or not any(labels) or all(labels):
        raise ValueError("metrics require both active and decoy records")
    ranking_scores = [-float(row["score"]) for row in ranked]
    active_total = sum(labels)
    precision_sum = sum(
        sum(labels[:rank]) / rank
        for rank, label in enumerate(labels, start=1)
        if label
    )

    total = len(labels)
    alpha = 20.0
    active_ranks = [
        rank for rank, label in enumerate(labels, start=1) if label
    ]

    def exponential_sum(ranks: list[int] | range) -> float:
        return sum(math.exp(-alpha * rank / total) for rank in ranks)

    random_sum = active_total * (
        exponential_sum(range(1, total + 1)) / total
    )
    observed_rie = exponential_sum(active_ranks) / random_sum
    maximum_rie = exponential_sum(range(1, active_total + 1)) / random_sum
    minimum_rie = exponential_sum(
        range(total - active_total + 1, total + 1)
    ) / random_sum
    result = {
        "roc_auc": float(roc_auc_score(labels, ranking_scores)),
        "pr_auc_average_precision": float(precision_sum / active_total),
        "bedroc_alpha_20": float(
            (observed_rie - minimum_rie) / (maximum_rie - minimum_rie)
        ),
    }
    if include_sklearn:
        result["pr_auc_sklearn"] = float(
            average_precision_score(labels, ranking_scores)
        )
    return result


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def independent_paired_bootstrap(
    first: dict[str, dict[str, object]],
    second: dict[str, dict[str, object]],
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    ligand_ids = sorted(first)
    if set(ligand_ids) != set(second):
        raise ValueError("paired bootstrap inputs have different ligand IDs")
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {
        metric: [] for metric in CORE_METRICS
    }
    skipped = 0
    for _ in range(iterations):
        selected = [rng.choice(ligand_ids) for _ in ligand_ids]
        first_records = [
            {**first[ligand_id], "ligand_id": f"{ligand_id}__{index}"}
            for index, ligand_id in enumerate(selected)
        ]
        second_records = [
            {**second[ligand_id], "ligand_id": f"{ligand_id}__{index}"}
            for index, ligand_id in enumerate(selected)
        ]
        if {row["label"] for row in first_records} != {"active", "decoy"}:
            skipped += 1
            continue
        first_metrics = independent_metrics(first_records, False)
        second_metrics = independent_metrics(second_records, False)
        for metric in CORE_METRICS:
            samples[metric].append(
                second_metrics[metric] - first_metrics[metric]
            )
    return {
        metric: {
            "mean_delta": sum(values) / len(values),
            "ci95_low": percentile(values, 0.025),
            "ci95_high": percentile(values, 0.975),
            "n_bootstrap_used": len(values),
            "n_bootstrap_skipped": skipped,
        }
        for metric, values in samples.items()
    }


def collect_exact_search_records(value: object) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    if isinstance(value, dict):
        if isinstance(value.get("exact_search"), dict):
            output.append(value["exact_search"])
        for child in value.values():
            output.extend(collect_exact_search_records(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(collect_exact_search_records(child))
    return output


def audit_consensus_constraints(
    outer_rows: list[dict[str, str]],
    minimum_frequency: float,
    core_size: int = 0,
) -> bool:
    """Verify consensus and core-plus-one rows independently."""
    consensus_rows = [
        row
        for row in outer_rows
        if row.get("method") in {"consensus_qubo", "core_plus_one_qubo"}
    ]
    for row in consensus_rows:
        required = set(json.loads(row["consensus_required_receptors"]))
        selected = set(row["subset"].split("+"))
        reference_inner_subsets = json.loads(
            row["consensus_reference_inner_subsets"]
        )
        reference_config = json.loads(row["consensus_reference_config"])
        if not reference_inner_subsets:
            return False
        counts = defaultdict(int)
        for subset in reference_inner_subsets:
            values = set(str(value) for value in subset)
            for receptor_id in values:
                counts[receptor_id] += 1
        expected = {
            receptor_id
            for receptor_id, count in counts.items()
            if count / len(reference_inner_subsets)
            >= minimum_frequency - 1e-12
        }
        selected_config = json.loads(row["selected_config"])
        config_required = set(selected_config.get("required_receptors", []))
        family = selected_config.get("family")
        if family == "core_plus_one_qubo":
            if core_size < 1:
                return False
            qualified = {
                receptor_id
                for receptor_id, count in counts.items()
                if count / len(reference_inner_subsets)
                >= minimum_frequency - 1e-12
            }
            expected = set(
                sorted(
                    qualified,
                    key=lambda receptor_id: (-counts[receptor_id], receptor_id),
                )[:core_size]
            )
        if (
            required != expected
            or reference_config.get("family") != "coverage_qubo"
            or config_required != required
            or not required.issubset(selected)
            or len(required) > int(row["target_size"])
            or (
                family == "core_plus_one_qubo"
                and len(required) != core_size
            )
        ):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)

    config = json.loads(args.config.read_text(encoding="ascii"))
    inputs = {key: Path(value) for key, value in config["inputs"].items()}
    run_directory = Path(config["outputs"]["run_directory"])
    summary_path = run_directory / "summary.json"
    output_paths = {
        "summary": summary_path,
        "method_metrics": run_directory / "method_metrics.csv",
        "oof_scores": run_directory / "oof_scores.csv",
        "fold_assignments": run_directory / "fold_assignments.csv",
        "outer_fold_results": run_directory / "outer_fold_results.csv",
    }
    for path in [*inputs.values(), *output_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    cv = config["cross_validation"]
    development_splits = set(cv["development_splits"])
    locked_split = str(cv["locked_split"])
    split_rows = read_csv(inputs["split_manifest"])
    manifest_by_id = {row["ligand_id"]: row for row in split_rows}
    development_ids = {
        row["ligand_id"]
        for row in split_rows
        if row["split"] in development_splits
    }
    locked_ids = {
        row["ligand_id"]
        for row in split_rows
        if row["split"] == locked_split
    }

    oof_rows = read_csv(output_paths["oof_scores"])
    groups: dict[
        tuple[str, str], dict[str, dict[str, object]]
    ] = defaultdict(dict)
    for row in oof_rows:
        key = (row["matrix"], row["method"])
        ligand_id = row["ligand_id"]
        if ligand_id in groups[key]:
            raise ValueError(f"duplicate OOF score: {key} {ligand_id}")
        if manifest_by_id[ligand_id]["label"] != row["label"]:
            raise ValueError(f"OOF label differs: {ligand_id}")
        groups[key][ligand_id] = {
            "ligand_id": ligand_id,
            "label": row["label"],
            "score": float(row["normalized_ensemble_score"]),
        }

    reported_metric_rows = read_csv(output_paths["method_metrics"])
    reported_metrics = {
        (row["matrix"], row["method"]): row
        for row in reported_metric_rows
    }
    recomputed_metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(
        dict
    )
    metric_differences: list[float] = []
    sklearn_ap_differences: list[float] = []
    group_id_checks: dict[str, bool] = {}
    for key, records_by_id in groups.items():
        group_name = "/".join(key)
        group_id_checks[group_name] = set(records_by_id) == development_ids
        observed = independent_metrics(list(records_by_id.values()))
        recomputed_metrics[key[0]][key[1]] = observed
        expected = reported_metrics[key]
        metric_differences.extend(
            abs(observed[metric] - float(expected[metric]))
            for metric in CORE_METRICS
        )
        sklearn_ap_differences.append(
            abs(
                observed["pr_auc_average_precision"]
                - observed["pr_auc_sklearn"]
            )
        )

    matrix_audit: dict[str, dict[str, object]] = {}
    for name, input_key in (
        ("primary", "primary_matrix"),
        ("sensitivity", "sensitivity_matrix"),
    ):
        matrix_ids = {
            row["ligand_id"] for row in read_csv(inputs[input_key])
        }
        matrix_audit[name] = {
            "ligand_count": len(matrix_ids),
            "equals_development_ids": matrix_ids == development_ids,
            "locked_test_overlap": len(matrix_ids & locked_ids),
        }

    fold_rows = read_csv(output_paths["fold_assignments"])
    scaffold_folds: dict[str, set[str]] = defaultdict(set)
    for row in fold_rows:
        scaffold_folds[row["scaffold_smiles"]].add(
            row["development_fold"]
        )
    scaffold_cross_fold_count = sum(
        len(folds) != 1 for folds in scaffold_folds.values()
    )

    summary = json.loads(summary_path.read_text(encoding="ascii"))
    selected_family = str(summary["selected_qubo_family"])
    baseline = groups[("primary", "single_best")]
    selected = groups[("primary", selected_family)]
    bootstrap = independent_paired_bootstrap(
        baseline,
        selected,
        int(cv["bootstrap_iterations"]),
        int(cv["bootstrap_seed"]),
    )
    reported_bootstrap = summary["comparison_to_single"][
        "paired_bootstrap"
    ]
    bootstrap_differences = [
        abs(
            float(bootstrap[metric][field])
            - float(reported_bootstrap[metric][field])
        )
        for metric in CORE_METRICS
        for field in ("mean_delta", "ci95_low", "ci95_high")
    ]

    outer_rows = read_csv(output_paths["outer_fold_results"])
    all_receptor_target_sizes = sorted(
        {
            int(row["target_size"])
            for row in outer_rows
            if row["method"] == "all_receptors"
        }
    )
    exact_search_records = collect_exact_search_records(summary)
    exact_search_methods = sorted(
        {str(row["method"]) for row in exact_search_records}
    )
    maximum_metric_difference = max(metric_differences, default=math.inf)
    maximum_bootstrap_difference = max(
        bootstrap_differences, default=math.inf
    )
    locked_oof_overlap = len(
        {row["ligand_id"] for row in oof_rows} & locked_ids
    )
    tolerance = 1e-12
    checks = {
        "all_oof_groups_equal_development_manifest": all(
            group_id_checks.values()
        ),
        "locked_test_absent_from_oof": locked_oof_overlap == 0,
        "locked_test_absent_from_matrices": all(
            row["locked_test_overlap"] == 0
            for row in matrix_audit.values()
        ),
        "matrices_equal_development_manifest": all(
            bool(row["equals_development_ids"])
            for row in matrix_audit.values()
        ),
        "scaffolds_do_not_cross_folds": scaffold_cross_fold_count == 0,
        "metrics_reproduce": maximum_metric_difference <= tolerance,
        "bootstrap_reproduces": maximum_bootstrap_difference <= tolerance,
        "all_receptor_metadata_uses_full_pool": (
            all_receptor_target_sizes == [len(config["receptor_ids"])]
        ),
        "summary_reports_test_locked": (
            summary["test_lock"]["scores_evaluated"] is False
        ),
        "exact_search_audited": bool(exact_search_records),
        "consensus_constraints_audited": audit_consensus_constraints(
            outer_rows,
            float(config["model"].get("consensus_min_inner_frequency", 0.0)),
            int(config["model"].get("consensus_core_size", 0)),
        ),
    }
    result = {
        "schema_version": "1.0",
        "operation": (
            "independent audit of development-only scaffold-CV outputs; "
            "no project metric or gate implementation was imported"
        ),
        "status": "ok" if all(checks.values()) else "failed",
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "audited_outputs": {
            key: {
                "path": path.as_posix(),
                "sha256": file_sha256(path),
            }
            for key, path in output_paths.items()
        },
        "checks": checks,
        "counts": {
            "oof_row_count": len(oof_rows),
            "oof_group_count": len(groups),
            "unique_ligands_per_oof_group": sorted(
                {len(rows) for rows in groups.values()}
            ),
            "development_ligand_count": len(development_ids),
            "locked_test_ligand_count": len(locked_ids),
            "locked_test_oof_overlap": locked_oof_overlap,
            "scaffold_group_count": len(scaffold_folds),
            "scaffold_cross_fold_count": scaffold_cross_fold_count,
        },
        "matrix_audit": matrix_audit,
        "recomputed_metrics": dict(recomputed_metrics),
        "maximum_recomputed_metric_abs_difference": (
            maximum_metric_difference
        ),
        "maximum_rankwise_vs_sklearn_ap_abs_difference": max(
            sklearn_ap_differences, default=math.inf
        ),
        "recomputed_paired_bootstrap": bootstrap,
        "maximum_recomputed_bootstrap_abs_difference": (
            maximum_bootstrap_difference
        ),
        "all_receptors_target_sizes": all_receptor_target_sizes,
        "exact_search": {
            "record_count": len(exact_search_records),
            "methods": exact_search_methods,
            "states_evaluated": sorted(
                {int(row["states_evaluated"]) for row in exact_search_records}
            ),
            "full_state_counts": sorted(
                {int(row["full_state_count"]) for row in exact_search_records}
            ),
        },
        "interpretation_boundary": (
            "A successful audit validates internal consistency and locked-test "
            "isolation. It does not change a rejected development gate into a "
            "passing result and does not authorize test release."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": checks,
                "maximum_recomputed_metric_abs_difference": (
                    maximum_metric_difference
                ),
                "maximum_recomputed_bootstrap_abs_difference": (
                    maximum_bootstrap_difference
                ),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
