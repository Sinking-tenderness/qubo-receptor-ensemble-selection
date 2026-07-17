"""Select a receptor protocol using fail-closed repeated-CV evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .cross_validate_ensemble_mvp import paired_bootstrap_delta
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from cross_validate_ensemble_mvp import paired_bootstrap_delta


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("candidate decision table is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "candidates",
        "baseline_method",
        "criteria",
        "bootstrap",
        "fallback_protocol",
        "outputs",
        "interpretation_boundary",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("reliability selector config is incomplete")
    candidates = config["candidates"]
    if (
        not isinstance(candidates, list)
        or not candidates
        or len({str(value["candidate_id"]) for value in candidates})
        != len(candidates)
    ):
        raise ValueError("candidate IDs must be nonempty and unique")
    criteria = config["criteria"]
    required_criteria = {
        "minimum_feasibility_fraction",
        "minimum_positive_primary_bedroc_fraction",
        "minimum_mean_primary_bedroc_delta",
        "minimum_mean_primary_roc_auc_delta",
        "minimum_mean_primary_pr_auc_delta",
        "minimum_mean_sensitivity_bedroc_delta",
        "minimum_aggregate_bedroc_bootstrap_ci95_low",
    }
    if not isinstance(criteria, dict) or not required_criteria.issubset(criteria):
        raise ValueError("reliability criteria are incomplete")
    for key in (
        "minimum_feasibility_fraction",
        "minimum_positive_primary_bedroc_fraction",
    ):
        if not 0.0 <= float(criteria[key]) <= 1.0:
            raise ValueError(f"{key} must be in [0, 1]")
    bootstrap = config["bootstrap"]
    if (
        int(bootstrap.get("iterations", 0)) <= 0
        or int(bootstrap.get("seed", 0)) <= 0
    ):
        raise ValueError("bootstrap settings must be positive")
    outputs = config["outputs"]
    if not {"decision_table_csv", "summary_json"}.issubset(outputs):
        raise ValueError("reliability selector outputs are incomplete")
    return config


def mean_delta(
    rows: list[dict[str, str]],
    candidate_method: str,
    baseline_method: str,
    matrix: str,
    metric: str,
) -> tuple[list[float], list[int]]:
    lookup = {
        (int(row["fold_seed"]), row["matrix"], row["method"]): row
        for row in rows
    }
    seeds = sorted(
        {
            int(row["fold_seed"])
            for row in rows
            if row["matrix"] == matrix and row["method"] == candidate_method
        }
    )
    values = [
        float(lookup[(seed, matrix, candidate_method)][metric])
        - float(lookup[(seed, matrix, baseline_method)][metric])
        for seed in seeds
    ]
    return values, seeds


def aggregate_records(
    rows: list[dict[str, str]], method: str, matrix: str
) -> dict[str, dict[str, object]]:
    records = {
        row["ligand_id"]: {
            "label": row["label"],
            "score": float(row["mean_normalized_ensemble_score"]),
        }
        for row in rows
        if row["method"] == method and row["matrix"] == matrix
    }
    if not records:
        raise ValueError(f"aggregate OOF records are missing for {method}")
    return records


def evaluate_candidate(
    specification: dict[str, object],
    baseline_method: str,
    criteria: dict[str, object],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    summary_path = Path(str(specification["summary_path"]))
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if file_sha256(summary_path) != str(specification["summary_sha256"]).upper():
        raise ValueError(f"summary SHA-256 differs: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    if summary["test_lock"]["scores_evaluated"] is not False:
        raise ValueError("candidate summary evaluated the locked test split")
    repeat_metrics_path = Path(
        summary["outputs"]["repeat_metrics_csv"]["path"]
    )
    aggregate_oof_path = Path(
        summary["outputs"]["aggregate_oof_scores_csv"]["path"]
    )
    for key, path in (
        ("repeat_metrics_csv", repeat_metrics_path),
        ("aggregate_oof_scores_csv", aggregate_oof_path),
    ):
        if file_sha256(path) != summary["outputs"][key]["sha256"]:
            raise ValueError(f"repeated output SHA-256 differs: {key}")
    method = str(specification["method"])
    metric_rows = read_csv(repeat_metrics_path)
    primary_bedroc, seeds = mean_delta(
        metric_rows,
        method,
        baseline_method,
        "primary",
        "bedroc_alpha_20",
    )
    primary_roc, _ = mean_delta(
        metric_rows, method, baseline_method, "primary", "roc_auc"
    )
    primary_pr, _ = mean_delta(
        metric_rows,
        method,
        baseline_method,
        "primary",
        "pr_auc_average_precision",
    )
    sensitivity_bedroc, _ = mean_delta(
        metric_rows,
        method,
        baseline_method,
        "sensitivity",
        "bedroc_alpha_20",
    )
    requested = int(summary["requested_repeat_count"])
    successful = len(seeds)
    feasibility_fraction = successful / requested
    positive_fraction = sum(value > 0.0 for value in primary_bedroc) / requested
    oof_rows = read_csv(aggregate_oof_path)
    baseline_records = aggregate_records(oof_rows, baseline_method, "primary")
    candidate_records = aggregate_records(oof_rows, method, "primary")
    aggregate_metrics = {
        "baseline": ranked_metrics_with_ids(baseline_records),
        "candidate": ranked_metrics_with_ids(candidate_records),
    }
    bootstrap = paired_bootstrap_delta(
        baseline_records,
        candidate_records,
        bootstrap_iterations,
        bootstrap_seed,
    )
    statistics_by_metric = {
        "primary_bedroc": {
            "mean": statistics.fmean(primary_bedroc),
            "population_std": statistics.pstdev(primary_bedroc),
            "minimum": min(primary_bedroc),
            "maximum": max(primary_bedroc),
            "positive_count": sum(value > 0.0 for value in primary_bedroc),
        },
        "primary_roc_auc": {"mean": statistics.fmean(primary_roc)},
        "primary_pr_auc": {"mean": statistics.fmean(primary_pr)},
        "sensitivity_bedroc": {
            "mean": statistics.fmean(sensitivity_bedroc)
        },
    }
    checks = {
        "feasibility_fraction": feasibility_fraction
        >= float(criteria["minimum_feasibility_fraction"]),
        "positive_primary_bedroc_fraction": positive_fraction
        >= float(criteria["minimum_positive_primary_bedroc_fraction"]),
        "mean_primary_bedroc_delta": statistics_by_metric[
            "primary_bedroc"
        ]["mean"]
        >= float(criteria["minimum_mean_primary_bedroc_delta"]),
        "mean_primary_roc_auc_delta": statistics_by_metric[
            "primary_roc_auc"
        ]["mean"]
        >= float(criteria["minimum_mean_primary_roc_auc_delta"]),
        "mean_primary_pr_auc_delta": statistics_by_metric[
            "primary_pr_auc"
        ]["mean"]
        >= float(criteria["minimum_mean_primary_pr_auc_delta"]),
        "mean_sensitivity_bedroc_delta": statistics_by_metric[
            "sensitivity_bedroc"
        ]["mean"]
        >= float(criteria["minimum_mean_sensitivity_bedroc_delta"]),
        "aggregate_bedroc_bootstrap_ci95_low": float(
            bootstrap["bedroc_alpha_20"]["ci95_low"]
        )
        >= float(criteria["minimum_aggregate_bedroc_bootstrap_ci95_low"]),
    }
    return {
        "candidate_id": specification["candidate_id"],
        "method": method,
        "summary": {
            "path": summary_path.as_posix(),
            "sha256": file_sha256(summary_path),
        },
        "requested_repeat_count": requested,
        "successful_repeat_count": successful,
        "fold_seeds": seeds,
        "feasibility_fraction": feasibility_fraction,
        "positive_primary_bedroc_fraction": positive_fraction,
        "delta_statistics": statistics_by_metric,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_paired_bootstrap": bootstrap,
        "checks": checks,
        "passes_reliability_gate": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = {key: Path(str(value)) for key, value in config["outputs"].items()}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("reliability selector outputs already exist")

    bootstrap = config["bootstrap"]
    candidates = [
        evaluate_candidate(
            specification,
            str(config["baseline_method"]),
            config["criteria"],
            int(bootstrap["iterations"]),
            int(bootstrap["seed"]),
        )
        for specification in config["candidates"]
    ]
    passing = [row for row in candidates if row["passes_reliability_gate"]]
    if passing:
        selected = max(
            passing,
            key=lambda row: (
                float(row["delta_statistics"]["primary_bedroc"]["mean"]),
                float(row["delta_statistics"]["primary_pr_auc"]["mean"]),
                str(row["candidate_id"]),
            ),
        )
        status = "reliable_candidate_nominated_test_still_locked"
        selected_protocol = {
            "protocol_id": selected["candidate_id"],
            "method": selected["method"],
            "reason": "all preregistered repeated-CV reliability checks passed",
        }
    else:
        status = "fallback_selected_no_reliable_qubo_candidate"
        selected_protocol = {
            **config["fallback_protocol"],
            "reason": (
                "no QUBO candidate passed all repeated-CV reliability checks"
            ),
        }

    decision_rows = [
        {
            "candidate_id": row["candidate_id"],
            "method": row["method"],
            "requested_repeat_count": row["requested_repeat_count"],
            "successful_repeat_count": row["successful_repeat_count"],
            "feasibility_fraction": row["feasibility_fraction"],
            "positive_primary_bedroc_fraction": row[
                "positive_primary_bedroc_fraction"
            ],
            "mean_primary_bedroc_delta": row["delta_statistics"][
                "primary_bedroc"
            ]["mean"],
            "mean_primary_roc_auc_delta": row["delta_statistics"][
                "primary_roc_auc"
            ]["mean"],
            "mean_primary_pr_auc_delta": row["delta_statistics"][
                "primary_pr_auc"
            ]["mean"],
            "mean_sensitivity_bedroc_delta": row["delta_statistics"][
                "sensitivity_bedroc"
            ]["mean"],
            "aggregate_bedroc_bootstrap_ci95_low": row[
                "aggregate_paired_bootstrap"
            ]["bedroc_alpha_20"]["ci95_low"],
            "passes_reliability_gate": row["passes_reliability_gate"],
            "checks": json.dumps(row["checks"], sort_keys=True),
        }
        for row in candidates
    ]
    write_csv(outputs["decision_table_csv"], decision_rows)
    implementation_path = Path(__file__)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "operation": "fail-closed repeated-CV receptor protocol selection",
        "status": status,
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "implementation": {
            "path": f"scripts/{implementation_path.name}",
            "sha256": file_sha256(implementation_path),
        },
        "criteria": config["criteria"],
        "candidate_results": candidates,
        "selected_protocol": selected_protocol,
        "test_lock": {
            "split": "test",
            "scores_evaluated": False,
            "release_authorized": False,
        },
        "outputs": {
            "decision_table_csv": {
                "path": outputs["decision_table_csv"].as_posix(),
                "sha256": file_sha256(outputs["decision_table_csv"]),
            }
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    outputs["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["summary_json"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": status,
                "selected_protocol": selected_protocol,
                "candidate_pass_count": len(passing),
                "test_evaluated": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
