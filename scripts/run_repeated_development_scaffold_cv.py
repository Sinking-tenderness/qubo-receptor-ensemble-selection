"""Run and aggregate repeated development-only scaffold-CV gates."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .cross_validate_ensemble_mvp import paired_bootstrap_delta
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from cross_validate_ensemble_mvp import paired_bootstrap_delta


METRIC_KEYS = (
    "roc_auc",
    "pr_auc_average_precision",
    "bedroc_alpha_20",
    "EF1%",
    "EF5%",
    "EF10%",
)


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
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_protocol(path: Path) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="ascii"))
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "base_config",
        "base_config_sha256",
        "fold_seeds",
        "methods",
        "aggregate_bootstrap",
        "outputs",
        "interpretation_boundary",
    }
    if not isinstance(protocol, dict) or not required.issubset(protocol):
        raise ValueError("repeated scaffold-CV protocol is incomplete")
    seeds = protocol["fold_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or len(set(int(value) for value in seeds)) != len(seeds)
        or any(int(value) <= 0 for value in seeds)
    ):
        raise ValueError("fold_seeds must contain at least three unique values")
    methods = protocol["methods"]
    if (
        not isinstance(methods, dict)
        or methods.get("baseline") != "single_best"
        or methods.get("candidate") != "core_plus_one_qubo"
        or methods.get("reference") != "coverage_qubo"
    ):
        raise ValueError("repeated scaffold-CV methods are invalid")
    bootstrap = protocol["aggregate_bootstrap"]
    if (
        not isinstance(bootstrap, dict)
        or int(bootstrap.get("iterations", 0)) <= 0
        or int(bootstrap.get("seed", 0)) <= 0
    ):
        raise ValueError("aggregate bootstrap settings must be positive")
    outputs = protocol["outputs"]
    required_outputs = {
        "run_directory",
        "derived_config_directory",
        "repeat_metrics_csv",
        "outer_selections_csv",
        "aggregate_oof_scores_csv",
        "summary_json",
    }
    if not isinstance(outputs, dict) or not required_outputs.issubset(outputs):
        raise ValueError("repeated scaffold-CV outputs are incomplete")
    return protocol


def derive_gate_config(
    base: dict[str, object],
    seed: int,
    run_directory: Path,
    base_path: Path,
    base_sha256: str,
) -> dict[str, object]:
    derived = copy.deepcopy(base)
    if (
        derived["cross_validation"]["evaluate_locked_test"] is not False
        or derived["cross_validation"]["locked_split"] != "test"
    ):
        raise ValueError("base gate config does not keep test locked")
    if "core_plus_one_qubo" not in derived["model"]["families"]:
        raise ValueError("base gate config lacks core_plus_one_qubo")
    derived["experiment_id"] = f"{base['experiment_id']}-fold-seed-{seed}"
    derived["purpose"] = (
        f"Repeated development-only scaffold-CV replicate with fold seed {seed}."
    )
    derived["cross_validation"]["fold_seed"] = int(seed)
    derived["repeat_provenance"] = {
        "base_config": base_path.as_posix(),
        "base_config_sha256": base_sha256,
        "fold_seed": int(seed),
    }
    base_outputs = base["outputs"]
    derived["outputs"] = {
        key: (
            run_directory.as_posix()
            if key == "run_directory"
            else (run_directory / Path(str(value)).name).as_posix()
        )
        for key, value in base_outputs.items()
    }
    return derived


def summary_statistics(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty values")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "positive_count": sum(value > 0.0 for value in values),
    }


def can_resume_run(
    summary_path: Path, audit_path: Path, derived_config_path: Path
) -> bool:
    if not summary_path.is_file() or not audit_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    return (
        summary.get("config", {}).get("sha256")
        == file_sha256(derived_config_path)
        and summary.get("test_lock", {}).get("scores_evaluated") is False
        and audit.get("status") == "ok"
        and all(audit.get("checks", {}).values())
    )


def run_gate_and_audit(
    repo_root: Path,
    derived_config_path: Path,
    run_directory: Path,
    resume: bool,
) -> dict[str, object]:
    summary_path = run_directory / "summary.json"
    audit_path = run_directory / "independent_audit.json"
    status_path = run_directory / "repeat_status.json"
    derived_hash = file_sha256(derived_config_path)
    if resume and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="ascii"))
        if status.get("derived_config_sha256") == derived_hash:
            print(f"resume: recorded fold status at {run_directory}")
            return status
    if resume and can_resume_run(
        summary_path, audit_path, derived_config_path
    ):
        print(f"resume: verified fold run at {run_directory}")
        status = {
            "status": "ok",
            "stage": "complete",
            "derived_config_sha256": derived_hash,
        }
        status_path.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        return status
    run_directory.mkdir(parents=True, exist_ok=True)
    commands = (
        (
            "gate",
            [
                sys.executable,
                "scripts/run_development_scaffold_cv_gate.py",
                "--config",
                derived_config_path.as_posix(),
                "--overwrite",
            ],
        ),
        (
            "audit",
            [
                sys.executable,
                "scripts/audit_development_scaffold_cv_gate.py",
                "--config",
                derived_config_path.as_posix(),
                "--output",
                audit_path.as_posix(),
                "--overwrite",
            ],
        ),
    )
    for stage, command in commands:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        (run_directory / f"{stage}_stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (run_directory / f"{stage}_stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if completed.returncode != 0:
            status = {
                "status": "infeasible_or_failed",
                "stage": stage,
                "returncode": completed.returncode,
                "derived_config_sha256": derived_hash,
                "stdout_log": (run_directory / f"{stage}_stdout.log").as_posix(),
                "stderr_log": (run_directory / f"{stage}_stderr.log").as_posix(),
                "error_tail": "\n".join(
                    completed.stderr.strip().splitlines()[-12:]
                ),
            }
            status_path.write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            return status
    status = {
        "status": "ok",
        "stage": "complete",
        "derived_config_sha256": derived_hash,
    }
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return status


def collect_repeat_outputs(
    repeat_records: list[dict[str, object]],
    methods: dict[str, str],
    bootstrap_iterations: int,
    bootstrap_seed: int,
    minimum_bedroc_delta: float,
) -> dict[str, object]:
    method_names = [
        str(methods["baseline"]),
        str(methods["reference"]),
        str(methods["candidate"]),
    ]
    repeat_metric_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    score_groups: dict[
        tuple[str, str, str], list[tuple[str, float]]
    ] = defaultdict(list)
    per_repeat: list[dict[str, object]] = []
    outer_counts: Counter[str] = Counter()
    core_counts: Counter[str] = Counter()
    residual_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()

    for record in repeat_records:
        seed = int(record["seed"])
        summary = record["summary"]
        audit = record["audit"]
        if (
            summary["test_lock"]["scores_evaluated"] is not False
            or audit["status"] != "ok"
            or not all(audit["checks"].values())
        ):
            raise ValueError(f"repeat {seed} failed lock or audit checks")
        metrics = summary["method_oof_metrics"]
        for matrix_name in ("primary", "sensitivity"):
            for method in method_names:
                values = metrics[matrix_name][method]
                repeat_metric_rows.append(
                    {
                        "fold_seed": seed,
                        "matrix": matrix_name,
                        "method": method,
                        **{key: values[key] for key in METRIC_KEYS},
                    }
                )
        candidate_primary = metrics["primary"][methods["candidate"]]
        baseline_primary = metrics["primary"][methods["baseline"]]
        candidate_sensitivity = metrics["sensitivity"][methods["candidate"]]
        baseline_sensitivity = metrics["sensitivity"][methods["baseline"]]
        deltas = {
            "primary_bedroc": float(candidate_primary["bedroc_alpha_20"])
            - float(baseline_primary["bedroc_alpha_20"]),
            "primary_roc_auc": float(candidate_primary["roc_auc"])
            - float(baseline_primary["roc_auc"]),
            "primary_pr_auc": float(
                candidate_primary["pr_auc_average_precision"]
            )
            - float(baseline_primary["pr_auc_average_precision"]),
            "sensitivity_bedroc": float(
                candidate_sensitivity["bedroc_alpha_20"]
            )
            - float(baseline_sensitivity["bedroc_alpha_20"]),
        }
        final_subset = list(summary["final_candidate"]["subset"])
        final_counts.update(final_subset)
        per_repeat.append(
            {
                "fold_seed": seed,
                "status": summary["status"],
                "selected_qubo_family": summary["selected_qubo_family"],
                "deltas": deltas,
                "final_subset": final_subset,
                "final_core": summary["final_candidate"].get(
                    "consensus_required_receptors", []
                ),
            }
        )
        for outer in summary["outer_fold_results"]:
            if outer["method"] != methods["candidate"]:
                continue
            subset = list(outer["subset"])
            core = list(outer["consensus_required_receptors"])
            residual = [value for value in subset if value not in set(core)]
            if len(core) != 2 or len(residual) != 1:
                raise ValueError("core-plus-one outer selection is malformed")
            outer_counts.update(subset)
            core_counts.update(core)
            residual_counts.update(residual)
            outer_rows.append(
                {
                    "fold_seed": seed,
                    "outer_fold": outer["outer_fold"],
                    "core": "+".join(core),
                    "residual_receptor": residual[0],
                    "subset": "+".join(subset),
                    "primary_bedroc_alpha_20": outer[
                        "primary_outer_metrics"
                    ]["bedroc_alpha_20"],
                    "sensitivity_bedroc_alpha_20": outer[
                        "sensitivity_outer_metrics"
                    ]["bedroc_alpha_20"],
                }
            )

        for row in read_csv(Path(str(record["oof_scores_path"]))):
            if row["method"] not in method_names:
                continue
            key = (row["matrix"], row["method"], row["ligand_id"])
            score_groups[key].append(
                (row["label"], float(row["normalized_ensemble_score"]))
            )

    repeat_count = len(repeat_records)
    aggregate_records: dict[
        tuple[str, str], dict[str, dict[str, object]]
    ] = defaultdict(dict)
    aggregate_oof_rows: list[dict[str, object]] = []
    for (matrix_name, method, ligand_id), values in sorted(score_groups.items()):
        labels = {label for label, _ in values}
        if len(values) != repeat_count or len(labels) != 1:
            raise ValueError("repeated OOF records are incomplete or inconsistent")
        score = statistics.fmean(value for _, value in values)
        label = values[0][0]
        aggregate_records[(matrix_name, method)][ligand_id] = {
            "label": label,
            "score": score,
        }
        aggregate_oof_rows.append(
            {
                "matrix": matrix_name,
                "method": method,
                "ligand_id": ligand_id,
                "label": label,
                "repeat_count": repeat_count,
                "mean_normalized_ensemble_score": score,
            }
        )

    aggregate_metrics = {
        matrix_name: {
            method: ranked_metrics_with_ids(
                aggregate_records[(matrix_name, method)]
            )
            for method in method_names
        }
        for matrix_name in ("primary", "sensitivity")
    }
    baseline_records = aggregate_records[("primary", methods["baseline"])]
    candidate_records = aggregate_records[("primary", methods["candidate"])]
    bootstrap = paired_bootstrap_delta(
        baseline_records,
        candidate_records,
        bootstrap_iterations,
        bootstrap_seed,
    )
    delta_keys = (
        "primary_bedroc",
        "primary_roc_auc",
        "primary_pr_auc",
        "sensitivity_bedroc",
    )
    delta_statistics = {
        key: summary_statistics(
            [float(row["deltas"][key]) for row in per_repeat]
        )
        for key in delta_keys
    }
    primary_bedroc_values = [
        float(row["deltas"]["primary_bedroc"]) for row in per_repeat
    ]
    return {
        "repeat_metric_rows": repeat_metric_rows,
        "outer_rows": outer_rows,
        "aggregate_oof_rows": aggregate_oof_rows,
        "per_repeat": per_repeat,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_paired_bootstrap": bootstrap,
        "delta_statistics": delta_statistics,
        "primary_bedroc_threshold": {
            "minimum_delta": minimum_bedroc_delta,
            "repeat_count_meeting_threshold": sum(
                value >= minimum_bedroc_delta
                for value in primary_bedroc_values
            ),
        },
        "selection_frequencies": {
            "outer_subset": dict(outer_counts),
            "outer_core": dict(core_counts),
            "outer_residual": dict(residual_counts),
            "final_subset": dict(final_counts),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    base_path = Path(str(protocol["base_config"]))
    if not base_path.is_absolute():
        base_path = repo_root / base_path
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    base_hash = file_sha256(base_path)
    if base_hash != str(protocol["base_config_sha256"]).upper():
        raise ValueError("base config SHA-256 differs")
    base = json.loads(base_path.read_text(encoding="ascii"))

    outputs = {
        key: Path(str(value)) for key, value in protocol["outputs"].items()
    }
    summary_path = outputs["summary_json"]
    if summary_path.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(summary_path)
    outputs["run_directory"].mkdir(parents=True, exist_ok=True)
    outputs["derived_config_directory"].mkdir(parents=True, exist_ok=True)

    repeat_records: list[dict[str, object]] = []
    execution_records: list[dict[str, object]] = []
    for seed_value in protocol["fold_seeds"]:
        seed = int(seed_value)
        run_directory = outputs["run_directory"] / f"fold_seed_{seed}"
        derived_config_path = (
            outputs["derived_config_directory"] / f"fold_seed_{seed}.json"
        )
        derived = derive_gate_config(
            base,
            seed,
            run_directory,
            Path(str(protocol["base_config"])),
            base_hash,
        )
        derived_config_path.write_text(
            json.dumps(derived, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        execution = run_gate_and_audit(
            repo_root,
            derived_config_path,
            run_directory,
            args.resume,
        )
        execution_record = {
            "fold_seed": seed,
            "derived_config": {
                "path": derived_config_path.as_posix(),
                "sha256": file_sha256(derived_config_path),
            },
            **execution,
        }
        execution_records.append(execution_record)
        if execution["status"] != "ok":
            print(f"repeat {seed}: recorded as {execution['status']}")
            continue
        repeat_records.append(
            {
                "seed": seed,
                "derived_config_path": derived_config_path.as_posix(),
                "derived_config_sha256": file_sha256(derived_config_path),
                "summary_path": (run_directory / "summary.json").as_posix(),
                "summary": json.loads(
                    (run_directory / "summary.json").read_text(
                        encoding="ascii"
                    )
                ),
                "audit_path": (
                    run_directory / "independent_audit.json"
                ).as_posix(),
                "audit": json.loads(
                    (run_directory / "independent_audit.json").read_text(
                        encoding="ascii"
                    )
                ),
                "oof_scores_path": (run_directory / "oof_scores.csv").as_posix(),
            }
        )

    if not repeat_records:
        raise RuntimeError("no repeated scaffold-CV run completed successfully")

    methods = {key: str(value) for key, value in protocol["methods"].items()}
    bootstrap = protocol["aggregate_bootstrap"]
    aggregate = collect_repeat_outputs(
        repeat_records,
        methods,
        int(bootstrap["iterations"]),
        int(bootstrap["seed"]),
        float(base["acceptance"]["minimum_primary_bedroc_delta"]),
    )
    write_csv(outputs["repeat_metrics_csv"], aggregate["repeat_metric_rows"])
    write_csv(outputs["outer_selections_csv"], aggregate["outer_rows"])
    write_csv(
        outputs["aggregate_oof_scores_csv"],
        aggregate["aggregate_oof_rows"],
    )

    source_runs = [
        {
            "fold_seed": record["seed"],
            "derived_config": {
                "path": record["derived_config_path"],
                "sha256": record["derived_config_sha256"],
            },
            "summary": {
                "path": record["summary_path"],
                "sha256": file_sha256(Path(str(record["summary_path"]))),
            },
            "independent_audit": {
                "path": record["audit_path"],
                "sha256": file_sha256(Path(str(record["audit_path"]))),
            },
        }
        for record in repeat_records
    ]
    implementation_path = Path(__file__)
    result = {
        "schema_version": "1.0",
        "experiment_id": protocol["experiment_id"],
        "operation": (
            "repeated nested scaffold-CV on development ligands with per-run "
            "independent audits and averaged OOF diagnostics"
        ),
        "status": (
            "ok"
            if len(repeat_records) == len(execution_records)
            else "ok_with_infeasible_repeats"
        ),
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "implementation": {
            "path": f"scripts/{implementation_path.name}",
            "sha256": file_sha256(implementation_path),
        },
        "base_config": {
            "path": protocol["base_config"],
            "sha256": base_hash,
        },
        "fold_seeds": [int(value) for value in protocol["fold_seeds"]],
        "requested_repeat_count": len(execution_records),
        "successful_repeat_count": len(repeat_records),
        "infeasible_or_failed_repeat_count": (
            len(execution_records) - len(repeat_records)
        ),
        "execution_records": execution_records,
        "methods": methods,
        "per_repeat": aggregate["per_repeat"],
        "aggregate_metrics": aggregate["aggregate_metrics"],
        "aggregate_paired_bootstrap": aggregate[
            "aggregate_paired_bootstrap"
        ],
        "delta_statistics": aggregate["delta_statistics"],
        "primary_bedroc_threshold": aggregate[
            "primary_bedroc_threshold"
        ],
        "selection_frequencies": aggregate["selection_frequencies"],
        "source_runs": source_runs,
        "test_lock": {
            "split": "test",
            "scores_evaluated": False,
            "all_successful_repeat_audits_passed": True,
        },
        "outputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in outputs.items()
            if key
            in {
                "repeat_metrics_csv",
                "outer_selections_csv",
                "aggregate_oof_scores_csv",
            }
        },
        "interpretation_boundary": protocol["interpretation_boundary"],
    }
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "requested_repeat_count": result["requested_repeat_count"],
                "successful_repeat_count": result["successful_repeat_count"],
                "infeasible_or_failed_repeat_count": result[
                    "infeasible_or_failed_repeat_count"
                ],
                "delta_statistics": result["delta_statistics"],
                "primary_bedroc_threshold": result[
                    "primary_bedroc_threshold"
                ],
                "test_evaluated": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
