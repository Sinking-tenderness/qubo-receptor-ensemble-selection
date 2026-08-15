"""Adjudicate the partially executed Stage85 Dirac-3 calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runner(execution_root: Path) -> Any:
    import sys

    sys.path.insert(0, str(execution_root))
    from scripts.experimental.quantum import (  # type: ignore
        run_stage85_mixed_radix_dirac_calibration as runner,
    )

    return runner


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    execution_root = root / config["inputs"]["execution_root"]
    runner = load_runner(execution_root)
    stage84 = runner.s84
    stage85_config = stage84.read_json(
        execution_root / "configs/stage85_mixed_radix_dirac_calibration.json"
    )
    prepared = stage84.read_json(
        execution_root / "data/stage85_mixed_radix_dirac_calibration_prepared.json"
    )
    _, lookup = runner.local_validate(execution_root, stage85_config, prepared)
    instances = {
        str(item["mapping"]["instance_id"]): item
        for item in runner.load_instances(execution_root, prepared)
    }
    output_root = (
        execution_root
        / "external_results/stage85_mixed_radix_dirac_calibration"
    )
    summaries = []
    total_rows = 0
    for descriptor in config["inputs"]["responses"]:
        path = execution_root / descriptor["path"]
        if sha256(path) != descriptor["sha256"]:
            raise ValueError(f"Stage85a response identity differs: {path}")
        instance = instances[descriptor["instance_id"]]
        mapping = instance["mapping"]
        cell = lookup[(str(mapping["target_id"]), int(mapping["outer_fold"]))]
        rows, summary = runner.run_job(
            None, instance, cell, prepared["hardware_protocol"], output_root
        )
        total_rows += sum(int(row["num_occurrences"]) for row in rows)
        count_distribution = Counter()
        receptor_constraints_ok = 0
        column_residuals_zero = 0
        for row in rows:
            occurrences = int(row["num_occurrences"])
            count_distribution[int(row["selected_count"])] += occurrences
            deficit_ok = int(row["deficit"]) <= int(mapping["quality_threshold"])
            cardinality_ok = int(row["selected_count"]) == int(mapping["k"])
            receptor_constraints_ok += occurrences * int(
                deficit_ok and cardinality_ok
            )
            residuals = [int(value) for value in row["constraint_residuals"].split("+")]
            column_residuals_zero += occurrences * int(all(value == 0 for value in residuals))
        summary["target_k"] = int(mapping["k"])
        summary["receptor_constraints_ok_count"] = receptor_constraints_ok
        summary["column_residuals_zero_count"] = column_residuals_zero
        summary["selected_count_distribution"] = {
            str(key): value for key, value in sorted(count_distribution.items())
        }
        summaries.append(summary)

    log_descriptor = config["inputs"]["execution_log"]
    log_path = execution_root / log_descriptor["path"]
    if sha256(log_path) != log_descriptor["sha256"]:
        raise ValueError("Stage85a execution-log identity differs")
    log_text = log_path.read_text(encoding="utf-16", errors="ignore")
    if "free-tier device limit '100'" not in log_text:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    expected = config["expected_execution"]
    recorded_usage = sum(float(item["device_usage_seconds"]) for item in summaries)
    checks = {
        "completed_job_count_matches": len(summaries)
        == int(expected["completed_device_jobs"]),
        "sample_count_matches": total_rows
        == len(summaries) * int(expected["samples_per_completed_job"]),
        "recorded_usage_matches": recorded_usage
        == float(expected["recorded_device_usage_seconds"]),
        "free_tier_rejection_recorded": "greater than the free-tier device limit '100'"
        in log_text,
        "no_feasible_sample": all(
            int(item["feasible_sample_count"]) == 0 for item in summaries
        ),
        "no_exact_optimum_sample": all(
            int(item["exact_optimum_sample_count"]) == 0 for item in summaries
        ),
        "no_below_certificate_sample": all(
            int(item["below_certified_optimum_count"]) == 0 for item in summaries
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Stage85a adjudication checks failed: {checks}")
    result = {
        "schema_version": "1.0",
        "status": "stage85_physical_calibration_failed_stop_hardware",
        "completed_device_jobs": len(summaries),
        "completed_device_samples": total_rows,
        "recorded_device_usage_seconds": recorded_usage,
        "summaries": summaries,
        "rejected_job": {
            "instance_id": expected["rejected_instance_id"],
            "variable_count": int(expected["rejected_variable_count"]),
            "free_tier_variable_limit": int(expected["free_tier_variable_limit"]),
            "device_job_executed": False,
        },
        "checks": checks,
        "primary_endpoint_passed": False,
        "additional_stage85_device_jobs_authorized": 0,
        "next_action": config["decision"]["failure_action"],
        "interpretation": (
            "The two physical jobs completed normally, but every returned sample "
            "violated the full unconstrained encoding. This is an encoding-fidelity "
            "failure, not evidence about biological efficacy or quantum advantage."
        ),
    }
    result_path = root / config["outputs"]["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    report_path = root / config["outputs"]["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Stage85a Dirac-3 failure adjudication",
                "",
                "Two frozen Dirac-3 jobs completed and used 71 device seconds. "
                "All 50 returned samples were infeasible and none recovered an exact optimum.",
                "",
                "| Instance | Samples | Feasible | Exact optimum | Device seconds |",
                "|---|---:|---:|---:|---:|",
                *[
                    f"| {item['instance_id']} | {item['sample_count']} | "
                    f"{item['feasible_sample_count']} | {item['exact_optimum_sample_count']} | "
                    f"{item['device_usage_seconds']:.0f} |"
                    for item in summaries
                ],
                "",
                "The third 103-variable PPARG job was rejected before execution because "
                "the free tier permits at most 100 variables for a quadratic polynomial.",
                "",
                "The primary endpoint failed. No replacement Stage85 job is authorized. "
                "The penalty encoding must be repaired and globally certified first.",
                "",
            ]
        ),
        encoding="ascii",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage85a_qci_dirac3_failure_adjudication.json"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run((root / args.config).resolve(), root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
