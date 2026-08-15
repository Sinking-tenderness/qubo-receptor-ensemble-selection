"""Validate, preflight, and run the externally authorized Stage79 Dirac-3 PoC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import dimod


TOLERANCE = 1e-10
DEVICE_ACKNOWLEDGEMENT = "I_ACCEPT_STAGE79_QCI_DEVICE_USAGE"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty Stage79 CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified(root: Path, value: dict[str, Any], label: str) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage79 {label} identity differs: {path}")
    if path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage79 {label} size differs: {path}")
    return path


def load_instances(root: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for item in result["outputs"]["instance_files"]:
        mapping = read_json(verified(root, item["qci_mapping"], "QCI mapping"))
        payload = read_json(verified(root, item["qci_polynomial"], "QCI polynomial"))
        metadata = read_json(
            verified(root, item["source_metadata"], "source metadata")
        )
        bqm = dimod.BinaryQuadraticModel.from_serializable(
            read_json(verified(root, item["source_bqm"], "source BQM"))
        )
        moves_path = verified(root, item["source_moves"], "source moves")
        with moves_path.open("r", encoding="utf-8", newline="") as handle:
            moves = list(csv.DictReader(handle))
        variables = [str(name) for name in bqm.variables]
        if mapping["variable_order"] != variables or len(moves) != len(variables):
            raise ValueError(f"Stage79 mapping differs: {mapping['instance_id']}")
        instances.append(
            {
                "mapping": mapping,
                "payload": payload,
                "metadata": metadata,
                "bqm": bqm,
                "moves": moves,
            }
        )
    return instances


def polynomial_energy(payload: dict[str, Any], sample: list[int]) -> float:
    energy = 0.0
    for term in payload["file_config"]["polynomial"]["data"]:
        product = 1
        for index in term["idx"]:
            if int(index) > 0:
                product *= int(sample[int(index) - 1])
        energy += float(term["val"]) * product
    return energy


def decode(instance: dict[str, Any], vector: list[int]) -> dict[str, Any]:
    variables = instance["mapping"]["variable_order"]
    if len(vector) != len(variables) or any(int(value) not in (0, 1) for value in vector):
        raise ValueError(f"Stage79 invalid binary solution for {instance['mapping']['instance_id']}")
    sample = {name: int(value) for name, value in zip(variables, vector)}
    selected = [index for index, value in enumerate(vector) if int(value)]
    removed = [instance["moves"][index]["removed_receptor_id"] for index in selected]
    added = [instance["moves"][index]["added_receptor_id"] for index in selected]
    conflict_free = len(removed) == len(set(removed)) and len(added) == len(set(added))
    deficit_delta = sum(
        int(instance["moves"][index]["deficit_delta"]) for index in selected
    )
    feasible = conflict_free and deficit_delta <= 0
    energy = float(instance["bqm"].energy(sample))
    warm = float(instance["metadata"]["warm_energy"])
    exact = float(instance["metadata"]["exact_reference"]["objective_energy"])
    return {
        "sample": sample,
        "selected_move_count": len(selected),
        "selected_variables": [variables[index] for index in selected],
        "conflict_free": conflict_free,
        "deficit_delta": deficit_delta,
        "feasible": feasible,
        "float64_energy": energy,
        "guarded_float64_energy": min(warm, energy) if feasible else warm,
        "strict_improvement": feasible and energy < warm - TOLERANCE,
        "exact_optimum": feasible and math.isclose(energy, exact, abs_tol=1e-8),
        "below_certified_optimum": energy < exact - 1e-8,
        "encoded_float32_energy": polynomial_energy(instance["payload"], vector),
    }


def local_validate(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for instance in load_instances(root, result):
        variables = instance["mapping"]["variable_order"]
        zero = [0] * len(variables)
        exact_variables = set(
            instance["metadata"]["exact_reference"]["selected_move_variables"]
        )
        exact = [int(name in exact_variables) for name in variables]
        warm_decoded = decode(instance, zero)
        exact_decoded = decode(instance, exact)
        if not math.isclose(
            warm_decoded["float64_energy"],
            float(instance["metadata"]["warm_energy"]),
            abs_tol=1e-8,
        ):
            raise ValueError(f"Stage79 warm identity failed: {instance['mapping']['instance_id']}")
        if not exact_decoded["exact_optimum"]:
            raise ValueError(f"Stage79 exact identity failed: {instance['mapping']['instance_id']}")
        rows.append(
            {
                "instance_id": instance["mapping"]["instance_id"],
                "role": instance["mapping"]["role"],
                "variable_count": len(variables),
                "warm_is_all_zero": True,
                "exact_certificate": True,
            }
        )
    return {
        "status": "stage79_local_execution_bundle_valid",
        "instance_count": len(rows),
        "instances": rows,
        "cloud_queries": 0,
        "qci_device_jobs": 0,
        "qci_device_samples": 0,
    }


def device_authorized(flag: bool) -> None:
    if not flag or os.environ.get("STAGE79_QCI_ACK") != DEVICE_ACKNOWLEDGEMENT:
        raise PermissionError(
            "Stage79 device execution requires --authorize-qci-device and "
            f"STAGE79_QCI_ACK={DEVICE_ACKNOWLEDGEMENT}"
        )


def qci_client(protocol: dict[str, Any]) -> Any:
    token = os.environ.get("QCI_TOKEN")
    if not token:
        raise PermissionError("Set QCI_TOKEN in the shell; never write it into a file")
    from qci_client import QciClient

    return QciClient(url=str(protocol["api_url"]), api_token=token)


def dirac_allocation(client: Any) -> dict[str, Any]:
    response = client.get_allocations()
    try:
        allocation = dict(response["allocations"]["dirac"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Stage79 could not parse Dirac allocation: {response}") from error
    return {"allocation": allocation, "response": response}


def check_free_allocation(
    allocation: dict[str, Any], protocol: dict[str, Any], minimum_seconds: int
) -> None:
    if bool(protocol["require_unpaid_allocation"]) and bool(allocation.get("paid")):
        raise PermissionError("Stage79 refuses a paid QCI allocation")
    if bool(allocation.get("metered", True)):
        remaining = int(allocation.get("seconds", 0))
        if remaining < int(minimum_seconds):
            raise RuntimeError(
                f"Stage79 needs at least {minimum_seconds} free Dirac seconds; found {remaining}"
            )


def preflight(root: Path, result: dict[str, Any], output_root: Path) -> dict[str, Any]:
    validation = local_validate(root, result)
    protocol = result["hardware_protocol"]
    client = qci_client(protocol)
    allocation_record = dirac_allocation(client)
    check_free_allocation(
        allocation_record["allocation"],
        protocol,
        int(protocol["required_initial_free_allocation_seconds"]),
    )
    record = {
        "schema_version": "1.0",
        "status": "stage79_qci_allocation_preflight_ok",
        "api_url": protocol["api_url"],
        "local_validation": validation,
        "dirac_allocation": allocation_record["allocation"],
        "cloud_queries": 1,
        "qci_device_jobs": 0,
        "qci_device_samples": 0,
        "token_recorded": False,
        "next_step": "Return this preflight for review before calibration.",
    }
    write_json(output_root / "preflight.json", record)
    return record


def response_device_usage(response: dict[str, Any]) -> float:
    try:
        return float(response["job_info"]["job_result"].get("device_usage_s", 0.0))
    except (KeyError, TypeError, ValueError):
        return 0.0


def evaluate_response(
    instance: dict[str, Any], response: dict[str, Any], schedule: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if str(response.get("status")) != "COMPLETED":
        raise RuntimeError(
            f"Stage79 QCI job did not complete: {instance['mapping']['instance_id']}"
        )
    results = response.get("results") or {}
    solutions = list(results.get("solutions") or [])
    counts = list(results.get("counts") or [1] * len(solutions))
    reported_energies = list(results.get("energies") or [None] * len(solutions))
    if not solutions or len(counts) != len(solutions):
        raise ValueError("Stage79 QCI response contains no usable solutions")
    if len(reported_energies) != len(solutions):
        reported_energies = [None] * len(solutions)

    rows: list[dict[str, Any]] = []
    for solution_index, (solution, count, reported) in enumerate(
        zip(solutions, counts, reported_energies)
    ):
        vector = [int(value) for value in solution]
        decoded = decode(instance, vector)
        rows.append(
            {
                "instance_id": instance["mapping"]["instance_id"],
                "role": instance["mapping"]["role"],
                "relaxation_schedule": int(schedule),
                "solution_index": solution_index,
                "num_occurrences": int(count),
                "solution": "".join(str(value) for value in vector),
                "selected_variables": "+".join(decoded["selected_variables"]),
                "selected_move_count": decoded["selected_move_count"],
                "feasible": decoded["feasible"],
                "conflict_free": decoded["conflict_free"],
                "deficit_delta": decoded["deficit_delta"],
                "float64_energy": decoded["float64_energy"],
                "guarded_float64_energy": decoded["guarded_float64_energy"],
                "encoded_float32_energy": decoded["encoded_float32_energy"],
                "device_reported_energy": reported,
                "strict_improvement": decoded["strict_improvement"],
                "exact_optimum": decoded["exact_optimum"],
                "below_certified_optimum": decoded["below_certified_optimum"],
            }
        )
    total = sum(int(row["num_occurrences"]) for row in rows)

    def weighted_count(field: str) -> int:
        return sum(
            int(row["num_occurrences"]) for row in rows if bool(row[field])
        )

    feasible = weighted_count("feasible")
    improving = weighted_count("strict_improvement")
    exact = weighted_count("exact_optimum")
    below = weighted_count("below_certified_optimum")
    summary = {
        "instance_id": instance["mapping"]["instance_id"],
        "role": instance["mapping"]["role"],
        "relaxation_schedule": int(schedule),
        "sample_count": total,
        "distinct_solution_count": len(rows),
        "feasible_sample_count": feasible,
        "feasible_sample_fraction": feasible / total,
        "strict_improvement_sample_count": improving,
        "strict_improvement_sample_fraction": improving / total,
        "exact_optimum_sample_count": exact,
        "exact_optimum_sample_fraction": exact / total,
        "below_certified_optimum_count": below,
        "best_guarded_float64_energy": min(
            float(row["guarded_float64_energy"]) for row in rows
        ),
        "device_usage_seconds": response_device_usage(response),
        "job_id": (response.get("job_info") or {}).get("job_id"),
    }
    return rows, summary


def run_job(
    client: Any,
    instance: dict[str, Any],
    schedule: int,
    samples: int,
    output_root: Path,
    job_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_path = output_root / "raw" / f"{job_key}.response.json"
    if raw_path.is_file():
        response = read_json(raw_path)
        return evaluate_response(instance, response, schedule)

    upload = client.upload_file(file=instance["payload"])
    write_json(output_root / "raw" / f"{job_key}.upload.json", upload)
    job_body = client.build_job_body(
        job_type="sample-hamiltonian-integer",
        job_name=f"stage79-{job_key}",
        job_tags=["stage79", instance["mapping"]["role"]],
        job_params={
            "device_type": "dirac-3",
            "num_samples": int(samples),
            "relaxation_schedule": int(schedule),
            "num_levels": instance["mapping"]["num_levels"],
        },
        polynomial_file_id=upload["file_id"],
    )
    write_json(output_root / "raw" / f"{job_key}.job.json", job_body)
    response = client.process_job(job_body=job_body)
    write_json(raw_path, response)
    job_id = (response.get("job_info") or {}).get("job_id")
    if job_id:
        try:
            metrics = client.get_job_metrics(job_id=job_id)
            write_json(output_root / "raw" / f"{job_key}.metrics.json", metrics)
        except Exception as error:
            write_json(
                output_root / "raw" / f"{job_key}.metrics_error.json",
                {"error_type": type(error).__name__, "error": str(error)},
            )
    return evaluate_response(instance, response, schedule)


def check_usage_limit(
    summaries: list[dict[str, Any]], prior_usage: float, protocol: dict[str, Any]
) -> float:
    usage = prior_usage + sum(float(row["device_usage_seconds"]) for row in summaries)
    if usage > float(protocol["maximum_recorded_device_usage_seconds"]):
        raise RuntimeError(
            f"Stage79 recorded device usage exceeded its hard limit: {usage}"
        )
    return usage


def run_calibration(
    root: Path, result: dict[str, Any], output_root: Path, authorized: bool
) -> dict[str, Any]:
    device_authorized(authorized)
    protocol = result["hardware_protocol"]
    client = qci_client(protocol)
    allocation_before = dirac_allocation(client)
    check_free_allocation(
        allocation_before["allocation"],
        protocol,
        int(protocol["minimum_remaining_seconds_before_job"]),
    )
    calibration_instances = [
        item
        for item in load_instances(root, result)
        if item["mapping"]["role"] == "calibration_diagnostic"
    ]
    if len(calibration_instances) != 1:
        raise ValueError("Stage79 requires one calibration diagnostic")
    instance = calibration_instances[0]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    calibration = protocol["calibration"]
    for schedule in calibration["relaxation_schedules"]:
        check_free_allocation(
            dirac_allocation(client)["allocation"],
            protocol,
            int(protocol["minimum_remaining_seconds_before_job"]),
        )
        job_rows, summary = run_job(
            client,
            instance,
            int(schedule),
            int(calibration["samples_per_schedule"]),
            output_root,
            f"calibration-s{int(schedule)}",
        )
        rows.extend(job_rows)
        summaries.append(summary)
        check_usage_limit(summaries, 0.0, protocol)
    selected = min(
        summaries,
        key=lambda row: (
            -float(row["strict_improvement_sample_fraction"]),
            -float(row["exact_optimum_sample_fraction"]),
            float(row["best_guarded_float64_energy"]),
            -float(row["feasible_sample_fraction"]),
            float(row["device_usage_seconds"]),
            int(row["relaxation_schedule"]),
        ),
    )
    record = {
        "schema_version": "1.0",
        "status": "stage79_qci_dirac3_calibration_complete",
        "instance_id": instance["mapping"]["instance_id"],
        "allocation_before": allocation_before["allocation"],
        "allocation_after": dirac_allocation(client)["allocation"],
        "summaries": summaries,
        "selected_relaxation_schedule": int(selected["relaxation_schedule"]),
        "recorded_device_usage_seconds": check_usage_limit(summaries, 0.0, protocol),
        "confirmation_outcomes_read": False,
    }
    write_csv(output_root / "calibration_samples.csv", rows)
    write_csv(output_root / "calibration_summaries.csv", summaries)
    write_json(output_root / "calibration.json", record)
    return record


def run_confirmation(
    root: Path, result: dict[str, Any], output_root: Path, authorized: bool
) -> dict[str, Any]:
    device_authorized(authorized)
    protocol = result["hardware_protocol"]
    calibration_path = output_root / "calibration.json"
    if not calibration_path.is_file():
        raise FileNotFoundError("Run and review Stage79 calibration first")
    calibration = read_json(calibration_path)
    if calibration["status"] != "stage79_qci_dirac3_calibration_complete":
        raise ValueError("Stage79 calibration is incomplete")
    schedule = int(calibration["selected_relaxation_schedule"])
    client = qci_client(protocol)
    allocation_before = dirac_allocation(client)
    check_free_allocation(
        allocation_before["allocation"],
        protocol,
        int(protocol["minimum_remaining_seconds_before_job"]),
    )
    roles = set(protocol["confirmation"]["instance_roles"])
    instances = [
        item for item in load_instances(root, result) if item["mapping"]["role"] in roles
    ]
    if len(instances) != int(protocol["confirmation"]["planned_instance_count"]):
        raise ValueError("Stage79 confirmation instance count differs")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    prior_usage = float(calibration["recorded_device_usage_seconds"])
    for instance in instances:
        check_free_allocation(
            dirac_allocation(client)["allocation"],
            protocol,
            int(protocol["minimum_remaining_seconds_before_job"]),
        )
        instance_id = instance["mapping"]["instance_id"]
        job_rows, summary = run_job(
            client,
            instance,
            schedule,
            int(protocol["confirmation"]["samples_per_instance"]),
            output_root,
            f"confirmation-{instance_id}-s{schedule}",
        )
        rows.extend(job_rows)
        summaries.append(summary)
        check_usage_limit(summaries, prior_usage, protocol)

    endpoints: list[dict[str, Any]] = []
    for summary in summaries:
        if summary["role"] == "confirmation_positive":
            passed = (
                int(summary["strict_improvement_sample_count"]) >= 1
                and int(summary["below_certified_optimum_count"]) == 0
            )
        else:
            passed = (
                int(summary["strict_improvement_sample_count"]) == 0
                and int(summary["below_certified_optimum_count"]) == 0
            )
        endpoints.append(
            {
                "instance_id": summary["instance_id"],
                "role": summary["role"],
                "strict_improvement_sample_count": summary[
                    "strict_improvement_sample_count"
                ],
                "exact_optimum_sample_count": summary["exact_optimum_sample_count"],
                "below_certified_optimum_count": summary[
                    "below_certified_optimum_count"
                ],
                "primary_endpoint_passed": passed,
            }
        )
    primary_passed = all(bool(row["primary_endpoint_passed"]) for row in endpoints)
    record = {
        "schema_version": "1.0",
        "status": (
            "stage79_qci_dirac3_physical_poc_passed"
            if primary_passed
            else "stage79_qci_dirac3_physical_poc_not_passed"
        ),
        "selected_relaxation_schedule": schedule,
        "allocation_before": allocation_before["allocation"],
        "allocation_after": dirac_allocation(client)["allocation"],
        "summaries": summaries,
        "endpoints": endpoints,
        "primary_endpoint_passed": primary_passed,
        "recorded_confirmation_device_usage_seconds": sum(
            float(row["device_usage_seconds"]) for row in summaries
        ),
        "recorded_cumulative_device_usage_seconds": check_usage_limit(
            summaries, prior_usage, protocol
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    write_csv(output_root / "confirmation_samples.csv", rows)
    write_csv(output_root / "confirmation_summaries.csv", summaries)
    write_json(output_root / "confirmation.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("data/stage79_qci_dirac3_poc_result.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("external_results/stage79_qci_dirac3_poc"),
    )
    parser.add_argument(
        "--phase",
        choices=("validate", "preflight", "calibration", "confirmation"),
        required=True,
    )
    parser.add_argument("--authorize-qci-device", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = read_json((root / args.result).resolve())
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "validate":
        record = local_validate(root, result)
        write_json(output_root / "local_validation.json", record)
    elif args.phase == "preflight":
        record = preflight(root, result, output_root)
    elif args.phase == "calibration":
        record = run_calibration(
            root, result, output_root, args.authorize_qci_device
        )
    else:
        record = run_confirmation(
            root, result, output_root, args.authorize_qci_device
        )
    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
