"""Run Stage85 allocation preflight or the three authorized Dirac-3 jobs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

try:
    import scripts.run_stage75_explicit_variable_k_cqm as s75
    import scripts.run_stage81_dirac_global_qubo_formulation_gate as s81
    import scripts.run_stage84_mixed_radix_dirac_iqp_gate as s84
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import run_stage75_explicit_variable_k_cqm as s75
    import run_stage81_dirac_global_qubo_formulation_gate as s81
    import run_stage84_mixed_radix_dirac_iqp_gate as s84


TOLERANCE = 1e-6


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Stage85 refuses to write an empty device table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified(root: Path, descriptor: dict[str, Any], label: str) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file() or s84.sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage85 {label} identity differs: {path}")
    if path.stat().st_size != int(descriptor["size_bytes"]):
        raise ValueError(f"Stage85 {label} size differs: {path}")
    return path


def payload_energy(payload: dict[str, Any], vector: list[int]) -> float:
    energy = 0.0
    for term in payload["file_config"]["polynomial"]["data"]:
        product = 1
        for index in term["idx"]:
            if int(index) > 0:
                product *= int(vector[int(index) - 1])
        energy += float(term["val"]) * product
    return energy


def load_instances(root: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for item in result["instances"]:
        mapping_path = verified(root, item["mapping"], "mapping")
        mapping = s84.read_json(mapping_path)
        payload_path = verified(root, mapping["qci_polynomial"], "polynomial")
        output.append(
            {
                "record": item,
                "mapping": mapping,
                "payload": s84.read_json(payload_path),
            }
        )
    return output


def vector_for_subset(mapping: dict[str, Any], subset_ids: set[str]) -> list[int]:
    names = list(mapping["variable_order"])
    positions = {name: index for index, name in enumerate(names)}
    receptor_ids = list(mapping["receptor_ids"])
    vector = [0] * len(names)
    deficit = 0
    for index, receptor_id in enumerate(receptor_ids):
        selected = int(receptor_id in subset_ids)
        vector[positions[f"x{index:03d}"]] = selected
        deficit += selected * int(mapping["integer_deficits"][index])
    slack = int(mapping["quality_threshold"]) - deficit
    if slack < 0:
        raise ValueError("Stage85 exact subset is quality infeasible")
    slack_digits = s84.digits(slack)
    deficit_digits = [s84.digits(int(value)) for value in mapping["integer_deficits"]]
    carry = 0
    for column in range(s84.DIGIT_COUNT):
        vector[positions[f"s{column}"]] = int(slack_digits[column])
        total = sum(
            deficit_digits[index][column]
            for index, receptor_id in enumerate(receptor_ids)
            if receptor_id in subset_ids
        )
        total += int(slack_digits[column]) + int(carry)
        if column < s84.DIGIT_COUNT - 1:
            carry = total // s84.RADIX
            vector[positions[f"c{column + 1}"]] = int(carry)
    return vector


def decode(
    instance: dict[str, Any],
    vector: list[int],
    cell: dict[str, Any],
) -> dict[str, Any]:
    mapping = instance["mapping"]
    names = list(mapping["variable_order"])
    if len(vector) != len(names):
        raise ValueError("Stage85 device vector length differs")
    if any(
        int(value) < 0 or int(value) >= int(mapping["num_levels"][index])
        for index, value in enumerate(vector)
    ):
        raise ValueError("Stage85 device vector contains an out-of-range level")
    values = {name: int(vector[index]) for index, name in enumerate(names)}
    selected_indices = tuple(
        index
        for index in range(len(mapping["receptor_ids"]))
        if values[f"x{index:03d}"] == 1
    )
    selected_ids = [mapping["receptor_ids"][index] for index in selected_indices]
    deficit = sum(int(mapping["integer_deficits"][index]) for index in selected_indices)
    threshold_digits = s84.digits(int(mapping["quality_threshold"]))
    deficit_digits = [s84.digits(int(value)) for value in mapping["integer_deficits"]]
    residuals = []
    for column in range(s84.DIGIT_COUNT):
        residual = sum(
            deficit_digits[index][column] * values[f"x{index:03d}"]
            for index in range(len(mapping["receptor_ids"]))
        )
        residual += values[f"s{column}"] - int(threshold_digits[column])
        if column > 0:
            residual += values[f"c{column}"]
        if column < s84.DIGIT_COUNT - 1:
            residual -= s84.RADIX * values[f"c{column + 1}"]
        residuals.append(int(residual))
    encoded = payload_energy(instance["payload"], vector)
    exact_energy = float(mapping["quantized_exact"]["normalized_energy"])
    feasible = (
        len(selected_indices) == int(mapping["k"])
        and deficit <= int(mapping["quality_threshold"])
        and all(value == 0 for value in residuals)
    )
    original = s75.variable_energy(
        cell["model"], selected_indices, float(mapping["reward_value"])
    )
    return {
        "selected_indices": selected_indices,
        "selected_ids": selected_ids,
        "selected_count": len(selected_indices),
        "deficit": deficit,
        "constraint_residuals": residuals,
        "feasible": feasible,
        "encoded_normalized_energy": encoded,
        "restored_energy": encoded * float(mapping["coefficient_scale"])
        + float(mapping["constant_offset_restored_after_sampling"]),
        "original_objective": original,
        "exact_quantized_optimum": feasible and encoded <= exact_energy + TOLERANCE,
        "below_certified_optimum": feasible and encoded < exact_energy - TOLERANCE,
    }


def local_validate(
    root: Path,
    config: dict[str, Any],
    result: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    cells = s81.canonical_cells(config, root)
    lookup = {
        (
            str(cell["model"]["record"]["target_id"]),
            int(cell["model"]["record"]["outer_fold"]),
        ): cell
        for cell in cells
    }
    records = []
    for instance in load_instances(root, result):
        mapping = instance["mapping"]
        subset = set(str(mapping["quantized_exact"]["selected_subset"]).split("+"))
        vector = vector_for_subset(mapping, subset)
        cell = lookup[(str(mapping["target_id"]), int(mapping["outer_fold"]))]
        decoded = decode(instance, vector, cell)
        records.append(
            {
                "instance_id": mapping["instance_id"],
                "variable_count": len(mapping["variable_order"]),
                "total_levels": sum(int(value) for value in mapping["num_levels"]),
                "exact_vector_feasible": decoded["feasible"],
                "exact_energy_match": abs(
                    float(decoded["encoded_normalized_energy"])
                    - float(mapping["quantized_exact"]["normalized_energy"])
                )
                <= TOLERANCE,
            }
        )
    if not all(
        row["exact_vector_feasible"] and row["exact_energy_match"] for row in records
    ):
        raise ValueError("Stage85 local external validation failed")
    return (
        {
            "status": "stage85_external_local_validation_ok",
            "instance_count": len(records),
            "instances": records,
            "qci_cloud_queries": 0,
            "qci_device_jobs": 0,
        },
        lookup,
    )


def qci_client(protocol: dict[str, Any]) -> Any:
    token = os.environ.get("QCI_TOKEN")
    if not token:
        raise PermissionError("Set QCI_TOKEN in the shell; never write it into a file")
    from qci_client import QciClient

    return QciClient(url=str(protocol["api_url"]), api_token=token)


def allocation(client: Any) -> dict[str, Any]:
    response = client.get_allocations()
    return dict(response["allocations"]["dirac"])


def check_allocation(value: dict[str, Any], protocol: dict[str, Any], minimum: int) -> None:
    if bool(protocol["require_unpaid_allocation"]) and bool(value.get("paid")):
        raise PermissionError("Stage85 refuses a paid QCI allocation")
    if bool(value.get("metered", True)) and int(value.get("seconds", 0)) < int(minimum):
        raise RuntimeError(
            f"Stage85 requires at least {minimum} free Dirac seconds; found {value.get('seconds')}"
        )


def preflight(
    root: Path,
    config: dict[str, Any],
    result: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    validation, _ = local_validate(root, config, result)
    protocol = result["hardware_protocol"]
    client = qci_client(protocol)
    value = allocation(client)
    check_allocation(
        value, protocol, int(protocol["required_free_seconds_before_calibration"])
    )
    record = {
        "schema_version": "1.0",
        "status": "stage85_qci_allocation_preflight_ok",
        "local_validation": validation,
        "dirac_allocation": value,
        "cloud_queries": 1,
        "qci_device_jobs": 0,
        "token_recorded": False,
        "next_step": "Return this preflight for review before using the authorized-device-run flag.",
    }
    s84.write_json(output_root / "preflight.json", record)
    return record


def run_job(
    client: Any,
    instance: dict[str, Any],
    cell: dict[str, Any],
    protocol: dict[str, Any],
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    instance_id = str(instance["mapping"]["instance_id"])
    raw_path = output_root / "raw" / f"{instance_id}.response.json"
    if raw_path.is_file():
        response = s84.read_json(raw_path)
    else:
        upload = client.upload_file(file=instance["payload"])
        s84.write_json(output_root / "raw" / f"{instance_id}.upload.json", upload)
        job_body = client.build_job_body(
            job_type="sample-hamiltonian-integer",
            job_name=f"stage85-{instance_id}",
            job_tags=["stage85", "mixed-radix", instance["mapping"]["role"]],
            job_params={
                "device_type": "dirac-3",
                "num_samples": int(protocol["samples_per_instance"]),
                "relaxation_schedule": int(protocol["relaxation_schedule"]),
                "num_levels": instance["mapping"]["num_levels"],
            },
            polynomial_file_id=upload["file_id"],
        )
        s84.write_json(output_root / "raw" / f"{instance_id}.job.json", job_body)
        response = client.process_job(job_body=job_body)
        s84.write_json(raw_path, response)
    if str(response.get("status")) != "COMPLETED":
        raise RuntimeError(f"Stage85 QCI job did not complete: {instance_id}")
    results = response.get("results") or {}
    solutions = list(results.get("solutions") or [])
    counts = list(results.get("counts") or [1] * len(solutions))
    if not solutions or len(solutions) != len(counts):
        raise ValueError(f"Stage85 response has no usable solutions: {instance_id}")
    rows = []
    for solution_index, (solution, count) in enumerate(zip(solutions, counts)):
        vector = [int(value) for value in solution]
        decoded = decode(instance, vector, cell)
        rows.append(
            {
                "instance_id": instance_id,
                "role": instance["mapping"]["role"],
                "solution_index": solution_index,
                "num_occurrences": int(count),
                "solution": "+".join(str(value) for value in vector),
                "selected_subset": "+".join(decoded["selected_ids"]),
                "selected_count": decoded["selected_count"],
                "deficit": decoded["deficit"],
                "constraint_residuals": "+".join(
                    str(value) for value in decoded["constraint_residuals"]
                ),
                "feasible": decoded["feasible"],
                "encoded_normalized_energy": decoded["encoded_normalized_energy"],
                "restored_energy": decoded["restored_energy"],
                "original_objective": decoded["original_objective"],
                "exact_quantized_optimum": decoded["exact_quantized_optimum"],
                "below_certified_optimum": decoded["below_certified_optimum"],
            }
        )
    total = sum(int(row["num_occurrences"]) for row in rows)

    def weighted(field: str) -> int:
        return sum(int(row["num_occurrences"]) for row in rows if bool(row[field]))

    usage = float(
        ((response.get("job_info") or {}).get("job_result") or {}).get(
            "device_usage_s", 0.0
        )
    )
    summary = {
        "instance_id": instance_id,
        "role": instance["mapping"]["role"],
        "sample_count": total,
        "distinct_solution_count": len(rows),
        "feasible_sample_count": weighted("feasible"),
        "feasible_sample_fraction": weighted("feasible") / total,
        "exact_optimum_sample_count": weighted("exact_quantized_optimum"),
        "exact_optimum_sample_fraction": weighted("exact_quantized_optimum") / total,
        "below_certified_optimum_count": weighted("below_certified_optimum"),
        "best_encoded_normalized_energy": min(
            float(row["encoded_normalized_energy"]) for row in rows
        ),
        "device_usage_seconds": usage,
        "job_id": (response.get("job_info") or {}).get("job_id"),
        "primary_endpoint_passed": weighted("exact_quantized_optimum") >= 1
        and weighted("below_certified_optimum") == 0,
    }
    return rows, summary


def calibration(
    root: Path,
    config: dict[str, Any],
    result: dict[str, Any],
    output_root: Path,
    authorized: bool,
) -> dict[str, Any]:
    if not authorized:
        raise PermissionError("Stage85 device execution requires --authorized-device-run")
    validation, lookup = local_validate(root, config, result)
    protocol = result["hardware_protocol"]
    client = qci_client(protocol)
    before = allocation(client)
    check_allocation(
        before, protocol, int(protocol["required_free_seconds_before_calibration"])
    )
    rows = []
    summaries = []
    for instance in load_instances(root, result):
        current = allocation(client)
        check_allocation(
            current, protocol, int(protocol["minimum_free_seconds_before_each_job"])
        )
        mapping = instance["mapping"]
        cell = lookup[(str(mapping["target_id"]), int(mapping["outer_fold"]))]
        local_rows, summary = run_job(
            client, instance, cell, protocol, output_root
        )
        rows.extend(local_rows)
        summaries.append(summary)
        if sum(float(item["device_usage_seconds"]) for item in summaries) > float(
            protocol["maximum_recorded_device_usage_seconds"]
        ):
            raise RuntimeError("Stage85 device usage exceeded the frozen hard limit")
    after = allocation(client)
    write_csv(output_root / "calibration_samples.csv", rows)
    write_csv(output_root / "calibration_summaries.csv", summaries)
    record = {
        "schema_version": "1.0",
        "status": "stage85_mixed_radix_dirac_calibration_complete",
        "local_validation": validation,
        "allocation_before": before,
        "allocation_after": after,
        "summaries": summaries,
        "recorded_device_usage_seconds": sum(
            float(item["device_usage_seconds"]) for item in summaries
        ),
        "primary_endpoint_passed": all(
            bool(item["primary_endpoint_passed"]) for item in summaries
        ),
        "cloud_queries": 2 + len(summaries),
        "qci_device_jobs": len(summaries),
        "token_recorded": False,
        "full_production_authorized": False,
        "quantum_advantage_claim_authorized": False,
    }
    s84.write_json(output_root / "calibration.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("local-validate", "preflight", "calibration")
    )
    parser.add_argument(
        "--config", default="configs/stage85_mixed_radix_dirac_calibration.json"
    )
    parser.add_argument(
        "--prepared-result",
        default="data/stage85_mixed_radix_dirac_calibration_prepared.json",
    )
    parser.add_argument(
        "--output-root",
        default="external_results/stage85_mixed_radix_dirac_calibration",
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--authorized-device-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = s84.read_json((root / args.config).resolve())
    result = s84.read_json((root / args.prepared_result).resolve())
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "local-validate":
        record, _ = local_validate(root, config, result)
        s84.write_json(output_root / "local_validation.json", record)
    elif args.mode == "preflight":
        record = preflight(root, config, result, output_root)
    else:
        record = calibration(
            root, config, result, output_root, args.authorized_device_run
        )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
