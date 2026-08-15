"""Run Stage86 local validation, allocation preflight, or one rescue job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import scripts.prepare_stage86_nonnegative_gauge_dirac_rescue as s86
    import scripts.run_stage75_explicit_variable_k_cqm as s75
    import scripts.run_stage81_dirac_global_qubo_formulation_gate as s81
    import scripts.run_stage84_mixed_radix_dirac_iqp_gate as s84
    from scripts.experimental.quantum import run_stage85_mixed_radix_dirac_calibration as s85
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import prepare_stage86_nonnegative_gauge_dirac_rescue as s86
    import run_stage75_explicit_variable_k_cqm as s75
    import run_stage81_dirac_global_qubo_formulation_gate as s81
    import run_stage84_mixed_radix_dirac_iqp_gate as s84
    from experimental.quantum import run_stage85_mixed_radix_dirac_calibration as s85


def verified(root: Path, descriptor: dict[str, Any]) -> Path:
    path = root / descriptor["path"]
    if not path.is_file() or s84.sha256(path) != descriptor["sha256"]:
        raise ValueError(f"Stage86 external identity differs: {path}")
    if path.stat().st_size != int(descriptor["size_bytes"]):
        raise ValueError(f"Stage86 external size differs: {path}")
    return path


def context(root: Path, config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    stage84_config = s84.read_json(root / config["inputs"]["stage84_config"])
    cells = s81.canonical_cells(stage84_config, root)
    selection = config["rescue_instance"]
    cell = next(
        cell
        for cell in cells
        if str(cell["model"]["record"]["target_id"]) == selection["target_id"]
        and int(cell["model"]["record"]["outer_fold"]) == int(selection["outer_fold"])
    )
    item = result["rescue_instance"]
    mapping = s84.read_json(verified(root, item["mapping"]))
    payload = s84.read_json(verified(root, item["payload"]))
    encoding = s86.encode_cell(cell, int(mapping["k"]))
    return {"cell": cell, "mapping": mapping, "payload": payload, "encoding": encoding}


def decode(ctx: dict[str, Any], vector: list[int]) -> dict[str, Any]:
    mapping = ctx["mapping"]
    names = mapping["variable_order"]
    if len(vector) != len(names):
        raise ValueError("Stage86 device vector length differs")
    if any(value < 0 or value >= int(mapping["num_levels"][i]) for i, value in enumerate(vector)):
        raise ValueError("Stage86 device vector contains an out-of-range level")
    values = {name: int(vector[i]) for i, name in enumerate(names)}
    selected = tuple(
        i for i in range(len(mapping["receptor_ids"])) if values[f"x{i:03d}"] == 1
    )
    deficit = sum(int(mapping["integer_deficits"][i]) for i in selected)
    deficit_digits = [s84.digits(int(value)) for value in mapping["integer_deficits"]]
    threshold_digits = s84.digits(int(mapping["quality_threshold"]))
    residuals = []
    for column in range(s84.DIGIT_COUNT):
        residual = sum(deficit_digits[i][column] * values[f"x{i:03d}"] for i in range(len(mapping["receptor_ids"])))
        residual += values[f"s{column}"] - int(threshold_digits[column])
        if column > 0:
            residual += values[f"c{column}"]
        if column < s84.DIGIT_COUNT - 1:
            residual -= s84.RADIX * values[f"c{column + 1}"]
        residuals.append(int(residual))
    feasible = len(selected) == int(mapping["k"]) and deficit <= int(mapping["quality_threshold"]) and all(value == 0 for value in residuals)
    energy = s85.payload_energy(ctx["payload"], vector)
    exact = float(mapping["quantized_exact"]["normalized_energy"])
    return {
        "selected": selected,
        "selected_ids": [mapping["receptor_ids"][i] for i in selected],
        "selected_count": len(selected),
        "deficit": deficit,
        "residuals": residuals,
        "feasible": feasible,
        "energy": energy,
        "exact": feasible and energy <= exact + 1e-6,
        "below_certificate": feasible and energy < exact - 1e-6,
        "original_objective": s75.variable_energy(
            ctx["cell"]["model"], selected, float(mapping["reward_value"])
        ),
    }


def local_validate(root: Path, config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    ctx = context(root, config, result)
    mapping = ctx["mapping"]
    selected_ids = set(mapping["quantized_exact"]["selected_subset"].split("+"))
    subset = tuple(i for i, receptor_id in enumerate(mapping["receptor_ids"]) if receptor_id in selected_ids)
    sample, residuals = s86.assignment_for_subset(ctx["encoding"], ctx["cell"]["model"], subset)
    vector = [sample[i + 1] for i in range(len(mapping["variable_order"]))]
    decoded = decode(ctx, vector)
    if residuals != [0] * s84.DIGIT_COUNT or not decoded["exact"]:
        raise ValueError("Stage86 external local validation failed")
    return {
        "status": "stage86_external_local_validation_ok",
        "instance_id": mapping["instance_id"],
        "variable_count": len(vector),
        "total_levels": sum(int(value) for value in mapping["num_levels"]),
        "exact_vector_feasible": decoded["feasible"],
        "exact_energy_match": decoded["exact"],
        "qci_cloud_queries": 0,
        "qci_device_jobs": 0,
    }


def check_allocation(value: dict[str, Any], protocol: dict[str, Any]) -> None:
    if bool(protocol["require_unpaid_allocation"]) and bool(value.get("paid")):
        raise PermissionError("Stage86 refuses a paid QCI allocation")
    if int(value.get("seconds", 0)) < int(protocol["required_free_seconds_before_job"]):
        raise RuntimeError("Stage86 has insufficient free Dirac seconds")


def preflight(root: Path, config: dict[str, Any], result: dict[str, Any], output: Path) -> dict[str, Any]:
    validation = local_validate(root, config, result)
    client = s85.qci_client(result["hardware_protocol"])
    allocation = s85.allocation(client)
    check_allocation(allocation, result["hardware_protocol"])
    record = {
        "schema_version": "1.0",
        "status": "stage86_allocation_preflight_ok",
        "local_validation": validation,
        "dirac_allocation": allocation,
        "cloud_queries": 1,
        "qci_device_jobs": 0,
        "token_recorded": False,
    }
    s84.write_json(output / "preflight.json", record)
    return record


def rescue(root: Path, config: dict[str, Any], result: dict[str, Any], output: Path, authorized: bool) -> dict[str, Any]:
    if not authorized:
        raise PermissionError("Stage86 device execution requires --authorized-device-run")
    validation = local_validate(root, config, result)
    protocol = result["hardware_protocol"]
    client = s85.qci_client(protocol)
    before = s85.allocation(client)
    check_allocation(before, protocol)
    ctx = context(root, config, result)
    mapping = ctx["mapping"]
    raw = output / "raw"
    response_path = raw / f"{mapping['instance_id']}.response.json"
    if response_path.is_file():
        response = s84.read_json(response_path)
    else:
        upload = client.upload_file(file=ctx["payload"])
        s84.write_json(raw / f"{mapping['instance_id']}.upload.json", upload)
        body = client.build_job_body(
            job_type=protocol["job_type"],
            job_name=f"stage86-{mapping['instance_id']}",
            job_tags=["stage86", "nonnegative-gauge", "exact-penalty"],
            job_params={
                "device_type": "dirac-3",
                "num_samples": int(protocol["samples"]),
                "relaxation_schedule": int(protocol["relaxation_schedule"]),
                "num_levels": mapping["num_levels"],
            },
            polynomial_file_id=upload["file_id"],
        )
        s84.write_json(raw / f"{mapping['instance_id']}.job.json", body)
        response = client.process_job(job_body=body)
        s84.write_json(response_path, response)
    if response.get("status") != "COMPLETED":
        raise RuntimeError("Stage86 QCI job did not complete")
    solutions = list((response.get("results") or {}).get("solutions") or [])
    counts = list((response.get("results") or {}).get("counts") or [1] * len(solutions))
    rows = []
    for index, (solution, count) in enumerate(zip(solutions, counts)):
        decoded = decode(ctx, [int(value) for value in solution])
        rows.append(
            {
                "solution_index": index,
                "num_occurrences": int(count),
                "selected_count": decoded["selected_count"],
                "deficit": decoded["deficit"],
                "residuals": "+".join(str(value) for value in decoded["residuals"]),
                "feasible": decoded["feasible"],
                "exact_optimum": decoded["exact"],
                "below_certificate": decoded["below_certificate"],
                "normalized_energy": decoded["energy"],
                "original_objective": decoded["original_objective"],
                "selected_subset": "+".join(decoded["selected_ids"]),
            }
        )
    total = sum(int(row["num_occurrences"]) for row in rows)
    weighted = lambda field: sum(int(row["num_occurrences"]) for row in rows if bool(row[field]))
    usage = float(((response.get("job_info") or {}).get("job_result") or {}).get("device_usage_s", 0.0))
    if usage > float(protocol["maximum_recorded_device_usage_seconds"]):
        raise RuntimeError("Stage86 exceeded the frozen device-usage limit")
    s85.write_csv(output / "rescue_samples.csv", rows)
    after = s85.allocation(client)
    record = {
        "schema_version": "1.0",
        "status": "stage86_dirac_rescue_complete",
        "local_validation": validation,
        "allocation_before": before,
        "allocation_after": after,
        "sample_count": total,
        "feasible_sample_count": weighted("feasible"),
        "exact_optimum_sample_count": weighted("exact_optimum"),
        "below_certificate_count": weighted("below_certificate"),
        "device_usage_seconds": usage,
        "primary_endpoint_passed": weighted("exact_optimum") >= 1 and weighted("below_certificate") == 0,
        "qci_device_jobs": 1,
        "full_production_authorized": False,
        "quantum_advantage_claim_authorized": False,
        "token_recorded": False,
    }
    s84.write_json(output / "rescue.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("local-validate", "preflight", "rescue"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/stage86_nonnegative_gauge_dirac_rescue.json")
    parser.add_argument("--prepared-result", default="data/stage86_nonnegative_gauge_dirac_rescue_prepared.json")
    parser.add_argument("--output-root", default="external_results/stage86_nonnegative_gauge_dirac_rescue")
    parser.add_argument("--authorized-device-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = s84.read_json(root / args.config)
    result = s84.read_json(root / args.prepared_result)
    output = root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "local-validate":
        record = local_validate(root, config, result)
        s84.write_json(output / "local_validation.json", record)
    elif args.mode == "preflight":
        record = preflight(root, config, result, output)
    else:
        record = rescue(root, config, result, output, args.authorized_device_run)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
