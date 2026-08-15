"""Independently audit the prepared Stage85 Dirac calibration package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import scripts.run_stage84_mixed_radix_dirac_iqp_gate as s84
except ImportError:
    import run_stage84_mixed_radix_dirac_iqp_gate as s84


def verified(root: Path, descriptor: dict[str, Any]) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file():
        raise ValueError(f"Stage85 audit is missing {path}")
    if s84.sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage85 audit identity differs: {path}")
    if path.stat().st_size != int(descriptor["size_bytes"]):
        raise ValueError(f"Stage85 audit size differs: {path}")
    return path


def polynomial_energy(payload: dict[str, Any], vector: list[int]) -> float:
    total = 0.0
    for term in payload["file_config"]["polynomial"]["data"]:
        product = 1
        for index in term["idx"]:
            if int(index) > 0:
                product *= int(vector[int(index) - 1])
        total += float(term["val"]) * product
    return total


def exact_vector(mapping: dict[str, Any]) -> list[int]:
    names = list(mapping["variable_order"])
    positions = {name: index for index, name in enumerate(names)}
    selected = set(str(mapping["quantized_exact"]["selected_subset"]).split("+"))
    vector = [0] * len(names)
    deficit = 0
    for index, receptor_id in enumerate(mapping["receptor_ids"]):
        value = int(receptor_id in selected)
        vector[positions[f"x{index:03d}"]] = value
        deficit += value * int(mapping["integer_deficits"][index])
    slack = int(mapping["quality_threshold"]) - deficit
    if slack < 0:
        raise ValueError("Stage85 certified vector is infeasible")
    slack_digits = s84.digits(slack)
    source_digits = [s84.digits(int(value)) for value in mapping["integer_deficits"]]
    carry = 0
    for column in range(s84.DIGIT_COUNT):
        vector[positions[f"s{column}"]] = int(slack_digits[column])
        total = sum(
            source_digits[index][column]
            for index, receptor_id in enumerate(mapping["receptor_ids"])
            if receptor_id in selected
        )
        total += int(slack_digits[column]) + int(carry)
        if column < s84.DIGIT_COUNT - 1:
            carry = total // s84.RADIX
            vector[positions[f"c{column + 1}"]] = int(carry)
    return vector


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = s84.read_json(config_path)
    result = s84.read_json(root / config["outputs"]["result_json"])
    implementation_checks = []
    for descriptor in config["implementation"].values():
        implementation_checks.append(verified(root, descriptor).is_file())
    instance_checks = []
    for item in result["instances"]:
        payload_path = verified(root, item["payload"])
        mapping_path = verified(root, item["mapping"])
        payload = s84.read_json(payload_path)
        mapping = s84.read_json(mapping_path)
        vector = exact_vector(mapping)
        energy = polynomial_energy(payload, vector)
        instance_checks.append(
            str(mapping["instance_id"]) == str(item["instance_id"])
            and len(vector) == int(item["integer_variable_count"])
            and sum(int(value) for value in mapping["num_levels"])
            == int(item["qci_total_levels"])
            and len(payload["file_config"]["polynomial"]["data"])
            == int(item["polynomial_term_count"])
            and int(mapping["quantized_exact"]["optimum_degeneracy"]) == 1
            and abs(
                energy - float(mapping["quantized_exact"]["normalized_energy"])
            )
            <= 1e-6
            and abs(float(item["quantized_optimum_original_delta"])) <= 1e-6
        )
    checks = {
        "implementation_identity": all(implementation_checks),
        "instance_identity_and_certificate": len(instance_checks) == 3
        and all(instance_checks),
        "exact_job_count": int(
            result["hardware_protocol"]["planned_device_job_count"]
        )
        == 3,
        "no_cloud_or_hardware": result["data_boundary"]["qci_cloud_queries"] == 0
        and result["data_boundary"]["quantum_hardware_jobs"] == 0,
        "production_locked": not bool(
            result["decision"]["full_qci_production_authorized"]
        )
        and int(result["decision"]["qci_device_jobs_authorized"]) == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage85 independent audit failed: {checks}")
    audit = {
        "schema_version": "1.0",
        "status": "stage85_mixed_radix_dirac_calibration_independent_audit_ok",
        "instance_count": len(instance_checks),
        "checks": checks,
        "allocation_only_preflight_authorized": True,
        "qci_device_jobs_authorized": 0,
        "qci_cloud_queries_observed": 0,
        "quantum_hardware_jobs_observed": 0,
    }
    output = root / config["outputs"]["audit_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage85_mixed_radix_dirac_calibration.json"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    audit = run((root / args.config).resolve(), root)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
