"""Independently audit the prepared Stage86 rescue instance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import scripts.prepare_stage86_nonnegative_gauge_dirac_rescue as s86
    import scripts.run_stage81_dirac_global_qubo_formulation_gate as s81
    import scripts.run_stage84_mixed_radix_dirac_iqp_gate as s84
except ImportError:
    import prepare_stage86_nonnegative_gauge_dirac_rescue as s86
    import run_stage81_dirac_global_qubo_formulation_gate as s81
    import run_stage84_mixed_radix_dirac_iqp_gate as s84


def verified(root: Path, descriptor: dict[str, Any]) -> Path:
    path = root / descriptor["path"]
    if not path.is_file() or s84.sha256(path) != descriptor["sha256"]:
        raise ValueError(f"Stage86 audit identity differs: {path}")
    if path.stat().st_size != int(descriptor["size_bytes"]):
        raise ValueError(f"Stage86 audit size differs: {path}")
    return path


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = s84.read_json(config_path)
    result = s84.read_json(root / config["outputs"]["result_json"])
    failure = s84.read_json(root / config["inputs"]["stage85a_failure"])
    metrics_path = root / config["outputs"]["metrics_csv"]
    with metrics_path.open("r", encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stage84_config = s84.read_json(root / config["inputs"]["stage84_config"])
    cells = s81.canonical_cells(stage84_config, root)
    selection = config["rescue_instance"]
    cell = next(
        cell
        for cell in cells
        if str(cell["model"]["record"]["target_id"]) == selection["target_id"]
        and int(cell["model"]["record"]["outer_fold"]) == int(selection["outer_fold"])
    )
    encoding = s86.encode_cell(cell, int(selection["k"]))
    item = result["rescue_instance"]
    mapping = s84.read_json(verified(root, item["mapping"]))
    payload = s84.read_json(verified(root, item["payload"]))
    selected_ids = set(mapping["quantized_exact"]["selected_subset"].split("+"))
    subset = tuple(
        index
        for index, receptor_id in enumerate(mapping["receptor_ids"])
        if receptor_id in selected_ids
    )
    sample, residuals = s86.assignment_for_subset(encoding, cell["model"], subset)
    vector = [sample[index + 1] for index in range(len(encoding["names"]))]
    payload_energy = 0.0
    for term in payload["file_config"]["polynomial"]["data"]:
        product = 1
        for index in term["idx"]:
            if int(index) > 0:
                product *= int(vector[int(index) - 1])
        payload_energy += float(term["val"]) * product
    gate = config["local_gate"]
    checks = {
        "stage85_failure_frozen": failure["status"]
        == "stage85_physical_calibration_failed_stop_hardware",
        "metric_count": len(rows) == int(gate["required_encoding_count"]),
        "global_certificates": all(
            float(row["global_penalty_margin"])
            >= float(gate["minimum_global_penalty_margin"])
            for row in rows
        ),
        "float32_retention": all(
            float(row["coefficient_retention_fraction"]) == 1.0 for row in rows
        ),
        "dynamic_range": all(
            float(row["normalized_dynamic_range"])
            <= float(gate["maximum_normalized_dynamic_range"])
            for row in rows
        ),
        "exact_vector_feasible": len(subset) == int(mapping["k"])
        and all(value == 0 for value in residuals),
        "exact_payload_energy": abs(
            payload_energy - float(mapping["quantized_exact"]["normalized_energy"])
        )
        <= 1e-6,
        "free_tier_limit": len(vector)
        <= int(gate["free_tier_quadratic_variable_limit"]),
        "no_local_hardware": int(result["decision"]["qci_device_jobs_authorized"]) == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage86 independent audit failed: {checks}")
    audit = {
        "schema_version": "1.0",
        "status": "stage86_nonnegative_gauge_independent_audit_ok",
        "checks": checks,
        "encoding_count": len(rows),
        "rescue_variable_count": len(vector),
        "allocation_only_preflight_authorized": True,
        "qci_device_jobs_authorized": 0,
    }
    output = root / config["outputs"]["audit_json"]
    s84.write_json(output, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage86_nonnegative_gauge_dirac_rescue.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run((root / args.config).resolve(), root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
