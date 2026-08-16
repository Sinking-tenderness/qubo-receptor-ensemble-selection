"""Freeze and validate the Stage79 QCI Dirac-3 local move-QUBO PoC."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import dimod
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()




def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any], label: str) -> Path:
    path = root / str(value["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Stage79 missing {label}: {path}")
    if sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage79 {label} hash differs: {path}")
    if path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage79 {label} size differs: {path}")
    return path


def load_moves(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_feasible(moves: list[dict[str, str]], variables: list[str], sample: dict[str, int]) -> bool:
    selected = [index for index, name in enumerate(variables) if int(sample[name])]
    removed = [moves[index]["removed_receptor_id"] for index in selected]
    added = [moves[index]["added_receptor_id"] for index in selected]
    deficit_delta = sum(int(moves[index]["deficit_delta"]) for index in selected)
    return (
        len(removed) == len(set(removed))
        and len(added) == len(set(added))
        and deficit_delta <= 0
    )


def exact_bqm_solution(bqm: dimod.BinaryQuadraticModel) -> dict[str, Any]:
    names = [str(name) for name in bqm.variables]
    index = {name: position for position, name in enumerate(names)}
    interactions = [
        (index[str(left)], index[str(right)], float(value))
        for (left, right), value in bqm.quadratic.items()
    ]
    variable_count = len(names)
    product_count = len(interactions)
    total_count = variable_count + product_count
    objective = np.zeros(total_count, dtype=float)
    for name, value in bqm.linear.items():
        objective[index[str(name)]] = float(value)
    for product_index, (_, _, value) in enumerate(interactions):
        objective[variable_count + product_index] = value

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: tuple[tuple[int, float], ...], lb: float, ub: float) -> None:
        row = len(lower)
        for column, coefficient in coefficients:
            row_indices.append(row)
            column_indices.append(column)
            values.append(float(coefficient))
        lower.append(float(lb))
        upper.append(float(ub))

    for product_index, (left, right, _) in enumerate(interactions):
        product = variable_count + product_index
        add(((product, 1.0), (left, -1.0)), -np.inf, 0.0)
        add(((product, 1.0), (right, -1.0)), -np.inf, 0.0)
        add(((left, 1.0), (right, 1.0), (product, -1.0)), -np.inf, 1.0)

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(lower), total_count),
    ).tocsr()
    result = milp(
        objective,
        integrality=np.ones(total_count, dtype=int),
        bounds=Bounds(np.zeros(total_count), np.ones(total_count)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"presolve": True, "time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or int(result.status) != 0 or result.x is None:
        raise ValueError(f"Stage79 quantized exact MILP failed: {result.message}")
    sample = {
        name: int(round(float(result.x[position])))
        for position, name in enumerate(names)
    }
    return {
        "sample": sample,
        "energy": float(bqm.energy(sample)),
        "selected_variables": [name for name in names if sample[name]],
        "mip_gap": float(getattr(result, "mip_gap", 0.0)),
        "mip_node_count": int(getattr(result, "mip_node_count", 0)),
    }


def normalized_float32_bqm(
    source: dimod.BinaryQuadraticModel,
) -> tuple[dimod.BinaryQuadraticModel, float]:
    coefficients = [abs(float(value)) for value in source.linear.values()]
    coefficients.extend(abs(float(value)) for value in source.quadratic.values())
    scale = max(coefficients)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Stage79 requires a finite nonzero QUBO scale")
    linear = {
        str(name): float(np.float32(float(value) / scale))
        for name, value in source.linear.items()
    }
    quadratic = {
        (str(left), str(right)): float(np.float32(float(value) / scale))
        for (left, right), value in source.quadratic.items()
    }
    return (
        dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY),
        scale,
    )


def qci_payload(
    instance_id: str,
    normalized: dimod.BinaryQuadraticModel,
    variables: list[str],
) -> tuple[dict[str, Any], list[str]]:
    index = {name: position + 1 for position, name in enumerate(variables)}
    terms: list[dict[str, Any]] = []
    for name in variables:
        value = float(normalized.get_linear(name))
        if value != 0.0:
            terms.append({"idx": [0, index[name]], "val": value})
    interactions = sorted(
        (
            min(index[str(left)], index[str(right)]),
            max(index[str(left)], index[str(right)]),
            float(value),
        )
        for (left, right), value in normalized.quadratic.items()
        if float(value) != 0.0
    )
    for left, right, value in interactions:
        terms.append({"idx": [left, right], "val": value})
    payload = {
        "file_name": f"stage79_{instance_id}",
        "file_config": {
            "polynomial": {
                "num_variables": len(variables),
                "min_degree": 1,
                "max_degree": 2,
                "data": terms,
            }
        },
    }
    return payload, variables


def polynomial_energy(payload: dict[str, Any], sample: list[int]) -> float:
    energy = 0.0
    for term in payload["file_config"]["polynomial"]["data"]:
        product = 1
        for index in term["idx"]:
            if int(index) > 0:
                product *= int(sample[int(index) - 1])
        energy += float(term["val"]) * product
    return energy


def translate_instance(
    root: Path,
    output_directory: Path,
    stage78_item: dict[str, Any],
) -> dict[str, Any]:
    metadata_path = verified(root, stage78_item["metadata"], "Stage78 metadata")
    metadata = read_json(metadata_path)
    bqm_path = verified(root, metadata["bqm"], "Stage78 BQM")
    moves_path = verified(root, metadata["moves"], "Stage78 moves")
    source = dimod.BinaryQuadraticModel.from_serializable(read_json(bqm_path))
    moves = load_moves(moves_path)
    variables = [str(name) for name in source.variables]
    if source.vartype is not dimod.BINARY:
        raise ValueError(f"Stage79 requires binary BQM: {metadata['instance_id']}")
    if len(moves) != len(variables):
        raise ValueError(f"Stage79 move count differs: {metadata['instance_id']}")
    zero = {name: 0 for name in variables}
    if not math.isclose(
        float(source.energy(zero)), float(metadata["warm_energy"]), abs_tol=1e-8
    ):
        raise ValueError(f"Stage79 warm state is not all zero: {metadata['instance_id']}")

    normalized, scale = normalized_float32_bqm(source)
    payload, payload_variables = qci_payload(
        metadata["instance_id"], normalized, variables
    )
    if payload_variables != variables:
        raise ValueError(f"Stage79 variable order changed: {metadata['instance_id']}")
    quantized_original = dimod.BinaryQuadraticModel(
        {name: float(normalized.get_linear(name)) * scale for name in variables},
        {
            (str(left), str(right)): float(value) * scale
            for (left, right), value in normalized.quadratic.items()
        },
        float(source.offset),
        dimod.BINARY,
    )
    quantized_exact = exact_bqm_solution(quantized_original)
    if not sample_feasible(moves, variables, quantized_exact["sample"]):
        raise ValueError(f"Stage79 quantized optimum is infeasible: {metadata['instance_id']}")
    original_energy_at_quantized = float(source.energy(quantized_exact["sample"]))
    warm = float(metadata["warm_energy"])
    role = str(metadata["role"])
    if role in {"confirmation_positive", "calibration_diagnostic"}:
        if original_energy_at_quantized >= warm - TOLERANCE:
            raise ValueError(
                f"Stage79 float32 translation erased the improvement: {metadata['instance_id']}"
            )
    elif role == "confirmation_negative":
        if original_energy_at_quantized < warm - TOLERANCE:
            raise ValueError(
                f"Stage79 negative became improving after translation: {metadata['instance_id']}"
            )
    else:
        raise ValueError(f"Stage79 unrecognized role: {role}")

    rng = np.random.default_rng(20267900 + int(metadata["source_subproblem_index"]))
    maximum_error = 0.0
    validation_samples = [zero, quantized_exact["sample"]]
    validation_samples.extend(
        {
            name: int(value)
            for name, value in zip(variables, rng.integers(0, 2, len(variables)))
        }
        for _ in range(128)
    )
    for sample in validation_samples:
        vector = [int(sample[name]) for name in variables]
        encoded = polynomial_energy(payload, vector)
        expected = float(normalized.energy(sample))
        maximum_error = max(maximum_error, abs(encoded - expected))
    if maximum_error > 1e-6:
        raise ValueError(f"Stage79 polynomial energy mismatch: {metadata['instance_id']}")

    payload_path = output_directory / f"{metadata['instance_id']}.qci-polynomial.json"
    mapping_path = output_directory / f"{metadata['instance_id']}.qci-mapping.json"
    write_json(payload_path, payload)
    mapping = {
        "schema_version": "1.0",
        "instance_id": metadata["instance_id"],
        "role": role,
        "source_bqm": metadata["bqm"],
        "source_moves": metadata["moves"],
        "source_metadata": descriptor(root, metadata_path),
        "variable_order": variables,
        "warm_bit_vector": [0] * len(variables),
        "warm_state_interpretation": "all-zero move vector; no receptor swap is applied",
        "source_offset_restored_after_sampling": float(source.offset),
        "coefficient_scale": scale,
        "submission_precision": "float32",
        "qci_job_type": "sample-hamiltonian-integer",
        "qci_device_type": "dirac-3",
        "num_levels": [2] * len(variables),
        "polynomial_term_count": len(payload["file_config"]["polynomial"]["data"]),
        "maximum_random_energy_identity_error": maximum_error,
        "quantized_exact": {
            "selected_variables": quantized_exact["selected_variables"],
            "quantized_original_energy": quantized_exact["energy"],
            "original_float64_energy": original_energy_at_quantized,
            "improvement_from_warm": original_energy_at_quantized - warm,
            "mip_gap": quantized_exact["mip_gap"],
            "mip_node_count": quantized_exact["mip_node_count"],
        },
    }
    write_json(mapping_path, mapping)
    return {
        "instance_id": metadata["instance_id"],
        "role": role,
        "target_id": metadata["target_id"],
        "logical_variable_count": len(variables),
        "interaction_count": source.num_interactions,
        "polynomial_term_count": mapping["polynomial_term_count"],
        "warm_energy": warm,
        "source_exact_energy": float(metadata["exact_reference"]["objective_energy"]),
        "source_exact_improvement": float(
            metadata["exact_reference"]["improvement_from_warm"]
        ),
        "coefficient_scale": scale,
        "quantized_exact_original_energy": original_energy_at_quantized,
        "quantized_exact_improvement": original_energy_at_quantized - warm,
        "qci_polynomial": descriptor(root, payload_path),
        "qci_mapping": descriptor(root, mapping_path),
        "source_bqm": metadata["bqm"],
        "source_moves": metadata["moves"],
        "source_metadata": descriptor(root, metadata_path),
    }


def report_text(result: dict[str, Any]) -> str:
    summary = result["instance_summary"]
    protocol = result["hardware_protocol"]
    return f"""# Stage79 QCI Dirac-3 Local Move-QUBO PoC Freeze

## Frozen panel

- Instances: `{summary['instance_count']}`.
- Confirmation positives: `{summary['confirmation_positive_count']}`.
- Confirmation negatives: `{summary['confirmation_negative_count']}`.
- Calibration diagnostics: `{summary['calibration_diagnostic_count']}`.
- Maximum variables: `{summary['maximum_variable_count']}`.
- Maximum quadratic interactions: `{summary['maximum_interaction_count']}`.

The Stage78 variables already describe local receptor swaps around a frozen
classical solution. The warm solution is the all-zero move vector, so an XOR
warm-start transformation would be the identity. Stage79 therefore submits the
same local search landscape to Dirac-3 as a degree-two integer polynomial with
two levels per variable.

## Translation

The unsupported constant offset is omitted from the QCI payload. Every other
coefficient is divided by its instance maximum absolute value and rounded to
float32. All returned bit vectors will be classified by the original Stage78
float64 BQM, not by the device-reported energy. Quantized exact MILP checks
preserved the expected positive/negative role for all instances.

## External stop

Local preparation made zero QCI queries and zero device submissions. The next
step is an allocation-only preflight. It requires a QCI token but consumes no
planned device sample. Calibration and confirmation require a separate double
acknowledgement.

The frozen plan uses `{protocol['calibration']['planned_job_count']}` calibration
jobs and `{protocol['confirmation']['planned_job_count']}` confirmation jobs,
for `{protocol['planned_total_sample_count']}` planned Dirac-3 samples. The
protocol refuses a paid allocation and caps recorded device use at
`{protocol['maximum_recorded_device_usage_seconds']}` seconds.

## Claim boundary

This is a cross-hardware physical optimization proof of concept. It does not
authorize a quantum-advantage, scaling, biological-generalization,
drug-discovery, or end-to-end speedup claim.
"""


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    for name, implementation in config["implementation"].items():
        verified(root, implementation, f"implementation {name}")
    stage78_result = read_json(
        verified(root, config["inputs"]["stage78_result"], "Stage78 result")
    )
    stage78_audit = read_json(
        verified(root, config["inputs"]["stage78_audit"], "Stage78 audit")
    )
    if stage78_result["status"] != "stage78_advantage2_reverse_annealing_poc_frozen":
        raise ValueError("Stage79 requires the frozen Stage78 panel")
    if stage78_audit["status"] != (
        "stage78_advantage2_reverse_annealing_poc_independent_audit_ok"
    ):
        raise ValueError("Stage79 requires the passing Stage78 audit")

    output_directory = root / config["outputs"]["instance_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    instances = [
        translate_instance(root, output_directory, item)
        for item in stage78_result["outputs"]["instance_files"]
    ]
    manifest_path = root / config["outputs"]["instance_manifest_csv"]
    write_csv(manifest_path, instances)
    roles = [row["role"] for row in instances]
    result = {
        "schema_version": "1.0",
        "status": "stage79_qci_dirac3_local_move_qubo_poc_frozen",
        "experiment_id": config["experiment_id"],
        "config": descriptor(root, config_path),
        "source_stage78": {
            "result": config["inputs"]["stage78_result"],
            "audit": config["inputs"]["stage78_audit"],
        },
        "translation": config["translation"],
        "hardware_protocol": config["hardware_protocol"],
        "instance_summary": {
            "instance_count": len(instances),
            "confirmation_positive_count": roles.count("confirmation_positive"),
            "confirmation_negative_count": roles.count("confirmation_negative"),
            "calibration_diagnostic_count": roles.count("calibration_diagnostic"),
            "maximum_variable_count": max(
                int(row["logical_variable_count"]) for row in instances
            ),
            "maximum_interaction_count": max(
                int(row["interaction_count"]) for row in instances
            ),
            "all_warm_states_are_zero": True,
            "all_float32_role_checks_passed": True,
        },
        "instances": instances,
        "data_boundary": {
            "historical_development_targets_read": 4,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "cloud_queries": 0,
            "quantum_hardware_jobs": 0,
        },
        "decision": {
            "ready_for_external_qci_allocation_preflight": True,
            "qci_device_execution_authorized": False,
            "quantum_advantage_claim_authorized": False,
        },
        "outputs": {
            "instance_manifest_csv": descriptor(root, manifest_path),
            "instance_files": [
                {
                    "instance_id": row["instance_id"],
                    "qci_polynomial": row["qci_polynomial"],
                    "qci_mapping": row["qci_mapping"],
                    "source_bqm": row["source_bqm"],
                    "source_moves": row["source_moves"],
                    "source_metadata": row["source_metadata"],
                }
                for row in instances
            ],
        },
    }
    result_path = root / config["outputs"]["result_json"]
    write_json(result_path, result)
    report_path = root / config["outputs"]["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text(result), encoding="ascii")
    result["outputs"]["report_md"] = descriptor(root, report_path)
    write_json(result_path, result)
    print(json.dumps(result["instance_summary"], indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage79_qci_dirac3_local_move_qubo_poc.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    run((root / args.config).resolve(), root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
