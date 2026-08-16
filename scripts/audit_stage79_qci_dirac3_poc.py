"""Independently audit the frozen Stage79 QCI Dirac-3 translation."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json  # noqa: F401 (deduped)
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import dimod
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()




def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any], label: str) -> Path:
    path = root / str(value["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Stage79 audit missing {label}: {path}")
    if sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage79 audit hash mismatch for {label}: {path}")
    if path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage79 audit size mismatch for {label}: {path}")
    return path


def payload_coefficients(payload: dict[str, Any]) -> dict[tuple[int, ...], float]:
    polynomial = payload["file_config"]["polynomial"]
    if int(polynomial["min_degree"]) != 1 or int(polynomial["max_degree"]) != 2:
        raise ValueError("Stage79 QCI polynomial degree differs")
    coefficients: dict[tuple[int, ...], float] = {}
    for term in polynomial["data"]:
        indices = tuple(int(value) for value in term["idx"])
        if len(indices) != 2 or list(indices) != sorted(indices):
            raise ValueError("Stage79 QCI polynomial index is invalid")
        if indices == (0, 0):
            raise ValueError("Stage79 QCI payload contains an unsupported constant")
        if indices in coefficients:
            raise ValueError("Stage79 QCI payload contains a duplicate term")
        coefficients[indices] = float(term["val"])
    return coefficients


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    result_path = root / config["outputs"]["result_json"]
    result = read_json(result_path)
    if result["status"] != "stage79_qci_dirac3_local_move_qubo_poc_frozen":
        raise ValueError("Stage79 result is not frozen")
    verified(root, config["inputs"]["stage78_result"], "Stage78 result")
    stage78_audit = read_json(
        verified(root, config["inputs"]["stage78_audit"], "Stage78 audit")
    )
    if not str(stage78_audit["status"]).endswith("independent_audit_ok"):
        raise ValueError("Stage79 source audit did not pass")

    audited = 0
    coefficient_terms = 0
    maximum_coefficient_error = 0.0
    for item in result["outputs"]["instance_files"]:
        mapping = read_json(verified(root, item["qci_mapping"], "QCI mapping"))
        payload = read_json(verified(root, item["qci_polynomial"], "QCI polynomial"))
        source = dimod.BinaryQuadraticModel.from_serializable(
            read_json(verified(root, item["source_bqm"], "source BQM"))
        )
        verified(root, item["source_moves"], "source moves")
        metadata = read_json(
            verified(root, item["source_metadata"], "source metadata")
        )
        variables = [str(name) for name in source.variables]
        if mapping["variable_order"] != variables:
            raise ValueError(f"Stage79 variable mapping differs: {mapping['instance_id']}")
        if mapping["warm_bit_vector"] != [0] * len(variables):
            raise ValueError(f"Stage79 warm vector differs: {mapping['instance_id']}")
        if not math.isclose(
            float(source.energy({name: 0 for name in variables})),
            float(metadata["warm_energy"]),
            abs_tol=1e-8,
        ):
            raise ValueError(f"Stage79 warm energy differs: {mapping['instance_id']}")

        values = [abs(float(value)) for value in source.linear.values()]
        values.extend(abs(float(value)) for value in source.quadratic.values())
        expected_scale = max(values)
        if not math.isclose(
            float(mapping["coefficient_scale"]), expected_scale, abs_tol=1e-12
        ):
            raise ValueError(f"Stage79 coefficient scale differs: {mapping['instance_id']}")
        coefficients = payload_coefficients(payload)
        if int(payload["file_config"]["polynomial"]["num_variables"]) != len(variables):
            raise ValueError(f"Stage79 QCI variable count differs: {mapping['instance_id']}")
        index = {name: position + 1 for position, name in enumerate(variables)}
        expected: dict[tuple[int, ...], float] = {}
        for name, value in source.linear.items():
            normalized = float(np.float32(float(value) / expected_scale))
            if normalized != 0.0:
                expected[(0, index[str(name)])] = normalized
        for (left, right), value in source.quadratic.items():
            normalized = float(np.float32(float(value) / expected_scale))
            if normalized != 0.0:
                pair = tuple(sorted((index[str(left)], index[str(right)])))
                expected[pair] = normalized
        if set(coefficients) != set(expected):
            raise ValueError(f"Stage79 QCI term set differs: {mapping['instance_id']}")
        for indices, value in expected.items():
            error = abs(coefficients[indices] - value)
            maximum_coefficient_error = max(maximum_coefficient_error, error)
            if error > 1e-12:
                raise ValueError(
                    f"Stage79 QCI coefficient differs: {mapping['instance_id']} {indices}"
                )

        exact_sample = {
            name: int(name in metadata["exact_reference"]["selected_move_variables"])
            for name in variables
        }
        exact_energy = float(source.energy(exact_sample))
        if not math.isclose(
            exact_energy,
            float(metadata["exact_reference"]["objective_energy"]),
            abs_tol=1e-8,
        ):
            raise ValueError(f"Stage79 exact certificate differs: {mapping['instance_id']}")
        audited += 1
        coefficient_terms += len(coefficients)

    if audited != 6:
        raise ValueError("Stage79 audit expected six frozen instances")
    if result["data_boundary"]["cloud_queries"] != 0:
        raise ValueError("Stage79 freeze unexpectedly contacted QCI")
    audit = {
        "schema_version": "1.0",
        "status": "stage79_qci_dirac3_local_move_qubo_poc_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "instances_audited": audited,
        "polynomial_terms_audited": coefficient_terms,
        "maximum_coefficient_error": maximum_coefficient_error,
        "all_warm_states_independently_verified_as_zero": True,
        "all_exact_certificates_independently_matched": True,
        "cloud_queries_observed": 0,
        "qci_device_jobs_observed": 0,
        "qci_device_execution_authorized": False,
        "quantum_advantage_claim_authorized": False,
    }
    output_path = root / config["outputs"]["audit_json"]
    write_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


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
