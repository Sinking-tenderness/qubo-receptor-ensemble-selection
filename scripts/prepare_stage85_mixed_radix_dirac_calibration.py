"""Prepare three exact mixed-radix Dirac-3 calibration instances."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

try:
    import scripts.run_stage75_explicit_variable_k_cqm as s75
    import scripts.run_stage81_dirac_global_qubo_formulation_gate as s81
    import scripts.run_stage84_mixed_radix_dirac_iqp_gate as s84
except ImportError:
    import run_stage75_explicit_variable_k_cqm as s75
    import run_stage81_dirac_global_qubo_formulation_gate as s81
    import run_stage84_mixed_radix_dirac_iqp_gate as s84


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": s84.sha256(path),
        "size_bytes": path.stat().st_size,
    }


def quantized_exact_feasible(
    cell: dict[str, Any], k: int, encoding: dict[str, Any]
) -> dict[str, Any]:
    model = cell["model"]
    threshold = int(encoding["threshold"])
    best_energy = math.inf
    best: list[tuple[tuple[int, ...], float]] = []
    feasible_count = 0
    total_count = 0
    for subset in itertools.combinations(range(model["count"]), int(k)):
        total_count += 1
        if s75.subset_deficit(model, subset) > threshold:
            continue
        feasible_count += 1
        sample, residuals = s84.assignment_for_subset(encoding, model, subset)
        if any(value != 0 for value in residuals):
            raise ValueError("Stage85 feasible enumeration produced a nonzero residual")
        energy = s84.polynomial_energy(encoding["normalized_terms"], sample)
        original = s75.variable_energy(model, subset, float(cell["reward"]))
        if energy < best_energy - 1e-10:
            best_energy = energy
            best = [(tuple(subset), original)]
        elif math.isclose(energy, best_energy, abs_tol=1e-10):
            best.append((tuple(subset), original))
    expected_feasible = int(cell["frontiers"][k]["fixed_k_feasible_state_count"])
    expected_total = int(cell["frontiers"][k]["fixed_k_total_state_count"])
    if feasible_count != expected_feasible or total_count != expected_total:
        raise ValueError("Stage85 exact enumeration count differs from Stage74")
    best.sort(key=lambda item: (item[1], item[0]))
    selected, original = best[0]
    restored = best_energy * float(encoding["full_scale"]) + float(
        encoding["offset"]
    )
    return {
        "total_fixed_k_state_count": total_count,
        "feasible_state_count": feasible_count,
        "quantized_optimum_degeneracy": len(best),
        "quantized_normalized_energy": best_energy,
        "quantized_restored_energy": restored,
        "selected_subset": selected,
        "selected_original_objective": original,
        "all_optimum_subsets": [item[0] for item in best],
    }


def prepare_instance(
    root: Path,
    output_directory: Path,
    cell: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    k = int(selection["k"])
    model = cell["model"]
    frontier = cell["frontiers"][k]
    if str(frontier["reference_type"]) != "exact_enumeration":
        raise ValueError("Stage85 calibration instances require an exact Stage74 oracle")
    encoding = s84.encode_cell(cell, k)
    exact = quantized_exact_feasible(cell, k, encoding)
    reference = tuple(frontier["reference_subset"])
    reference_objective = s75.variable_energy(model, reference, float(cell["reward"]))
    tolerance = float(selection["maximum_original_objective_delta"])
    if float(exact["selected_original_objective"]) > reference_objective + tolerance:
        raise ValueError("Stage85 float32 optimum is not frontier competitive")
    instance_id = str(selection["instance_id"])
    payload = dict(encoding["payload"])
    payload["file_name"] = f"stage85_{instance_id}"
    payload_path = output_directory / f"{instance_id}.qci-polynomial.json"
    mapping_path = output_directory / f"{instance_id}.qci-mapping.json"
    s84.write_json(payload_path, payload)
    mapping = {
        "schema_version": "1.0",
        "instance_id": instance_id,
        "role": str(selection["role"]),
        "target_id": str(model["record"]["target_id"]),
        "outer_fold": int(model["record"]["outer_fold"]),
        "k": k,
        "reward_quantile": 0.5,
        "reward_value": float(cell["reward"]),
        "quality_threshold": int(encoding["threshold"]),
        "receptor_ids": list(model["receptor_ids"]),
        "integer_deficits": [int(value) for value in model["deficits"]],
        "variable_order": list(encoding["names"]),
        "num_levels": [int(value) for value in encoding["levels"]],
        "qci_total_levels": sum(int(value) for value in encoding["levels"]),
        "polynomial_term_count": len(encoding["terms"]),
        "coefficient_scale": float(encoding["full_scale"]),
        "constant_offset_restored_after_sampling": float(encoding["offset"]),
        "pair_scale": float(encoding["pair_scale"]),
        "constraint_weight": float(encoding["constraint_weight"]),
        "carry_upper_bounds": [int(value) for value in encoding["carry_bounds"]],
        "source_reference": {
            "subset": s75.subset_name(model, reference),
            "original_objective": reference_objective,
            "reference_type": str(frontier["reference_type"]),
        },
        "quantized_exact": {
            "total_fixed_k_state_count": int(exact["total_fixed_k_state_count"]),
            "feasible_state_count": int(exact["feasible_state_count"]),
            "optimum_degeneracy": int(exact["quantized_optimum_degeneracy"]),
            "normalized_energy": float(exact["quantized_normalized_energy"]),
            "restored_energy": float(exact["quantized_restored_energy"]),
            "selected_subset": s75.subset_name(
                model, tuple(exact["selected_subset"])
            ),
            "selected_original_objective": float(
                exact["selected_original_objective"]
            ),
            "all_optimum_subsets": [
                s75.subset_name(model, tuple(subset))
                for subset in exact["all_optimum_subsets"]
            ],
        },
        "qci_polynomial": descriptor(root, payload_path),
        "classification_rule": "Require exact k, original integer deficit <= D_k, zero radix-column residuals, and locally recomputed normalized energy at the frozen quantized optimum tolerance.",
    }
    s84.write_json(mapping_path, mapping)
    return {
        "instance_id": instance_id,
        "role": str(selection["role"]),
        "target_id": str(model["record"]["target_id"]),
        "outer_fold": int(model["record"]["outer_fold"]),
        "k": k,
        "integer_variable_count": len(encoding["names"]),
        "qci_total_levels": sum(int(value) for value in encoding["levels"]),
        "polynomial_term_count": len(encoding["terms"]),
        "fixed_k_total_state_count": int(exact["total_fixed_k_state_count"]),
        "feasible_state_count": int(exact["feasible_state_count"]),
        "quantized_optimum_degeneracy": int(exact["quantized_optimum_degeneracy"]),
        "quantized_optimum_original_delta": float(exact["selected_original_objective"])
        - reference_objective,
        "payload": descriptor(root, payload_path),
        "mapping": descriptor(root, mapping_path),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Stage85 Mixed-radix Dirac Calibration Preparation",
        "",
        "Three exact-oracle fixed-k calibration instances were translated into",
        "native mixed-integer degree-two Dirac-3 polynomials.",
        "",
    ]
    for item in result["instances"]:
        lines.append(
            f"- `{item['instance_id']}`: {item['integer_variable_count']} variables, "
            f"{item['qci_total_levels']} levels, {item['polynomial_term_count']} terms."
        )
    lines.extend(
        [
            "",
            "No QCI query or device job was performed. Run allocation-only preflight",
            "first; the three physical jobs remain separately authorization-gated.",
            "This package tests physical solver fidelity, not efficacy or quantum advantage.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = s84.read_json(config_path)
    for name in ("stage84_result", "stage84_audit", "stage79_physical_audit"):
        s84.verified(root, config["inputs"][name], name)
    audit = s84.read_json(root / config["inputs"]["stage84_audit"]["path"])
    if audit["status"] != "stage84_mixed_radix_dirac_iqp_independent_audit_ok":
        raise ValueError("Stage85 requires the passing Stage84 audit")
    cells = s81.canonical_cells(config, root)
    lookup = {
        (
            str(cell["model"]["record"]["target_id"]),
            int(cell["model"]["record"]["outer_fold"]),
        ): cell
        for cell in cells
    }
    output_directory = root / config["outputs"]["instance_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    instances = []
    for selection in config["calibration_instances"]:
        key = (str(selection["target_id"]), int(selection["outer_fold"]))
        instances.append(
            prepare_instance(root, output_directory, lookup[key], selection)
        )
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage85_mixed_radix_dirac_calibration_prepared",
        "instances": instances,
        "hardware_protocol": config["hardware_protocol"],
        "decision": {
            "allocation_only_preflight_authorized": True,
            "qci_device_jobs_authorized": 0,
            "physical_calibration_requires_separate_flag": True,
            "full_qci_production_authorized": False,
            "quantum_advantage_claim_authorized": False,
        },
        "data_boundary": {
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "qci_cloud_queries": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {},
    }
    result_path = root / config["outputs"]["result_json"]
    s84.write_json(result_path, result)
    report_path = root / config["outputs"]["report_md"]
    write_report(report_path, result)
    result["outputs"] = {
        "report_md": descriptor(root, report_path),
    }
    s84.write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage85_mixed_radix_dirac_calibration.json"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run((root / args.config).resolve(), root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
