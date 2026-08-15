"""Prepare the globally exact nonnegative-gauge Stage86 Dirac rescue."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

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


def encode_cell(cell: dict[str, Any], k: int) -> dict[str, Any]:
    model = cell["model"]
    count = int(model["count"])
    x_names = [f"x{index:03d}" for index in range(count)]
    slack_names = [f"s{column}" for column in range(s84.DIGIT_COUNT)]
    carry_names = [f"c{column}" for column in range(1, s84.DIGIT_COUNT)]
    names = x_names + slack_names + carry_names
    index = {name: position + 1 for position, name in enumerate(names)}
    x_indices = {index[name] for name in x_names}
    carry_bounds = s84.carry_upper_bounds(model)
    levels = [2] * count + [s84.RADIX] * s84.DIGIT_COUNT + [
        value + 1 for value in carry_bounds
    ]

    raw = np.asarray(model["raw_coefficients"], dtype=float)
    pair_min = float(np.min(raw))
    pair_max = float(np.max(raw))
    pair_span = pair_max - pair_min
    if pair_span <= 1e-12:
        raise ValueError("Stage86 requires a nonconstant pair landscape")
    gauged = (raw - pair_min) / pair_span
    if float(np.min(gauged)) < -1e-12 or float(np.max(gauged)) > 1.0 + 1e-12:
        raise ValueError("Stage86 nonnegative pair gauge escaped [0,1]")

    pair_count = math.comb(int(k), 2)
    constraint_weight = float(pair_count + 1)
    terms: dict[tuple[int, ...], float] = {}
    for (left, right), value in zip(model["pairs"], gauged):
        s84.add_term(
            terms,
            (index[x_names[left]], index[x_names[right]]),
            float(value),
        )
    offset = s84.add_square(
        terms,
        {index[name]: 1 for name in x_names},
        int(k),
        constraint_weight,
        x_indices,
    )
    threshold = int(cell["frontiers"][k]["quality_threshold"])
    threshold_digits = s84.digits(threshold)
    deficit_digits = [s84.digits(int(value)) for value in model["deficits"]]
    for column in range(s84.DIGIT_COUNT):
        coefficients = {
            index[x_names[position]]: int(value[column])
            for position, value in enumerate(deficit_digits)
            if int(value[column]) != 0
        }
        coefficients[index[slack_names[column]]] = 1
        if column > 0:
            coefficients[index[carry_names[column - 1]]] = 1
        if column < s84.DIGIT_COUNT - 1:
            coefficients[index[carry_names[column]]] = -s84.RADIX
        offset += s84.add_square(
            terms,
            coefficients,
            int(threshold_digits[column]),
            constraint_weight,
            x_indices,
        )
    terms = {key: value for key, value in terms.items() if abs(value) > 1e-12}
    full_scale = max(abs(float(value)) for value in terms.values())
    normalized = {
        key: float(np.float32(float(value) / full_scale))
        for key, value in terms.items()
    }
    retained = [abs(value) for value in normalized.values() if value != 0.0]
    payload_terms = []
    for key, value in sorted(normalized.items(), key=lambda item: (len(item[0]), item[0])):
        encoded_index = [0, int(key[0])] if len(key) == 1 else [int(key[0]), int(key[1])]
        payload_terms.append({"idx": encoded_index, "val": float(value)})
    return {
        "names": names,
        "index": index,
        "levels": levels,
        "terms": terms,
        "normalized_terms": normalized,
        "payload": {
            "file_name": f"stage86_{model['record']['target_id']}_of{model['record']['outer_fold']}_k{k}",
            "file_config": {
                "polynomial": {
                    "num_variables": len(names),
                    "min_degree": 1,
                    "max_degree": 2,
                    "data": payload_terms,
                }
            },
        },
        "offset": float(offset),
        "full_scale": float(full_scale),
        "pair_min": pair_min,
        "pair_max": pair_max,
        "pair_span": pair_span,
        "constraint_weight": constraint_weight,
        "objective_upper_bound": float(pair_count),
        "global_penalty_margin": constraint_weight - float(pair_count),
        "threshold": threshold,
        "threshold_digits": threshold_digits,
        "carry_bounds": carry_bounds,
        "coefficient_retention_fraction": len(retained) / len(terms),
        "normalized_dynamic_range": max(retained) / min(retained),
    }


def assignment_for_subset(
    encoding: dict[str, Any], model: dict[str, Any], subset: tuple[int, ...]
) -> tuple[dict[int, int], list[int]]:
    threshold = int(encoding["threshold"])
    deficit = s75.subset_deficit(model, subset)
    if deficit > threshold:
        raise ValueError("Stage86 assignment requires a quality-feasible subset")
    slack_digits = s84.digits(threshold - deficit)
    source_digits = [s84.digits(int(value)) for value in model["deficits"]]
    sample = {position + 1: 0 for position in range(len(encoding["names"]))}
    chosen = set(subset)
    for position in range(int(model["count"])):
        sample[encoding["index"][f"x{position:03d}"]] = int(position in chosen)
    carry = 0
    residuals = []
    threshold_digits = encoding["threshold_digits"]
    for column in range(s84.DIGIT_COUNT):
        sample[encoding["index"][f"s{column}"]] = int(slack_digits[column])
        total = sum(source_digits[position][column] for position in chosen)
        total += int(slack_digits[column]) + carry
        next_carry = total // s84.RADIX if column < s84.DIGIT_COUNT - 1 else 0
        if column < s84.DIGIT_COUNT - 1:
            sample[encoding["index"][f"c{column + 1}"]] = next_carry
        residuals.append(total - int(threshold_digits[column]) - s84.RADIX * next_carry)
        carry = next_carry
    return sample, residuals


def original_from_encoded_feasible(
    encoding: dict[str, Any], encoded_with_offset: float, cell: dict[str, Any], k: int
) -> float:
    constant = (float(encoding["pair_min"]) - float(cell["reward"])) * math.comb(k, 2)
    return float(encoding["pair_span"]) * encoded_with_offset + constant


def exact_feasible(
    cell: dict[str, Any], k: int, encoding: dict[str, Any]
) -> dict[str, Any]:
    model = cell["model"]
    best_energy = math.inf
    best: list[tuple[tuple[int, ...], float]] = []
    feasible_count = 0
    total_count = 0
    for subset in itertools.combinations(range(int(model["count"])), k):
        total_count += 1
        if s75.subset_deficit(model, subset) > int(encoding["threshold"]):
            continue
        feasible_count += 1
        sample, residuals = assignment_for_subset(encoding, model, subset)
        if any(residuals):
            raise ValueError("Stage86 feasible assignment has a radix residual")
        energy = s84.polynomial_energy(encoding["normalized_terms"], sample)
        original = s75.variable_energy(model, subset, float(cell["reward"]))
        if energy < best_energy - 1e-10:
            best_energy = energy
            best = [(tuple(subset), original)]
        elif math.isclose(energy, best_energy, abs_tol=1e-10):
            best.append((tuple(subset), original))
    frontier = cell["frontiers"][k]
    if total_count != int(frontier["fixed_k_total_state_count"]):
        raise ValueError("Stage86 fixed-k total count differs")
    if feasible_count != int(frontier["fixed_k_feasible_state_count"]):
        raise ValueError("Stage86 feasible count differs")
    best.sort(key=lambda item: (item[1], item[0]))
    selected, original = best[0]
    encoded_with_offset = best_energy * float(encoding["full_scale"]) + float(encoding["offset"])
    restored = original_from_encoded_feasible(encoding, encoded_with_offset, cell, k)
    return {
        "total_count": total_count,
        "feasible_count": feasible_count,
        "optimum_degeneracy": len(best),
        "normalized_energy": best_energy,
        "encoded_with_offset": encoded_with_offset,
        "selected_subset": selected,
        "selected_original_objective": original,
        "restored_original_objective": restored,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = s84.read_json(config_path)
    failure = s84.read_json(root / config["inputs"]["stage85a_failure"])
    if failure["status"] != "stage85_physical_calibration_failed_stop_hardware":
        raise ValueError("Stage86 requires the frozen Stage85a failure")
    stage84_config = s84.read_json(root / config["inputs"]["stage84_config"])
    cells = s81.canonical_cells(stage84_config, root)
    rows = []
    cell_lookup = {}
    for cell in cells:
        model = cell["model"]
        key = (str(model["record"]["target_id"]), int(model["record"]["outer_fold"]))
        cell_lookup[key] = cell
        for k_value in cell["frontiers"]:
            k = int(k_value)
            encoding = encode_cell(cell, k)
            reference = tuple(cell["frontiers"][k]["reference_subset"])
            sample, residuals = assignment_for_subset(encoding, model, reference)
            float64_energy = s84.polynomial_energy(encoding["terms"], sample)
            float64_with_offset = float64_energy + float(encoding["offset"])
            float64_restored = original_from_encoded_feasible(
                encoding, float64_with_offset, cell, k
            )
            normalized_energy = s84.polynomial_energy(encoding["normalized_terms"], sample)
            encoded_with_offset = normalized_energy * float(encoding["full_scale"]) + float(encoding["offset"])
            restored = original_from_encoded_feasible(encoding, encoded_with_offset, cell, k)
            expected = s75.variable_energy(model, reference, float(cell["reward"]))
            rows.append(
                {
                    "target_id": key[0],
                    "outer_fold": key[1],
                    "candidate_count": int(model["count"]),
                    "k": k,
                    "integer_variable_count": len(encoding["names"]),
                    "total_levels": sum(int(value) for value in encoding["levels"]),
                    "pair_min": encoding["pair_min"],
                    "pair_max": encoding["pair_max"],
                    "pair_gauge_min": 0.0,
                    "pair_gauge_max": 1.0,
                    "constraint_weight": encoding["constraint_weight"],
                    "objective_upper_bound": encoding["objective_upper_bound"],
                    "global_penalty_margin": encoding["global_penalty_margin"],
                    "normalized_dynamic_range": encoding["normalized_dynamic_range"],
                    "coefficient_retention_fraction": encoding["coefficient_retention_fraction"],
                    "reference_residuals_zero": all(value == 0 for value in residuals),
                    "reference_float64_identity_error": abs(float64_restored - expected),
                    "reference_float32_restored_error": abs(restored - expected),
                    "free_tier_compatible": len(encoding["names"])
                    <= int(config["local_gate"]["free_tier_quadratic_variable_limit"]),
                }
            )
    if len(rows) != int(config["local_gate"]["required_encoding_count"]):
        raise ValueError("Stage86 encoding count differs")

    selection = config["rescue_instance"]
    cell = cell_lookup[(selection["target_id"], int(selection["outer_fold"]))]
    k = int(selection["k"])
    encoding = encode_cell(cell, k)
    exact = exact_feasible(cell, k, encoding)
    reference = tuple(cell["frontiers"][k]["reference_subset"])
    reference_objective = s75.variable_energy(cell["model"], reference, float(cell["reward"]))
    if abs(float(exact["selected_original_objective"]) - reference_objective) > float(
        selection["maximum_original_objective_delta"]
    ):
        raise ValueError("Stage86 exact quantized optimum differs from the oracle")
    if len(encoding["names"]) > int(config["local_gate"]["free_tier_quadratic_variable_limit"]):
        raise ValueError("Stage86 rescue exceeds the free-tier variable limit")

    instance_dir = root / config["outputs"]["instance_directory"]
    instance_dir.mkdir(parents=True, exist_ok=True)
    instance_id = selection["instance_id"]
    payload = dict(encoding["payload"])
    payload["file_name"] = f"stage86_{instance_id}"
    payload_path = instance_dir / f"{instance_id}.qci-polynomial.json"
    mapping_path = instance_dir / f"{instance_id}.qci-mapping.json"
    s84.write_json(payload_path, payload)
    model = cell["model"]
    mapping = {
        "schema_version": "1.0",
        "instance_id": instance_id,
        "target_id": selection["target_id"],
        "outer_fold": int(selection["outer_fold"]),
        "k": k,
        "reward_value": float(cell["reward"]),
        "quality_threshold": int(encoding["threshold"]),
        "receptor_ids": list(model["receptor_ids"]),
        "integer_deficits": [int(value) for value in model["deficits"]],
        "variable_order": list(encoding["names"]),
        "num_levels": [int(value) for value in encoding["levels"]],
        "coefficient_scale": float(encoding["full_scale"]),
        "constant_offset": float(encoding["offset"]),
        "pair_min": float(encoding["pair_min"]),
        "pair_span": float(encoding["pair_span"]),
        "constraint_weight": float(encoding["constraint_weight"]),
        "global_penalty_margin": float(encoding["global_penalty_margin"]),
        "carry_upper_bounds": [int(value) for value in encoding["carry_bounds"]],
        "quantized_exact": {
            "normalized_energy": float(exact["normalized_energy"]),
            "selected_subset": s75.subset_name(model, tuple(exact["selected_subset"])),
            "selected_original_objective": float(exact["selected_original_objective"]),
            "optimum_degeneracy": int(exact["optimum_degeneracy"]),
            "feasible_count": int(exact["feasible_count"]),
            "total_count": int(exact["total_count"]),
        },
        "qci_polynomial": descriptor(root, payload_path),
    }
    s84.write_json(mapping_path, mapping)
    write_csv(root / config["outputs"]["metrics_csv"], rows)
    gate = config["local_gate"]
    checks = {
        "all_pair_gauges_nonnegative": all(
            float(row["pair_gauge_min"]) >= 0.0
            and float(row["pair_gauge_max"]) <= 1.0
            for row in rows
        ),
        "all_global_penalty_certificates": all(
            float(row["global_penalty_margin"]) >= float(gate["minimum_global_penalty_margin"])
            for row in rows
        ),
        "all_float32_coefficients_retained": all(
            float(row["coefficient_retention_fraction"])
            >= float(gate["minimum_float32_coefficient_retention"])
            for row in rows
        ),
        "all_dynamic_ranges_pass": all(
            float(row["normalized_dynamic_range"])
            <= float(gate["maximum_normalized_dynamic_range"])
            for row in rows
        ),
        "all_reference_identities_pass": all(
            float(row["reference_float64_identity_error"])
            <= float(gate["maximum_float64_identity_error"])
            and float(row["reference_float32_restored_error"])
            <= float(gate["maximum_float32_restored_error"])
            and bool(row["reference_residuals_zero"])
            for row in rows
        ),
        "rescue_free_tier_compatible": len(encoding["names"])
        <= int(gate["free_tier_quadratic_variable_limit"]),
        "rescue_exact_oracle_preserved": abs(
            float(exact["selected_original_objective"]) - reference_objective
        )
        <= float(selection["maximum_original_objective_delta"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Stage86 local gate failed: {checks}")
    result = {
        "schema_version": "1.0",
        "status": "stage86_nonnegative_gauge_local_gate_ok",
        "encoding_count": len(rows),
        "checks": checks,
        "summary": {
            "maximum_normalized_dynamic_range": max(float(row["normalized_dynamic_range"]) for row in rows),
            "minimum_global_penalty_margin": min(float(row["global_penalty_margin"]) for row in rows),
            "free_tier_compatible_encoding_count": sum(bool(row["free_tier_compatible"]) for row in rows),
        },
        "rescue_instance": {
            "instance_id": instance_id,
            "integer_variable_count": len(encoding["names"]),
            "total_levels": sum(int(value) for value in encoding["levels"]),
            "polynomial_term_count": len(encoding["terms"]),
            "payload": descriptor(root, payload_path),
            "mapping": descriptor(root, mapping_path),
        },
        "hardware_protocol": config["hardware_protocol"],
        "decision": {
            "allocation_only_preflight_authorized": True,
            "qci_device_jobs_authorized": 0,
            "production_authorized": False,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    result_path = root / config["outputs"]["result_json"]
    s84.write_json(result_path, result)
    report_path = root / config["outputs"]["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Stage86 nonnegative-gauge Dirac rescue",
                "",
                f"- Encodings certified: `{len(rows)}`.",
                f"- Maximum normalized dynamic range: `{result['summary']['maximum_normalized_dynamic_range']:.3f}`.",
                f"- Minimum global exact-penalty margin: `{result['summary']['minimum_global_penalty_margin']:.1f}`.",
                f"- Rescue variables: `{len(encoding['names'])}` / `{gate['free_tier_quadratic_variable_limit']}`.",
                "",
                "The fixed-k-invariant nonnegative gauge removes the Stage84 incentive "
                "for incorrect cardinalities while preserving the exact classical oracle.",
                "Only an allocation preflight is authorized locally; no device job was submitted.",
                "",
            ]
        ),
        encoding="ascii",
    )
    return result


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
