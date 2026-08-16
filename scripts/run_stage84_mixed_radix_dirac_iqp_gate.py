"""Build and audit an exact mixed-radix Dirac-3 integer quadratic encoding."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import scripts.run_stage75_explicit_variable_k_cqm as s75
    import scripts.run_stage81_dirac_global_qubo_formulation_gate as s81
except ImportError:
    import run_stage75_explicit_variable_k_cqm as s75
    import run_stage81_dirac_global_qubo_formulation_gate as s81


RADIX = 16
DIGIT_COUNT = 4


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
        raise ValueError("Stage84 refuses to write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified(root: Path, descriptor: dict[str, Any], label: str) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file() or sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage84 {label} identity differs: {path}")
    if path.stat().st_size != int(descriptor["size_bytes"]):
        raise ValueError(f"Stage84 {label} size differs: {path}")
    return path


def digits(value: int) -> list[int]:
    if int(value) < 0:
        raise ValueError("Stage84 radix digits require a nonnegative integer")
    output = []
    remaining = int(value)
    for _ in range(DIGIT_COUNT):
        output.append(remaining % RADIX)
        remaining //= RADIX
    if remaining:
        raise ValueError(f"Stage84 value exceeds {DIGIT_COUNT} radix digits: {value}")
    return output


def add_term(terms: dict[tuple[int, ...], float], key: tuple[int, ...], value: float) -> None:
    if abs(float(value)) <= 1e-15:
        return
    canonical = tuple(sorted(int(item) for item in key))
    terms[canonical] = terms.get(canonical, 0.0) + float(value)


def add_square(
    terms: dict[tuple[int, ...], float],
    coefficients: dict[int, int],
    rhs: int,
    weight: float,
    binary_indices: set[int],
) -> float:
    items = sorted(coefficients.items())
    for index, coefficient in items:
        square = float(weight) * int(coefficient) ** 2
        if index in binary_indices:
            add_term(terms, (index,), square)
        else:
            add_term(terms, (index, index), square)
        add_term(
            terms,
            (index,),
            -2 * float(weight) * int(rhs) * int(coefficient),
        )
    for (left, left_value), (right, right_value) in itertools.combinations(items, 2):
        add_term(
            terms,
            (left, right),
            2 * float(weight) * int(left_value) * int(right_value),
        )
    return float(weight) * int(rhs) ** 2


def carry_upper_bounds(model: dict[str, Any]) -> list[int]:
    source_digits = [digits(int(value)) for value in model["deficits"]]
    carry = 0
    bounds = []
    for column in range(DIGIT_COUNT - 1):
        maximum_total = sum(value[column] for value in source_digits)
        maximum_total += RADIX - 1 + carry
        carry = maximum_total // RADIX
        bounds.append(carry)
    return bounds


def encode_cell(cell: dict[str, Any], k: int) -> dict[str, Any]:
    model = cell["model"]
    count = int(model["count"])
    x_names = [f"x{index:03d}" for index in range(count)]
    slack_names = [f"s{column}" for column in range(DIGIT_COUNT)]
    carry_names = [f"c{column}" for column in range(1, DIGIT_COUNT)]
    names = x_names + slack_names + carry_names
    index = {name: position + 1 for position, name in enumerate(names)}
    x_indices = {index[name] for name in x_names}
    carry_bounds = carry_upper_bounds(model)
    levels = [2] * count + [RADIX] * DIGIT_COUNT + [value + 1 for value in carry_bounds]
    reward = float(cell["reward"])
    shifted = np.asarray(model["raw_coefficients"], dtype=float) - reward
    pair_scale = max(float(np.max(np.abs(shifted))), 1e-12)
    terms: dict[tuple[int, ...], float] = {}
    for (left, right), value in zip(model["pairs"], shifted):
        add_term(
            terms,
            (index[x_names[left]], index[x_names[right]]),
            float(value) / pair_scale,
        )
    objective_range_bound = 2 * math.comb(int(k), 2)
    constraint_weight = float(objective_range_bound + 1)
    offset = add_square(
        terms,
        {index[name]: 1 for name in x_names},
        int(k),
        constraint_weight,
        x_indices,
    )
    threshold = int(cell["frontiers"][k]["quality_threshold"])
    threshold_digits = digits(threshold)
    deficit_digits = [digits(int(value)) for value in model["deficits"]]
    for column in range(DIGIT_COUNT):
        coefficients = {
            index[x_names[position]]: int(value[column])
            for position, value in enumerate(deficit_digits)
            if int(value[column]) != 0
        }
        coefficients[index[slack_names[column]]] = 1
        if column > 0:
            coefficients[index[carry_names[column - 1]]] = 1
        if column < DIGIT_COUNT - 1:
            coefficients[index[carry_names[column]]] = -RADIX
        offset += add_square(
            terms,
            coefficients,
            int(threshold_digits[column]),
            constraint_weight,
            x_indices,
        )
    terms = {
        key: value for key, value in terms.items() if abs(float(value)) > 1e-12
    }
    coefficients = [abs(float(value)) for value in terms.values()]
    full_scale = max(coefficients)
    normalized = {
        key: float(np.float32(float(value) / full_scale))
        for key, value in terms.items()
    }
    retained = [abs(value) for value in normalized.values() if value != 0.0]
    payload_terms = []
    for key, value in sorted(normalized.items(), key=lambda item: (len(item[0]), item[0])):
        if len(key) == 1:
            encoded_index = [0, int(key[0])]
        elif len(key) == 2:
            encoded_index = [int(key[0]), int(key[1])]
        else:
            raise ValueError("Stage84 emitted a term above degree two")
        payload_terms.append({"idx": encoded_index, "val": float(value)})
    return {
        "names": names,
        "index": index,
        "levels": levels,
        "terms": terms,
        "normalized_terms": normalized,
        "payload": {
            "file_name": f"stage84_{model['record']['target_id']}_of{model['record']['outer_fold']}_k{k}",
            "file_config": {
                "polynomial": {
                    "num_variables": len(names),
                    "min_degree": 1,
                    "max_degree": 2,
                    "data": payload_terms,
                }
            },
        },
        "offset": offset,
        "full_scale": full_scale,
        "pair_scale": pair_scale,
        "constraint_weight": constraint_weight,
        "objective_range_bound": float(objective_range_bound),
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
    slack = max(threshold - deficit, 0)
    slack_digits = digits(slack)
    sample = {position + 1: 0 for position in range(len(encoding["names"]))}
    chosen = set(subset)
    for position in range(model["count"]):
        sample[encoding["index"][f"x{position:03d}"]] = int(position in chosen)
    for column, value in enumerate(slack_digits):
        sample[encoding["index"][f"s{column}"]] = int(value)
    source_digits = [digits(int(value)) for value in model["deficits"]]
    carry = 0
    residuals = []
    threshold_digits = encoding["threshold_digits"]
    for column in range(DIGIT_COUNT):
        total = sum(source_digits[position][column] for position in subset)
        total += int(slack_digits[column]) + int(carry)
        if column < DIGIT_COUNT - 1:
            next_carry = total // RADIX
            sample[encoding["index"][f"c{column + 1}"]] = int(next_carry)
            residuals.append(total - int(threshold_digits[column]) - RADIX * next_carry)
            carry = next_carry
        else:
            residuals.append(total - int(threshold_digits[column]))
    for position, level_count in enumerate(encoding["levels"], start=1):
        if sample[position] < 0 or sample[position] >= int(level_count):
            raise ValueError("Stage84 generated an out-of-range integer assignment")
    return sample, residuals


def polynomial_energy(terms: dict[tuple[int, ...], float], sample: dict[int, int]) -> float:
    return float(
        sum(
            float(value) * math.prod(int(sample[index]) for index in key)
            for key, value in terms.items()
        )
    )


def validate_cell(
    config: dict[str, Any], cell: dict[str, Any], k: int, seed: int
) -> dict[str, Any]:
    encoding = encode_cell(cell, k)
    model = cell["model"]
    frontier = cell["frontiers"][k]
    reference = tuple(frontier["reference_subset"])
    reference_sample, reference_residuals = assignment_for_subset(
        encoding, model, reference
    )
    reference_full = polynomial_energy(encoding["terms"], reference_sample)
    reference_full += float(encoding["offset"])
    reference_expected = s75.variable_energy(model, reference, float(cell["reward"]))
    reference_expected /= float(encoding["pair_scale"])
    reference_normalized = polynomial_energy(
        encoding["normalized_terms"], reference_sample
    )
    reference_restored = reference_normalized * float(encoding["full_scale"])
    reference_restored += float(encoding["offset"])
    rng = np.random.default_rng(seed)
    maximum_float64_error = abs(reference_full - reference_expected)
    maximum_float32_error = abs(reference_restored - reference_expected)
    feasible_zero_residual_count = int(all(value == 0 for value in reference_residuals))
    infeasible_positive_penalty_count = 0
    feasible_random_count = 0
    for _ in range(int(config["validation"]["random_subsets_per_cell"])):
        subset = tuple(
            sorted(
                int(value)
                for value in rng.choice(model["count"], int(k), replace=False)
            )
        )
        sample, residuals = assignment_for_subset(encoding, model, subset)
        expected_objective = s75.variable_energy(model, subset, float(cell["reward"]))
        expected_objective /= float(encoding["pair_scale"])
        expected_full = expected_objective + float(encoding["constraint_weight"]) * (
            sum(int(value) ** 2 for value in residuals)
        )
        full = polynomial_energy(encoding["terms"], sample) + float(
            encoding["offset"]
        )
        restored = polynomial_energy(encoding["normalized_terms"], sample)
        restored = restored * float(encoding["full_scale"]) + float(
            encoding["offset"]
        )
        maximum_float64_error = max(maximum_float64_error, abs(full - expected_full))
        maximum_float32_error = max(maximum_float32_error, abs(restored - expected_full))
        if s75.subset_deficit(model, subset) <= int(encoding["threshold"]):
            feasible_random_count += 1
            feasible_zero_residual_count += int(all(value == 0 for value in residuals))
        else:
            infeasible_positive_penalty_count += int(any(value != 0 for value in residuals))
    record = model["record"]
    minimum_normalized_constraint_penalty = float(encoding["constraint_weight"])
    minimum_normalized_constraint_penalty /= float(encoding["full_scale"])
    return {
        "target_id": str(record["target_id"]),
        "outer_fold": int(record["outer_fold"]),
        "k": int(k),
        "candidate_receptor_count": int(model["count"]),
        "integer_variable_count": len(encoding["names"]),
        "binary_receptor_variable_count": int(model["count"]),
        "slack_digit_variable_count": DIGIT_COUNT,
        "carry_variable_count": DIGIT_COUNT - 1,
        "total_qci_levels": sum(int(value) for value in encoding["levels"]),
        "qci_level_limit_ok": sum(int(value) for value in encoding["levels"])
        <= int(config["experiment"]["qci_total_level_limit"]),
        "polynomial_max_degree": 2,
        "polynomial_term_count": len(encoding["terms"]),
        "constraint_weight": float(encoding["constraint_weight"]),
        "objective_range_bound": float(encoding["objective_range_bound"]),
        "constraint_weight_dominates_objective_range": float(
            encoding["constraint_weight"]
        )
        > float(encoding["objective_range_bound"]),
        "maximum_absolute_coefficient": float(encoding["full_scale"]),
        "normalized_dynamic_range": float(encoding["normalized_dynamic_range"]),
        "coefficient_retention_fraction": float(
            encoding["coefficient_retention_fraction"]
        ),
        "minimum_normalized_constraint_penalty": minimum_normalized_constraint_penalty,
        "maximum_float64_energy_identity_error": maximum_float64_error,
        "maximum_float32_restored_energy_error": maximum_float32_error,
        "constraint_penalty_to_float32_error_ratio": float(
            encoding["constraint_weight"]
        )
        / max(maximum_float32_error, 1e-15),
        "reference_constraint_residual_zero": all(
            value == 0 for value in reference_residuals
        ),
        "feasible_zero_residual_count": feasible_zero_residual_count,
        "feasible_assignment_count_tested": feasible_random_count + 1,
        "infeasible_positive_penalty_count": infeasible_positive_penalty_count,
        "infeasible_assignment_count_tested": int(
            config["validation"]["random_subsets_per_cell"]
        )
        - feasible_random_count,
    }


def summarize(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    gate = config["decision_gate"]
    maximum_float32_error = max(
        float(row["maximum_float32_restored_energy_error"]) for row in rows
    )
    minimum_penalty = min(
        float(row["minimum_normalized_constraint_penalty"]) for row in rows
    )
    minimum_precision_margin = min(
        float(row["constraint_penalty_to_float32_error_ratio"]) for row in rows
    )
    checks = {
        "all_level_limits_ok": all(bool(row["qci_level_limit_ok"]) for row in rows),
        "all_degree_two": all(int(row["polynomial_max_degree"]) == 2 for row in rows),
        "all_weights_dominate_objective_range": all(
            bool(row["constraint_weight_dominates_objective_range"]) for row in rows
        ),
        "all_reference_residuals_zero": all(
            bool(row["reference_constraint_residual_zero"]) for row in rows
        ),
        "all_random_feasible_residuals_zero": all(
            int(row["feasible_zero_residual_count"])
            == int(row["feasible_assignment_count_tested"])
            for row in rows
        ),
        "all_random_infeasible_penalties_positive": all(
            int(row["infeasible_positive_penalty_count"])
            == int(row["infeasible_assignment_count_tested"])
            for row in rows
        ),
        "coefficient_retention": min(
            float(row["coefficient_retention_fraction"]) for row in rows
        )
        >= float(gate["minimum_coefficient_retention_fraction"]),
        "float64_identity": max(
            float(row["maximum_float64_energy_identity_error"]) for row in rows
        )
        <= float(gate["maximum_float64_energy_identity_error"]),
        "float32_error": maximum_float32_error
        <= float(gate["maximum_float32_restored_energy_error"]),
        "precision_margin": minimum_precision_margin
        >= float(gate["minimum_constraint_penalty_to_error_ratio"]),
    }
    return {
        "fixed_k_encoding_count": len(rows),
        "maximum_integer_variable_count": max(
            int(row["integer_variable_count"]) for row in rows
        ),
        "maximum_total_qci_levels": max(int(row["total_qci_levels"]) for row in rows),
        "maximum_polynomial_term_count": max(
            int(row["polynomial_term_count"]) for row in rows
        ),
        "maximum_absolute_coefficient": max(
            float(row["maximum_absolute_coefficient"]) for row in rows
        ),
        "maximum_normalized_dynamic_range": max(
            float(row["normalized_dynamic_range"]) for row in rows
        ),
        "minimum_coefficient_retention_fraction": min(
            float(row["coefficient_retention_fraction"]) for row in rows
        ),
        "maximum_float64_energy_identity_error": max(
            float(row["maximum_float64_energy_identity_error"]) for row in rows
        ),
        "maximum_float32_restored_energy_error": maximum_float32_error,
        "minimum_normalized_constraint_penalty": minimum_penalty,
        "minimum_constraint_penalty_to_error_ratio": minimum_precision_margin,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    text = f"""# Stage84 Mixed-radix Dirac Integer Quadratic Gate

Stage84 represents each exact fixed-k quality slack with four base-16 digits
and three carry qudits. Four local column equations replace the single
large-coefficient quality square.

- Encodings audited: `{summary['fixed_k_encoding_count']}`.
- Maximum integer variables: `{summary['maximum_integer_variable_count']}`.
- Maximum total Dirac levels: `{summary['maximum_total_qci_levels']}` / 949.
- Maximum polynomial terms: `{summary['maximum_polynomial_term_count']}`.
- Maximum normalized dynamic range: `{summary['maximum_normalized_dynamic_range']:.4f}`.
- Float32 constraint-penalty/error margin: `{summary['minimum_constraint_penalty_to_error_ratio']:.2f}`.

External Dirac calibration preparation authorized:
`{result['decision']['external_dirac_calibration_preparation_authorized']}`.
This is an exact encoding and precision gate, not a physical solver result,
speedup measurement, efficacy validation, or quantum-advantage claim.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    for name in ("stage83_result", "stage83_audit", "stage79_physical_audit"):
        verified(root, config["inputs"][name], name)
    stage83 = read_json(root / config["inputs"]["stage83_audit"]["path"])
    if stage83["status"] != "stage83_quality_shell_qubo_independent_audit_ok":
        raise ValueError("Stage84 requires the passing Stage83 audit")
    cells = s81.canonical_cells(config, root)
    rows = []
    seed = int(config["validation"]["seed_base"])
    for cell in cells:
        for k in cell["frontiers"]:
            rows.append(validate_cell(config, cell, int(k), seed + len(rows)))
    summary = summarize(config, rows)
    outputs = config["outputs"]
    metrics_path = root / outputs["metrics_csv"]
    write_csv(metrics_path, rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage84_mixed_radix_dirac_iqp_gate_complete",
        "summary": summary,
        "decision": {
            "mixed_radix_exact_quality_encoding_frozen": bool(summary["gate_passed"]),
            "external_dirac_calibration_preparation_authorized": bool(
                summary["gate_passed"]
            ),
            "qci_device_jobs_authorized": 0,
            "full_qci_production_authorized": False,
            "quantum_advantage_claim_authorized": False,
        },
        "data_boundary": {
            "historical_development_models_read": len(cells),
            "fixed_k_encodings_audited": len(rows),
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "qci_cloud_queries": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            "metrics_csv": {
                "path": outputs["metrics_csv"],
                "sha256": sha256(metrics_path),
                "size_bytes": metrics_path.stat().st_size,
            }
        },
    }
    result_path = root / outputs["result_json"]
    write_json(result_path, result)
    report_path = root / outputs["report_md"]
    write_report(report_path, result)
    result["outputs"]["report_md"] = {
        "path": outputs["report_md"],
        "sha256": sha256(report_path),
        "size_bytes": report_path.stat().st_size,
    }
    write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage84_mixed_radix_dirac_iqp_gate.json"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run((root / args.config).resolve(), root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
