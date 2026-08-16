"""Independently audit Stage70 constraint-aware QUBO encoding outputs."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json, write_json  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


TOLERANCE = 1e-9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()




def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))




def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def checked(root: Path, value: dict[str, Any], label: str) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage70 {label} identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage70 {label} size differs: {path}")
    return path


def close(observed: Any, expected: Any, label: str) -> None:
    if not math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=TOLERANCE
    ):
        raise ValueError(f"Stage70 numeric value differs: {label}")


def truth(value: Any) -> bool:
    return str(value).lower() == "true"


def interval_complete(weights: list[int], maximum: int) -> bool:
    covered = 0
    for weight in sorted(weights):
        if weight > covered + 1:
            return False
        covered += weight
    return covered == maximum


def independent_qubo_summary(
    redundancy: np.ndarray,
    deficits: np.ndarray,
    maximum_deficit: int,
    slack_weights: list[int],
    subset_size: int,
    center: int,
    cardinality_penalty: float,
    quality_penalty: float,
) -> dict[str, Any]:
    receptor_count = len(deficits)
    centered = deficits.astype(float) - center
    slack = np.asarray(slack_weights, dtype=float)
    rhs = maximum_deficit - subset_size * center
    linear_parts = [
        cardinality_penalty * (1 - 2 * subset_size)
        + quality_penalty * (centered * centered - 2 * rhs * centered)
    ]
    if len(slack):
        linear_parts.append(quality_penalty * (slack * slack - 2 * rhs * slack))
    linear = np.concatenate(linear_parts)
    receptor_upper = np.triu_indices(receptor_count, 1)
    quadratic_parts = [
        2 * cardinality_penalty
        + redundancy[receptor_upper]
        + 2
        * quality_penalty
        * (centered[:, None] * centered[None, :])[receptor_upper]
    ]
    if len(slack):
        quadratic_parts.append(
            (2 * quality_penalty * centered[:, None] * slack[None, :]).ravel()
        )
        slack_upper = np.triu_indices(len(slack), 1)
        quadratic_parts.append(
            (2 * quality_penalty * slack[:, None] * slack[None, :])[
                slack_upper
            ]
        )
    quadratic = np.concatenate(quadratic_parts)
    coefficients = np.abs(np.concatenate([linear, quadratic]))
    coefficients = coefficients[coefficients > 1e-10]
    result = {
        "logical_variable_count": receptor_count + len(slack_weights),
        "receptor_variable_count": receptor_count,
        "slack_variable_count": len(slack_weights),
        "linear_coefficient_count": int(np.sum(np.abs(linear) > 1e-10)),
        "quadratic_coefficient_count": int(
            np.sum(np.abs(quadratic) > 1e-10)
        ),
        "constant": float(
            cardinality_penalty * subset_size**2 + quality_penalty * rhs**2
        ),
        "minimum_absolute_nonzero_coefficient": float(np.min(coefficients)),
        "maximum_absolute_coefficient": float(np.max(coefficients)),
        "coefficient_dynamic_range": float(
            np.max(coefficients) / np.min(coefficients)
        ),
        "centered_rhs": rhs,
    }
    result["qubo_sha256"] = canonical_sha256(
        {
            "center": center,
            "centered_rhs": rhs,
            "linear": [float(value) for value in linear],
            "quadratic": [float(value) for value in quadratic],
            "constant": result["constant"],
        }
    )
    return result


def recompute_summaries(
    rows: list[dict[str, str]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    gates = config["encoding_gate"]
    stage69 = read_json(Path(config["_root"]) / config["inputs"]["stage69_result"]["path"])
    stage69_maximum = float(
        stage69["best_uniform_near_miss"]["maximum_coefficient_dynamic_range"]
    )
    output: list[dict[str, Any]] = []
    for cap in config["encoding_screen"]["slack_weight_caps"]:
        selected = [row for row in rows if int(row["slack_weight_cap"]) == int(cap)]
        summary = {
            "candidate_id": f"tight_cap{int(cap)}_centered_pair_upper",
            "slack_weight_cap": int(cap),
            "cell_count": len(selected),
            "analytic_exact_penalty_certificate_count": sum(
                truth(row["analytic_exact_penalty_certificate"]) for row in selected
            ),
            "exact_subset_match_count_vs_continuous": sum(
                truth(row["source_exact_subset_match_vs_continuous"])
                for row in selected
            ),
            "mean_subset_jaccard_vs_continuous": statistics.fmean(
                float(row["source_subset_jaccard_vs_continuous"])
                for row in selected
            ),
            "minimum_subset_jaccard_vs_continuous": min(
                float(row["source_subset_jaccard_vs_continuous"])
                for row in selected
            ),
            "mean_absolute_holdout_bedroc_gap": statistics.fmean(
                float(row["source_absolute_holdout_bedroc_gap"])
                for row in selected
            ),
            "maximum_absolute_holdout_bedroc_gap": max(
                float(row["source_absolute_holdout_bedroc_gap"])
                for row in selected
            ),
            "minimum_actual_quality_floor_margin": min(
                float(row["source_actual_quality_floor_margin"])
                for row in selected
            ),
            "minimum_analytic_invalid_state_gap_lower_bound": min(
                float(row["analytic_invalid_state_gap_lower_bound"])
                for row in selected
            ),
            "maximum_factorized_energy_residual": max(
                float(row["factorized_energy_residual"]) for row in selected
            ),
            "maximum_logical_variable_count": max(
                int(row["logical_variable_count"]) for row in selected
            ),
            "maximum_quadratic_coefficient_count": max(
                int(row["quadratic_coefficient_count"]) for row in selected
            ),
            "maximum_slack_variable_count": max(
                int(row["slack_variable_count"]) for row in selected
            ),
            "maximum_coefficient_dynamic_range": max(
                float(row["coefficient_dynamic_range"]) for row in selected
            ),
            "maximum_absolute_coefficient": max(
                float(row["maximum_absolute_coefficient"]) for row in selected
            ),
        }
        summary["dynamic_range_improvement_factor_vs_stage69"] = (
            stage69_maximum
            / float(summary["maximum_coefficient_dynamic_range"])
        )
        summary["encoding_gate_passed"] = bool(
            int(summary["cell_count"]) >= int(gates["required_cell_count"])
            and int(summary["analytic_exact_penalty_certificate_count"])
            >= int(gates["required_exact_penalty_certificate_count"])
            and int(summary["exact_subset_match_count_vs_continuous"])
            >= int(gates["minimum_exact_subset_match_count_vs_continuous"])
            and float(summary["mean_subset_jaccard_vs_continuous"])
            >= float(gates["minimum_mean_subset_jaccard"])
            and float(summary["minimum_subset_jaccard_vs_continuous"])
            >= float(gates["minimum_subset_jaccard"])
            and float(summary["mean_absolute_holdout_bedroc_gap"])
            <= float(gates["maximum_mean_absolute_holdout_bedroc_gap"])
            and float(summary["maximum_absolute_holdout_bedroc_gap"])
            <= float(gates["maximum_absolute_holdout_bedroc_gap"])
            and float(summary["minimum_actual_quality_floor_margin"])
            >= -float(gates["maximum_quality_floor_violation"])
            and float(summary["minimum_analytic_invalid_state_gap_lower_bound"])
            >= float(gates["minimum_analytic_invalid_state_gap"])
            and float(summary["maximum_factorized_energy_residual"])
            <= float(gates["maximum_factorized_energy_residual"])
            and int(summary["maximum_logical_variable_count"])
            <= int(gates["maximum_logical_variable_count"])
            and int(summary["maximum_quadratic_coefficient_count"])
            <= int(gates["maximum_quadratic_coefficient_count"])
            and float(summary["maximum_coefficient_dynamic_range"])
            <= float(gates["maximum_coefficient_dynamic_range"])
            and float(summary["dynamic_range_improvement_factor_vs_stage69"])
            >= float(gates["minimum_dynamic_range_improvement_factor"])
        )
        output.append(summary)
    return output


def compare_summaries(
    observed: list[dict[str, str]], expected: list[dict[str, Any]]
) -> None:
    lookup = {row["candidate_id"]: row for row in observed}
    if set(lookup) != {row["candidate_id"] for row in expected}:
        raise ValueError("Stage70 candidate-summary key set differs")
    for expected_row in expected:
        observed_row = lookup[expected_row["candidate_id"]]
        for field, value in expected_row.items():
            if isinstance(value, bool):
                if truth(observed_row[field]) != value:
                    raise ValueError(f"Stage70 boolean differs: {field}")
            elif isinstance(value, (float, int)):
                close(observed_row[field], value, f"summary:{field}")
            elif observed_row[field] != str(value):
                raise ValueError(f"Stage70 value differs: {field}")


def audit_models(model: dict[str, Any], config: dict[str, Any]) -> int:
    for record in model["models"]:
        receptor_ids = list(record["receptor_ids"])
        receptor_index = {value: index for index, value in enumerate(receptor_ids)}
        selected = tuple(
            sorted(receptor_index[value] for value in record["selected_subset"].split("+"))
        )
        deficits = np.asarray(record["integer_deficits"], dtype=int)
        maximum = int(record["maximum_integer_deficit"])
        subset_size = int(record["reference_k"])
        minimum = int(np.sum(np.sort(deficits)[:subset_size]))
        if minimum != int(record["minimum_fixed_k_deficit"]):
            raise ValueError("Stage70 tight lower deficit differs")
        tight_maximum = maximum - minimum
        weights = [int(value) for value in record["slack_weights"]]
        if tight_maximum != int(record["tight_slack_maximum"]) or not interval_complete(
            weights, tight_maximum
        ):
            raise ValueError("Stage70 tight slack encoding differs")
        redundancy = np.zeros((len(receptor_ids), len(receptor_ids)), dtype=float)
        triangle = iter(record["stable_redundancy_upper_triangle"])
        for left, right in itertools.combinations(range(len(receptor_ids)), 2):
            value = float(next(triangle))
            redundancy[left, right] = value
            redundancy[right, left] = value
        try:
            next(triangle)
            raise ValueError("Stage70 redundancy triangle has excess values")
        except StopIteration:
            pass
        center = int(record["integer_center"])
        penalty_k = float(record["cardinality_penalty"])
        penalty_q = float(record["quality_penalty"])
        if penalty_k != penalty_q:
            raise ValueError("Stage70 penalty policy differs")
        close(
            penalty_k,
            float(record["pair_off_redundancy_upper_bound"])
            + float(config["encoding_screen"]["penalty_margin"]),
            "model penalty",
        )
        selected_objective = sum(
            redundancy[left, right]
            for left, right in itertools.combinations(selected, 2)
        )
        close(selected_objective, record["selected_redundancy_sum"], "model objective")
        slack_value = int(record["selected_slack_value"])
        centered_sum = int(np.sum(deficits[list(selected)])) - center * len(selected)
        centered_rhs = maximum - center * subset_size
        if centered_rhs != int(record["centered_rhs"]):
            raise ValueError("Stage70 centered right-hand side differs")
        energy = (
            selected_objective
            + penalty_k * (len(selected) - subset_size) ** 2
            + penalty_q * (centered_sum + slack_value - centered_rhs) ** 2
        )
        close(energy, record["selected_factorized_energy"], "model energy")
        close(
            abs(energy - selected_objective),
            record["factorized_energy_residual"],
            "model energy residual",
        )
        close(
            penalty_k - selected_objective,
            record["analytic_invalid_state_gap_lower_bound"],
            "model invalid-state gap",
        )
        expected = independent_qubo_summary(
            redundancy,
            deficits,
            maximum,
            weights,
            subset_size,
            center,
            penalty_k,
            penalty_q,
        )
        observed = record["qubo_summary"]
        for field, value in expected.items():
            if field == "centered_rhs":
                close(record["centered_rhs"], value, "model QUBO:centered_rhs")
                continue
            if isinstance(value, (float, int)):
                close(observed[field], value, f"model QUBO:{field}")
            elif observed[field] != value:
                raise ValueError(f"Stage70 model QUBO differs: {field}")
        alternatives = []
        for alternative in range(int(np.max(deficits)) + 1):
            summary = independent_qubo_summary(
                redundancy,
                deficits,
                maximum,
                weights,
                subset_size,
                alternative,
                penalty_k,
                penalty_q,
            )
            alternatives.append(
                (
                    float(summary["coefficient_dynamic_range"]),
                    float(summary["maximum_absolute_coefficient"]),
                    alternative,
                )
            )
        if center != min(alternatives)[2]:
            raise ValueError("Stage70 selected integer center is not coefficient-optimal")
    return len(model["models"])


def run(
    config_path: Path, result_path: Path, root: Path, output_path: Path
) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path.resolve())
    config["_root"] = str(root)
    result = read_json(result_path.resolve())
    if result.get("status") != "stage70_constraint_aware_qubo_encoding_complete":
        raise ValueError("Stage70 source result did not complete")
    if checked(root, result["config"], "config").resolve() != config_path.resolve():
        raise ValueError("Stage70 result config differs")
    for key, value in config["implementation"].items():
        checked(root, value, key)
    for key, value in config["inputs"].items():
        checked(root, value, key)
    outputs = {
        key: checked(root, value, key) for key, value in result["outputs"].items()
    }
    rows = read_csv(outputs["cell_metrics_csv"])
    expected_count = len(config["encoding_screen"]["slack_weight_caps"]) * int(
        config["encoding_gate"]["required_cell_count"]
    )
    if len(rows) != expected_count:
        raise ValueError("Stage70 cell-metric count differs")
    for row in rows:
        if not truth(row["slack_interval_complete"]):
            raise ValueError("Stage70 row has an incomplete slack interval")
        close(
            float(row["cardinality_penalty"]),
            float(row["pair_off_redundancy_upper_bound"])
            + float(config["encoding_screen"]["penalty_margin"]),
            "row exact penalty",
        )
        close(
            float(row["cardinality_penalty"])
            - float(row["selected_redundancy_sum"]),
            row["analytic_invalid_state_gap_lower_bound"],
            "row invalid-state gap",
        )
        if float(row["factorized_energy_residual"]) > TOLERANCE:
            raise ValueError("Stage70 row factorized energy differs")
        if float(row["source_actual_quality_floor_margin"]) < -TOLERANCE:
            raise ValueError("Stage70 row violates the source quality floor")
        close(
            float(row["stage69_coefficient_dynamic_range"])
            / float(row["coefficient_dynamic_range"]),
            row["cell_dynamic_range_improvement_factor"],
            "row dynamic-range improvement",
        )
    expected_summaries = recompute_summaries(rows, config)
    compare_summaries(
        read_csv(outputs["candidate_summary_csv"]), expected_summaries
    )
    eligible = [row for row in expected_summaries if row["encoding_gate_passed"]]
    expected_selected = (
        min(
            eligible,
            key=lambda row: (
                float(row["maximum_coefficient_dynamic_range"]),
                int(row["maximum_logical_variable_count"]),
                int(row["maximum_quadratic_coefficient_count"]),
                int(row["slack_weight_cap"]),
            ),
        )
        if eligible
        else {}
    )
    selected = result["selected_encoding"]
    if bool(selected) != bool(expected_selected):
        raise ValueError("Stage70 selection presence differs")
    for field, value in expected_selected.items():
        if isinstance(value, bool):
            if bool(selected[field]) != value:
                raise ValueError(f"Stage70 selected boolean differs: {field}")
        elif isinstance(value, (float, int)):
            close(selected[field], value, f"selected:{field}")
        elif str(selected[field]) != str(value):
            raise ValueError(f"Stage70 selected value differs: {field}")
    model_count = audit_models(read_json(outputs["model_record_json"]), config)
    expected_models = (
        len(config["development"]["target_order"])
        * int(config["development"]["outer_fold_count"])
        if expected_selected
        else 0
    )
    if model_count != expected_models:
        raise ValueError("Stage70 model count differs")
    direct_expected = bool(expected_selected) and float(
        expected_selected["maximum_coefficient_dynamic_range"]
    ) <= float(config["direct_qpu_gate"]["maximum_coefficient_dynamic_range"])
    if bool(result["direct_qpu_gate"]["direct_qpu_precision_gate_passed"]) != (
        direct_expected
    ):
        raise ValueError("Stage70 direct-QPU precision decision differs")
    freeze_expected = bool(expected_selected)
    if bool(
        result["encoding_gate"]["compact_logical_qubo_freeze_authorized"]
    ) != freeze_expected:
        raise ValueError("Stage70 compact-QUBO authorization differs")
    if bool(result["decision"]["coefficient_noise_simulation_authorized"]) != (
        freeze_expected
    ):
        raise ValueError("Stage70 noise-simulation authorization differs")
    if bool(result["decision"]["direct_qpu_execution_authorized"]) != (
        freeze_expected and direct_expected
    ):
        raise ValueError("Stage70 direct-QPU authorization differs")
    audit = {
        "schema_version": "1.0",
        "status": "stage70_constraint_aware_qubo_encoding_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "cell_metrics_independently_checked": len(rows),
        "candidate_summaries_independently_recomputed": len(expected_summaries),
        "selected_candidate_independently_verified": expected_selected.get(
            "candidate_id", ""
        ),
        "selected_slack_weight_cap_independently_verified": int(
            expected_selected.get("slack_weight_cap", 0)
        ),
        "factorized_qubo_models_independently_checked": model_count,
        "compact_logical_qubo_freeze_authorized": freeze_expected,
        "coefficient_noise_simulation_authorized": result["decision"][
            "coefficient_noise_simulation_authorized"
        ],
        "direct_qpu_execution_authorized": result["decision"][
            "direct_qpu_execution_authorized"
        ],
        "quantum_advantage_claim_authorized": False,
        "data_boundary": result["data_boundary"],
    }
    write_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage70_constraint_aware_qubo_encoding.json"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("data/stage70_constraint_aware_qubo_encoding_result.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage70_constraint_aware_qubo_encoding_audit.json"),
    )
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
