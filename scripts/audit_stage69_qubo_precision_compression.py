"""Independently audit Stage69 QUBO precision-compression outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any


TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def checked(root: Path, value: dict[str, Any], label: str) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage69 {label} identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage69 {label} size differs: {path}")
    return path


def close(observed: Any, expected: Any, label: str) -> None:
    if not math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=TOLERANCE
    ):
        raise ValueError(f"Stage69 numeric value differs: {label}")


def recompute_scale_summary(
    rows: list[dict[str, str]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    gates = config["compression_gate"]
    scales = [int(value) for value in config["development"]["quality_integer_scales"]]
    for scale in scales:
        selected = [
            row for row in rows if int(row["quality_integer_scale"]) == scale
        ]
        valid = [row for row in selected if row["status"] == "ok"]
        summary: dict[str, Any] = {
            "model_id": f"quality_scale_{scale}",
            "quality_integer_scale": scale,
            "cell_count": len(selected),
            "feasible_cell_count": len(valid),
            "pair_off_infeasible_cell_count": len(selected) - len(valid),
        }
        if valid:
            summary.update(
                {
                    "exact_subset_match_count": sum(
                        row["exact_subset_match"].lower() == "true" for row in valid
                    ),
                    "mean_subset_jaccard_vs_continuous": statistics.fmean(
                        float(row["subset_jaccard_vs_continuous"]) for row in valid
                    ),
                    "minimum_subset_jaccard_vs_continuous": min(
                        float(row["subset_jaccard_vs_continuous"]) for row in valid
                    ),
                    "mean_absolute_holdout_bedroc_gap": statistics.fmean(
                        abs(float(row["quantized_minus_continuous_holdout_bedroc"]))
                        for row in valid
                    ),
                    "maximum_absolute_holdout_bedroc_gap": max(
                        abs(float(row["quantized_minus_continuous_holdout_bedroc"]))
                        for row in valid
                    ),
                    "mean_holdout_gain_over_pair_off": statistics.fmean(
                        float(row["quantized_minus_pair_off_holdout_bedroc"])
                        for row in valid
                    ),
                    "minimum_actual_quality_floor_margin": min(
                        float(row["actual_quality_floor_margin"]) for row in valid
                    ),
                    "maximum_logical_variable_count": max(
                        int(row["logical_variable_count"]) for row in valid
                    ),
                    "maximum_quadratic_coefficient_count": max(
                        int(row["quadratic_coefficient_count"]) for row in valid
                    ),
                    "maximum_coefficient_dynamic_range": max(
                        float(row["coefficient_dynamic_range"]) for row in valid
                    ),
                    "maximum_factorized_energy_residual": max(
                        float(row["factorized_energy_residual"]) for row in valid
                    ),
                }
            )
        summary["compression_gate_passed"] = bool(
            len(valid) >= int(gates["minimum_feasible_cell_count"])
            and valid
            and int(summary["exact_subset_match_count"])
            >= int(gates["minimum_exact_subset_match_count"])
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
            and float(summary["maximum_coefficient_dynamic_range"])
            <= float(gates["maximum_compressed_dynamic_range"])
            and float(summary["maximum_factorized_energy_residual"])
            <= float(gates["maximum_factorized_energy_residual"])
        )
        output.append(summary)
    reference = next(
        row
        for row in output
        if int(row["quality_integer_scale"])
        == int(config["development"]["reference_quality_integer_scale"])
    )
    for row in output:
        row["dynamic_range_compression_factor_vs_4095"] = (
            float(reference["maximum_coefficient_dynamic_range"])
            / float(row["maximum_coefficient_dynamic_range"])
            if "maximum_coefficient_dynamic_range" in row
            else 0.0
        )
    return output


def compare_summaries(
    observed: list[dict[str, str]], expected: list[dict[str, Any]]
) -> None:
    lookup = {int(row["quality_integer_scale"]): row for row in observed}
    if set(lookup) != {int(row["quality_integer_scale"]) for row in expected}:
        raise ValueError("Stage69 scale-summary key set differs")
    for expected_row in expected:
        scale = int(expected_row["quality_integer_scale"])
        observed_row = lookup[scale]
        for field, value in expected_row.items():
            if isinstance(value, bool):
                if observed_row[field].lower() != str(value).lower():
                    raise ValueError(f"Stage69 boolean differs: {scale}:{field}")
            elif isinstance(value, (float, int)):
                close(observed_row[field], value, f"{scale}:{field}")
            elif observed_row[field] != str(value):
                raise ValueError(f"Stage69 value differs: {scale}:{field}")


def audit_models(model: dict[str, Any]) -> int:
    for record in model["models"]:
        receptor_ids = list(record["receptor_ids"])
        index = {value: position for position, value in enumerate(receptor_ids)}
        selected = tuple(
            sorted(index[value] for value in record["selected_subset"].split("+"))
        )
        deficits = [int(value) for value in record["integer_deficits"]]
        deficit = sum(deficits[value] for value in selected)
        slack = int(record["selected_slack_value"])
        maximum = int(record["maximum_integer_deficit"])
        if deficit + slack != maximum:
            raise ValueError("Stage69 model slack equality differs")
        triangle = iter(record["stable_redundancy_upper_triangle"])
        selected_set = set(selected)
        redundancy = 0.0
        for left, right in itertools.combinations(range(len(receptor_ids)), 2):
            value = float(next(triangle))
            if left in selected_set and right in selected_set:
                redundancy += value
        try:
            next(triangle)
            raise ValueError("Stage69 redundancy triangle has excess values")
        except StopIteration:
            pass
        energy = (
            redundancy
            + float(model["cardinality_penalty"])
            * (len(selected) - int(record["reference_k"])) ** 2
            + float(model["quality_penalty"])
            * (deficit + slack - maximum) ** 2
        )
        close(energy, record["selected_factorized_energy"], "model energy")
        close(redundancy, record["selected_redundancy_sum"], "model redundancy")
        close(
            abs(energy - redundancy), record["energy_residual"], "model residual"
        )
    return len(model["models"])


def run(
    config_path: Path, result_path: Path, root: Path, output_path: Path
) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path.resolve())
    result = read_json(result_path.resolve())
    if result.get("status") != "stage69_qubo_precision_compression_complete":
        raise ValueError("Stage69 source result did not complete")
    if checked(root, result["config"], "config").resolve() != config_path.resolve():
        raise ValueError("Stage69 result config differs")
    for key, value in config["implementation"].items():
        checked(root, value, key)
    for key, value in config["inputs"].items():
        checked(root, value, key)
    outputs = {
        key: checked(root, value, key) for key, value in result["outputs"].items()
    }
    rows = read_csv(outputs["cell_metrics_csv"])
    if len(rows) != 640:
        raise ValueError("Stage69 cell-metric count differs")
    for row in rows:
        if row["status"] != "ok":
            if row["quantized_subset"]:
                raise ValueError("Stage69 infeasible cell has a selected subset")
            continue
        if int(row["integer_deficit"]) > int(row["maximum_integer_deficit"]):
            raise ValueError("Stage69 integer quality inequality differs")
        if float(row["actual_quality_floor_margin"]) < -TOLERANCE:
            raise ValueError("Stage69 continuous quality floor is violated")
        if float(row["factorized_energy_residual"]) > TOLERANCE:
            raise ValueError("Stage69 factorized energy differs")
    expected_summaries = recompute_scale_summary(rows, config)
    compare_summaries(read_csv(outputs["scale_summary_csv"]), expected_summaries)
    eligible = [row for row in expected_summaries if row["compression_gate_passed"]]
    expected_selected = (
        min(eligible, key=lambda row: int(row["quality_integer_scale"]))
        if eligible
        else {}
    )
    selected = result["selected_compression"]
    if bool(selected) != bool(expected_selected):
        raise ValueError("Stage69 compression selection presence differs")
    if expected_selected:
        if int(selected["quality_integer_scale"]) != int(
            expected_selected["quality_integer_scale"]
        ):
            raise ValueError("Stage69 selected scale differs")
        for field, value in expected_selected.items():
            if isinstance(value, bool):
                if bool(selected[field]) != value:
                    raise ValueError(f"Stage69 selected boolean differs: {field}")
            elif isinstance(value, (float, int)):
                close(selected[field], value, f"selected:{field}")
            elif str(selected[field]) != str(value):
                raise ValueError(f"Stage69 selected value differs: {field}")

    uniformly_feasible = [
        row
        for row in expected_summaries
        if int(row["feasible_cell_count"])
        >= int(config["compression_gate"]["minimum_feasible_cell_count"])
    ]
    expected_near_miss = (
        min(
            uniformly_feasible,
            key=lambda row: int(row["quality_integer_scale"]),
        )
        if uniformly_feasible
        else {}
    )
    near_miss = result["best_uniform_near_miss"]
    if bool(near_miss) != bool(expected_near_miss):
        raise ValueError("Stage69 uniform near-miss presence differs")
    for field, value in expected_near_miss.items():
        if isinstance(value, bool):
            if bool(near_miss[field]) != value:
                raise ValueError(f"Stage69 near-miss boolean differs: {field}")
        elif isinstance(value, (float, int)):
            close(near_miss[field], value, f"near_miss:{field}")
        elif str(near_miss[field]) != str(value):
            raise ValueError(f"Stage69 near-miss value differs: {field}")

    model_record = read_json(outputs["model_record_json"])
    model_count = audit_models(model_record)
    diagnostic_summary = expected_selected or expected_near_miss
    diagnostic_scale = int(diagnostic_summary.get("quality_integer_scale", 0))
    expected_model_count = (
        len(config["development"]["target_order"])
        * int(config["development"]["outer_fold_count"])
        if diagnostic_scale
        else 0
    )
    if model_count != expected_model_count:
        raise ValueError("Stage69 diagnostic model count differs")
    if int(model_record["selected_quality_integer_scale"]) != int(
        expected_selected.get("quality_integer_scale", 0)
    ):
        raise ValueError("Stage69 model selected scale differs")
    if int(model_record["diagnostic_model_quality_integer_scale"]) != diagnostic_scale:
        raise ValueError("Stage69 model diagnostic scale differs")

    direct_expected = bool(expected_selected) and float(
        expected_selected["maximum_coefficient_dynamic_range"]
    ) <= float(config["direct_qpu_gate"]["maximum_coefficient_dynamic_range"])
    direct_gate = result["direct_qpu_gate"]
    if bool(direct_gate["direct_qpu_precision_gate_passed"]) != direct_expected:
        raise ValueError("Stage69 direct-QPU precision decision differs")
    close(
        direct_gate["observed_maximum_dynamic_range"],
        diagnostic_summary.get("maximum_coefficient_dynamic_range", 0.0),
        "direct-QPU observed dynamic range",
    )
    compression_authorized = bool(expected_selected)
    if bool(
        result["compression_gate"]["compressed_qubo_freeze_authorized"]
    ) != compression_authorized:
        raise ValueError("Stage69 compression authorization differs")
    if bool(result["decision"]["compact_solver_prototype_authorized"]) != (
        compression_authorized
    ):
        raise ValueError("Stage69 compact-prototype authorization differs")
    if bool(result["decision"]["direct_qpu_execution_authorized"]) != (
        compression_authorized and direct_expected
    ):
        raise ValueError("Stage69 direct-QPU authorization differs")
    audit = {
        "schema_version": "1.0",
        "status": "stage69_qubo_precision_compression_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "cell_metrics_independently_checked": len(rows),
        "scale_summaries_independently_recomputed": len(expected_summaries),
        "selected_scale_independently_verified": int(
            expected_selected.get("quality_integer_scale", 0)
        ),
        "diagnostic_near_miss_scale_independently_verified": int(
            expected_near_miss.get("quality_integer_scale", 0)
        ),
        "factorized_qubo_models_independently_checked": model_count,
        "compressed_qubo_freeze_authorized": result["compression_gate"][
            "compressed_qubo_freeze_authorized"
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
        default=Path("configs/stage69_qubo_precision_compression.json"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("data/stage69_qubo_precision_compression_result.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage69_qubo_precision_compression_audit.json"),
    )
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
