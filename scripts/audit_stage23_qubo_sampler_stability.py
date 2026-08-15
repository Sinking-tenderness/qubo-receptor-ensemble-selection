"""Independently audit Stage 23 sampler-stability outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    distance_matrix,
    file_sha256,
    load_target,
    read_csv,
    read_json,
    rooted,
)
from scripts.run_stage22_structural_state_coverage_qubo import (
    build_coverage_terms,
    direct_greedy,
    objective_components,
)


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def parse_subset(value: str) -> tuple[str, ...]:
    subset = tuple(sorted(part for part in value.split("+") if part))
    if len(subset) != len(set(subset)):
        raise ValueError("duplicate conformer in subset")
    return subset


def audit(config_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    if result.get("status") != "stage23_qubo_sampler_stability_complete":
        raise ValueError("unexpected Stage23 status")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("config hash differs")
    implementation = rooted(root, result["implementation"]["path"])
    if result["implementation"]["sha256"] != file_sha256(implementation):
        raise ValueError("implementation hash differs")
    baseline_path = rooted(root, result["strong_classical_baseline"]["path"])
    if result["strong_classical_baseline"]["sha256"] != file_sha256(baseline_path):
        raise ValueError("baseline hash differs")
    baseline = read_json(baseline_path)
    output_paths: dict[str, Path] = {}
    for key, descriptor in result["outputs"].items():
        path = rooted(root, descriptor["path"])
        if descriptor["sha256"] != file_sha256(path):
            raise ValueError(f"output hash differs: {key}")
        output_paths[key] = path
    reads = read_csv(output_paths["read_csv"])
    batches = read_csv(output_paths["batch_csv"])
    objective = config["objective"]
    sampler = config["sampler"]
    gate = config["gate"]
    fractions = [float(value) for value in objective["neighborhood_fractions"]]
    target_k = int(objective["k"])
    expected_reads = (
        len(config["targets"])
        * len(fractions)
        * int(sampler["batch_count"])
        * int(sampler["reads_per_batch"])
    )
    expected_batches = (
        len(config["targets"]) * len(fractions) * int(sampler["batch_count"])
    )
    if len(reads) != expected_reads or len(batches) != expected_batches:
        raise ValueError("Stage23 row count differs")
    read_groups: dict[tuple[str, float, int], list[dict[str, str]]] = {}
    for row in reads:
        key = (
            row["target_id"],
            float(row["neighborhood_fraction"]),
            int(row["batch"]),
        )
        read_groups.setdefault(key, []).append(row)
    batch_index = {
        (row["target_id"], float(row["neighborhood_fraction"]), int(row["batch"])): row
        for row in batches
    }
    recomputed_records: dict[str, Any] = {}
    checked_reads = 0
    for target_id, spec in config["targets"].items():
        target = load_target(root, target_id, spec)
        ids = target["ids"]
        matrix = distance_matrix(ids, target["distances"])
        recomputed_records[target_id] = {"fractions": {}}
        for fraction in fractions:
            terms = build_coverage_terms(ids, matrix, fraction)
            greedy = direct_greedy(
                ids, matrix, terms, target_k, float(objective["diversity_weight"])
            )
            greedy_value = objective_components(
                greedy, ids, matrix, terms, float(objective["diversity_weight"])
            )["composite_objective"]
            classical_rows = [
                row
                for row in baseline["rows"]
                if row["target_id"] == target_id
                and math.isclose(
                    float(row["neighborhood_fraction"]), fraction, abs_tol=1e-12
                )
            ]
            strong_value = max(
                float(row["refined_composite_objective"]) for row in classical_rows
            )
            batch_values: list[float] = []
            batch_deltas: list[float] = []
            for batch in range(int(sampler["batch_count"])):
                key = (target_id, fraction, batch)
                group = read_groups[key]
                if len(group) != int(sampler["reads_per_batch"]):
                    raise ValueError(f"{key}: read count differs")
                values = []
                for row in group:
                    subset = parse_subset(row["selected_subset"])
                    if len(subset) != target_k or not set(subset).issubset(ids):
                        raise ValueError(f"{key}: invalid subset")
                    metrics = objective_components(
                        subset,
                        ids,
                        matrix,
                        terms,
                        float(objective["diversity_weight"]),
                    )
                    for metric in (
                        "coverage_fraction",
                        "mean_pair_distance_normalized",
                        "minimum_pair_distance_normalized",
                        "composite_objective",
                    ):
                        close(float(row[metric]), float(metrics[metric]), f"{key}/{metric}")
                    close(
                        float(row["delta_vs_direct_greedy"]),
                        float(metrics["composite_objective"]) - greedy_value,
                        f"{key}/greedy-delta",
                    )
                    close(
                        float(row["delta_vs_strong_classical"]),
                        float(metrics["composite_objective"]) - strong_value,
                        f"{key}/strong-delta",
                    )
                    values.append(float(metrics["composite_objective"]))
                    checked_reads += 1
                best = max(values)
                recorded = batch_index[key]
                close(float(recorded["best_objective"]), best, f"{key}/batch-best")
                close(
                    float(recorded["delta_vs_strong_classical"]),
                    best - strong_value,
                    f"{key}/batch-strong-delta",
                )
                batch_values.append(best)
                batch_deltas.append(best - strong_value)
            best = max(batch_values)
            within = sum(
                value >= best - float(gate["objective_tolerance"])
                for value in batch_values
            )
            above = sum(
                delta > float(gate["minimum_gain_vs_strong_classical"])
                for delta in batch_deltas
            )
            recomputed_records[target_id]["fractions"][f"{fraction:.6f}"] = {
                "best_batch_objective": best,
                "within_tolerance_batch_fraction": within / len(batch_values),
                "above_strong_classical_batch_fraction": above / len(batch_values),
            }
            recorded = result["target_records"][target_id]["fractions"][f"{fraction:.6f}"]
            for key in (
                "best_batch_objective",
                "within_tolerance_batch_fraction",
                "above_strong_classical_batch_fraction",
            ):
                close(float(recorded[key]), float(recomputed_records[target_id]["fractions"][f"{fraction:.6f}"][key]), f"{target_id}/{fraction}/{key}")
    primary = float(objective["primary_neighborhood_fraction"])
    primary_pass = all(
        recomputed_records[target_id]["fractions"][f"{primary:.6f}"][
            "within_tolerance_batch_fraction"
        ]
        >= float(gate["minimum_batch_fraction_within_tolerance"])
        and recomputed_records[target_id]["fractions"][f"{primary:.6f}"][
            "above_strong_classical_batch_fraction"
        ]
        >= float(gate["minimum_batch_fraction_above_strong_classical"])
        for target_id in config["targets"]
    )
    sensitivity = {}
    for target_id in sorted(config["targets"]):
        positive = sum(
            recomputed_records[target_id]["fractions"][f"{fraction:.6f}"][
                "above_strong_classical_batch_fraction"
            ]
            >= float(gate["minimum_batch_fraction_above_strong_classical"])
            for fraction in fractions
        )
        sensitivity[target_id] = {
            "positive_fraction_count": positive,
            "fraction_count": len(fractions),
            "passed": positive >= int(gate["minimum_positive_sensitivity_fractions"]),
        }
    gate_passed = primary_pass and all(value["passed"] for value in sensitivity.values())
    if result["decision"]["primary_pass"] != primary_pass:
        raise ValueError("primary decision differs")
    if result["decision"]["sensitivity"] != sensitivity:
        raise ValueError("sensitivity decision differs")
    if result["decision"]["sampler_stability_gate_passed"] != gate_passed:
        raise ValueError("gate decision differs")
    if any(int(value) != 0 for value in result["data_boundary"].values()):
        raise ValueError("nonzero data boundary")
    audit_result = {
        "schema_version": "1.0",
        "status": "stage23_qubo_sampler_stability_audit_ok",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "result": {"path": result_path.relative_to(root).as_posix(), "sha256": file_sha256(result_path)},
        "coverage": {
            "read_rows_recomputed": checked_reads,
            "batch_rows_recomputed": len(batches),
            "target_count": len(config["targets"]),
            "fraction_count": len(fractions),
        },
        "decision_recomputed": {
            "primary_pass": primary_pass,
            "sensitivity": sensitivity,
            "sampler_stability_gate_passed": gate_passed,
        },
        "checks": {
            "input_and_output_hashes_verified": True,
            "all_read_objectives_recomputed": True,
            "all_batch_best_values_recomputed": True,
            "strong_classical_deltas_recomputed": True,
            "decision_recomputed": True,
            "data_boundary_zero": True,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
    }
    output = rooted(root, output_path.as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage23_qubo_sampler_stability.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/stage23_qubo_sampler_stability_audit.json"))
    args = parser.parse_args()
    audit(args.config, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
