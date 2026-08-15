"""Independently audit the Stage36 consensus-objective landscape screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import file_sha256, read_json, rooted, write_json
from scripts.run_stage31_pparg_objective_landscape_screen import all_assignments, build_cohort, load_inputs, successor_and_local_metrics
from scripts.run_stage36_pparg_consensus_objective_landscape import component_arrays, objective_values


def audit(config_path: Path, root: Path) -> dict:
    config = read_json(config_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    result = read_json(outputs["result_json"])
    loaded = load_inputs(root, config)
    assignments = all_assignments()
    tolerance = float(config["difficulty_gate"]["objective_tie_tolerance"])
    checks = {
        "result_status": result.get("status") == "stage36_pparg_consensus_objective_landscape_complete",
        "screen_complete": result["decision"]["exact_landscape_screen_complete"] is True,
        "data_boundary_zero": all(int(value) == 0 for value in result["data_boundary"].values()),
    }
    maximum_differences = {"exact_optimum": 0.0, "objective_minimum": 0.0, "strict_local_count": 0, "optimum_basin_fraction": 0.0}
    cell_checks = {}
    recomputed_passing = {spec["objective_id"]: 0 for spec in config["objective_families_in_priority_order"]}
    threshold_ranges = []
    for cohort_spec in config["exact_candidate_cohorts"]:
        cohort = build_cohort(cohort_spec, loaded)
        components = component_arrays(cohort, assignments)
        for spec in config["objective_families_in_priority_order"]:
            objective_id = spec["objective_id"]
            scores = objective_values(objective_id, components)
            landscape = successor_and_local_metrics(scores, assignments, tolerance)
            stored = result["objective_records"][objective_id]["cohorts"][cohort["cohort_id"]]
            differences = {
                "exact_optimum": abs(float(scores.max()) - float(stored["exact_optimum"])),
                "objective_minimum": abs(float(scores.min()) - float(stored["objective_minimum"])),
                "strict_local_count": abs(int(landscape["strict_local"].sum()) - int(stored["strict_local_optimum_count"])),
                "optimum_basin_fraction": abs(float(landscape["optimum_basin_fraction"]) - float(stored["global_optimum_basin_fraction"])),
            }
            for key, value in differences.items():
                maximum_differences[key] = max(maximum_differences[key], value)
            cell_checks[f"{cohort['cohort_id']}::{objective_id}"] = all(value <= 1e-10 for value in differences.values())
            recomputed_passing[objective_id] += int(bool(stored["difficulty_gate_passed"]))
            if objective_id == "threshold_consensus_control":
                threshold_ranges.append(float(scores.max() - scores.min()))
    checks["all_cell_recalculations"] = all(cell_checks.values())
    checks["passing_count_recalculation"] = all(
        recomputed_passing[objective_id] == int(record["passing_cohort_count"])
        for objective_id, record in result["objective_records"].items()
    )
    selected = next(
        (spec["objective_id"] for spec in config["objective_families_in_priority_order"] if recomputed_passing[spec["objective_id"]] >= int(config["difficulty_gate"]["minimum_passing_cohorts_per_objective"])),
        None,
    )
    checks["decision_recalculation"] = result["decision"]["selected_objective_id"] == selected and result["decision"]["candidate_objective_found"] == (selected is not None)
    checks["cross_start_threshold_degeneracy_confirmed"] = all(value <= tolerance for value in threshold_ranges)
    status = "stage36_pparg_consensus_objective_landscape_audit_ok" if all(checks.values()) else "stage36_pparg_consensus_objective_landscape_audit_failed"
    record = {
        "schema_version": "1.0", "status": status,
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "result": {"path": outputs["result_json"].relative_to(root).as_posix(), "sha256": file_sha256(outputs["result_json"])},
        "checks": checks, "cell_checks": cell_checks, "maximum_absolute_differences": maximum_differences,
        "recomputed_passing_cohort_counts": recomputed_passing,
        "threshold_control_objective_ranges": threshold_ranges,
    }
    write_json(outputs["audit_json"], record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage36_pparg_consensus_objective_landscape.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    record = audit(rooted(root, args.config), root)
    print(json.dumps({"status": record["status"], "checks": record["checks"]}, indent=2, sort_keys=True))
    return 0 if record["status"].endswith("_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
