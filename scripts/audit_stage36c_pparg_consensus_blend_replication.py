"""Audit the Stage36c held-out consensus-blend replication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import file_sha256, read_json, rooted, write_json
from scripts.run_stage31_pparg_objective_landscape_screen import all_assignments, build_cohort, cyclic_orders, load_inputs, successor_and_local_metrics
from scripts.run_stage36_pparg_consensus_objective_landscape import component_arrays
from scripts.run_stage36b_pparg_start_centered_consensus_landscape import start_centered_labels
from scripts.run_stage36c_pparg_consensus_blend_replication import greedy_state, objective_values


def audit(config_path: Path, root: Path) -> dict:
    config = read_json(config_path)
    stage36b_config = read_json(rooted(root, config["inputs"]["stage36b_config"]))
    parent = read_json(rooted(root, stage36b_config["inputs"]["parent_config"]))
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    result = read_json(outputs["result_json"])
    loaded = load_inputs(root, parent)
    labels, representation = start_centered_labels(loaded, rooted(root, config["inputs"]["feature_archive"]), stage36b_config)
    loaded["labels"] = labels
    assignments = all_assignments()
    tolerance = float(config["difficulty_gate"]["objective_tie_tolerance"])
    primary_id = config["primary_objective"]["objective_id"]
    objective_ids = [primary_id, *config["diagnostic_controls"]]
    checks = {
        "result_status": result.get("status") == "stage36c_pparg_consensus_blend_replication_complete",
        "replication_complete": result["decision"]["heldout_replication_complete"] is True,
        "data_boundary_zero": all(int(value) == 0 for value in result["data_boundary"].values()),
        "representation_recalculation": representation == result["representation_statistics"],
    }
    cell_checks = {}
    maximum = {"exact": 0.0, "greedy": 0.0, "strict_count": 0, "basin": 0.0}
    passing = {objective_id: 0 for objective_id in objective_ids}
    for cohort_spec in config["replication_cohorts"]:
        cohort = build_cohort(cohort_spec, loaded)
        components = component_arrays(cohort, assignments)
        for objective_id in objective_ids:
            scores = objective_values(objective_id, components)
            landscape = successor_and_local_metrics(scores, assignments, tolerance)
            starts = [greedy_state(cohort, objective_id, order, landscape["strides"]) for order in cyclic_orders()]
            endpoints = [int(landscape["endpoints"][state]) for state in starts]
            greedy_index = max(endpoints, key=lambda state: (float(scores[state]), -state))
            stored = result["objective_records"][objective_id]["cohorts"][cohort["cohort_id"]]
            differences = {
                "exact": abs(float(scores.max()) - float(stored["exact_optimum"])),
                "greedy": abs(float(scores[greedy_index]) - float(stored["strong_greedy_objective"])),
                "strict_count": abs(int(landscape["strict_local"].sum()) - int(stored["strict_local_optimum_count"])),
                "basin": abs(float(landscape["optimum_basin_fraction"]) - float(stored["global_optimum_basin_fraction"])),
            }
            for key, value in differences.items():
                maximum[key] = max(maximum[key], value)
            cell_checks[f"{cohort['cohort_id']}::{objective_id}"] = all(value <= 1e-10 for value in differences.values())
            passing[objective_id] += int(bool(stored["difficulty_gate_passed"]))
    checks["all_exact_and_greedy_recalculations"] = all(cell_checks.values())
    primary_pass = passing[primary_id] >= int(config["difficulty_gate"]["minimum_passing_replication_cohorts"])
    checks["decision_recalculation"] = result["decision"]["primary_passing_cohort_count"] == passing[primary_id] and result["decision"]["primary_difficulty_replication_passed"] == primary_pass
    checks["no_posthoc_control_selection"] = result["decision"]["primary_objective_id"] == primary_id
    status = "stage36c_pparg_consensus_blend_replication_audit_ok" if all(checks.values()) else "stage36c_pparg_consensus_blend_replication_audit_failed"
    record = {"schema_version": "1.0", "status": status, "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)}, "result": {"path": outputs["result_json"].relative_to(root).as_posix(), "sha256": file_sha256(outputs["result_json"])}, "checks": checks, "cell_checks": cell_checks, "maximum_absolute_differences": maximum, "recomputed_passing_counts": passing}
    write_json(outputs["audit_json"], record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage36c_pparg_consensus_blend_replication.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    record = audit(rooted(root, args.config), root)
    print(json.dumps({"status": record["status"], "checks": record["checks"]}, indent=2, sort_keys=True))
    return 0 if record["status"].endswith("_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
