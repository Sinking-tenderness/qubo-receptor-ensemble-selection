"""Run the Stage36c held-out consensus-blend landscape replication."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import descriptor, file_sha256, read_json, rooted, write_csv, write_json
from scripts.run_stage31_pparg_objective_landscape_screen import all_assignments, build_cohort, cyclic_orders, load_inputs, successor_and_local_metrics
from scripts.run_stage36_pparg_consensus_objective_landscape import component_arrays, greedy_state as parent_greedy_state, objective_values as parent_objective_values, partial_components
from scripts.run_stage36b_pparg_start_centered_consensus_landscape import start_centered_labels


def objective_values(objective_id: str, values: dict[str, np.ndarray]) -> np.ndarray:
    if objective_id == "consensus_support_blend_v1":
        return (
            0.175 * values["triple_supported_state8"]
            + 0.125 * values["double_supported_state8"]
            + 0.275 * values["double_supported_state16"]
            + 0.10 * values["single_state32"]
            + 0.10 * values["supported_frame_fraction32"]
            + 0.125 * values["mean_pair_distance"]
            + 0.10 * values["mean_within_start_centrality"]
        )
    return parent_objective_values(objective_id, values)


def greedy_state(cohort: dict[str, Any], objective_id: str, order: tuple[int, ...], strides: np.ndarray) -> int:
    selected: list[int] = []
    digits = np.zeros(8, dtype=int)
    for group in order:
        scored = []
        for digit, candidate in enumerate(cohort["groups_local"][group]):
            arrays = {key: np.asarray([value]) for key, value in partial_components(cohort, [*selected, candidate]).items()}
            scored.append((float(objective_values(objective_id, arrays)[0]), -digit, candidate, digit))
        _, _, candidate, digit = max(scored)
        selected.append(candidate)
        digits[group] = digit
    return int(np.dot(digits, strides))


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    stage36b_config_path = rooted(root, config["inputs"]["stage36b_config"])
    stage36b_config = read_json(stage36b_config_path)
    stage36b_result = read_json(rooted(root, config["inputs"]["stage36b_result"]))
    if stage36b_result.get("status") != "stage36b_pparg_start_centered_consensus_landscape_complete":
        raise ValueError("Stage36b result status differs")
    parent_path = rooted(root, stage36b_config["inputs"]["parent_config"])
    parent = read_json(parent_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if outputs["result_json"].exists() and not overwrite:
        raise FileExistsError(outputs["result_json"])
    loaded = load_inputs(root, parent)
    labels, representation_statistics = start_centered_labels(loaded, rooted(root, config["inputs"]["feature_archive"]), stage36b_config)
    loaded["labels"] = labels
    assignments = all_assignments()
    tolerance = float(config["difficulty_gate"]["objective_tie_tolerance"])
    objective_ids = [config["primary_objective"]["objective_id"], *config["diagnostic_controls"]]
    candidate_rows, metric_rows, optimum_rows = [], [], []
    records = {objective_id: {"cohorts": {}} for objective_id in objective_ids}
    for cohort_spec in config["replication_cohorts"]:
        cohort = build_cohort(cohort_spec, loaded)
        print(json.dumps({"cohort_id": cohort["cohort_id"], "status": "heldout_enumeration"}), flush=True)
        for local_index, global_index in enumerate(cohort["global_indices"]):
            frame = loaded["frames"][int(global_index)]
            candidate_rows.append({"cohort_id": cohort["cohort_id"], "local_candidate_index": local_index, "group_index": local_index // 4, "within_group_choice": local_index % 4, "global_frame_index": int(global_index), "frame_id": frame["frame_id"], "conformer_id": frame["conformer_id"]})
        components = component_arrays(cohort, assignments)
        for objective_id in objective_ids:
            started = time.perf_counter()
            scores = objective_values(objective_id, components)
            landscape = successor_and_local_metrics(scores, assignments, tolerance)
            greedy_starts = [greedy_state(cohort, objective_id, order, landscape["strides"]) for order in cyclic_orders()]
            greedy_endpoints = [int(landscape["endpoints"][state]) for state in greedy_starts]
            greedy_index = max(greedy_endpoints, key=lambda state: (float(scores[state]), -state))
            exact, greedy_value = float(landscape["optimum"]), float(scores[greedy_index])
            value_range = float(scores.max() - scores.min())
            gap = exact - greedy_value
            normalized_gap = gap / value_range if value_range else 0.0
            strict_count = int(landscape["strict_local"].sum())
            gate = config["difficulty_gate"]
            passed = gap >= float(gate["minimum_absolute_greedy_gap"]) - tolerance and normalized_gap >= float(gate["minimum_normalized_greedy_gap"]) - tolerance and strict_count >= int(gate["minimum_strict_local_optimum_count"]) and float(landscape["optimum_basin_fraction"]) <= float(gate["maximum_global_optimum_basin_fraction"]) + tolerance
            row = {"cohort_id": cohort["cohort_id"], "objective_id": objective_id, "objective_role": "primary" if objective_id == config["primary_objective"]["objective_id"] else "diagnostic_control", "state_count": len(scores), "exact_optimum": exact, "strong_greedy_objective": greedy_value, "absolute_greedy_gap": gap, "normalized_greedy_gap": normalized_gap, "objective_minimum": float(scores.min()), "objective_range": value_range, "strict_local_optimum_count": strict_count, "weak_local_optimum_count": int(landscape["weak_local"].sum()), "tie_broken_sink_count": len(landscape["basin_by_endpoint"]), "global_optimum_basin_fraction": float(landscape["optimum_basin_fraction"]), "difficulty_gate_passed": passed, "runtime_seconds": time.perf_counter() - started, "exact_best_state_index": min(landscape["optimum_states"]), "strong_greedy_state_index": greedy_index}
            metric_rows.append(row)
            records[objective_id]["cohorts"][cohort["cohort_id"]] = row
            for endpoint, basin_size in sorted(landscape["basin_by_endpoint"].items(), key=lambda item: (-item[1], item[0])):
                optimum_rows.append({"cohort_id": cohort["cohort_id"], "objective_id": objective_id, "state_index": endpoint, "objective": float(scores[endpoint]), "basin_size": basin_size, "basin_fraction": basin_size / len(scores), "is_global_optimum": endpoint in landscape["optimum_states"], "is_strict_local_optimum": bool(landscape["strict_local"][endpoint]), "digits": "".join(str(int(value)) for value in assignments[endpoint])})
    for objective_id in objective_ids:
        records[objective_id]["passing_cohort_count"] = sum(bool(row["difficulty_gate_passed"]) for row in records[objective_id]["cohorts"].values())
    primary_id = config["primary_objective"]["objective_id"]
    primary_pass = records[primary_id]["passing_cohort_count"] >= int(config["difficulty_gate"]["minimum_passing_replication_cohorts"])
    write_csv(outputs["candidate_manifest_csv"], candidate_rows)
    write_csv(outputs["landscape_metrics_csv"], metric_rows)
    write_csv(outputs["local_optima_csv"], optimum_rows)
    decision = {"heldout_replication_complete": len(metric_rows) == 9, "primary_objective_id": primary_id, "primary_passing_cohort_count": records[primary_id]["passing_cohort_count"], "primary_difficulty_replication_passed": primary_pass, "stage37_sparse_qubo_encoding_authorized": primary_pass, "new_docking_jobs_authorized_by_this_stage": False, "quantum_hardware_authorized": False}
    report = ["# Stage36c held-out consensus-blend replication", "", "| Objective | Role | Passing cohorts | Max greedy gap | Max local optima | Min optimum basin |", "|---|---|---:|---:|---:|---:|"]
    for objective_id in objective_ids:
        rows = list(records[objective_id]["cohorts"].values())
        report.append(f"| {objective_id} | {'primary' if objective_id == primary_id else 'control'} | {records[objective_id]['passing_cohort_count']}/3 | {max(float(row['absolute_greedy_gap']) for row in rows):.8g} | {max(int(row['strict_local_optimum_count']) for row in rows)} | {min(float(row['global_optimum_basin_fraction']) for row in rows):.6f} |")
    report.extend(["", f"Primary replication gate: **{'PASS' if primary_pass else 'NO-GO'}**.", "", config["interpretation_boundary"]])
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report) + "\n", encoding="ascii")
    result = {"schema_version": "1.0", "status": "stage36c_pparg_consensus_blend_replication_complete", "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)}, "implementation": descriptor(root, Path(__file__).resolve()), "inputs": {key: descriptor(root, rooted(root, value)) for key, value in config["inputs"].items()}, "representation_statistics": representation_statistics, "objective_records": records, "decision": decision, "data_boundary": {"docking_scores_read": 0, "ligand_labels_read": 0, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0}, "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key not in {"result_json", "audit_json"}}, "interpretation_boundary": config["interpretation_boundary"]}
    write_json(outputs["result_json"], result)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage36c_pparg_consensus_blend_replication.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run(rooted(root, args.config), root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
