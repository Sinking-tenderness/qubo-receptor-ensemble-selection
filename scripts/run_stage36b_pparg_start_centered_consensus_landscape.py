"""Run Stage36b with start-centered dynamic pocket-state labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import descriptor, file_sha256, read_json, rooted, write_csv, write_json
from scripts.run_stage31_pparg_objective_landscape_screen import (
    all_assignments, build_cohort, canonical_labels, cyclic_orders, load_inputs, successor_and_local_metrics,
)
from scripts.run_stage36_pparg_consensus_objective_landscape import component_arrays, greedy_state, objective_values


def start_centered_labels(loaded: dict[str, Any], feature_path: Path, config: dict[str, Any]) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    archive = np.load(feature_path)
    features = archive["standardized_features"].astype(float)
    if features.shape != (1200, 870):
        raise ValueError("Stage36b feature dimensions differ")
    residual = np.empty_like(features)
    for _, indices in sorted(loaded["by_start"].items()):
        local = np.asarray(indices, dtype=int)
        residual[local] = features[local] - features[local].mean(axis=0)
    standard_deviation = residual.std(axis=0)
    keep = standard_deviation > float(config["state_representation"]["minimum_residual_standard_deviation"])
    scaled = residual[:, keep] / standard_deviation[keep]
    hierarchy = linkage(scaled, method="ward", optimal_ordering=False)
    labels = {
        int(count): canonical_labels(fcluster(hierarchy, int(count), criterion="maxclust"))
        for count in config["state_representation"]["cluster_counts"]
    }
    statistics = {
        "raw_feature_count": features.shape[1], "retained_residual_feature_count": int(keep.sum()),
        "maximum_absolute_start_residual_mean": max(float(np.abs(residual[np.asarray(indices, dtype=int)].mean(axis=0)).max()) for indices in loaded["by_start"].values()),
    }
    return labels, statistics


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    parent_path = rooted(root, config["inputs"]["parent_config"])
    parent = read_json(parent_path)
    if read_json(rooted(root, config["inputs"]["stage36_audit"])).get("status") != "stage36_pparg_consensus_objective_landscape_audit_ok":
        raise ValueError("Stage36 audit status differs")
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if outputs["result_json"].exists() and not overwrite:
        raise FileExistsError(outputs["result_json"])
    loaded = load_inputs(root, parent)
    labels, representation_statistics = start_centered_labels(loaded, rooted(root, config["inputs"]["feature_archive"]), config)
    loaded["labels"] = labels
    state_rows = []
    state_statistics = {}
    frame_starts = np.asarray([int(row["start_index"]) for row in loaded["frames"]], dtype=int)
    for count, values in sorted(labels.items()):
        shared_states = 0
        shared_frames = 0
        for state in range(count):
            members = np.flatnonzero(values == state)
            starts = np.unique(frame_starts[members])
            shared = len(starts) >= 2
            shared_states += int(shared)
            shared_frames += len(members) if shared else 0
            state_rows.append({"cluster_count": count, "state_id": state, "frame_count": len(members), "represented_start_count": len(starts), "cross_start_shared": shared})
        state_statistics[str(count)] = {"state_count": count, "cross_start_shared_state_count": shared_states, "cross_start_shared_frame_fraction": shared_frames / 1200.0}
    assignments = all_assignments()
    tolerance = float(parent["difficulty_gate"]["objective_tie_tolerance"])
    candidate_rows, metric_rows, optimum_rows = [], [], []
    records = {spec["objective_id"]: {"cohorts": {}} for spec in parent["objective_families_in_priority_order"]}
    for cohort_spec in parent["exact_candidate_cohorts"]:
        cohort = build_cohort(cohort_spec, loaded)
        print(json.dumps({"cohort_id": cohort["cohort_id"], "status": "enumerating_start_centered"}), flush=True)
        for local_index, global_index in enumerate(cohort["global_indices"]):
            frame = loaded["frames"][int(global_index)]
            candidate_rows.append({"cohort_id": cohort["cohort_id"], "local_candidate_index": local_index, "group_index": local_index // 4, "within_group_choice": local_index % 4, "global_frame_index": int(global_index), "frame_id": frame["frame_id"], "conformer_id": frame["conformer_id"]})
        components = component_arrays(cohort, assignments)
        for spec in parent["objective_families_in_priority_order"]:
            objective_id = spec["objective_id"]
            started = time.perf_counter()
            scores = objective_values(objective_id, components)
            landscape = successor_and_local_metrics(scores, assignments, tolerance)
            greedy_starts = [greedy_state(cohort, objective_id, order, landscape["strides"]) for order in cyclic_orders()]
            greedy_endpoints = [int(landscape["endpoints"][state]) for state in greedy_starts]
            greedy_index = max(greedy_endpoints, key=lambda state: (float(scores[state]), -state))
            exact = float(landscape["optimum"])
            greedy_value = float(scores[greedy_index])
            value_range = float(scores.max() - scores.min())
            gap = exact - greedy_value
            normalized_gap = gap / value_range if value_range else 0.0
            strict_count = int(landscape["strict_local"].sum())
            gate = parent["difficulty_gate"]
            passed = gap >= float(gate["minimum_absolute_greedy_gap"]) - tolerance and normalized_gap >= float(gate["minimum_normalized_greedy_gap"]) - tolerance and strict_count >= int(gate["minimum_strict_local_optimum_count"]) and float(landscape["optimum_basin_fraction"]) <= float(gate["maximum_global_optimum_basin_fraction"]) + tolerance
            row = {"cohort_id": cohort["cohort_id"], "objective_id": objective_id, "state_count": len(scores), "exact_optimum": exact, "exact_optimum_state_count": len(landscape["optimum_states"]), "strong_greedy_objective": greedy_value, "absolute_greedy_gap": gap, "normalized_greedy_gap": normalized_gap, "objective_minimum": float(scores.min()), "objective_range": value_range, "strict_local_optimum_count": strict_count, "weak_local_optimum_count": int(landscape["weak_local"].sum()), "tie_broken_sink_count": len(landscape["basin_by_endpoint"]), "global_optimum_basin_fraction": float(landscape["optimum_basin_fraction"]), "difficulty_gate_passed": passed, "runtime_seconds": time.perf_counter() - started, "exact_best_state_index": min(landscape["optimum_states"]), "strong_greedy_state_index": greedy_index}
            metric_rows.append(row)
            records[objective_id]["cohorts"][cohort["cohort_id"]] = row
            for endpoint, basin_size in sorted(landscape["basin_by_endpoint"].items(), key=lambda item: (-item[1], item[0])):
                optimum_rows.append({"cohort_id": cohort["cohort_id"], "objective_id": objective_id, "state_index": endpoint, "objective": float(scores[endpoint]), "basin_size": basin_size, "basin_fraction": basin_size / len(scores), "is_global_optimum": endpoint in landscape["optimum_states"], "is_strict_local_optimum": bool(landscape["strict_local"][endpoint]), "digits": "".join(str(int(value)) for value in assignments[endpoint])})
    selected_objective = None
    for spec in parent["objective_families_in_priority_order"]:
        objective_id = spec["objective_id"]
        passing = sum(bool(row["difficulty_gate_passed"]) for row in records[objective_id]["cohorts"].values())
        records[objective_id]["passing_cohort_count"] = passing
        records[objective_id]["objective_difficulty_gate_passed"] = passing >= int(parent["difficulty_gate"]["minimum_passing_cohorts_per_objective"])
        if selected_objective is None and records[objective_id]["objective_difficulty_gate_passed"]:
            selected_objective = objective_id
    write_csv(outputs["state_manifest_csv"], state_rows)
    write_csv(outputs["candidate_manifest_csv"], candidate_rows)
    write_csv(outputs["landscape_metrics_csv"], metric_rows)
    write_csv(outputs["local_optima_csv"], optimum_rows)
    decision = {"start_centered_state_screen_complete": len(metric_rows) == 24, "candidate_objective_found": selected_objective is not None, "selected_objective_id": selected_objective, "stage37_sparse_qubo_encoding_authorized": selected_objective is not None, "new_docking_jobs_authorized_by_this_stage": False, "quantum_hardware_authorized": False}
    report = ["# Stage36b start-centered consensus landscape", "", "| Objective | Passing cohorts | Max greedy gap | Max strict local optima | Min optimum basin | Gate |", "|---|---:|---:|---:|---:|---|"]
    for spec in parent["objective_families_in_priority_order"]:
        objective_id = spec["objective_id"]
        rows = list(records[objective_id]["cohorts"].values())
        report.append(f"| {objective_id} | {records[objective_id]['passing_cohort_count']}/3 | {max(float(row['absolute_greedy_gap']) for row in rows):.8g} | {max(int(row['strict_local_optimum_count']) for row in rows)} | {min(float(row['global_optimum_basin_fraction']) for row in rows):.6f} | {'PASS' if records[objective_id]['objective_difficulty_gate_passed'] else 'NO-GO'} |")
    report.extend(["", f"Selected Stage37 objective: **{selected_objective or 'NONE'}**.", "", config["interpretation_boundary"]])
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report) + "\n", encoding="ascii")
    result = {"schema_version": "1.0", "status": "stage36b_pparg_start_centered_consensus_landscape_complete", "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)}, "parent_config": descriptor(root, parent_path), "implementation": descriptor(root, Path(__file__).resolve()), "inputs": {key: descriptor(root, rooted(root, value)) for key, value in config["inputs"].items()}, "representation_statistics": representation_statistics, "state_statistics": state_statistics, "objective_records": records, "decision": decision, "data_boundary": {"docking_scores_read": 0, "ligand_labels_read": 0, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0}, "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key not in {"result_json", "audit_json"}}, "interpretation_boundary": config["interpretation_boundary"]}
    write_json(outputs["result_json"], result)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage36b_pparg_start_centered_consensus_landscape.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run(rooted(root, args.config), root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
