"""Run the Stage36 exact consensus-objective landscape screen."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import descriptor, file_sha256, read_json, rooted, write_csv, write_json
from scripts.run_stage31_pparg_objective_landscape_screen import (
    all_assignments,
    build_cohort,
    cyclic_orders,
    load_inputs,
    successor_and_local_metrics,
)


COMPONENT_KEYS = (
    "mean_pair_same_state8", "mean_pair_same_state16", "mean_pair_same_state32",
    "mean_pair_different_state16", "mean_pair_different_state32", "mean_pair_distance",
    "mean_within_start_centrality", "single_state32", "double_supported_state8",
    "double_supported_state16", "triple_supported_state8", "supported_frame_fraction32",
)


def support_components(labels: np.ndarray, cluster_count: int) -> dict[str, np.ndarray]:
    rows = len(labels)
    counts = np.zeros((rows, cluster_count), dtype=np.uint8)
    row_index = np.repeat(np.arange(rows), labels.shape[1])
    np.add.at(counts, (row_index, labels.reshape(-1)), 1)
    return {
        "single": (counts >= 1).sum(axis=1) / 8.0,
        "double": (counts >= 2).sum(axis=1) / 4.0,
        "triple": (counts >= 3).sum(axis=1) / 2.0,
        "supported_fraction": (counts * (counts >= 2)).sum(axis=1) / 8.0,
    }


def component_arrays(cohort: dict[str, Any], assignments: np.ndarray) -> dict[str, np.ndarray]:
    components = {key: np.empty(len(assignments), dtype=float) for key in COMPONENT_KEYS}
    chunk_size = 4096
    for start in range(0, len(assignments), chunk_size):
        stop = min(len(assignments), start + chunk_size)
        digits = assignments[start:stop]
        selected = digits + 4 * np.arange(8, dtype=int)[None, :]
        pair_distance = np.zeros(len(digits), dtype=float)
        same = {count: np.zeros(len(digits), dtype=float) for count in (8, 16, 32)}
        for left_group in range(8):
            for right_group in range(left_group + 1, 8):
                left = selected[:, left_group]
                right = selected[:, right_group]
                pair_distance += cohort["distance"][left, right]
                for count in same:
                    same[count] += cohort["labels"][count][left] == cohort["labels"][count][right]
        components["mean_pair_distance"][start:stop] = pair_distance / 28.0
        for count in same:
            components[f"mean_pair_same_state{count}"][start:stop] = same[count] / 28.0
        components["mean_pair_different_state16"][start:stop] = 1.0 - same[16] / 28.0
        components["mean_pair_different_state32"][start:stop] = 1.0 - same[32] / 28.0
        components["mean_within_start_centrality"][start:stop] = cohort["centrality"][selected].mean(axis=1)
        support8 = support_components(cohort["labels"][8][selected], 8)
        support16 = support_components(cohort["labels"][16][selected], 16)
        support32 = support_components(cohort["labels"][32][selected], 32)
        components["single_state32"][start:stop] = support32["single"]
        components["double_supported_state8"][start:stop] = support8["double"]
        components["double_supported_state16"][start:stop] = support16["double"]
        components["triple_supported_state8"][start:stop] = support8["triple"]
        components["supported_frame_fraction32"][start:stop] = support32["supported_fraction"]
    return components


def objective_values(objective_id: str, value: dict[str, np.ndarray]) -> np.ndarray:
    if objective_id == "coarse_consensus_fine_diversity":
        return 0.45 * value["mean_pair_same_state8"] + 0.30 * value["mean_pair_different_state32"] + 0.15 * value["mean_pair_distance"] + 0.10 * value["mean_within_start_centrality"]
    if objective_id == "mid_consensus_fine_diversity":
        return 0.45 * value["mean_pair_same_state16"] + 0.30 * value["mean_pair_different_state32"] + 0.15 * value["mean_pair_distance"] + 0.10 * value["mean_within_start_centrality"]
    if objective_id == "frustrated_hierarchical_pair":
        return 0.40 * value["mean_pair_same_state8"] + 0.35 * value["mean_pair_different_state16"] - 0.15 * value["mean_pair_same_state32"] + 0.10 * value["mean_within_start_centrality"]
    if objective_id == "geometry_frustrated_pair":
        return 0.40 * value["mean_pair_same_state8"] + 0.40 * value["mean_pair_distance"] - 0.10 * value["mean_pair_same_state32"] + 0.10 * value["mean_within_start_centrality"]
    if objective_id == "coarse_triple_supported_portfolio":
        return 0.35 * value["triple_supported_state8"] + 0.25 * value["double_supported_state16"] + 0.20 * value["single_state32"] + 0.10 * value["mean_pair_distance"] + 0.10 * value["mean_within_start_centrality"]
    if objective_id == "hierarchical_double_consensus":
        return 0.25 * value["double_supported_state8"] + 0.30 * value["double_supported_state16"] + 0.20 * value["supported_frame_fraction32"] + 0.15 * value["mean_pair_distance"] + 0.10 * value["mean_within_start_centrality"]
    if objective_id == "threshold_consensus_control":
        return 0.50 * value["triple_supported_state8"] + 0.50 * value["double_supported_state16"]
    if objective_id == "smooth_stage30_control":
        return 0.30 * value["mean_within_start_centrality"] + 0.50 * value["mean_pair_distance"] + 0.20 * value["mean_pair_different_state32"]
    raise ValueError(f"unknown Stage36 objective: {objective_id}")


def partial_components(cohort: dict[str, Any], selected: list[int]) -> dict[str, float]:
    if not selected:
        return {key: 0.0 for key in COMPONENT_KEYS}
    pairs = list(itertools.combinations(selected, 2))
    pair_count = len(pairs)
    output = {key: 0.0 for key in COMPONENT_KEYS}
    if pairs:
        output["mean_pair_distance"] = float(np.mean([cohort["distance"][left, right] for left, right in pairs]))
        for count in (8, 16, 32):
            output[f"mean_pair_same_state{count}"] = float(np.mean([cohort["labels"][count][left] == cohort["labels"][count][right] for left, right in pairs]))
        output["mean_pair_different_state16"] = 1.0 - output["mean_pair_same_state16"]
        output["mean_pair_different_state32"] = 1.0 - output["mean_pair_same_state32"]
    output["mean_within_start_centrality"] = float(cohort["centrality"][selected].mean())
    supports = {}
    for count in (8, 16, 32):
        counts = np.bincount(cohort["labels"][count][selected], minlength=count)
        supports[count] = {
            "single": float((counts >= 1).sum() / 8.0),
            "double": float((counts >= 2).sum() / 4.0),
            "triple": float((counts >= 3).sum() / 2.0),
            "supported_fraction": float((counts * (counts >= 2)).sum() / 8.0),
        }
    output["single_state32"] = supports[32]["single"]
    output["double_supported_state8"] = supports[8]["double"]
    output["double_supported_state16"] = supports[16]["double"]
    output["triple_supported_state8"] = supports[8]["triple"]
    output["supported_frame_fraction32"] = supports[32]["supported_fraction"]
    return output


def greedy_state(cohort: dict[str, Any], objective_id: str, order: tuple[int, ...], strides: np.ndarray) -> int:
    selected: list[int] = []
    digits = np.zeros(8, dtype=int)
    for group in order:
        candidates = cohort["groups_local"][group]
        scored = []
        for digit, candidate in enumerate(candidates):
            values = {key: np.asarray([value]) for key, value in partial_components(cohort, [*selected, candidate]).items()}
            scored.append((float(objective_values(objective_id, values)[0]), -digit, candidate, digit))
        _, _, candidate, digit = max(scored)
        selected.append(candidate)
        digits[group] = digit
    return int(np.dot(digits, strides))


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    if read_json(rooted(root, config["inputs"]["stage34_audit"])).get("status") != "stage34_pparg_sparse_fidelity_pareto_audit_ok":
        raise ValueError("Stage34 audit status differs")
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if outputs["result_json"].exists() and not overwrite:
        raise FileExistsError(outputs["result_json"])
    loaded = load_inputs(root, config)
    assignments = all_assignments()
    tolerance = float(config["difficulty_gate"]["objective_tie_tolerance"])
    candidate_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    optimum_rows: list[dict[str, Any]] = []
    records = {spec["objective_id"]: {"cohorts": {}} for spec in config["objective_families_in_priority_order"]}
    for cohort_spec in config["exact_candidate_cohorts"]:
        cohort = build_cohort(cohort_spec, loaded)
        print(json.dumps({"cohort_id": cohort["cohort_id"], "status": "enumerating"}), flush=True)
        for local_index, global_index in enumerate(cohort["global_indices"]):
            frame = loaded["frames"][int(global_index)]
            candidate_rows.append({"cohort_id": cohort["cohort_id"], "local_candidate_index": local_index, "group_index": local_index // 4, "within_group_choice": local_index % 4, "global_frame_index": int(global_index), "frame_id": frame["frame_id"], "conformer_id": frame["conformer_id"]})
        components = component_arrays(cohort, assignments)
        for spec in config["objective_families_in_priority_order"]:
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
            gate = config["difficulty_gate"]
            passed = gap >= float(gate["minimum_absolute_greedy_gap"]) - tolerance and normalized_gap >= float(gate["minimum_normalized_greedy_gap"]) - tolerance and strict_count >= int(gate["minimum_strict_local_optimum_count"]) and float(landscape["optimum_basin_fraction"]) <= float(gate["maximum_global_optimum_basin_fraction"]) + tolerance
            row = {"cohort_id": cohort["cohort_id"], "objective_id": objective_id, "state_count": len(scores), "exact_optimum": exact, "exact_optimum_state_count": len(landscape["optimum_states"]), "strong_greedy_objective": greedy_value, "absolute_greedy_gap": gap, "normalized_greedy_gap": normalized_gap, "objective_minimum": float(scores.min()), "objective_range": value_range, "strict_local_optimum_count": strict_count, "weak_local_optimum_count": int(landscape["weak_local"].sum()), "tie_broken_sink_count": len(landscape["basin_by_endpoint"]), "global_optimum_basin_fraction": float(landscape["optimum_basin_fraction"]), "difficulty_gate_passed": passed, "runtime_seconds": time.perf_counter() - started, "exact_best_state_index": min(landscape["optimum_states"]), "strong_greedy_state_index": greedy_index}
            metric_rows.append(row)
            records[objective_id]["cohorts"][cohort["cohort_id"]] = row
            for endpoint, basin_size in sorted(landscape["basin_by_endpoint"].items(), key=lambda item: (-item[1], item[0])):
                optimum_rows.append({"cohort_id": cohort["cohort_id"], "objective_id": objective_id, "state_index": endpoint, "objective": float(scores[endpoint]), "basin_size": basin_size, "basin_fraction": basin_size / len(scores), "is_global_optimum": endpoint in landscape["optimum_states"], "is_strict_local_optimum": bool(landscape["strict_local"][endpoint]), "digits": "".join(str(int(value)) for value in assignments[endpoint])})
    selected_objective = None
    for spec in config["objective_families_in_priority_order"]:
        objective_id = spec["objective_id"]
        passing = sum(bool(row["difficulty_gate_passed"]) for row in records[objective_id]["cohorts"].values())
        records[objective_id]["passing_cohort_count"] = passing
        records[objective_id]["objective_difficulty_gate_passed"] = passing >= int(config["difficulty_gate"]["minimum_passing_cohorts_per_objective"])
        if selected_objective is None and records[objective_id]["objective_difficulty_gate_passed"]:
            selected_objective = objective_id
    write_csv(outputs["candidate_manifest_csv"], candidate_rows)
    write_csv(outputs["landscape_metrics_csv"], metric_rows)
    write_csv(outputs["local_optima_csv"], optimum_rows)
    decision = {"exact_landscape_screen_complete": len(metric_rows) == 24, "candidate_objective_found": selected_objective is not None, "selected_objective_id": selected_objective, "stage37_sparse_qubo_encoding_authorized": selected_objective is not None, "new_docking_jobs_authorized_by_this_stage": False, "quantum_hardware_authorized": False}
    report = ["# Stage36 PPARG consensus-objective landscape", "", "Each cell exhaustively enumerates 4^8 = 65,536 feasible selections.", "", "| Objective | Passing cohorts | Max greedy gap | Max strict local optima | Min optimum basin | Gate |", "|---|---:|---:|---:|---:|---|"]
    for spec in config["objective_families_in_priority_order"]:
        objective_id = spec["objective_id"]
        rows = list(records[objective_id]["cohorts"].values())
        report.append(f"| {objective_id} | {records[objective_id]['passing_cohort_count']}/3 | {max(float(row['absolute_greedy_gap']) for row in rows):.8g} | {max(int(row['strict_local_optimum_count']) for row in rows)} | {min(float(row['global_optimum_basin_fraction']) for row in rows):.6f} | {'PASS' if records[objective_id]['objective_difficulty_gate_passed'] else 'NO-GO'} |")
    report.extend(["", f"Selected Stage37 objective: **{selected_objective or 'NONE'}**.", "", config["interpretation_boundary"]])
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report) + "\n", encoding="ascii")
    result = {"schema_version": "1.0", "status": "stage36_pparg_consensus_objective_landscape_complete", "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)}, "implementation": descriptor(root, Path(__file__).resolve()), "inputs": {key: descriptor(root, path) for key, path in loaded["paths"].items()}, "input_statistics": {"frame_count": 1200, "cohort_count": 3, "states_per_cohort": 65536}, "objective_records": records, "decision": decision, "data_boundary": {"docking_scores_read": 0, "ligand_labels_read": 0, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0}, "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key not in {"result_json", "audit_json"}}, "interpretation_boundary": config["interpretation_boundary"]}
    write_json(outputs["result_json"], result)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage36_pparg_consensus_objective_landscape.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run(rooted(root, args.config), root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
