"""Run the Stage31 exact PPARG objective-landscape difficulty screen."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import descriptor, file_sha256, read_json, rooted, write_csv, write_json
from scripts.run_stage29_pparg_md_qubo_scaling import temporal_maximin_order


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_labels(raw: np.ndarray) -> np.ndarray:
    unique = sorted(np.unique(raw), key=lambda label: int(np.flatnonzero(raw == label)[0]))
    mapping = {int(label): index for index, label in enumerate(unique)}
    return np.asarray([mapping[int(label)] for label in raw], dtype=int)


def load_inputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = {key: rooted(root, value) for key, value in config["inputs"].items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if read_json(paths["stage30_audit"]).get("status") != "stage30_pparg_group_balanced_state_qubo_audit_ok":
        raise ValueError("Stage30 audit status differs")
    frames = read_rows(paths["frame_manifest"])
    features = np.load(paths["feature_archive"])
    distance_archive = np.load(paths["distance_archive"])
    frame_ids = np.asarray([row["frame_id"] for row in frames])
    if len(frames) != 1200 or not np.array_equal(features["frame_ids"], frame_ids) or not np.array_equal(distance_archive["frame_ids"], frame_ids):
        raise ValueError("Stage31 frame coverage differs")
    standardized = features["standardized_features"].astype(float)
    condensed = distance_archive["condensed_distances"].astype(float)
    if standardized.shape != (1200, 870) or condensed.shape != (719400,):
        raise ValueError("Stage31 input dimensions differ")
    distances = squareform(condensed)
    maximum = float(distances.max())
    distances /= maximum
    by_start: dict[int, list[int]] = {}
    for index, row in enumerate(frames):
        by_start.setdefault(int(row["start_index"]), []).append(index)
    if sorted(len(values) for values in by_start.values()) != [150] * 8:
        raise ValueError("Stage31 requires eight starts with 150 frames each")
    ordered: dict[int, list[int]] = {}
    centrality = np.zeros(1200, dtype=float)
    for start, indices in sorted(by_start.items()):
        by_local = {int(frames[index]["local_frame_index"]): index for index in indices}
        ordered[start] = [by_local[value] for value in temporal_maximin_order(by_local)]
        local = np.asarray(indices, dtype=int)
        centrality[local] = 1.0 - distances[np.ix_(local, local)].sum(axis=1) / 149
    hierarchy = linkage(condensed, method="ward", optimal_ordering=False)
    labels = {
        int(count): canonical_labels(fcluster(hierarchy, int(count), criterion="maxclust"))
        for count in config["coverage"]["smooth_control_ward_cluster_counts"]
    }
    off_diagonal = condensed / maximum
    quantiles = {
        float(value): float(np.quantile(off_diagonal, float(value)))
        for value in config["coverage"]["global_distance_quantiles"]
    }
    return {
        "paths": paths,
        "frames": frames,
        "distances": distances,
        "distance_maximum": maximum,
        "ordered": ordered,
        "by_start": by_start,
        "centrality": centrality,
        "labels": labels,
        "quantiles": quantiles,
    }


def build_cohort(spec: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    ranks = [int(value) for value in spec["within_start_temporal_maximin_ranks"]]
    if len(ranks) != 4 or len(set(ranks)) != 4:
        raise ValueError("Stage31 cohorts require four distinct ranks")
    groups_global = [tuple(loaded["ordered"][start][rank] for rank in ranks) for start in sorted(loaded["ordered"])]
    global_indices = np.asarray([value for group in groups_global for value in group], dtype=int)
    groups_local = [tuple(range(group * 4, (group + 1) * 4)) for group in range(8)]
    incidence = {
        quantile: loaded["distances"][np.ix_(global_indices, np.arange(1200))] <= threshold + 1e-15
        for quantile, threshold in loaded["quantiles"].items()
    }
    labels = {count: values[global_indices] for count, values in loaded["labels"].items()}
    return {
        "cohort_id": spec["cohort_id"],
        "ranks": tuple(ranks),
        "global_indices": global_indices,
        "groups_local": groups_local,
        "incidence": incidence,
        "distance": loaded["distances"][np.ix_(global_indices, global_indices)],
        "centrality": loaded["centrality"][global_indices],
        "labels": labels,
        "frames": loaded["frames"],
        "by_start": loaded["by_start"],
    }


def all_assignments() -> np.ndarray:
    return np.indices((4,) * 8, dtype=np.int8).reshape(8, -1).T


def component_arrays(cohort: dict[str, Any], assignments: np.ndarray) -> dict[str, np.ndarray]:
    count = len(assignments)
    components = {
        "single_q05": np.empty(count, dtype=float),
        "single_q10": np.empty(count, dtype=float),
        "single_q20": np.empty(count, dtype=float),
        "double_q10": np.empty(count, dtype=float),
        "worst_start_single_q10": np.empty(count, dtype=float),
        "worst_two_start_single_q10": np.empty(count, dtype=float),
        "mean_pair_distance": np.empty(count, dtype=float),
        "mean_within_start_centrality": np.empty(count, dtype=float),
        "mean_multiscale_state_separation": np.empty(count, dtype=float),
    }
    reference_groups = [np.asarray(cohort["by_start"][start], dtype=int) for start in sorted(cohort["by_start"])]
    chunk_size = 4096
    for start_index in range(0, count, chunk_size):
        stop = min(count, start_index + chunk_size)
        digits = assignments[start_index:stop]
        selected = digits + 4 * np.arange(8, dtype=int)[None, :]
        coverage_by_quantile: dict[float, np.ndarray] = {}
        for quantile, incidence in cohort["incidence"].items():
            coverage = np.zeros((len(digits), 1200), dtype=np.uint8)
            for group in range(8):
                coverage += incidence[selected[:, group]]
            coverage_by_quantile[quantile] = coverage
        for quantile, key in ((0.05, "single_q05"), (0.10, "single_q10"), (0.20, "single_q20")):
            components[key][start_index:stop] = (coverage_by_quantile[quantile] >= 1).mean(axis=1)
        q10 = coverage_by_quantile[0.10]
        components["double_q10"][start_index:stop] = (q10 >= 2).mean(axis=1)
        per_start = np.stack([(q10[:, indices] >= 1).mean(axis=1) for indices in reference_groups], axis=1)
        ordered = np.sort(per_start, axis=1)
        components["worst_start_single_q10"][start_index:stop] = ordered[:, 0]
        components["worst_two_start_single_q10"][start_index:stop] = ordered[:, :2].mean(axis=1)
        pair_sum = np.zeros(len(digits), dtype=float)
        state_sum = np.zeros(len(digits), dtype=float)
        for first in range(8):
            for second in range(first + 1, 8):
                left, right = selected[:, first], selected[:, second]
                pair_sum += cohort["distance"][left, right]
                state_sum += np.mean(
                    np.stack([cohort["labels"][count][left] != cohort["labels"][count][right] for count in sorted(cohort["labels"])], axis=1),
                    axis=1,
                )
        components["mean_pair_distance"][start_index:stop] = pair_sum / 28
        components["mean_multiscale_state_separation"][start_index:stop] = state_sum / 28
        components["mean_within_start_centrality"][start_index:stop] = cohort["centrality"][selected].mean(axis=1)
    return components


def objective_values(objective_id: str, values: dict[str, np.ndarray]) -> np.ndarray:
    if objective_id == "robust_coverage_pair":
        return 0.15 * values["single_q05"] + 0.15 * values["single_q10"] + 0.25 * values["double_q10"] + 0.30 * values["worst_two_start_single_q10"] + 0.15 * values["mean_pair_distance"]
    if objective_id == "multiscale_robust_coverage":
        return 0.20 * values["single_q05"] + 0.20 * values["single_q10"] + 0.20 * values["single_q20"] + 0.20 * values["double_q10"] + 0.20 * values["worst_start_single_q10"]
    if objective_id == "worst_two_single_double":
        return 0.30 * values["single_q10"] + 0.30 * values["double_q10"] + 0.40 * values["worst_two_start_single_q10"]
    if objective_id == "worst_start_single":
        return values["worst_start_single_q10"].copy()
    if objective_id == "global_single_double":
        return 0.25 * values["single_q10"] + 0.75 * values["double_q10"]
    if objective_id == "smooth_pair_control":
        return 0.30 * values["mean_within_start_centrality"] + 0.50 * values["mean_pair_distance"] + 0.20 * values["mean_multiscale_state_separation"]
    raise ValueError(f"unknown Stage31 objective: {objective_id}")


def partial_components(cohort: dict[str, Any], selected: list[int]) -> dict[str, float]:
    if not selected:
        return {key: 0.0 for key in (
            "single_q05", "single_q10", "single_q20", "double_q10",
            "worst_start_single_q10", "worst_two_start_single_q10", "mean_pair_distance",
            "mean_within_start_centrality", "mean_multiscale_state_separation",
        )}
    coverage = {
        quantile: cohort["incidence"][quantile][selected].sum(axis=0)
        for quantile in cohort["incidence"]
    }
    per_start = sorted(float((coverage[0.10][indices] >= 1).mean()) for _, indices in sorted(cohort["by_start"].items()))
    pairs = list(itertools.combinations(selected, 2))
    pair_distance = 0.0 if not pairs else float(np.mean([cohort["distance"][first, second] for first, second in pairs]))
    state_separation = 0.0 if not pairs else float(np.mean([
        np.mean([cohort["labels"][count][first] != cohort["labels"][count][second] for count in sorted(cohort["labels"])])
        for first, second in pairs
    ]))
    return {
        "single_q05": float((coverage[0.05] >= 1).mean()),
        "single_q10": float((coverage[0.10] >= 1).mean()),
        "single_q20": float((coverage[0.20] >= 1).mean()),
        "double_q10": float((coverage[0.10] >= 2).mean()),
        "worst_start_single_q10": per_start[0],
        "worst_two_start_single_q10": float(np.mean(per_start[:2])),
        "mean_pair_distance": pair_distance,
        "mean_within_start_centrality": float(cohort["centrality"][selected].mean()),
        "mean_multiscale_state_separation": state_separation,
    }


def scalar_objective(objective_id: str, components: dict[str, float]) -> float:
    arrays = {key: np.asarray([value], dtype=float) for key, value in components.items()}
    return float(objective_values(objective_id, arrays)[0])


def greedy_state(cohort: dict[str, Any], objective_id: str, order: tuple[int, ...], strides: np.ndarray) -> int:
    selected: list[int] = []
    digits = np.zeros(8, dtype=int)
    for group in order:
        candidates = cohort["groups_local"][group]
        scored = []
        for digit, candidate in enumerate(candidates):
            value = scalar_objective(objective_id, partial_components(cohort, [*selected, candidate]))
            scored.append((value, -digit, candidate, digit))
        _, _, candidate, digit = max(scored)
        selected.append(candidate)
        digits[group] = digit
    return int(np.dot(digits, strides))


def successor_and_local_metrics(scores: np.ndarray, assignments: np.ndarray, tolerance: float) -> dict[str, Any]:
    state_count = len(scores)
    strides = np.asarray([4 ** (7 - group) for group in range(8)], dtype=int)
    indices = np.arange(state_count, dtype=int)
    successor = indices.copy()
    best_score = scores.copy()
    strict_local = np.ones(state_count, dtype=bool)
    weak_local = np.ones(state_count, dtype=bool)
    for group in range(8):
        current_digit = assignments[:, group].astype(int)
        for alternative in range(4):
            neighbor = indices + (alternative - current_digit) * strides[group]
            different = alternative != current_digit
            neighbor_score = scores[neighbor]
            weak_local &= (~different) | (neighbor_score <= scores + tolerance)
            strict_local &= (~different) | (neighbor_score < scores - tolerance)
            improve = neighbor_score > best_score + tolerance
            tie = np.isclose(neighbor_score, best_score, atol=tolerance, rtol=0.0) & (neighbor < successor)
            update = different & (improve | tie)
            successor[update] = neighbor[update]
            best_score[update] = neighbor_score[update]
    endpoints = np.full(state_count, -1, dtype=int)
    for start in range(state_count):
        if endpoints[start] >= 0:
            continue
        path = []
        value = start
        while endpoints[value] < 0 and successor[value] != value:
            path.append(value)
            value = int(successor[value])
        endpoint = int(endpoints[value]) if endpoints[value] >= 0 else value
        endpoints[value] = endpoint
        for member in reversed(path):
            endpoints[member] = endpoint
    unique, basin_sizes = np.unique(endpoints, return_counts=True)
    basin_by_endpoint = {int(endpoint): int(size) for endpoint, size in zip(unique, basin_sizes)}
    optimum = float(scores.max())
    optimum_states = set(np.flatnonzero(np.isclose(scores, optimum, atol=tolerance, rtol=0.0)).tolist())
    optimum_basin = sum(size for endpoint, size in basin_by_endpoint.items() if endpoint in optimum_states) / state_count
    return {
        "successor": successor,
        "endpoints": endpoints,
        "basin_by_endpoint": basin_by_endpoint,
        "weak_local": weak_local,
        "strict_local": strict_local,
        "optimum": optimum,
        "optimum_states": optimum_states,
        "optimum_basin_fraction": optimum_basin,
        "strides": strides,
    }


def cyclic_orders() -> list[tuple[int, ...]]:
    forward = tuple(range(8))
    reverse = tuple(reversed(forward))
    return [base[offset:] + base[:offset] for base in (forward, reverse) for offset in range(8)]


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if outputs["result_json"].exists() and not overwrite:
        raise FileExistsError(f"result exists: {outputs['result_json']}; pass --overwrite")
    loaded = load_inputs(root, config)
    assignments = all_assignments()
    if len(assignments) != int(config["landscape"]["state_count_per_cohort"]):
        raise ValueError("Stage31 enumerated state count differs")
    objective_specs = config["objective_families_in_priority_order"]
    tolerance = float(config["difficulty_gate"]["objective_tie_tolerance"])
    candidate_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    optimum_rows: list[dict[str, Any]] = []
    records: dict[str, Any] = {spec["objective_id"]: {"cohorts": {}} for spec in objective_specs}
    for cohort_spec in config["exact_candidate_cohorts"]:
        cohort = build_cohort(cohort_spec, loaded)
        print(json.dumps({"cohort_id": cohort["cohort_id"], "status": "enumerating"}), flush=True)
        for local_index, global_index in enumerate(cohort["global_indices"]):
            frame = loaded["frames"][int(global_index)]
            candidate_rows.append({
                "cohort_id": cohort["cohort_id"],
                "local_candidate_index": local_index,
                "group_index": local_index // 4,
                "within_group_choice": local_index % 4,
                "temporal_maximin_rank": cohort["ranks"][local_index % 4],
                "global_frame_index": int(global_index),
                "frame_id": frame["frame_id"],
                "conformer_id": frame["conformer_id"],
                "local_frame_index": frame["local_frame_index"],
                "time_ps": frame["time_ps"],
            })
        components = component_arrays(cohort, assignments)
        for spec in objective_specs:
            objective_id = spec["objective_id"]
            started = time.perf_counter()
            scores = objective_values(objective_id, components)
            landscape = successor_and_local_metrics(scores, assignments, tolerance)
            greedy_starts = [greedy_state(cohort, objective_id, order, landscape["strides"]) for order in cyclic_orders()]
            greedy_endpoints = [int(landscape["endpoints"][state]) for state in greedy_starts]
            greedy_state_index = max(greedy_endpoints, key=lambda state: (float(scores[state]), -state))
            exact = float(landscape["optimum"])
            greedy_value = float(scores[greedy_state_index])
            value_range = float(scores.max() - scores.min())
            absolute_gap = exact - greedy_value
            normalized_gap = absolute_gap / value_range if value_range > 0 else 0.0
            strict_count = int(landscape["strict_local"].sum())
            weak_count = int(landscape["weak_local"].sum())
            sink_count = len(landscape["basin_by_endpoint"])
            gate = config["difficulty_gate"]
            passed = (
                absolute_gap >= float(gate["minimum_absolute_greedy_gap"]) - tolerance
                and normalized_gap >= float(gate["minimum_normalized_greedy_gap"]) - tolerance
                and strict_count >= int(gate["minimum_strict_local_optimum_count"])
                and float(landscape["optimum_basin_fraction"]) <= float(gate["maximum_global_optimum_basin_fraction"]) + tolerance
            )
            row = {
                "cohort_id": cohort["cohort_id"],
                "objective_id": objective_id,
                "state_count": len(scores),
                "exact_optimum": exact,
                "exact_optimum_state_count": len(landscape["optimum_states"]),
                "strong_greedy_objective": greedy_value,
                "absolute_greedy_gap": absolute_gap,
                "normalized_greedy_gap": normalized_gap,
                "objective_minimum": float(scores.min()),
                "objective_range": value_range,
                "strict_local_optimum_count": strict_count,
                "weak_local_optimum_count": weak_count,
                "tie_broken_sink_count": sink_count,
                "global_optimum_basin_fraction": float(landscape["optimum_basin_fraction"]),
                "difficulty_gate_passed": passed,
                "runtime_seconds": time.perf_counter() - started,
                "exact_best_state_index": min(landscape["optimum_states"]),
                "strong_greedy_state_index": greedy_state_index,
            }
            metric_rows.append(row)
            records[objective_id]["cohorts"][cohort["cohort_id"]] = row
            for endpoint, basin_size in sorted(landscape["basin_by_endpoint"].items(), key=lambda item: (-item[1], item[0])):
                optimum_rows.append({
                    "cohort_id": cohort["cohort_id"],
                    "objective_id": objective_id,
                    "state_index": endpoint,
                    "objective": float(scores[endpoint]),
                    "basin_size": basin_size,
                    "basin_fraction": basin_size / len(scores),
                    "is_global_optimum": endpoint in landscape["optimum_states"],
                    "is_strict_local_optimum": bool(landscape["strict_local"][endpoint]),
                    "digits": "".join(str(int(value)) for value in assignments[endpoint]),
                })
    selected_objective = None
    for spec in objective_specs:
        objective_id = spec["objective_id"]
        passing = sum(bool(row["difficulty_gate_passed"]) for row in records[objective_id]["cohorts"].values())
        records[objective_id]["passing_cohort_count"] = passing
        records[objective_id]["objective_difficulty_gate_passed"] = passing >= int(config["difficulty_gate"]["minimum_passing_cohorts_per_objective"])
        if selected_objective is None and records[objective_id]["objective_difficulty_gate_passed"]:
            selected_objective = objective_id
    write_csv(outputs["candidate_manifest_csv"], candidate_rows)
    write_csv(outputs["landscape_metrics_csv"], metric_rows)
    write_csv(outputs["local_optima_csv"], optimum_rows)
    decision = {
        "exact_landscape_screen_complete": len(metric_rows) == len(config["exact_candidate_cohorts"]) * len(objective_specs),
        "candidate_objective_found": selected_objective is not None,
        "selected_objective_id": selected_objective,
        "stage32_qubo_encoding_authorized": selected_objective is not None,
        "new_docking_jobs_authorized_by_this_stage": False,
        "quantum_hardware_authorized": False,
    }
    report = [
        "# Stage 31: PPARG objective-landscape difficulty screen",
        "",
        "Each cell exhaustively enumerates 4^8 = 65,536 feasible start-balanced selections.",
        "",
        "| Objective | Passing cohorts | Max greedy gap | Max strict local optima | Min optimum basin | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for spec in objective_specs:
        objective_id = spec["objective_id"]
        local = list(records[objective_id]["cohorts"].values())
        report.append(f"| {objective_id} | {records[objective_id]['passing_cohort_count']}/3 | {max(float(row['absolute_greedy_gap']) for row in local):.8g} | {max(int(row['strict_local_optimum_count']) for row in local)} | {min(float(row['global_optimum_basin_fraction']) for row in local):.6f} | {'PASS' if records[objective_id]['objective_difficulty_gate_passed'] else 'NO-GO'} |")
    report += [
        "",
        f"Selected Stage32 objective: **{selected_objective or 'NONE'}**.",
        f"Stage32 QUBO encoding authorization: **{'PASS' if decision['stage32_qubo_encoding_authorized'] else 'NO-GO'}**.",
        "",
        "No docking scores, ligand labels, validation/test rows, new docking jobs, or quantum-hardware outputs were used.",
    ]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report) + "\n", encoding="ascii")
    result = {
        "schema_version": "1.0",
        "status": "stage31_pparg_objective_landscape_screen_complete",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "implementation": descriptor(root, Path(__file__).resolve()),
        "inputs": {key: descriptor(root, path) for key, path in loaded["paths"].items()},
        "input_statistics": {
            "frame_count": len(loaded["frames"]),
            "distance_maximum_before_normalization": loaded["distance_maximum"],
            "normalized_distance_quantiles": {str(key): value for key, value in loaded["quantiles"].items()},
        },
        "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key not in {"result_json", "audit_json"}},
        "objective_records": records,
        "decision": decision,
        "data_boundary": {"docking_scores_read": 0, "ligand_labels_read": 0, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage31_pparg_objective_landscape_screen.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
