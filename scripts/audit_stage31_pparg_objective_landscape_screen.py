"""Independently audit the Stage31 PPARG objective-landscape screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_descriptor(root: Path, record: dict[str, Any]) -> None:
    path = root / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"descriptor differs: {record['path']}")
    if "size_bytes" in record and path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"descriptor size differs: {record['path']}")


def canonical(raw: np.ndarray) -> np.ndarray:
    unique = sorted(np.unique(raw), key=lambda value: int(np.flatnonzero(raw == value)[0]))
    mapping = {int(value): index for index, value in enumerate(unique)}
    return np.asarray([mapping[int(value)] for value in raw], dtype=int)


def objective(objective_id: str, value: dict[str, np.ndarray]) -> np.ndarray:
    if objective_id == "robust_coverage_pair":
        return 0.15 * value["s05"] + 0.15 * value["s10"] + 0.25 * value["d10"] + 0.30 * value["w2"] + 0.15 * value["pair"]
    if objective_id == "multiscale_robust_coverage":
        return 0.20 * value["s05"] + 0.20 * value["s10"] + 0.20 * value["s20"] + 0.20 * value["d10"] + 0.20 * value["w1"]
    if objective_id == "worst_two_single_double":
        return 0.30 * value["s10"] + 0.30 * value["d10"] + 0.40 * value["w2"]
    if objective_id == "worst_start_single":
        return value["w1"]
    if objective_id == "global_single_double":
        return 0.25 * value["s10"] + 0.75 * value["d10"]
    if objective_id == "smooth_pair_control":
        return 0.30 * value["central"] + 0.50 * value["pair"] + 0.20 * value["states"]
    raise ValueError(objective_id)


def landscape_statistics(scores: np.ndarray, assignments: np.ndarray, tolerance: float) -> tuple[int, int, int, float]:
    state_count = len(scores)
    indices = np.arange(state_count, dtype=int)
    strides = np.asarray([4 ** (7 - group) for group in range(8)], dtype=int)
    strict = np.ones(state_count, dtype=bool)
    weak = np.ones(state_count, dtype=bool)
    successor = indices.copy()
    successor_score = scores.copy()
    for group in range(8):
        current = assignments[:, group].astype(int)
        for alternative in range(4):
            neighbor = indices + (alternative - current) * strides[group]
            different = alternative != current
            neighbor_score = scores[neighbor]
            strict &= (~different) | (neighbor_score < scores - tolerance)
            weak &= (~different) | (neighbor_score <= scores + tolerance)
            improve = neighbor_score > successor_score + tolerance
            tie = np.isclose(neighbor_score, successor_score, atol=tolerance, rtol=0) & (neighbor < successor)
            update = different & (improve | tie)
            successor[update] = neighbor[update]
            successor_score[update] = neighbor_score[update]
    endpoints = np.full(state_count, -1, dtype=int)
    for origin in range(state_count):
        if endpoints[origin] >= 0:
            continue
        path: list[int] = []
        current = origin
        while endpoints[current] < 0 and successor[current] != current:
            path.append(current)
            current = int(successor[current])
        endpoint = int(endpoints[current]) if endpoints[current] >= 0 else current
        endpoints[current] = endpoint
        for member in reversed(path):
            endpoints[member] = endpoint
    optimum = float(scores.max())
    optimum_states = set(np.flatnonzero(np.isclose(scores, optimum, atol=tolerance, rtol=0)).tolist())
    optimum_basin = float(np.mean(np.isin(endpoints, list(optimum_states))))
    return int(strict.sum()), int(weak.sum()), len(set(endpoints.tolist())), optimum_basin


def audit(config_path: Path, result_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    result_path = result_path.resolve()
    config = read_json(config_path)
    result = read_json(result_path)
    if result.get("status") != "stage31_pparg_objective_landscape_screen_complete":
        raise ValueError("unexpected Stage31 result status")
    if result["config"]["sha256"] != sha256(config_path):
        raise ValueError("Stage31 config hash differs")
    for record in result["inputs"].values():
        verify_descriptor(root, record)
    for record in result["outputs"].values():
        verify_descriptor(root, record)

    inputs = {key: root / value for key, value in config["inputs"].items()}
    outputs = {key: root / value for key, value in config["outputs"].items()}
    if read_json(inputs["stage30_audit"]).get("status") != "stage30_pparg_group_balanced_state_qubo_audit_ok":
        raise ValueError("Stage30 audit differs")
    frames = read_csv(inputs["frame_manifest"])
    features = np.load(inputs["feature_archive"])
    distance_archive = np.load(inputs["distance_archive"])
    frame_ids = np.asarray([row["frame_id"] for row in frames])
    if len(frames) != 1200 or not np.array_equal(features["frame_ids"], frame_ids) or not np.array_equal(distance_archive["frame_ids"], frame_ids):
        raise ValueError("Stage31 frame coverage differs")
    condensed = distance_archive["condensed_distances"].astype(float)
    distance = squareform(condensed)
    distance /= float(distance.max())
    thresholds = {float(q): float(np.quantile(condensed / condensed.max(), float(q))) for q in config["coverage"]["global_distance_quantiles"]}
    ward = linkage(condensed, method="ward", optimal_ordering=False)
    labels = {int(count): canonical(fcluster(ward, int(count), criterion="maxclust")) for count in config["coverage"]["smooth_control_ward_cluster_counts"]}
    by_start: dict[int, np.ndarray] = {}
    for start in range(8):
        by_start[start] = np.asarray([index for index, row in enumerate(frames) if int(row["start_index"]) == start], dtype=int)
    centrality = np.zeros(1200, dtype=float)
    for indices in by_start.values():
        centrality[indices] = 1.0 - distance[np.ix_(indices, indices)].sum(axis=1) / 149

    candidates = read_csv(outputs["candidate_manifest_csv"])
    metric_rows = read_csv(outputs["landscape_metrics_csv"])
    if len(candidates) != 96 or len(metric_rows) != 18:
        raise ValueError("Stage31 output row coverage differs")
    candidate_by_cohort: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        candidate_by_cohort.setdefault(row["cohort_id"], []).append(row)
    metric_by_key = {(row["cohort_id"], row["objective_id"]): row for row in metric_rows}
    assignments = np.indices((4,) * 8, dtype=np.int8).reshape(8, -1).T
    max_difference = 0.0
    gate_rows: dict[str, list[bool]] = {spec["objective_id"]: [] for spec in config["objective_families_in_priority_order"]}
    for cohort_spec in config["exact_candidate_cohorts"]:
        cohort_id = cohort_spec["cohort_id"]
        rows = sorted(candidate_by_cohort[cohort_id], key=lambda row: int(row["local_candidate_index"]))
        if len(rows) != 32:
            raise ValueError(f"{cohort_id}: candidate count differs")
        global_indices = np.asarray([int(row["global_frame_index"]) for row in rows], dtype=int)
        for local, row in enumerate(rows):
            if int(row["local_candidate_index"]) != local or int(row["group_index"]) != local // 4 or int(row["within_group_choice"]) != local % 4:
                raise ValueError(f"{cohort_id}: candidate indexing differs")
            if row["frame_id"] != frames[global_indices[local]]["frame_id"]:
                raise ValueError(f"{cohort_id}: candidate provenance differs")
        selected = assignments + 4 * np.arange(8, dtype=int)[None, :]
        incidence = {q: distance[np.ix_(global_indices, np.arange(1200))] <= threshold + 1e-15 for q, threshold in thresholds.items()}
        values = {key: np.empty(len(assignments), dtype=float) for key in ("s05", "s10", "s20", "d10", "w1", "w2", "pair", "central", "states")}
        for begin in range(0, len(assignments), 4096):
            end = min(begin + 4096, len(assignments))
            chosen = selected[begin:end]
            coverage: dict[float, np.ndarray] = {}
            for quantile in thresholds:
                count = np.zeros((end - begin, 1200), dtype=np.uint8)
                for group in range(8):
                    count += incidence[quantile][chosen[:, group]]
                coverage[quantile] = count
            values["s05"][begin:end] = (coverage[0.05] >= 1).mean(axis=1)
            values["s10"][begin:end] = (coverage[0.10] >= 1).mean(axis=1)
            values["s20"][begin:end] = (coverage[0.20] >= 1).mean(axis=1)
            values["d10"][begin:end] = (coverage[0.10] >= 2).mean(axis=1)
            per_start = np.sort(np.stack([(coverage[0.10][:, indices] >= 1).mean(axis=1) for indices in by_start.values()], axis=1), axis=1)
            values["w1"][begin:end] = per_start[:, 0]
            values["w2"][begin:end] = per_start[:, :2].mean(axis=1)
            pair_sum = np.zeros(end - begin)
            state_sum = np.zeros(end - begin)
            for first in range(8):
                for second in range(first + 1, 8):
                    left, right = chosen[:, first], chosen[:, second]
                    pair_sum += distance[global_indices[left], global_indices[right]]
                    state_sum += np.mean(np.stack([labels[count][global_indices[left]] != labels[count][global_indices[right]] for count in sorted(labels)], axis=1), axis=1)
            values["pair"][begin:end] = pair_sum / 28
            values["states"][begin:end] = state_sum / 28
            values["central"][begin:end] = centrality[global_indices[chosen]].mean(axis=1)
        for spec in config["objective_families_in_priority_order"]:
            objective_id = spec["objective_id"]
            scores = objective(objective_id, values)
            row = metric_by_key[(cohort_id, objective_id)]
            strict, weak, sinks, basin = landscape_statistics(scores, assignments, float(config["difficulty_gate"]["objective_tie_tolerance"]))
            max_difference = max(max_difference, abs(float(scores.max()) - float(row["exact_optimum"])), abs(float(scores.min()) - float(row["objective_minimum"])), abs(basin - float(row["global_optimum_basin_fraction"])))
            if strict != int(row["strict_local_optimum_count"]) or weak != int(row["weak_local_optimum_count"]) or sinks != int(row["tie_broken_sink_count"]):
                raise ValueError(f"{cohort_id}/{objective_id}: landscape counts differ")
            gap = float(row["exact_optimum"]) - float(row["strong_greedy_objective"])
            max_difference = max(max_difference, abs(gap - float(row["absolute_greedy_gap"])))
            gate = config["difficulty_gate"]
            passed = gap >= float(gate["minimum_absolute_greedy_gap"]) - 1e-12 and float(row["normalized_greedy_gap"]) >= float(gate["minimum_normalized_greedy_gap"]) - 1e-12 and strict >= int(gate["minimum_strict_local_optimum_count"]) and basin <= float(gate["maximum_global_optimum_basin_fraction"]) + 1e-12
            if passed != (row["difficulty_gate_passed"].lower() == "true"):
                raise ValueError(f"{cohort_id}/{objective_id}: gate differs")
            gate_rows[objective_id].append(passed)

    selected = None
    for spec in config["objective_families_in_priority_order"]:
        objective_id = spec["objective_id"]
        if selected is None and sum(gate_rows[objective_id]) >= int(config["difficulty_gate"]["minimum_passing_cohorts_per_objective"]):
            selected = objective_id
    decision = {
        "exact_landscape_screen_complete": True,
        "candidate_objective_found": selected is not None,
        "selected_objective_id": selected,
        "stage32_qubo_encoding_authorized": selected is not None,
        "new_docking_jobs_authorized_by_this_stage": False,
        "quantum_hardware_authorized": False,
    }
    if decision != result["decision"]:
        raise ValueError("Stage31 decision does not independently reproduce")
    data_boundary_zero = all(int(value) == 0 for value in result["data_boundary"].values())
    checks = {
        "all_input_and_output_descriptors_verified": True,
        "candidate_cohorts_and_provenance_verified": True,
        "all_196608_feasible_states_reenumerated": True,
        "all_18_exact_optima_recomputed": max_difference <= 1e-10,
        "all_18_local_landscapes_recomputed": True,
        "difficulty_gates_recomputed": True,
        "decision_recomputed": True,
        "data_boundary_zero": data_boundary_zero,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage31 audit failed: {checks}")
    audit_result = {
        "schema_version": "1.0",
        "status": "stage31_pparg_objective_landscape_screen_audit_ok",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": sha256(config_path)},
        "result": {"path": result_path.relative_to(root).as_posix(), "sha256": sha256(result_path)},
        "checks": checks,
        "coverage": {"cohort_count": 3, "objective_count": 6, "states_per_cohort": 65536, "total_state_rows_scored": 196608, "landscape_cell_count": 18, "maximum_recomputed_abs_difference": max_difference},
        "decision": decision,
        "data_boundary": result["data_boundary"],
        "interpretation_boundary": config["interpretation_boundary"],
    }
    outputs["audit_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["audit_json"].write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage31_pparg_objective_landscape_screen.json"))
    parser.add_argument("--result", type=Path, default=Path("data/stage31_pparg_objective_landscape_screen_result.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    audit(args.config, args.result, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
