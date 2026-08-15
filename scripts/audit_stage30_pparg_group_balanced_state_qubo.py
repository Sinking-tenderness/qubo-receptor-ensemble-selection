"""Independently audit the Stage30 group-balanced PPARG state QUBO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
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
    unique = sorted(np.unique(raw), key=lambda label: int(np.flatnonzero(raw == label)[0]))
    mapping = {int(label): index for index, label in enumerate(unique)}
    return np.asarray([mapping[int(label)] for label in raw], dtype=int)


def audit(config_path: Path, result_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    result_path = result_path.resolve()
    config = read_json(config_path)
    result = read_json(result_path)
    if result.get("status") != "stage30_pparg_group_balanced_state_qubo_complete":
        raise ValueError("unexpected Stage30 result status")
    if result["config"]["sha256"] != sha256(config_path):
        raise ValueError("Stage30 config hash differs")
    for record in result["inputs"].values():
        verify_descriptor(root, record)
    for record in result["outputs"].values():
        verify_descriptor(root, record)
    inputs = {key: root / value for key, value in config["inputs"].items()}
    outputs = {key: root / value for key, value in config["outputs"].items()}
    if read_json(inputs["stage28b_audit"]).get("status") != "stage28_pparg_multistart_md_ensemble_audit_ok":
        raise ValueError("Stage28b audit differs")
    if read_json(inputs["stage29_audit"]).get("status") != "stage29_pparg_md_qubo_solver_scaling_audit_ok":
        raise ValueError("Stage29 audit differs")
    frames = read_csv(inputs["frame_manifest"])
    features = np.load(inputs["feature_archive"])
    distance_archive = np.load(inputs["distance_archive"])
    expected_ids = np.asarray([row["frame_id"] for row in frames])
    if len(frames) != 1200 or not np.array_equal(features["frame_ids"], expected_ids) or not np.array_equal(distance_archive["frame_ids"], expected_ids):
        raise ValueError("Stage30 frame coverage differs")
    condensed = distance_archive["condensed_distances"].astype(float)
    distances = squareform(condensed)
    distances /= float(distances.max())
    hierarchy = linkage(condensed, method="ward", optimal_ordering=False)
    labels = {
        int(count): canonical(fcluster(hierarchy, int(count), criterion="maxclust"))
        for count in config["structural_states"]["cluster_counts"]
    }
    state_manifest = read_csv(outputs["state_manifest_csv"])
    if len(state_manifest) != 1200:
        raise ValueError("Stage30 state manifest count differs")
    by_start: dict[int, list[int]] = {}
    for index, row in enumerate(frames):
        by_start.setdefault(int(row["start_index"]), []).append(index)
    centrality = np.zeros(1200, dtype=float)
    for indices in by_start.values():
        local = np.asarray(indices, dtype=int)
        centrality[local] = 1.0 - distances[np.ix_(local, local)].sum(axis=1) / 149
    maximum_difference = 0.0
    for index, row in enumerate(state_manifest):
        if row["frame_id"] != frames[index]["frame_id"]:
            raise ValueError("Stage30 state frame order differs")
        maximum_difference = max(maximum_difference, abs(float(row["within_start_centrality"]) - centrality[index]))
        for count, values in labels.items():
            if int(row[f"ward_state_{count}"]) != int(values[index]):
                raise ValueError("Stage30 Ward state label differs")
    state_separation = np.zeros_like(distances)
    for values in labels.values():
        state_separation += values[:, None] != values[None, :]
    state_separation /= len(labels)
    candidates = read_csv(outputs["candidate_manifest_csv"])
    by_pool: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        by_pool.setdefault(row["pool_id"], []).append(row)
    expected_per_start = [int(value) for value in config["candidate_scaling"]["frames_per_start"]]
    if len(by_pool) != len(expected_per_start):
        raise ValueError("Stage30 scaling cell count differs")
    models: dict[str, dict[str, Any]] = {}
    previous_by_start: dict[int, tuple[int, ...]] | None = None
    objective = config["objective"]
    k = int(objective["selected_count"])
    for per_start in expected_per_start:
        pool_id = f"balanced_m{per_start:03d}_n{per_start * 8:04d}"
        rows = sorted(by_pool[pool_id], key=lambda row: int(row["local_candidate_index"]))
        if len(rows) != per_start * 8:
            raise ValueError(f"{pool_id}: candidate count differs")
        local_groups: dict[int, list[int]] = {}
        global_indices = []
        for local_index, row in enumerate(rows):
            if local_index != int(row["local_candidate_index"]):
                raise ValueError(f"{pool_id}: local index differs")
            global_index = int(row["global_frame_index"])
            if row["frame_id"] != frames[global_index]["frame_id"]:
                raise ValueError(f"{pool_id}: frame provenance differs")
            local_groups.setdefault(int(row["start_index"]), []).append(local_index)
            global_indices.append(global_index)
        if sorted(len(values) for values in local_groups.values()) != [per_start] * 8:
            raise ValueError(f"{pool_id}: start balance differs")
        current_by_start = {
            start: tuple(global_indices[index] for index in values)
            for start, values in local_groups.items()
        }
        if previous_by_start is not None:
            for start, old in previous_by_start.items():
                if current_by_start[start][: len(old)] != old:
                    raise ValueError("Stage30 candidate pools are not nested within start")
        previous_by_start = current_by_start
        global_array = np.asarray(global_indices, dtype=int)
        linear = float(objective["within_start_centrality_weight"]) * centrality[global_array] / k
        pair = (
            float(objective["cross_start_pair_diversity_weight"]) * distances[np.ix_(global_array, global_array)]
            + float(objective["multiscale_state_separation_weight"]) * state_separation[np.ix_(global_array, global_array)]
        ) / math.comb(k, 2)
        groups = [tuple(local_groups[start]) for start in sorted(local_groups)]
        for group in groups:
            pair[np.ix_(group, group)] = 0.0
        models[pool_id] = {
            "linear": linear,
            "pair": pair,
            "groups": groups,
            "local_by_frame": {row["frame_id"]: index for index, row in enumerate(rows)},
        }

    def selected(pool_id: str, text: str) -> tuple[int, ...]:
        ids = tuple(value for value in text.split("+") if value)
        model = models[pool_id]
        if len(ids) != k or len(ids) != len(set(ids)):
            raise ValueError("Stage30 selected cardinality differs")
        chosen = tuple(sorted(model["local_by_frame"][value] for value in ids))
        if any(sum(value in set(chosen) for value in group) != 1 for group in model["groups"]):
            raise ValueError("Stage30 exactly-one constraint differs")
        return chosen

    def score(pool_id: str, text: str) -> float:
        model = models[pool_id]
        chosen = np.asarray(selected(pool_id, text), dtype=int)
        return float(model["linear"][chosen].sum() + np.triu(model["pair"][np.ix_(chosen, chosen)], 1).sum())

    solver_rows = read_csv(outputs["solver_results_csv"])
    read_rows = read_csv(outputs["read_results_csv"])
    batch_rows = read_csv(outputs["batch_results_csv"])
    model_rows = read_csv(outputs["model_scaling_csv"])
    for row in itertools.chain(solver_rows, read_rows):
        maximum_difference = max(maximum_difference, abs(score(row["pool_id"], row["selected_frame_ids"]) - float(row["objective"])))
        if int(row["represented_start_count"]) != 8:
            raise ValueError("Stage30 reported solution does not cover all starts")
    reads_by_batch: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in read_rows:
        reads_by_batch.setdefault((row["pool_id"], int(row["batch"])), []).append(row)
    for row in batch_rows:
        expected = max(float(value["objective"]) for value in reads_by_batch[(row["pool_id"], int(row["batch"]))])
        maximum_difference = max(maximum_difference, abs(expected - float(row["best_objective"])))
    solver_by_pool: dict[str, list[dict[str, str]]] = {}
    batch_by_pool: dict[str, list[dict[str, str]]] = {}
    for row in solver_rows:
        solver_by_pool.setdefault(row["pool_id"], []).append(row)
    for row in batch_rows:
        batch_by_pool.setdefault(row["pool_id"], []).append(row)
    model_by_pool = {row["pool_id"]: row for row in model_rows}
    strict_wins = 0
    stable_cells = 0
    exact_total = 0
    exact_passed = 0
    equivalence = True
    for pool_id, record in result["pool_records"].items():
        rows = solver_by_pool[pool_id]
        classical = [row for row in rows if row["method"] in config["classical_baselines"]["methods"]]
        annealed = next(row for row in rows if row["method"] == "group_feasible_qubo_annealing")
        strong = max(float(row["objective"]) for row in classical)
        annealed_value = float(annealed["objective"])
        batches = batch_by_pool[pool_id]
        best_batch = max(float(row["best_objective"]) for row in batches)
        within = sum(float(row["best_objective"]) >= best_batch - float(config["gate"]["objective_tolerance"]) for row in batches) / len(batches)
        stable = within >= float(config["gate"]["minimum_batch_fraction_within_tolerance"])
        stable_cells += int(stable)
        strict_wins += int(annealed_value - strong > float(config["gate"]["minimum_strict_gain"]))
        maximum_difference = max(
            maximum_difference,
            abs(strong - float(record["strong_classical_objective"])),
            abs(annealed_value - float(record["annealing_objective"])),
            abs(within - float(record["annealing_batch_fraction_within_tolerance"])),
        )
        exact = [row for row in rows if row["method"] == "exact_oracle"]
        if exact:
            exact_total += 1
            gap = float(exact[0]["objective"]) - annealed_value
            exact_passed += int(gap <= float(config["gate"]["maximum_exact_gap"]) + 1e-12)
        model = models[pool_id]
        chosen = np.asarray(selected(pool_id, annealed["selected_frame_ids"]), dtype=int)
        penalty = float(objective["exactly_one_group_penalty"])
        q_linear = -model["linear"] - penalty
        q_pair = -model["pair"].copy()
        for group in model["groups"]:
            for first, second in itertools.combinations(group, 2):
                q_pair[first, second] = q_pair[second, first] = 2 * penalty
        energy = float(penalty * 8 + q_linear[chosen].sum() + np.triu(q_pair[np.ix_(chosen, chosen)], 1).sum())
        residual = abs(energy + annealed_value)
        maximum_difference = max(maximum_difference, residual, abs(residual - float(model_by_pool[pool_id]["equivalence_residual"])))
        equivalence = equivalence and residual <= float(config["gate"]["maximum_qubo_equivalence_residual"])
    cell_count = len(expected_per_start)
    decision = {
        "input_gate_passed": True,
        "scaling_complete": len(by_pool) == int(config["gate"]["required_scaling_cell_count"]),
        "exactly_one_per_start_verified": True,
        "qubo_equivalence_gate_passed": equivalence,
        "exactness_gate_passed": exact_total > 0 and exact_passed == exact_total,
        "annealing_stability_gate_passed": stable_cells == cell_count,
        "cells_strictly_above_strong_classical": strict_wins,
        "solver_novelty_gate_passed": strict_wins >= int(config["gate"]["minimum_primary_cells_strictly_above_strong_classical"]),
        "direct_qpu_readiness_gate_passed": all(row["direct_qpu_ready_under_frozen_thresholds"].lower() == "true" for row in model_rows),
        "new_docking_jobs_authorized_by_this_stage": False,
        "quantum_hardware_authorized": False,
    }
    if decision != result["decision"]:
        raise ValueError("Stage30 decision does not independently reproduce")
    data_boundary_zero = all(int(value) == 0 for value in result["data_boundary"].values())
    checks = {
        "all_input_and_output_descriptors_verified": True,
        "ward_state_labels_recomputed": True,
        "within_start_centrality_recomputed": True,
        "candidate_scaling_nested_and_balanced": True,
        "all_reported_subsets_satisfy_exactly_one_per_start": True,
        "all_solver_and_read_objectives_recomputed": maximum_difference <= 1e-9,
        "batch_summaries_recomputed": True,
        "qubo_energy_equivalence_recomputed": equivalence,
        "decision_recomputed": True,
        "data_boundary_zero": data_boundary_zero,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage30 audit failed: {checks}")
    audit_result = {
        "schema_version": "1.0",
        "status": "stage30_pparg_group_balanced_state_qubo_audit_ok",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": sha256(config_path)},
        "result": {"path": result_path.relative_to(root).as_posix(), "sha256": sha256(result_path)},
        "checks": checks,
        "coverage": {
            "frame_count": len(frames),
            "state_resolution_count": len(labels),
            "scaling_cell_count": len(by_pool),
            "solver_result_count": len(solver_rows),
            "annealing_batch_count": len(batch_rows),
            "annealing_read_count": len(read_rows),
            "maximum_recomputed_objective_abs_difference": maximum_difference,
        },
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
    parser.add_argument("--config", type=Path, default=Path("configs/stage30_pparg_group_balanced_state_qubo.json"))
    parser.add_argument("--result", type=Path, default=Path("data/stage30_pparg_group_balanced_state_qubo_result.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    audit(args.config, args.result, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
