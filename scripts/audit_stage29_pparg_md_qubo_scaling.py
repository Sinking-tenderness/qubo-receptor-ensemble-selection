"""Independently audit the Stage29 PPARG MD pure-QUBO scaling benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
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
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != record["sha256"]:
        raise ValueError(f"descriptor hash differs: {record['path']}")
    if "size_bytes" in record and path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"descriptor size differs: {record['path']}")


def selected_indices(value: str, local_by_frame: dict[str, int], k: int) -> tuple[int, ...]:
    frame_ids = tuple(part for part in value.split("+") if part)
    if len(frame_ids) != k or len(frame_ids) != len(set(frame_ids)):
        raise ValueError("reported Stage29 subset has wrong cardinality")
    try:
        return tuple(sorted(local_by_frame[value] for value in frame_ids))
    except KeyError as error:
        raise ValueError(f"reported frame is outside its candidate pool: {error}") from error


def audit(config_path: Path, result_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    result_path = result_path.resolve()
    config = read_json(config_path)
    result = read_json(result_path)
    if result.get("status") != "stage29_pparg_md_qubo_solver_scaling_complete":
        raise ValueError("unexpected Stage29 result status")
    if result["config"]["sha256"] != sha256(config_path):
        raise ValueError("Stage29 frozen config hash differs")
    for record in result["inputs"].values():
        verify_descriptor(root, record)
    for record in result["outputs"].values():
        verify_descriptor(root, record)
    inputs = {key: root / value for key, value in config["inputs"].items()}
    outputs = {key: root / value for key, value in config["outputs"].items()}
    stage28_audit = read_json(inputs["stage28b_audit"])
    if stage28_audit.get("status") != "stage28_pparg_multistart_md_ensemble_audit_ok":
        raise ValueError("Stage28b audit is not valid")
    if not stage28_audit["decision"]["stage29_solver_scaling_authorized"]:
        raise ValueError("Stage28b did not authorize Stage29")
    frames = read_csv(inputs["frame_manifest"])
    if len(frames) != 1200:
        raise ValueError("Stage29 frame manifest is not complete")
    frame_by_id = {row["frame_id"]: row for row in frames}
    if len(frame_by_id) != len(frames):
        raise ValueError("duplicate frame ID")
    feature_archive = np.load(inputs["feature_archive"])
    distance_archive = np.load(inputs["distance_archive"])
    expected_ids = np.asarray([row["frame_id"] for row in frames])
    if not np.array_equal(feature_archive["frame_ids"], expected_ids):
        raise ValueError("feature frame IDs differ")
    if not np.array_equal(distance_archive["frame_ids"], expected_ids):
        raise ValueError("distance frame IDs differ")
    features = feature_archive["standardized_features"]
    distances = squareform(distance_archive["condensed_distances"].astype(float))
    if features.shape != (1200, 870) or distances.shape != (1200, 1200):
        raise ValueError("Stage29 input arrays have unexpected shape")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(distances)):
        raise ValueError("Stage29 input arrays contain non-finite values")
    distances /= float(distances.max())
    centrality = 1.0 - distances.sum(axis=1) / 1199
    pool_manifest = read_csv(outputs["pool_manifest_csv"])
    pool_rows: dict[str, list[dict[str, str]]] = {}
    for row in pool_manifest:
        pool_rows.setdefault(row["pool_id"], []).append(row)
    expected_pool_count = len(config["candidate_pools"]["primary_scaling_sizes"]) + len(config["candidate_pools"]["sensitivity_pools"])
    if len(pool_rows) != expected_pool_count:
        raise ValueError("Stage29 pool count differs")
    primary_ids = [f"primary_n{int(value):04d}" for value in config["candidate_pools"]["primary_scaling_sizes"]]
    primary_global: list[tuple[int, ...]] = []
    for pool_id in primary_ids:
        local_rows = sorted(pool_rows[pool_id], key=lambda row: int(row["local_candidate_index"]))
        indices = tuple(int(row["global_frame_index"]) for row in local_rows)
        if len(indices) != int(pool_id.rsplit("n", 1)[1]):
            raise ValueError(f"{pool_id}: wrong primary pool size")
        counts: dict[str, int] = {}
        for row in local_rows:
            counts[row["conformer_id"]] = counts.get(row["conformer_id"], 0) + 1
        if max(counts.values()) - min(counts.values()) > 1 or len(counts) != 8:
            raise ValueError(f"{pool_id}: primary pool is not start-balanced")
        primary_global.append(indices)
    for smaller, larger in zip(primary_global, primary_global[1:]):
        if larger[: len(smaller)] != smaller:
            raise ValueError("primary pools are not nested prefixes")
    sensitivity_expected = {row["pool_id"]: row for row in config["candidate_pools"]["sensitivity_pools"]}
    for pool_id, spec in sensitivity_expected.items():
        local_rows = pool_rows[pool_id]
        if len(local_rows) != int(spec["expected_count"]):
            raise ValueError(f"{pool_id}: sensitivity count differs")
        if pool_id == "uniform_100ps_n240" and any((int(row["local_frame_index"]) + 1) % 5 for row in local_rows):
            raise ValueError("100 ps sensitivity rule differs")
        if pool_id == "uniform_200ps_n120" and any((int(row["local_frame_index"]) + 1) % 10 for row in local_rows):
            raise ValueError("200 ps sensitivity rule differs")
        if pool_id == "exclude_3d6d_n1050" and any(row["conformer_id"] == "PPARG_3D6D_aligned" for row in local_rows):
            raise ValueError("3D6D exclusion rule differs")
    objective = config["objective"]
    k = int(objective["selected_count"])
    models: dict[str, dict[str, Any]] = {}
    for pool_id, local_rows in pool_rows.items():
        local_rows = sorted(local_rows, key=lambda row: int(row["local_candidate_index"]))
        global_indices = np.asarray([int(row["global_frame_index"]) for row in local_rows], dtype=int)
        starts = np.asarray([int(frame_by_id[row["frame_id"]]["start_index"]) for row in local_rows])
        times = np.asarray([float(frame_by_id[row["frame_id"]]["time_ps"]) for row in local_rows])
        redundancy = np.where(
            starts[:, None] == starts[None, :],
            np.exp(-np.abs(times[:, None] - times[None, :]) / float(objective["temporal_redundancy_decay_ps"])),
            0.0,
        )
        np.fill_diagonal(redundancy, 0.0)
        local_distance = distances[np.ix_(global_indices, global_indices)]
        linear = float(objective["centrality_weight"]) * centrality[global_indices] / k
        pair = (
            float(objective["pair_diversity_weight"]) * local_distance
            - float(objective["temporal_redundancy_weight"]) * redundancy
        ) / math.comb(k, 2)
        np.fill_diagonal(pair, 0.0)
        models[pool_id] = {
            "linear": linear,
            "pair": pair,
            "local_by_frame": {row["frame_id"]: index for index, row in enumerate(local_rows)},
        }
    maximum_difference = 0.0

    def recompute(pool_id: str, value: str) -> float:
        model = models[pool_id]
        selected = selected_indices(value, model["local_by_frame"], k)
        array = np.asarray(selected, dtype=int)
        return float(model["linear"][array].sum() + np.triu(model["pair"][np.ix_(array, array)], 1).sum())

    solver_rows = read_csv(outputs["solver_results_csv"])
    read_rows = read_csv(outputs["read_results_csv"])
    batch_rows = read_csv(outputs["batch_results_csv"])
    model_rows = read_csv(outputs["model_scaling_csv"])
    for row in solver_rows:
        observed = recompute(row["pool_id"], row["selected_frame_ids"])
        maximum_difference = max(maximum_difference, abs(observed - float(row["objective"])))
    for row in read_rows:
        observed = recompute(row["pool_id"], row["selected_frame_ids"])
        maximum_difference = max(maximum_difference, abs(observed - float(row["objective"])))
    reads_by_batch: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in read_rows:
        reads_by_batch.setdefault((row["pool_id"], int(row["batch"])), []).append(row)
    for row in batch_rows:
        local = reads_by_batch[(row["pool_id"], int(row["batch"]))]
        expected = max(float(value["objective"]) for value in local)
        maximum_difference = max(maximum_difference, abs(expected - float(row["best_objective"])))
    solver_by_pool: dict[str, list[dict[str, str]]] = {}
    batch_by_pool: dict[str, list[dict[str, str]]] = {}
    for row in solver_rows:
        solver_by_pool.setdefault(row["pool_id"], []).append(row)
    for row in batch_rows:
        batch_by_pool.setdefault(row["pool_id"], []).append(row)
    primary_strict_wins = 0
    primary_stable = 0
    exact_total = 0
    exact_passed = 0
    qubo_equivalence = True
    model_by_pool = {row["pool_id"]: row for row in model_rows}
    for pool_id, record in result["pool_records"].items():
        rows = solver_by_pool[pool_id]
        classical = [row for row in rows if row["method"] in config["classical_baselines"]["methods"]]
        annealing = next(row for row in rows if row["method"] == "fixed_cardinality_qubo_annealing")
        strong = max(float(row["objective"]) for row in classical)
        annealed = float(annealing["objective"])
        batches = batch_by_pool[pool_id]
        best_batch = max(float(row["best_objective"]) for row in batches)
        within = sum(float(row["best_objective"]) >= best_batch - float(config["gate"]["objective_tolerance"]) for row in batches) / len(batches)
        stable = within >= float(config["gate"]["minimum_batch_fraction_within_tolerance"])
        maximum_difference = max(
            maximum_difference,
            abs(strong - float(record["strong_classical_objective"])),
            abs(annealed - float(record["annealing_objective"])),
            abs((annealed - strong) - float(record["delta_vs_strong_classical"])),
            abs(within - float(record["annealing_batch_fraction_within_tolerance"])),
        )
        if record["pool_class"] == "primary_scaling":
            primary_strict_wins += int(annealed - strong > float(config["gate"]["minimum_strict_gain"]))
            primary_stable += int(stable)
        exact_rows = [row for row in rows if row["method"] == "exact_oracle"]
        if exact_rows:
            exact_total += 1
            gap = float(exact_rows[0]["objective"]) - annealed
            exact_ok = gap <= float(config["gate"]["maximum_exact_gap"]) + 1e-12
            exact_passed += int(exact_ok)
        model_row = model_by_pool[pool_id]
        n = int(model_row["candidate_count"])
        if int(model_row["logical_variable_count"]) != n or int(model_row["quadratic_coupler_count"]) != n * (n - 1) // 2:
            raise ValueError(f"{pool_id}: QUBO dimensions differ")
        selected = selected_indices(annealing["selected_frame_ids"], models[pool_id]["local_by_frame"], k)
        chosen = np.asarray(selected, dtype=int)
        penalty = float(objective["cardinality_penalty"])
        q_linear = -models[pool_id]["linear"] + penalty * (1 - 2 * k)
        q_pair = -models[pool_id]["pair"] + 2 * penalty
        energy = float(penalty * k * k + q_linear[chosen].sum() + np.triu(q_pair[np.ix_(chosen, chosen)], 1).sum())
        residual = abs(energy + annealed)
        maximum_difference = max(maximum_difference, residual, abs(residual - float(model_row["equivalence_residual"])))
        qubo_equivalence = qubo_equivalence and residual <= float(config["gate"]["maximum_qubo_equivalence_residual"])
    primary_count = len(primary_ids)
    recomputed_decision = {
        "input_gate_passed": True,
        "solver_scaling_complete": primary_count == int(config["gate"]["required_primary_cell_count"]),
        "qubo_equivalence_gate_passed": qubo_equivalence,
        "exactness_gate_passed": exact_total > 0 and exact_passed == exact_total,
        "annealing_stability_gate_passed": primary_stable == primary_count,
        "primary_cells_strictly_above_strong_classical": primary_strict_wins,
        "solver_novelty_gate_passed": primary_strict_wins >= int(config["gate"]["minimum_primary_cells_strictly_above_strong_classical"]),
        "direct_qpu_readiness_gate_passed": all(row["direct_qpu_ready_under_frozen_thresholds"].lower() == "true" for row in model_rows if row["pool_class"] == "primary_scaling"),
        "new_docking_jobs_authorized_by_this_stage": False,
        "quantum_hardware_authorized": False,
    }
    if recomputed_decision != result["decision"]:
        raise ValueError("Stage29 decision does not independently reproduce")
    data_boundary_zero = all(int(value) == 0 for value in result["data_boundary"].values())
    checks = {
        "stage28b_gate_verified": True,
        "all_input_and_output_descriptors_verified": True,
        "pool_rules_and_counts_reproduced": True,
        "primary_pools_nested_and_start_balanced": True,
        "all_solver_and_read_objectives_recomputed": maximum_difference <= 1e-9,
        "batch_summaries_recomputed": True,
        "qubo_energy_equivalence_recomputed": qubo_equivalence,
        "decision_recomputed": True,
        "data_boundary_zero": data_boundary_zero,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage29 audit failed: {checks}")
    audit_result = {
        "schema_version": "1.0",
        "status": "stage29_pparg_md_qubo_solver_scaling_audit_ok",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": sha256(config_path)},
        "result": {"path": result_path.relative_to(root).as_posix(), "sha256": sha256(result_path)},
        "checks": checks,
        "coverage": {
            "candidate_frame_count": len(frames),
            "pool_count": len(pool_rows),
            "primary_pool_count": primary_count,
            "solver_result_count": len(solver_rows),
            "annealing_batch_count": len(batch_rows),
            "annealing_read_count": len(read_rows),
            "maximum_recomputed_objective_abs_difference": maximum_difference,
        },
        "decision": recomputed_decision,
        "data_boundary": result["data_boundary"],
        "interpretation_boundary": config["interpretation_boundary"],
    }
    outputs["audit_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["audit_json"].write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage29_pparg_md_qubo_solver_scaling.json"))
    parser.add_argument("--result", type=Path, default=Path("data/stage29_pparg_md_qubo_solver_scaling_result.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    audit(args.config, args.result, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
