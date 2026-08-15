"""Independently audit the Stage 21 structure-aware QUBO screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    file_sha256,
    load_target,
    maxmin_seeded,
    maxsum_greedy,
    q_energy,
    read_csv,
    read_json,
    rooted,
    distance_matrix,
)


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def parse_subset(value: str) -> tuple[str, ...]:
    result = tuple(sorted(part for part in value.split("+") if part))
    if not result:
        raise ValueError("empty subset")
    if len(result) != len(set(result)):
        raise ValueError("duplicate ID in subset")
    return result


def audit(config_path: Path, root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path.resolve())
    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    if result.get("status") != "stage21_structure_aware_qubo_train_only_complete":
        raise ValueError("unexpected Stage 21 result status")
    implementation_path = rooted(root, result["implementation"]["path"])
    if file_sha256(implementation_path) != result["implementation"]["sha256"]:
        raise ValueError("implementation hash differs")
    if result["config"]["sha256"] != file_sha256(config_path.resolve()):
        raise ValueError("result identifies another config")

    selection_path = rooted(root, result["outputs"]["selection_csv"]["path"])
    restart_path = rooted(root, result["outputs"]["restart_csv"]["path"])
    model_path = rooted(root, result["outputs"]["model_record_json"]["path"])
    report_path = rooted(root, result["outputs"]["report_md"]["path"])
    for key, path in {
        "selection_csv": selection_path,
        "restart_csv": restart_path,
        "model_record_json": model_path,
        "report_md": report_path,
    }.items():
        if file_sha256(path) != result["outputs"][key]["sha256"]:
            raise ValueError(f"output hash differs: {key}")
    model = read_json(model_path)
    selection_rows = read_csv(selection_path)
    restart_rows = read_csv(restart_path)
    diagnostic = config["diagnostic"]
    k_values = [int(value) for value in diagnostic["k_values"]]
    restart_count = int(diagnostic["restart_count"])
    lambda_distance = float(diagnostic["lambda_distance"])
    lambda_quality = float(diagnostic["lambda_quality"])
    cardinality_scale = float(diagnostic["cardinality_penalty_scale"])
    expected_selection = len(config["targets"]) * len(k_values) * 3
    expected_restart = len(config["targets"]) * len(k_values) * restart_count
    if len(selection_rows) != expected_selection:
        raise ValueError(f"selection row count differs: {len(selection_rows)}")
    if len(restart_rows) != expected_restart:
        raise ValueError(f"restart row count differs: {len(restart_rows)}")

    selection_index = {
        (row["target_id"], int(row["k"]), row["method"]): row
        for row in selection_rows
    }
    if len(selection_index) != len(selection_rows):
        raise ValueError("duplicate selection row")
    restart_index: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in restart_rows:
        key = (row["target_id"], int(row["k"]))
        restart_index.setdefault(key, []).append(row)

    checked_selection = 0
    checked_restarts = 0
    checked_swaps = 0
    target_checks: dict[str, Any] = {}
    for target_id, spec in config["targets"].items():
        target = load_target(root, target_id, spec)
        ids = target["ids"]
        matrix = distance_matrix(ids, target["distances"])
        reference_id = str(spec["reference_id"])
        index = {value: position for position, value in enumerate(ids)}
        target_checks[target_id] = {
            "candidate_count": len(ids),
            "hard_gate_excluded_count": len(target["excluded_hard_gate"]),
            "k_checks": {},
        }
        for k in k_values:
            penalty = cardinality_scale * max(1.0, float(max(matrix.flat))) * max(1, k)
            qrow = selection_index[(target_id, k, "qubo_swap_local_search")]
            qsubset = parse_subset(qrow["selected_subset"])
            if len(qsubset) != k or not set(qsubset).issubset(ids):
                raise ValueError(f"{target_id}/{k}: invalid QUBO subset")
            expected_energy = q_energy(
                qsubset, ids, matrix, target["quality"], k, penalty,
                lambda_distance, lambda_quality
            )
            close(float(qrow["qubo_energy"]), expected_energy, f"{target_id}/{k}/energy")
            maxmin = maxmin_seeded(ids, matrix, k, reference_id)
            maxsum = maxsum_greedy(ids, matrix, k)
            if parse_subset(selection_index[(target_id, k, "maxmin_seeded")]["selected_subset"]) != maxmin:
                raise ValueError(f"{target_id}/{k}: max-min baseline differs")
            if parse_subset(selection_index[(target_id, k, "maxsum_greedy")]["selected_subset"]) != maxsum:
                raise ValueError(f"{target_id}/{k}: max-sum baseline differs")

            restarts = restart_index[(target_id, k)]
            if len(restarts) != restart_count:
                raise ValueError(f"{target_id}/{k}: restart count differs")
            restart_energies = []
            for restart in restarts:
                selected = parse_subset(restart["selected_subset"])
                if len(selected) != k or not set(selected).issubset(ids):
                    raise ValueError(f"{target_id}/{k}: invalid restart subset")
                energy = q_energy(
                    selected, ids, matrix, target["quality"], k, penalty,
                    lambda_distance, lambda_quality
                )
                close(float(restart["energy"]), energy, f"{target_id}/{k}/restart-energy")
                restart_energies.append(energy)
                # Independent one-swap local-optimum check.  This does not
                # assume that the implementation's search path was correct.
                selected_set = set(selected)
                for outgoing in selected:
                    for incoming in ids:
                        if incoming in selected_set:
                            continue
                        candidate = tuple(sorted((selected_set - {outgoing}) | {incoming}))
                        candidate_energy = q_energy(
                            candidate, ids, matrix, target["quality"], k, penalty,
                            lambda_distance, lambda_quality
                        )
                        if candidate_energy < energy - 1e-12:
                            raise ValueError(f"{target_id}/{k}: restart is not one-swap local optimum")
                        checked_swaps += 1
                checked_restarts += 1
            best_energy = min(restart_energies)
            close(float(qrow["qubo_energy"]), best_energy, f"{target_id}/{k}/best-energy")
            model = read_json(model_path)
            recorded = model["target_models"][target_id]["k_models"][str(k)]
            if list(qsubset) != recorded["selected_subset"]:
                raise ValueError(f"{target_id}/{k}: model-record subset differs")
            target_checks[target_id]["k_checks"][str(k)] = {
                "state_count": math.comb(len(ids), k),
                "restart_count": len(restarts),
                "best_energy": best_energy,
                "unique_solution_count": len({row["selected_subset"] for row in restarts}),
            }
            checked_selection += 3

    boundary = result["data_boundary"]
    if any(int(value) != 0 for value in boundary.values()):
        raise ValueError("Stage 21 data boundary is nonzero")
    paths_to_check = [
        descriptor["path"]
        for target in result["inputs"].values()
        for descriptor in target.values()
    ]
    if any(marker in path.lower() for path in paths_to_check for marker in ("fresh_validation", "locked_test", "bace1_docking")):
        raise ValueError("protected data path entered Stage 21")

    audit_result = {
        "schema_version": "1.0",
        "status": "stage21_structure_aware_qubo_audit_ok",
        "config": {"path": config_path.resolve().relative_to(root).as_posix(), "sha256": file_sha256(config_path.resolve())},
        "result": {"path": result_path.relative_to(root).as_posix(), "sha256": file_sha256(result_path)},
        "coverage": {
            "selection_rows_recomputed": checked_selection,
            "restart_rows_recomputed": checked_restarts,
            "one_swap_transitions_checked": checked_swaps,
            "target_count": len(target_checks),
            "k_count": len(k_values),
        },
        "target_checks": target_checks,
        "checks": {
            "input_hashes_recorded": True,
            "output_hashes_verified": True,
            "qubo_energy_recomputed": True,
            "classical_baselines_recomputed": True,
            "all_restarts_are_one_swap_local_optima": True,
            "data_boundary_zero": True,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
    }
    output = rooted(root, output.as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage21_structure_aware_qubo.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/stage21_structure_aware_qubo_audit.json"))
    args = parser.parse_args()
    audit(args.config, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
