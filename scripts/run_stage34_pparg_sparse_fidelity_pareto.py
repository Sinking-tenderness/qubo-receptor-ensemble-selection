"""Run the Stage34 PPARG sparse-QUBO fidelity Pareto screen."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import file_sha256, read_json, rooted, write_csv, write_json
from scripts.run_stage30_pparg_group_balanced_state_qubo import load_inputs as load_stage30_inputs
from scripts.run_stage33_pparg_sparse_hardware_qubo import (
    anneal,
    build_domain_wall_qubo,
    exact_oracle,
    qubo_energy,
    selected_to_domain_wall,
    sparse_objective,
    strong_classical,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def quantize(value: float, step: float) -> float:
    return float(round(value / step) * step)


def validate_upstream(config: dict[str, Any], root: Path) -> None:
    expected = {
        "stage28b_audit": "stage28_pparg_multistart_md_ensemble_audit_ok",
        "stage29_audit": "stage29_pparg_md_qubo_solver_scaling_audit_ok",
        "stage30_audit": "stage30_pparg_group_balanced_state_qubo_audit_ok",
        "stage33_audit": "stage33_pparg_sparse_hardware_qubo_audit_ok",
    }
    for key, status in expected.items():
        record = read_json(rooted(root, config["inputs"][key]))
        if record.get("status") != status:
            raise ValueError(f"{key} status differs: {record.get('status')}")


def selected_edges(priority: np.ndarray, groups: list[tuple[int, ...]], q: int) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for left_group in range(8):
        for right_group in range(left_group + 1, 8):
            left = np.asarray(groups[left_group], dtype=int)
            right = np.asarray(groups[right_group], dtype=int)
            block = priority[np.ix_(left, right)]
            for row, left_index in enumerate(left):
                order = np.argsort(-block[row], kind="stable")[:q]
                edges.update((int(left_index), int(right[column])) for column in order)
            for column, right_index in enumerate(right):
                order = np.argsort(-block[:, column], kind="stable")[:q]
                edges.update((int(left[row]), int(right_index)) for row in order)
    return edges


def build_model(per_start: int, loaded: dict[str, Any], encoding: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    groups_global = [tuple(loaded["ordered_global"][start][:per_start]) for start in sorted(loaded["ordered_global"])]
    global_indices = np.asarray([value for group in groups_global for value in group], dtype=int)
    groups = [tuple(range(group * per_start, (group + 1) * per_start)) for group in range(8)]
    distance = loaded["distance"][np.ix_(global_indices, global_indices)]
    state = loaded["state_separation"][np.ix_(global_indices, global_indices)]
    reference = config["dense_reference_objective"]
    step = float(config["sparse_encoding"]["coefficient_quantization_step"])
    floor = float(config["sparse_encoding"]["minimum_absolute_pair_coefficient"])
    dense_centrality_unary = float(reference["within_start_centrality_weight"]) * loaded["centrality"][global_indices] / 8.0
    unary = dense_centrality_unary.copy()
    dense_pair_reward = (
        float(reference["cross_start_pair_diversity_weight"]) * distance
        + float(reference["multiscale_state_separation_weight"]) * state
    ) / float(reference["pair_count"])
    pair: dict[tuple[int, int], float] = {}
    q = int(encoding["neighbors_per_direction"])
    if encoding["family"] == "local_redundancy":
        redundancy = (
            float(reference["cross_start_pair_diversity_weight"]) * (1.0 - distance)
            + float(reference["multiscale_state_separation_weight"]) * (1.0 - state)
        ) / float(reference["pair_count"])
        edges = selected_edges(redundancy, groups, q)
        for edge in edges:
            value = quantize(-float(redundancy[edge]), step)
            if abs(value) >= floor:
                pair[edge] = value
    elif encoding["family"] == "centered_residual":
        residual_priority = np.zeros_like(dense_pair_reward)
        residuals: dict[tuple[int, int], float] = {}
        for left_group in range(8):
            for right_group in range(left_group + 1, 8):
                left = np.asarray(groups[left_group], dtype=int)
                right = np.asarray(groups[right_group], dtype=int)
                block = dense_pair_reward[np.ix_(left, right)]
                grand = float(block.mean())
                row_effect = block.mean(axis=1) - grand
                column_effect = block.mean(axis=0) - grand
                unary[left] += row_effect
                unary[right] += column_effect
                residual = block - grand - row_effect[:, None] - column_effect[None, :]
                residual_priority[np.ix_(left, right)] = np.abs(residual)
                residual_priority[np.ix_(right, left)] = np.abs(residual.T)
                for row, left_index in enumerate(left):
                    for column, right_index in enumerate(right):
                        residuals[(int(left_index), int(right_index))] = float(residual[row, column])
        edges = selected_edges(residual_priority, groups, q)
        for edge in edges:
            value = quantize(residuals[edge], step)
            if abs(value) >= floor:
                pair[edge] = value
    else:
        raise ValueError(f"unknown encoding family: {encoding['family']}")
    unary = np.asarray([quantize(float(value), step) for value in unary], dtype=float)
    pair_matrix = np.zeros((len(global_indices), len(global_indices)), dtype=float)
    for edge, value in pair.items():
        pair_matrix[edge] = value
        pair_matrix[edge[::-1]] = value
    return {
        "per_start": per_start,
        "global_indices": global_indices,
        "groups": groups,
        "unary": unary,
        "pair": pair,
        "pair_matrix": pair_matrix,
        "dense_centrality_unary": dense_centrality_unary,
        "distance": distance,
        "state_separation": state,
        "frame_ids": tuple(loaded["frames"][index]["frame_id"] for index in global_indices),
        "source_ids": tuple(loaded["frames"][index]["conformer_id"] for index in global_indices),
        "k": 8,
    }


def dense_quality_objective(selected: tuple[int, ...], model: dict[str, Any]) -> float:
    chosen = np.asarray(sorted(selected), dtype=int)
    upper = np.triu_indices(len(chosen), 1)
    return float(
        model["dense_centrality_unary"][chosen].sum()
        + 0.5 * model["distance"][np.ix_(chosen, chosen)][upper].mean()
        + 0.2 * model["state_separation"][np.ix_(chosen, chosen)][upper].mean()
    )


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    config = read_json(config_path)
    validate_upstream(config, root)
    loaded = load_stage30_inputs(root, config)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if outputs["result_json"].exists() and not overwrite:
        raise FileExistsError(outputs["result_json"])
    stage30_rows = read_csv(rooted(root, config["inputs"]["stage30_solver_results"]))
    stage30_best = {
        count: max(float(row["objective"]) for row in stage30_rows if int(row["candidate_count"]) == count)
        for count in config["screening_scales"]["candidate_counts"]
    }
    solver_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    gate = config["pareto_gate"]
    sampler = config["annealing_sampler"]
    all_residuals: list[float] = []
    for encoding_index, encoding in enumerate(config["candidate_encodings"]):
        encoding_cells: list[dict[str, Any]] = []
        for scale_index, per_start in enumerate(config["screening_scales"]["frames_per_start"]):
            model = build_model(int(per_start), loaded, encoding, config)
            count = len(model["unary"])
            cell_id = f"{encoding['encoding_id']}__n{count:04d}"
            rng = random.Random(int(sampler["base_seed"]) + encoding_index * 1000003 + scale_index * 10007)
            started = time.perf_counter()
            strong = strong_classical(model, int(config["classical_baseline"]["random_coordinate_restart_count"]), rng)
            strong_runtime = time.perf_counter() - started
            exact = exact_oracle(model, int(config["exact_oracle"]["maximum_states"]))
            reference_record = exact or strong
            batch_best: list[dict[str, Any]] = []
            for batch_index in range(int(sampler["batch_count"])):
                batch_started = time.perf_counter()
                reads = [
                    anneal(model, sampler, random.Random(int(sampler["base_seed"]) + encoding_index * 1000003 + scale_index * 10007 + batch_index * 101 + read_index))
                    for read_index in range(int(sampler["reads_per_batch"]))
                ]
                best = max(reads, key=lambda item: (item["objective"], tuple(-value for value in item["selected"])))
                batch_best.append(best)
                batch_rows.append({
                    "encoding_id": encoding["encoding_id"], "candidate_count": count, "cell_id": cell_id,
                    "batch_index": batch_index, "best_sparse_objective": best["objective"],
                    "reference_gap": float(reference_record["objective"]) - float(best["objective"]),
                    "within_tolerance": float(reference_record["objective"]) - float(best["objective"]) <= float(gate["objective_tolerance"]),
                    "runtime_seconds": time.perf_counter() - batch_started,
                })
            annealed = max(batch_best, key=lambda item: (item["objective"], tuple(-value for value in item["selected"])))
            method_records = [("strong_classical", strong, strong_runtime), ("group_feasible_annealing", annealed, 0.0)]
            if exact:
                method_records.append(("exact_oracle", exact, 0.0))
            for method, record, runtime in method_records:
                solver_rows.append({
                    "encoding_id": encoding["encoding_id"], "family": encoding["family"], "neighbors_per_direction": encoding["neighbors_per_direction"],
                    "candidate_count": count, "cell_id": cell_id, "method": method, "runtime_seconds": runtime,
                    "selected_frame_ids": "+".join(model["frame_ids"][index] for index in record["selected"]),
                    "sparse_objective": record["objective"], "dense_quality_objective": dense_quality_objective(record["selected"], model),
                    "sparse_reference_gap": float(reference_record["objective"]) - float(record["objective"]),
                    "unique_local_optima": record.get("unique_local_optima", ""), "state_count": record.get("state_count", ""),
                })
            qubo = build_domain_wall_qubo(model, float(config["sparse_encoding"]["domain_wall_violation_penalty"]))
            sample_sets = [strong["selected"], annealed["selected"]]
            if exact:
                sample_sets.append(exact["selected"])
            sample_sets.extend(tuple(rng.choice(group) for group in model["groups"]) for _ in range(24))
            residuals = [abs(qubo_energy(selected_to_domain_wall(selected, model), qubo) + sparse_objective(selected, model)) for selected in sample_sets]
            all_residuals.extend(residuals)
            coefficients = [abs(value) for _, value in qubo["linear"]] + [abs(value) for _, _, value in qubo["quadratic"]]
            couplers = len(qubo["quadratic"])
            dense_couplers = math.comb(count, 2)
            reduction = 1.0 - couplers / dense_couplers
            strong_dense = dense_quality_objective(strong["selected"], model)
            quality_loss = stage30_best[count] - strong_dense
            direct_ready = (
                int(qubo["logical_variable_count"]) <= int(gate["direct_qpu_max_logical_variables"])
                and couplers <= int(gate["direct_qpu_max_quadratic_couplers"])
                and max(coefficients) / min(coefficients) <= float(gate["direct_qpu_max_coefficient_dynamic_range"])
            )
            cell = {
                "encoding_id": encoding["encoding_id"], "family": encoding["family"], "neighbors_per_direction": encoding["neighbors_per_direction"],
                "candidate_count": count, "frames_per_start": per_start, "sparse_x_edge_count": len(model["pair"]),
                "domain_wall_variable_count": int(qubo["logical_variable_count"]), "domain_wall_coupler_count": couplers,
                "dense_one_hot_coupler_count": dense_couplers, "coupler_reduction_fraction": reduction,
                "coefficient_dynamic_range": max(coefficients) / min(coefficients), "maximum_equivalence_residual": max(residuals),
                "strong_dense_quality_objective": strong_dense, "stage30_dense_reference_objective": stage30_best[count],
                "dense_quality_loss": quality_loss,
                "annealing_batch_fraction_within_tolerance": sum(float(reference_record["objective"]) - item["objective"] <= float(gate["objective_tolerance"]) for item in batch_best) / len(batch_best),
                "small_exact_gap": "" if not exact else float(exact["objective"]) - float(strong["objective"]),
                "direct_qpu_ready_under_frozen_thresholds": direct_ready,
                "qubo_sha256": qubo["sha256"],
            }
            cell_rows.append(cell)
            encoding_cells.append(cell)
        small = next(cell for cell in encoding_cells if int(cell["candidate_count"]) == 32)
        reference = next(cell for cell in encoding_cells if int(cell["candidate_count"]) == int(gate["direct_qpu_reference_candidate_count"]))
        full = next(cell for cell in encoding_cells if int(cell["candidate_count"]) == 1200)
        quality_pass = all(float(cell["dense_quality_loss"]) <= float(gate["maximum_dense_quality_loss_at_each_gate_scale"]) for cell in encoding_cells)
        equivalence_pass = all(float(cell["maximum_equivalence_residual"]) <= float(gate["maximum_qubo_equivalence_residual"]) for cell in encoding_cells)
        exact_pass = abs(float(small["small_exact_gap"])) <= float(gate["maximum_small_exact_gap"])
        stability_pass = all(float(cell["annealing_batch_fraction_within_tolerance"]) >= float(gate["minimum_batch_fraction_within_tolerance"]) for cell in encoding_cells)
        sparsity_pass = float(full["coupler_reduction_fraction"]) >= float(gate["minimum_full_pool_coupler_reduction_fraction"])
        direct_pass = bool(reference["direct_qpu_ready_under_frozen_thresholds"])
        eligible = bool(quality_pass and equivalence_pass and exact_pass and stability_pass and sparsity_pass and direct_pass)
        summary_rows.append({
            "encoding_id": encoding["encoding_id"], "family": encoding["family"], "neighbors_per_direction": encoding["neighbors_per_direction"],
            "maximum_dense_quality_loss": max(float(cell["dense_quality_loss"]) for cell in encoding_cells),
            "full_pool_coupler_count": int(full["domain_wall_coupler_count"]), "full_pool_coupler_reduction_fraction": full["coupler_reduction_fraction"],
            "reference_cell_variable_count": int(reference["domain_wall_variable_count"]), "reference_cell_coupler_count": int(reference["domain_wall_coupler_count"]),
            "quality_gate_passed": quality_pass, "equivalence_gate_passed": equivalence_pass, "small_exactness_gate_passed": exact_pass,
            "stability_gate_passed": stability_pass, "sparsity_gate_passed": sparsity_pass, "direct_qpu_reference_gate_passed": direct_pass,
            "pareto_eligible": eligible,
        })
    eligible = [row for row in summary_rows if row["pareto_eligible"]]
    selected = min(eligible, key=lambda row: (int(row["full_pool_coupler_count"]), float(row["maximum_dense_quality_loss"]), row["encoding_id"])) if eligible else None
    write_csv(outputs["encoding_summary_csv"], summary_rows)
    write_csv(outputs["cell_metrics_csv"], cell_rows)
    write_csv(outputs["solver_results_csv"], solver_rows)
    write_csv(outputs["batch_results_csv"], batch_rows)
    result = {
        "schema_version": "1.0",
        "status": "stage34_pparg_sparse_fidelity_pareto_complete",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "inputs": {key: {"path": value, "sha256": file_sha256(rooted(root, value))} for key, value in config["inputs"].items()},
        "screen_statistics": {"encoding_count": len(summary_rows), "scale_count": 3, "cell_count": len(cell_rows), "maximum_equivalence_residual": max(all_residuals)},
        "encoding_records": {row["encoding_id"]: row for row in summary_rows},
        "decision": {
            "pareto_eligible_encoding_count": len(eligible),
            "selected_encoding_id": None if selected is None else selected["encoding_id"],
            "sparse_fidelity_gate_passed": selected is not None,
            "small_quantum_annealing_application_pilot_authorized": selected is not None,
            "quantum_advantage_claim_authorized": False,
            "new_docking_jobs_authorized_by_this_stage": False,
        },
        "data_boundary": {"docking_scores_read": 0, "ligand_labels_read": 0, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "outputs": {key: value for key, value in config["outputs"].items() if key not in {"result_json", "audit_json"}},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    report_lines = [
        "# Stage34 PPARG sparse-fidelity Pareto screen", "", "## Decision", "",
        f"- Pareto-eligible encodings: **{len(eligible)}**",
        f"- Selected encoding: **{result['decision']['selected_encoding_id']}**",
        f"- Small quantum-annealing application pilot authorized: **{result['decision']['small_quantum_annealing_application_pilot_authorized']}**",
        "", "## Encoding summary", "",
        "| Encoding | Max dense loss | Full couplers | Reduction | 128-candidate direct ready | Eligible |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['encoding_id']} | {float(row['maximum_dense_quality_loss']):.6f} | {int(row['full_pool_coupler_count'])} | {float(row['full_pool_coupler_reduction_fraction']):.3%} | {row['direct_qpu_reference_gate_passed']} | {row['pareto_eligible']} |"
        )
    report_lines.extend(["", "## Boundary", "", config["interpretation_boundary"]])
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage34_pparg_sparse_fidelity_pareto.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run(rooted(root, args.config), root, args.overwrite)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
