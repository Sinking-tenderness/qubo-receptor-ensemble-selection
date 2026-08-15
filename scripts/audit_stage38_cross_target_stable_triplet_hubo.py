"""Independently audit the Stage38 stable triplet-residual screen."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage19e_cross_target_qubo_v2 import load_target, read_json, rooted, verified
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage37_cross_target_robust_functional_qubo import build_rank_arrays
from scripts.run_stage38_cross_target_stable_triplet_hubo import (
    TOLERANCE,
    all_subsets,
    build_coefficients,
    fixed_size_strong_search,
    objective_values,
    safe_spearman,
    subset_name,
    utilities_by_size,
    write_json,
)


def audit(config_path: Path, root: Path) -> dict:
    config = read_json(config_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    result = read_json(outputs["result_json"])
    parent = read_json(verified(root, config["inputs"]["stage19e_config"]))
    stored = {(str(row["target_id"]), int(row["outer_fold"]), int(row["subset_size"])): row for row in result["cells"]}
    checks = {
        "result_status": result.get("status") == "stage38_cross_target_stable_triplet_hubo_complete",
        "config_identity": result["config"]["sha256"] == file_sha256(config_path),
        "implementation_identity": result["implementation"]["sha256"] == config["implementation"]["sha256"],
        "input_identities": all(result["inputs"][key]["sha256"] == value["sha256"] for key, value in config["inputs"].items()),
        "train_only_boundary": int(result["data_boundary"]["fresh_validation_rows_read"]) == 0 and int(result["data_boundary"]["locked_test_rows_read"]) == 0,
        "no_execution_boundary": int(result["data_boundary"]["new_docking_jobs"]) == 0 and int(result["data_boundary"]["quantum_hardware_jobs"]) == 0,
    }
    screen = config["screen"]
    objective_config = config["objective"]
    cell_checks: dict[str, bool] = {}
    recalculated: list[dict[str, float | str | int]] = []
    maximum_difference = 0.0

    for target_id, target_spec in parent["targets"].items():
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        ligand_ids = sorted(row["ligand_id"] for row in ligands)
        manifest = {row["ligand_id"]: row for row in ligands}
        assignments = make_frozen_group_folds(ligands, int(screen["outer_fold_count"]), int(screen["fold_seed"]))
        group_folds: dict[str, set[int]] = defaultdict(set)
        for row in ligands:
            group_folds[row["split_group_id"]].add(assignments[row["ligand_id"]])
        checks[f"{target_id}_groups_fold_disjoint"] = all(len(values) == 1 for values in group_folds.values())
        subsets = all_subsets(len(receptor_ids), int(screen["maximum_subset_size"]))
        subsets_by_size = {size: [subset for subset in subsets if len(subset) == size] for size in range(1, int(screen["maximum_subset_size"]) + 1)}
        labels = np.asarray([int(manifest[ligand_id]["label"] == "active") for ligand_id in ligand_ids], dtype=int)
        for outer_fold in range(int(screen["outer_fold_count"])):
            train_ids = {ligand_id for ligand_id in ligand_ids if assignments[ligand_id] != outer_fold}
            train_mask = np.asarray([ligand_id in train_ids for ligand_id in ligand_ids])
            holdout_mask = ~train_mask
            ranks = build_rank_arrays(ligand_ids, train_ids, receptor_ids, matrices)
            train_utility = utilities_by_size(ranks[:, train_mask, :], labels[train_mask], subsets_by_size, float(screen["bedroc_alpha"]))
            holdout_utility = utilities_by_size(ranks[:, holdout_mask, :], labels[holdout_mask], subsets_by_size, float(screen["bedroc_alpha"]))
            block_utilities = []
            for block_fold in range(int(screen["outer_fold_count"])):
                if block_fold == outer_fold:
                    continue
                block_mask = np.asarray([assignments[ligand_id] == block_fold for ligand_id in ligand_ids])
                values = utilities_by_size(ranks[:, block_mask, :], labels[block_mask], {size: subsets_by_size[size] for size in (1, 2, 3)}, float(screen["bedroc_alpha"]))
                block_utilities.append({size: {subset: float(value) for subset, value in zip(subsets_by_size[size], values[size])} for size in (1, 2, 3)})
            coefficients = build_coefficients(block_utilities, objective_config)
            retained_pair_count = sum(value > 0 for value in coefficients["pair_raw"].values())
            retained_triplet_count = sum(value > 0 for value in coefficients["triplet_raw"].values())
            all_objective = objective_values(subsets, coefficients, objective_config["weights"])
            all_objective_map = {subset: float(value) for subset, value in zip(subsets, all_objective)}
            all_utility_map = {subset: float(value) for size, values in train_utility.items() for subset, value in zip(subsets_by_size[size], values)}
            for size in [int(value) for value in screen["evaluated_subset_sizes"]]:
                size_subsets = subsets_by_size[size]
                objective = np.asarray([all_objective_map[subset] for subset in size_subsets])
                linear = objective_values(size_subsets, coefficients, {"linear": 1.0, "pair": 0.0, "triplet": 0.0})
                utility = train_utility[size]
                exact_index = min(range(len(size_subsets)), key=lambda index: (-float(objective[index]), size_subsets[index]))
                exact_subset = size_subsets[exact_index]
                classical_subset, _ = fixed_size_strong_search(all_objective_map, len(receptor_ids), size, int(screen["classical_beam_width"]))
                direct_subset, _ = fixed_size_strong_search(all_utility_map, len(receptor_ids), size, int(screen["classical_beam_width"]))
                linear_index = min(range(len(size_subsets)), key=lambda index: (-float(linear[index]), size_subsets[index]))
                linear_subset = size_subsets[linear_index]
                index_by_subset = {subset: index for index, subset in enumerate(size_subsets)}
                holdout = holdout_utility[size]
                current = stored[(target_id, outer_fold, size)]
                recomputed = {
                    "fidelity": safe_spearman(objective, utility),
                    "gap": float(objective[exact_index] - all_objective_map[classical_subset]),
                    "vs_direct": float(holdout[index_by_subset[exact_subset]] - holdout[index_by_subset[direct_subset]]),
                    "vs_linear": float(holdout[index_by_subset[exact_subset]] - holdout[index_by_subset[linear_subset]]),
                }
                differences = [
                    abs(recomputed["fidelity"] - float(current["train_objective_utility_spearman"])),
                    abs(recomputed["gap"] - float(current["train_triplet_exact_minus_classical_gap"])),
                    abs(recomputed["vs_direct"] - float(current["holdout_triplet_exact_minus_direct_classical"])),
                    abs(recomputed["vs_linear"] - float(current["holdout_triplet_exact_minus_linear_exact"])),
                ]
                maximum_difference = max(maximum_difference, *differences)
                names_match = (
                    current["triplet_exact_subset"] == subset_name(exact_subset, receptor_ids)
                    and current["triplet_classical_subset"] == subset_name(classical_subset, receptor_ids)
                    and current["direct_classical_subset"] == subset_name(direct_subset, receptor_ids)
                    and current["linear_exact_subset"] == subset_name(linear_subset, receptor_ids)
                )
                cell_checks[f"{target_id}::fold{outer_fold}::k{size}"] = names_match and max(differences) <= 1e-12 and int(current["retained_pair_count"]) == retained_pair_count and int(current["retained_triplet_count"]) == retained_triplet_count
                recalculated.append({"target_id": target_id, "fidelity": recomputed["fidelity"], "gap": recomputed["gap"], "vs_direct": recomputed["vs_direct"], "vs_linear": recomputed["vs_linear"], "retained_triplet_count": retained_triplet_count})

    checks["all_cell_recalculations"] = all(cell_checks.values())
    gate = config["support_gate"]
    target_checks = {}
    for target_id in sorted({str(row["target_id"]) for row in recalculated}):
        rows = [row for row in recalculated if row["target_id"] == target_id]
        target_checks[target_id] = (
            float(np.mean([float(row["fidelity"]) for row in rows])) >= float(gate["minimum_target_mean_train_spearman"]) - TOLERANCE
            and float(np.mean([float(row["vs_direct"]) for row in rows])) >= float(gate["minimum_target_mean_holdout_delta_vs_direct_classical"]) - TOLERANCE
            and float(np.mean([float(row["vs_linear"]) for row in rows])) >= float(gate["minimum_target_mean_holdout_delta_vs_linear"]) - TOLERANCE
        )
    retained_count = sum(int(row["retained_triplet_count"]) > 0 for row in recalculated)
    gap_count = sum(float(row["gap"]) > TOLERANCE for row in recalculated)
    supported = retained_count >= int(gate["minimum_models_with_retained_triplet"]) and gap_count >= int(gate["minimum_positive_solver_gap_cells"]) and all(target_checks.values())
    checks["decision_recalculation"] = result["decision"]["stable_triplet_objective_supported"] == supported and result["decision"]["target_checks"] == target_checks and result["decision"]["stage39_quadratization_authorized"] == supported
    checks["failed_gate_is_binding"] = supported is False
    checks["output_hashes"] = all(result["outputs"][key]["sha256"] == file_sha256(rooted(root, config["outputs"][key])) for key in result["outputs"])
    status = "stage38_cross_target_stable_triplet_hubo_audit_ok" if all(checks.values()) else "stage38_cross_target_stable_triplet_hubo_audit_failed"
    record = {
        "schema_version": "1.0",
        "status": status,
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "result": {"path": outputs["result_json"].relative_to(root).as_posix(), "sha256": file_sha256(outputs["result_json"])},
        "checks": checks,
        "cell_checks": cell_checks,
        "maximum_absolute_recalculation_difference": maximum_difference,
        "recomputed_retained_triplet_cell_count": retained_count,
        "recomputed_positive_solver_gap_cell_count": gap_count,
    }
    write_json(outputs["audit_json"], record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage38_cross_target_stable_triplet_hubo.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    record = audit(rooted(root, args.config), root)
    print(json.dumps({"status": record["status"], "checks": record["checks"]}, indent=2, sort_keys=True))
    return 0 if record["status"].endswith("_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
