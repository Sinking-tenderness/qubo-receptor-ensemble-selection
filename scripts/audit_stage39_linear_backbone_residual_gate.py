"""Independently audit the Stage39 gated residual correction."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage19e_cross_target_qubo_v2 import load_target, read_json, rooted, verified
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage37_cross_target_robust_functional_qubo import build_rank_arrays
from scripts.run_stage38_cross_target_stable_triplet_hubo import all_subsets, build_coefficients, objective_values, robust_utilities, subset_name
from scripts.run_stage39_linear_backbone_residual_gate import (
    TOLERANCE,
    aggregate,
    conservative_delta,
    parse_subset,
    subgroup_counts,
    write_json,
)


def audit(config_path: Path, root: Path) -> dict:
    config = read_json(config_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    result = read_json(outputs["result_json"])
    parent = read_json(verified(root, config["inputs"]["stage19e_config"]))
    stage38_config = read_json(verified(root, config["inputs"]["stage38_config"]))
    stage38_result = read_json(verified(root, config["inputs"]["stage38_result"]))
    verified(root, config["inputs"]["stage38_audit"])
    prior = {(str(row["target_id"]), int(row["outer_fold"]), int(row["subset_size"])): row for row in stage38_result["cells"]}
    stored = {(str(row["target_id"]), int(row["outer_fold"]), int(row["subset_size"])): row for row in result["cells"]}
    checks = {
        "result_status": result.get("status") == "stage39_linear_backbone_residual_gate_complete",
        "config_identity": result["config"]["sha256"] == file_sha256(config_path),
        "implementation_identity": result["implementation"]["sha256"] == config["implementation"]["sha256"],
        "input_identities": all(result["inputs"][key]["sha256"] == value["sha256"] for key, value in config["inputs"].items()),
        "train_only_boundary": int(result["data_boundary"]["fresh_validation_rows_read"]) == 0 and int(result["data_boundary"]["locked_test_rows_read"]) == 0,
        "no_execution_boundary": int(result["data_boundary"]["new_docking_jobs"]) == 0 and int(result["data_boundary"]["quantum_hardware_jobs"]) == 0,
    }
    screen = config["screen"]
    gate_config = config["correction_gate"]
    objective_config = stage38_config["objective"]
    recomputed_rows: list[dict[str, object]] = []
    cell_checks: dict[str, bool] = {}
    maximum_difference = 0.0

    for target_id, target_spec in parent["targets"].items():
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        ligand_ids = sorted(row["ligand_id"] for row in ligands)
        manifest = {row["ligand_id"]: row for row in ligands}
        assignments = make_frozen_group_folds(ligands, int(screen["outer_fold_count"]), int(screen["fold_seed"]))
        labels = np.asarray([int(manifest[ligand_id]["label"] == "active") for ligand_id in ligand_ids], dtype=int)
        subsets = all_subsets(len(receptor_ids), int(screen["maximum_subset_size"]))
        subsets_by_size = {size: [subset for subset in subsets if len(subset) == size] for size in range(1, int(screen["maximum_subset_size"]) + 1)}
        for outer_fold in range(int(screen["outer_fold_count"])):
            train_ids = {ligand_id for ligand_id in ligand_ids if assignments[ligand_id] != outer_fold}
            train_mask = np.asarray([ligand_id in train_ids for ligand_id in ligand_ids])
            holdout_mask = ~train_mask
            ranks = build_rank_arrays(ligand_ids, train_ids, receptor_ids, matrices)
            block_masks = []
            block_utilities = []
            for block_fold in range(int(screen["outer_fold_count"])):
                if block_fold == outer_fold:
                    continue
                mask = np.asarray([assignments[ligand_id] == block_fold for ligand_id in ligand_ids])
                block_masks.append(mask)
                values = {size: robust_utilities(ranks[:, mask, :], labels[mask], subsets_by_size[size], float(screen["bedroc_alpha"])) for size in (1, 2, 3)}
                block_utilities.append({size: {subset: float(value) for subset, value in zip(subsets_by_size[size], values[size])} for size in (1, 2, 3)})
            coefficients = build_coefficients(block_utilities, objective_config)
            for size in [int(value) for value in screen["evaluated_subset_sizes"]]:
                size_subsets = subsets_by_size[size]
                triplet_values = objective_values(size_subsets, coefficients, objective_config["weights"])
                linear_values = objective_values(size_subsets, coefficients, {"linear": 1.0, "pair": 0.0, "triplet": 0.0})
                triplet_index = min(range(len(size_subsets)), key=lambda index: (-float(triplet_values[index]), size_subsets[index]))
                linear_index = min(range(len(size_subsets)), key=lambda index: (-float(linear_values[index]), size_subsets[index]))
                triplet_subset, linear_subset = size_subsets[triplet_index], size_subsets[linear_index]
                direct_subset = parse_subset(str(prior[(target_id, outer_fold, size)]["direct_classical_subset"]), receptor_ids)
                block_deltas = []
                for mask in block_masks:
                    values = robust_utilities(ranks[:, mask, :], labels[mask], [triplet_subset, linear_subset], float(screen["bedroc_alpha"]))
                    block_deltas.append(float(values[0] - values[1]))
                lcb = conservative_delta(block_deltas, float(gate_config["risk_kappa"]))
                positive_fraction = sum(value > 0 for value in block_deltas) / len(block_deltas)
                value_range = float(linear_values.max() - linear_values.min())
                loss = float((linear_values[linear_index] - linear_values[triplet_index]) / value_range) if value_range > TOLERANCE else 0.0
                correction = triplet_subset != linear_subset and positive_fraction + TOLERANCE >= float(gate_config["minimum_positive_block_fraction"]) and lcb + TOLERANCE >= float(gate_config["minimum_block_delta_lcb"]) and loss <= float(gate_config["maximum_normalized_linear_backbone_loss"]) + TOLERANCE
                hybrid_subset = triplet_subset if correction else linear_subset
                holdout = robust_utilities(ranks[:, holdout_mask, :], labels[holdout_mask], [hybrid_subset, linear_subset, direct_subset], float(screen["bedroc_alpha"]))
                train_counts = subgroup_counts(ranks[:, train_mask, :], labels[train_mask], linear_subset, triplet_subset, float(screen["favorable_rank_fraction"]))
                holdout_counts = subgroup_counts(ranks[:, holdout_mask, :], labels[holdout_mask], linear_subset, triplet_subset, float(screen["favorable_rank_fraction"]))
                current = stored[(target_id, outer_fold, size)]
                numeric = [*block_deltas, positive_fraction, lcb, loss, float(holdout[0] - holdout[1]), float(holdout[0] - holdout[2])]
                stored_numeric = [float(current[f"block_delta_{index}"]) for index in range(3)] + [float(current["positive_block_fraction"]), float(current["block_delta_lcb"]), float(current["normalized_linear_backbone_loss"]), float(current["holdout_hybrid_minus_linear"]), float(current["holdout_hybrid_minus_direct_classical"])]
                differences = [abs(left - right) for left, right in zip(numeric, stored_numeric)]
                maximum_difference = max(maximum_difference, *differences)
                names_match = current["linear_subset"] == subset_name(linear_subset, receptor_ids) and current["triplet_subset"] == subset_name(triplet_subset, receptor_ids) and current["hybrid_subset"] == subset_name(hybrid_subset, receptor_ids)
                counts_match = all(int(current[f"train_{key}"]) == value for key, value in train_counts.items()) and all(int(current[f"holdout_{key}"]) == value for key, value in holdout_counts.items())
                cell_checks[f"{target_id}::fold{outer_fold}::k{size}"] = names_match and counts_match and bool(current["correction_applied"]) == correction and max(differences) <= 1e-12
                recomputed_rows.append({"target_id": target_id, "correction_applied": correction, "holdout_hybrid_minus_linear": float(holdout[0] - holdout[1]), "holdout_hybrid_minus_direct_classical": float(holdout[0] - holdout[2]), **{f"holdout_{key}": value for key, value in holdout_counts.items()}})

    checks["all_cell_recalculations"] = all(cell_checks.values())
    summary = aggregate(recomputed_rows)
    support = config["support_gate"]
    target_checks = {
        target_id: float(values["mean_holdout_hybrid_minus_linear"]) >= float(support["minimum_target_mean_holdout_delta_vs_linear"]) - TOLERANCE and float(values["mean_holdout_hybrid_minus_direct_classical"]) >= float(support["minimum_target_mean_holdout_delta_vs_direct_classical"]) - TOLERANCE and int(values["negative_corrected_holdout_cell_count"]) <= int(support["maximum_target_negative_corrected_cells"])
        for target_id, values in summary["per_target"].items()
    }
    supported = int(summary["correction_cell_count"]) >= int(support["minimum_correction_cells"]) and int(summary["positive_corrected_holdout_cell_count"]) >= int(support["minimum_positive_corrected_holdout_cells"]) and all(target_checks.values())
    checks["summary_recalculation"] = summary == result["summary"]
    checks["decision_recalculation"] = result["decision"]["gated_residual_correction_supported"] == supported and result["decision"]["target_checks"] == target_checks and result["decision"]["stage40_trust_region_qubo_authorized"] == supported
    checks["failed_gate_is_binding"] = supported is False
    checks["output_hashes"] = all(result["outputs"][key]["sha256"] == file_sha256(rooted(root, config["outputs"][key])) for key in result["outputs"])
    status = "stage39_linear_backbone_residual_gate_audit_ok" if all(checks.values()) else "stage39_linear_backbone_residual_gate_audit_failed"
    record = {"schema_version": "1.0", "status": status, "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)}, "result": {"path": outputs["result_json"].relative_to(root).as_posix(), "sha256": file_sha256(outputs["result_json"])}, "checks": checks, "cell_checks": cell_checks, "maximum_absolute_recalculation_difference": maximum_difference}
    write_json(outputs["audit_json"], record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage39_linear_backbone_residual_gate.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    record = audit(rooted(root, args.config), root)
    print(json.dumps({"status": record["status"], "checks": record["checks"]}, indent=2, sort_keys=True))
    return 0 if record["status"].endswith("_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
