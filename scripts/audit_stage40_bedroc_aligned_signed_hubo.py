"""Independently audit the Stage40 BEDROC-aligned signed HUBO screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage19e_cross_target_qubo_v2 import load_target, read_json, rooted, verified
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage37_cross_target_robust_functional_qubo import build_rank_arrays
from scripts.run_stage38_cross_target_stable_triplet_hubo import TOLERANCE, all_subsets, fixed_size_strong_search, robust_utilities, safe_spearman, subset_name
from scripts.run_stage39_linear_backbone_residual_gate import parse_subset
from scripts.run_stage40_bedroc_aligned_signed_hubo import (
    aggregate,
    build_signed_coefficients,
    early_rank_utilities,
    signed_objective_values,
    write_json,
)


def audit(config_path: Path, root: Path) -> dict:
    config = read_json(config_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    result = read_json(outputs["result_json"])
    parent = read_json(verified(root, config["inputs"]["stage19e_config"]))
    stage38_result = read_json(verified(root, config["inputs"]["stage38_result"]))
    for key in ("stage38_audit", "stage39_result", "stage39_audit"):
        verified(root, config["inputs"][key])
    prior = {(str(row["target_id"]), int(row["outer_fold"]), int(row["subset_size"])): row for row in stage38_result["cells"]}
    stored = {(str(row["target_id"]), int(row["outer_fold"]), int(row["subset_size"])): row for row in result["cells"]}
    checks = {
        "result_status": result.get("status") == "stage40_bedroc_aligned_signed_hubo_complete",
        "config_identity": result["config"]["sha256"] == file_sha256(config_path),
        "implementation_identity": result["implementation"]["sha256"] == config["implementation"]["sha256"],
        "input_identities": all(result["inputs"][key]["sha256"] == value["sha256"] for key, value in config["inputs"].items()),
        "train_only_boundary": int(result["data_boundary"]["fresh_validation_rows_read"]) == 0 and int(result["data_boundary"]["locked_test_rows_read"]) == 0,
        "no_execution_boundary": int(result["data_boundary"]["new_docking_jobs"]) == 0 and int(result["data_boundary"]["quantum_hardware_jobs"]) == 0,
    }
    screen = config["screen"]
    objective_config = config["objective"]
    cell_checks: dict[str, bool] = {}
    recomputed_rows: list[dict[str, Any]] = []
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
            block_values = []
            for block_fold in range(int(screen["outer_fold_count"])):
                if block_fold == outer_fold:
                    continue
                mask = np.asarray([assignments[ligand_id] == block_fold for ligand_id in ligand_ids])
                values = {size: early_rank_utilities(ranks[:, mask, :], labels[mask], subsets_by_size[size], float(screen["bedroc_alpha"])) for size in (1, 2, 3)}
                block_values.append({size: {subset: float(value) for subset, value in zip(subsets_by_size[size], values[size])} for size in (1, 2, 3)})
            coefficients = build_signed_coefficients(block_values, objective_config)
            retained_pairs = sum(abs(value) > 0 for value in coefficients["pair"].values())
            retained_triplets = sum(abs(value) > 0 for value in coefficients["triplet"].values())
            all_objective = signed_objective_values(subsets, coefficients)
            all_objective_map = {subset: float(value) for subset, value in zip(subsets, all_objective)}
            for size in [int(value) for value in screen["evaluated_subset_sizes"]]:
                size_subsets = subsets_by_size[size]
                objective = np.asarray([all_objective_map[subset] for subset in size_subsets])
                train_early = early_rank_utilities(ranks[:, train_mask, :], labels[train_mask], size_subsets, float(screen["bedroc_alpha"]))
                train_bedroc = robust_utilities(ranks[:, train_mask, :], labels[train_mask], size_subsets, float(screen["bedroc_alpha"]))
                exact_index = min(range(len(size_subsets)), key=lambda index: (-float(objective[index]), size_subsets[index]))
                model_subset = size_subsets[exact_index]
                classical_subset, _ = fixed_size_strong_search(all_objective_map, len(receptor_ids), size, int(screen["classical_beam_width"]))
                legacy_linear = parse_subset(str(prior[(target_id, outer_fold, size)]["linear_exact_subset"]), receptor_ids)
                direct_classical = parse_subset(str(prior[(target_id, outer_fold, size)]["direct_classical_subset"]), receptor_ids)
                early_linear_values = np.asarray([sum(coefficients["linear"][(item,)] for item in subset) for subset in size_subsets])
                early_linear_index = min(range(len(size_subsets)), key=lambda index: (-float(early_linear_values[index]), size_subsets[index]))
                early_linear = size_subsets[early_linear_index]
                holdout_subsets = [model_subset, classical_subset, legacy_linear, direct_classical, early_linear]
                holdout_bedroc = robust_utilities(ranks[:, holdout_mask, :], labels[holdout_mask], holdout_subsets, float(screen["bedroc_alpha"]))
                holdout_early = early_rank_utilities(ranks[:, holdout_mask, :], labels[holdout_mask], holdout_subsets, float(screen["bedroc_alpha"]))
                values = {
                    "fidelity_bedroc": safe_spearman(objective, train_bedroc),
                    "fidelity_early": safe_spearman(objective, train_early),
                    "exact_objective": float(objective[exact_index]),
                    "classical_objective": all_objective_map[classical_subset],
                    "gap": float(objective[exact_index] - all_objective_map[classical_subset]),
                    "holdout_model": float(holdout_bedroc[0]),
                    "holdout_classical": float(holdout_bedroc[1]),
                    "holdout_legacy": float(holdout_bedroc[2]),
                    "holdout_direct": float(holdout_bedroc[3]),
                    "holdout_early_linear": float(holdout_bedroc[4]),
                    "holdout_model_early": float(holdout_early[0]),
                }
                current = stored[(target_id, outer_fold, size)]
                stored_values = {
                    "fidelity_bedroc": float(current["train_objective_bedroc_spearman"]),
                    "fidelity_early": float(current["train_objective_early_rank_spearman"]),
                    "exact_objective": float(current["train_exact_objective"]),
                    "classical_objective": float(current["train_classical_objective"]),
                    "gap": float(current["train_exact_minus_classical_objective_gap"]),
                    "holdout_model": float(current["holdout_model_bedroc"]),
                    "holdout_classical": float(current["holdout_classical_bedroc"]),
                    "holdout_legacy": float(current["holdout_legacy_linear_bedroc"]),
                    "holdout_direct": float(current["holdout_direct_classical_bedroc"]),
                    "holdout_early_linear": float(current["holdout_early_linear_bedroc"]),
                    "holdout_model_early": float(current["holdout_model_early_rank"]),
                }
                differences = [abs(values[key] - stored_values[key]) for key in values]
                maximum_difference = max(maximum_difference, *differences)
                names_match = current["model_exact_subset"] == subset_name(model_subset, receptor_ids) and current["model_classical_subset"] == subset_name(classical_subset, receptor_ids) and current["legacy_linear_subset"] == subset_name(legacy_linear, receptor_ids) and current["direct_classical_subset"] == subset_name(direct_classical, receptor_ids) and current["early_linear_subset"] == subset_name(early_linear, receptor_ids)
                cell_checks[f"{target_id}::fold{outer_fold}::k{size}"] = names_match and int(current["retained_pair_count"]) == retained_pairs and int(current["retained_triplet_count"]) == retained_triplets and max(differences) <= 1e-12
                recomputed_rows.append({
                    "target_id": target_id,
                    "train_objective_bedroc_spearman": values["fidelity_bedroc"],
                    "train_objective_early_rank_spearman": values["fidelity_early"],
                    "holdout_model_minus_legacy_linear": values["holdout_model"] - values["holdout_legacy"],
                    "holdout_model_minus_direct_classical": values["holdout_model"] - values["holdout_direct"],
                    "train_exact_minus_classical_objective_gap": values["gap"],
                })

    checks["all_cell_recalculations"] = all(cell_checks.values())
    summary = aggregate(recomputed_rows)
    efficacy_gate = config["efficacy_gate"]
    target_checks = {
        target_id: float(values["mean_train_objective_bedroc_spearman"]) >= float(efficacy_gate["minimum_target_mean_train_bedroc_spearman"]) - TOLERANCE and float(values["mean_holdout_model_minus_legacy_linear"]) >= float(efficacy_gate["minimum_target_mean_holdout_delta_vs_legacy_linear"]) - TOLERANCE and float(values["mean_holdout_model_minus_direct_classical"]) >= float(efficacy_gate["minimum_target_mean_holdout_delta_vs_direct_classical"]) - TOLERANCE
        for target_id, values in summary["per_target"].items()
    }
    efficacy = int(summary["positive_holdout_vs_legacy_cell_count"]) >= int(efficacy_gate["minimum_positive_holdout_vs_legacy_cells"]) and all(target_checks.values())
    difficulty = int(summary["positive_solver_gap_cell_count"]) >= int(config["difficulty_gate"]["minimum_positive_solver_gap_cells"])
    checks["summary_recalculation"] = summary == result["summary"]
    checks["decision_recalculation"] = result["decision"]["bedroc_aligned_objective_supported"] == efficacy and result["decision"]["target_efficacy_checks"] == target_checks and result["decision"]["small_pool_classical_difficulty_detected"] == difficulty and result["decision"]["stage41_independent_target_preregistration_authorized"] == efficacy
    checks["failed_gates_are_binding"] = efficacy is False and difficulty is False
    checks["output_hashes"] = all(result["outputs"][key]["sha256"] == file_sha256(rooted(root, config["outputs"][key])) for key in result["outputs"])
    status = "stage40_bedroc_aligned_signed_hubo_audit_ok" if all(checks.values()) else "stage40_bedroc_aligned_signed_hubo_audit_failed"
    record = {"schema_version": "1.0", "status": status, "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)}, "result": {"path": outputs["result_json"].relative_to(root).as_posix(), "sha256": file_sha256(outputs["result_json"])}, "checks": checks, "cell_checks": cell_checks, "maximum_absolute_recalculation_difference": maximum_difference}
    write_json(outputs["audit_json"], record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage40_bedroc_aligned_signed_hubo.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    record = audit(rooted(root, args.config), root)
    print(json.dumps({"status": record["status"], "checks": record["checks"]}, indent=2, sort_keys=True))
    return 0 if record["status"].endswith("_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
