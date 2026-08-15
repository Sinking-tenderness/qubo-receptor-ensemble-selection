"""Independently audit the Stage37 robust functional objective screen."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage19e_cross_target_qubo_v2 import load_target, read_json, rooted, verified
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage37_cross_target_robust_functional_qubo import (
    TOLERANCE,
    build_rank_arrays,
    enumerate_subsets,
    score_landscape,
    strong_classical_search,
    subset_names,
    write_json,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(config_path: Path, root: Path) -> dict:
    config = read_json(config_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    result = read_json(outputs["result_json"])
    parent = read_json(verified(root, config["inputs"]["stage19e_config"]))
    stored_cells = {
        (str(row["target_id"]), int(row["outer_fold"])): row for row in result["cells"]
    }
    checks = {
        "result_status": result.get("status") == "stage37_cross_target_robust_functional_qubo_complete",
        "config_identity": result["config"]["sha256"] == file_sha256(config_path),
        "implementation_identity": result["implementation"]["sha256"] == config["implementation"]["sha256"],
        "input_identities": all(result["inputs"][key]["sha256"] == value["sha256"] for key, value in config["inputs"].items()),
        "train_only_boundary": int(result["data_boundary"]["fresh_validation_rows_read"]) == 0
        and int(result["data_boundary"]["locked_test_rows_read"]) == 0,
        "no_execution_boundary": int(result["data_boundary"]["new_docking_jobs"]) == 0
        and int(result["data_boundary"]["quantum_hardware_jobs"]) == 0,
    }
    cell_checks: dict[str, bool] = {}
    gaps: list[float] = []
    target_deltas: dict[str, list[tuple[float, float]]] = defaultdict(list)
    maximum_difference = 0.0
    fold_count = int(config["screen"]["outer_fold_count"])
    objective = config["objective"]

    for target_id, target_spec in parent["targets"].items():
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        ligand_ids = sorted(row["ligand_id"] for row in ligands)
        manifest = {row["ligand_id"]: row for row in ligands}
        assignments = make_frozen_group_folds(ligands, fold_count, int(config["screen"]["fold_seed"]))
        group_folds: dict[str, set[int]] = defaultdict(set)
        for row in ligands:
            group_folds[row["split_group_id"]].add(assignments[row["ligand_id"]])
        checks[f"{target_id}_groups_fold_disjoint"] = all(len(values) == 1 for values in group_folds.values())
        subsets = enumerate_subsets(len(receptor_ids), int(objective["minimum_subset_size"]), int(objective["maximum_subset_size"]))
        subset_index = {subset: index for index, subset in enumerate(subsets)}
        all_labels = np.asarray([int(manifest[ligand_id]["label"] == "active") for ligand_id in ligand_ids], dtype=int)
        for outer_fold in range(fold_count):
            train_ids = {ligand_id for ligand_id in ligand_ids if assignments[ligand_id] != outer_fold}
            train_mask = np.asarray([ligand_id in train_ids for ligand_id in ligand_ids])
            ranks = build_rank_arrays(ligand_ids, train_ids, receptor_ids, matrices)
            values, _ = score_landscape(ranks[:, train_mask, :], all_labels[train_mask], subsets, objective)
            exact_index = min(range(len(subsets)), key=lambda index: (-float(values[index]), len(subsets[index]), subsets[index]))
            exact_subset = subsets[exact_index]
            classical_subset, _ = strong_classical_search(
                subsets,
                values,
                len(receptor_ids),
                int(objective["minimum_subset_size"]),
                int(objective["maximum_subset_size"]),
                int(config["screen"]["classical_beam_width"]),
            )
            stored = stored_cells[(target_id, outer_fold)]
            recomputed_gap = float(values[exact_index] - values[subset_index[classical_subset]])
            differences = [
                abs(float(stored["train_exact_objective"]) - float(values[exact_index])),
                abs(float(stored["train_classical_objective"]) - float(values[subset_index[classical_subset]])),
                abs(float(stored["train_exact_minus_classical_gap"]) - recomputed_gap),
            ]
            maximum_difference = max(maximum_difference, *differences)
            cell_checks[f"{target_id}::fold{outer_fold}"] = (
                stored["exact_subset"] == subset_names(exact_subset, receptor_ids)
                and stored["classical_subset"] == subset_names(classical_subset, receptor_ids)
                and max(differences) <= 1e-12
            )
            gaps.append(recomputed_gap)
            target_deltas[target_id].append(
                (float(stored["holdout_objective_delta"]), float(stored["holdout_robust_bedroc_delta"]))
            )

    checks["all_exact_and_classical_recalculations"] = all(cell_checks.values())
    gate = config["support_gate"]
    target_checks = {
        target_id: (
            float(np.mean([row[0] for row in values])) >= float(gate["minimum_target_mean_holdout_objective_delta"]) - TOLERANCE
            and float(np.mean([row[1] for row in values])) >= float(gate["minimum_target_mean_holdout_robust_bedroc_delta"]) - TOLERANCE
        )
        for target_id, values in target_deltas.items()
    }
    supported = (
        sum(value > TOLERANCE for value in gaps) >= int(gate["minimum_positive_train_gap_cells"])
        and float(np.mean(gaps)) >= float(gate["minimum_mean_train_gap"]) - TOLERANCE
        and all(target_checks.values())
    )
    checks["decision_recalculation"] = (
        result["decision"]["functional_objective_supported"] == supported
        and result["decision"]["target_holdout_checks"] == target_checks
        and result["decision"]["stage38_sparse_auxiliary_qubo_authorized"] == supported
    )
    checks["failed_gate_is_binding"] = supported is False
    checks["output_hashes"] = all(
        result["outputs"][key]["sha256"] == file_sha256(rooted(root, config["outputs"][key]))
        for key in result["outputs"]
    )
    status = (
        "stage37_cross_target_robust_functional_qubo_audit_ok"
        if all(checks.values())
        else "stage37_cross_target_robust_functional_qubo_audit_failed"
    )
    record = {
        "schema_version": "1.0",
        "status": status,
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "result": {"path": outputs["result_json"].relative_to(root).as_posix(), "sha256": file_sha256(outputs["result_json"])},
        "checks": checks,
        "cell_checks": cell_checks,
        "maximum_absolute_recalculation_difference": maximum_difference,
        "recomputed_positive_train_gap_cell_count": sum(value > TOLERANCE for value in gaps),
        "recomputed_mean_train_gap": float(np.mean(gaps)),
    }
    write_json(outputs["audit_json"], record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage37_cross_target_robust_functional_qubo.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    record = audit(rooted(root, args.config), root)
    print(json.dumps({"status": record["status"], "checks": record["checks"]}, indent=2, sort_keys=True))
    return 0 if record["status"].endswith("_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
