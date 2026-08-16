"""Run the Stage39 linear-backbone, gated triplet-residual correction."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import write_json  # noqa: F401 (deduped)
import argparse
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path
from typing import Any

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
    objective_values,
    robust_utilities,
    subset_name,
)




def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def parse_subset(value: str, receptor_ids: list[str]) -> tuple[int, ...]:
    index = {receptor_id: position for position, receptor_id in enumerate(receptor_ids)}
    return tuple(sorted(index[item] for item in value.split("+")))


def conservative_delta(values: list[float], risk_kappa: float) -> float:
    return statistics.fmean(values) - risk_kappa * statistics.pstdev(values)


def subgroup_counts(
    ranks: np.ndarray,
    labels: np.ndarray,
    linear_subset: tuple[int, ...],
    triplet_subset: tuple[int, ...],
    threshold: float,
) -> dict[str, int]:
    def favorable(subset: tuple[int, ...]) -> np.ndarray:
        per_seed = np.min(ranks[:, :, subset], axis=2)
        return np.median(per_seed, axis=0) <= threshold

    linear = favorable(linear_subset)
    triplet = favorable(triplet_subset)
    active = labels == 1
    decoy = labels == 0
    return {
        "active_rescued": int(np.sum(active & ~linear & triplet)),
        "active_lost": int(np.sum(active & linear & ~triplet)),
        "decoy_newly_promoted": int(np.sum(decoy & ~linear & triplet)),
        "decoy_removed": int(np.sum(decoy & linear & ~triplet)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_target: dict[str, dict[str, Any]] = {}
    for target_id in sorted({str(row["target_id"]) for row in rows}):
        selected = [row for row in rows if row["target_id"] == target_id]
        corrected = [row for row in selected if bool(row["correction_applied"])]
        per_target[target_id] = {
            "cell_count": len(selected),
            "correction_cell_count": len(corrected),
            "mean_holdout_hybrid_minus_linear": statistics.fmean(float(row["holdout_hybrid_minus_linear"]) for row in selected),
            "mean_holdout_hybrid_minus_direct_classical": statistics.fmean(float(row["holdout_hybrid_minus_direct_classical"]) for row in selected),
            "positive_corrected_holdout_cell_count": sum(float(row["holdout_hybrid_minus_linear"]) > TOLERANCE for row in corrected),
            "negative_corrected_holdout_cell_count": sum(float(row["holdout_hybrid_minus_linear"]) < -TOLERANCE for row in corrected),
            "corrected_holdout_mean_delta": statistics.fmean(float(row["holdout_hybrid_minus_linear"]) for row in corrected) if corrected else 0.0,
            "holdout_active_rescued": sum(int(row["holdout_active_rescued"]) for row in corrected),
            "holdout_active_lost": sum(int(row["holdout_active_lost"]) for row in corrected),
            "holdout_decoy_newly_promoted": sum(int(row["holdout_decoy_newly_promoted"]) for row in corrected),
            "holdout_decoy_removed": sum(int(row["holdout_decoy_removed"]) for row in corrected),
        }
    return {
        "cell_count": len(rows),
        "correction_cell_count": sum(bool(row["correction_applied"]) for row in rows),
        "positive_corrected_holdout_cell_count": sum(bool(row["correction_applied"]) and float(row["holdout_hybrid_minus_linear"]) > TOLERANCE for row in rows),
        "negative_corrected_holdout_cell_count": sum(bool(row["correction_applied"]) and float(row["holdout_hybrid_minus_linear"]) < -TOLERANCE for row in rows),
        "per_target": per_target,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Stage39 gated residual correction",
        "",
        "The linear exact subset is the default. A stable-triplet subset replaces it only when the frozen train-block evidence and trust-region checks pass.",
        "",
        "| Target | Corrections | Mean hybrid-linear | Corrected wins | Corrected losses | Active rescued/lost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target_id, values in result["summary"]["per_target"].items():
        lines.append(
            f"| {target_id} | {values['correction_cell_count']}/{values['cell_count']} | "
            f"{float(values['mean_holdout_hybrid_minus_linear']):+.6f} | {values['positive_corrected_holdout_cell_count']} | "
            f"{values['negative_corrected_holdout_cell_count']} | {values['holdout_active_rescued']}/{values['holdout_active_lost']} |"
        )
    lines.extend([
        "",
        f"Support gate: **{'PASS' if result['decision']['gated_residual_correction_supported'] else 'NO-GO'}**.",
        "",
        result["interpretation_boundary"],
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage39 implementation identity differs")
    parent = read_json(verified(root, config["inputs"]["stage19e_config"]))
    stage38_config = read_json(verified(root, config["inputs"]["stage38_config"]))
    stage38_result = read_json(verified(root, config["inputs"]["stage38_result"]))
    verified(root, config["inputs"]["stage38_audit"])
    prior_cells = {(str(row["target_id"]), int(row["outer_fold"]), int(row["subset_size"])): row for row in stage38_result["cells"]}
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage39 outputs exist; pass --overwrite")

    screen = config["screen"]
    gate_config = config["correction_gate"]
    objective_config = stage38_config["objective"]
    cell_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []

    for target_id, target_spec in parent["targets"].items():
        print(f"loading_target={target_id}", flush=True)
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
            block_utilities = []
            block_masks = []
            for block_fold in range(int(screen["outer_fold_count"])):
                if block_fold == outer_fold:
                    continue
                block_mask = np.asarray([assignments[ligand_id] == block_fold for ligand_id in ligand_ids])
                block_masks.append(block_mask)
                values = {
                    size: robust_utilities(ranks[:, block_mask, :], labels[block_mask], subsets_by_size[size], float(screen["bedroc_alpha"]))
                    for size in (1, 2, 3)
                }
                block_utilities.append({size: {subset: float(value) for subset, value in zip(subsets_by_size[size], values[size])} for size in (1, 2, 3)})
            coefficients = build_coefficients(block_utilities, objective_config)

            for size in [int(value) for value in screen["evaluated_subset_sizes"]]:
                size_subsets = subsets_by_size[size]
                triplet_objective = objective_values(size_subsets, coefficients, objective_config["weights"])
                linear_objective = objective_values(size_subsets, coefficients, {"linear": 1.0, "pair": 0.0, "triplet": 0.0})
                triplet_index = min(range(len(size_subsets)), key=lambda index: (-float(triplet_objective[index]), size_subsets[index]))
                linear_index = min(range(len(size_subsets)), key=lambda index: (-float(linear_objective[index]), size_subsets[index]))
                triplet_subset = size_subsets[triplet_index]
                linear_subset = size_subsets[linear_index]
                prior = prior_cells[(target_id, outer_fold, size)]
                if prior["triplet_exact_subset"] != subset_name(triplet_subset, receptor_ids) or prior["linear_exact_subset"] != subset_name(linear_subset, receptor_ids):
                    raise ValueError("Stage39 candidate identity differs from audited Stage38")
                direct_subset = parse_subset(str(prior["direct_classical_subset"]), receptor_ids)
                block_deltas = []
                for block_mask in block_masks:
                    values = robust_utilities(ranks[:, block_mask, :], labels[block_mask], [triplet_subset, linear_subset], float(screen["bedroc_alpha"]))
                    block_deltas.append(float(values[0] - values[1]))
                lcb = conservative_delta(block_deltas, float(gate_config["risk_kappa"]))
                positive_fraction = sum(value > 0 for value in block_deltas) / len(block_deltas)
                linear_range = float(linear_objective.max() - linear_objective.min())
                linear_loss_fraction = float((linear_objective[linear_index] - linear_objective[triplet_index]) / linear_range) if linear_range > TOLERANCE else 0.0
                correction_applied = (
                    triplet_subset != linear_subset
                    and positive_fraction + TOLERANCE >= float(gate_config["minimum_positive_block_fraction"])
                    and lcb + TOLERANCE >= float(gate_config["minimum_block_delta_lcb"])
                    and linear_loss_fraction <= float(gate_config["maximum_normalized_linear_backbone_loss"]) + TOLERANCE
                )
                hybrid_subset = triplet_subset if correction_applied else linear_subset
                holdout_values = robust_utilities(ranks[:, holdout_mask, :], labels[holdout_mask], [hybrid_subset, linear_subset, direct_subset, triplet_subset], float(screen["bedroc_alpha"]))
                train_counts = subgroup_counts(ranks[:, train_mask, :], labels[train_mask], linear_subset, triplet_subset, float(screen["favorable_rank_fraction"]))
                holdout_counts = subgroup_counts(ranks[:, holdout_mask, :], labels[holdout_mask], linear_subset, triplet_subset, float(screen["favorable_rank_fraction"]))
                row = {
                    "target_id": target_id,
                    "outer_fold": outer_fold,
                    "subset_size": size,
                    "linear_subset": subset_name(linear_subset, receptor_ids),
                    "triplet_subset": subset_name(triplet_subset, receptor_ids),
                    "hybrid_subset": subset_name(hybrid_subset, receptor_ids),
                    "direct_classical_subset": subset_name(direct_subset, receptor_ids),
                    "candidate_subsets_differ": triplet_subset != linear_subset,
                    "block_delta_0": block_deltas[0],
                    "block_delta_1": block_deltas[1],
                    "block_delta_2": block_deltas[2],
                    "positive_block_fraction": positive_fraction,
                    "block_delta_lcb": lcb,
                    "normalized_linear_backbone_loss": linear_loss_fraction,
                    "correction_applied": correction_applied,
                    "holdout_hybrid_utility": float(holdout_values[0]),
                    "holdout_linear_utility": float(holdout_values[1]),
                    "holdout_direct_classical_utility": float(holdout_values[2]),
                    "holdout_triplet_utility": float(holdout_values[3]),
                    "holdout_hybrid_minus_linear": float(holdout_values[0] - holdout_values[1]),
                    "holdout_hybrid_minus_direct_classical": float(holdout_values[0] - holdout_values[2]),
                    **{f"train_{key}": value for key, value in train_counts.items()},
                    **{f"holdout_{key}": value for key, value in holdout_counts.items()},
                }
                cell_rows.append(row)
                subgroup_rows.extend(
                    {"target_id": target_id, "outer_fold": outer_fold, "subset_size": size, "split": split, **counts}
                    for split, counts in (("outer_train", train_counts), ("outer_holdout", holdout_counts))
                )
                print(f"cell={target_id}/fold{outer_fold}/k{size} correction={int(correction_applied)} lcb={lcb:+.6f} loss={linear_loss_fraction:.4f}", flush=True)

    summary = aggregate(cell_rows)
    support = config["support_gate"]
    target_checks = {
        target_id: (
            float(values["mean_holdout_hybrid_minus_linear"]) >= float(support["minimum_target_mean_holdout_delta_vs_linear"]) - TOLERANCE
            and float(values["mean_holdout_hybrid_minus_direct_classical"]) >= float(support["minimum_target_mean_holdout_delta_vs_direct_classical"]) - TOLERANCE
            and int(values["negative_corrected_holdout_cell_count"]) <= int(support["maximum_target_negative_corrected_cells"])
        )
        for target_id, values in summary["per_target"].items()
    }
    supported = (
        int(summary["correction_cell_count"]) >= int(support["minimum_correction_cells"])
        and int(summary["positive_corrected_holdout_cell_count"]) >= int(support["minimum_positive_corrected_holdout_cells"])
        and all(target_checks.values())
    )
    decision = {
        "gated_residual_correction_supported": supported,
        "target_checks": target_checks,
        "stage40_trust_region_qubo_authorized": supported,
        "new_docking_authorized": False,
        "quantum_hardware_authorized": False,
    }
    write_csv(outputs["cell_metrics_csv"], cell_rows)
    write_csv(outputs["subgroup_diagnostics_csv"], subgroup_rows)
    result = {
        "schema_version": "1.0",
        "status": "stage39_linear_backbone_residual_gate_complete",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation),
        "inputs": {key: descriptor(root, verified(root, value)) for key, value in config["inputs"].items()},
        "cells": cell_rows,
        "summary": summary,
        "decision": decision,
        "data_boundary": {"mk14_train_rows_read": 696, "pparg_train_rows_read": 668, "fresh_validation_rows_read": 0, "locked_test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_report(outputs["report_md"], result)
    result["outputs"] = {key: descriptor(root, path) for key, path in outputs.items() if key not in {"result_json", "audit_json"}}
    write_json(outputs["result_json"], result)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage39_linear_backbone_residual_gate.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run(rooted(root, args.config), root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
