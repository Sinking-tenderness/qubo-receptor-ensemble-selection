"""Run the Stage40 BEDROC-aligned signed Mobius HUBO screen."""

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
    fixed_size_strong_search,
    robust_utilities,
    safe_spearman,
    subset_name,
)
from scripts.run_stage39_linear_backbone_residual_gate import parse_subset




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


def class_contrast(ranks: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    weights = np.exp(-alpha * ranks)
    return np.mean(weights[labels == 1], axis=0) - np.mean(weights[labels == 0], axis=0)


def early_rank_utilities(
    ranks: np.ndarray,
    labels: np.ndarray,
    subsets: list[tuple[int, ...]],
    alpha: float,
    batch_size: int = 512,
) -> np.ndarray:
    values = np.empty(len(subsets), dtype=float)
    for start in range(0, len(subsets), batch_size):
        batch = subsets[start : start + batch_size]
        if len({len(subset) for subset in batch}) != 1:
            for offset, subset in enumerate(batch):
                values[start + offset] = early_rank_utilities(ranks, labels, [subset], alpha, 1)[0]
            continue
        columns = np.asarray(batch, dtype=int)
        per_seed_ranks = np.stack(
            [np.min(ranks[seed_index][:, columns], axis=2) for seed_index in range(ranks.shape[0])]
        )
        seed_values = np.vstack([class_contrast(per_seed_ranks[seed_index], labels, alpha) for seed_index in range(ranks.shape[0])])
        consensus_values = class_contrast(np.median(per_seed_ranks, axis=0), labels, alpha)
        values[start : start + len(batch)] = (consensus_values + np.mean(seed_values, axis=0) + np.min(seed_values, axis=0)) / 3.0
    return values


def signed_stable(values: list[float], minimum_sign_fraction: float, risk_kappa: float, minimum_magnitude: float) -> float:
    mean_value = statistics.fmean(values)
    if abs(mean_value) <= TOLERANCE:
        return 0.0
    sign = 1.0 if mean_value > 0 else -1.0
    sign_fraction = sum(value * sign > 0 for value in values) / len(values)
    magnitude = abs(mean_value) - risk_kappa * statistics.pstdev(values)
    return sign * magnitude if sign_fraction + TOLERANCE >= minimum_sign_fraction and magnitude + TOLERANCE >= minimum_magnitude else 0.0


def build_signed_coefficients(
    block_values: list[dict[int, dict[tuple[int, ...], float]]],
    config: dict[str, Any],
) -> dict[str, dict[tuple[int, ...], float]]:
    singles = sorted(block_values[0][1])
    pairs = sorted(block_values[0][2])
    triples = sorted(block_values[0][3])
    linear = {subset: statistics.fmean(block[1][subset] for block in block_values) for subset in singles}
    pair: dict[tuple[int, ...], float] = {}
    for key in pairs:
        residuals = [block[2][key] - block[1][(key[0],)] - block[1][(key[1],)] for block in block_values]
        pair[key] = signed_stable(residuals, float(config["minimum_same_sign_block_fraction"]), float(config["risk_kappa"]), float(config["minimum_interaction_magnitude"]))
    triplet: dict[tuple[int, ...], float] = {}
    for key in triples:
        residuals = []
        for block in block_values:
            pair_residual = {
                pair_key: block[2][pair_key] - block[1][(pair_key[0],)] - block[1][(pair_key[1],)]
                for pair_key in itertools.combinations(key, 2)
            }
            residuals.append(
                block[3][key]
                - sum(block[1][(item,)] for item in key)
                - sum(pair_residual.values())
            )
        triplet[key] = signed_stable(residuals, float(config["minimum_same_sign_block_fraction"]), float(config["risk_kappa"]), float(config["minimum_interaction_magnitude"]))
    return {"linear": linear, "pair": pair, "triplet": triplet}


def signed_objective_values(
    subsets: list[tuple[int, ...]], coefficients: dict[str, dict[tuple[int, ...], float]]
) -> np.ndarray:
    return np.asarray(
        [
            sum(coefficients["linear"][(item,)] for item in subset)
            + sum(coefficients["pair"].get(key, 0.0) for key in itertools.combinations(subset, 2))
            + sum(coefficients["triplet"].get(key, 0.0) for key in itertools.combinations(subset, 3))
            for subset in subsets
        ],
        dtype=float,
    )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_target: dict[str, dict[str, Any]] = {}
    for target_id in sorted({str(row["target_id"]) for row in rows}):
        selected = [row for row in rows if row["target_id"] == target_id]
        per_target[target_id] = {
            "cell_count": len(selected),
            "mean_train_objective_bedroc_spearman": statistics.fmean(float(row["train_objective_bedroc_spearman"]) for row in selected),
            "mean_train_objective_early_rank_spearman": statistics.fmean(float(row["train_objective_early_rank_spearman"]) for row in selected),
            "mean_holdout_model_minus_legacy_linear": statistics.fmean(float(row["holdout_model_minus_legacy_linear"]) for row in selected),
            "mean_holdout_model_minus_direct_classical": statistics.fmean(float(row["holdout_model_minus_direct_classical"]) for row in selected),
            "positive_holdout_vs_legacy_cell_count": sum(float(row["holdout_model_minus_legacy_linear"]) > TOLERANCE for row in selected),
            "negative_holdout_vs_legacy_cell_count": sum(float(row["holdout_model_minus_legacy_linear"]) < -TOLERANCE for row in selected),
            "positive_solver_gap_cell_count": sum(float(row["train_exact_minus_classical_objective_gap"]) > TOLERANCE for row in selected),
        }
    return {
        "cell_count": len(rows),
        "positive_holdout_vs_legacy_cell_count": sum(float(row["holdout_model_minus_legacy_linear"]) > TOLERANCE for row in rows),
        "positive_solver_gap_cell_count": sum(float(row["train_exact_minus_classical_objective_gap"]) > TOLERANCE for row in rows),
        "per_target": per_target,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Stage40 BEDROC-aligned signed HUBO",
        "",
        "The objective uses continuous exp(-alpha*r) early-rank contrasts and scaffold-stable signed Mobius pair/triplet interactions.",
        "",
        "| Target | BEDROC fidelity | Early-rank fidelity | Holdout vs legacy linear | Holdout wins | Solver gaps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target_id, values in result["summary"]["per_target"].items():
        lines.append(
            f"| {target_id} | {values['mean_train_objective_bedroc_spearman']:.4f} | "
            f"{values['mean_train_objective_early_rank_spearman']:.4f} | {values['mean_holdout_model_minus_legacy_linear']:+.6f} | "
            f"{values['positive_holdout_vs_legacy_cell_count']}/{values['cell_count']} | {values['positive_solver_gap_cell_count']}/{values['cell_count']} |"
        )
    lines.extend([
        "",
        f"Objective support gate: **{'PASS' if result['decision']['bedroc_aligned_objective_supported'] else 'NO-GO'}**.",
        f"Small-pool difficulty gate: **{'PASS' if result['decision']['small_pool_classical_difficulty_detected'] else 'NO-GO'}**.",
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
        raise ValueError("Stage40 implementation identity differs")
    parent = read_json(verified(root, config["inputs"]["stage19e_config"]))
    stage38_result = read_json(verified(root, config["inputs"]["stage38_result"]))
    for key in ("stage38_audit", "stage39_result", "stage39_audit"):
        verified(root, config["inputs"][key])
    prior = {(str(row["target_id"]), int(row["outer_fold"]), int(row["subset_size"])): row for row in stage38_result["cells"]}
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage40 outputs exist; pass --overwrite")

    screen = config["screen"]
    objective_config = config["objective"]
    model_records: dict[str, Any] = {}
    cell_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for target_id, target_spec in parent["targets"].items():
        print(f"loading_target={target_id}", flush=True)
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        ligand_ids = sorted(row["ligand_id"] for row in ligands)
        manifest = {row["ligand_id"]: row for row in ligands}
        assignments = make_frozen_group_folds(ligands, int(screen["outer_fold_count"]), int(screen["fold_seed"]))
        labels = np.asarray([int(manifest[ligand_id]["label"] == "active") for ligand_id in ligand_ids], dtype=int)
        subsets = all_subsets(len(receptor_ids), int(screen["maximum_subset_size"]))
        subsets_by_size = {size: [subset for subset in subsets if len(subset) == size] for size in range(1, int(screen["maximum_subset_size"]) + 1)}
        model_records[target_id] = {}

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
                values = {
                    size: early_rank_utilities(ranks[:, mask, :], labels[mask], subsets_by_size[size], float(screen["bedroc_alpha"]))
                    for size in (1, 2, 3)
                }
                block_values.append({size: {subset: float(value) for subset, value in zip(subsets_by_size[size], values[size])} for size in (1, 2, 3)})
            coefficients = build_signed_coefficients(block_values, objective_config)
            retained_pairs = {key: value for key, value in coefficients["pair"].items() if abs(value) > 0}
            retained_triplets = {key: value for key, value in coefficients["triplet"].items() if abs(value) > 0}
            model_records[target_id][str(outer_fold)] = {
                "retained_pair_count": len(retained_pairs),
                "retained_triplet_count": len(retained_triplets),
                "positive_pair_count": sum(value > 0 for value in retained_pairs.values()),
                "negative_pair_count": sum(value < 0 for value in retained_pairs.values()),
                "positive_triplet_count": sum(value > 0 for value in retained_triplets.values()),
                "negative_triplet_count": sum(value < 0 for value in retained_triplets.values()),
                "linear": {subset_name(key, receptor_ids): value for key, value in coefficients["linear"].items()},
                "pairs": {subset_name(key, receptor_ids): value for key, value in retained_pairs.items()},
                "triplets": {subset_name(key, receptor_ids): value for key, value in retained_triplets.items()},
            }
            coefficient_rows.append({"target_id": target_id, "outer_fold": outer_fold, **{key: value for key, value in model_records[target_id][str(outer_fold)].items() if not isinstance(value, dict)}})
            all_objective = signed_objective_values(subsets, coefficients)
            all_objective_map = {subset: float(value) for subset, value in zip(subsets, all_objective)}

            for size in [int(value) for value in screen["evaluated_subset_sizes"]]:
                size_subsets = subsets_by_size[size]
                objective = np.asarray([all_objective_map[subset] for subset in size_subsets])
                train_early = early_rank_utilities(ranks[:, train_mask, :], labels[train_mask], size_subsets, float(screen["bedroc_alpha"]))
                train_bedroc = robust_utilities(ranks[:, train_mask, :], labels[train_mask], size_subsets, float(screen["bedroc_alpha"]))
                exact_index = min(range(len(size_subsets)), key=lambda index: (-float(objective[index]), size_subsets[index]))
                model_subset = size_subsets[exact_index]
                classical_subset, search = fixed_size_strong_search(all_objective_map, len(receptor_ids), size, int(screen["classical_beam_width"]))
                current_prior = prior[(target_id, outer_fold, size)]
                legacy_linear = parse_subset(str(current_prior["linear_exact_subset"]), receptor_ids)
                direct_classical = parse_subset(str(current_prior["direct_classical_subset"]), receptor_ids)
                early_linear_values = np.asarray([sum(coefficients["linear"][(item,)] for item in subset) for subset in size_subsets])
                early_linear_index = min(range(len(size_subsets)), key=lambda index: (-float(early_linear_values[index]), size_subsets[index]))
                early_linear = size_subsets[early_linear_index]
                holdout_subsets = [model_subset, classical_subset, legacy_linear, direct_classical, early_linear]
                holdout_bedroc = robust_utilities(ranks[:, holdout_mask, :], labels[holdout_mask], holdout_subsets, float(screen["bedroc_alpha"]))
                holdout_early = early_rank_utilities(ranks[:, holdout_mask, :], labels[holdout_mask], holdout_subsets, float(screen["bedroc_alpha"]))
                row = {
                    "target_id": target_id,
                    "outer_fold": outer_fold,
                    "subset_size": size,
                    "state_count": len(size_subsets),
                    "retained_pair_count": len(retained_pairs),
                    "retained_triplet_count": len(retained_triplets),
                    "model_exact_subset": subset_name(model_subset, receptor_ids),
                    "model_classical_subset": subset_name(classical_subset, receptor_ids),
                    "legacy_linear_subset": subset_name(legacy_linear, receptor_ids),
                    "direct_classical_subset": subset_name(direct_classical, receptor_ids),
                    "early_linear_subset": subset_name(early_linear, receptor_ids),
                    "train_objective_bedroc_spearman": safe_spearman(objective, train_bedroc),
                    "train_objective_early_rank_spearman": safe_spearman(objective, train_early),
                    "train_exact_objective": float(objective[exact_index]),
                    "train_classical_objective": all_objective_map[classical_subset],
                    "train_exact_minus_classical_objective_gap": float(objective[exact_index] - all_objective_map[classical_subset]),
                    "holdout_model_bedroc": float(holdout_bedroc[0]),
                    "holdout_classical_bedroc": float(holdout_bedroc[1]),
                    "holdout_legacy_linear_bedroc": float(holdout_bedroc[2]),
                    "holdout_direct_classical_bedroc": float(holdout_bedroc[3]),
                    "holdout_early_linear_bedroc": float(holdout_bedroc[4]),
                    "holdout_model_early_rank": float(holdout_early[0]),
                    "holdout_model_minus_legacy_linear": float(holdout_bedroc[0] - holdout_bedroc[2]),
                    "holdout_model_minus_direct_classical": float(holdout_bedroc[0] - holdout_bedroc[3]),
                    "holdout_model_minus_early_linear": float(holdout_bedroc[0] - holdout_bedroc[4]),
                    **{f"search_{key}": value for key, value in search.items()},
                }
                cell_rows.append(row)
                print(f"cell={target_id}/fold{outer_fold}/k{size} fidelity={row['train_objective_bedroc_spearman']:.4f} gap={row['train_exact_minus_classical_objective_gap']:.8f}", flush=True)

    summary = aggregate(cell_rows)
    efficacy_gate = config["efficacy_gate"]
    target_checks = {
        target_id: (
            float(values["mean_train_objective_bedroc_spearman"]) >= float(efficacy_gate["minimum_target_mean_train_bedroc_spearman"]) - TOLERANCE
            and float(values["mean_holdout_model_minus_legacy_linear"]) >= float(efficacy_gate["minimum_target_mean_holdout_delta_vs_legacy_linear"]) - TOLERANCE
            and float(values["mean_holdout_model_minus_direct_classical"]) >= float(efficacy_gate["minimum_target_mean_holdout_delta_vs_direct_classical"]) - TOLERANCE
        )
        for target_id, values in summary["per_target"].items()
    }
    efficacy_pass = int(summary["positive_holdout_vs_legacy_cell_count"]) >= int(efficacy_gate["minimum_positive_holdout_vs_legacy_cells"]) and all(target_checks.values())
    difficulty_pass = int(summary["positive_solver_gap_cell_count"]) >= int(config["difficulty_gate"]["minimum_positive_solver_gap_cells"])
    decision = {
        "bedroc_aligned_objective_supported": efficacy_pass,
        "target_efficacy_checks": target_checks,
        "small_pool_classical_difficulty_detected": difficulty_pass,
        "stage41_independent_target_preregistration_authorized": efficacy_pass,
        "hardware_scaling_study_authorized": efficacy_pass,
        "quantum_hardware_authorized": False,
        "new_docking_authorized_by_this_stage": False,
    }
    write_csv(outputs["coefficient_summary_csv"], coefficient_rows)
    write_csv(outputs["cell_metrics_csv"], cell_rows)
    write_json(outputs["model_record_json"], {"schema_version": "1.0", "status": "stage40_bedroc_aligned_signed_hubo_models_frozen", "objective": objective_config, "models": model_records})
    result = {
        "schema_version": "1.0",
        "status": "stage40_bedroc_aligned_signed_hubo_complete",
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
    parser.add_argument("--config", default="configs/stage40_bedroc_aligned_signed_hubo.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run(rooted(root, args.config), root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
