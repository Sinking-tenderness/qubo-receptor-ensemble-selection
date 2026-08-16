"""Run the Stage38 scaffold-stable triplet-residual HUBO/QUBO screen."""

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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage19e_cross_target_qubo_v2 import load_target, read_json, rooted, vectorized_bedroc, verified
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage37_cross_target_robust_functional_qubo import build_rank_arrays


SEED_IDS = ("seed0", "seed1", "seed2")
TOLERANCE = 1e-12




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


def all_subsets(receptor_count: int, maximum_size: int) -> list[tuple[int, ...]]:
    return [subset for size in range(1, maximum_size + 1) for subset in itertools.combinations(range(receptor_count), size)]


def robust_utilities(
    ranks: np.ndarray,
    labels: np.ndarray,
    subsets: list[tuple[int, ...]],
    alpha: float,
    batch_size: int = 512,
) -> np.ndarray:
    values = np.empty(len(subsets), dtype=float)
    for start in range(0, len(subsets), batch_size):
        batch = subsets[start : start + batch_size]
        maximum_size = max(len(subset) for subset in batch)
        # Variable-size batches are grouped by callers, but keep a defensive path.
        if any(len(subset) != maximum_size for subset in batch):
            for offset, subset in enumerate(batch):
                values[start + offset] = robust_utilities(ranks, labels, [subset], alpha, 1)[0]
            continue
        columns = np.asarray(batch, dtype=int)
        per_seed = []
        for seed_index in range(ranks.shape[0]):
            aggregate = np.min(ranks[seed_index][:, columns], axis=2)
            per_seed.append(vectorized_bedroc(aggregate, labels, alpha))
        seed_values = np.vstack(per_seed)
        consensus = np.median(
            np.stack([np.min(ranks[seed_index][:, columns], axis=2) for seed_index in range(ranks.shape[0])]),
            axis=0,
        )
        primary = vectorized_bedroc(consensus, labels, alpha)
        values[start : start + len(batch)] = (primary + np.mean(seed_values, axis=0) + np.min(seed_values, axis=0)) / 3.0
    return values


def utilities_by_size(
    ranks: np.ndarray,
    labels: np.ndarray,
    subsets_by_size: dict[int, list[tuple[int, ...]]],
    alpha: float,
) -> dict[int, np.ndarray]:
    return {size: robust_utilities(ranks, labels, subsets, alpha) for size, subsets in subsets_by_size.items()}


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def normalized(values: dict[tuple[int, ...], float]) -> dict[tuple[int, ...], float]:
    minimum, maximum = min(values.values()), max(values.values())
    if maximum - minimum <= TOLERANCE:
        return {key: 0.0 for key in values}
    return {key: (value - minimum) / (maximum - minimum) for key, value in values.items()}


def stable_lcb(values: list[float], minimum_positive_fraction: float, risk_kappa: float, minimum_lcb: float) -> float:
    positive_fraction = sum(value > 0.0 for value in values) / len(values)
    lcb = statistics.fmean(values) - risk_kappa * statistics.pstdev(values)
    return max(lcb, 0.0) if positive_fraction + TOLERANCE >= minimum_positive_fraction and lcb + TOLERANCE >= minimum_lcb else 0.0


def build_coefficients(
    block_utilities: list[dict[int, dict[tuple[int, ...], float]]],
    config: dict[str, Any],
) -> dict[str, dict[tuple[int, ...], float]]:
    singles = sorted(block_utilities[0][1])
    pairs = sorted(block_utilities[0][2])
    triples = sorted(block_utilities[0][3])
    singleton_raw = {subset: statistics.fmean(block[1][subset] for block in block_utilities) for subset in singles}
    pair_raw: dict[tuple[int, ...], float] = {}
    for pair in pairs:
        residuals = [block[2][pair] - max(block[1][(pair[0],)], block[1][(pair[1],)]) for block in block_utilities]
        pair_raw[pair] = stable_lcb(residuals, float(config["minimum_positive_block_fraction"]), float(config["risk_kappa"]), float(config["minimum_pair_lcb"]))
    triple_raw: dict[tuple[int, ...], float] = {}
    for triple in triples:
        pair_terms = list(itertools.combinations(triple, 2))
        residuals = [block[3][triple] - max(block[2][pair] for pair in pair_terms) for block in block_utilities]
        triple_raw[triple] = stable_lcb(residuals, float(config["minimum_positive_block_fraction"]), float(config["risk_kappa"]), float(config["minimum_triplet_lcb"]))
    return {
        "linear": normalized(singleton_raw),
        "pair": normalized(pair_raw) if max(pair_raw.values(), default=0.0) > 0 else pair_raw,
        "triplet": normalized(triple_raw) if max(triple_raw.values(), default=0.0) > 0 else triple_raw,
        "pair_raw": pair_raw,
        "triplet_raw": triple_raw,
    }


def objective_values(
    subsets: list[tuple[int, ...]], coefficients: dict[str, dict[tuple[int, ...], float]], weights: dict[str, float]
) -> np.ndarray:
    output = np.empty(len(subsets), dtype=float)
    for index, subset in enumerate(subsets):
        output[index] = (
            float(weights["linear"]) * sum(coefficients["linear"][(item,)] for item in subset)
            + float(weights["pair"]) * sum(coefficients["pair"].get(pair, 0.0) for pair in itertools.combinations(subset, 2))
            + float(weights["triplet"]) * sum(coefficients["triplet"].get(triple, 0.0) for triple in itertools.combinations(subset, 3))
        )
    return output


def fixed_size_strong_search(
    values_by_subset: dict[tuple[int, ...], float], receptor_count: int, target_size: int, beam_width: int
) -> tuple[tuple[int, ...], dict[str, int]]:
    beam = [(index,) for index in range(receptor_count)]
    starts: set[tuple[int, ...]] = set(beam)
    for size in range(2, target_size + 1):
        expanded = {
            tuple(sorted((*subset, item)))
            for subset in beam
            for item in range(receptor_count)
            if item not in subset
        }
        beam = sorted(expanded, key=lambda subset: (-values_by_subset[subset], subset))[:beam_width]
        starts.update(beam)
    target_starts = [subset for subset in starts if len(subset) == target_size]
    endpoints: set[tuple[int, ...]] = set()
    for start in target_starts:
        current = start
        while True:
            current_value = values_by_subset[current]
            current_set = set(current)
            neighbors = {
                tuple(sorted((current_set - {removed}) | {added}))
                for removed in current
                for added in range(receptor_count)
                if added not in current_set
            }
            improving = [subset for subset in neighbors if values_by_subset[subset] > current_value + TOLERANCE]
            if not improving:
                endpoints.add(current)
                break
            current = min(improving, key=lambda subset: (-values_by_subset[subset], subset))
    best = min(endpoints, key=lambda subset: (-values_by_subset[subset], subset))
    return best, {"beam_target_start_count": len(target_starts), "local_endpoint_count": len(endpoints)}


def subset_name(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(receptor_ids[index] for index in subset)


def aggregate_cells(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_summary: dict[str, dict[str, Any]] = {}
    for target_id in sorted({str(row["target_id"]) for row in rows}):
        selected = [row for row in rows if row["target_id"] == target_id]
        target_summary[target_id] = {
            "cell_count": len(selected),
            "mean_train_objective_spearman": statistics.fmean(float(row["train_objective_utility_spearman"]) for row in selected),
            "mean_holdout_triplet_exact_minus_direct_classical": statistics.fmean(float(row["holdout_triplet_exact_minus_direct_classical"]) for row in selected),
            "mean_holdout_triplet_exact_minus_linear_exact": statistics.fmean(float(row["holdout_triplet_exact_minus_linear_exact"]) for row in selected),
            "positive_solver_gap_cell_count": sum(float(row["train_triplet_exact_minus_classical_gap"]) > TOLERANCE for row in selected),
            "mean_solver_gap": statistics.fmean(float(row["train_triplet_exact_minus_classical_gap"]) for row in selected),
        }
    return {
        "cell_count": len(rows),
        "retained_triplet_model_count": sum(int(row["retained_triplet_count"]) > 0 for row in rows),
        "positive_solver_gap_cell_count": sum(float(row["train_triplet_exact_minus_classical_gap"]) > TOLERANCE for row in rows),
        "per_target": target_summary,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Stage38 stable triplet-residual objective",
        "",
        "The objective retains only pair and triplet residuals that recur across scaffold blocks. It is a cubic HUBO that can be reduced exactly to a QUBO with Rosenberg auxiliaries.",
        "",
        "| Target | k | Mean fidelity | Mean holdout vs direct classical | Solver-gap cells |",
        "|---|---:|---:|---:|---:|",
    ]
    for target_id, values in result["summary"]["per_target"].items():
        for size in sorted({int(row["subset_size"]) for row in result["cells"] if row["target_id"] == target_id}):
            rows = [row for row in result["cells"] if row["target_id"] == target_id and int(row["subset_size"]) == size]
            lines.append(
                f"| {target_id} | {size} | {statistics.fmean(float(row['train_objective_utility_spearman']) for row in rows):.4f} | "
                f"{statistics.fmean(float(row['holdout_triplet_exact_minus_direct_classical']) for row in rows):+.6f} | "
                f"{sum(float(row['train_triplet_exact_minus_classical_gap']) > TOLERANCE for row in rows)}/{len(rows)} |"
            )
    lines.extend([
        "",
        f"Support gate: **{'PASS' if result['decision']['stable_triplet_objective_supported'] else 'NO-GO'}**.",
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
        raise ValueError("Stage38 implementation identity differs")
    parent = read_json(verified(root, config["inputs"]["stage19e_config"]))
    for key in ("stage19g_result", "stage19g_audit", "stage37_audit"):
        verified(root, config["inputs"][key])
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage38 outputs exist; pass --overwrite")

    screen = config["screen"]
    objective_config = config["objective"]
    fold_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    coefficient_records: dict[str, Any] = {}

    for target_id, target_spec in parent["targets"].items():
        print(f"loading_target={target_id}", flush=True)
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        ligand_ids = sorted(row["ligand_id"] for row in ligands)
        manifest = {row["ligand_id"]: row for row in ligands}
        assignments = make_frozen_group_folds(ligands, int(screen["outer_fold_count"]), int(screen["fold_seed"]))
        subsets = all_subsets(len(receptor_ids), int(screen["maximum_subset_size"]))
        subsets_by_size = {size: [subset for subset in subsets if len(subset) == size] for size in range(1, int(screen["maximum_subset_size"]) + 1)}
        all_labels = np.asarray([int(manifest[ligand_id]["label"] == "active") for ligand_id in ligand_ids], dtype=int)
        for ligand_id in ligand_ids:
            fold_rows.append({"target_id": target_id, "ligand_id": ligand_id, "label": manifest[ligand_id]["label"], "split_group_id": manifest[ligand_id]["split_group_id"], "outer_fold": assignments[ligand_id]})
        coefficient_records[target_id] = {}

        for outer_fold in range(int(screen["outer_fold_count"])):
            train_ids = {ligand_id for ligand_id in ligand_ids if assignments[ligand_id] != outer_fold}
            train_mask = np.asarray([ligand_id in train_ids for ligand_id in ligand_ids])
            holdout_mask = ~train_mask
            ranks = build_rank_arrays(ligand_ids, train_ids, receptor_ids, matrices)
            train_utility = utilities_by_size(ranks[:, train_mask, :], all_labels[train_mask], subsets_by_size, float(screen["bedroc_alpha"]))
            holdout_utility = utilities_by_size(ranks[:, holdout_mask, :], all_labels[holdout_mask], subsets_by_size, float(screen["bedroc_alpha"]))

            block_utilities: list[dict[int, dict[tuple[int, ...], float]]] = []
            for block_fold in range(int(screen["outer_fold_count"])):
                if block_fold == outer_fold:
                    continue
                block_mask = np.asarray([assignments[ligand_id] == block_fold for ligand_id in ligand_ids])
                by_size = utilities_by_size(ranks[:, block_mask, :], all_labels[block_mask], {size: subsets_by_size[size] for size in (1, 2, 3)}, float(screen["bedroc_alpha"]))
                block_utilities.append({size: {subset: float(value) for subset, value in zip(subsets_by_size[size], by_size[size])} for size in (1, 2, 3)})
            coefficients = build_coefficients(block_utilities, objective_config)
            retained_pair_count = sum(value > 0 for value in coefficients["pair_raw"].values())
            retained_triplet_count = sum(value > 0 for value in coefficients["triplet_raw"].values())
            coefficient_records[target_id][str(outer_fold)] = {
                "retained_pair_count": retained_pair_count,
                "retained_triplet_count": retained_triplet_count,
                "linear": {subset_name(key, receptor_ids): value for key, value in coefficients["linear"].items()},
                "retained_pairs": {subset_name(key, receptor_ids): value for key, value in coefficients["pair_raw"].items() if value > 0},
                "retained_triplets": {subset_name(key, receptor_ids): value for key, value in coefficients["triplet_raw"].items() if value > 0},
            }
            model_rows.append({"target_id": target_id, "outer_fold": outer_fold, "retained_pair_count": retained_pair_count, "retained_triplet_count": retained_triplet_count})
            all_objective = objective_values(subsets, coefficients, objective_config["weights"])
            all_objective_map = {subset: float(value) for subset, value in zip(subsets, all_objective)}
            all_utility_map = {
                subset: float(value)
                for size, values in train_utility.items()
                for subset, value in zip(subsets_by_size[size], values)
            }

            for size in [int(value) for value in screen["evaluated_subset_sizes"]]:
                size_subsets = subsets_by_size[size]
                objective = objective_values(size_subsets, coefficients, objective_config["weights"])
                linear_objective = objective_values(size_subsets, coefficients, {"linear": 1.0, "pair": 0.0, "triplet": 0.0})
                utility = train_utility[size]
                utility_map = {subset: float(value) for subset, value in zip(size_subsets, utility)}
                objective_map = {subset: float(value) for subset, value in zip(size_subsets, objective)}
                exact_index = min(range(len(size_subsets)), key=lambda index: (-float(objective[index]), size_subsets[index]))
                exact_subset = size_subsets[exact_index]
                classical_subset, search = fixed_size_strong_search(all_objective_map, len(receptor_ids), size, int(screen["classical_beam_width"]))
                direct_subset, direct_search = fixed_size_strong_search(all_utility_map, len(receptor_ids), size, int(screen["classical_beam_width"]))
                oracle_index = min(range(len(size_subsets)), key=lambda index: (-float(utility[index]), size_subsets[index]))
                oracle_subset = size_subsets[oracle_index]
                linear_index = min(range(len(size_subsets)), key=lambda index: (-float(linear_objective[index]), size_subsets[index]))
                linear_subset = size_subsets[linear_index]
                index_by_subset = {subset: index for index, subset in enumerate(size_subsets)}
                holdout = holdout_utility[size]
                row = {
                    "target_id": target_id,
                    "outer_fold": outer_fold,
                    "subset_size": size,
                    "state_count": len(size_subsets),
                    "retained_pair_count": retained_pair_count,
                    "retained_triplet_count": retained_triplet_count,
                    "triplet_exact_subset": subset_name(exact_subset, receptor_ids),
                    "triplet_classical_subset": subset_name(classical_subset, receptor_ids),
                    "direct_classical_subset": subset_name(direct_subset, receptor_ids),
                    "true_oracle_subset": subset_name(oracle_subset, receptor_ids),
                    "linear_exact_subset": subset_name(linear_subset, receptor_ids),
                    "train_objective_utility_spearman": safe_spearman(objective, utility),
                    "train_triplet_exact_objective": float(objective[exact_index]),
                    "train_triplet_classical_objective": objective_map[classical_subset],
                    "train_triplet_exact_minus_classical_gap": float(objective[exact_index] - objective_map[classical_subset]),
                    "train_triplet_exact_true_utility": utility_map[exact_subset],
                    "train_direct_classical_true_utility": utility_map[direct_subset],
                    "train_true_oracle_utility": utility_map[oracle_subset],
                    "holdout_triplet_exact_utility": float(holdout[index_by_subset[exact_subset]]),
                    "holdout_triplet_classical_utility": float(holdout[index_by_subset[classical_subset]]),
                    "holdout_direct_classical_utility": float(holdout[index_by_subset[direct_subset]]),
                    "holdout_true_oracle_utility": float(holdout[index_by_subset[oracle_subset]]),
                    "holdout_linear_exact_utility": float(holdout[index_by_subset[linear_subset]]),
                    "holdout_triplet_exact_minus_direct_classical": float(holdout[index_by_subset[exact_subset]] - holdout[index_by_subset[direct_subset]]),
                    "holdout_triplet_exact_minus_linear_exact": float(holdout[index_by_subset[exact_subset]] - holdout[index_by_subset[linear_subset]]),
                    **{f"triplet_search_{key}": value for key, value in search.items()},
                    **{f"direct_search_{key}": value for key, value in direct_search.items()},
                }
                cell_rows.append(row)
                print(f"cell={target_id}/fold{outer_fold}/k{size} triples={retained_triplet_count} gap={row['train_triplet_exact_minus_classical_gap']:.8f}", flush=True)

    summary = aggregate_cells(cell_rows)
    gate = config["support_gate"]
    target_checks = {
        target_id: (
            float(values["mean_train_objective_spearman"]) >= float(gate["minimum_target_mean_train_spearman"]) - TOLERANCE
            and float(values["mean_holdout_triplet_exact_minus_direct_classical"]) >= float(gate["minimum_target_mean_holdout_delta_vs_direct_classical"]) - TOLERANCE
            and float(values["mean_holdout_triplet_exact_minus_linear_exact"]) >= float(gate["minimum_target_mean_holdout_delta_vs_linear"]) - TOLERANCE
        )
        for target_id, values in summary["per_target"].items()
    }
    supported = (
        int(summary["retained_triplet_model_count"]) >= int(gate["minimum_models_with_retained_triplet"])
        and int(summary["positive_solver_gap_cell_count"]) >= int(gate["minimum_positive_solver_gap_cells"])
        and all(target_checks.values())
    )
    decision = {
        "stable_triplet_objective_supported": supported,
        "target_checks": target_checks,
        "stage39_quadratization_authorized": supported,
        "new_docking_authorized": False,
        "quantum_hardware_authorized": False,
    }
    write_csv(outputs["fold_assignments_csv"], fold_rows)
    write_csv(outputs["model_summary_csv"], model_rows)
    write_csv(outputs["cell_metrics_csv"], cell_rows)
    model_record = {"schema_version": "1.0", "status": "stage38_stable_triplet_hubo_models_frozen", "objective": objective_config, "quadratization": config["quadratization"], "models": coefficient_records}
    write_json(outputs["model_record_json"], model_record)
    result = {
        "schema_version": "1.0",
        "status": "stage38_cross_target_stable_triplet_hubo_complete",
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
    parser.add_argument("--config", default="configs/stage38_cross_target_stable_triplet_hubo.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run(rooted(root, args.config), root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
