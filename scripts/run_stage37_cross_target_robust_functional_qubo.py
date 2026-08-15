"""Run the frozen Stage37 cross-target robust functional QUBO screen."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage19e_cross_target_qubo_v2 import (
    load_target,
    read_json,
    rooted,
    vectorized_bedroc,
    verified,
)
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds


SEED_IDS = ("seed0", "seed1", "seed2")
TOLERANCE = 1e-12


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


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
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def fit_rank_transform(train_scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map lower-is-better scores to empirical train CDF ranks with tie midranks."""
    if train_scores.ndim != 1:
        raise ValueError("rank-transform training values must be one dimensional")
    ordered = np.sort(train_scores, kind="stable")
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return (left + 0.5 * (right - left) + 0.5) / (len(ordered) + 1.0)


def build_rank_arrays(
    ligand_ids: list[str],
    train_ids: set[str],
    receptor_ids: list[str],
    matrices: dict[str, dict[str, dict[str, Any]]],
) -> np.ndarray:
    output = np.empty((len(SEED_IDS), len(ligand_ids), len(receptor_ids)), dtype=float)
    train_positions = [index for index, ligand_id in enumerate(ligand_ids) if ligand_id in train_ids]
    for seed_index, seed_id in enumerate(SEED_IDS):
        raw = np.asarray(
            [
                [float(matrices[seed_id][ligand_id][receptor_id]) for receptor_id in receptor_ids]
                for ligand_id in ligand_ids
            ],
            dtype=float,
        )
        for receptor_index in range(len(receptor_ids)):
            output[seed_index, :, receptor_index] = fit_rank_transform(
                raw[train_positions, receptor_index], raw[:, receptor_index]
            )
    return output


def enumerate_subsets(receptor_count: int, minimum_size: int, maximum_size: int) -> list[tuple[int, ...]]:
    return [
        subset
        for size in range(minimum_size, maximum_size + 1)
        for subset in itertools.combinations(range(receptor_count), size)
    ]


def objective_components(
    favorable: np.ndarray,
    labels: np.ndarray,
    subset: tuple[int, ...],
) -> dict[str, float]:
    receptor_hits = favorable[:, :, subset].sum(axis=2)
    covered_seed_count = (receptor_hits >= 1).sum(axis=0)
    double_seed_count = (receptor_hits >= 2).sum(axis=0)
    active = labels == 1
    decoy = labels == 0
    return {
        "active_majority_seed_coverage": float(np.mean(covered_seed_count[active] >= 2)),
        "active_all_seed_coverage": float(np.mean(covered_seed_count[active] == len(SEED_IDS))),
        "active_double_receptor_majority_seed_support": float(np.mean(double_seed_count[active] >= 2)),
        "decoy_any_seed_exposure": float(np.mean(covered_seed_count[decoy] >= 1)),
    }


def objective_value(components: dict[str, float], size: int, config: dict[str, Any]) -> float:
    weights = config["weights"]
    return (
        float(weights["active_majority_seed_coverage"]) * components["active_majority_seed_coverage"]
        + float(weights["active_all_seed_coverage"]) * components["active_all_seed_coverage"]
        + float(weights["active_double_receptor_majority_seed_support"])
        * components["active_double_receptor_majority_seed_support"]
        - float(weights["decoy_any_seed_exposure"]) * components["decoy_any_seed_exposure"]
        - float(weights["receptor_cost"]) * size / int(config["maximum_subset_size"])
    )


def score_landscape(
    ranks: np.ndarray,
    labels: np.ndarray,
    subsets: list[tuple[int, ...]],
    objective: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    favorable = ranks <= float(objective["favorable_rank_fraction"])
    components: list[dict[str, float]] = []
    values = np.empty(len(subsets), dtype=float)
    for index, subset in enumerate(subsets):
        row = objective_components(favorable, labels, subset)
        components.append(row)
        values[index] = objective_value(row, len(subset), objective)
    return values, components


def subset_key(values: np.ndarray, subsets: list[tuple[int, ...]], index: int) -> tuple[Any, ...]:
    return (-float(values[index]), len(subsets[index]), subsets[index])


def local_improve(
    start: tuple[int, ...],
    receptor_count: int,
    minimum_size: int,
    maximum_size: int,
    value_by_subset: dict[tuple[int, ...], float],
) -> tuple[int, ...]:
    current = start
    while True:
        current_value = value_by_subset[current]
        current_set = set(current)
        neighbors: set[tuple[int, ...]] = set()
        if len(current) < maximum_size:
            neighbors.update(tuple(sorted((*current, item))) for item in range(receptor_count) if item not in current_set)
        if len(current) > minimum_size:
            neighbors.update(tuple(value for value in current if value != item) for item in current)
        for removed in current:
            neighbors.update(
                tuple(sorted((current_set - {removed}) | {added}))
                for added in range(receptor_count)
                if added not in current_set
            )
        improving = [subset for subset in neighbors if value_by_subset[subset] > current_value + TOLERANCE]
        if not improving:
            return current
        current = min(improving, key=lambda subset: (-value_by_subset[subset], len(subset), subset))


def strong_classical_search(
    subsets: list[tuple[int, ...]],
    values: np.ndarray,
    receptor_count: int,
    minimum_size: int,
    maximum_size: int,
    beam_width: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    value_by_subset = {subset: float(value) for subset, value in zip(subsets, values)}
    starts: set[tuple[int, ...]] = {(index,) for index in range(receptor_count)}

    beam = sorted(starts, key=lambda subset: (-value_by_subset[subset], subset))[:beam_width]
    starts.update(beam)
    for size in range(2, maximum_size + 1):
        expanded = {
            tuple(sorted((*subset, item)))
            for subset in beam
            for item in range(receptor_count)
            if item not in subset
        }
        beam = sorted(expanded, key=lambda subset: (-value_by_subset[subset], subset))[:beam_width]
        starts.update(beam)

    for initial in range(receptor_count):
        current = (initial,)
        starts.add(current)
        while len(current) < maximum_size:
            candidates = [
                tuple(sorted((*current, item)))
                for item in range(receptor_count)
                if item not in current
            ]
            current = min(candidates, key=lambda subset: (-value_by_subset[subset], subset))
            starts.add(current)

    endpoints = {local_improve(start, receptor_count, minimum_size, maximum_size, value_by_subset) for start in starts}
    best = min(endpoints, key=lambda subset: (-value_by_subset[subset], len(subset), subset))
    return best, {
        "beam_width": beam_width,
        "start_state_count": len(starts),
        "local_endpoint_count": len(endpoints),
    }


def bedroc_metrics(
    ranks: np.ndarray,
    labels: np.ndarray,
    subset: tuple[int, ...],
    alpha: float,
) -> dict[str, float]:
    per_seed_rank = np.min(ranks[:, :, subset], axis=2)
    seed_bedroc = vectorized_bedroc(per_seed_rank.T, labels, alpha)
    consensus_rank = np.median(per_seed_rank, axis=0)
    primary = float(vectorized_bedroc(consensus_rank[:, None], labels, alpha)[0])
    mean_seed = float(np.mean(seed_bedroc))
    worst_seed = float(np.min(seed_bedroc))
    return {
        "primary_bedroc": primary,
        "mean_seed_bedroc": mean_seed,
        "worst_seed_bedroc": worst_seed,
        "robust_bedroc_composite": (primary + mean_seed + worst_seed) / 3.0,
    }


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def subset_names(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(receptor_ids[index] for index in subset)


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_target: dict[str, dict[str, Any]] = {}
    for target_id in sorted({str(row["target_id"]) for row in rows}):
        selected = [row for row in rows if row["target_id"] == target_id]
        objective_deltas = [float(row["holdout_objective_delta"]) for row in selected]
        bedroc_deltas = [float(row["holdout_robust_bedroc_delta"]) for row in selected]
        per_target[target_id] = {
            "fold_count": len(selected),
            "mean_holdout_objective_delta": statistics.fmean(objective_deltas),
            "mean_holdout_robust_bedroc_delta": statistics.fmean(bedroc_deltas),
            "positive_holdout_objective_fold_count": sum(value > 0 for value in objective_deltas),
            "positive_holdout_robust_bedroc_fold_count": sum(value > 0 for value in bedroc_deltas),
        }
    train_gaps = [float(row["train_exact_minus_classical_gap"]) for row in rows]
    return {
        "cell_count": len(rows),
        "positive_train_gap_cell_count": sum(value > TOLERANCE for value in train_gaps),
        "mean_train_gap": statistics.fmean(train_gaps),
        "maximum_train_gap": max(train_gaps),
        "per_target": per_target,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Stage37 cross-target robust functional QUBO screen",
        "",
        "This frozen train-only screen compares the exact optimum of a robust functional set objective with a beam-64, multi-start, add/drop/swap classical search.",
        "No fresh-validation or test row was read.",
        "",
        "| Target | Fold | Exact subset | Classical subset | Train gap | Holdout objective delta | Holdout BEDROC delta |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in result["cells"]:
        lines.append(
            f"| {row['target_id']} | {row['outer_fold']} | {row['exact_subset']} | {row['classical_subset']} | "
            f"{float(row['train_exact_minus_classical_gap']):.8f} | {float(row['holdout_objective_delta']):+.6f} | "
            f"{float(row['holdout_robust_bedroc_delta']):+.6f} |"
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Functional objective support gate: **{'PASS' if decision['functional_objective_supported'] else 'NO-GO'}**.",
            f"- Sparse auxiliary-QUBO encoding authorized: `{decision['stage38_sparse_auxiliary_qubo_authorized']}`.",
            f"- Quantum hardware authorized: `{decision['quantum_hardware_authorized']}`.",
            "",
            "## Boundary",
            "",
            result["interpretation_boundary"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage37 implementation identity differs")
    parent_path = verified(root, config["inputs"]["stage19e_config"])
    for key in ("stage19e_audit", "stage19h_audit", "stage36c_audit"):
        verified(root, config["inputs"][key])
    parent = read_json(parent_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage37 outputs exist; pass --overwrite")

    screen = config["screen"]
    objective = config["objective"]
    subset_list: list[tuple[int, ...]] | None = None
    fold_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []

    for target_id, target_spec in parent["targets"].items():
        print(f"loading_target={target_id}", flush=True)
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        ligand_ids = sorted(row["ligand_id"] for row in ligands)
        manifest = {row["ligand_id"]: row for row in ligands}
        assignments = make_frozen_group_folds(ligands, int(screen["outer_fold_count"]), int(screen["fold_seed"]))
        subsets = enumerate_subsets(
            len(receptor_ids), int(objective["minimum_subset_size"]), int(objective["maximum_subset_size"])
        )
        if subset_list is None:
            subset_list = subsets
        elif subset_list != subsets:
            raise ValueError("target receptor counts differ")
        for ligand_id in ligand_ids:
            row = manifest[ligand_id]
            fold_rows.append(
                {
                    "target_id": target_id,
                    "ligand_id": ligand_id,
                    "label": row["label"],
                    "split_group_id": row["split_group_id"],
                    "outer_fold": assignments[ligand_id],
                }
            )

        all_ids = set(ligand_ids)
        all_labels = np.asarray([int(manifest[ligand_id]["label"] == "active") for ligand_id in ligand_ids], dtype=int)
        for outer_fold in range(int(screen["outer_fold_count"])):
            train_ids = {ligand_id for ligand_id in ligand_ids if assignments[ligand_id] != outer_fold}
            holdout_ids = all_ids - train_ids
            train_mask = np.asarray([ligand_id in train_ids for ligand_id in ligand_ids])
            holdout_mask = ~train_mask
            ranks = build_rank_arrays(ligand_ids, train_ids, receptor_ids, matrices)
            train_values, train_components = score_landscape(ranks[:, train_mask, :], all_labels[train_mask], subsets, objective)
            exact_index = min(range(len(subsets)), key=lambda index: subset_key(train_values, subsets, index))
            exact_subset = subsets[exact_index]
            classical_subset, search_record = strong_classical_search(
                subsets,
                train_values,
                len(receptor_ids),
                int(objective["minimum_subset_size"]),
                int(objective["maximum_subset_size"]),
                int(screen["classical_beam_width"]),
            )
            subset_index = {subset: index for index, subset in enumerate(subsets)}
            classical_index = subset_index[classical_subset]
            holdout_values, holdout_components = score_landscape(
                ranks[:, holdout_mask, :], all_labels[holdout_mask], [exact_subset, classical_subset], objective
            )
            exact_bedroc = bedroc_metrics(ranks[:, holdout_mask, :], all_labels[holdout_mask], exact_subset, float(screen["bedroc_alpha"]))
            classical_bedroc = bedroc_metrics(ranks[:, holdout_mask, :], all_labels[holdout_mask], classical_subset, float(screen["bedroc_alpha"]))
            cell = {
                "target_id": target_id,
                "outer_fold": outer_fold,
                "train_ligand_count": int(train_mask.sum()),
                "holdout_ligand_count": int(holdout_mask.sum()),
                "state_count": len(subsets),
                "exact_subset": subset_names(exact_subset, receptor_ids),
                "exact_subset_size": len(exact_subset),
                "classical_subset": subset_names(classical_subset, receptor_ids),
                "classical_subset_size": len(classical_subset),
                "train_exact_objective": float(train_values[exact_index]),
                "train_classical_objective": float(train_values[classical_index]),
                "train_exact_minus_classical_gap": float(train_values[exact_index] - train_values[classical_index]),
                "holdout_exact_objective": float(holdout_values[0]),
                "holdout_classical_objective": float(holdout_values[1]),
                "holdout_objective_delta": float(holdout_values[0] - holdout_values[1]),
                "holdout_robust_bedroc_delta": float(exact_bedroc["robust_bedroc_composite"] - classical_bedroc["robust_bedroc_composite"]),
                **{f"classical_{key}": value for key, value in search_record.items()},
                **prefixed("holdout_exact", exact_bedroc),
                **prefixed("holdout_classical", classical_bedroc),
            }
            cell_rows.append(cell)
            for method, subset, train_index, holdout_index in (
                ("exact_functional_objective", exact_subset, exact_index, 0),
                ("strong_classical_search", classical_subset, classical_index, 1),
            ):
                selection_rows.append(
                    {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "method": method,
                        "selected_subset": subset_names(subset, receptor_ids),
                        "subset_size": len(subset),
                        "train_objective": float(train_values[train_index]),
                        "holdout_objective": float(holdout_values[holdout_index]),
                        **prefixed("train", train_components[train_index]),
                        **prefixed("holdout", holdout_components[holdout_index]),
                    }
                )
            top_indices = sorted(range(len(subsets)), key=lambda index: subset_key(train_values, subsets, index))[: int(screen["top_landscape_rows_per_cell"])]
            for rank, index in enumerate(top_indices, start=1):
                top_rows.append(
                    {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "rank": rank,
                        "subset": subset_names(subsets[index], receptor_ids),
                        "subset_size": len(subsets[index]),
                        "train_objective": float(train_values[index]),
                        **train_components[index],
                    }
                )
            print(
                f"cell={target_id}/fold{outer_fold} states={len(subsets)} gap={cell['train_exact_minus_classical_gap']:.8f}",
                flush=True,
            )

    summary = paired_summary(cell_rows)
    gate = config["support_gate"]
    target_checks = {
        target_id: (
            float(values["mean_holdout_objective_delta"]) >= float(gate["minimum_target_mean_holdout_objective_delta"]) - TOLERANCE
            and float(values["mean_holdout_robust_bedroc_delta"]) >= float(gate["minimum_target_mean_holdout_robust_bedroc_delta"]) - TOLERANCE
        )
        for target_id, values in summary["per_target"].items()
    }
    supported = (
        int(summary["positive_train_gap_cell_count"]) >= int(gate["minimum_positive_train_gap_cells"])
        and float(summary["mean_train_gap"]) >= float(gate["minimum_mean_train_gap"]) - TOLERANCE
        and all(target_checks.values())
    )
    decision = {
        "functional_objective_supported": supported,
        "target_holdout_checks": target_checks,
        "stage38_sparse_auxiliary_qubo_authorized": supported,
        "new_docking_authorized": False,
        "quantum_hardware_authorized": False,
    }
    write_csv(outputs["fold_assignments_csv"], fold_rows)
    write_csv(outputs["cell_metrics_csv"], cell_rows)
    write_csv(outputs["selection_metrics_csv"], selection_rows)
    write_csv(outputs["top_landscape_csv"], top_rows)
    model_record = {
        "schema_version": "1.0",
        "status": "stage37_robust_functional_objective_frozen",
        "objective": objective,
        "qubo_encoding_note": config["qubo_encoding_note"],
        "classical_comparator": config["classical_comparator"],
    }
    write_json(outputs["model_record_json"], model_record)
    result = {
        "schema_version": "1.0",
        "status": "stage37_cross_target_robust_functional_qubo_complete",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation),
        "inputs": {key: descriptor(root, verified(root, value)) for key, value in config["inputs"].items()},
        "objective": objective,
        "cells": cell_rows,
        "summary": summary,
        "decision": decision,
        "data_boundary": {
            "mk14_train_rows_read": 696,
            "pparg_train_rows_read": 668,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_report(outputs["report_md"], result)
    result["outputs"] = {
        key: descriptor(root, path)
        for key, path in outputs.items()
        if key not in {"result_json", "audit_json"}
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage37_cross_target_robust_functional_qubo.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run(rooted(root, args.config), root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
