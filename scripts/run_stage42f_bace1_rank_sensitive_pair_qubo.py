"""Evaluate an analytic rank-sensitive pair QUBO on BACE1 Train-266."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage42d_bace1_large_pool_qubo_screen import (
    bedroc_metrics,
    build_score_cube,
    descriptor,
    rank_cube,
    read_csv,
    read_json,
    verified,
    write_csv,
    write_json,
)


TOLERANCE = 1e-12


def pair_coefficients(
    ranks: np.ndarray, labels: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    active = labels == 1
    decoy = labels == 0
    receptor_count = ranks.shape[2]

    def robust_discrimination(subset: tuple[int, ...]) -> float:
        ensemble = np.min(ranks[:, :, subset], axis=2)
        utility = np.exp(-alpha * ensemble)
        per_seed = utility[:, active].mean(axis=1) - utility[:, decoy].mean(axis=1)
        return float((per_seed.mean() + per_seed.min()) / 2.0)

    singleton = np.asarray(
        [robust_discrimination((index,)) for index in range(receptor_count)], dtype=float
    )
    complement = np.zeros((receptor_count, receptor_count), dtype=float)
    for left in range(receptor_count):
        for right in range(left + 1, receptor_count):
            pair_value = robust_discrimination((left, right))
            complement[left, right] = complement[right, left] = (
                pair_value - max(singleton[left], singleton[right])
            )
    return singleton, complement


def qubo_value(
    subset: tuple[int, ...], singleton: np.ndarray, complement: np.ndarray
) -> float:
    linear = float(np.mean(singleton[list(subset)]))
    if len(subset) == 1:
        return linear
    pair_values = [complement[left, right] for left, right in itertools.combinations(subset, 2)]
    return linear + float(np.mean(pair_values))


def selection_key(
    subset: tuple[int, ...], singleton: np.ndarray, complement: np.ndarray
) -> tuple[Any, ...]:
    return (-qubo_value(subset, singleton, complement), subset)


def exact_by_size(
    receptor_count: int, maximum_size: int, singleton: np.ndarray, complement: np.ndarray
) -> tuple[dict[int, tuple[int, ...]], int, float]:
    started = time.perf_counter()
    output: dict[int, tuple[int, ...]] = {}
    states = 0
    for size in range(1, maximum_size + 1):
        subsets = itertools.combinations(range(receptor_count), size)
        best: tuple[int, ...] | None = None
        for subset in subsets:
            states += 1
            if best is None or selection_key(subset, singleton, complement) < selection_key(best, singleton, complement):
                best = subset
        if best is None:
            raise ValueError("empty fixed-k QUBO landscape")
        output[size] = best
    return output, states, time.perf_counter() - started


def local_swap(
    subset: tuple[int, ...], receptor_count: int, singleton: np.ndarray, complement: np.ndarray
) -> tuple[int, ...]:
    current = subset
    while True:
        selected = set(current)
        neighbors = {
            tuple(sorted((selected - {removed}) | {added}))
            for removed in current
            for added in range(receptor_count)
            if added not in selected
        }
        improving = [
            value
            for value in neighbors
            if qubo_value(value, singleton, complement)
            > qubo_value(current, singleton, complement) + TOLERANCE
        ]
        if not improving:
            return current
        current = min(improving, key=lambda value: selection_key(value, singleton, complement))


def classical_by_size(
    receptor_count: int,
    maximum_size: int,
    singleton: np.ndarray,
    complement: np.ndarray,
    beam_width: int,
) -> tuple[dict[int, tuple[int, ...]], dict[int, dict[str, int]]]:
    starts_by_size: dict[int, set[tuple[int, ...]]] = {
        1: {(index,) for index in range(receptor_count)}
    }
    beam = sorted(starts_by_size[1], key=lambda value: selection_key(value, singleton, complement))[:beam_width]
    for size in range(2, maximum_size + 1):
        expanded = {
            tuple(sorted((*subset, added)))
            for subset in beam
            for added in range(receptor_count)
            if added not in subset
        }
        beam = sorted(expanded, key=lambda value: selection_key(value, singleton, complement))[:beam_width]
        starts_by_size[size] = set(beam)

    for initial in range(receptor_count):
        current = (initial,)
        for size in range(2, maximum_size + 1):
            selected = set(current)
            candidates = [
                tuple(sorted((*current, added)))
                for added in range(receptor_count)
                if added not in selected
            ]
            current = min(candidates, key=lambda value: selection_key(value, singleton, complement))
            starts_by_size[size].add(current)

    selected: dict[int, tuple[int, ...]] = {}
    records: dict[int, dict[str, int]] = {}
    for size in range(1, maximum_size + 1):
        endpoints = {
            local_swap(value, receptor_count, singleton, complement)
            for value in starts_by_size[size]
        }
        selected[size] = min(endpoints, key=lambda value: selection_key(value, singleton, complement))
        records[size] = {
            "start_state_count": len(starts_by_size[size]),
            "local_endpoint_count": len(endpoints),
        }
    return selected, records


def direct_greedy_by_size(
    receptor_count: int, maximum_size: int, singleton: np.ndarray, complement: np.ndarray
) -> dict[int, tuple[int, ...]]:
    current = (int(np.argmax(singleton)),)
    output = {1: current}
    for size in range(2, maximum_size + 1):
        selected = set(current)
        candidates = [
            tuple(sorted((*current, added)))
            for added in range(receptor_count)
            if added not in selected
        ]
        current = min(candidates, key=lambda value: selection_key(value, singleton, complement))
        output[size] = current
    return output


def best_variable_k(
    subsets: dict[int, tuple[int, ...]], singleton: np.ndarray, complement: np.ndarray
) -> tuple[int, ...]:
    return min(
        subsets.values(),
        key=lambda value: (-qubo_value(value, singleton, complement), len(value), value),
    )


def subset_name(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(receptor_ids[index] for index in subset)


def write_report(path: Path, rows: list[dict[str, Any]], decision: dict[str, Any], boundary: str) -> None:
    lines = [
        "# Stage42f BACE1 rank-sensitive pair QUBO",
        "",
        "| k | Exact QUBO | Classical | Gap | Exact BEDROC20 | Classical BEDROC20 |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['subset_size']} | {row['exact_subset']} | {row['classical_subset']} | "
            f"{row['exact_minus_classical_gap']:.8g} | {row['exact_robust_bedroc']:.6f} | "
            f"{row['classical_robust_bedroc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Rank-sensitive pair QUBO supported: **{'PASS' if decision['rank_sensitive_pair_qubo_supported'] else 'NO-GO'}**.",
            "",
            boundary,
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
        raise ValueError("Stage 42f implementation identity differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    diagnosis = read_json(inputs["stage42e_result"])
    if not diagnosis["diagnosis"]["analytically_weighted_rank_sensitive_redesign_authorized"]:
        raise ValueError("Stage 42e did not authorize a rank-sensitive redesign")
    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage 42f outputs exist; pass --overwrite")

    ligands = read_csv(inputs["ligand_manifest"])
    receptors = read_csv(inputs["receptor_manifest"])
    score_rows = read_csv(inputs["scores"])
    ligand_ids = [row["ligand_id"] for row in ligands]
    receptor_ids = [row["conformer_id"] for row in receptors]
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=int)
    scores = build_score_cube(score_rows, ligand_ids, receptor_ids)
    folds = make_frozen_group_folds(
        ligands, int(config["screen"]["outer_fold_count"]), int(config["screen"]["fold_seed"])
    )
    fold_assignments = [
        {"ligand_id": row["ligand_id"], "label": row["label"], "split_group_id": row["split_group_id"], "outer_fold": folds[row["ligand_id"]]}
        for row in ligands
    ]
    maximum_size = int(config["objective"]["maximum_subset_size"])
    alpha = float(config["objective"]["bedroc_alpha"])
    beam_width = int(config["screen"]["classical_beam_width"])
    fold_rows: list[dict[str, Any]] = []
    for fold in range(int(config["screen"]["outer_fold_count"])):
        train_mask = np.asarray([folds[value] != fold for value in ligand_ids])
        holdout_mask = ~train_mask
        ranks = rank_cube(scores, train_mask)
        singleton, complement = pair_coefficients(ranks[:, train_mask, :], labels[train_mask], alpha)
        exact, state_count, elapsed = exact_by_size(len(receptor_ids), maximum_size, singleton, complement)
        classical, records = classical_by_size(len(receptor_ids), maximum_size, singleton, complement, beam_width)
        holdout_singleton, holdout_complement = pair_coefficients(ranks[:, holdout_mask, :], labels[holdout_mask], alpha)
        for size in range(1, maximum_size + 1):
            exact_subset = exact[size]
            classical_subset = classical[size]
            exact_bedroc = bedroc_metrics(ranks[:, holdout_mask, :], labels[holdout_mask], exact_subset, alpha)
            classical_bedroc = bedroc_metrics(ranks[:, holdout_mask, :], labels[holdout_mask], classical_subset, alpha)
            fold_rows.append({
                "outer_fold": fold,
                "subset_size": size,
                "train_ligand_count": int(train_mask.sum()),
                "holdout_ligand_count": int(holdout_mask.sum()),
                "state_count_all_k": state_count,
                "exact_subset": subset_name(exact_subset, receptor_ids),
                "classical_subset": subset_name(classical_subset, receptor_ids),
                "train_exact_qubo": qubo_value(exact_subset, singleton, complement),
                "train_classical_qubo": qubo_value(classical_subset, singleton, complement),
                "train_exact_minus_classical_gap": qubo_value(exact_subset, singleton, complement) - qubo_value(classical_subset, singleton, complement),
                "holdout_exact_qubo": qubo_value(exact_subset, holdout_singleton, holdout_complement),
                "holdout_classical_qubo": qubo_value(classical_subset, holdout_singleton, holdout_complement),
                "holdout_qubo_delta": qubo_value(exact_subset, holdout_singleton, holdout_complement) - qubo_value(classical_subset, holdout_singleton, holdout_complement),
                "holdout_exact_robust_bedroc": exact_bedroc["robust_bedroc_composite"],
                "holdout_classical_robust_bedroc": classical_bedroc["robust_bedroc_composite"],
                "holdout_robust_bedroc_delta": exact_bedroc["robust_bedroc_composite"] - classical_bedroc["robust_bedroc_composite"],
                "exact_enumeration_seconds_all_k": elapsed,
                **records[size],
            })
        print(json.dumps({"outer_fold": fold, "state_count": state_count, "elapsed_seconds": elapsed}), flush=True)

    full_ranks = rank_cube(scores, np.ones(len(ligands), dtype=bool))
    singleton, complement = pair_coefficients(full_ranks, labels, alpha)
    exact, state_count, elapsed = exact_by_size(len(receptor_ids), maximum_size, singleton, complement)
    classical, records = classical_by_size(len(receptor_ids), maximum_size, singleton, complement, beam_width)
    greedy = direct_greedy_by_size(len(receptor_ids), maximum_size, singleton, complement)
    full_rows: list[dict[str, Any]] = []
    for size in range(1, maximum_size + 1):
        exact_subset = exact[size]
        classical_subset = classical[size]
        greedy_subset = greedy[size]
        exact_bedroc = bedroc_metrics(full_ranks, labels, exact_subset, alpha)
        classical_bedroc = bedroc_metrics(full_ranks, labels, classical_subset, alpha)
        greedy_bedroc = bedroc_metrics(full_ranks, labels, greedy_subset, alpha)
        full_rows.append({
            "subset_size": size,
            "exact_subset": subset_name(exact_subset, receptor_ids),
            "classical_subset": subset_name(classical_subset, receptor_ids),
            "direct_greedy_subset": subset_name(greedy_subset, receptor_ids),
            "exact_qubo": qubo_value(exact_subset, singleton, complement),
            "classical_qubo": qubo_value(classical_subset, singleton, complement),
            "direct_greedy_qubo": qubo_value(greedy_subset, singleton, complement),
            "exact_minus_classical_gap": qubo_value(exact_subset, singleton, complement) - qubo_value(classical_subset, singleton, complement),
            "exact_robust_bedroc": exact_bedroc["robust_bedroc_composite"],
            "classical_robust_bedroc": classical_bedroc["robust_bedroc_composite"],
            "direct_greedy_robust_bedroc": greedy_bedroc["robust_bedroc_composite"],
            **records[size],
        })

    positive_gap_cells = sum(row["train_exact_minus_classical_gap"] > TOLERANCE for row in fold_rows)
    positive_gap_folds = len({row["outer_fold"] for row in fold_rows if row["train_exact_minus_classical_gap"] > TOLERANCE})
    mean_holdout_qubo_delta = statistics.fmean(row["holdout_qubo_delta"] for row in fold_rows)
    mean_holdout_bedroc_delta = statistics.fmean(row["holdout_robust_bedroc_delta"] for row in fold_rows)
    full_gap_rows = [row for row in full_rows if row["exact_minus_classical_gap"] > TOLERANCE]
    best_exact_bedroc = max(row["exact_robust_bedroc"] for row in full_rows)
    best_single_bedroc = full_rows[0]["exact_robust_bedroc"]
    gate = config["support_gate"]
    checks = {
        "minimum_positive_gap_folds": positive_gap_folds >= int(gate["minimum_positive_gap_folds"]),
        "minimum_positive_gap_cells": positive_gap_cells >= int(gate["minimum_positive_gap_cells"]),
        "full_data_positive_gap_exists": len(full_gap_rows) >= int(gate["minimum_full_data_positive_gap_k_count"]),
        "nonnegative_mean_holdout_qubo_delta": mean_holdout_qubo_delta >= float(gate["minimum_mean_holdout_qubo_delta"]),
        "nonnegative_mean_holdout_bedroc_delta": mean_holdout_bedroc_delta >= float(gate["minimum_mean_holdout_bedroc_delta"]),
        "minimum_combination_over_single_bedroc_gain": best_exact_bedroc - best_single_bedroc >= float(gate["minimum_combination_over_single_bedroc_gain"]),
    }
    supported = all(checks.values())
    decision = {
        "rank_sensitive_pair_qubo_supported": supported,
        "positive_gap_fold_count": positive_gap_folds,
        "positive_gap_cell_count": positive_gap_cells,
        "full_data_positive_gap_k_values": [row["subset_size"] for row in full_gap_rows],
        "mean_holdout_qubo_delta": mean_holdout_qubo_delta,
        "mean_holdout_bedroc_delta": mean_holdout_bedroc_delta,
        "best_combination_over_single_bedroc_gain": best_exact_bedroc - best_single_bedroc,
        "checks": checks,
        "fresh_validation_authorized": supported,
        "sparse_qubo_encoding_authorized": supported,
        "quantum_hardware_authorized": False,
        "same_data_retuning_authorized": False,
    }
    write_csv(outputs["fold_assignments_csv"], fold_assignments)
    write_csv(outputs["fold_metrics_csv"], fold_rows)
    write_csv(outputs["full_metrics_csv"], full_rows)
    write_report(outputs["report_md"], full_rows, decision, config["interpretation_boundary"])
    result = {
        "schema_version": "1.0",
        "status": "stage42f_bace1_rank_sensitive_pair_qubo_complete",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, Path(__file__).resolve()),
        "objective": config["objective"],
        "input_statistics": {
            "receptor_count": 34,
            "ligand_count": 266,
            "seed_count": 3,
            "state_count_k1_to_k6": state_count,
            "exact_full_enumeration_seconds": elapsed,
        },
        "decision": decision,
        "data_boundary": {
            "train_rows_read": 266,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            key: descriptor(root, path)
            for key, path in outputs.items()
            if key != "result_json"
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage42f_bace1_rank_sensitive_pair_qubo.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
