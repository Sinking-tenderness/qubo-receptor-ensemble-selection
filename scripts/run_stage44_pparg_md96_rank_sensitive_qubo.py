"""Evaluate the frozen Stage42f rank-sensitive QUBO on PPARG MD-96."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage42d_bace1_large_pool_qubo_screen import bedroc_metrics, rank_cube
from scripts.run_stage42f_bace1_rank_sensitive_pair_qubo import (
    classical_by_size,
    direct_greedy_by_size,
    exact_by_size,
    pair_coefficients,
    qubo_value,
    subset_name,
)


TOLERANCE = 1e-12
SEED_IDS = ("seed0", "seed1", "seed2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / value["path"]
    if not path.is_file() or sha256(path) != value["sha256"].upper():
        raise ValueError(f"Stage44 input identity differs: {path}")
    return path


def build_score_cube(
    rows: list[dict[str, str]], ligand_ids: list[str], receptor_ids: list[str]
) -> np.ndarray:
    ligand_index = {value: index for index, value in enumerate(ligand_ids)}
    receptor_index = {value: index for index, value in enumerate(receptor_ids)}
    seed_index = {value: index for index, value in enumerate(SEED_IDS)}
    cube = np.full((3, len(ligand_ids), len(receptor_ids)), np.nan, dtype=float)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["seed_id"], row["ligand_id"], row["receptor_id"])
        if key in seen:
            raise ValueError(f"duplicate Stage43 score key: {key}")
        seen.add(key)
        cube[seed_index[key[0]], ligand_index[key[1]], receptor_index[key[2]]] = float(row["gpu_score"])
    expected = 3 * len(ligand_ids) * len(receptor_ids)
    if len(seen) != expected or not np.isfinite(cube).all():
        raise ValueError("Stage43 score cube is incomplete")
    return cube


def anneal_fixed_k(
    singleton: np.ndarray,
    complement: np.ndarray,
    size: int,
    sampler: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    receptor_count = len(singleton)
    read_count = int(sampler["reads_per_batch"])
    pair_denominator = math.comb(size, 2) if size > 1 else 1
    coefficient_scale = max(
        float(np.max(np.abs(singleton))) / size,
        float(np.max(np.abs(complement))) / pair_denominator if size > 1 else 0.0,
        1e-8,
    )
    start_temperature = coefficient_scale * float(sampler["temperature_start_scale"])
    end_temperature = coefficient_scale * float(sampler["temperature_end_scale"])
    sweeps = int(sampler["sweeps_per_read"])
    selected = np.vstack(
        [rng.choice(receptor_count, size=size, replace=False) for _ in range(read_count)]
    ).astype(int)
    row_indices = np.arange(read_count)
    values = singleton[selected].mean(axis=1)
    if size > 1:
        for left in range(size):
            for right in range(left + 1, size):
                values += complement[selected[:, left], selected[:, right]] / pair_denominator
    best_selected = selected.copy()
    best_values = values.copy()
    for step in range(sweeps):
        removed_positions = rng.integers(0, size, size=read_count)
        removed = selected[row_indices, removed_positions]
        added = rng.integers(0, receptor_count, size=read_count)
        conflict = np.any(selected == added[:, None], axis=1)
        while np.any(conflict):
            added[conflict] = rng.integers(0, receptor_count, size=int(np.sum(conflict)))
            conflict = np.any(selected == added[:, None], axis=1)
        delta = (singleton[added] - singleton[removed]) / size
        if size > 1:
            for position in range(size):
                retained = position != removed_positions
                delta[retained] += (
                    complement[added[retained], selected[retained, position]]
                    - complement[removed[retained], selected[retained, position]]
                ) / pair_denominator
        fraction = step / max(1, sweeps - 1)
        temperature = start_temperature * (end_temperature / start_temperature) ** fraction
        accept = (delta >= 0.0) | (rng.random(read_count) < np.exp(np.minimum(0.0, delta / temperature)))
        accepted_rows = row_indices[accept]
        selected[accepted_rows, removed_positions[accept]] = added[accept]
        values[accept] += delta[accept]
        improved = values > best_values + TOLERANCE
        best_values[improved] = values[improved]
        best_selected[improved] = selected[improved]
    best_value = float(np.max(best_values))
    tied = np.flatnonzero(np.abs(best_values - best_value) <= TOLERANCE)
    best_subset = min(tuple(sorted(int(value) for value in best_selected[index])) for index in tied)
    read_values = best_values.tolist()
    return {
        "subset": best_subset,
        "objective": best_value,
        "read_best_minimum": min(read_values),
        "read_best_median": statistics.median(read_values),
        "read_best_maximum": max(read_values),
    }


def annealing_by_size(
    singleton: np.ndarray, complement: np.ndarray, maximum_size: int,
    sampler: dict[str, Any], seed_offset: int,
) -> tuple[dict[int, tuple[int, ...]], list[dict[str, Any]]]:
    selected: dict[int, tuple[int, ...]] = {}
    batch_rows: list[dict[str, Any]] = []
    for size in range(1, maximum_size + 1):
        results = []
        for batch in range(int(sampler["batch_count"])):
            result = anneal_fixed_k(
                singleton, complement, size, sampler,
                int(sampler["base_seed"]) + seed_offset + size * 1009 + batch,
            )
            results.append(result)
            batch_rows.append({"subset_size": size, "batch_id": batch, **result, "subset": "+".join(map(str, result["subset"]))})
        best = max(results, key=lambda row: (row["objective"], tuple(-value for value in row["subset"])))
        selected[size] = best["subset"]
        print(f"annealing complete: k={size}", flush=True)
    return selected, batch_rows


def metric_record(
    scope: str, fold: int | str, method: str, size: int, subset: tuple[int, ...],
    singleton: np.ndarray, complement: np.ndarray, ranks: np.ndarray,
    labels: np.ndarray, mask: np.ndarray, receptor_ids: list[str], alpha: float,
) -> dict[str, Any]:
    metrics = bedroc_metrics(ranks[:, mask, :], labels[mask], subset, alpha)
    return {
        "scope": scope, "fold": fold, "method": method, "subset_size": size,
        "selected_subset": subset_name(subset, receptor_ids),
        "qubo_objective": qubo_value(subset, singleton, complement),
        **metrics,
    }


def solve_landscape(
    ranks: np.ndarray, labels: np.ndarray, fit_mask: np.ndarray,
    config: dict[str, Any], seed_offset: int,
) -> dict[str, Any]:
    singleton, complement = pair_coefficients(ranks[:, fit_mask, :], labels[fit_mask], float(config["objective"]["bedroc_alpha"]))
    maximum_size = int(config["objective"]["maximum_subset_size"])
    exact_maximum = int(config["solver"]["exact_maximum_subset_size"])
    exact, exact_states, exact_seconds = exact_by_size(len(singleton), exact_maximum, singleton, complement)
    classical, classical_records = classical_by_size(len(singleton), maximum_size, singleton, complement, int(config["solver"]["beam_width"]))
    direct = direct_greedy_by_size(len(singleton), maximum_size, singleton, complement)
    annealing, annealing_batches = annealing_by_size(singleton, complement, maximum_size, config["solver"]["annealing"], seed_offset)
    return {
        "singleton": singleton, "complement": complement, "exact": exact,
        "classical": classical, "direct": direct, "annealing": annealing,
        "exact_states": exact_states, "exact_seconds": exact_seconds,
        "classical_records": classical_records, "annealing_batches": annealing_batches,
    }


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage44 implementation identity differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    if read_json(inputs["stage43_audit"]).get("status") != "stage43_pparg_md96_unidock_matrix_independent_audit_ok":
        raise ValueError("Stage43 independent matrix audit did not pass")
    stage42f = read_json(inputs["stage42f_result"])
    if stage42f["objective"]["objective_id"] != config["objective"]["source_objective_id"]:
        raise ValueError("Stage44 changed the frozen Stage42f objective identity")
    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage44 outputs exist; pass --overwrite")

    ligands = read_csv(inputs["ligand_manifest"])
    receptors = read_csv(inputs["receptor_manifest"])
    scores = read_csv(inputs["scores"])
    ligand_ids = [row["ligand_id"] for row in ligands]
    receptor_ids = [row["conformer_id"] for row in receptors]
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=int)
    cube = build_score_cube(scores, ligand_ids, receptor_ids)
    folds = make_frozen_group_folds(ligands, int(config["validation"]["outer_fold_count"]), int(config["validation"]["fold_seed"]))
    fold_assignments = [{"ligand_id": row["ligand_id"], "label": row["label"], "split_group_id": row["split_group_id"], "outer_fold": folds[row["ligand_id"]]} for row in ligands]
    metric_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    annealing_rows: list[dict[str, Any]] = []
    alpha = float(config["objective"]["bedroc_alpha"])
    primary_k = int(config["objective"]["primary_replication_subset_size"])
    holdout_k3_gains: list[float] = []
    positive_solver_cells = 0
    started = time.perf_counter()

    for fold in range(int(config["validation"]["outer_fold_count"])):
        train_mask = np.asarray([folds[value] != fold for value in ligand_ids], dtype=bool)
        holdout_mask = ~train_mask
        ranks = rank_cube(cube, train_mask)
        print(f"Stage44 outer fold {fold + 1}/4", flush=True)
        solved = solve_landscape(ranks, labels, train_mask, config, fold * 100000)
        singleton, complement = solved["singleton"], solved["complement"]
        for size in range(1, int(config["objective"]["maximum_subset_size"]) + 1):
            methods = {"strong_classical": solved["classical"][size], "direct_greedy": solved["direct"][size], "simulated_annealing": solved["annealing"][size]}
            if size in solved["exact"]:
                methods["exact"] = solved["exact"][size]
            for method, subset in methods.items():
                metric_rows.append(metric_record("outer_train", fold, method, size, subset, singleton, complement, ranks, labels, train_mask, receptor_ids, alpha))
                metric_rows.append(metric_record("outer_holdout", fold, method, size, subset, singleton, complement, ranks, labels, holdout_mask, receptor_ids, alpha))
            exact_value = qubo_value(solved["exact"][size], singleton, complement) if size in solved["exact"] else None
            classical_value = qubo_value(solved["classical"][size], singleton, complement)
            annealing_value = qubo_value(solved["annealing"][size], singleton, complement)
            gap = annealing_value - classical_value
            positive_solver_cells += int(gap > TOLERANCE)
            solver_rows.append({
                "fold": fold, "subset_size": size,
                "exact_available": size in solved["exact"], "exact_objective": exact_value,
                "strong_classical_objective": classical_value,
                "annealing_objective": annealing_value,
                "annealing_minus_classical_gap": gap,
                "classical_exact_gap": None if exact_value is None else exact_value - classical_value,
                "annealing_exact_gap": None if exact_value is None else exact_value - annealing_value,
                "exact_state_count_k1_to_k3": solved["exact_states"],
                "exact_seconds": solved["exact_seconds"],
            })
        holdout_exact = metric_record("outer_holdout", fold, "exact", primary_k, solved["exact"][primary_k], singleton, complement, ranks, labels, holdout_mask, receptor_ids, alpha)
        holdout_single = metric_record("outer_holdout", fold, "exact", 1, solved["exact"][1], singleton, complement, ranks, labels, holdout_mask, receptor_ids, alpha)
        holdout_k3_gains.append(holdout_exact["robust_bedroc_composite"] - holdout_single["robust_bedroc_composite"])
        for row in solved["annealing_batches"]:
            annealing_rows.append({"fold": fold, **row})

    full_mask = np.ones(len(ligands), dtype=bool)
    full_ranks = rank_cube(cube, full_mask)
    print("Stage44 full-data landscape", flush=True)
    full = solve_landscape(full_ranks, labels, full_mask, config, 900000)
    for size in range(1, int(config["objective"]["maximum_subset_size"]) + 1):
        methods = {"strong_classical": full["classical"][size], "direct_greedy": full["direct"][size], "simulated_annealing": full["annealing"][size]}
        if size in full["exact"]:
            methods["exact"] = full["exact"][size]
        for method, subset in methods.items():
            metric_rows.append(metric_record("full_data", "full", method, size, subset, full["singleton"], full["complement"], full_ranks, labels, full_mask, receptor_ids, alpha))
        exact_value = qubo_value(full["exact"][size], full["singleton"], full["complement"]) if size in full["exact"] else None
        classical_value = qubo_value(full["classical"][size], full["singleton"], full["complement"])
        annealing_value = qubo_value(full["annealing"][size], full["singleton"], full["complement"])
        gap = annealing_value - classical_value
        positive_solver_cells += int(gap > TOLERANCE)
        solver_rows.append({
            "fold": "full", "subset_size": size, "exact_available": size in full["exact"],
            "exact_objective": exact_value, "strong_classical_objective": classical_value,
            "annealing_objective": annealing_value, "annealing_minus_classical_gap": gap,
            "classical_exact_gap": None if exact_value is None else exact_value - classical_value,
            "annealing_exact_gap": None if exact_value is None else exact_value - annealing_value,
            "exact_state_count_k1_to_k3": full["exact_states"], "exact_seconds": full["exact_seconds"],
        })
    for row in full["annealing_batches"]:
        annealing_rows.append({"fold": "full", **row})

    def full_metric(method: str, size: int) -> dict[str, Any]:
        return next(row for row in metric_rows if row["scope"] == "full_data" and row["method"] == method and row["subset_size"] == size)

    primary = full_metric("exact", primary_k)
    best_single = full_metric("exact", 1)
    full_gain = primary["robust_bedroc_composite"] - best_single["robust_bedroc_composite"]
    mean_holdout_gain = statistics.fmean(holdout_k3_gains)
    gates = config["support_gate"]
    application_checks = {
        "minimum_full_k3_over_single_bedroc_gain": full_gain >= float(gates["minimum_full_k3_over_single_bedroc_gain"]),
        "nonnegative_mean_holdout_k3_over_single_gain": mean_holdout_gain >= float(gates["minimum_mean_holdout_k3_over_single_gain"]),
        "minimum_positive_holdout_fold_count": sum(value > 0 for value in holdout_k3_gains) >= int(gates["minimum_positive_holdout_fold_count"]),
    }
    solver_checks = {
        "minimum_positive_solver_gap_cells": positive_solver_cells >= int(gates["minimum_positive_solver_gap_cells"]),
        "exact_k1_to_k3_completed": all(bool(row["exact_available"]) for row in solver_rows if int(row["subset_size"]) <= 3),
    }
    decision = {
        "application_replication_supported": all(application_checks.values()),
        "solver_novelty_supported": all(solver_checks.values()),
        "full_k3_over_single_robust_bedroc_gain": full_gain,
        "mean_outer_holdout_k3_over_single_gain": mean_holdout_gain,
        "positive_outer_holdout_fold_count": sum(value > 0 for value in holdout_k3_gains),
        "positive_solver_gap_cell_count": positive_solver_cells,
        "application_checks": application_checks, "solver_checks": solver_checks,
        "same_data_retuning_authorized": False,
        "fresh_validation_authorized": False,
        "quantum_hardware_authorized": False,
    }
    write_csv(outputs["fold_assignments_csv"], fold_assignments)
    write_csv(outputs["selection_metrics_csv"], metric_rows)
    write_csv(outputs["solver_comparison_csv"], solver_rows)
    write_csv(outputs["annealing_batches_csv"], annealing_rows)
    report = [
        "# Stage44 PPARG MD-96 rank-sensitive QUBO", "",
        f"Primary frozen k=3 exact QUBO subset: {primary['selected_subset']}.",
        f"Full-data robust BEDROC20: k=3 {primary['robust_bedroc_composite']:.6f}; best single {best_single['robust_bedroc_composite']:.6f}; gain {full_gain:+.6f}.",
        f"Mean outer-holdout k=3 gain: {mean_holdout_gain:+.6f}.",
        f"Positive annealing-over-classical cells: {positive_solver_cells}.", "",
        f"Application replication: **{'PASS' if decision['application_replication_supported'] else 'NO-GO'}**.",
        f"Solver novelty: **{'PASS' if decision['solver_novelty_supported'] else 'NO-GO'}**.", "",
        config["interpretation_boundary"], "",
    ]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report), encoding="ascii")
    result = {
        "schema_version": "1.0", "status": "stage44_pparg_md96_rank_sensitive_qubo_complete",
        "config": descriptor(root, config_path), "implementation": descriptor(root, Path(__file__).resolve()),
        "input_statistics": {"receptor_count": 96, "ligand_count": 160, "active_count": int(labels.sum()), "decoy_count": int((labels == 0).sum()), "seed_count": 3},
        "primary_result": {"k3": primary, "best_single": best_single},
        "decision": decision,
        "runtime_seconds": time.perf_counter() - started,
        "data_boundary": {"train_rows_read": 160, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key != "result_json"},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage44_pparg_md96_rank_sensitive_qubo.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
