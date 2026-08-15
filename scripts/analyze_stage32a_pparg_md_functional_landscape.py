"""Analyze exact and strong-greedy Stage32 PPARG MD functional landscapes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage19e_cross_target_qubo_v2 import vectorized_bedroc


SCENARIOS = ("primary", "sensitivity", "seed0", "seed1", "seed2")
SEEDS = ("seed0", "seed1", "seed2")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {"path": path.resolve().relative_to(root.resolve()).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def fold_assignments(ligands: list[dict[str, str]], config: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    fold_config = config["folds"]
    assignment = np.full(len(ligands), -1, dtype=int)
    rows = []
    for label in ("active", "decoy"):
        indices = [index for index, row in enumerate(ligands) if row["label"] == label]
        ordered = sorted(
            indices,
            key=lambda index: (
                hashlib.sha256(f"{fold_config['assignment_seed']}|{label}|{ligands[index]['split_group_id']}".encode("ascii")).hexdigest(),
                ligands[index]["ligand_id"],
            ),
        )
        if len({ligands[index]["split_group_id"] for index in ordered}) != len(ordered):
            raise ValueError(f"Stage32a {label} scaffold groups are not unique")
        for rank, index in enumerate(ordered):
            assignment[index] = rank % int(fold_config["fold_count"])
    for index, ligand in enumerate(ligands):
        rows.append({"ligand_id": ligand["ligand_id"], "label": ligand["label"], "split_group_id": ligand["split_group_id"], "outer_fold": int(assignment[index])})
    for fold in range(int(fold_config["fold_count"])):
        labels = Counter(ligands[index]["label"] for index in np.flatnonzero(assignment == fold))
        if labels != Counter({"active": 20, "decoy": 20}):
            raise ValueError(f"Stage32a fold {fold} is not balanced: {labels}")
    return assignment, rows


def load_matrices(root: Path, config: dict[str, Any], ligands: list[dict[str, str]], receptor_ids: list[str]) -> dict[str, np.ndarray]:
    inputs = config["inputs"]
    ligand_index = {row["ligand_id"]: index for index, row in enumerate(ligands)}
    receptor_index = {value: index for index, value in enumerate(receptor_ids)}
    matrices = {scenario: np.empty((len(ligands), len(receptor_ids)), dtype=float) for scenario in SCENARIOS}
    for scenario, key in (("primary", "median_matrix_csv"), ("sensitivity", "minimum_matrix_csv")):
        rows = read_csv(root / inputs[key])
        for row in rows:
            i = ligand_index[row["ligand_id"]]
            matrices[scenario][i] = [float(row[receptor]) for receptor in receptor_ids]
    scores = read_csv(root / inputs["scores_csv"])
    seen = {seed: set() for seed in SEEDS}
    for row in scores:
        seed = row["seed_id"]
        i = ligand_index[row["ligand_id"]]
        j = receptor_index[row["receptor_id"]]
        matrices[seed][i, j] = float(row["gpu_score"])
        seen[seed].add((i, j))
    expected = len(ligands) * len(receptor_ids)
    if any(len(seen[seed]) != expected for seed in SEEDS) or any(not np.all(np.isfinite(matrix)) for matrix in matrices.values()):
        raise ValueError("Stage32a matrix coverage differs")
    return matrices


def normalize_from_train(matrix: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    output = np.empty_like(matrix, dtype=float)
    train_count = int(train_mask.sum())
    for receptor in range(matrix.shape[1]):
        frozen = np.sort(matrix[train_mask, receptor], kind="stable")
        values = matrix[:, receptor]
        left = np.searchsorted(frozen, values, side="left")
        right = np.searchsorted(frozen, values, side="right")
        output[:, receptor] = (left + 0.5 * (right - left) + 0.5) / (train_count + 1.0)
    return output


def build_catalog(receptor_count: int, sizes: list[int]) -> dict[str, Any]:
    masks_by_size = {
        size: np.asarray([sum(1 << index for index in subset) for subset in itertools.combinations(range(receptor_count), size)], dtype=np.int32)
        for size in sizes
    }
    all_masks = np.asarray([0, *[int(mask) for size in sizes for mask in masks_by_size[size]]], dtype=np.int32)
    column = np.full(1 << receptor_count, -1, dtype=np.int32)
    column[all_masks] = np.arange(len(all_masks), dtype=np.int32)
    added_bit = np.full(1 << receptor_count, -1, dtype=np.int16)
    for mask in all_masks[1:]:
        bit = int(mask) & -int(mask)
        added_bit[int(mask)] = bit.bit_length() - 1
    return {"masks_by_size": masks_by_size, "all_masks": all_masks, "nonempty_masks": all_masks[1:], "column": column, "added_bit": added_bit}


def landscape(normalized: dict[str, np.ndarray], labels: np.ndarray, row_mask: np.ndarray, catalog: dict[str, Any], alpha: float) -> dict[str, np.ndarray]:
    count = len(catalog["all_masks"])
    output: dict[str, np.ndarray] = {}
    for scenario in SCENARIOS:
        matrix = normalized[scenario][row_mask]
        aggregate = np.full((matrix.shape[0], count), np.inf, dtype=float)
        for mask in catalog["nonempty_masks"]:
            value = int(mask)
            parent = value & (value - 1)
            bit = catalog["added_bit"][value]
            aggregate[:, catalog["column"][value]] = np.minimum(aggregate[:, catalog["column"][parent]], matrix[:, bit])
        metric = np.empty(count - 1, dtype=float)
        for begin in range(1, count, 1024):
            end = min(count, begin + 1024)
            metric[begin - 1 : end - 1] = vectorized_bedroc(aggregate[:, begin:end], labels[row_mask], alpha)
        output[scenario] = metric
    seed_values = np.vstack([output[seed] for seed in SEEDS])
    output["mean_seed"] = seed_values.mean(axis=0)
    output["worst_seed"] = seed_values.min(axis=0)
    output["robust"] = (output["primary"] + output["mean_seed"] + output["worst_seed"]) / 3.0
    return output


def mask_subset(mask: int, receptor_ids: list[str]) -> tuple[str, ...]:
    return tuple(sorted(receptor_ids[index] for index in range(len(receptor_ids)) if mask & (1 << index)))


def value(values: dict[str, np.ndarray], catalog: dict[str, Any], mask: int, metric: str = "robust") -> float:
    return float(values[metric][int(catalog["column"][mask]) - 1])


def best_mask(values: dict[str, np.ndarray], catalog: dict[str, Any], receptor_ids: list[str], masks: list[int] | np.ndarray, tolerance: float) -> int:
    best = -1
    best_value = -np.inf
    for candidate in masks:
        candidate = int(candidate)
        local = value(values, catalog, candidate)
        if local > best_value + tolerance or (abs(local - best_value) <= tolerance and (best < 0 or mask_subset(candidate, receptor_ids) < mask_subset(best, receptor_ids))):
            best = candidate
            best_value = local
    return best


def one_swap(mask: int, values: dict[str, np.ndarray], catalog: dict[str, Any], receptor_ids: list[str], tolerance: float) -> int:
    while True:
        selected = [index for index in range(len(receptor_ids)) if mask & (1 << index)]
        absent = [index for index in range(len(receptor_ids)) if not mask & (1 << index)]
        candidates = [mask, *[(mask ^ (1 << remove)) | (1 << add) for remove in selected for add in absent]]
        candidate = best_mask(values, catalog, receptor_ids, candidates, tolerance)
        candidate_value = value(values, catalog, candidate)
        current_value = value(values, catalog, mask)
        lex_improves = abs(candidate_value - current_value) <= tolerance and mask_subset(candidate, receptor_ids) < mask_subset(mask, receptor_ids)
        if candidate_value > current_value + tolerance or lex_improves:
            mask = candidate
        else:
            return mask


def strong_greedy(size: int, values: dict[str, np.ndarray], catalog: dict[str, Any], receptor_ids: list[str], tolerance: float) -> int:
    finals = []
    for start in range(len(receptor_ids)):
        mask = 1 << start
        while mask.bit_count() < size:
            candidates = [mask | (1 << index) for index in range(len(receptor_ids)) if not mask & (1 << index)]
            mask = best_mask(values, catalog, receptor_ids, candidates, tolerance)
        finals.append(one_swap(mask, values, catalog, receptor_ids, tolerance))
    return best_mask(values, catalog, receptor_ids, finals, tolerance)


def metric_record(prefix: str, values: dict[str, np.ndarray], catalog: dict[str, Any], mask: int) -> dict[str, float]:
    return {f"{prefix}_{metric}": value(values, catalog, mask, metric) for metric in ("primary", "sensitivity", "mean_seed", "worst_seed", "robust")}


def analyze(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    matrix_audit = read_json(root / config["outputs"]["matrix_audit_json"])
    if matrix_audit.get("status") != "stage32_pparg_md_functional_pilot_matrix_audit_ok":
        raise ValueError("Stage32 matrix audit gate differs")
    ligands = read_csv(root / config["inputs"]["ligand_manifest"])
    receptors = read_csv(root / config["inputs"]["prepared_receptor_manifest"])
    receptor_ids = [row["conformer_id"] for row in receptors]
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=np.int8)
    assignments, assignment_rows = fold_assignments(ligands, config)
    matrices = load_matrices(root, config, ligands, receptor_ids)
    sizes = [int(value) for value in config["landscape"]["subset_sizes"]]
    catalog = build_catalog(len(receptor_ids), sizes)
    alpha = float(config["landscape"]["bedroc_alpha"])
    tolerance = float(config["landscape"]["objective_tolerance"])
    comparison_rows = []
    for fold in range(int(config["folds"]["fold_count"])):
        holdout = assignments == fold
        train = ~holdout
        normalized = {scenario: normalize_from_train(matrix, train) for scenario, matrix in matrices.items()}
        train_values = landscape(normalized, labels, train, catalog, alpha)
        holdout_values = landscape(normalized, labels, holdout, catalog, alpha)
        for size in sizes:
            exact = best_mask(train_values, catalog, receptor_ids, catalog["masks_by_size"][size], tolerance)
            greedy = strong_greedy(size, train_values, catalog, receptor_ids, tolerance)
            row: dict[str, Any] = {
                "outer_fold": fold,
                "train_count": int(train.sum()),
                "holdout_count": int(holdout.sum()),
                "subset_size": size,
                "exact_subset": "+".join(mask_subset(exact, receptor_ids)),
                "strong_greedy_subset": "+".join(mask_subset(greedy, receptor_ids)),
                "subset_differs": exact != greedy,
                "train_exact_minus_greedy_robust": value(train_values, catalog, exact) - value(train_values, catalog, greedy),
                "holdout_exact_minus_greedy_robust": value(holdout_values, catalog, exact) - value(holdout_values, catalog, greedy),
            }
            row.update(metric_record("train_exact", train_values, catalog, exact))
            row.update(metric_record("train_greedy", train_values, catalog, greedy))
            row.update(metric_record("holdout_exact", holdout_values, catalog, exact))
            row.update(metric_record("holdout_greedy", holdout_values, catalog, greedy))
            comparison_rows.append(row)
    gate = config["stage33_gate"]
    aggregate_rows = []
    qualifying_sizes = []
    for size in sizes:
        rows = [row for row in comparison_rows if int(row["subset_size"]) == size]
        mean_train_gap = statistics.fmean(float(row["train_exact_minus_greedy_robust"]) for row in rows)
        mean_holdout_gain = statistics.fmean(float(row["holdout_exact_minus_greedy_robust"]) for row in rows)
        positive_folds = sum(float(row["train_exact_minus_greedy_robust"]) > float(gate["positive_gap_tolerance"]) for row in rows)
        qualifies = mean_train_gap >= float(gate["minimum_mean_outer_train_greedy_gap"]) and positive_folds >= int(gate["minimum_positive_gap_folds"]) and mean_holdout_gain >= float(gate["minimum_mean_outer_holdout_gain"])
        if qualifies:
            qualifying_sizes.append(size)
        aggregate_rows.append({
            "subset_size": size,
            "mean_outer_train_exact_minus_greedy_robust": mean_train_gap,
            "maximum_outer_train_exact_minus_greedy_robust": max(float(row["train_exact_minus_greedy_robust"]) for row in rows),
            "positive_train_gap_fold_count": positive_folds,
            "subset_difference_fold_count": sum(str(row["subset_differs"]).lower() == "true" or row["subset_differs"] is True for row in rows),
            "mean_outer_holdout_exact_minus_greedy_robust": mean_holdout_gain,
            "mean_holdout_exact_robust": statistics.fmean(float(row["holdout_exact_robust"]) for row in rows),
            "mean_holdout_greedy_robust": statistics.fmean(float(row["holdout_greedy_robust"]) for row in rows),
            "stage33_size_gate_passed": qualifies,
        })
    stage33_authorized = len(qualifying_sizes) >= int(gate["minimum_qualifying_subset_sizes"])
    outputs = {key: root / value for key, value in config["outputs"].items()}
    write_csv(outputs["fold_assignments_csv"], assignment_rows)
    write_csv(outputs["fold_comparisons_csv"], comparison_rows)
    write_csv(outputs["size_aggregate_csv"], aggregate_rows)
    decision = {"qualifying_subset_sizes": qualifying_sizes, "qualifying_subset_size_count": len(qualifying_sizes), "stage33_qubo_model_authorized": stage33_authorized, "new_docking_jobs_authorized": False, "quantum_hardware_authorized": False, "route": "freeze_stage33_qubo_model" if stage33_authorized else "stop_md_functional_qubo_efficacy_route"}
    result = {
        "schema_version": "1.0",
        "status": "stage32a_pparg_md_functional_landscape_analysis_complete",
        "config": descriptor(root, config_path),
        "matrix_audit": descriptor(root, root / config["outputs"]["matrix_audit_json"]),
        "coverage": {"receptor_count": len(receptor_ids), "ligand_count": len(ligands), "fold_count": 4, "subset_sizes": sizes, "nonempty_subset_count_per_fold": len(catalog["nonempty_masks"]), "fold_comparison_count": len(comparison_rows)},
        "size_aggregate": aggregate_rows,
        "decision": decision,
        "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key in {"fold_assignments_csv", "fold_comparisons_csv", "size_aggregate_csv"}},
        "data_boundary": {"train_rows_read": 160, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    outputs["result_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["result_json"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    report = [
        "# Stage 32a: PPARG MD functional landscape",
        "",
        "| k | Mean train exact-greedy | Positive folds | Mean holdout exact | Mean holdout exact-greedy | Gate |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in aggregate_rows:
        report.append(f"| {row['subset_size']} | {row['mean_outer_train_exact_minus_greedy_robust']:.6f} | {row['positive_train_gap_fold_count']}/4 | {row['mean_holdout_exact_robust']:.6f} | {row['mean_outer_holdout_exact_minus_greedy_robust']:+.6f} | {'PASS' if row['stage33_size_gate_passed'] else 'NO-GO'} |")
    report += ["", f"Qualifying subset sizes: **{qualifying_sizes or 'none'}**.", f"Stage33 QUBO model authorization: **{'PASS' if stage33_authorized else 'NO-GO'}**.", "", config["interpretation_boundary"]]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32a_pparg_md_functional_landscape_analysis.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    analyze(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
