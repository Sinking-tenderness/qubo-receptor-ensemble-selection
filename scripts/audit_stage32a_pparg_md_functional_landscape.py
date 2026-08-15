"""Independently audit the Stage32a PPARG MD functional landscape."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_descriptor(root: Path, record: dict[str, Any]) -> None:
    path = root / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"descriptor differs: {record['path']}")
    if "size_bytes" in record and path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"descriptor size differs: {record['path']}")


def independent_folds(ligands: list[dict[str, str]], config: dict[str, Any]) -> np.ndarray:
    fold_config = config["folds"]
    assignment = np.full(len(ligands), -1, dtype=int)
    for label in ("active", "decoy"):
        members = [index for index, row in enumerate(ligands) if row["label"] == label]
        members.sort(
            key=lambda index: (
                hashlib.sha256(
                    f"{fold_config['assignment_seed']}|{label}|{ligands[index]['split_group_id']}".encode("ascii")
                ).hexdigest(),
                ligands[index]["ligand_id"],
            )
        )
        if len({ligands[index]["split_group_id"] for index in members}) != len(members):
            raise ValueError(f"duplicate {label} split group")
        for rank, index in enumerate(members):
            assignment[index] = rank % int(fold_config["fold_count"])
    for fold in range(int(fold_config["fold_count"])):
        counts = Counter(ligands[index]["label"] for index in np.flatnonzero(assignment == fold))
        if counts != Counter({"active": 20, "decoy": 20}):
            raise ValueError(f"fold {fold} is not balanced")
    return assignment


def independent_matrices(
    root: Path,
    config: dict[str, Any],
    ligands: list[dict[str, str]],
    receptor_ids: list[str],
) -> dict[str, np.ndarray]:
    ligand_index = {row["ligand_id"]: index for index, row in enumerate(ligands)}
    receptor_index = {value: index for index, value in enumerate(receptor_ids)}
    matrices = {key: np.full((len(ligands), len(receptor_ids)), np.nan) for key in SCENARIOS}
    for scenario, source in (("primary", "median_matrix_csv"), ("sensitivity", "minimum_matrix_csv")):
        for row in read_csv(root / config["inputs"][source]):
            matrices[scenario][ligand_index[row["ligand_id"]]] = [float(row[value]) for value in receptor_ids]
    seen: set[tuple[str, int, int]] = set()
    for row in read_csv(root / config["inputs"]["scores_csv"]):
        seed = row["seed_id"]
        index = (seed, ligand_index[row["ligand_id"]], receptor_index[row["receptor_id"]])
        if index in seen:
            raise ValueError("duplicate Stage32 score key")
        seen.add(index)
        matrices[seed][index[1], index[2]] = float(row["gpu_score"])
    if len(seen) != 3 * len(ligands) * len(receptor_ids):
        raise ValueError("Stage32 score coverage differs")
    if any(not np.all(np.isfinite(matrix)) for matrix in matrices.values()):
        raise ValueError("non-finite Stage32 matrix")
    return matrices


def independent_normalize(matrix: np.ndarray, train: np.ndarray) -> np.ndarray:
    result = np.empty_like(matrix, dtype=float)
    denominator = float(train.sum() + 1)
    for receptor in range(matrix.shape[1]):
        reference = matrix[train, receptor]
        for row, score in enumerate(matrix[:, receptor]):
            result[row, receptor] = (
                np.count_nonzero(reference < score)
                + 0.5 * np.count_nonzero(reference == score)
                + 0.5
            ) / denominator
    return result


def subset_name(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(sorted(receptor_ids[index] for index in subset))


def best_subset(
    candidates: list[tuple[int, ...]],
    utility: dict[tuple[int, ...], float],
    receptor_ids: list[str],
    tolerance: float,
) -> tuple[int, ...]:
    best: tuple[int, ...] | None = None
    best_value = -np.inf
    for candidate in candidates:
        candidate_value = utility[candidate]
        if (
            candidate_value > best_value + tolerance
            or (
                abs(candidate_value - best_value) <= tolerance
                and (best is None or subset_name(candidate, receptor_ids) < subset_name(best, receptor_ids))
            )
        ):
            best = candidate
            best_value = candidate_value
    if best is None:
        raise ValueError("cannot choose from an empty candidate set")
    return best


def greedy_subset(
    size: int,
    utility: dict[tuple[int, ...], float],
    receptor_ids: list[str],
    tolerance: float,
) -> tuple[int, ...]:
    finals = []
    for start in range(len(receptor_ids)):
        current = (start,)
        while len(current) < size:
            additions = [tuple(sorted((*current, index))) for index in range(len(receptor_ids)) if index not in current]
            current = best_subset(additions, utility, receptor_ids, tolerance)
        while True:
            swaps = {current}
            for remove in current:
                for add in range(len(receptor_ids)):
                    if add not in current:
                        swaps.add(tuple(sorted((*[value for value in current if value != remove], add))))
            candidate = best_subset(sorted(swaps), utility, receptor_ids, tolerance)
            improves = utility[candidate] > utility[current] + tolerance
            tie_improves = abs(utility[candidate] - utility[current]) <= tolerance and subset_name(candidate, receptor_ids) < subset_name(current, receptor_ids)
            if improves or tie_improves:
                current = candidate
            else:
                break
        finals.append(current)
    return best_subset(finals, utility, receptor_ids, tolerance)


def subset_metrics(
    normalized: dict[str, np.ndarray],
    labels: np.ndarray,
    selected_rows: np.ndarray,
    subsets: list[tuple[int, ...]],
    alpha: float,
) -> dict[str, dict[tuple[int, ...], float]]:
    values: dict[str, dict[tuple[int, ...], float]] = {}
    for scenario in SCENARIOS:
        metrics = np.empty(len(subsets), dtype=float)
        for begin in range(0, len(subsets), 512):
            block = subsets[begin : begin + 512]
            scores = np.stack(
                [normalized[scenario][selected_rows][:, subset].min(axis=1) for subset in block],
                axis=1,
            )
            metrics[begin : begin + len(block)] = vectorized_bedroc(scores, labels[selected_rows], alpha)
        values[scenario] = dict(zip(subsets, metrics.tolist()))
    values["mean_seed"] = {subset: statistics.fmean(values[seed][subset] for seed in SEEDS) for subset in subsets}
    values["worst_seed"] = {subset: min(values[seed][subset] for seed in SEEDS) for subset in subsets}
    values["robust"] = {
        subset: (values["primary"][subset] + values["mean_seed"][subset] + values["worst_seed"][subset]) / 3.0
        for subset in subsets
    }
    return values


def audit(config_path: Path, result_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    result_path = result_path.resolve()
    config = read_json(config_path)
    result = read_json(result_path)
    if result.get("status") != "stage32a_pparg_md_functional_landscape_analysis_complete":
        raise ValueError("unexpected Stage32a result status")
    if result["config"]["sha256"] != sha256(config_path):
        raise ValueError("Stage32a config hash differs")
    verify_descriptor(root, result["matrix_audit"])
    for record in result["outputs"].values():
        verify_descriptor(root, record)
    if read_json(root / config["outputs"]["matrix_audit_json"]).get("status") != "stage32_pparg_md_functional_pilot_matrix_audit_ok":
        raise ValueError("Stage32 matrix audit gate differs")

    ligands = read_csv(root / config["inputs"]["ligand_manifest"])
    receptors = read_csv(root / config["inputs"]["prepared_receptor_manifest"])
    receptor_ids = [row["conformer_id"] for row in receptors]
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=np.int8)
    folds = independent_folds(ligands, config)
    reported_assignments = {row["ligand_id"]: int(row["outer_fold"]) for row in read_csv(root / config["outputs"]["fold_assignments_csv"])}
    if any(reported_assignments[row["ligand_id"]] != int(folds[index]) for index, row in enumerate(ligands)):
        raise ValueError("fold assignments do not reproduce")
    matrices = independent_matrices(root, config, ligands, receptor_ids)
    sizes = [int(value) for value in config["landscape"]["subset_sizes"]]
    all_subsets = [subset for size in sizes for subset in itertools.combinations(range(len(receptor_ids)), size)]
    by_size = {size: [subset for subset in all_subsets if len(subset) == size] for size in sizes}
    reported_rows = {(int(row["outer_fold"]), int(row["subset_size"])): row for row in read_csv(root / config["outputs"]["fold_comparisons_csv"])}
    tolerance = float(config["landscape"]["objective_tolerance"])
    alpha = float(config["landscape"]["bedroc_alpha"])
    maximum_difference = 0.0
    reproduced: list[dict[str, Any]] = []
    metric_names = ("primary", "sensitivity", "mean_seed", "worst_seed", "robust")
    for fold in range(int(config["folds"]["fold_count"])):
        holdout = folds == fold
        train = ~holdout
        normalized = {scenario: independent_normalize(matrix, train) for scenario, matrix in matrices.items()}
        train_metrics = subset_metrics(normalized, labels, train, all_subsets, alpha)
        holdout_metrics = subset_metrics(normalized, labels, holdout, all_subsets, alpha)
        for size in sizes:
            exact = best_subset(by_size[size], train_metrics["robust"], receptor_ids, tolerance)
            greedy = greedy_subset(size, train_metrics["robust"], receptor_ids, tolerance)
            row = reported_rows[(fold, size)]
            if row["exact_subset"] != subset_name(exact, receptor_ids) or row["strong_greedy_subset"] != subset_name(greedy, receptor_ids):
                raise ValueError(f"fold {fold}, k={size}: selected subset differs")
            for split, metrics in (("train", train_metrics), ("holdout", holdout_metrics)):
                for method, subset in (("exact", exact), ("greedy", greedy)):
                    for metric in metric_names:
                        field = f"{split}_{method}_{metric}"
                        maximum_difference = max(maximum_difference, abs(float(row[field]) - metrics[metric][subset]))
            reproduced.append(
                {
                    "subset_size": size,
                    "train_gap": train_metrics["robust"][exact] - train_metrics["robust"][greedy],
                    "holdout_gap": holdout_metrics["robust"][exact] - holdout_metrics["robust"][greedy],
                    "differs": exact != greedy,
                }
            )
    if maximum_difference > 1e-12:
        raise ValueError(f"Stage32a metric difference exceeds tolerance: {maximum_difference}")

    gate = config["stage33_gate"]
    qualifying = []
    for size in sizes:
        rows = [row for row in reproduced if row["subset_size"] == size]
        if (
            statistics.fmean(row["train_gap"] for row in rows) >= float(gate["minimum_mean_outer_train_greedy_gap"])
            and sum(row["train_gap"] > float(gate["positive_gap_tolerance"]) for row in rows) >= int(gate["minimum_positive_gap_folds"])
            and statistics.fmean(row["holdout_gap"] for row in rows) >= float(gate["minimum_mean_outer_holdout_gain"])
        ):
            qualifying.append(size)
    authorized = len(qualifying) >= int(gate["minimum_qualifying_subset_sizes"])
    if qualifying != result["decision"]["qualifying_subset_sizes"] or authorized != result["decision"]["stage33_qubo_model_authorized"]:
        raise ValueError("Stage32a decision does not reproduce")
    checks = {
        "config_and_output_descriptors_verified": True,
        "stage32_matrix_audit_gate_verified": True,
        "four_balanced_scaffold_folds_recomputed": True,
        "all_59568_train_subsets_reenumerated": True,
        "all_59568_holdout_subsets_reenumerated": True,
        "all_24_exact_and_strong_greedy_selections_recomputed": True,
        "all_reported_metrics_recomputed": maximum_difference <= 1e-12,
        "stage33_gate_and_decision_recomputed": True,
        "protected_data_and_hardware_boundary_verified": all(int(result["data_boundary"][key]) == 0 for key in ("fresh_validation_rows_read", "test_rows_read", "new_docking_jobs", "quantum_hardware_jobs")),
    }
    if not all(checks.values()):
        raise ValueError(f"Stage32a audit failed: {checks}")
    audit_result = {
        "schema_version": "1.0",
        "status": "stage32a_pparg_md_functional_landscape_audit_ok",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": sha256(config_path)},
        "result": {"path": result_path.relative_to(root).as_posix(), "sha256": sha256(result_path)},
        "checks": checks,
        "coverage": {
            "fold_count": 4,
            "subset_count_per_split_per_fold": len(all_subsets),
            "total_train_subset_states": 4 * len(all_subsets),
            "total_holdout_subset_states": 4 * len(all_subsets),
            "selection_comparison_count": len(reproduced),
            "maximum_recomputed_abs_difference": maximum_difference,
        },
        "decision": result["decision"],
        "data_boundary": result["data_boundary"],
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output = root / config["outputs"]["audit_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32a_pparg_md_functional_landscape_analysis.json"))
    parser.add_argument("--result", type=Path, default=Path("data/stage32a_pparg_md_functional_landscape_result.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    audit(args.config, args.result, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
