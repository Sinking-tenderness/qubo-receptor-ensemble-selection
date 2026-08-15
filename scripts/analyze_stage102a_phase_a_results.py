"""Audit Stage102A matrices and run frozen nested comparator analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage99_qubo_objective_repair_screen as s99
from scripts import run_stage100_adaptive_stopping_qubo as s100


TARGETS = {
    "EGFR": "stage102a_egfr_phase_a_production",
    "FA10": "stage102a_fa10_phase_a_production",
}
METHODS = (
    "single",
    "fixed_k2",
    "fixed_k3",
    "maximum_inner_mean",
    "one_standard_error_smallest_k",
    "sequential_paired_marginal_lcb",
    "outer_oracle_k",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_target(received_root: Path, target: str) -> tuple[list[str], list[dict[str, str]], np.ndarray]:
    lower = target.lower()
    run = TARGETS[target]
    matrix_path = received_root / "results" / "runs" / run / "primary_median_score_matrix.csv"
    manifest_path = received_root / "data" / "processed" / f"stage102a_{lower}_phase_a_pdbqt_manifest.csv"
    receptors, matrix_rows, values = s99.read_matrix(matrix_path)
    manifest_rows = read_csv(manifest_path)
    manifest_by_id = {row["ligand_id"]: row for row in manifest_rows}
    if len(matrix_rows) != 600 or len(manifest_rows) != 600:
        raise ValueError(f"{target}: expected 600 matrix and manifest rows")
    if values.shape != (600, len(receptors)) or not np.isfinite(values).all():
        raise ValueError(f"{target}: invalid score matrix shape or non-finite score")
    merged = []
    for row in matrix_rows:
        manifest = manifest_by_id.get(row["ligand_id"])
        if manifest is None:
            raise ValueError(f"{target}: matrix ligand absent from manifest: {row['ligand_id']}")
        if row["label"] != manifest["label"]:
            raise ValueError(f"{target}: label mismatch for {row['ligand_id']}")
        merged.append(manifest)
    return receptors, merged, values


def load_seed_values(
    received_root: Path,
    target: str,
    receptors: list[str],
    rows: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    run = TARGETS[target]
    score_path = received_root / "results" / "runs" / run / "scores.csv"
    score_rows = read_csv(score_path)
    ligand_index = {row["ligand_id"]: index for index, row in enumerate(rows)}
    receptor_index = {receptor: index for index, receptor in enumerate(receptors)}
    seed_ids = sorted({row["seed_id"] for row in score_rows})
    matrices = {seed: np.full((len(rows), len(receptors)), np.nan) for seed in seed_ids}
    for row in score_rows:
        matrices[row["seed_id"]][
            ligand_index[row["ligand_id"]], receptor_index[row["receptor_id"]]
        ] = float(row["gpu_score"])
    for seed, matrix in matrices.items():
        if not np.isfinite(matrix).all():
            raise ValueError(f"{target}/{seed}: incomplete seed matrix")
    return matrices


def evaluate_target(
    target: str,
    receptors: list[str],
    rows: list[dict[str, str]],
    values: np.ndarray,
    parent: dict[str, Any],
    adaptive: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = s99.label_array(rows)
    groups = np.asarray([row["split_group_id"] for row in rows])
    outer_labels = np.asarray([int(row["outer_fold"]) for row in rows])
    fold_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for fold in sorted(set(outer_labels.tolist())):
        train_index = np.flatnonzero(outer_labels != fold)
        test_index = np.flatnonzero(outer_labels == fold)
        train_values = values[train_index]
        train_labels = labels[train_index]
        profiles, _ = s100.inner_profiles(
            train_values,
            train_labels,
            groups[train_index],
            adaptive,
            parent,
        )
        one_se_k, _ = s100.choose_one_standard_error(profiles)
        maximum_k = s100.choose_maximum_mean(profiles)
        marginal_k, _ = s100.choose_marginal_lcb(profiles)

        train_ranks = s99.percentile_ranks(train_values, train_values)
        test_ranks = s99.percentile_ranks(train_values, values[test_index])
        node, pair = s99.coefficients(train_ranks, train_labels, parent, "repair")
        subsets = {k: s99.exact_select(len(receptors), k, node, pair) for k in (1, 2, 3)}
        scores = {
            k: s99.bedroc(
                s99.aggregate(test_ranks[:, list(subsets[k])], "minimum"),
                labels[test_index],
                float(parent["objective"]["bedroc_alpha"]),
            )
            for k in (1, 2, 3)
        }
        choices = {
            "single": 1,
            "fixed_k2": 2,
            "fixed_k3": 3,
            "maximum_inner_mean": maximum_k,
            "one_standard_error_smallest_k": one_se_k,
            "sequential_paired_marginal_lcb": marginal_k,
            "outer_oracle_k": max((1, 2, 3), key=lambda k: (scores[k], -k)),
        }
        for method, selected_k in choices.items():
            fold_rows.append(
                {
                    "target_id": target,
                    "outer_fold": fold,
                    "method": method,
                    "selected_k": selected_k,
                    "selected_receptors": s99.subset_name(subsets[selected_k], receptors),
                    "outer_bedroc_alpha20": scores[selected_k],
                    "gain_over_single": scores[selected_k] - scores[1],
                    "uses_outer_labels_for_selection": method == "outer_oracle_k",
                }
            )
        for current in (2, 3):
            differences = np.asarray(profiles[current]) - np.asarray(profiles[current - 1])
            outer_gain = scores[current] - scores[current - 1]
            edge_rows.append(
                {
                    "target_id": target,
                    "outer_fold": fold,
                    "from_k": current - 1,
                    "to_k": current,
                    "inner_mean_gain": float(np.mean(differences)),
                    "inner_gain_se": s100.standard_error(differences.tolist()),
                    "outer_gain": outer_gain,
                    "sign_correct": int(np.sign(np.mean(differences)) == np.sign(outer_gain)),
                }
            )
    return fold_rows, edge_rows


def summarize(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for target in TARGETS:
        for method in METHODS:
            selected = [row for row in fold_rows if row["target_id"] == target and row["method"] == method]
            values = np.asarray([row["outer_bedroc_alpha20"] for row in selected], dtype=float)
            gains = np.asarray([row["gain_over_single"] for row in selected], dtype=float)
            summaries.append(
                {
                    "target_id": target,
                    "method": method,
                    "mean_outer_bedroc_alpha20": float(np.mean(values)),
                    "std_outer_bedroc_alpha20": float(np.std(values, ddof=1)),
                    "gain_over_single": float(np.mean(gains)),
                    "mean_selected_k": float(np.mean([row["selected_k"] for row in selected])),
                    "selected_k_values": "|".join(str(row["selected_k"]) for row in selected),
                }
            )
    return summaries


def summarize_seed_robustness(
    received_root: Path,
    target_data: dict[str, tuple[list[str], list[dict[str, str]]]],
    parent: dict[str, Any],
    adaptive: dict[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target, (receptors, rows) in target_data.items():
        for seed, values in load_seed_values(received_root, target, receptors, rows).items():
            folds, _ = evaluate_target(target, receptors, rows, values, parent, adaptive)
            for method in ("single", "fixed_k2", "fixed_k3", "one_standard_error_smallest_k"):
                selected = [row for row in folds if row["method"] == method]
                scores = np.asarray([row["outer_bedroc_alpha20"] for row in selected], dtype=float)
                gains = np.asarray([row["gain_over_single"] for row in selected], dtype=float)
                result.append(
                    {
                        "target_id": target,
                        "seed_id": seed,
                        "method": method,
                        "mean_outer_bedroc_alpha20": float(np.mean(scores)),
                        "gain_over_single": float(np.mean(gains)),
                        "positive_fold_count": int(np.sum(gains > 0.0)),
                    }
                )
    return result


def summarize_frozen_selection_seed_robustness(
    received_root: Path,
    target_data: dict[str, tuple[list[str], list[dict[str, str]]]],
    median_fold_rows: list[dict[str, Any]],
    parent: dict[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target, (receptors, rows) in target_data.items():
        labels = s99.label_array(rows)
        outer_labels = np.asarray([int(row["outer_fold"]) for row in rows])
        receptor_index = {receptor: index for index, receptor in enumerate(receptors)}
        target_median = [row for row in median_fold_rows if row["target_id"] == target]
        for seed, values in load_seed_values(received_root, target, receptors, rows).items():
            records: dict[str, list[float]] = {
                method: []
                for method in ("single", "fixed_k2", "fixed_k3", "one_standard_error_smallest_k")
            }
            for fold in sorted(set(outer_labels.tolist())):
                train_index = np.flatnonzero(outer_labels != fold)
                test_index = np.flatnonzero(outer_labels == fold)
                test_ranks = s99.percentile_ranks(values[train_index], values[test_index])
                for method in records:
                    median_row = next(
                        row
                        for row in target_median
                        if row["outer_fold"] == fold and row["method"] == method
                    )
                    subset = tuple(
                        receptor_index[name]
                        for name in median_row["selected_receptors"].split("+")
                    )
                    records[method].append(
                        s99.bedroc(
                            s99.aggregate(test_ranks[:, list(subset)], "minimum"),
                            labels[test_index],
                            float(parent["objective"]["bedroc_alpha"]),
                        )
                    )
            single = np.asarray(records["single"], dtype=float)
            for method, method_scores in records.items():
                scores = np.asarray(method_scores, dtype=float)
                gains = scores - single
                result.append(
                    {
                        "target_id": target,
                        "seed_id": seed,
                        "method": method,
                        "mean_outer_bedroc_alpha20": float(np.mean(scores)),
                        "gain_over_median_selected_single": float(np.mean(gains)),
                        "positive_fold_count": int(np.sum(gains > 0.0)),
                    }
                )
    return result


def provisional_gate(root: Path, summaries: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    previous = json.loads((root / "data/stage100_adaptive_stopping_qubo_result.json").read_text(encoding="utf-8"))
    old_gains = [float(row["gain"]) for row in previous["primary_target_rows"]]
    new_rows = [row for row in summaries if row["method"] == "one_standard_error_smallest_k"]
    new_gains = [float(row["gain_over_single"]) for row in new_rows]
    gains = np.asarray(old_gains + new_gains, dtype=float)
    new_sign_accuracy = float(np.mean([row["sign_correct"] for row in edges]))
    nontrivial = int(previous["gate"]["nontrivial_selected_fold_count"]) + sum(
        int(value) > 1
        for row in new_rows
        for value in row["selected_k_values"].split("|")
    )
    metrics = {
        "mean_target_gain": float(np.mean(gains)),
        "worst_target_gain": float(np.min(gains)),
        "target_count_gain_at_least_0p02": int(np.sum(gains >= 0.02)),
        "nontrivial_outer_fold_count": nontrivial,
        "new_target_count_with_positive_gain": int(np.sum(np.asarray(new_gains) > 0.0)),
        "new_target_marginal_sign_accuracy": new_sign_accuracy,
    }
    checks = {
        "mean_target_gain": metrics["mean_target_gain"] >= 0.02,
        "worst_target_gain": metrics["worst_target_gain"] >= -0.02,
        "positive_target_count": metrics["target_count_gain_at_least_0p02"] >= 4,
        "nontrivial_outer_fold_count": metrics["nontrivial_outer_fold_count"] >= 12,
        "new_positive_target_count": metrics["new_target_count_with_positive_gain"] >= 1,
        "new_target_sign_accuracy": metrics["new_target_marginal_sign_accuracy"] >= 0.60,
    }
    return {
        "status": "diagnostic_only_not_the_frozen_stage102_candidate_gate",
        "reason": "The preregistration names mechanistic-bootstrap and held-target Ridge candidates but does not fully specify executable bootstrap and feature transformations. This calculation applies the frozen one-standard-error comparator and cannot release PARP1.",
        "metrics": metrics,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--received-root",
        type=Path,
        default=Path("analysis/stage102a_received_20260813/core"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage102a_received_20260813/analysis"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    received_root = (root / args.received_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    parent = json.loads((root / "configs/stage99_qubo_objective_repair_screen.json").read_text(encoding="utf-8"))
    adaptive = json.loads((root / "configs/stage100_adaptive_stopping_qubo.json").read_text(encoding="utf-8"))

    fold_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    matrix_audit: dict[str, Any] = {}
    target_data: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for target in TARGETS:
        receptors, rows, values = load_target(received_root, target)
        target_data[target] = (receptors, rows)
        target_fold_rows, target_edges = evaluate_target(target, receptors, rows, values, parent, adaptive)
        fold_rows.extend(target_fold_rows)
        edge_rows.extend(target_edges)
        labels = s99.label_array(rows)
        matrix_audit[target] = {
            "ligand_count": len(rows),
            "active_count": int(np.sum(labels == 1)),
            "decoy_count": int(np.sum(labels == 0)),
            "receptor_count": len(receptors),
            "score_count": int(values.size),
            "nonfinite_score_count": int(np.sum(~np.isfinite(values))),
            "outer_fold_counts": {
                str(fold): int(np.sum(np.asarray([int(row["outer_fold"]) for row in rows]) == fold))
                for fold in range(1, 6)
            },
        }
    summaries = summarize(fold_rows)
    seed_summaries = summarize_seed_robustness(received_root, target_data, parent, adaptive)
    frozen_seed_summaries = summarize_frozen_selection_seed_robustness(
        received_root, target_data, fold_rows, parent
    )
    gate = provisional_gate(root, summaries, edge_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fold_metrics.csv", fold_rows)
    write_csv(output_dir / "marginal_edges.csv", edge_rows)
    write_csv(output_dir / "target_summary.csv", summaries)
    write_csv(output_dir / "seed_target_summary.csv", seed_summaries)
    write_csv(output_dir / "frozen_selection_seed_summary.csv", frozen_seed_summaries)
    result = {
        "schema_version": "1.0",
        "status": "stage102a_received_results_analyzed",
        "matrix_audit": matrix_audit,
        "target_summary": summaries,
        "seed_target_summary": seed_summaries,
        "frozen_selection_seed_summary": frozen_seed_summaries,
        "new_target_marginal_sign_accuracy": float(np.mean([row["sign_correct"] for row in edge_rows])),
        "provisional_one_se_gate": gate,
        "interpretation": "Stage102A docking is technically complete. Nested metrics are development diagnostics; PARP1 and hardware remain locked.",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
