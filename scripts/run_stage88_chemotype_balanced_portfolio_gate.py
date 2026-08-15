"""Evaluate whether receptor selection must balance active-ligand chemotypes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.run_stage68_quality_plateau_portfolio_qubo as s68


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty Stage88 CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified(root: Path, descriptor: dict[str, Any]) -> Path:
    path = root / descriptor["path"]
    if not path.is_file() or sha256(path) != descriptor["sha256"]:
        raise ValueError(f"Stage88 frozen input identity differs: {path}")
    return path


def fingerprint_distance(rows: list[dict[str, str]], radius: int, bit_count: int) -> tuple[list[Any], np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bit_count)
    fingerprints = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        if molecule is None:
            raise ValueError(f"Stage88 invalid canonical SMILES: {row['ligand_id']}")
        fingerprints.append(generator.GetFingerprint(molecule))
    distance = np.zeros((len(rows), len(rows)), dtype=float)
    for index, fingerprint in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)
        distance[index, :] = 1.0 - np.asarray(similarities, dtype=float)
    np.fill_diagonal(distance, 0.0)
    return fingerprints, distance


def cluster_target(
    target_id: str,
    target: dict[str, Any],
    clustering: dict[str, Any],
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    active_rows = [row for row in target["ligands"] if row["label"] == "active"]
    _, distance = fingerprint_distance(
        active_rows, int(clustering["radius"]), int(clustering["bit_count"])
    )
    labels = AgglomerativeClustering(
        n_clusters=int(clustering["cluster_count"]),
        metric="precomputed",
        linkage="average",
    ).fit_predict(distance)
    raw_groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        raw_groups.setdefault(int(label), []).append(index)
    ordered = sorted(
        raw_groups,
        key=lambda label: min(active_rows[index]["ligand_id"] for index in raw_groups[label]),
    )
    relabel = {old: new for new, old in enumerate(ordered)}
    assignments = {
        row["ligand_id"]: relabel[int(label)]
        for row, label in zip(active_rows, labels)
    }
    rows = [
        {
            "target_id": target_id,
            "ligand_id": row["ligand_id"],
            "chemotype_id": assignments[row["ligand_id"]],
            "canonical_smiles": row["canonical_smiles"],
            "scaffold_smiles": row["scaffold_smiles"],
        }
        for row in active_rows
    ]
    counts = {str(group): sum(value == group for value in assignments.values()) for group in range(len(ordered))}
    return assignments, rows, {
        "target_id": target_id,
        "active_count": len(active_rows),
        "cluster_counts": counts,
        "minimum_cluster_count": min(counts.values()),
        "silhouette": float(silhouette_score(distance, labels, metric="precomputed")),
    }


def singleton_utilities(
    ranks: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    receptor_count: int,
    alpha: float,
) -> np.ndarray:
    return np.asarray(
        [
            s68.bedroc_metrics(ranks[:, mask, :], labels[mask], (index,), alpha)[
                "robust_bedroc_composite"
            ]
            for index in range(receptor_count)
        ],
        dtype=float,
    )


def balanced_key(subset: tuple[int, ...], utilities: np.ndarray) -> tuple[Any, ...]:
    coverage = np.max(utilities[:, subset], axis=1)
    return (-float(np.min(coverage)), -float(np.mean(coverage)), subset)


def exact_balanced(utilities: np.ndarray, subset_size: int) -> tuple[int, ...]:
    return min(
        itertools.combinations(range(utilities.shape[1]), subset_size),
        key=lambda subset: balanced_key(subset, utilities),
    )


def greedy_balanced(utilities: np.ndarray, subset_size: int) -> tuple[int, ...]:
    selected: tuple[int, ...] = ()
    while len(selected) < subset_size:
        selected = min(
            (
                tuple(sorted((*selected, candidate)))
                for candidate in range(utilities.shape[1])
                if candidate not in selected
            ),
            key=lambda subset: balanced_key(subset, utilities),
        )
    return selected


def overall_topk(utilities: np.ndarray, subset_size: int) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in np.argsort(-utilities, kind="stable")[:subset_size]))


def evaluate_holdout(
    ranks: np.ndarray,
    labels: np.ndarray,
    holdout_mask: np.ndarray,
    ligand_ids: list[str],
    assignments: dict[str, int],
    subset: tuple[int, ...],
    alpha: float,
    group_count: int,
) -> dict[str, float | None]:
    group_values: list[float] = []
    all_groups_evaluable = True
    for group in range(group_count):
        group_mask = np.asarray(
            [
                bool(holdout_mask[index])
                and (
                    labels[index] == 0
                    or assignments.get(ligand_id) == group
                )
                for index, ligand_id in enumerate(ligand_ids)
            ]
        )
        group_labels = labels[group_mask]
        if int(np.sum(group_labels == 1)) == 0 or int(np.sum(group_labels == 0)) == 0:
            all_groups_evaluable = False
            continue
        value = s68.bedroc_metrics(
            ranks[:, group_mask, :], group_labels, subset, alpha
        )["robust_bedroc_composite"]
        group_values.append(float(value))
    overall = s68.bedroc_metrics(
        ranks[:, holdout_mask, :], labels[holdout_mask], subset, alpha
    )["robust_bedroc_composite"]
    return {
        "all_groups_evaluable": all_groups_evaluable,
        "worst_group_holdout_robust_bedroc": (
            min(group_values) if all_groups_evaluable else None
        ),
        "mean_group_holdout_robust_bedroc": (
            statistics.fmean(group_values) if all_groups_evaluable else None
        ),
        "overall_holdout_robust_bedroc": float(overall),
    }


def subset_name(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(receptor_ids[index] for index in subset)


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    stage87 = read_json(inputs["stage87_result"])
    if stage87["constraint_preserving_qaoa_simulation_authorized"]:
        raise ValueError("Stage88 expected Stage87 to block the current QAOA route")
    stage64 = read_json(inputs["stage64_config"])
    clustering = config["chemotype_clustering"]
    portfolio = config["portfolio"]
    alpha = float(portfolio["bedroc_alpha"])
    subset_size = int(portfolio["subset_size"])
    group_count = int(clustering["cluster_count"])

    assignment_rows: list[dict[str, Any]] = []
    cluster_summaries = []
    fold_rows: list[dict[str, Any]] = []
    for target_id in stage64["targets"]:
        target = s68.load_target(root, target_id, stage64["targets"][target_id])
        assignments, rows, cluster_summary = cluster_target(
            target_id, target, clustering
        )
        assignment_rows.extend(rows)
        cluster_summaries.append(cluster_summary)
        labels = target["labels"]
        ligand_ids = target["ligand_ids"]
        receptor_ids = target["receptor_ids"]
        active_groups = np.asarray(
            [assignments.get(ligand_id, -1) for ligand_id in ligand_ids], dtype=int
        )
        for outer_fold in range(4):
            train_mask = np.asarray(
                [target["outer"][ligand_id] != outer_fold for ligand_id in ligand_ids]
            )
            holdout_mask = ~train_mask
            ranks = s68.rank_cube(target["scores"], train_mask)
            group_utilities = []
            train_counts = []
            holdout_counts = []
            for group in range(group_count):
                train_count = int(
                    np.sum(train_mask & (labels == 1) & (active_groups == group))
                )
                holdout_count = int(
                    np.sum(holdout_mask & (labels == 1) & (active_groups == group))
                )
                train_counts.append(train_count)
                holdout_counts.append(holdout_count)
                mask = train_mask & ((labels == 0) | (active_groups == group))
                group_utilities.append(
                    (
                        singleton_utilities(
                            ranks, labels, mask, len(receptor_ids), alpha
                        )
                        if train_count > 0
                        else None
                    )
                )
            if any(value is None for value in group_utilities) or min(holdout_counts) == 0:
                fold_rows.append(
                    {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "receptor_count": len(receptor_ids),
                        "subset_size": subset_size,
                        "chemotype_fold_evaluable": False,
                        "minimum_train_group_active_count": min(train_counts),
                        "minimum_holdout_group_active_count": min(holdout_counts),
                        "distinct_group_best_receptor_count": 0,
                        "median_group_utility_spearman": 1.0,
                        "exact_subset": "",
                        "greedy_subset": "",
                        "overall_top3_subset": "",
                        "exact_differs_from_greedy": False,
                        "exact_differs_from_overall_top3": False,
                        "exact_worst_train_group_utility": None,
                        "greedy_worst_train_group_utility": None,
                        "exact_train_objective_gap_over_greedy": 0.0,
                        "exact_all_groups_evaluable": False,
                        "greedy_all_groups_evaluable": False,
                        "overall_top3_all_groups_evaluable": False,
                        "exact_worst_group_gain_vs_overall_top3": None,
                    }
                )
                continue
            utility_matrix = np.stack(group_utilities)
            overall_utility = singleton_utilities(
                ranks, labels, train_mask, len(receptor_ids), alpha
            )
            exact = exact_balanced(utility_matrix, subset_size)
            greedy = greedy_balanced(utility_matrix, subset_size)
            baseline = overall_topk(overall_utility, subset_size)
            exact_key = balanced_key(exact, utility_matrix)
            greedy_key = balanced_key(greedy, utility_matrix)
            correlations = [
                float(spearmanr(utility_matrix[left], utility_matrix[right]).statistic)
                for left, right in itertools.combinations(range(group_count), 2)
            ]
            methods = {
                "chemotype_balanced_exact": exact,
                "chemotype_balanced_greedy": greedy,
                "overall_singleton_top3": baseline,
            }
            evaluated = {
                method: evaluate_holdout(
                    ranks,
                    labels,
                    holdout_mask,
                    ligand_ids,
                    assignments,
                    subset,
                    alpha,
                    group_count,
                )
                for method, subset in methods.items()
            }
            fold_rows.append(
                {
                    "target_id": target_id,
                    "outer_fold": outer_fold,
                    "receptor_count": len(receptor_ids),
                    "subset_size": subset_size,
                    "chemotype_fold_evaluable": True,
                    "minimum_train_group_active_count": min(train_counts),
                    "minimum_holdout_group_active_count": min(holdout_counts),
                    "distinct_group_best_receptor_count": len(
                        set(int(value) for value in np.argmax(utility_matrix, axis=1))
                    ),
                    "median_group_utility_spearman": statistics.median(correlations),
                    "exact_subset": subset_name(exact, receptor_ids),
                    "greedy_subset": subset_name(greedy, receptor_ids),
                    "overall_top3_subset": subset_name(baseline, receptor_ids),
                    "exact_differs_from_greedy": exact != greedy,
                    "exact_differs_from_overall_top3": exact != baseline,
                    "exact_worst_train_group_utility": -float(exact_key[0]),
                    "greedy_worst_train_group_utility": -float(greedy_key[0]),
                    "exact_train_objective_gap_over_greedy": float(greedy_key[0] - exact_key[0]),
                    **{
                        f"exact_{key}": value
                        for key, value in evaluated["chemotype_balanced_exact"].items()
                    },
                    **{
                        f"greedy_{key}": value
                        for key, value in evaluated["chemotype_balanced_greedy"].items()
                    },
                    **{
                        f"overall_top3_{key}": value
                        for key, value in evaluated["overall_singleton_top3"].items()
                    },
                    "exact_worst_group_gain_vs_overall_top3": (
                        float(
                            evaluated["chemotype_balanced_exact"][
                                "worst_group_holdout_robust_bedroc"
                            ]
                        )
                        - float(
                            evaluated["overall_singleton_top3"][
                                "worst_group_holdout_robust_bedroc"
                            ]
                        )
                        if evaluated["chemotype_balanced_exact"][
                            "all_groups_evaluable"
                        ]
                        and evaluated["overall_singleton_top3"][
                            "all_groups_evaluable"
                        ]
                        else None
                    ),
                }
            )

    gate = config["gate"]
    gains = [
        float(row["exact_worst_group_gain_vs_overall_top3"])
        for row in fold_rows
        if row["exact_worst_group_gain_vs_overall_top3"] is not None
    ]
    checks = {
        "global_cluster_sizes": min(
            int(summary["minimum_cluster_count"]) for summary in cluster_summaries
        )
        >= int(gate["minimum_active_count_per_global_cluster"]),
        "fold_train_cluster_sizes": min(
            int(row["minimum_train_group_active_count"]) for row in fold_rows
        )
        >= int(gate["minimum_train_active_count_per_fold_cluster"]),
        "fold_holdout_cluster_sizes": min(
            int(row["minimum_holdout_group_active_count"]) for row in fold_rows
        )
        >= int(gate["minimum_holdout_active_count_per_fold_cluster"]),
        "group_specialization": statistics.median(
            float(row["median_group_utility_spearman"]) for row in fold_rows
        )
        <= float(gate["maximum_median_train_group_utility_spearman"]),
        "distinct_group_best_receptors": sum(
            int(row["distinct_group_best_receptor_count"]) >= 2 for row in fold_rows
        )
        >= int(gate["minimum_fold_count_with_two_distinct_group_best_receptors"]),
        "balanced_selection_differs": sum(
            bool(row["exact_differs_from_overall_top3"]) for row in fold_rows
        )
        >= int(gate["minimum_exact_selection_difference_folds_vs_overall_top3"]),
        "positive_worst_group_holdout_gain_folds": sum(value > 1e-12 for value in gains)
        >= int(gate["minimum_positive_worst_group_holdout_gain_folds"]),
        "mean_worst_group_holdout_gain": bool(gains)
        and statistics.fmean(gains)
        >= float(gate["minimum_mean_worst_group_holdout_gain_vs_overall_top3"]),
        "balanced_greedy_trap_exists": sum(
            float(row["exact_train_objective_gap_over_greedy"]) > 1e-12
            for row in fold_rows
        )
        >= int(gate["minimum_exact_objective_gap_folds_over_balanced_greedy"]),
    }
    passed = all(checks.values())
    outputs = config["outputs"]
    write_csv(root / outputs["cluster_assignments_csv"], assignment_rows)
    write_csv(root / outputs["fold_metrics_csv"], fold_rows)
    result = {
        "schema_version": "1.0",
        "status": (
            "stage88_chemotype_balanced_portfolio_gate_passed"
            if passed
            else "stage88_chemotype_balanced_portfolio_gate_failed"
        ),
        "cluster_summaries": cluster_summaries,
        "summary": {
            "target_count": len(cluster_summaries),
            "fold_count": len(fold_rows),
            "minimum_global_cluster_count": min(
                int(value["minimum_cluster_count"]) for value in cluster_summaries
            ),
            "minimum_train_group_active_count": min(
                int(row["minimum_train_group_active_count"]) for row in fold_rows
            ),
            "minimum_holdout_group_active_count": min(
                int(row["minimum_holdout_group_active_count"]) for row in fold_rows
            ),
            "median_group_utility_spearman": statistics.median(
                float(row["median_group_utility_spearman"]) for row in fold_rows
            ),
            "folds_with_distinct_group_best_receptors": sum(
                int(row["distinct_group_best_receptor_count"]) >= 2 for row in fold_rows
            ),
            "exact_selection_difference_fold_count": sum(
                bool(row["exact_differs_from_overall_top3"]) for row in fold_rows
            ),
            "balanced_greedy_trap_fold_count": sum(
                float(row["exact_train_objective_gap_over_greedy"]) > 1e-12
                for row in fold_rows
            ),
            "positive_worst_group_holdout_gain_fold_count": sum(
                value > 1e-12 for value in gains
            ),
            "evaluable_worst_group_holdout_fold_count": len(gains),
            "mean_worst_group_holdout_gain_vs_overall_top3": (
                statistics.fmean(gains) if gains else 0.0
            ),
            "per_target_mean_worst_group_gain": {
                target_id: (
                    statistics.fmean(
                        float(row["exact_worst_group_gain_vs_overall_top3"])
                        for row in fold_rows
                        if row["target_id"] == target_id
                        and row["exact_worst_group_gain_vs_overall_top3"] is not None
                    )
                    if any(
                        row["target_id"] == target_id
                        and row["exact_worst_group_gain_vs_overall_top3"] is not None
                        for row in fold_rows
                    )
                    else None
                )
                for target_id in stage64["targets"]
            },
        },
        "checks": checks,
        "chemotype_balanced_cqm_design_authorized": passed,
        "constraint_preserving_qaoa_simulation_authorized": False,
        "new_quantum_hardware_jobs_authorized": 0,
        "new_docking_jobs_authorized": 0,
        "next_action": (
            "Freeze and formulate the chemotype-balanced constraint-native CQM, then compare exact MILP, multistart greedy, tabu, and annealing on historical data."
            if passed
            else "Do not formulate a new chemotype-balanced CQM; the structural grouping or held-out benefit gate did not pass."
        ),
        "data_boundary": {
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
    }
    write_json(root / outputs["result_json"], result)
    report = [
        "# Stage88 chemotype-balanced portfolio gate",
        "",
        "Active ligands were grouped from chemical structure only using four deterministic Morgan-fingerprint clusters per target. Existing docking outcomes were then evaluated under the frozen four-fold splits.",
        "",
        f"- Minimum global cluster size: `{result['summary']['minimum_global_cluster_count']}`.",
        f"- Median train group-utility Spearman: `{result['summary']['median_group_utility_spearman']:.6f}`.",
        f"- Folds with distinct group-best receptors: `{result['summary']['folds_with_distinct_group_best_receptors']}` / `{len(fold_rows)}`.",
        f"- Exact selections differing from overall Top-3: `{result['summary']['exact_selection_difference_fold_count']}` / `{len(fold_rows)}`.",
        f"- Balanced-greedy trap folds: `{result['summary']['balanced_greedy_trap_fold_count']}` / `{len(fold_rows)}`.",
        f"- Positive worst-group holdout gains: `{result['summary']['positive_worst_group_holdout_gain_fold_count']}` / `{len(fold_rows)}`.",
        f"- Mean worst-group holdout gain: `{result['summary']['mean_worst_group_holdout_gain_vs_overall_top3']:+.6f}`.",
        "",
        "## Decision",
        "",
        (
            "The scientific intake gate passed. A chemotype-balanced constraint-native CQM may be designed next, but QAOA and hardware remain blocked."
            if passed
            else "The scientific intake gate failed. No new chemotype-balanced CQM, QAOA, docking, or hardware task is authorized."
        ),
        "",
    ]
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="ascii")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage88_chemotype_balanced_portfolio_gate.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run((root / args.config).resolve(), root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
