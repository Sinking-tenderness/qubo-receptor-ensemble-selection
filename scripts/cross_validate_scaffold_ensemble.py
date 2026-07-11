"""Run scaffold-group outer CV for receptor subset selection."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

try:
    from .cross_validate_ensemble_mvp import (
        collect_scores,
        normalize_rows_by_train_minmax,
        paired_bootstrap_delta,
    )
    from .run_receptor_ensemble_mvp import (
        choose_train_best,
        select_validation_tuned_qubo,
        validate_inputs,
    )
    from .select_receptor_baselines import metrics_for_subset, read_csv
    from .split_ligand_benchmark import read_rows
    from .split_ligand_scaffold import scaffold_for_smiles
    from .compare_receptor_screening import ranked_metrics_with_ids
except ImportError:
    from cross_validate_ensemble_mvp import collect_scores, normalize_rows_by_train_minmax, paired_bootstrap_delta
    from run_receptor_ensemble_mvp import choose_train_best, select_validation_tuned_qubo, validate_inputs
    from select_receptor_baselines import metrics_for_subset, read_csv
    from split_ligand_benchmark import read_rows
    from split_ligand_scaffold import scaffold_for_smiles
    from compare_receptor_screening import ranked_metrics_with_ids


def make_scaffold_folds(rows: list[dict[str, str]], fold_count: int, seed: int) -> dict[str, int]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[scaffold_for_smiles(row["canonical_smiles"])].append(row)
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    total = {label: sum(row["label"] == label for row in rows) for label in ("active", "decoy")}
    targets = {label: total[label] / fold_count for label in total}
    counts = [{label: 0 for label in total} for _ in range(fold_count)]
    assignment: dict[str, int] = {}
    for scaffold, group in items:
        group_counts = {label: sum(row["label"] == label for row in group) for label in total}

        def cost(fold: int) -> tuple[float, float, int]:
            projected = {label: counts[fold][label] + group_counts[label] for label in total}
            overflow = sum(max(0, projected[label] - targets[label]) / max(1, targets[label]) for label in total)
            distance = sum(abs(projected[label] - targets[label]) / max(1, targets[label]) for label in total)
            return overflow, distance, fold

        fold = min(range(fold_count), key=cost)
        for row in group:
            assignment[row["ligand_id"]] = fold
        for label in total:
            counts[fold][label] += group_counts[label]
    if len(assignment) != len(rows):
        raise ValueError("scaffold fold assignment is incomplete")
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--ligands", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--target-size", type=int, default=2)
    parser.add_argument("--utility-metric", choices=["roc_auc", "bedroc", "ef5"], default="roc_auc")
    parser.add_argument("--validation-metric", choices=["roc_auc", "pr_auc_average_precision", "bedroc_alpha_20"], default="roc_auc")
    parser.add_argument("--aggregation", choices=["min_score", "mean_score"], default="mean_score")
    parser.add_argument("--score-normalization", choices=["raw", "train_minmax"], default="raw")
    args = parser.parse_args()
    if args.folds < 3:
        raise ValueError("at least three folds are required")
    matrix_rows = read_csv(args.matrix)
    ligand_rows = read_rows(args.ligands)
    by_id = {row["ligand_id"]: row for row in ligand_rows}
    if {row["ligand_id"] for row in matrix_rows} != set(by_id):
        raise ValueError("matrix and ligand manifest IDs do not match")
    folds = make_scaffold_folds(ligand_rows, args.folds, args.seed)
    scaffold_by_id = {row["ligand_id"]: scaffold_for_smiles(row["canonical_smiles"]) for row in ligand_rows}
    fold_results: list[dict[str, object]] = []
    oof_scores = {"single": {}, "qubo": {}, "all_mean": {}}
    for outer_fold in range(args.folds):
        validation_fold = (outer_fold + 1) % args.folds
        raw_train_rows = [row for row in matrix_rows if folds[row["ligand_id"]] not in {outer_fold, validation_fold}]
        raw_validation_rows = [row for row in matrix_rows if folds[row["ligand_id"]] == validation_fold]
        raw_test_rows = [row for row in matrix_rows if folds[row["ligand_id"]] == outer_fold]
        if args.score_normalization == "train_minmax":
            train_rows, normalized = normalize_rows_by_train_minmax(
                raw_train_rows, [raw_validation_rows, raw_test_rows], args.receptor
            )
            validation_rows, test_rows = normalized
        else:
            train_rows, validation_rows, test_rows = raw_train_rows, raw_validation_rows, raw_test_rows
        train_ids = {row["ligand_id"] for row in raw_train_rows}
        validation_ids = {row["ligand_id"] for row in raw_validation_rows}
        split_manifest = [
            {
                "ligand_id": row["ligand_id"],
                "split": "train" if row["ligand_id"] in train_ids else "validation" if row["ligand_id"] in validation_ids else "test",
            }
            for row in matrix_rows
        ]
        validate_inputs(matrix_rows, split_manifest, args.receptor)
        tuned = select_validation_tuned_qubo(
            train_rows, validation_rows, args.receptor, args.target_size,
            [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.25, 0.5],
            args.utility_metric, args.validation_metric, args.aggregation,
        )
        qubo_subset = tuple(tuned["chosen"]["subset"])
        single_subset = choose_train_best(train_rows, args.receptor, 1, "min_score")
        fold_results.append({
            "outer_fold": outer_fold,
            "validation_fold": validation_fold,
            "train_active_count": sum(row["label"] == "active" for row in train_rows),
            "validation_active_count": sum(row["label"] == "active" for row in validation_rows),
            "test_active_count": sum(row["label"] == "active" for row in test_rows),
            "single_subset": list(single_subset),
            "qubo_subset": list(qubo_subset),
            "qubo_weights": {key: tuned["chosen"][key] for key in ("coverage_weight", "overlap_weight", "redundancy_weight")},
            "test_single_metrics": metrics_for_subset(test_rows, single_subset, args.aggregation),
            "test_qubo_metrics": metrics_for_subset(test_rows, qubo_subset, args.aggregation),
            "test_scaffold_count": len({scaffold_by_id[row["ligand_id"]] for row in test_rows}),
        })
        for name, subset in (("single", single_subset), ("qubo", qubo_subset)):
            oof_scores[name].update(collect_scores(test_rows, subset, args.aggregation))
        oof_scores["all_mean"].update(collect_scores(test_rows, tuple(args.receptor), args.aggregation))
    result = {
        "matrix": str(args.matrix), "ligands": str(args.ligands), "receptor_ids": args.receptor,
        "folds": args.folds, "seed": args.seed, "split_type": "Bemis-Murcko scaffold groups",
        "utility_metric": args.utility_metric, "validation_metric": args.validation_metric,
        "aggregation": args.aggregation,
        "score_normalization": args.score_normalization,
        "aggregate_out_of_fold_metrics": {name: ranked_metrics_with_ids(values) for name, values in oof_scores.items()},
        "oof_scores": oof_scores,
        "fold_results": fold_results,
        "paired_bootstrap_qubo_minus_single": paired_bootstrap_delta(oof_scores["single"], oof_scores["qubo"], 2000, args.seed + 1),
        "scaffold_disjoint": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result["aggregate_out_of_fold_metrics"], indent=2, ensure_ascii=True))
    print(json.dumps(result["paired_bootstrap_qubo_minus_single"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
