"""Evaluate a fixed-weight discriminative coverage QUBO with outer CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .cross_validate_ensemble_mvp import (
        collect_scores,
        make_folds,
        paired_bootstrap_delta,
    )
    from .run_receptor_ensemble_mvp import choose_train_best, validate_inputs
    from .select_receptor_baselines import metrics_for_subset, read_csv
    from .solve_discriminative_coverage_qubo import select_discriminative_subset
    from .split_ligand_benchmark import read_rows
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from cross_validate_ensemble_mvp import collect_scores, make_folds, paired_bootstrap_delta
    from run_receptor_ensemble_mvp import choose_train_best, validate_inputs
    from select_receptor_baselines import metrics_for_subset, read_csv
    from solve_discriminative_coverage_qubo import select_discriminative_subset
    from split_ligand_benchmark import read_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--ligands", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--target-size", type=int, default=2)
    parser.add_argument("--active-weight", type=float, default=0.5)
    parser.add_argument("--decoy-weight", type=float, default=0.5)
    parser.add_argument("--active-overlap-weight", type=float, default=0.5)
    args = parser.parse_args()

    matrix_rows = read_csv(args.matrix)
    ligand_rows = read_rows(args.ligands)
    if {row["ligand_id"] for row in matrix_rows} != {row["ligand_id"] for row in ligand_rows}:
        raise ValueError("matrix and ligand manifest IDs do not match")
    folds = make_folds(ligand_rows, args.folds, args.seed)
    fold_results = []
    oof = {"single": {}, "discriminative_qubo": {}, "all_mean": {}}
    for outer_fold in range(args.folds):
        validation_fold = (outer_fold + 1) % args.folds
        train_rows = [row for row in matrix_rows if folds[row["ligand_id"]] not in {outer_fold, validation_fold}]
        validation_rows = [row for row in matrix_rows if folds[row["ligand_id"]] == validation_fold]
        test_rows = [row for row in matrix_rows if folds[row["ligand_id"]] == outer_fold]
        split_manifest = [
            {"ligand_id": row["ligand_id"], "split": "train" if row in train_rows else "validation" if row in validation_rows else "test"}
            for row in matrix_rows
        ]
        validate_inputs(matrix_rows, split_manifest, args.receptor)
        subset, details = select_discriminative_subset(
            train_rows,
            args.receptor,
            args.target_size,
            utility_metric="bedroc",
            utility_normalization="minmax",
            active_weight=args.active_weight,
            decoy_weight=args.decoy_weight,
            active_overlap_weight=args.active_overlap_weight,
        )
        single = choose_train_best(train_rows, args.receptor, 1, "min_score")
        fold_results.append({
            "outer_fold": outer_fold,
            "validation_fold": validation_fold,
            "train_active_count": sum(row["label"] == "active" for row in train_rows),
            "validation_active_count": sum(row["label"] == "active" for row in validation_rows),
            "test_active_count": sum(row["label"] == "active" for row in test_rows),
            "single_subset": list(single),
            "discriminative_qubo_subset": list(subset),
            "best_energy": details["best_energy"],
            "test_single_metrics": metrics_for_subset(test_rows, single, "mean_score"),
            "test_discriminative_qubo_metrics": metrics_for_subset(test_rows, subset, "mean_score"),
        })
        oof["single"].update(collect_scores(test_rows, single, "mean_score"))
        oof["discriminative_qubo"].update(collect_scores(test_rows, subset, "mean_score"))
        oof["all_mean"].update(collect_scores(test_rows, tuple(args.receptor), "mean_score"))

    result = {
        "matrix": str(args.matrix),
        "ligands": str(args.ligands),
        "receptor_ids": args.receptor,
        "folds": args.folds,
        "seed": args.seed,
        "fixed_weights": {
            "active_weight": args.active_weight,
            "decoy_weight": args.decoy_weight,
            "active_overlap_weight": args.active_overlap_weight,
        },
        "fold_results": fold_results,
        "aggregate_out_of_fold_metrics": {
            name: ranked_metrics_with_ids(values) for name, values in oof.items()
        },
        "paired_bootstrap_qubo_minus_single": paired_bootstrap_delta(
            oof["single"], oof["discriminative_qubo"], 2000, args.seed + 1
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result["aggregate_out_of_fold_metrics"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
