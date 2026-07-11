"""Repeat leakage-aware ensemble selection over several stratified splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .run_receptor_ensemble_mvp import (
        choose_train_best,
        select_validation_tuned_qubo,
        validate_inputs,
    )
    from .select_receptor_baselines import metrics_for_subset, read_csv
    from .split_ligand_benchmark import read_rows, split_rows
except ImportError:
    from run_receptor_ensemble_mvp import (
        choose_train_best,
        select_validation_tuned_qubo,
        validate_inputs,
    )
    from select_receptor_baselines import metrics_for_subset, read_csv
    from split_ligand_benchmark import read_rows, split_rows


def run_one(
    matrix_rows: list[dict[str, str]],
    ligand_rows: list[dict[str, str]],
    receptor_ids: list[str],
    seed: int,
    target_size: int,
) -> dict[str, object]:
    split_manifest = split_rows(ligand_rows, 0.60, 0.20, seed)
    by_split = validate_inputs(matrix_rows, split_manifest, receptor_ids)
    tuned = select_validation_tuned_qubo(
        by_split["train"],
        by_split["validation"],
        receptor_ids,
        target_size,
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.25, 0.5],
        "roc_auc",
        "roc_auc",
        "mean_score",
    )
    qubo_subset = tuple(tuned["chosen"]["subset"])
    single_receptor = choose_train_best(by_split["train"], receptor_ids, 1, "min_score")
    qubo_test = metrics_for_subset(by_split["test"], qubo_subset, "mean_score")
    single_test = metrics_for_subset(by_split["test"], single_receptor, "min_score")
    return {
        "seed": seed,
        "split_counts": {
            split: {
                "ligand_count": len(rows),
                "active_count": sum(row["label"] == "active" for row in rows),
            }
            for split, rows in by_split.items()
        },
        "single_best_train": list(single_receptor),
        "qubo_subset": list(qubo_subset),
        "qubo_weights": {
            key: tuned["chosen"][key]
            for key in ("coverage_weight", "overlap_weight", "redundancy_weight")
        },
        "test_qubo": qubo_test,
        "test_single": single_test,
        "test_delta_qubo_minus_single": {
            metric: float(qubo_test[metric]) - float(single_test[metric])
            for metric in ("roc_auc", "pr_auc_average_precision", "bedroc_alpha_20")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--ligands", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=2)
    args = parser.parse_args()

    matrix_rows = read_csv(args.matrix)
    ligand_rows = read_rows(args.ligands)
    results = [
        run_one(matrix_rows, ligand_rows, args.receptor, seed, args.target_size)
        for seed in args.seeds
    ]
    summary = {
        "matrix": str(args.matrix),
        "ligands": str(args.ligands),
        "receptor_ids": args.receptor,
        "seeds": args.seeds,
        "results": results,
        "interpretation": "Repeated split analysis is exploratory because each test split contains only two actives.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
