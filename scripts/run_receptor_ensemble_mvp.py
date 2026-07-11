"""Run the leakage-aware receptor-ensemble selection MVP.

The score matrix is treated as a fixed observation table. Receptor selection is
performed on train only; QUBO hyperparameters are selected on validation; the
test split is evaluated once at the end.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

try:
    from .select_receptor_baselines import metrics_for_subset, read_csv
    from .solve_coverage_qubo import coverage_objective, coverage_terms
    from .solve_qubo_receptor_subset import build_qubo
except ImportError:
    from select_receptor_baselines import metrics_for_subset, read_csv
    from solve_coverage_qubo import coverage_objective, coverage_terms
    from solve_qubo_receptor_subset import build_qubo


def parse_grid(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("a parameter grid cannot be empty")
    return values


def validate_inputs(
    matrix_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
    receptor_ids: list[str],
) -> dict[str, list[dict[str, str]]]:
    matrix_ids = {row["ligand_id"] for row in matrix_rows}
    split_by_ligand = {row["ligand_id"]: row["split"] for row in split_rows}
    if matrix_ids != set(split_by_ligand):
        raise ValueError("matrix and split manifest ligand IDs do not match")
    if set(split_by_ligand.values()) != {"train", "validation", "test"}:
        raise ValueError("split manifest must contain train, validation, and test")
    for row in matrix_rows:
        for receptor_id in receptor_ids:
            if not row.get(receptor_id, ""):
                raise ValueError(f"missing score: {row['ligand_id']} / {receptor_id}")
            float(row[receptor_id])
    return {
        split: [row for row in matrix_rows if split_by_ligand[row["ligand_id"]] == split]
        for split in ("train", "validation", "test")
    }


def enumerate_fixed_size(
    receptor_ids: list[str], target_size: int
) -> list[tuple[str, ...]]:
    if not 1 <= target_size <= len(receptor_ids):
        raise ValueError("target_size must be between 1 and receptor count")
    return list(itertools.combinations(receptor_ids, target_size))


def choose_train_best(
    train_rows: list[dict[str, str]],
    receptor_ids: list[str],
    target_size: int,
    aggregation: str,
) -> tuple[str, ...]:
    candidates = enumerate_fixed_size(receptor_ids, target_size)
    return max(
        candidates,
        key=lambda subset: (
            float(metrics_for_subset(train_rows, subset, aggregation)["roc_auc"]),
            tuple(subset),
        ),
    )


def solve_coverage_qubo(
    train_rows: list[dict[str, str]],
    receptor_ids: list[str],
    target_size: int,
    coverage_weight: float,
    overlap_weight: float,
    redundancy_weight: float,
    utility_metric: str,
) -> tuple[tuple[str, ...], float]:
    base = build_qubo(
        train_rows,
        receptor_ids,
        target_size,
        redundancy_weight,
        0.0,
        1.0,
        utility_metric,
        "minmax",
    )
    _, rewards, overlaps = coverage_terms(train_rows, receptor_ids, 0.10)
    candidates = enumerate_fixed_size(receptor_ids, target_size)
    scored = [
        (
            coverage_objective(
                subset,
                base,
                rewards,
                overlaps,
                coverage_weight,
                overlap_weight,
            ),
            subset,
        )
        for subset in candidates
    ]
    return min(scored, key=lambda item: (item[0], item[1]))[1], min(scored)[0]


def select_validation_tuned_qubo(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    receptor_ids: list[str],
    target_size: int,
    coverage_weights: list[float],
    overlap_weights: list[float],
    redundancy_weights: list[float],
    utility_metric: str,
    validation_metric: str,
    validation_aggregation: str,
) -> dict[str, object]:
    trials: list[dict[str, object]] = []
    for coverage_weight, overlap_weight, redundancy_weight in itertools.product(
        coverage_weights, overlap_weights, redundancy_weights
    ):
        subset, objective = solve_coverage_qubo(
            train_rows,
            receptor_ids,
            target_size,
            coverage_weight,
            overlap_weight,
            redundancy_weight,
            utility_metric,
        )
        validation_metrics = metrics_for_subset(
            validation_rows, subset, validation_aggregation
        )
        trials.append(
            {
                "subset": list(subset),
                "objective_train": objective,
                "coverage_weight": coverage_weight,
                "overlap_weight": overlap_weight,
                "redundancy_weight": redundancy_weight,
                "validation_metrics": validation_metrics,
            }
        )
    chosen = max(
        trials,
        key=lambda row: (
            float(row["validation_metrics"][validation_metric]),
            float(row["validation_metrics"]["bedroc_alpha_20"]),
            tuple(row["subset"]),
        ),
    )
    return {"chosen": chosen, "trials": trials, "validation_metric": validation_metric}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=2)
    parser.add_argument("--coverage-grid", default="0,0.5,1")
    parser.add_argument("--overlap-grid", default="0,0.5,1")
    parser.add_argument("--redundancy-grid", default="0,0.25,0.5")
    parser.add_argument("--utility-metric", choices=["roc_auc", "bedroc", "ef5"], default="roc_auc")
    parser.add_argument(
        "--validation-metric",
        choices=["roc_auc", "pr_auc_average_precision", "bedroc_alpha_20"],
        default="roc_auc",
    )
    parser.add_argument(
        "--qubo-aggregation",
        choices=["min_score", "mean_score"],
        default="min_score",
        help="Aggregation used when selecting QUBO weights on validation.",
    )
    args = parser.parse_args()

    matrix_rows = read_csv(args.matrix)
    split_rows = read_csv(args.split_manifest)
    by_split = validate_inputs(matrix_rows, split_rows, args.receptor)
    if any(len(by_split[name]) == 0 for name in by_split):
        raise ValueError("each split must contain at least one ligand")

    train_best_min = choose_train_best(by_split["train"], args.receptor, args.target_size, "min_score")
    train_best_mean = choose_train_best(by_split["train"], args.receptor, args.target_size, "mean_score")
    tuned = select_validation_tuned_qubo(
        by_split["train"],
        by_split["validation"],
        args.receptor,
        args.target_size,
        parse_grid(args.coverage_grid),
        parse_grid(args.overlap_grid),
        parse_grid(args.redundancy_grid),
        args.utility_metric,
        args.validation_metric,
        args.qubo_aggregation,
    )
    qubo_subset = tuple(tuned["chosen"]["subset"])
    methods: list[tuple[str, tuple[str, ...], str]] = [
        ("single_best_train", (choose_train_best(by_split["train"], args.receptor, 1, "min_score")[0],), "min_score"),
        ("train_best_min_score", train_best_min, "min_score"),
        ("train_best_mean_score", train_best_mean, "mean_score"),
        ("qubo_validation_tuned", qubo_subset, args.qubo_aggregation),
        ("all_receptors_min_score", tuple(args.receptor), "min_score"),
        ("all_receptors_mean_score", tuple(args.receptor), "mean_score"),
    ]

    comparison: list[dict[str, object]] = []
    for split_name in ("train", "validation", "test"):
        for method_name, subset, aggregation in methods:
            metrics = metrics_for_subset(by_split[split_name], subset, aggregation)
            comparison.append(
                {
                    "split": split_name,
                    "method": method_name,
                    "subset": "+".join(subset),
                    "aggregation": aggregation,
                    **metrics,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_output = args.output_dir / "receptor_ensemble_mvp.json"
    csv_output = args.output_dir / "receptor_ensemble_mvp_comparison.csv"
    report_output = args.output_dir / "receptor_ensemble_mvp_report.md"
    result = {
        "mvp_version": "1.0",
        "matrix": str(args.matrix),
        "split_manifest": str(args.split_manifest),
        "receptor_ids": args.receptor,
        "target_size": args.target_size,
        "split_counts": {
            split: {
                "ligand_count": len(rows),
                "active_count": sum(row["label"] == "active" for row in rows),
            }
            for split, rows in by_split.items()
        },
        "selection_protocol": {
            "classical_baselines_selected_on": "train",
            "qubo_subset_selected_from_train": True,
            "qubo_hyperparameters_selected_on": "validation",
            "test_used_only_for_final_evaluation": True,
            "validation_metric": args.validation_metric,
            "utility_metric": args.utility_metric,
            "qubo_validation_aggregation": args.qubo_aggregation,
        },
        "selected_methods": {
            "single_best_train": [methods[0][1][0]],
            "train_best_min_score": list(train_best_min),
            "train_best_mean_score": list(train_best_mean),
            "qubo_validation_tuned": list(qubo_subset),
            "all_receptors": args.receptor,
        },
        "qubo_tuning": tuned,
        "comparison": comparison,
        "limitations": [
            "Only three receptor conformers and a 10-active/50-decoy DUD-E teaching subset are included.",
            "Validation and test each contain only two actives, so early-enrichment estimates are unstable.",
            "DUD-E decoys are benchmark decoys, not experimentally confirmed inactives.",
            "The QUBO uses docking-derived utility and score correlation; this is not evidence of quantum advantage.",
        ],
    }
    json_output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    test_rows = [row for row in comparison if row["split"] == "test"]
    report_lines = [
        "# CDK2 Receptor-Ensemble Selection MVP",
        "",
        "## Protocol",
        "",
        "- Receptor selection uses the train split only.",
        "- QUBO weights are selected on validation only.",
        "- The test split is used once for final comparison.",
        f"- Receptor pool: {', '.join(args.receptor)}",
        f"- Target subset size: {args.target_size}",
        "",
        "## Selected subsets",
        "",
        f"- Train-selected single receptor: {methods[0][1][0]}",
        f"- Train-selected min-score subset: {' + '.join(train_best_min)}",
        f"- Validation-tuned QUBO subset: {' + '.join(qubo_subset)}",
        "",
        "## Held-out test comparison",
        "",
        "| Method | Subset | ROC-AUC | PR-AUC | BEDROC | EF1% | EF5% |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in test_rows:
        report_lines.append(
            f"| {row['method']} | {row['subset']} | {float(row['roc_auc']):.3f} | "
            f"{float(row['pr_auc_average_precision']):.3f} | {float(row['bedroc_alpha_20']):.3f} | "
            f"{float(row['EF1%']):.3f} | {float(row['EF5%']):.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a reproducible engineering MVP, not a claim of quantum advantage or biological optimality.",
            "The current benchmark is too small to support a strong generalization claim; the next scientific step is to expand the conformer pool and ligand benchmark while keeping this protocol frozen.",
        ]
    )
    report_output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_output), "csv": str(csv_output), "report": str(report_output), "qubo_subset": list(qubo_subset)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
