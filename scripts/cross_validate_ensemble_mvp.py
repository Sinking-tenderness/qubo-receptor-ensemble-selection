"""Run outer stratified cross-validation for receptor-subset selection."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .run_receptor_ensemble_mvp import (
        choose_train_best,
        select_validation_tuned_qubo,
        validate_inputs,
    )
    from .select_receptor_baselines import metrics_for_subset, read_csv
    from .split_ligand_benchmark import read_rows
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from run_receptor_ensemble_mvp import (
        choose_train_best,
        select_validation_tuned_qubo,
        validate_inputs,
    )
    from select_receptor_baselines import metrics_for_subset, read_csv
    from split_ligand_benchmark import read_rows


def make_folds(rows: list[dict[str, str]], fold_count: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    assignment: dict[str, int] = {}
    for label in ("active", "decoy"):
        ids = [row["ligand_id"] for row in rows if row["label"] == label]
        rng.shuffle(ids)
        for index, ligand_id in enumerate(ids):
            assignment[ligand_id] = index % fold_count
    return assignment


def collect_scores(
    rows: list[dict[str, str]], subset: tuple[str, ...], aggregation: str
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        scores = [float(row[receptor_id]) for receptor_id in subset]
        score = min(scores) if aggregation == "min_score" else sum(scores) / len(scores)
        output[row["ligand_id"]] = {"label": row["label"], "score": score}
    return output


def normalize_rows_by_train_minmax(
    train_rows: list[dict[str, str]],
    other_rows: list[list[dict[str, str]]],
    receptor_ids: list[str],
) -> tuple[list[dict[str, object]], list[list[dict[str, object]]]]:
    """Normalize each receptor score using train extrema only.

    Lower Vina scores remain better. Test and validation values are transformed
    with statistics fitted on train, so no held-out score distribution is used
    to choose the receptor subset.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for receptor_id in receptor_ids:
        values = [float(row[receptor_id]) for row in train_rows]
        bounds[receptor_id] = (min(values), max(values))

    def transform(rows: list[dict[str, str]]) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        for row in rows:
            transformed: dict[str, object] = dict(row)
            for receptor_id in receptor_ids:
                lower, upper = bounds[receptor_id]
                value = float(row[receptor_id])
                transformed[receptor_id] = (
                    0.0 if upper == lower else (value - lower) / (upper - lower)
                )
            output.append(transformed)
        return output

    return transform(train_rows), [transform(rows) for rows in other_rows]


def ranked_metric_values(records: dict[str, dict[str, object]]) -> dict[str, float]:
    ranked_ids = sorted(records, key=lambda ligand_id: (-float(records[ligand_id]["score"]), ligand_id))
    ranked = [
        {
            "label": records[ligand_id]["label"],
            "binary_label": int(records[ligand_id]["label"] == "active"),
            "ranking_score": float(records[ligand_id]["score"]),
        }
        for ligand_id in ranked_ids
    ]
    return {
        "roc_auc": ranked_metrics_with_ids(records)["roc_auc"],
        "pr_auc_average_precision": ranked_metrics_with_ids(records)["pr_auc_average_precision"],
        "bedroc_alpha_20": ranked_metrics_with_ids(records)["bedroc_alpha_20"],
    }


def paired_bootstrap_delta(
    first: dict[str, dict[str, object]],
    second: dict[str, dict[str, object]],
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    ids = sorted(first)
    if set(ids) != set(second):
        raise ValueError("paired bootstrap inputs have different ligand IDs")
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {key: [] for key in ("roc_auc", "pr_auc_average_precision", "bedroc_alpha_20")}
    skipped = 0
    for _ in range(iterations):
        selected = [rng.choice(ids) for _ in ids]
        first_sample = {
            f"{ligand_id}__{index}": first[ligand_id]
            for index, ligand_id in enumerate(selected)
        }
        second_sample = {
            f"{ligand_id}__{index}": second[ligand_id]
            for index, ligand_id in enumerate(selected)
        }
        if {row["label"] for row in first_sample.values()} != {"active", "decoy"}:
            skipped += 1
            continue
        first_metrics = ranked_metric_values(first_sample)
        second_metrics = ranked_metric_values(second_sample)
        for key in samples:
            delta = float(second_metrics[key]) - float(first_metrics[key])
            if not math.isnan(delta):
                samples[key].append(delta)

    def percentile(values: list[float], q: float) -> float:
        values = sorted(values)
        if not values:
            return math.nan
        position = (len(values) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    return {
        key: {
            "mean_delta": sum(values) / len(values) if values else math.nan,
            "ci95_low": percentile(values, 0.025),
            "ci95_high": percentile(values, 0.975),
            "n_bootstrap_used": len(values),
            "n_bootstrap_skipped": skipped,
        }
        for key, values in samples.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--ligands", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--target-size", type=int, default=2)
    parser.add_argument("--utility-metric", choices=["roc_auc", "bedroc", "ef5"], default="roc_auc")
    parser.add_argument(
        "--validation-metric",
        choices=["roc_auc", "pr_auc_average_precision", "bedroc_alpha_20"],
        default="roc_auc",
    )
    parser.add_argument("--aggregation", choices=["min_score", "mean_score"], default="mean_score")
    parser.add_argument(
        "--score-normalization",
        choices=["raw", "train_minmax"],
        default="raw",
        help="Normalize receptor scores using train extrema before selection and evaluation.",
    )
    args = parser.parse_args()
    if args.folds < 3:
        raise ValueError("at least three folds are required")

    matrix_rows = read_csv(args.matrix)
    ligand_rows = read_rows(args.ligands)
    by_id = {row["ligand_id"]: row for row in ligand_rows}
    if {row["ligand_id"] for row in matrix_rows} != set(by_id):
        raise ValueError("matrix and ligand manifest IDs do not match")
    folds = make_folds(ligand_rows, args.folds, args.seed)
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
        by_split = validate_inputs(matrix_rows, split_manifest, args.receptor)
        del by_split
        tuned = select_validation_tuned_qubo(
            train_rows,
            validation_rows,
            args.receptor,
            args.target_size,
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.25, 0.5],
            args.utility_metric,
            args.validation_metric,
            args.aggregation,
        )
        qubo_subset = tuple(tuned["chosen"]["subset"])
        single_subset = choose_train_best(train_rows, args.receptor, 1, "min_score")
        fold_results.append(
            {
                "outer_fold": outer_fold,
                "validation_fold": validation_fold,
                "train_active_count": sum(row["label"] == "active" for row in train_rows),
                "validation_active_count": sum(row["label"] == "active" for row in validation_rows),
                "test_active_count": sum(row["label"] == "active" for row in test_rows),
                "single_subset": list(single_subset),
                "qubo_subset": list(qubo_subset),
                "qubo_weights": {
                    key: tuned["chosen"][key]
                    for key in ("coverage_weight", "overlap_weight", "redundancy_weight")
                },
                "test_single_metrics": metrics_for_subset(test_rows, single_subset, args.aggregation),
                "test_qubo_metrics": metrics_for_subset(test_rows, qubo_subset, args.aggregation),
            }
        )
        for name, subset in (("single", single_subset), ("qubo", qubo_subset)):
            oof_scores[name].update(collect_scores(test_rows, subset, args.aggregation))
        oof_scores["all_mean"].update(collect_scores(test_rows, tuple(args.receptor), args.aggregation))

    aggregate = {
        name: ranked_metrics_with_ids(values)
        for name, values in oof_scores.items()
    }
    result = {
        "matrix": str(args.matrix),
        "ligands": str(args.ligands),
        "receptor_ids": args.receptor,
        "folds": args.folds,
        "seed": args.seed,
        "utility_metric": args.utility_metric,
        "validation_metric": args.validation_metric,
        "aggregation": args.aggregation,
        "score_normalization": args.score_normalization,
        "protocol": "outer test fold is never used for selection; next fold is validation; remaining folds are train",
        "fold_results": fold_results,
        "aggregate_out_of_fold_metrics": aggregate,
        "oof_scores": oof_scores,
        "paired_bootstrap_qubo_minus_single": paired_bootstrap_delta(
            oof_scores["single"], oof_scores["qubo"], 2000, args.seed + 1
        ),
        "limitations": [
            "This is cross-validation on a small benchmark, not an independent external test set.",
            "DUD-E decoys are benchmark decoys rather than experimentally confirmed inactives.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result["aggregate_out_of_fold_metrics"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
