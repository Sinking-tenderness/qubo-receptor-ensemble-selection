"""Scan development-only receptor subsets as an optimistic diagnostic."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

try:
    from .prepare_receptor import file_sha256
    from .run_receptor_selection_validation_gate import (
        normalize_from_train,
        read_csv,
        subset_metrics,
    )
except ImportError:
    from prepare_receptor import file_sha256
    from run_receptor_selection_validation_gate import (
        normalize_from_train,
        read_csv,
        subset_metrics,
    )


METADATA_COLUMNS = {"target_id", "ligand_id", "label"}


def rank_subset_candidates(
    rows: list[dict[str, str]],
    receptor_ids: list[str],
    max_size: int,
    aggregation: str,
) -> list[dict[str, object]]:
    if aggregation not in {"min_score", "mean_score"}:
        raise ValueError("unsupported aggregation")
    if not 1 <= max_size <= len(receptor_ids):
        raise ValueError("max_size is outside the receptor pool")
    normalized, _, _ = normalize_from_train(rows, rows, receptor_ids)
    candidates: list[dict[str, object]] = []
    for size in range(1, max_size + 1):
        for subset in itertools.combinations(receptor_ids, size):
            metrics = subset_metrics(normalized, subset, aggregation)
            candidates.append(
                {
                    "subset": list(subset),
                    "target_size": size,
                    "aggregation": aggregation,
                    "metrics": metrics,
                }
            )
    candidates.sort(
        key=lambda row: (
            -float(row["metrics"]["bedroc_alpha_20"]),
            -float(row["metrics"]["pr_auc_average_precision"]),
            -float(row["metrics"]["roc_auc"]),
            tuple(row["subset"]),
        )
    )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-matrix", type=Path, required=True)
    parser.add_argument("--sensitivity-matrix", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    if args.top_n < 1:
        raise ValueError("top-n must be positive")

    split_rows = read_csv(args.split_manifest)
    development_ids = {
        row["ligand_id"]
        for row in split_rows
        if row["split"] in {"train", "validation"}
    }
    locked_ids = {
        row["ligand_id"]
        for row in split_rows
        if row["split"] == "test"
    }
    matrix_rows: dict[str, list[dict[str, str]]] = {}
    receptor_ids: list[str] | None = None
    for name, path in (
        ("primary", args.primary_matrix),
        ("sensitivity", args.sensitivity_matrix),
    ):
        rows = read_csv(path)
        ids = {row["ligand_id"] for row in rows}
        if ids != development_ids:
            raise ValueError(f"{name} matrix is not development-only")
        if ids & locked_ids:
            raise ValueError(f"{name} matrix contains locked test IDs")
        current_receptors = [
            field for field in rows[0] if field not in METADATA_COLUMNS
        ]
        if receptor_ids is None:
            receptor_ids = current_receptors
        elif current_receptors != receptor_ids:
            raise ValueError("primary and sensitivity receptor columns differ")
        matrix_rows[name] = rows

    assert receptor_ids is not None
    results: dict[str, dict[str, list[dict[str, object]]]] = {}
    for matrix_name, rows in matrix_rows.items():
        results[matrix_name] = {}
        for aggregation in ("min_score", "mean_score"):
            results[matrix_name][aggregation] = rank_subset_candidates(
                rows, receptor_ids, args.max_size, aggregation
            )[: args.top_n]

    output = {
        "schema_version": "1.0",
        "operation": (
            "optimistic full-development subset scan; not nested CV and not "
            "a test evaluation"
        ),
        "status": "ok",
        "input_sha256": {
            "primary_matrix": file_sha256(args.primary_matrix),
            "sensitivity_matrix": file_sha256(args.sensitivity_matrix),
            "split_manifest": file_sha256(args.split_manifest),
        },
        "development_ligand_count": len(development_ids),
        "locked_test_ligand_count": len(locked_ids),
        "locked_test_scores_evaluated": False,
        "receptor_ids": receptor_ids,
        "max_subset_size": args.max_size,
        "top_n_per_matrix_and_aggregation": args.top_n,
        "results": results,
        "interpretation_boundary": (
            "This scan uses all development ligands to estimate an optimistic "
            "upper bound. It is diagnostic only and cannot release or validate "
            "the locked test split."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "development_ligand_count": output[
                    "development_ligand_count"
                ],
                "locked_test_scores_evaluated": output[
                    "locked_test_scores_evaluated"
                ],
                "top_primary_mean_bedroc": results["primary"][
                    "mean_score"
                ][0],
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
