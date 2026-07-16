"""Audit receptor-subset stability across nested development folds."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
from collections import Counter
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def jaccard(first: list[str], second: list[str]) -> float:
    left = set(first)
    right = set(second)
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def inclusion_rows(
    subsets: list[list[str]], receptor_ids: list[str]
) -> list[dict[str, object]]:
    counts = Counter(
        receptor_id for subset in subsets for receptor_id in set(subset)
    )
    total = len(subsets)
    return [
        {
            "receptor_id": receptor_id,
            "selection_count": counts[receptor_id],
            "selection_frequency": counts[receptor_id] / total,
        }
        for receptor_id in sorted(
            receptor_ids, key=lambda value: (-counts[value], value)
        )
    ]


def pairwise_jaccard_summary(
    subsets: list[list[str]],
) -> dict[str, float | int]:
    values = [
        jaccard(first, second)
        for first, second in itertools.combinations(subsets, 2)
    ]
    return {
        "comparison_count": len(values),
        "mean": statistics.fmean(values) if values else 1.0,
        "minimum": min(values, default=1.0),
        "maximum": max(values, default=1.0),
    }


def summarize_method(
    rows: list[dict[str, object]], receptor_ids: list[str]
) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: int(row["outer_fold"]))
    outer_subsets = [list(row["subset"]) for row in ordered]
    if any("inner_subsets" not in row for row in ordered):
        raise ValueError("outer results do not contain inner subsets")
    inner_subsets = [
        list(subset)
        for row in ordered
        for subset in row["inner_subsets"]
    ]
    outer_to_inner = [
        jaccard(list(row["subset"]), list(inner_subset))
        for row in ordered
        for inner_subset in row["inner_subsets"]
    ]
    outer_inclusion = inclusion_rows(outer_subsets, receptor_ids)
    inner_inclusion = inclusion_rows(inner_subsets, receptor_ids)
    outer_by_id = {
        row["receptor_id"]: float(row["selection_frequency"])
        for row in outer_inclusion
    }
    inner_by_id = {
        row["receptor_id"]: float(row["selection_frequency"])
        for row in inner_inclusion
    }
    primary_bedroc = [
        float(row["primary_outer_metrics"]["bedroc_alpha_20"])
        for row in ordered
    ]
    subset_counts = Counter("+".join(subset) for subset in outer_subsets)
    return {
        "outer_fold_count": len(outer_subsets),
        "inner_fit_count": len(inner_subsets),
        "outer_subsets": outer_subsets,
        "unique_outer_subset_count": len(subset_counts),
        "outer_subset_counts": dict(sorted(subset_counts.items())),
        "mean_outer_subset_size": statistics.fmean(
            len(subset) for subset in outer_subsets
        ),
        "outer_inclusion": outer_inclusion,
        "inner_inclusion": inner_inclusion,
        "outer_pairwise_jaccard": pairwise_jaccard_summary(outer_subsets),
        "inner_pairwise_jaccard": pairwise_jaccard_summary(inner_subsets),
        "outer_to_inner_jaccard": {
            "comparison_count": len(outer_to_inner),
            "mean": statistics.fmean(outer_to_inner),
            "minimum": min(outer_to_inner),
            "maximum": max(outer_to_inner),
        },
        "stable_receptors": [
            receptor_id
            for receptor_id in receptor_ids
            if outer_by_id[receptor_id] >= 0.75
            and inner_by_id[receptor_id] >= 0.5
        ],
        "primary_outer_bedroc": {
            "mean": statistics.fmean(primary_bedroc),
            "population_std": statistics.pstdev(primary_bedroc),
            "minimum": min(primary_bedroc),
            "maximum": max(primary_bedroc),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)

    summary = json.loads(args.summary.read_text(encoding="ascii"))
    if summary["test_lock"]["scores_evaluated"] is not False:
        raise ValueError("stability analysis requires a locked test split")
    receptor_ids = list(summary["inputs"]["primary_matrix"].get(
        "receptor_ids", []
    ))
    if not receptor_ids:
        receptor_ids = sorted(
            {
                receptor_id
                for row in summary["outer_fold_results"]
                for receptor_id in row["subset"]
            }
        )
        coefficient_ids = {
            receptor_id
            for row in summary["outer_fold_results"]
            for receptor_id in row.get("fit_details", {})
            .get("coefficients", {})
            .get("linear", {})
        }
        receptor_ids = sorted(set(receptor_ids) | coefficient_ids)
    by_method: dict[str, list[dict[str, object]]] = {}
    for row in summary["outer_fold_results"]:
        by_method.setdefault(str(row["method"]), []).append(row)
    method_summaries = {
        method: summarize_method(rows, receptor_ids)
        for method, rows in sorted(by_method.items())
    }
    coverage = method_summaries.get("coverage_qubo")
    pair_bedroc = method_summaries.get("pair_bedroc_qubo")
    output = {
        "schema_version": "1.0",
        "operation": (
            "development-only nested-fold receptor selection stability audit"
        ),
        "status": "ok",
        "summary": {
            "path": args.summary.as_posix(),
            "sha256": file_sha256(args.summary),
        },
        "locked_test_scores_evaluated": False,
        "receptor_count": len(receptor_ids),
        "receptor_ids": receptor_ids,
        "methods": method_summaries,
        "coverage_pair_bedroc_outer_subsets_identical": (
            coverage is not None
            and pair_bedroc is not None
            and coverage["outer_subsets"] == pair_bedroc["outer_subsets"]
        ),
        "stable_definition": (
            "outer selection frequency >= 0.75 and inner selection frequency "
            ">= 0.50"
        ),
        "interpretation_boundary": (
            "This audit measures selection consistency on nested development "
            "folds. It does not evaluate, rank, or release locked-test scores."
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
                "locked_test_scores_evaluated": False,
                "coverage_qubo": method_summaries.get("coverage_qubo"),
                "pair_bedroc_qubo": method_summaries.get(
                    "pair_bedroc_qubo"
                ),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
