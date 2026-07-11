"""Analyze complementarity and simple ensemble baselines from a score matrix."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids


def read_matrix(path: Path, receptor_ids: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ligand_id", "label", *receptor_ids}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"matrix is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("matrix contains no rows")
    for row in rows:
        for receptor_id in receptor_ids:
            if row[receptor_id] == "":
                raise ValueError(f"missing score for {row['ligand_id']} / {receptor_id}")
    return rows


def receptor_data(rows: list[dict[str, str]], receptor_id: str) -> dict[str, dict[str, object]]:
    return {
        row["ligand_id"]: {"label": row["label"], "score": float(row[receptor_id])}
        for row in rows
    }


def top_ids(data: dict[str, dict[str, object]], n: int = 10) -> list[str]:
    return sorted(data, key=lambda ligand_id: (float(data[ligand_id]["score"]), ligand_id))[:n]


def ensemble_data(
    rows: list[dict[str, str]], receptor_ids: list[str], method: str
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        scores = [float(row[receptor_id]) for receptor_id in receptor_ids]
        value = min(scores) if method == "min_score" else sum(scores) / len(scores)
        output[row["ligand_id"]] = {"label": row["label"], method: value}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_matrix(args.matrix, args.receptor)
    data = {receptor_id: receptor_data(rows, receptor_id) for receptor_id in args.receptor}
    top10 = {receptor_id: top_ids(data[receptor_id]) for receptor_id in args.receptor}
    correlations: dict[str, dict[str, float]] = {}
    for first, second in combinations(args.receptor, 2):
        first_scores = [float(row[first]) for row in rows]
        second_scores = [float(row[second]) for row in rows]
        correlations[f"{first}__{second}"] = {
            "spearman": float(spearmanr(first_scores, second_scores).statistic),
            "pearson": float(pearsonr(first_scores, second_scores).statistic),
        }

    active_ids = {row["ligand_id"] for row in rows if row["label"] == "active"}
    top10_active = {
        receptor_id: sorted(set(top10[receptor_id]) & active_ids)
        for receptor_id in args.receptor
    }
    top10_union = set().union(*(set(ids) for ids in top10_active.values()))
    best_receptor_counts = {receptor_id: 0 for receptor_id in args.receptor}
    for row in rows:
        best = min(args.receptor, key=lambda receptor_id: float(row[receptor_id]))
        best_receptor_counts[best] += 1

    summary = {
        "ligand_count": len(rows),
        "active_count": len(active_ids),
        "receptor_ids": args.receptor,
        "score_correlations": correlations,
        "top10": {
            "ligand_ids": top10,
            "active_ids": top10_active,
            "active_union_ids": sorted(top10_union),
            "active_union_count": len(top10_union),
            "active_union_fraction": len(top10_union) / len(active_ids),
        },
        "pairwise_top10_overlap": {
            f"{first}__{second}": {
                "overlap_count": len(set(top10[first]) & set(top10[second])),
                "jaccard": len(set(top10[first]) & set(top10[second]))
                / len(set(top10[first]) | set(top10[second])),
                "active_overlap_count": len(
                    set(top10_active[first]) & set(top10_active[second])
                ),
            }
            for first, second in combinations(args.receptor, 2)
        },
        "best_receptor_by_ligand_count": best_receptor_counts,
        "metrics": {
            receptor_id: ranked_metrics_with_ids(data[receptor_id])
            for receptor_id in args.receptor
        },
    }
    for method in ("min_score", "mean_score"):
        ensemble = ensemble_data(rows, args.receptor, method)
        summary["metrics"][f"ensemble_{method}"] = ranked_metrics_with_ids(ensemble, method)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
