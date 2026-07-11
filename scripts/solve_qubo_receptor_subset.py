"""Build and exhaustively solve a small QUBO receptor-subset prototype."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

from scipy.stats import spearmanr

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def train_data(rows: list[dict[str, str]], receptor_id: str) -> dict[str, dict[str, object]]:
    return {
        row["ligand_id"]: {"label": row["label"], "score": float(row[receptor_id])}
        for row in rows
    }


def build_qubo(
    rows: list[dict[str, str]],
    receptor_ids: list[str],
    target_size: int,
    redundancy_weight: float,
    count_weight: float,
    size_weight: float,
) -> dict[str, object]:
    utilities: dict[str, float] = {}
    train_scores: dict[str, list[float]] = {}
    for receptor_id in receptor_ids:
        data = train_data(rows, receptor_id)
        utilities[receptor_id] = float(ranked_metrics_with_ids(data)["roc_auc"])
        train_scores[receptor_id] = [float(row[receptor_id]) for row in rows]

    redundancy: dict[str, float] = {}
    for first, second in itertools.combinations(receptor_ids, 2):
        value = float(spearmanr(train_scores[first], train_scores[second]).statistic)
        redundancy[f"{first}__{second}"] = max(0.0, value)

    linear = {
        receptor_id: -utilities[receptor_id]
        + count_weight
        + size_weight * (1 - 2 * target_size)
        for receptor_id in receptor_ids
    }
    quadratic = {
        key: redundancy[key] * redundancy_weight + 2 * size_weight
        for key in redundancy
    }

    return {
        "target_size": target_size,
        "weights": {
            "redundancy": redundancy_weight,
            "count": count_weight,
            "size": size_weight,
        },
        "utilities_train_roc_auc": utilities,
        "redundancy_train_spearman_clipped": redundancy,
        "linear_coefficients": linear,
        "quadratic_coefficients": quadratic,
    }


def objective(
    subset: tuple[str, ...], qubo: dict[str, object]
) -> float:
    utilities = qubo["utilities_train_roc_auc"]
    redundancy = qubo["redundancy_train_spearman_clipped"]
    weights = qubo["weights"]
    target_size = qubo["target_size"]
    value = -sum(utilities[receptor_id] for receptor_id in subset)
    value += weights["count"] * len(subset)
    value += weights["size"] * (len(subset) - target_size) ** 2
    for first, second in itertools.combinations(subset, 2):
        value += weights["redundancy"] * redundancy[f"{first}__{second}"]
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=2)
    parser.add_argument("--redundancy-weight", type=float, default=0.25)
    parser.add_argument("--count-weight", type=float, default=0.10)
    parser.add_argument("--size-weight", type=float, default=1.0)
    args = parser.parse_args()

    matrix_rows = read_csv(args.matrix)
    split_manifest = read_csv(args.split_manifest)
    split_by_ligand = {row["ligand_id"]: row["split"] for row in split_manifest}
    train_rows = [
        row for row in matrix_rows if split_by_ligand.get(row["ligand_id"]) == "train"
    ]
    if not train_rows:
        raise ValueError("no train rows found")
    if not 0 <= args.target_size <= len(args.receptor):
        raise ValueError("target size must be between zero and receptor count")

    qubo = build_qubo(
        train_rows,
        args.receptor,
        args.target_size,
        args.redundancy_weight,
        args.count_weight,
        args.size_weight,
    )
    candidates = []
    for size in range(len(args.receptor) + 1):
        for subset in itertools.combinations(args.receptor, size):
            candidates.append(
                {
                    "subset": list(subset),
                    "size": size,
                    "objective": objective(subset, qubo),
                }
            )
    candidates.sort(key=lambda row: (row["objective"], row["subset"]))
    result = {
        "selection_split": "train",
        "receptor_ids": args.receptor,
        "qubo": qubo,
        "best_subset": candidates[0],
        "all_candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result["best_subset"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
