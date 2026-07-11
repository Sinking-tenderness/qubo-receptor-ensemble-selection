"""Select classical receptor subsets on train and evaluate them on all splits."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

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


def metrics_for_subset(
    rows: list[dict[str, str]], subset: tuple[str, ...], method: str
) -> dict[str, object]:
    data: dict[str, dict[str, object]] = {}
    for row in rows:
        scores = [float(row[receptor_id]) for receptor_id in subset]
        score = min(scores) if method == "min_score" else sum(scores) / len(scores)
        data[row["ligand_id"]] = {"label": row["label"], method: score}
    return ranked_metrics_with_ids(data, method)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection-metric",
        choices=["roc_auc", "pr_auc_average_precision", "bedroc_alpha_20"],
        default="roc_auc",
    )
    args = parser.parse_args()

    matrix_rows = read_csv(args.matrix)
    split_rows = read_csv(args.split_manifest)
    split_by_ligand = {row["ligand_id"]: row["split"] for row in split_rows}
    if {row["ligand_id"] for row in matrix_rows} != set(split_by_ligand):
        raise ValueError("matrix and split manifest ligand IDs do not match")
    for row in matrix_rows:
        for receptor_id in args.receptor:
            if row.get(receptor_id, "") == "":
                raise ValueError(f"missing score: {row['ligand_id']} / {receptor_id}")

    split_rows_by_name = {
        split: [row for row in matrix_rows if split_by_ligand[row["ligand_id"]] == split]
        for split in ("train", "validation", "test")
    }
    result: dict[str, object] = {
        "selection_split": "train",
        "selection_metric": args.selection_metric,
        "receptor_ids": args.receptor,
        "subsets": [],
    }
    for size in range(1, len(args.receptor) + 1):
        for subset in itertools.combinations(args.receptor, size):
            subset_name = "+".join(subset)
            for method in ("min_score", "mean_score"):
                train_metrics = metrics_for_subset(
                    split_rows_by_name["train"], subset, method
                )
                entry = {
                    "subset": list(subset),
                    "subset_name": subset_name,
                    "size": size,
                    "method": method,
                    "train_selection_metric": train_metrics[args.selection_metric],
                    "train_metrics": train_metrics,
                    "evaluation": {
                        split: metrics_for_subset(rows, subset, method)
                        for split, rows in split_rows_by_name.items()
                    },
                }
                result["subsets"].append(entry)

    selected: dict[str, object] = {}
    for size in range(1, len(args.receptor) + 1):
        candidates = [
            entry
            for entry in result["subsets"]
            if entry["size"] == size
        ]
        for method in ("min_score", "mean_score"):
            method_candidates = [entry for entry in candidates if entry["method"] == method]
            chosen = max(
                method_candidates,
                key=lambda entry: (
                    float(entry["train_selection_metric"]),
                    tuple(entry["subset"]),
                ),
            )
            selected[f"size_{size}_{method}"] = chosen
    result["selected_on_train"] = selected

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result["selected_on_train"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
