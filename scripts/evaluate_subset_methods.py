"""Create one comparable train/validation/test table for receptor subsets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .select_receptor_baselines import metrics_for_subset, read_csv
except ImportError:
    from select_receptor_baselines import metrics_for_subset, read_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--qubo-output", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    matrix_rows = read_csv(args.matrix)
    split_rows = read_csv(args.split_manifest)
    split_by_ligand = {row["ligand_id"]: row["split"] for row in split_rows}
    qubo = json.loads(args.qubo_output.read_text(encoding="utf-8"))
    qubo_subset = tuple(qubo["best_subset"]["subset"])

    methods: list[tuple[str, tuple[str, ...]]] = [
        (f"single_{receptor_id}", (receptor_id,))
        for receptor_id in args.receptor
    ]
    methods.extend(
        [
            ("qubo_selected", qubo_subset),
            ("all_receptors", tuple(args.receptor)),
        ]
    )
    output_rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        split_rows_matrix = [
            row for row in matrix_rows if split_by_ligand[row["ligand_id"]] == split
        ]
        for method_name, subset in methods:
            for aggregation in ("min_score", "mean_score"):
                metrics = metrics_for_subset(split_rows_matrix, subset, aggregation)
                output_rows.append(
                    {
                        "split": split,
                        "method": method_name,
                        "subset": "+".join(subset),
                        "aggregation": aggregation,
                        "ligand_count": metrics["ligand_count"],
                        "active_count": metrics["active_count"],
                        "roc_auc": metrics["roc_auc"],
                        "pr_auc": metrics["pr_auc_average_precision"],
                        "bedroc": metrics["bedroc_alpha_20"],
                        "EF1%": metrics["EF1%"],
                        "EF5%": metrics["EF5%"],
                        "EF10%": metrics["EF10%"],
                    }
                )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(
            {
                "qubo_subset": list(qubo_subset),
                "rows": output_rows,
                "score_direction": "lower Vina score is better",
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="ascii",
    )
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"rows={len(output_rows)}")
    print(f"qubo_subset={'+'.join(qubo_subset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
