"""Run a transparent sensitivity grid for the QUBO receptor subset prototype."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

try:
    from .select_receptor_baselines import metrics_for_subset, read_csv
    from .solve_qubo_receptor_subset import build_qubo, objective
except ImportError:
    from select_receptor_baselines import metrics_for_subset, read_csv
    from solve_qubo_receptor_subset import build_qubo, objective


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-sizes", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--redundancy-weights", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 1.0]
    )
    parser.add_argument("--count-weight", type=float, default=0.10)
    parser.add_argument("--size-weight", type=float, default=1.0)
    args = parser.parse_args()

    matrix_rows = read_csv(args.matrix)
    split_rows = read_csv(args.split_manifest)
    split_by_ligand = {row["ligand_id"]: row["split"] for row in split_rows}
    train_rows = [row for row in matrix_rows if split_by_ligand[row["ligand_id"]] == "train"]
    rows_by_split = {
        split: [row for row in matrix_rows if split_by_ligand[row["ligand_id"]] == split]
        for split in ("train", "validation", "test")
    }
    results = []
    for target_size in args.target_sizes:
        if not 0 <= target_size <= len(args.receptor):
            raise ValueError(f"invalid target size: {target_size}")
        for redundancy_weight in args.redundancy_weights:
            qubo = build_qubo(
                train_rows,
                args.receptor,
                target_size,
                redundancy_weight,
                args.count_weight,
                args.size_weight,
            )
            candidates = []
            for size in range(len(args.receptor) + 1):
                for subset in itertools.combinations(args.receptor, size):
                    candidates.append((objective(subset, qubo), subset))
            _, selected = min(candidates, key=lambda item: (item[0], item[1]))
            results.append(
                {
                    "target_size": target_size,
                    "redundancy_weight": redundancy_weight,
                    "selected_subset": list(selected),
                    "objective": objective(selected, qubo),
                    "metrics": {
                        split: {
                            method: metrics_for_subset(rows, selected, method)
                            for method in ("min_score", "mean_score")
                        }
                        for split, rows in rows_by_split.items()
                    },
                }
            )

    output = {
        "selection_split": "train",
        "weights_fixed": {
            "count": args.count_weight,
            "size": args.size_weight,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    for result in results:
        print(
            f"K={result['target_size']} redundancy={result['redundancy_weight']} "
            f"subset={'+'.join(result['selected_subset'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
