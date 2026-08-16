"""Build and exhaustively solve a small QUBO receptor-subset prototype.

Thin CLI wrapper; the core logic lives in
``qubo_receptor_ensemble.qubo`` and ``qubo_receptor_ensemble.io``.
"""



from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import argparse
import itertools
import json
from pathlib import Path

from qubo_receptor_ensemble.io import read_csv
from qubo_receptor_ensemble.qubo import build_qubo, objective, train_data

__all__ = ["read_csv", "train_data", "build_qubo", "objective"]


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
    parser.add_argument(
        "--utility-metric",
        choices=["roc_auc", "bedroc", "ef5"],
        default="roc_auc",
    )
    parser.add_argument(
        "--utility-normalization",
        choices=["none", "minmax"],
        default="none",
    )
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
        args.utility_metric,
        args.utility_normalization,
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
