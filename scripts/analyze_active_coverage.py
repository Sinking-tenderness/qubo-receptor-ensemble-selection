"""Measure ligand-level active coverage and complementarity across receptors."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def top_active_ids(rows: list[dict[str, str]], receptor_id: str, fraction: float) -> set[str]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    top_n = max(1, math.ceil(len(rows) * fraction))
    ranked = sorted(rows, key=lambda row: (float(row[receptor_id]), row["ligand_id"]))
    return {
        row["ligand_id"] for row in ranked[:top_n] if row["label"] == "active"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    args = parser.parse_args()

    matrix_rows = read_csv(args.matrix)
    split_rows = read_csv(args.split_manifest)
    split_by_ligand = {row["ligand_id"]: row["split"] for row in split_rows}
    if {row["ligand_id"] for row in matrix_rows} != set(split_by_ligand):
        raise ValueError("matrix and split manifest ligand IDs do not match")

    result: dict[str, object] = {
        "fraction": args.fraction,
        "receptor_ids": args.receptor,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        rows = [row for row in matrix_rows if split_by_ligand[row["ligand_id"]] == split]
        active_total = {row["ligand_id"] for row in rows if row["label"] == "active"}
        coverage = {
            receptor_id: sorted(top_active_ids(rows, receptor_id, args.fraction))
            for receptor_id in args.receptor
        }
        pairwise = {}
        for first, second in itertools.combinations(args.receptor, 2):
            first_set = set(coverage[first])
            second_set = set(coverage[second])
            pairwise[f"{first}__{second}"] = {
                "union_active_ids": sorted(first_set | second_set),
                "union_count": len(first_set | second_set),
                "new_active_from_second": sorted(second_set - first_set),
                "new_active_from_first": sorted(first_set - second_set),
                "overlap_count": len(first_set & second_set),
            }
        union = set().union(*(set(ids) for ids in coverage.values()))
        result["splits"][split] = {
            "ligand_count": len(rows),
            "active_count": len(active_total),
            "active_top_ids_by_receptor": coverage,
            "active_union_ids": sorted(union),
            "active_union_count": len(union),
            "active_union_fraction": len(union) / len(active_total) if active_total else None,
            "pairwise": pairwise,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
