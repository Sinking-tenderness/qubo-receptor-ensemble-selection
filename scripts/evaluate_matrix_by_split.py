"""Evaluate each receptor column separately on fixed ligand splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .analyze_receptor_complementarity import receptor_data
    from .compare_receptor_screening import ranked_metrics_with_ids
except ImportError:
    from analyze_receptor_complementarity import receptor_data
    from compare_receptor_screening import ranked_metrics_with_ids


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix_rows = read_csv(args.matrix)
    split_rows = read_csv(args.split_manifest)
    split_by_ligand = {row["ligand_id"]: row["split"] for row in split_rows}
    matrix_ids = {row["ligand_id"] for row in matrix_rows}
    if matrix_ids != set(split_by_ligand):
        raise ValueError("matrix and split manifest ligand IDs do not match")
    for row in matrix_rows:
        for receptor_id in args.receptor:
            if row.get(receptor_id, "") == "":
                raise ValueError(f"missing score: {row['ligand_id']} / {receptor_id}")

    summary: dict[str, object] = {
        "matrix": str(args.matrix),
        "split_manifest": str(args.split_manifest),
        "receptor_ids": args.receptor,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        rows = [row for row in matrix_rows if split_by_ligand[row["ligand_id"]] == split]
        split_summary: dict[str, object] = {
            "ligand_count": len(rows),
            "active_count": sum(row["label"] == "active" for row in rows),
            "metrics": {},
        }
        for receptor_id in args.receptor:
            split_summary["metrics"][receptor_id] = ranked_metrics_with_ids(
                receptor_data(rows, receptor_id)
            )
        summary["splits"][split] = split_summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
