"""Audit a ligand manifest with RDKit before 3D ligand preparation.

Thin CLI wrapper; the core logic lives in ``qubo_receptor_ensemble.ligand``.
"""



from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import argparse
import csv
import json
from pathlib import Path

from qubo_receptor_ensemble.ligand import (
    REQUIRED_COLUMNS,
    audit_row,
    build_summary,
    read_rows,
    validate_columns,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "validate_columns",
    "audit_row",
    "read_rows",
    "write_csv",
    "build_summary",
]


def write_csv(output_csv: Path, rows: list[dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input ligand manifest CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output audited CSV")
    parser.add_argument("--summary", type=Path, required=True, help="Output JSON summary")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_rows, _ = read_rows(args.input)
    audited_rows = [audit_row(row) for row in input_rows]
    write_csv(args.output, audited_rows)
    summary = build_summary(audited_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"output={args.output}")
    print(f"summary={args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
