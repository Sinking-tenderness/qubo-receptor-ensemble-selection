"""Build ligand-by-receptor docking score matrices from long docking tables.

Thin CLI wrapper; the core logic lives in ``qubo_receptor_ensemble.matrix``.
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

from qubo_receptor_ensemble.matrix import (
    REQUIRED_COLUMNS,
    build_summary,
    build_wide_matrix,
    parse_pose_rank,
    read_score_tables,
    select_representative_scores,
    validate_columns,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "validate_columns",
    "read_score_tables",
    "parse_pose_rank",
    "select_representative_scores",
    "build_wide_matrix",
    "write_csv",
    "build_summary",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table", type=Path, nargs="+", required=True)
    parser.add_argument("--long-output", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--representative",
        choices=["pose_rank_1", "min_score"],
        default="pose_rank_1",
        help="How to choose one score per ligand-receptor pair.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_rows = read_score_tables(args.score_table)
    long_rows = select_representative_scores(raw_rows, args.representative)
    matrix_rows = build_wide_matrix(long_rows)
    summary = build_summary(long_rows, matrix_rows)

    write_csv(args.long_output, long_rows)
    write_csv(args.matrix_output, matrix_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"long_output={args.long_output}")
    print(f"matrix_output={args.matrix_output}")
    print(f"summary_output={args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
