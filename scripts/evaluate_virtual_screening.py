"""Evaluate virtual-screening ranking metrics from a docking score table.

Thin CLI wrapper; the core logic lives in ``qubo_receptor_ensemble.screening``.
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

from qubo_receptor_ensemble.screening import (
    REQUIRED_COLUMNS,
    average_precision,
    bedroc,
    bootstrap_confidence_intervals,
    build_metrics,
    enrichment_factor,
    percentile,
    read_rows,
    roc_auc_pairwise,
    scalar_metrics,
    select_best_pose,
    validate_columns,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "validate_columns",
    "read_rows",
    "select_best_pose",
    "roc_auc_pairwise",
    "average_precision",
    "bedroc",
    "enrichment_factor",
    "scalar_metrics",
    "percentile",
    "bootstrap_confidence_intervals",
    "build_metrics",
    "write_csv",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows as CSV; unlike ``qubo_receptor_ensemble.io.write_csv`` this
    variant accepts an empty row list (header-only output)."""
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
    parser.add_argument("--score-table", type=Path, required=True)
    parser.add_argument("--ranking-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument(
        "--top-fractions",
        nargs="+",
        type=float,
        default=[0.01, 0.05],
        help="Fractions for enrichment factor, e.g. 0.01 0.05.",
    )
    parser.add_argument("--bedroc-alpha", type=float, default=20.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows(args.score_table)
    ranked = select_best_pose(rows)
    metrics = build_metrics(
        ranked=ranked,
        top_fractions=args.top_fractions,
        bedroc_alpha=args.bedroc_alpha,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_csv(args.ranking_output, ranked)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"ranking_output={args.ranking_output}")
    print(f"metrics_output={args.metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
