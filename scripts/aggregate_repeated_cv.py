"""Aggregate repeated out-of-fold score predictions across CV seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .cross_validate_ensemble_mvp import paired_bootstrap_delta
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from cross_validate_ensemble_mvp import paired_bootstrap_delta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    methods = ("single", "qubo", "all_mean")
    aggregate: dict[str, dict[str, dict[str, object]]] = {}
    for method in methods:
        by_ligand: dict[str, list[dict[str, object]]] = defaultdict(list)
        for document in documents:
            for ligand_id, record in document["oof_scores"][method].items():
                by_ligand[ligand_id].append(record)
        aggregate[method] = {
            ligand_id: {
                "label": records[0]["label"],
                "score": sum(float(record["score"]) for record in records) / len(records),
                "replicate_count": len(records),
            }
            for ligand_id, records in by_ligand.items()
        }
    result = {
        "input_files": [str(path) for path in args.input],
        "replicate_count": len(documents),
        "aggregate_metrics": {
            method: ranked_metrics_with_ids(records) for method, records in aggregate.items()
        },
        "paired_bootstrap_qubo_minus_single": paired_bootstrap_delta(
            aggregate["single"], aggregate["qubo"], 5000, 20260801
        ),
        "oof_scores": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result["aggregate_metrics"], indent=2, ensure_ascii=True))
    print(json.dumps(result["paired_bootstrap_qubo_minus_single"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
