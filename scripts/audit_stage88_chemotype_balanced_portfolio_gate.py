"""Independently audit Stage88 chemotype-balanced gate outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    outputs = config["outputs"]
    result = read_json(root / outputs["result_json"])
    assignments = read_csv(root / outputs["cluster_assignments_csv"])
    folds = read_csv(root / outputs["fold_metrics_csv"])
    target_cluster_counts = Counter((row["target_id"], row["chemotype_id"]) for row in assignments)
    gains = [
        float(row["exact_worst_group_gain_vs_overall_top3"])
        for row in folds
        if row["exact_worst_group_gain_vs_overall_top3"]
    ]
    checks = {
        "four_targets": len({row["target_id"] for row in assignments}) == 4,
        "four_clusters_per_target": all(
            len({row["chemotype_id"] for row in assignments if row["target_id"] == target}) == 4
            for target in {row["target_id"] for row in assignments}
        ),
        "assignment_counts_match": min(target_cluster_counts.values())
        == int(result["summary"]["minimum_global_cluster_count"]),
        "fold_count": len(folds) == 16,
        "gain_count_matches": sum(value > 1e-12 for value in gains)
        == int(result["summary"]["positive_worst_group_holdout_gain_fold_count"]),
        "no_qaoa": result["constraint_preserving_qaoa_simulation_authorized"] is False,
        "no_hardware": result["new_quantum_hardware_jobs_authorized"] == 0,
        "no_docking": result["new_docking_jobs_authorized"] == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage88 independent audit failed: {checks}")
    audit = {
        "schema_version": "1.0",
        "status": "stage88_chemotype_balanced_portfolio_gate_independent_audit_ok",
        "checks": checks,
        "gate_passed": bool(result["chemotype_balanced_cqm_design_authorized"]),
        "qaoa_authorized": False,
        "quantum_hardware_jobs_authorized": 0,
    }
    path = root / outputs["audit_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage88_chemotype_balanced_portfolio_gate.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    audit = run((root / args.config).resolve(), root)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
