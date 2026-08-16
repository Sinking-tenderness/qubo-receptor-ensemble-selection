"""Independently audit Stage81 condition aggregation and no-go decision."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()




def truth(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean: {value}")


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    result = read_json(root / config["outputs"]["result_json"])
    direct_path = root / result["outputs"]["direct_metrics_csv"]["path"]
    prefilter_path = root / result["outputs"]["prefilter_metrics_csv"]["path"]
    if sha256(direct_path) != result["outputs"]["direct_metrics_csv"]["sha256"]:
        raise ValueError("Stage81 direct metrics identity differs")
    if sha256(prefilter_path) != result["outputs"]["prefilter_metrics_csv"]["sha256"]:
        raise ValueError("Stage81 prefilter metrics identity differs")
    with direct_path.open("r", encoding="utf-8", newline="") as handle:
        direct = list(csv.DictReader(handle))
    with prefilter_path.open("r", encoding="utf-8", newline="") as handle:
        prefilter = list(csv.DictReader(handle))

    grouped: dict[tuple[float, int], list[dict[str, str]]] = defaultdict(list)
    for row in direct:
        grouped[(float(row["penalty_factor"]), int(row["signed_precision_bits"]))].append(row)
    summaries = {
        (float(row["penalty_factor"]), int(row["signed_precision_bits"])): row
        for row in result["direct_penalty_summary"]
    }
    direct_checks = []
    for key, rows in grouped.items():
        summary = summaries[key]
        count = len(rows)
        direct_checks.append(
            count == int(summary["cell_count"])
            and abs(
                sum(truth(row["cold_raw_best_feasible"]) for row in rows) / count
                - float(summary["cold_raw_best_feasible_fraction"])
            )
            <= 1e-12
            and abs(
                sum(truth(row["cold_frontier_competitive"]) for row in rows) / count
                - float(summary["cold_frontier_competitive_fraction"])
            )
            <= 1e-12
        )
    prefilter_summary = result["conservative_prefilter_summary"]
    prefilter_checks = {
        "cell_count": len(prefilter) == int(prefilter_summary["cell_count"]),
        "pool_count": sum(truth(row["pool_can_select_k"]) for row in prefilter)
        == int(prefilter_summary["pool_can_select_k_count"]),
        "reference_count": sum(
            truth(row["reference_fully_retained"]) for row in prefilter
        )
        == int(prefilter_summary["reference_fully_retained_count"]),
        "mean_retention": abs(
            sum(float(row["reference_retained_fraction"]) for row in prefilter)
            / len(prefilter)
            - float(prefilter_summary["mean_reference_retained_fraction"])
        )
        <= 1e-12,
    }
    direct_pass = any(bool(row["condition_passed"]) for row in result["direct_penalty_summary"])
    prefilter_pass = (
        int(prefilter_summary["pool_can_select_k_count"]) == len(prefilter)
        and int(prefilter_summary["reference_fully_retained_count"])
        / len(prefilter)
        >= float(config["decision_gate"]["minimum_prefilter_reference_retention_fraction"])
    )
    checks = {
        "direct_aggregation": all(direct_checks),
        "prefilter_aggregation": all(prefilter_checks.values()),
        "decision_consistent": bool(
            result["decision"]["additional_qci_hardware_submission_authorized"]
        )
        == (direct_pass or prefilter_pass),
        "no_cloud_or_hardware": result["data_boundary"]["qci_cloud_queries"] == 0
        and result["data_boundary"]["quantum_hardware_jobs"] == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage81 audit failed: {checks}")
    audit = {
        "schema_version": "1.0",
        "status": "stage81_dirac_global_qubo_formulation_independent_audit_ok",
        "direct_rows_audited": len(direct),
        "prefilter_rows_audited": len(prefilter),
        "checks": checks,
        "additional_qci_hardware_submission_authorized": direct_pass or prefilter_pass,
        "qci_cloud_queries_observed": 0,
        "quantum_hardware_jobs_observed": 0,
    }
    output = root / config["outputs"]["audit_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage81_dirac_global_qubo_formulation_gate.json"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    audit = run((root / args.config).resolve(), root)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
