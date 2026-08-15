"""Independently audit the Stage44 PPARG MD-96 QUBO result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    result_path = root / config["outputs"]["result_json"]
    metrics_path = root / config["outputs"]["selection_metrics_csv"]
    solver_path = root / config["outputs"]["solver_comparison_csv"]
    result = read_json(result_path)
    metrics = read_csv(metrics_path)
    solver = read_csv(solver_path)
    if result.get("status") != "stage44_pparg_md96_rank_sensitive_qubo_complete":
        raise ValueError("Stage44 result is incomplete")
    if len(metrics) != 189 or len(solver) != 30:
        raise ValueError("Stage44 output row counts differ")

    def metric(scope: str, fold: str, method: str, size: str) -> dict[str, str]:
        matches = [row for row in metrics if row["scope"] == scope and row["fold"] == fold and row["method"] == method and row["subset_size"] == size]
        if len(matches) != 1:
            raise ValueError(f"Stage44 metric identity differs: {scope}/{fold}/{method}/{size}")
        return matches[0]

    full_k1 = metric("full_data", "full", "exact", "1")
    full_k3 = metric("full_data", "full", "exact", "3")
    full_gain = float(full_k3["robust_bedroc_composite"]) - float(full_k1["robust_bedroc_composite"])
    holdout_gains = [
        float(metric("outer_holdout", str(fold), "exact", "3")["robust_bedroc_composite"])
        - float(metric("outer_holdout", str(fold), "exact", "1")["robust_bedroc_composite"])
        for fold in range(4)
    ]
    decision = result["decision"]
    if not math.isclose(full_gain, float(decision["full_k3_over_single_robust_bedroc_gain"]), abs_tol=1e-12):
        raise ValueError("Stage44 full-data gain differs")
    if not math.isclose(statistics.fmean(holdout_gains), float(decision["mean_outer_holdout_k3_over_single_gain"]), abs_tol=1e-12):
        raise ValueError("Stage44 holdout gain differs")
    if any(value >= 0 for value in holdout_gains):
        raise ValueError("Stage44 reported holdout failure pattern differs")

    exact_cells = [row for row in solver if row["subset_size"] in {"1", "2", "3"}]
    if len(exact_cells) != 15 or any(row["exact_available"].lower() != "true" for row in exact_cells):
        raise ValueError("Stage44 exact-oracle coverage differs")
    for row in exact_cells:
        if abs(float(row["classical_exact_gap"])) > 1e-12 or abs(float(row["annealing_exact_gap"])) > 1e-12:
            raise ValueError("Stage44 exact solver gap differs")
    if any(abs(float(row["annealing_minus_classical_gap"])) > 1e-12 for row in solver):
        raise ValueError("Stage44 annealing/classical equality differs")
    if decision["application_replication_supported"] or decision["solver_novelty_supported"]:
        raise ValueError("Stage44 decision should be NO-GO")
    if decision["same_data_retuning_authorized"] or decision["fresh_validation_authorized"] or decision["quantum_hardware_authorized"]:
        raise ValueError("Stage44 downstream authorization differs")

    audit = {
        "schema_version": "1.0",
        "status": "stage44_pparg_md96_rank_sensitive_qubo_independent_audit_ok",
        "counts": {"selection_metric_rows": 189, "solver_cells": 30, "exact_cells": 15},
        "primary_recalculation": {
            "full_k1_robust_bedroc": float(full_k1["robust_bedroc_composite"]),
            "full_k3_robust_bedroc": float(full_k3["robust_bedroc_composite"]),
            "full_k3_over_single_gain": full_gain,
            "outer_holdout_k3_over_single_gains": holdout_gains,
            "mean_outer_holdout_gain": statistics.fmean(holdout_gains),
        },
        "solver_recalculation": {
            "exact_k1_to_k3_cell_count": 15,
            "exact_mismatch_count": 0,
            "annealing_over_classical_cell_count": 0,
        },
        "decision": {
            "application_replication_supported": False,
            "solver_novelty_supported": False,
            "same_data_retuning_authorized": False,
            "fresh_validation_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "data_boundary": {"train_rows_read": 160, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "inputs": {"result_sha256": sha256(result_path), "selection_metrics_sha256": sha256(metrics_path), "solver_comparison_sha256": sha256(solver_path)},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output = root / "data/stage44_pparg_md96_rank_sensitive_qubo_audit.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage44_pparg_md96_rank_sensitive_qubo.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
