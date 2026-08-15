"""Independent consistency audit for the Stage99 objective-repair screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    result = json.loads((root / "data/stage99_qubo_objective_repair_screen_result.json").read_text(encoding="utf-8"))
    fold_rows = read_csv(root / "results/runs/stage99_qubo_objective_repair_screen/fold_metrics.csv")
    target_rows = read_csv(root / "results/runs/stage99_qubo_objective_repair_screen/target_summary.csv")
    solver_rows = read_csv(root / "results/runs/stage99_qubo_objective_repair_screen/solver_diagnostics.csv")
    adaptive_rows = read_csv(root / "results/runs/stage99_qubo_objective_repair_screen/adaptive_k_metrics.csv")
    if result["status"] != "stage99_qubo_objective_repair_screen_complete":
        raise ValueError("unexpected Stage99 status")
    if result["data_boundary"]["historical_consumed_fresh_validation_rows_read_posthoc"] != 1576:
        raise ValueError("historical MK14 post-hoc row count mismatch")
    if result["data_boundary"]["protected_fresh_validation_rows_read"] != 0:
        raise ValueError("protected fresh-validation data were read")
    if result["target_ids"] != ["BACE1", "MK14", "PPARA", "PPARD", "PPARG"]:
        raise ValueError("target coverage mismatch")
    if (len(fold_rows), len(target_rows), len(solver_rows), len(adaptive_rows)) != (375, 75, 75, 25):
        raise ValueError("output row-count mismatch")
    if any(row["selector_used_test_labels"] != "False" for row in fold_rows):
        raise ValueError("outer-test labels leaked into selector")
    index = {
        (row["target_id"], row["fold"], int(row["ensemble_size"]), row["method"]): float(row["primary_bedroc_alpha20"])
        for row in fold_rows
    }
    fixed_gains = []
    for target in result["target_ids"]:
        repair = np.mean([index[(target, str(fold), 3, "repair_pair_qubo_exact")] for fold in range(1, 6)])
        single = np.mean([index[(target, str(fold), 1, "single_best")] for fold in range(1, 6)])
        fixed_gains.append(float(repair - single))
    fixed_gate = result["gate"]
    if not close(float(np.mean(fixed_gains)), float(fixed_gate["mean_gain_over_single"])):
        raise ValueError("fixed-k mean gain mismatch")
    if not close(float(np.min(fixed_gains)), float(fixed_gate["worst_target_gain"])):
        raise ValueError("fixed-k worst-target gain mismatch")
    adaptive_gains = []
    for target in result["target_ids"]:
        adaptive = np.mean([float(row["outer_exact_bedroc_alpha20"]) for row in adaptive_rows if row["target_id"] == target])
        single = np.mean([index[(target, str(fold), 1, "single_best")] for fold in range(1, 6)])
        adaptive_gains.append(float(adaptive - single))
    if not close(float(np.mean(adaptive_gains)), float(fixed_gate["adaptive_k"]["mean_gain_over_single"])):
        raise ValueError("adaptive-k mean gain mismatch")
    if not close(float(np.min(adaptive_gains)), float(fixed_gate["adaptive_k"]["worst_target_gain"])):
        raise ValueError("adaptive-k worst-target gain mismatch")
    differing = [row for row in solver_rows if row["exact_differs"] == "True"]
    if len(differing) != int(fixed_gate["solver_exact_differs_from_one_swap_count"]):
        raise ValueError("solver-difference count mismatch")
    if fixed_gate["passes"] is not False or fixed_gate["fixed_k3_passes"] is not False:
        raise ValueError("conservative NO-GO changed")
    audit = {
        "schema_version": "1.0",
        "status": "stage99_independent_audit_ok",
        "target_count": 5,
        "fold_rows": len(fold_rows),
        "target_rows": len(target_rows),
        "solver_rows": len(solver_rows),
        "adaptive_k_rows": len(adaptive_rows),
        "fixed_k3_mean_gain_recomputed": float(np.mean(fixed_gains)),
        "adaptive_k_mean_gain_recomputed": float(np.mean(adaptive_gains)),
        "exact_differs_from_one_swap_count": len(differing),
        "test_label_leakage_detected": False,
        "historical_consumed_fresh_validation_rows_read_posthoc": 1576,
        "protected_fresh_validation_rows_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
        "gate_passes": False,
    }
    output = root / "data/stage99_qubo_objective_repair_screen_audit.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
