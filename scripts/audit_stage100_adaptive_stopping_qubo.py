"""Independent audit for Stage100 adaptive receptor-count stopping."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    result = json.loads((root / "data/stage100_adaptive_stopping_qubo_result.json").read_text(encoding="utf-8"))
    folds = read_csv(root / "results/runs/stage100_adaptive_stopping_qubo/fold_metrics.csv")
    inner = read_csv(root / "results/runs/stage100_adaptive_stopping_qubo/inner_k_profiles.csv")
    targets = read_csv(root / "results/runs/stage100_adaptive_stopping_qubo/target_summary.csv")
    if result["status"] != "stage100_adaptive_stopping_qubo_complete":
        raise ValueError("unexpected Stage100 status")
    if (len(folds), len(inner), len(targets)) != (125, 225, 25):
        raise ValueError("Stage100 row-count mismatch")
    if any(row["selector_used_outer_test_labels"] != "False" for row in folds):
        raise ValueError("outer-test label leakage detected")
    primary = [row for row in folds if row["method"] == "one_standard_error_smallest_k"]
    if any(int(row["selected_k"]) not in {1, 2, 3} for row in primary):
        raise ValueError("invalid adaptive k")
    nontrivial = sum(int(row["selected_k"]) > 1 for row in primary)
    if nontrivial != result["gate"]["nontrivial_selected_fold_count"]:
        raise ValueError("nontrivial selection count mismatch")
    gains = []
    for target in result["target_ids"]:
        adaptive = np.mean([float(row["outer_bedroc_alpha20"]) for row in primary if row["target_id"] == target])
        single = np.mean([float(row["outer_bedroc_alpha20"]) for row in folds if row["target_id"] == target and row["method"] == "single"])
        gains.append(float(adaptive - single))
    if not math.isclose(float(np.mean(gains)), result["gate"]["mean_gain_over_single"], abs_tol=1e-12):
        raise ValueError("mean gain mismatch")
    if not math.isclose(float(np.min(gains)), result["gate"]["worst_target_gain"], abs_tol=1e-12):
        raise ValueError("worst-target gain mismatch")
    if result["gate"]["passes"] is not False:
        raise ValueError("expected conservative NO-GO changed")
    audit = {
        "schema_version": "1.0",
        "status": "stage100_independent_audit_ok",
        "fold_rows": len(folds),
        "inner_profile_rows": len(inner),
        "target_rows": len(targets),
        "primary_nontrivial_fold_count": nontrivial,
        "mean_gain_recomputed": float(np.mean(gains)),
        "worst_target_gain_recomputed": float(np.min(gains)),
        "outer_test_label_leakage_detected": False,
        "protected_fresh_validation_rows_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
        "gate_passes": False,
    }
    output = root / "data/stage100_adaptive_stopping_qubo_audit.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
