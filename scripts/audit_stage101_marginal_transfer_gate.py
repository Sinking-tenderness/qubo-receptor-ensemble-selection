"""Independent consistency audit for Stage101."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    result = json.loads((root / "data/stage101_marginal_transfer_gate_result.json").read_text(encoding="utf-8"))
    edges = read_csv(root / "results/runs/stage101_marginal_transfer_gate/marginal_edges.csv")
    policies = read_csv(root / "results/runs/stage101_marginal_transfer_gate/policy_target_summary.csv")
    folds = read_csv(root / "results/runs/stage101_marginal_transfer_gate/policy_fold_decisions.csv")
    if (len(edges), len(policies), len(folds)) != (50, 50, 250):
        raise ValueError("Stage101 row-count mismatch")
    targets = sorted({row["target_id"] for row in edges})
    for row in edges:
        training = set(row["loto_training_targets"].split("|"))
        if row["target_id"] in training or training != set(targets) - {row["target_id"]}:
            raise ValueError("leave-one-target-out leakage detected")
    k2 = [row for row in edges if int(row["to_k"]) == 2]
    rho = float(spearmanr([float(row["inner_mean_gain"]) for row in k2], [float(row["outer_gain"]) for row in k2]).statistic)
    recorded = float(result["correlations"]["k1_to_k2"]["spearman_r"])
    if not math.isclose(rho, recorded, abs_tol=1e-12):
        raise ValueError("k=2 Spearman mismatch")
    oracle_rows = [row for row in policies if row["policy"] == "outer_oracle_k"]
    oracle_mean = float(np.mean([float(row["gain_over_single"]) for row in oracle_rows]))
    if not math.isclose(oracle_mean, float(result["oracle_ceiling_mean_target_gain"]), abs_tol=1e-12):
        raise ValueError("oracle ceiling mismatch")
    if result["decision"]["hardware_authorized"] or result["decision"]["same_matrix_threshold_tuning_allowed"]:
        raise ValueError("Stage101 decision boundary was weakened")
    audit = {
        "schema_version": "1.0",
        "status": "stage101_independent_audit_ok",
        "edge_rows": len(edges),
        "policy_target_rows": len(policies),
        "policy_fold_rows": len(folds),
        "target_count": len(targets),
        "k1_to_k2_spearman_recomputed": rho,
        "oracle_ceiling_recomputed": oracle_mean,
        "held_target_leakage_detected": False,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
        "hardware_authorized": False,
    }
    output = root / "data/stage101_marginal_transfer_gate_audit.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
