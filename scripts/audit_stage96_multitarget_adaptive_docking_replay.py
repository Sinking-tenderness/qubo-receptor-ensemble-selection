from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    result = json.loads((root / "data/stage96_multitarget_adaptive_docking_replay_result.json").read_text(encoding="utf-8"))
    trajectories = read_csv(root / "results/runs/stage96_multitarget_adaptive_docking_replay/trajectories.csv")
    checkpoints = read_csv(root / "results/runs/stage96_multitarget_adaptive_docking_replay/checkpoints.csv")
    solver = read_csv(root / "results/runs/stage96_multitarget_adaptive_docking_replay/qubo_solver_comparisons.csv")
    required_targets = {"PPARG", "BACE1"}
    required_policies = {"random", "predicted_mean", "predictive_uncertainty", "qubo_direct_greedy", "qubo_greedy_one_swap", "qubo_exact_milp"}
    if result["status"] != "stage96_replay_complete":
        raise ValueError("unexpected result status")
    if result["audit"] != {"labels_used_by_selector": False, "docking_scores_revealed_only_after_task_selection": True, "synthetic_scores": 0, "new_docking_jobs": 0, "fresh_validation_rows": 0}:
        raise ValueError("data-boundary audit mismatch")
    if {row["target_id"] for row in checkpoints} != required_targets:
        raise ValueError("checkpoint target coverage mismatch")
    if {row["policy"] for row in checkpoints} != required_policies:
        raise ValueError("checkpoint policy coverage mismatch")
    if {float(row["checkpoint_fraction"]) for row in checkpoints} != {0.1, 0.2, 0.3}:
        raise ValueError("checkpoint fraction mismatch")
    if len(trajectories) != 774 or len(checkpoints) != 108 or len(solver) != 387:
        raise ValueError("unexpected output row count")
    if max(abs(float(row["exact_minus_one_swap"])) for row in solver) >= 1e-9:
        raise ValueError("exact and one-swap are not equivalent within audit tolerance")
    if result["policy_gate"]["passes"] or result["solver_value"]["passes"]:
        raise ValueError("conservative negative gate changed unexpectedly")
    audit = {"schema_version": "1.0", "status": "stage96_audit_ok", "targets": sorted(required_targets), "trajectory_rows": len(trajectories), "checkpoint_rows": len(checkpoints), "solver_rows": len(solver), "labels_used_by_selector": False, "policy_gate_passes": False, "solver_value_passes": False}
    output = root / "data/stage96_multitarget_adaptive_docking_replay_audit.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
