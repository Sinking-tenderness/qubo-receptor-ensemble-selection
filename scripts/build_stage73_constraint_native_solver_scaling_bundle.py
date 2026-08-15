"""Build the audited Stage73 constraint-native solver-scaling core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage72_constraint_native_cqm_bundle import (
        PATHS as STAGE72_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage72_constraint_native_cqm_bundle import PATHS as STAGE72_PATHS


PATHS = tuple(
    sorted(
        set(
            STAGE72_PATHS
            + (
                "configs/stage73_constraint_native_solver_scaling.json",
                "scripts/run_stage73_constraint_native_solver_scaling.py",
                "scripts/audit_stage73_constraint_native_solver_scaling.py",
                "scripts/build_stage73_constraint_native_solver_scaling_bundle.py",
                "tests/test_stage73_constraint_native_solver_scaling.py",
                "results/runs/stage73_constraint_native_solver_scaling/workload_metrics.csv",
                "results/runs/stage73_constraint_native_solver_scaling/solver_trials.csv",
                "results/runs/stage73_constraint_native_solver_scaling/solver_summary.csv",
                "data/stage73_constraint_native_solver_scaling_result.json",
                "data/stage73_constraint_native_solver_scaling_audit.json",
                "reports/stage-73/constraint_native_solver_scaling.md",
            )
        )
    )
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = json.loads(
        (
            root / "data/stage73_constraint_native_solver_scaling_result.json"
        ).read_text(encoding="ascii")
    )
    result = write_bundle(root, args.output, list(PATHS))
    scale = source["scaling_summary"]
    result.update(
        {
            "operation": "audited Stage73 constraint-native classical solver-scaling benchmark",
            "target_count": 4,
            "logical_model_count": scale["model_count"],
            "workload_cell_count": scale["workload_cell_count"],
            "solver_trial_count": scale["solver_trial_count"],
            "maximum_pool_size": scale["maximum_pool_size"],
            "maximum_total_fixed_k_subset_count": scale[
                "maximum_total_fixed_k_subset_count"
            ],
            "maximum_full_pool_frozen_feasible_subset_count": scale[
                "maximum_full_pool_frozen_feasible_subset_count"
            ],
            "current_k3_exact_enumeration_tractable": source["route_gate"][
                "current_k3_exact_enumeration_tractable"
            ],
            "constraint_native_solver_gate_passed": source["route_gate"][
                "constraint_native_solver_gate_passed"
            ],
            "larger_k_scaling_study_authorized": source["decision"][
                "larger_k_scaling_study_authorized"
            ],
            "direct_qpu_execution_authorized": source["decision"][
                "direct_qpu_execution_authorized"
            ],
            "quantum_scaling_claim_authorized": source["decision"][
                "quantum_scaling_claim_authorized"
            ],
            "quantum_advantage_claim_authorized": source["decision"][
                "quantum_advantage_claim_authorized"
            ],
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "cloud_cqm_jobs": 0,
            "quantum_hardware_jobs": 0,
        }
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
