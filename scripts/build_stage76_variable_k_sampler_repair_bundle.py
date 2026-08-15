"""Build the audited Stage76 variable-k sampler-repair core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage75_explicit_variable_k_cqm_bundle import (
        PATHS as STAGE75_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage75_explicit_variable_k_cqm_bundle import (
        PATHS as STAGE75_PATHS,
    )


PATHS = tuple(
    sorted(
        set(
            STAGE75_PATHS
            + (
                "configs/stage76_variable_k_sampler_repair.json",
                "scripts/run_stage76_variable_k_sampler_repair.py",
                "scripts/audit_stage76_variable_k_sampler_repair.py",
                "scripts/build_stage76_variable_k_sampler_repair_bundle.py",
                "tests/test_stage76_variable_k_sampler_repair.py",
                "results/runs/stage76_variable_k_sampler_repair/cqm_identity.csv",
                "results/runs/stage76_variable_k_sampler_repair/solver_trials.csv",
                "results/runs/stage76_variable_k_sampler_repair/cell_method_comparison.csv",
                "results/runs/stage76_variable_k_sampler_repair/method_summary.csv",
                "data/stage76_variable_k_sampler_repair_result.json",
                "data/stage76_variable_k_sampler_repair_audit.json",
                "reports/stage-76/variable_k_sampler_repair.md",
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
        (root / "data/stage76_variable_k_sampler_repair_result.json").read_text(
            encoding="ascii"
        )
    )
    summaries = {row["method"]: row for row in source["method_summaries"]}
    final = summaries["frontier_warm_parallel_tempering"]
    result = write_bundle(root, args.output, list(PATHS))
    result.update(
        {
            "operation": "audited Stage76 variable-k sampler-mechanism repair benchmark",
            "target_count": 4,
            "cqm_model_count": source["encoding_summary"]["cqm_model_count"],
            "objective_or_constraint_changes": source["encoding_summary"][
                "objective_or_constraint_changes"
            ],
            "method_count": source["benchmark_summary"]["method_count"],
            "solver_trial_count": source["benchmark_summary"]["solver_trial_count"],
            "cell_method_count": source["benchmark_summary"]["method_cell_count"],
            "warm_pt_exact_frontier_match_rate": final[
                "exact_frontier_match_rate"
            ],
            "warm_pt_joint_competitive_fraction": final[
                "joint_competitive_fraction"
            ],
            "warm_pt_frontier_competitive_fraction": final[
                "frontier_competitive_fraction"
            ],
            "warm_pt_frontier_improvement_cell_count": final[
                "strict_frontier_improvement_cell_count"
            ],
            "standalone_cold_sampler_ready": source["decision"][
                "standalone_cold_sampler_ready"
            ],
            "local_warm_start_hardware_shaped_emulation_authorized": source[
                "decision"
            ]["local_warm_start_hardware_shaped_emulation_authorized"],
            "cloud_cqm_execution_authorized": False,
            "direct_qpu_execution_authorized": False,
            "quantum_scaling_claim_authorized": False,
            "quantum_advantage_claim_authorized": False,
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
