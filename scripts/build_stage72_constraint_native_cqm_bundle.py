"""Build the audited Stage72 constraint-native CQM core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage71_qubo_coefficient_noise_robustness_bundle import (
        PATHS as STAGE71_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage71_qubo_coefficient_noise_robustness_bundle import (
        PATHS as STAGE71_PATHS,
    )


PATHS = tuple(
    sorted(
        set(
            STAGE71_PATHS
            + (
                "configs/stage72_constraint_native_cqm.json",
                "scripts/run_stage72_constraint_native_cqm.py",
                "scripts/audit_stage72_constraint_native_cqm.py",
                "scripts/build_stage72_constraint_native_cqm_bundle.py",
                "tests/test_stage72_constraint_native_cqm.py",
                "results/runs/stage72_constraint_native_cqm/model_metrics.csv",
                "results/runs/stage72_constraint_native_cqm/noise_trials.csv",
                "results/runs/stage72_constraint_native_cqm/noise_summary.csv",
                "data/stage72_constraint_native_cqm_model_record.json",
                "data/stage72_constraint_native_cqm_result.json",
                "data/stage72_constraint_native_cqm_audit.json",
                "reports/stage-72/constraint_native_cqm.md",
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
        (root / "data/stage72_constraint_native_cqm_result.json").read_text(
            encoding="ascii"
        )
    )
    result = write_bundle(root, args.output, list(PATHS))
    summary = source["formulation_summary"]
    result.update(
        {
            "operation": "audited Stage72 constraint-native CQM precision-rescue experiment",
            "target_count": 4,
            "logical_model_count": summary["model_count"],
            "noise_trial_count": source["noise_trial_count"],
            "noise_summary_count": source["noise_summary_count"],
            "minimum_normalized_gap_improvement_factor_vs_stage71": summary[
                "minimum_normalized_gap_improvement_factor_vs_stage71"
            ],
            "maximum_normalized_gap_improvement_factor_vs_stage71": summary[
                "maximum_normalized_gap_improvement_factor_vs_stage71"
            ],
            "constraint_native_formulation_gate_passed": source[
                "formulation_gate"
            ]["constraint_native_formulation_gate_passed"],
            "constraint_native_formulation_freeze_authorized": source["decision"][
                "constraint_native_formulation_freeze_authorized"
            ],
            "solver_scaling_benchmark_authorized": source["decision"][
                "solver_scaling_benchmark_authorized"
            ],
            "direct_qpu_execution_authorized": source["decision"][
                "direct_qpu_execution_authorized"
            ],
            "new_target_preregistration_remains_authorized": source["decision"][
                "new_target_preregistration_remains_authorized"
            ],
            "quantum_advantage_claim_authorized": source["decision"][
                "quantum_advantage_claim_authorized"
            ],
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
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
