"""Build the audited Stage71 coefficient-noise robustness core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage70_constraint_aware_qubo_encoding_bundle import (
        PATHS as STAGE70_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage70_constraint_aware_qubo_encoding_bundle import (
        PATHS as STAGE70_PATHS,
    )


PATHS = tuple(
    sorted(
        set(
            STAGE70_PATHS
            + (
                "configs/stage71_qubo_coefficient_noise_robustness.json",
                "scripts/run_stage71_qubo_coefficient_noise_robustness.py",
                "scripts/audit_stage71_qubo_coefficient_noise_robustness.py",
                "scripts/build_stage71_qubo_coefficient_noise_robustness_bundle.py",
                "tests/test_stage71_qubo_coefficient_noise_robustness.py",
                "results/runs/stage71_qubo_coefficient_noise_robustness/exact_feasible_landscape.csv",
                "results/runs/stage71_qubo_coefficient_noise_robustness/zero_noise_sampler_calibration.csv",
                "results/runs/stage71_qubo_coefficient_noise_robustness/noise_trials.csv",
                "results/runs/stage71_qubo_coefficient_noise_robustness/noise_summary.csv",
                "data/stage71_qubo_coefficient_noise_robustness_result.json",
                "data/stage71_qubo_coefficient_noise_robustness_audit.json",
                "reports/stage-71/qubo_coefficient_noise_robustness.md",
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
            root / "data/stage71_qubo_coefficient_noise_robustness_result.json"
        ).read_text(encoding="ascii")
    )
    result = write_bundle(root, args.output, list(PATHS))
    result.update(
        {
            "operation": "audited Stage71 logical-QUBO coefficient-noise robustness diagnosis",
            "target_count": 4,
            "logical_model_count": source["exact_landscape"]["model_count"],
            "noise_trial_count": source["noise_trial_count"],
            "noise_summary_count": source["noise_summary_count"],
            "minimum_normalized_feasible_energy_gap": source["exact_landscape"][
                "minimum_normalized_feasible_energy_gap"
            ],
            "zero_noise_calibration_gate_passed": source["sampler_calibration"][
                "calibration_gate_passed"
            ],
            "coefficient_robust_logical_bqm_gate_passed": source[
                "robustness_gate"
            ]["coefficient_robust_logical_bqm_gate_passed"],
            "direct_qpu_execution_authorized": source["decision"][
                "direct_qpu_execution_authorized"
            ],
            "constraint_native_reformulation_authorized": source["decision"][
                "constraint_native_reformulation_authorized"
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
