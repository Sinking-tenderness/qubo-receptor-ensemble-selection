"""Build the audited Stage75 explicit variable-k CQM core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage74_larger_k_solver_scaling_bundle import (
        PATHS as STAGE74_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage74_larger_k_solver_scaling_bundle import (
        PATHS as STAGE74_PATHS,
    )


PATHS = tuple(
    sorted(
        set(
            STAGE74_PATHS
            + (
                "configs/stage75_explicit_variable_k_cqm.json",
                "scripts/run_stage75_explicit_variable_k_cqm.py",
                "scripts/audit_stage75_explicit_variable_k_cqm.py",
                "scripts/build_stage75_explicit_variable_k_cqm_bundle.py",
                "tests/test_stage75_explicit_variable_k_cqm.py",
                "results/runs/stage75_explicit_variable_k_cqm/cqm_metrics.csv",
                "results/runs/stage75_explicit_variable_k_cqm/solver_trials.csv",
                "results/runs/stage75_explicit_variable_k_cqm/cell_comparison.csv",
                "results/runs/stage75_explicit_variable_k_cqm/solver_summary.csv",
                "data/stage75_explicit_variable_k_cqm_model_record.json",
                "data/stage75_explicit_variable_k_cqm_result.json",
                "data/stage75_explicit_variable_k_cqm_audit.json",
                "reports/stage-75/explicit_variable_k_cqm.md",
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
        (root / "data/stage75_explicit_variable_k_cqm_result.json").read_text(
            encoding="ascii"
        )
    )
    result = write_bundle(root, args.output, list(PATHS))
    encoding = source["encoding_summary"]
    performance = source["solver_performance"]
    result.update(
        {
            "operation": "audited Stage75 explicit variable-k constraint-native CQM benchmark",
            "target_count": 4,
            "source_model_count": 16,
            "cqm_model_count": encoding["cqm_model_count"],
            "solver_trial_count": 1440,
            "exact_frontier_cell_count": performance["exact_frontier_cell_count"],
            "maximum_logical_variable_count": encoding[
                "maximum_logical_variable_count"
            ],
            "maximum_quadratic_coupler_count": encoding[
                "maximum_quadratic_coupler_count"
            ],
            "distinct_frontier_selected_k": encoding[
                "distinct_frontier_selected_k"
            ],
            "joint_classical_exact_frontier_match_rate": performance[
                "joint_classical_exact_frontier_match_rate"
            ],
            "sampler_exact_frontier_match_rate": performance[
                "sampler_exact_frontier_match_rate"
            ],
            "sampler_joint_classical_competitive_fraction": performance[
                "sampler_joint_classical_competitive_fraction"
            ],
            "explicit_variable_k_cqm_freeze_authorized": source["decision"][
                "explicit_variable_k_cqm_freeze_authorized"
            ],
            "local_hardware_shaped_emulation_authorized": source["decision"][
                "local_hardware_shaped_emulation_authorized"
            ],
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
