"""Build the audited Stage74 larger-k solver-scaling core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage73_constraint_native_solver_scaling_bundle import (
        PATHS as STAGE73_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage73_constraint_native_solver_scaling_bundle import (
        PATHS as STAGE73_PATHS,
    )


PATHS = tuple(
    sorted(
        set(
            STAGE73_PATHS
            + (
                "configs/stage74_larger_k_solver_scaling.json",
                "scripts/run_stage74_larger_k_solver_scaling.py",
                "scripts/audit_stage74_larger_k_solver_scaling.py",
                "scripts/build_stage74_larger_k_solver_scaling_bundle.py",
                "tests/test_stage74_larger_k_solver_scaling.py",
                "results/runs/stage74_larger_k_solver_scaling/workload_metrics.csv",
                "results/runs/stage74_larger_k_solver_scaling/solver_trials.csv",
                "results/runs/stage74_larger_k_solver_scaling/cell_comparison.csv",
                "results/runs/stage74_larger_k_solver_scaling/solver_summary.csv",
                "data/stage74_larger_k_solver_scaling_result.json",
                "data/stage74_larger_k_solver_scaling_audit.json",
                "reports/stage-74/larger_k_solver_scaling.md",
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
        (root / "data/stage74_larger_k_solver_scaling_result.json").read_text(
            encoding="ascii"
        )
    )
    result = write_bundle(root, args.output, list(PATHS))
    scale = source["scaling_summary"]
    hardness = source["hardness_summary"]
    result.update(
        {
            "operation": "audited Stage74 larger-k constraint-native solver-scaling benchmark",
            "target_count": 4,
            "logical_model_count": scale["model_count"],
            "model_k_count": scale["model_k_count"],
            "workload_cell_count": scale["workload_cell_count"],
            "solver_trial_count": scale["solver_trial_count"],
            "exact_oracle_cell_count": scale["exact_oracle_cell_count"],
            "maximum_k": scale["maximum_k"],
            "maximum_total_fixed_k_subset_count": scale[
                "maximum_total_fixed_k_subset_count"
            ],
            "nonexact_solver_disagreement_cell_count": hardness[
                "nonexact_solver_disagreement_cell_count"
            ],
            "nonexact_solver_disagreement_fraction": hardness[
                "nonexact_solver_disagreement_fraction"
            ],
            "sampler_strict_win_cell_count": hardness[
                "sampler_strict_win_cell_count"
            ],
            "strong_classical_strict_win_cell_count": hardness[
                "strong_classical_strict_win_cell_count"
            ],
            "explicit_variable_k_cqm_design_authorized": source["decision"][
                "explicit_variable_k_cqm_design_authorized"
            ],
            "hardware_shaped_sampler_poc_authorized": source["decision"][
                "hardware_shaped_sampler_poc_authorized"
            ],
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
