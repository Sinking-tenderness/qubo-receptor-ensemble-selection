"""Build the audited Stage70 constraint-aware QUBO encoding core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage69_qubo_precision_compression_bundle import (
        PATHS as STAGE69_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage69_qubo_precision_compression_bundle import (
        PATHS as STAGE69_PATHS,
    )


PATHS = tuple(
    sorted(
        set(
            STAGE69_PATHS
            + (
                "configs/stage70_constraint_aware_qubo_encoding.json",
                "scripts/run_stage70_constraint_aware_qubo_encoding.py",
                "scripts/audit_stage70_constraint_aware_qubo_encoding.py",
                "scripts/build_stage70_constraint_aware_qubo_encoding_bundle.py",
                "tests/test_stage70_constraint_aware_qubo_encoding.py",
                "results/runs/stage70_constraint_aware_qubo_encoding/cell_metrics.csv",
                "results/runs/stage70_constraint_aware_qubo_encoding/candidate_summary.csv",
                "data/stage70_constraint_aware_qubo_encoding_model_record.json",
                "data/stage70_constraint_aware_qubo_encoding_result.json",
                "data/stage70_constraint_aware_qubo_encoding_audit.json",
                "reports/stage-70/constraint_aware_qubo_encoding.md",
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
        (root / "data/stage70_constraint_aware_qubo_encoding_result.json").read_text(
            encoding="ascii"
        )
    )
    result = write_bundle(root, args.output, list(PATHS))
    selected = source["selected_encoding"]
    result.update(
        {
            "operation": "audited Stage70 constraint-aware exact-penalty QUBO encoding",
            "target_count": 4,
            "candidate_count": source["candidate_count"],
            "cell_metric_count": source["cell_metric_count"],
            "selected_candidate_id": selected.get("candidate_id", ""),
            "selected_slack_weight_cap": int(
                selected.get("slack_weight_cap", 0)
            ),
            "selected_maximum_coefficient_dynamic_range": float(
                selected.get("maximum_coefficient_dynamic_range", 0.0)
            ),
            "dynamic_range_improvement_factor_vs_stage69": float(
                selected.get("dynamic_range_improvement_factor_vs_stage69", 0.0)
            ),
            "compact_logical_qubo_freeze_authorized": source["encoding_gate"][
                "compact_logical_qubo_freeze_authorized"
            ],
            "coefficient_noise_simulation_authorized": source["decision"][
                "coefficient_noise_simulation_authorized"
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
