"""Build the audited Stage66 auxiliary coverage QUBO core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage65_cross_target_pair_sign_mechanism_bundle import (
        PATHS as STAGE65_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage65_cross_target_pair_sign_mechanism_bundle import (
        PATHS as STAGE65_PATHS,
    )


PATHS = tuple(
    sorted(
        set(
            STAGE65_PATHS
            + (
                "configs/stage66_cross_target_auxiliary_coverage_qubo.json",
                "scripts/run_stage66_cross_target_auxiliary_coverage_qubo.py",
                "scripts/audit_stage66_cross_target_auxiliary_coverage_qubo.py",
                "scripts/build_stage66_cross_target_auxiliary_coverage_qubo_bundle.py",
                "tests/test_stage66_cross_target_auxiliary_coverage_qubo.py",
                "results/runs/stage66_cross_target_auxiliary_coverage_qubo/fixed_k_metrics.csv",
                "results/runs/stage66_cross_target_auxiliary_coverage_qubo/target_summary.csv",
                "results/runs/stage66_cross_target_auxiliary_coverage_qubo/global_summary.csv",
                "results/runs/stage66_cross_target_auxiliary_coverage_qubo/loto_summary.csv",
                "data/stage66_auxiliary_coverage_qubo_model_record.json",
                "data/stage66_cross_target_auxiliary_coverage_qubo_result.json",
                "data/stage66_cross_target_auxiliary_coverage_qubo_audit.json",
                "reports/stage-66/cross_target_auxiliary_coverage_qubo.md",
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
        (root / "data/stage66_cross_target_auxiliary_coverage_qubo_result.json")
        .read_text(encoding="ascii")
    )
    result = write_bundle(root, args.output, list(PATHS))
    result.update(
        {
            "operation": "audited Stage66 cross-target auxiliary coverage QUBO development",
            "target_count": 4,
            "ligand_count": 1040,
            "receptor_count": 179,
            "score_row_count": 116532,
            "candidate_count": source["candidate_count"],
            "fixed_k_metric_count": source["fixed_k_metric_count"],
            "pair_off_reproduction_cell_count": source[
                "pair_off_reproduction_cell_count"
            ],
            "coverage_objective_freeze_authorized": source["freeze_gate"][
                "coverage_objective_freeze_authorized"
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
