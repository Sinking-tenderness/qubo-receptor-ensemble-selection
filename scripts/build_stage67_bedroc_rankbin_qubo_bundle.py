"""Build the audited Stage67 BEDROC rank-bin QUBO core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage66_cross_target_auxiliary_coverage_qubo_bundle import (
        PATHS as STAGE66_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage66_cross_target_auxiliary_coverage_qubo_bundle import (
        PATHS as STAGE66_PATHS,
    )


PATHS = tuple(
    sorted(
        set(
            STAGE66_PATHS
            + (
                "configs/stage67_bedroc_rankbin_qubo.json",
                "scripts/run_stage67_bedroc_rankbin_qubo.py",
                "scripts/audit_stage67_bedroc_rankbin_qubo.py",
                "scripts/build_stage67_bedroc_rankbin_qubo_bundle.py",
                "tests/test_stage67_bedroc_rankbin_qubo.py",
                "results/runs/stage67_bedroc_rankbin_qubo/fixed_k_metrics.csv",
                "results/runs/stage67_bedroc_rankbin_qubo/target_summary.csv",
                "results/runs/stage67_bedroc_rankbin_qubo/resolution_summary.csv",
                "data/stage67_bedroc_rankbin_qubo_model_record.json",
                "data/stage67_bedroc_rankbin_qubo_result.json",
                "data/stage67_bedroc_rankbin_qubo_audit.json",
                "reports/stage-67/bedroc_rankbin_qubo.md",
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
        (root / "data/stage67_bedroc_rankbin_qubo_result.json").read_text(
            encoding="ascii"
        )
    )
    result = write_bundle(root, args.output, list(PATHS))
    result.update(
        {
            "operation": "audited Stage67 BEDROC-aligned rank-bin QUBO fidelity adjudication",
            "target_count": 4,
            "ligand_count": 1040,
            "receptor_count": 179,
            "score_row_count": 116532,
            "objective_count": source["objective_count"],
            "fixed_k_metric_count": source["fixed_k_metric_count"],
            "pair_off_reproduction_cell_count": source[
                "pair_off_reproduction_cell_count"
            ],
            "continuous_objective_supported": source["route_gate"][
                "continuous_objective_supported"
            ],
            "rankbin_qubo_freeze_authorized": source["route_gate"][
                "rankbin_qubo_freeze_authorized"
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
