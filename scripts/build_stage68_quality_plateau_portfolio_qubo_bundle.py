"""Build the audited Stage68 quality-plateau portfolio QUBO core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage67_bedroc_rankbin_qubo_bundle import (
        PATHS as STAGE67_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage67_bedroc_rankbin_qubo_bundle import (
        PATHS as STAGE67_PATHS,
    )


PATHS = tuple(
    sorted(
        set(
            STAGE67_PATHS
            + (
                "configs/stage68_quality_plateau_portfolio_qubo.json",
                "scripts/run_stage68_quality_plateau_portfolio_qubo.py",
                "scripts/audit_stage68_quality_plateau_portfolio_qubo.py",
                "scripts/build_stage68_quality_plateau_portfolio_qubo_bundle.py",
                "tests/test_stage68_quality_plateau_portfolio_qubo.py",
                "results/runs/stage68_quality_plateau_portfolio_qubo/fixed_k_metrics.csv",
                "results/runs/stage68_quality_plateau_portfolio_qubo/target_summary.csv",
                "results/runs/stage68_quality_plateau_portfolio_qubo/global_summary.csv",
                "results/runs/stage68_quality_plateau_portfolio_qubo/loto_summary.csv",
                "results/runs/stage68_quality_plateau_portfolio_qubo/qubo_fidelity.csv",
                "data/stage68_quality_plateau_portfolio_qubo_model_record.json",
                "data/stage68_quality_plateau_portfolio_qubo_result.json",
                "data/stage68_quality_plateau_portfolio_qubo_audit.json",
                "reports/stage-68/quality_plateau_portfolio_qubo.md",
                "analysis/stage68_unfrozen_partition_probe_20260806/result.json",
                "analysis/stage68_unfrozen_partition_probe_20260806/global_summary.csv",
                "analysis/stage68_unfrozen_partition_probe_20260806/target_summary.csv",
                "analysis/stage68_unfrozen_partition_probe_20260806/loto_summary.csv",
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
        (root / "data/stage68_quality_plateau_portfolio_qubo_result.json").read_text(
            encoding="ascii"
        )
    )
    result = write_bundle(root, args.output, list(PATHS))
    result.update(
        {
            "operation": "audited Stage68 quality-preserving functional-diversity portfolio QUBO",
            "target_count": 4,
            "ligand_count": 1040,
            "receptor_count": 179,
            "score_row_count": 116532,
            "candidate_count": source["candidate_count"],
            "fixed_k_metric_count": source["fixed_k_metric_count"],
            "milp_certificate_count": source["milp_certificate_count"],
            "selected_candidate_id": source["selected_candidate"]["candidate_id"],
            "quality_plateau_qubo_freeze_authorized": source["route_gate"][
                "quality_plateau_qubo_freeze_authorized"
            ],
            "future_new_target_preregistration_authorized": source["decision"][
                "future_new_target_preregistration_authorized"
            ],
            "robustness_claim_authorized": source["decision"][
                "robustness_claim_authorized"
            ],
            "quantum_hardware_authorized": source["decision"][
                "quantum_hardware_authorized"
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
