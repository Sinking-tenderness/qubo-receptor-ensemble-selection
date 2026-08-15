"""Build the audited Stage69 QUBO precision-compression core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage68_quality_plateau_portfolio_qubo_bundle import (
        PATHS as STAGE68_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage68_quality_plateau_portfolio_qubo_bundle import (
        PATHS as STAGE68_PATHS,
    )


PATHS = tuple(
    sorted(
        set(
            STAGE68_PATHS
            + (
                "configs/stage69_qubo_precision_compression.json",
                "scripts/run_stage69_qubo_precision_compression.py",
                "scripts/audit_stage69_qubo_precision_compression.py",
                "scripts/build_stage69_qubo_precision_compression_bundle.py",
                "tests/test_stage69_qubo_precision_compression.py",
                "results/runs/stage69_qubo_precision_compression/cell_metrics.csv",
                "results/runs/stage69_qubo_precision_compression/scale_summary.csv",
                "data/stage69_qubo_precision_compression_model_record.json",
                "data/stage69_qubo_precision_compression_result.json",
                "data/stage69_qubo_precision_compression_audit.json",
                "reports/stage-69/qubo_precision_compression.md",
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
        (root / "data/stage69_qubo_precision_compression_result.json").read_text(
            encoding="ascii"
        )
    )
    result = write_bundle(root, args.output, list(PATHS))
    near_miss = source["best_uniform_near_miss"]
    result.update(
        {
            "operation": "audited Stage69 optimum-preserving QUBO precision-compression screen",
            "target_count": 4,
            "scale_count": source["scale_count"],
            "cell_metric_count": source["cell_metric_count"],
            "continuous_milp_certificate_count": source[
                "continuous_milp_certificate_count"
            ],
            "quantized_milp_certificate_count": source[
                "quantized_milp_certificate_count"
            ],
            "selected_quality_integer_scale": int(
                source["selected_compression"].get("quality_integer_scale", 0)
            ),
            "diagnostic_near_miss_quality_integer_scale": int(
                near_miss.get("quality_integer_scale", 0)
            ),
            "diagnostic_near_miss_dynamic_range_compression_factor": float(
                near_miss.get("dynamic_range_compression_factor_vs_4095", 0.0)
            ),
            "compressed_qubo_freeze_authorized": source["compression_gate"][
                "compressed_qubo_freeze_authorized"
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
