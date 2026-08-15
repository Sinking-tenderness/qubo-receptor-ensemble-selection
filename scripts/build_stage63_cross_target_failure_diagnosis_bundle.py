"""Build the audited Stage61b diagnostics and Stage63 mechanism bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle


PATHS = (
    "configs/stage63_cross_target_rank_pair_failure_diagnosis.json",
    "scripts/audit_stage61b_ppard_diagnostics_archive.py",
    "scripts/run_stage63_cross_target_rank_pair_failure_diagnosis.py",
    "scripts/audit_stage63_cross_target_rank_pair_failure_diagnosis.py",
    "scripts/build_stage63_cross_target_failure_diagnosis_bundle.py",
    "tests/test_stage63_cross_target_rank_pair_failure_diagnosis.py",
    "data/stage61b_ppard_diagnostics_archive_local_audit.json",
    "data/stage42f_bace1_rank_sensitive_pair_qubo_result.json",
    "data/stage42f_bace1_rank_sensitive_pair_qubo_audit.json",
    "data/stage44_pparg_md96_rank_sensitive_qubo_result.json",
    "data/stage44_pparg_md96_rank_sensitive_qubo_audit.json",
    "data/stage53_ppara_large_pool_qubo_transfer_result.json",
    "data/stage53_ppara_large_pool_qubo_transfer_audit.json",
    "data/stage62_ppard_train240_nested_qubo_result.json",
    "data/stage62_ppard_train240_nested_qubo_audit.json",
    "data/stage63_cross_target_rank_pair_failure_diagnosis_result.json",
    "data/stage63_cross_target_rank_pair_failure_diagnosis_audit.json",
    "results/runs/stage42f_bace1_rank_sensitive_pair_qubo/fold_metrics.csv",
    "results/runs/stage42f_bace1_rank_sensitive_pair_qubo/full_metrics.csv",
    "results/runs/stage44_pparg_md96_rank_sensitive_qubo/selection_metrics.csv",
    "results/runs/stage44_pparg_md96_rank_sensitive_qubo/solver_comparison.csv",
    "results/runs/stage53_ppara_large_pool_qubo_transfer/fixed_k_landscape.csv",
    "results/runs/stage62_ppard_train240_nested_qubo/outer_k_metrics.csv",
    "results/runs/stage62_ppard_train240_nested_qubo/inner_k_selection.csv",
    "results/runs/stage62_ppard_train240_nested_qubo/objective_gap_cells.csv",
    "results/runs/stage63_cross_target_rank_pair_failure_diagnosis/fixed_k_landscape.csv",
    "results/runs/stage63_cross_target_rank_pair_failure_diagnosis/fold_diagnostics.csv",
    "results/runs/stage63_cross_target_rank_pair_failure_diagnosis/target_k_summary.csv",
    "results/runs/stage63_cross_target_rank_pair_failure_diagnosis/target_summary.csv",
    "results/runs/stage63_cross_target_rank_pair_failure_diagnosis/ppard_nested_k_diagnostics.csv",
    "results/runs/stage63_cross_target_rank_pair_failure_diagnosis/solver_diagnostics.csv",
    "reports/stage-63/cross_target_rank_pair_failure_diagnosis.md",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, sorted(PATHS))
    result.update(
        {
            "operation": "audited Stage61b diagnostics and Stage63 cross-target rank-pair failure diagnosis",
            "target_count": 4,
            "outer_fold_count": 16,
            "fixed_k_cell_count": 96,
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
