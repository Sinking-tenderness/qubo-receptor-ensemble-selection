"""Build the audited Stage65 pair-sign mechanism core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle


PATHS = (
    "configs/stage64_cross_target_uncertainty_shrunk_qubo.json",
    "configs/stage65_cross_target_pair_sign_mechanism.json",
    "scripts/run_stage05_mk14_method_gate.py",
    "scripts/run_stage42d_bace1_large_pool_qubo_screen.py",
    "scripts/run_stage42f_bace1_rank_sensitive_pair_qubo.py",
    "scripts/run_stage64_cross_target_uncertainty_shrunk_qubo.py",
    "scripts/run_stage65_cross_target_pair_sign_mechanism.py",
    "scripts/audit_stage65_cross_target_pair_sign_mechanism.py",
    "scripts/build_stage65_cross_target_pair_sign_mechanism_bundle.py",
    "tests/test_stage65_cross_target_pair_sign_mechanism.py",
    "data/stage64_cross_target_uncertainty_shrunk_qubo_result.json",
    "data/stage64_cross_target_uncertainty_shrunk_qubo_audit.json",
    "results/runs/stage64_cross_target_uncertainty_shrunk_qubo/fixed_k_metrics.csv",
    "results/runs/stage42c_bace1_train266_unidock113_production/scores.csv",
    "data/processed/stage42b_bace1_train266_unidock_pdbqt_manifest.csv",
    "data/processed/stage42c_bace1_redocking_qualified34_receptor_manifest.csv",
    "results/runs/stage42f_bace1_rank_sensitive_pair_qubo/fold_assignments.csv",
    "results/runs/stage43_pparg_md96_unidock113_production/scores.csv",
    "data/processed/stage32_pparg_train160_ligand_manifest.csv",
    "data/processed/stage43_pparg_md96_prepared_receptor_manifest.csv",
    "results/runs/stage44_pparg_md96_rank_sensitive_qubo/fold_assignments.csv",
    "results/runs/stage52c_ppara_target_id_amendment/scores.csv",
    "data/processed/stage52a_ppara_train374_unidock_pdbqt_manifest.csv",
    "data/processed/stage52b_ppara_stage51_passing20_receptor_manifest.csv",
    "results/runs/stage53_ppara_large_pool_qubo_transfer/fold_assignments.csv",
    "results/runs/stage62_ppard_train240_nested_qubo/merged_scores.csv",
    "data/processed/stage56_ppard_train240_ligand_manifest.csv",
    "data/processed/stage58b_ppard_stage57_passing29_receptor_manifest.csv",
    "data/processed/stage60_ppard_full_development_outer_fold_assignments.csv",
    "results/runs/stage65_cross_target_pair_sign_mechanism/edge_transfer.csv",
    "results/runs/stage65_cross_target_pair_sign_mechanism/edge_fold_summary.csv",
    "results/runs/stage65_cross_target_pair_sign_mechanism/edge_target_summary.csv",
    "results/runs/stage65_cross_target_pair_sign_mechanism/fixed_k_metrics.csv",
    "results/runs/stage65_cross_target_pair_sign_mechanism/target_summary.csv",
    "results/runs/stage65_cross_target_pair_sign_mechanism/global_summary.csv",
    "data/stage65_cross_target_pair_sign_mechanism_result.json",
    "data/stage65_cross_target_pair_sign_mechanism_audit.json",
    "reports/stage-65/cross_target_pair_sign_mechanism.md",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = json.loads(
        (root / "data/stage65_cross_target_pair_sign_mechanism_result.json")
        .read_text(encoding="ascii")
    )
    result = write_bundle(root, args.output, sorted(PATHS))
    result.update(
        {
            "operation": "audited Stage65 cross-target pair-sign mechanism adjudication",
            "target_count": 4,
            "ligand_count": 1040,
            "receptor_count": 179,
            "score_row_count": 116532,
            "pair_edge_observation_count": source["edge_transfer_row_count"],
            "candidate_count": source["candidate_count"],
            "fixed_k_metric_count": source["fixed_k_metric_count"],
            "pair_residual_route_supported": source["decision_gate"][
                "pair_residual_route_supported"
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
