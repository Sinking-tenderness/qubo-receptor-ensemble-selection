"""Build the deterministic Stage 52b/c and Stage 53 PPARA evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


PATHS = (
    "configs/stage52c_ppara_target_id_amendment.json",
    "configs/stage53_ppara_large_pool_qubo_transfer.json",
    "configs/stage37_cross_target_robust_functional_qubo.json",
    "configs/stage42f_bace1_rank_sensitive_pair_qubo.json",
    "data/stage52b_ppara_train374_downloaded_result_audit.json",
    "data/stage52b_ppara_train374_unidock113_production_audit.json",
    "data/stage52c_ppara_target_id_amendment_result.json",
    "data/stage53_ppara_large_pool_qubo_transfer_result.json",
    "data/stage53_ppara_large_pool_qubo_transfer_audit.json",
    "data/processed/stage52a_ppara_train374_unidock_pdbqt_manifest.csv",
    "data/processed/stage52b_ppara_stage51_passing20_receptor_manifest.csv",
    "results/runs/stage52b_ppara_train374_unidock113_production/summary.json",
    "results/runs/stage52b_ppara_train374_unidock113_production/scores.csv",
    "results/runs/stage52b_ppara_train374_unidock113_production/batch_runs.csv",
    "results/runs/stage52b_ppara_train374_unidock113_production/primary_median_score_matrix.csv",
    "results/runs/stage52b_ppara_train374_unidock113_production/sensitivity_minimum_score_matrix.csv",
    "results/runs/stage52c_ppara_target_id_amendment/scores.csv",
    "results/runs/stage53_ppara_large_pool_qubo_transfer/fold_assignments.csv",
    "results/runs/stage53_ppara_large_pool_qubo_transfer/selection_metrics.csv",
    "results/runs/stage53_ppara_large_pool_qubo_transfer/fixed_k_landscape.csv",
    "reports/stage-53/ppara_large_pool_qubo_transfer.md",
    "scripts/audit_stage52b_ppara_result_archives.py",
    "scripts/amend_stage52c_ppara_target_id.py",
    "scripts/run_stage53_ppara_large_pool_qubo_transfer.py",
    "scripts/audit_stage53_ppara_large_pool_qubo_transfer.py",
    "scripts/build_stage53_ppara_large_pool_qubo_transfer_bundle.py",
    "tests/test_stage52b_c_53_ppara.py",
    "scripts/build_stage05_mk14_remote_bundle.py"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, list(PATHS))
    result.update(
        {
            "operation": "Stage52b downloaded-result audit, Stage52c metadata amendment, and Stage53 frozen PPARA QUBO-transfer evidence",
            "target_id": "PPARA",
            "receptor_count": 20,
            "ligand_count": 374,
            "seed_count": 3,
            "pair_count": 22440,
            "stage52b_technical_gate": "pass",
            "stage53_application_transfer": "no_go",
            "stage53_solver_novelty": "no_go",
            "fresh_validation_rows": 0,
            "locked_test_rows": 0,
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
