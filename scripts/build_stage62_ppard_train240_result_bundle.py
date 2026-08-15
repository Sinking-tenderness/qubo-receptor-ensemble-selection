"""Build the audited Stage61c and Stage62 PPARD core result bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_stage05_mk14_remote_bundle import write_bundle


PATHS = (
    "configs/stage61c_ppard_target_id_amendment.json",
    "configs/stage62_ppard_train240_nested_qubo.json",
    "scripts/amend_stage61c_ppard_target_id.py",
    "scripts/audit_stage61c_ppard_target_id.py",
    "scripts/run_stage62_ppard_train240_nested_qubo.py",
    "scripts/audit_stage62_ppard_train240_nested_qubo.py",
    "scripts/build_stage62_ppard_train240_result_bundle.py",
    "data/stage58b_ppard_pilot96_unidock113_production_audit.json",
    "data/stage58c_ppard_target_id_amendment_result.json",
    "data/stage58c_ppard_target_id_amendment_audit.json",
    "data/stage60_ppard_transferred_qubo_freeze_result.json",
    "data/stage60_ppard_transferred_qubo_freeze_audit.json",
    "data/stage60_ppard_transferred_qubo_model_record.json",
    "data/stage61b_ppard_progress_descriptor_amendment01.json",
    "data/stage61b_ppard_remaining144_unidock113_production_audit.json",
    "data/stage61c_ppard_target_id_amendment_result.json",
    "data/stage61c_ppard_target_id_amendment_audit.json",
    "data/stage62_ppard_train240_nested_qubo_result.json",
    "data/stage62_ppard_train240_nested_qubo_audit.json",
    "data/stage62_ppard_train240_final_model_record.json",
    "data/processed/stage56_ppard_train240_ligand_manifest.csv",
    "data/processed/stage58a_ppard_pilot96_unidock_pdbqt_manifest.csv",
    "data/processed/stage58b_ppard_stage57_passing29_receptor_manifest.csv",
    "data/processed/stage60_ppard_full_development_outer_fold_assignments.csv",
    "data/processed/stage60_ppard_full_development_inner_fold_assignments.csv",
    "data/processed/stage61a_ppard_remaining144_unidock_pdbqt_manifest.csv",
    "results/runs/stage58b_ppard_pilot96_unidock113_production/summary.json",
    "results/runs/stage58b_ppard_pilot96_unidock113_production/scores.csv",
    "results/runs/stage58c_ppard_target_id_amendment/scores.csv",
    "results/runs/stage61b_ppard_remaining144_unidock113_production/summary.json",
    "results/runs/stage61b_ppard_remaining144_unidock113_production/scores.csv",
    "results/runs/stage61c_ppard_target_id_amendment/scores.csv",
    "results/runs/stage62_ppard_train240_nested_qubo/merged_scores.csv",
    "results/runs/stage62_ppard_train240_nested_qubo/inner_k_metrics.csv",
    "results/runs/stage62_ppard_train240_nested_qubo/inner_k_selection.csv",
    "results/runs/stage62_ppard_train240_nested_qubo/outer_k_metrics.csv",
    "results/runs/stage62_ppard_train240_nested_qubo/nested_outer_metrics.csv",
    "results/runs/stage62_ppard_train240_nested_qubo/objective_gap_cells.csv",
    "results/runs/stage62_ppard_train240_nested_qubo/final_method_metrics.csv",
    "reports/stage-62/ppard_train240_nested_qubo.md",
    "reports/stage-62/ppard_train240_interpretation.md",
    "tests/test_stage61c_62_ppard.py",
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
            "operation": "audited Stage61c metadata amendment and frozen Stage62 PPARD Train-240 nested QUBO analysis",
            "target_id": "PPARD",
            "development_ligand_count": 240,
            "receptor_count": 29,
            "seed_count": 3,
            "score_row_count": 20880,
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
