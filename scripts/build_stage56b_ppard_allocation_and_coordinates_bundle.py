"""Build the deterministic Stage 56b PPARD allocation and coordinate bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


STATIC_PATHS = (
    "configs/stage56_ppard_ligand_panel_allocation.json",
    "configs/stage56_ppard_coordinate_pool_hard_gate.json",
    "configs/stage56a_ppard_numbering_failure_adjudication.json",
    "configs/stage56b_ppard_sequence_remapped_coordinates.json",
    "configs/stage56b_ppard_coordinate_pool_amendment01.json",
    "configs/stage56b_ppard_allocation_and_coordinates_audit.json",
    "data/stage55_ppard_small_pilot_preregistration_result.json",
    "data/stage55_ppard_small_pilot_preregistration_audit.json",
    "data/stage56_ppard_ligand_panel_allocation_summary.json",
    "data/stage56_ppard_coordinate_pool_summary.json",
    "data/stage56a_ppard_numbering_failure_adjudication_result.json",
    "data/stage56b_ppard_sequence_remapped_coordinates_result.json",
    "data/stage56b_ppard_coordinate_pool_amendment01_summary.json",
    "data/stage56b_ppard_allocation_and_coordinates_audit.json",
    "data/processed/stage47b_expanded_new_target_candidate_metadata.csv",
    "data/processed/stage56_ppard_selected_ligand_panel_manifest.csv",
    "data/processed/stage56_ppard_train240_ligand_manifest.csv",
    "data/processed/stage56_ppard_pilot96_ligand_manifest.csv",
    "data/processed/stage56_ppard_pilot96_fold_assignments.csv",
    "data/processed/stage56_ppard_coordinate_candidate_audit.csv",
    "data/processed/stage56a_ppard_sequence_numbering_diagnostic.csv",
    "data/processed/stage56b_ppard_sequence_remapped_coordinate_manifest.csv",
    "data/processed/stage56b_ppard_coordinate_candidate_audit_amendment01.csv",
    "data/processed/stage56b_ppard_coordinate_eligible_pool_amendment01.csv",
    "data/processed/stage56b_ppard_hard_gate_redocking_pool_amendment01.csv",
    "data/raw/external_targets/ppard_dude/ppard/actives_final.ism",
    "data/raw/external_targets/ppard_dude/ppard/decoys_final.ism",
    "reports/stage-56/ppard_coordinate_pool_hard_gate.md",
    "reports/stage-56b/ppard_coordinate_pool_amendment01.md",
    "scripts/allocate_stage56_ppard_ligand_panels.py",
    "scripts/audit_stage56_ppard_coordinate_pool.py",
    "scripts/diagnose_stage56a_ppard_numbering_failure.py",
    "scripts/prepare_stage56b_ppard_sequence_remapped_coordinates.py",
    "scripts/audit_stage56b_ppard_allocation_and_coordinates.py",
    "scripts/build_stage56b_ppard_allocation_and_coordinates_bundle.py",
    "scripts/select_stage13_egfr_coordinate_pool.py",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "tests/test_stage56_ppard_allocation_and_coordinates.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = list(STATIC_PATHS)
    for directory in (
        root / "data/raw/rcsb/ppard",
        root / "results/runs/stage56b_ppard_sequence_remapped_coordinates/mmcif",
        root / "results/runs/stage56b_ppard_coordinate_pool_amendment01/aligned",
    ):
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted(value for value in directory.rglob("*") if value.is_file())
        )
    paths = sorted(set(paths))
    result = write_bundle(root, args.output, paths)
    result.update(
        {
            "operation": "Stage56 PPARD disjoint pilot allocation, numbering adjudication, and Amendment01 coordinate hard-gate",
            "target_id": "PPARD",
            "selected_panel_row_count": 2760,
            "pilot_ligand_count": 96,
            "pilot_outer_fold_count": 4,
            "metadata_candidate_count": 51,
            "original_coordinate_eligible_count": 17,
            "amended_coordinate_eligible_count": 51,
            "coordinate_thresholds_changed": False,
            "raw_coordinates_modified": False,
            "max_min_compression_used": False,
            "cognate_redocking_input_preparation_authorized": True,
            "pilot_docking_authorized": False,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
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
