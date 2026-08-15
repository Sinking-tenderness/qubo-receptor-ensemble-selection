"""Build the Stage61a PPARD Remaining-144 preparation input bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle
from scripts.experimental.unidock.prepare_development_ligand_inputs import read_json
from scripts.experimental.unidock.prepare_stage61a_ppard_remaining144_inputs import (
    validate_source,
)


CONFIG = "configs/stage61a_ppard_remaining144_unidock_input_preparation.json"
PATHS = (
    "configs/stage55_ppard_small_pilot_preregistration.json",
    "configs/stage60_ppard_transferred_qubo_freeze.json",
    CONFIG,
    "data/stage56_ppard_ligand_panel_allocation_summary.json",
    "data/processed/stage56_ppard_train240_ligand_manifest.csv",
    "data/processed/stage56_ppard_pilot96_ligand_manifest.csv",
    "data/processed/stage60_ppard_full_development_outer_fold_assignments.csv",
    "data/processed/stage60_ppard_full_development_inner_fold_assignments.csv",
    "data/stage60_ppard_transferred_qubo_model_record.json",
    "data/stage60_ppard_transferred_qubo_freeze_result.json",
    "data/stage60_ppard_transferred_qubo_freeze_audit.json",
    "data/processed/stage61a_ppard_remaining144_ligand_manifest.csv",
    "data/stage61a_ppard_remaining144_manifest_freeze.json",
    "environment/stage08_unidock_gpu.yml",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/freeze_stage61a_ppard_remaining_development_manifest.py",
    "scripts/prepare_ligand_3d_sdf.py",
    "scripts/batch_prepare_ligand_pdbqt.py",
    "scripts/audit_stage61a_ppard_remaining144_inputs.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/prepare_development_ligand_inputs.py",
    "scripts/experimental/unidock/prepare_stage61a_ppard_remaining144_inputs.py",
    "scripts/experimental/unidock/run_stage61a_ppard_remaining144_input_preparation_remote.sh",
    "scripts/experimental/unidock/build_stage61a_ppard_remaining144_input_bundle.py",
    "reports/stage-61a/ppard_remaining144_input_preparation.md",
    "tests/test_stage60_61a_ppard_freezes.py",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = read_json(root / CONFIG)
    rows, _, labels = validate_source(root, config)
    if len(rows) != 144 or labels != {"active": 72, "decoy": 72}:
        raise ValueError("Stage61a frozen Remaining-144 differs")
    if {row["pilot_selected"] for row in rows} != {"False"}:
        raise ValueError("Stage61a contains a Pilot-96 row")
    stage60 = read_json(root / "data/stage60_ppard_transferred_qubo_freeze_result.json")
    stage60_audit = read_json(root / "data/stage60_ppard_transferred_qubo_freeze_audit.json")
    if not stage60["decision"]["remaining_development_ligand_preparation_authorized"]:
        raise ValueError("Stage61a is not authorized by Stage60")
    if stage60_audit["status"] != "stage60_ppard_transferred_qubo_independent_audit_ok":
        raise ValueError("Stage60 independent audit did not pass")
    if any(
        marker in path.lower()
        for path in PATHS
        for marker in ("fresh_validation", "locked_test", "protected/")
    ):
        raise ValueError("Stage61a bundle crossed a protected split boundary")
    return sorted(set(PATHS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, bundle_paths(root))
    result.update(
        {
            "operation": "Stage61a PPARD Remaining-144 checkpointed ligand input preparation",
            "target_id": "PPARD",
            "analysis_class": "prospective remaining development train",
            "development_active_count": 72,
            "development_decoy_count": 72,
            "development_ligand_count": 144,
            "frozen_receptor_count": 29,
            "future_seed_count": 3,
            "future_pair_count": 12528,
            "gpu_docking_jobs_in_this_bundle": 0,
            "fresh_validation_structures_prepared": 0,
            "locked_test_structures_prepared": 0,
            "resume_supported": True,
            "gpu_required_for_this_stage": False,
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
