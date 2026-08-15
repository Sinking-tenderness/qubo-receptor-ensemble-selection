"""Build the deterministic Stage 55 PPARD preregistration evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


PATHS = (
    "configs/stage55_ppard_small_pilot_preregistration.json",
    "data/stage47b_expanded_new_target_feasibility_screen_result.json",
    "data/processed/stage47b_expanded_new_target_feasibility_screen.csv",
    "data/stage54_future_target_intake_criteria.json",
    "data/stage55_ppard_small_pilot_preregistration_result.json",
    "data/stage55_ppard_small_pilot_preregistration_audit.json",
    "data/raw/external_targets/ppard_dude/ppard.tar.gz",
    "data/raw/external_targets/ppard_dude/ppard/actives_final.ism",
    "data/raw/external_targets/ppard_dude/ppard/decoys_final.ism",
    "data/raw/external_targets/ppard_dude/ppard/receptor.pdb",
    "data/raw/external_targets/ppard_dude/ppard/crystal_ligand.mol2",
    "data/raw/rcsb/ppard/2ZNP.pdb",
    "data/raw/rcsb/ppard/2ZNP.cif",
    "data/raw/rcsb/ppard/2ZNP_K55_A922.sdf",
    "scripts/audit_stage55_ppard_small_pilot_source.py",
    "scripts/audit_stage55_ppard_small_pilot_result.py",
    "scripts/audit_external_target_intake.py",
    "scripts/build_stage55_ppard_small_pilot_bundle.py",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "tests/test_stage55_ppard_small_pilot.py",
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
            "operation": "Stage55 outcome-blind PPARD source audit and small-pilot preregistration",
            "target_id": "PPARD",
            "source_active_count": 240,
            "source_decoy_count": 12250,
            "metadata_eligible_receptor_count": 51,
            "pilot_active_count": 48,
            "pilot_decoy_count": 48,
            "maximum_seeded_docking_jobs": 14688,
            "source_gate": "pass",
            "pilot_docking_authorized": False,
            "full_training_matrix_authorized": False,
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
