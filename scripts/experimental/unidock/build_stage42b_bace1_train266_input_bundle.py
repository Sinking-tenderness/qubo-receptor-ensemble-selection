"""Build the Stage 42a/42b BACE1 Train-266 input-preparation bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import write_bundle


PATHS = (
    "configs/stage42_bace1_redocking_qualified_development_preregistration.json",
    "configs/stage42a_bace1_ligand_panel_allocation.json",
    "configs/stage42b_bace1_train266_unidock_input_preparation.json",
    "data/stage21a_bace1_source_and_active_allocation_summary.json",
    "data/processed/stage21a_bace1_active_panel_allocation.csv",
    "data/raw/external_targets/bace1_dude/bace1/decoys_final.ism",
    "environment/stage08_unidock_gpu.yml",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/allocate_stage42a_bace1_ligand_panels.py",
    "scripts/prepare_ligand_3d_sdf.py",
    "scripts/batch_prepare_ligand_pdbqt.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/prepare_stage42b_bace1_train266_inputs.py",
    "scripts/experimental/unidock/run_stage42b_bace1_train266_input_preparation_remote.sh",
    "scripts/experimental/unidock/build_stage42b_bace1_train266_input_bundle.py",
    "reports/stage-42/bace1_train266_input_preparation.md",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, list(PATHS))
    result.update({
        "operation": "Stage42 BACE1 panel freeze and Train-266 input preparation",
        "target_id": "BACE1",
        "receptor_count_frozen_for_later_production": 34,
        "development_active_count": 133,
        "development_decoy_count": 133,
        "development_ligand_count": 266,
        "production_seed_count_frozen_for_later_production": 3,
        "prospective_production_pair_count": 27132,
        "gpu_docking_jobs_in_this_bundle": 0,
        "fresh_validation_structures_prepared": 0,
        "locked_test_structures_prepared": 0,
    })
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
