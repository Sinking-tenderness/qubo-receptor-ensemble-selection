"""Build the deterministic Stage57 PPARD preparation and redocking bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle
from scripts.prepare_stage57_ppard_redocking_inputs import validate_inputs


PREP_CONFIG = "configs/stage57_ppard_cognate_redocking_input_preparation.json"
DOCK_CONFIG = "configs/stage57_ppard_cognate_redocking.json"
FIXED_PATHS = (
    PREP_CONFIG,
    DOCK_CONFIG,
    "data/raw/rcsb/ppard/2ZNP_K55_A922.sdf",
    "environment/stage57_ppard_unidock_gpu.yml",
    "reports/stage-57/ppard_cognate_redocking_execution.md",
    "scripts/__init__.py",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/build_stage57_ppard_cognate_redocking_input_bundle.py",
    "scripts/prepare_stage57_ppard_redocking_inputs.py",
    "scripts/prepare_receptor_stage57.py",
    "scripts/prepare_receptor.py",
    "scripts/prepare_stage50_ppara_large_pool_redocking_inputs.py",
    "scripts/prepare_stage18d_pparg_redocking_inputs.py",
    "scripts/prepare_stage13e_egfr_redocking_inputs.py",
    "scripts/prepare_stage14c_fa10_redocking_inputs.py",
    "scripts/select_mk14_rcsb_coordinate_pool.py",
    "scripts/select_stage13_egfr_coordinate_pool.py",
    "scripts/select_stage13b_egfr_expanded_coordinate_pool.py",
    "scripts/select_stage13c_egfr_local_pocket_pool.py",
    "scripts/batch_prepare_ligand_pdbqt.py",
    "scripts/run_mk14_expanded_redocking_gate.py",
    "scripts/evaluate_redocking_rmsd.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
    "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
    "scripts/experimental/unidock/run_stage14d_fa10_cognate_redocking.py",
    "scripts/experimental/unidock/run_stage57_ppard_cognate_redocking.py",
    "scripts/experimental/unidock/run_stage57_ppard_cognate_redocking_remote.sh",
    "tests/test_stage57_ppard_cognate_redocking.py",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config, _, rows, audit = validate_inputs(root / PREP_CONFIG, root)
    if audit["frozen_receptor_count"] != 51 or audit["cognate_ligand_count"] != 51:
        raise ValueError("Stage57 bundle count differs")
    if any(int(value) != 0 for value in audit["data_boundary"].values()):
        raise ValueError("Stage57 bundle crossed a protected boundary")
    paths = list(FIXED_PATHS)
    paths.extend(value["path"] for value in config["inputs"].values())
    paths.extend(value["path"] for value in config.get("dependencies", []))
    for row in rows:
        paths.extend((row["mmcif_path"], row["aligned_protein_pdb_path"]))
    normalized = sorted(set(path.replace("\\", "/") for path in paths))
    forbidden = (
        "ppard_fresh_validation",
        "ppard_locked_test",
        "stage57_ppard_cognate_redocking_results.csv",
        "stage57_ppard_receptor_gate_results.csv",
    )
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage57 input bundle contains a protected or outcome path")
    return normalized


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
            "operation": "Stage57 PPARD 51-receptor preparation and three-seed cognate redocking",
            "target_id": "PPARD",
            "frozen_receptor_count": 51,
            "cognate_ligand_count": 51,
            "minimum_prepared_receptor_count": 24,
            "seed_count": 3,
            "maximum_gpu_pair_count": 153,
            "profile_id": "enhanced",
            "exhaustiveness": 1024,
            "max_step": 80,
            "resume_supported": True,
            "auto_poweroff_supported": True,
            "protected_rows_read": 0,
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
