"""Build the Stage50 PPARA 64-receptor input-preparation bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle
from scripts.prepare_stage50_ppara_large_pool_redocking_inputs import validate_inputs


CONFIG = "configs/stage50_ppara_large_pool_redocking_input_preparation.json"
FIXED_PATHS = (
    CONFIG,
    "environment/stage50_ppara_input_preparation.yml",
    "data/stage50a_ppara_input_preparation_runtime_adjudication.json",
    "reports/stage-50/ppara_large_pool_input_preparation_remote.md",
    "scripts/__init__.py",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/build_stage50_ppara_large_pool_input_bundle.py",
    "scripts/prepare_stage50_ppara_large_pool_redocking_inputs.py",
    "scripts/run_stage50_ppara_large_pool_input_preparation_remote.sh",
    "scripts/prepare_receptor.py",
    "scripts/prepare_receptor_stage50.py",
    "scripts/prepare_stage18d_pparg_redocking_inputs.py",
    "scripts/prepare_stage13e_egfr_redocking_inputs.py",
    "scripts/prepare_stage14c_fa10_redocking_inputs.py",
    "scripts/batch_prepare_ligand_pdbqt.py",
    "scripts/run_mk14_expanded_redocking_gate.py",
    "scripts/select_mk14_rcsb_coordinate_pool.py",
    "scripts/select_stage13_egfr_coordinate_pool.py",
    "scripts/select_stage13b_egfr_expanded_coordinate_pool.py",
    "scripts/select_stage13c_egfr_local_pocket_pool.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "tests/test_stage49_ppara_intake.py",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config, _, rows, audit = validate_inputs(root / CONFIG, root)
    if audit["receptor_count"] != 64 or audit["cognate_ligand_count"] != 64:
        raise ValueError("Stage50 bundle count differs")
    if any(int(value) != 0 for value in audit["data_boundary"].values()):
        raise ValueError("Stage50 bundle crossed a protected boundary")
    paths = list(FIXED_PATHS)
    paths.extend(value["path"] for value in config["inputs"].values())
    paths.extend(value["path"] for value in config.get("dependencies", []))
    for row in rows:
        paths.extend((row["mmcif_path"], row["aligned_protein_pdb_path"]))
    normalized = sorted(set(path.replace("\\", "/") for path in paths))
    forbidden = ("fresh_validation", "locked_test", "stage11_mk14")
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage50 bundle contains a protected path")
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
            "operation": "Stage50 PPARA 64-receptor cognate-redocking input preparation",
            "target_id": "PPARA",
            "receptor_count": 64,
            "cognate_ligand_count": 64,
            "minimum_prepared_receptor_count": 24,
            "docking_jobs": 0,
            "gpu_required_for_execution": False,
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
