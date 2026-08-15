"""Build the deterministic Stage91c BACE1 ChEMBL-365 production bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
from scripts.experimental.unidock import (
    run_stage91c_bace1_chembl365_production as runner,
)


CONFIG = "configs/stage91c_bace1_chembl365_unidock113_production.json"
FIXED_PATHS = (
    CONFIG,
    "configs/stage91c_bace1_group_robust_development_docking_preregistration.json",
    "environment/stage08_unidock_gpu.yml",
    "scripts/__init__.py",
    "scripts/experimental/__init__.py",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
    "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
    "scripts/experimental/unidock/run_stage09_mk14_train696_production.py",
    "scripts/experimental/unidock/audit_stage09_mk14_train696_production.py",
    "scripts/experimental/unidock/run_stage91c_bace1_chembl365_production.py",
    "scripts/experimental/unidock/audit_stage91c_bace1_chembl365_production.py",
    "scripts/experimental/unidock/run_stage91c_bace1_chembl365_production_remote.sh",
    "scripts/build_stage91c_bace1_chembl365_production_bundle.py",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = runner.common.read_json(root / CONFIG)
    receptors, ligands, audit = runner.validate_inputs(root, config)
    if audit["confirmation_rows"] != 0 or audit["locked_test_rows"] != 0:
        raise ValueError("Stage91c bundle crossed a data boundary")
    if len(receptors) != 34 or len(ligands) != 365:
        raise ValueError("Stage91c bundle dimensions differ")
    if audit["expected_pair_count"] != 37230:
        raise ValueError("Stage91c bundle pair count differs")

    paths = list(FIXED_PATHS)
    for descriptor in config["implementation"].values():
        path = str(descriptor["path"]).replace("\\", "/")
        if runner.common.file_sha256(root / path) != str(descriptor["sha256"]).upper():
            raise ValueError(f"Stage91c implementation hash differs: {path}")
        paths.append(path)
    for value in config["inputs"].values():
        if isinstance(value, dict) and "path" in value:
            paths.append(str(value["path"]))
    receptor_manifest = str(config["inputs"]["receptor_manifest"]["path"])
    ligand_manifest = str(config["inputs"]["ligand_manifest"]["path"])
    paths.extend(manifest_paths(root, receptor_manifest, "receptor_pdbqt"))
    paths.extend(manifest_paths(root, ligand_manifest, "pdbqt_path"))
    normalized = sorted(set(path.replace("\\", "/") for path in paths))
    forbidden = ("confirmation_a", "confirmation_b", "locked_test", "fresh_validation")
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage91c bundle contains a protected panel path")
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
            "operation": "Stage91c BACE1 ChEMBL-365 x qualified34 x three-seed Uni-Dock production bundle",
            "target_id": "BACE1",
            "experiment_class": "prospective_objective_frozen_development",
            "receptor_count": 34,
            "ligand_count": 365,
            "seed_count": 3,
            "gpu_batch_count": 102,
            "gpu_pair_count": 37230,
            "profile_id": "enhanced",
            "exhaustiveness": 1024,
            "max_step": 80,
            "confirmation_rows": 0,
            "locked_test_rows": 0,
            "gpu_required_for_execution": True,
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
