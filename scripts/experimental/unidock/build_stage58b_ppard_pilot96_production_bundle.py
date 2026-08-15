"""Build the deterministic PPARD Pilot-96 Uni-Dock production bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from scripts.experimental.unidock.run_stage58b_ppard_pilot96_production import (
        common,
        validate_inputs,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from scripts.experimental.unidock.run_stage58b_ppard_pilot96_production import (
        common,
        validate_inputs,
    )


CONFIG = "configs/stage58b_ppard_pilot96_unidock113_production.json"
FIXED_PATHS = (
    CONFIG,
    "configs/stage55_ppard_small_pilot_preregistration.json",
    "environment/stage08_unidock_gpu.yml",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/build_stage58b_ppard_passing29_receptor_manifest.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
    "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
    "scripts/experimental/unidock/run_stage09_mk14_train696_production.py",
    "scripts/experimental/unidock/audit_stage09_mk14_train696_production.py",
    "scripts/experimental/unidock/run_stage52b_ppara_train374_production.py",
    "scripts/experimental/unidock/run_stage58b_ppard_pilot96_production.py",
    "scripts/experimental/unidock/audit_stage58b_ppard_pilot96_production.py",
    "scripts/experimental/unidock/run_stage58b_ppard_pilot96_production_remote.sh",
    "scripts/experimental/unidock/build_stage58b_ppard_pilot96_production_bundle.py",
    "reports/stage-58/ppard_pilot96_unidock113_production.md",
    "tests/test_stage58b_ppard_pilot96_production.py",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = common.read_json(root / CONFIG)
    receptors, ligands, audit = validate_inputs(root, config)
    if audit["fresh_validation_rows"] != 0 or audit["locked_test_rows"] != 0:
        raise ValueError("Stage58b bundle crossed a protected data boundary")
    if len(receptors) != 29 or len(ligands) != 96:
        raise ValueError("Stage58b bundle input dimensions differ")
    if audit["expected_pair_count"] != 8352:
        raise ValueError("Stage58b bundle pair count differs")

    paths = list(FIXED_PATHS)
    for descriptor in config["implementation"].values():
        path = str(descriptor["path"]).replace("\\", "/")
        if common.file_sha256(root / path) != str(descriptor["sha256"]).upper():
            raise ValueError(f"Stage58b implementation hash differs: {path}")
        paths.append(path)
    for value in config["inputs"].values():
        if isinstance(value, dict) and "path" in value:
            paths.append(str(value["path"]))
    paths.extend(
        manifest_paths(
            root, str(config["inputs"]["receptor_manifest"]["path"]), "receptor_pdbqt"
        )
    )
    paths.extend(
        manifest_paths(
            root, str(config["inputs"]["ligand_manifest"]["path"]), "pdbqt_path"
        )
    )
    normalized = sorted(set(path.replace("\\", "/") for path in paths))
    forbidden = ("fresh_validation", "locked_test", "data/protected")
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage58b bundle contains a protected panel path")
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
            "operation": "Stage58b PPARD Pilot-96 x Stage57-passing29 x three-seed Uni-Dock production bundle",
            "target_id": "PPARD",
            "experiment_class": "prospective outcome-blind development pilot",
            "receptor_count": 29,
            "ligand_count": 96,
            "active_count": 48,
            "decoy_count": 48,
            "seed_count": 3,
            "gpu_batch_count": 87,
            "gpu_pair_count": 8352,
            "profile_id": "enhanced",
            "exhaustiveness": 1024,
            "max_step": 80,
            "fresh_validation_rows": 0,
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
