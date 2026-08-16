"""Build the deterministic BACE1 Train-266 Uni-Dock production bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from scripts.experimental.unidock.run_stage42c_bace1_train266_production import (
        common,
        validate_inputs,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from scripts.experimental.unidock.run_stage42c_bace1_train266_production import (
        common,
        validate_inputs,
    )


CONFIG = "configs/stage42c_bace1_train266_unidock113_production.json"
FIXED_PATHS = (
    CONFIG,
    "configs/stage42b_bace1_train266_unidock_input_preparation.json",
    "environment/stage08_unidock_gpu.yml",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
    "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
    "scripts/experimental/unidock/run_stage09_mk14_train696_production.py",
    "scripts/experimental/unidock/audit_stage09_mk14_train696_production.py",
    "scripts/experimental/unidock/run_stage42c_bace1_train266_production.py",
    "scripts/experimental/unidock/audit_stage42c_bace1_train266_production.py",
    "scripts/experimental/unidock/run_stage42c_bace1_train266_production_remote.sh",
    "scripts/experimental/unidock/build_stage42c_bace1_train266_production_bundle.py",
    "reports/stage-42c/bace1_train266_unidock113_production_execution.md",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = common.read_json(root / CONFIG)
    receptors, ligands, audit = validate_inputs(root, config)
    if audit["validation_rows"] != 0 or audit["test_rows"] != 0:
        raise ValueError("Stage 42c bundle crossed a data boundary")
    if len(receptors) != 34 or len(ligands) != 266:
        raise ValueError("Stage 42c bundle input dimensions differ")
    if audit["expected_pair_count"] != 27132:
        raise ValueError("Stage 42c bundle pair count differs")

    paths = list(FIXED_PATHS)
    for descriptor in config["implementation"].values():
        path = str(descriptor["path"]).replace("\\", "/")
        if common.file_sha256(root / path) != str(descriptor["sha256"]).upper():
            raise ValueError(f"Stage 42c implementation hash differs: {path}")
        paths.append(path)
    for value in config["inputs"].values():
        if isinstance(value, dict) and "path" in value:
            paths.append(str(value["path"]))
    receptor_manifest = str(config["inputs"]["receptor_manifest"]["path"])
    ligand_manifest = str(config["inputs"]["ligand_manifest"]["path"])
    paths.extend(manifest_paths(root, receptor_manifest, "receptor_pdbqt"))
    paths.extend(manifest_paths(root, ligand_manifest, "pdbqt_path"))
    normalized = sorted(set(path.replace("\\", "/") for path in paths))
    forbidden = ("fresh_validation", "locked_test", "selected_ligand_panel_manifest")
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage 42c bundle contains a protected panel path")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, bundle_paths(root))
    result.update({
        "operation": "Stage 42c BACE1 Train-266 x redocking-qualified34 x three-seed Uni-Dock production bundle",
        "target_id": "BACE1",
        "experiment_class": "outcome_informed_posthoc_development",
        "stage41c_gate_status": "closed_failed_34_of_49_below_40_required",
        "receptor_count": 34,
        "ligand_count": 266,
        "active_count": 133,
        "decoy_count": 133,
        "seed_count": 3,
        "gpu_batch_count": 102,
        "gpu_pair_count": 27132,
        "profile_id": "enhanced",
        "exhaustiveness": 1024,
        "max_step": 80,
        "validation_rows": 0,
        "test_rows": 0,
        "gpu_required_for_execution": True,
    })
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
