"""Build the deterministic Stage 07c warning-adjudication remote bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from .run_stage07c_unidock_warning_adjudication import (
        file_sha256,
        read_json,
        validate_config,
        validate_inputs,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from run_stage07c_unidock_warning_adjudication import (
        file_sha256,
        read_json,
        validate_config,
        validate_inputs,
    )


CONFIG = "configs/stage07c_mk14_unidock113_warning_adjudication.json"
FIXED_PATHS = (
    CONFIG,
    "environment/stage07_unidock_gpu.yml",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/compare_receptor_screening.py",
    "scripts/evaluate_virtual_screening.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
    "scripts/experimental/unidock/prepare_stage07c_existing_evidence.py",
    "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
    "scripts/experimental/unidock/evaluate_stage07c_unidock_warning_adjudication.py",
    "scripts/experimental/unidock/build_stage07c_pose_diagnostics.py",
    "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication_remote.sh",
    "reports/stage-07/mk14_unidock113_warning_adjudication_execution.md",
    "data/processed/stage07c_mk14_unidock113_enhanced_seed012_scores.csv",
    "data/processed/stage07c_mk14_unidock113_enhanced_warning_replay_reference.csv",
    "data/stage07c_mk14_unidock113_existing_evidence_provenance.json",
)


def bundle_paths(root: Path) -> list[str]:
    config = read_json(root / CONFIG)
    validate_config(config)
    _, _, _, _, audit = validate_inputs(root.resolve(), config)
    if audit["validation_rows"] != 0 or audit["test_rows"] != 0:
        raise ValueError("Stage 07c bundle crossed a data boundary")
    if audit["macrocycle_closure_pseudoatom_ligand_count"] != 0:
        raise ValueError("Stage 07c bundle contains closure pseudoatoms")
    for descriptor in config["implementation"].values():
        path = root / str(descriptor["path"])
        if file_sha256(path) != str(descriptor["sha256"]):
            raise ValueError(f"Stage 07c implementation hash differs: {path}")
    receptor_manifest = str(config["inputs"]["receptor_manifest"]["path"])
    ligand_manifest = str(config["inputs"]["ligand_manifest"]["path"])
    paths = list(FIXED_PATHS)
    paths.extend(manifest_paths(root, receptor_manifest, "receptor_pdbqt"))
    paths.extend(manifest_paths(root, ligand_manifest, "pdbqt_path"))
    paths.extend((receptor_manifest, ligand_manifest))
    return sorted(set(path.replace("\\", "/") for path in paths))


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
            "operation": "Stage 07c consumed Train-160 warning-adjudication bundle",
            "profile_id": "enhanced",
            "receptor_count": 4,
            "ligand_count": 160,
            "combined_seed_count": 4,
            "new_seed_pair_count": 640,
            "warning_replay_pair_count": 160,
            "gpu_pair_count": 800,
            "fresh_validation_rows": 0,
            "test_rows": 0,
            "gpu_required_for_execution": True,
        }
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
