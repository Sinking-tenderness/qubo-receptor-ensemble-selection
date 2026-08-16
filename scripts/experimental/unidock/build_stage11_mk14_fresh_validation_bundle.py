"""Build the deterministic Stage 11 fresh-validation Uni-Dock input bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from .prepare_stage11_mk14_fresh_validation_inputs import validate_source_inputs
    from .run_unidock_gpu_equivalence import file_sha256, read_json
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from scripts.experimental.unidock.prepare_stage11_mk14_fresh_validation_inputs import (
        validate_source_inputs,
    )
    from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
        file_sha256,
        read_json,
    )


CONFIG = "configs/stage11_mk14_fresh_validation_unidock113_confirmation.json"
FIXED_PATHS = (
    CONFIG,
    "environment/stage08_unidock_gpu.yml",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/batch_prepare_ligand_pdbqt.py",
    "scripts/compare_receptor_screening.py",
    "scripts/evaluate_virtual_screening.py",
    "scripts/prepare_receptor.py",
    "scripts/select_receptor_baselines.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/prepare_stage09_mk14_train696_inputs.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
    "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
    "scripts/experimental/unidock/run_stage09_mk14_train696_production.py",
    "scripts/experimental/unidock/audit_stage09_mk14_train696_production.py",
    "scripts/experimental/unidock/prepare_stage11_mk14_fresh_validation_inputs.py",
    "scripts/experimental/unidock/run_stage11_mk14_fresh_validation_confirmation.py",
    "scripts/experimental/unidock/audit_stage11_mk14_fresh_validation_confirmation.py",
    "scripts/experimental/unidock/evaluate_stage11_mk14_fresh_validation_confirmation.py",
    "scripts/experimental/unidock/build_stage11_mk14_fresh_validation_bundle.py",
    "scripts/experimental/unidock/run_stage11_mk14_fresh_validation_remote.sh",
    "reports/stage-11/mk14_fresh_validation_unidock113_execution.md",
    "reports/stage-10/mk14_expanded16_qubo_greedy_screen_adjudication.md",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = read_json(root / CONFIG)
    receptors, ligands, macrocycles, audit = validate_source_inputs(root, config)
    if len(receptors) != 6 or len(ligands) != 1576 or len(macrocycles) != 54:
        raise ValueError("Stage 11 bundle source dimensions differ")
    if audit["train_score_rows"] != 0 or audit["test_rows"] != 0:
        raise ValueError("Stage 11 bundle crossed a data boundary")

    paths = list(FIXED_PATHS)
    for descriptor in dict(config["implementation"]).values():
        path = str(descriptor["path"]).replace("\\", "/")
        if file_sha256(root / path) != str(descriptor["sha256"]).upper():
            raise ValueError(f"Stage 11 implementation hash differs: {path}")
        paths.append(path)
    for value in dict(config["inputs"]).values():
        if isinstance(value, dict) and "path" in value:
            paths.append(str(value["path"]))

    paths.extend(row["receptor_pdbqt"] for row in receptors)
    source_ligand_manifest = str(
        dict(config["inputs"])["source_ligand_manifest"]["path"]
    )
    paths.extend(manifest_paths(root, source_ligand_manifest, "pdbqt_path"))
    paths.extend(row["sdf_path"] for row in macrocycles)
    normalized = sorted(set(path.replace("\\", "/") for path in paths))
    forbidden = (
        "locked_test",
        "fresh_validation_e32_",
        "fresh_validation_result.json",
        "frozen_method_metrics.csv",
        "normalized_method_scores.csv",
        "enopt_xgboost",
    )
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage 11 bundle contains a forbidden score/result path")
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
            "operation": "Stage 11 frozen six-receptor fresh-validation Uni-Dock bundle",
            "receptor_count": 6,
            "ligand_count": 1576,
            "seed_count": 3,
            "gpu_batch_count": 18,
            "gpu_pair_count": 28368,
            "profile_id": "enhanced",
            "exhaustiveness": 1024,
            "max_step": 80,
            "rigid_macrocycle_ligand_count": 54,
            "validation_rows": 1576,
            "train_score_rows": 0,
            "test_rows": 0,
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
