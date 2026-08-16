"""Build the deterministic PPARA Train-374 Uni-Dock production bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from scripts.experimental.unidock.run_stage52b_ppara_train374_production import (
        common,
        validate_inputs,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from scripts.experimental.unidock.run_stage52b_ppara_train374_production import (
        common,
        validate_inputs,
    )


CONFIG = "configs/stage52b_ppara_train374_unidock113_production.json"
FIXED_PATHS = (
    CONFIG,
    "configs/stage52_ppara_posthoc_exploratory_development_preregistration.json",
    "configs/stage52a_ppara_train374_unidock_input_preparation.json",
    "data/processed/stage49_ppara_train374_ligand_manifest.csv",
    "data/processed/stage50_ppara_large_pool_prepared_receptor_manifest.csv",
    "data/processed/stage51_ppara_large_pool_receptor_gate_results.csv",
    "data/stage49_ppara_ligand_panel_allocation_summary.json",
    "data/stage51b_ppara_redocking_bias_diagnostic_result.json",
    "data/stage51b_ppara_redocking_bias_diagnostic_audit.json",
    "environment/stage08_unidock_gpu.yml",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/build_stage52b_ppara_passing20_receptor_manifest.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
    "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
    "scripts/experimental/unidock/run_stage09_mk14_train696_production.py",
    "scripts/experimental/unidock/audit_stage09_mk14_train696_production.py",
    "scripts/experimental/unidock/run_stage52b_ppara_train374_production.py",
    "scripts/experimental/unidock/audit_stage52b_ppara_train374_production.py",
    "scripts/experimental/unidock/run_stage52b_ppara_train374_production_remote.sh",
    "scripts/experimental/unidock/build_stage52b_ppara_train374_production_bundle.py",
    "reports/stage-52b/ppara_train374_unidock113_production_execution.md",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = common.read_json(root / CONFIG)
    receptors, ligands, audit = validate_inputs(root, config)
    if audit["fresh_validation_rows"] != 0 or audit["locked_test_rows"] != 0:
        raise ValueError("Stage 52b bundle crossed a data boundary")
    if len(receptors) != 20 or len(ligands) != 374:
        raise ValueError("Stage 52b bundle input dimensions differ")
    if audit["expected_pair_count"] != 22440:
        raise ValueError("Stage 52b bundle pair count differs")

    paths = list(FIXED_PATHS)
    for descriptor in config["implementation"].values():
        path = str(descriptor["path"]).replace("\\", "/")
        if common.file_sha256(root / path) != str(descriptor["sha256"]).upper():
            raise ValueError(f"Stage 52b implementation hash differs: {path}")
        paths.append(path)
    for value in config["inputs"].values():
        if isinstance(value, dict) and "path" in value:
            paths.append(str(value["path"]))
    receptor_manifest = str(config["inputs"]["receptor_manifest"]["path"])
    ligand_manifest = str(config["inputs"]["ligand_manifest"]["path"])
    paths.extend(manifest_paths(root, receptor_manifest, "receptor_pdbqt"))
    paths.extend(manifest_paths(root, ligand_manifest, "pdbqt_path"))
    normalized = sorted(set(path.replace("\\", "/") for path in paths))
    forbidden = ("fresh_validation", "locked_test", "data/protected")
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage 52b bundle contains a protected panel path")
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
            "operation": (
                "Stage 52b PPARA Train-374 x Stage51-passing20 x three-seed "
                "Uni-Dock production bundle"
            ),
            "target_id": "PPARA",
            "experiment_class": "post-hoc exploratory development-only",
            "stage51_gate_status": "closed_failed_20_of_64_below_24_required",
            "receptor_count": 20,
            "ligand_count": 374,
            "active_count": 187,
            "decoy_count": 187,
            "seed_count": 3,
            "gpu_batch_count": 60,
            "gpu_pair_count": 22440,
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
