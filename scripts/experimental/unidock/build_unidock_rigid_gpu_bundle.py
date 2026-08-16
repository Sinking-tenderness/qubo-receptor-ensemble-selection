"""Build the deterministic rigid-macrocycle Train-160 GPU diagnostic bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from .run_unidock_gpu_equivalence import read_json, validate_inputs
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from run_unidock_gpu_equivalence import read_json, validate_inputs


CONFIGS = (
    "configs/stage05_mk14_unidock_rigid_train160_detail_gpu_equivalence.json",
    "configs/stage05_mk14_unidock_rigid_train160_enhanced_gpu_equivalence.json",
)
FIXED_PATHS = (
    "environment/stage05_unidock_gpu.yml",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/audit_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/run_unidock_rigid_gpu_diagnostics_remote.sh",
    "reports/stage-05/mk14_unidock_gpu_equivalence_pilot.md",
)


def bundle_paths(root: Path) -> list[str]:
    paths = list(FIXED_PATHS) + list(CONFIGS)
    common_receptor_manifest: str | None = None
    common_ligand_manifest: str | None = None
    for config_value in CONFIGS:
        config = read_json(root / config_value)
        _, _, audit = validate_inputs(root.resolve(), config)
        if audit["validation_rows"] != 0 or audit["test_rows"] != 0:
            raise ValueError("rigid GPU diagnostic crossed a data boundary")
        if audit["macrocycle_closure_pseudoatom_ligand_count"] != 0:
            raise ValueError("rigid GPU diagnostic contains closure pseudoatoms")
        receptor_manifest = str(config["inputs"]["receptor_manifest"]["path"])
        ligand_manifest = str(config["inputs"]["ligand_manifest"]["path"])
        if common_receptor_manifest not in (None, receptor_manifest):
            raise ValueError("GPU diagnostic receptor manifests differ")
        if common_ligand_manifest not in (None, ligand_manifest):
            raise ValueError("GPU diagnostic ligand manifests differ")
        common_receptor_manifest = receptor_manifest
        common_ligand_manifest = ligand_manifest
        paths.extend(
            str(seed[key])
            for seed in config["inputs"]["cpu_seed_runs"]
            for key in ("summary_path", "scores_path")
        )
    if common_receptor_manifest is None or common_ligand_manifest is None:
        raise ValueError("GPU diagnostic manifests are missing")
    paths.extend(
        manifest_paths(root, common_receptor_manifest, "receptor_pdbqt")
    )
    paths.extend(manifest_paths(root, common_ligand_manifest, "pdbqt_path"))
    paths.extend([common_receptor_manifest, common_ligand_manifest])
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
            "operation": "consumed Train-160 rigid-macrocycle Uni-Dock detail and enhanced-search diagnostic bundle",
            "config_count": len(CONFIGS),
            "receptor_count": 5,
            "ligand_count": 160,
            "seed_count": 3,
            "gpu_pairs_per_profile": 2400,
            "validation_rows": 0,
            "test_rows": 0,
            "fresh_validation_scores_included": False,
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
