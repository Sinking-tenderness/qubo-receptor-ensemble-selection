"""Build a deterministic Linux bundle for the complete MAPK14 e32 rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .build_stage05_mk14_expanded_train_remote_bundle import (
        validate_train_only_manifest,
    )
    from .build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
except ImportError:
    from build_stage05_mk14_expanded_train_remote_bundle import (
        validate_train_only_manifest,
    )
    from build_stage05_mk14_remote_bundle import manifest_paths, write_bundle


FIXED_PATHS = (
    "configs/stage05_mk14_expanded_train_docking_preregistration.json",
    "configs/stage05_mk14_expanded_train_search_protocol_amendment01.json",
    "configs/stage05_mk14_expanded_train_e32_cpu2.txt",
    "configs/stage05_mk14_expanded8_train160_e32_seed0_linux.json",
    "configs/stage05_mk14_expanded8_train160_e32_seed1_linux.json",
    "configs/stage05_mk14_expanded8_train160_e32_seed2_linux.json",
    "configs/stage05_mk14_expanded8_train160_e32_seed_aggregation.json",
    "data/processed/stage05_mk14_expanded8_receptor_manifest.csv",
    "data/processed/stage05_mk14_expanded_train_80a80d_pdbqt_manifest.csv",
    "data/stage05_mk14_expanded_e32_diagnostic_audit_summary.json",
    "environment/bin/vina_1.2.7_linux_x86_64",
    "scripts/aggregate_seed_replicates.py",
    "scripts/audit_stage05_development_matrix.py",
    "scripts/audit_stage05_expanded_train_matrix.py",
    "scripts/audit_stage05_expanded_e32_matrix.py",
    "scripts/batch_vina_docking.py",
    "scripts/batch_vina_docking_parallel.py",
    "scripts/build_score_matrix.py",
    "scripts/prepare_receptor.py",
    "scripts/run_md_receptor_ligand_benchmark.py",
    "scripts/run_stage05_mk14_expanded_train_e32_remote.sh",
)


def e32_bundle_paths(root: Path) -> list[str]:
    ligand_manifest = root / (
        "data/processed/stage05_mk14_expanded_train_80a80d_pdbqt_manifest.csv"
    )
    validate_train_only_manifest(ligand_manifest)
    paths = list(FIXED_PATHS)
    paths.extend(
        manifest_paths(
            root,
            "data/processed/stage05_mk14_expanded8_receptor_manifest.csv",
            "receptor_pdbqt",
        )
    )
    paths.extend(
        manifest_paths(
            root,
            "data/processed/stage05_mk14_expanded_train_80a80d_pdbqt_manifest.csv",
            "pdbqt_path",
        )
    )
    return sorted(set(paths))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(args.root, args.output, e32_bundle_paths(args.root))
    result["receptor_count"] = 8
    result["ligand_count"] = 160
    result["seed_count"] = 3
    result["expected_vina_runs"] = 3840
    result["validation_rows"] = 0
    result["test_rows"] = 0
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
