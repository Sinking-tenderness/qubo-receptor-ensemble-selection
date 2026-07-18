"""Build a deterministic Linux bundle for expanded MAPK14 train docking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
except ImportError:
    from build_stage05_mk14_remote_bundle import manifest_paths, write_bundle


FIXED_PATHS = (
    "configs/stage05_mk14_expanded_train_docking_preregistration.json",
    "configs/stage05_mk14_expanded_train_input_build.json",
    "configs/stage05_mk14_expanded_train_e16_cpu2.txt",
    "configs/stage05_mk14_expanded8_train160_e16_seed0_linux.json",
    "configs/stage05_mk14_expanded8_train160_e16_seed1_linux.json",
    "configs/stage05_mk14_expanded8_train160_e16_seed2_linux.json",
    "configs/stage05_mk14_expanded8_train160_seed_aggregation.json",
    "data/processed/stage05_mk14_expanded8_receptor_manifest.csv",
    "data/processed/stage05_mk14_expanded_train_80a80d_pdbqt_manifest.csv",
    "data/stage05_mk14_expanded_train_input_summary.json",
    "data/stage05_mk14_expanded_redocking_v4_summary.json",
    "data/stage05_mk14_expanded_redocking_v4_audit.json",
    "environment/bin/vina_1.2.7_linux_x86_64",
    "scripts/run_md_receptor_ligand_benchmark.py",
    "scripts/batch_vina_docking.py",
    "scripts/batch_vina_docking_parallel.py",
    "scripts/build_score_matrix.py",
    "scripts/prepare_receptor.py",
    "scripts/aggregate_seed_replicates.py",
    "scripts/run_stage05_mk14_expanded_train_remote.sh",
    "reports/stage-05/mk14_expanded_eight_receptor_redocking_record.md",
)


def validate_train_only_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 160:
        raise ValueError(f"expanded train bundle requires 160 ligands, got {len(rows)}")
    roles = {row.get("selection_role", "") for row in rows}
    splits = {row.get("split", "") for row in rows}
    if roles != {"development_train"} or splits != {"train"}:
        raise ValueError(f"bundle ligand roles are not train-only: {roles}, {splits}")
    if any(row.get("selection_role") != "development_train" for row in rows):
        raise ValueError("a prohibited ligand role entered the bundle")
    return rows


def expanded_train_bundle_paths(root: Path) -> list[str]:
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
    result = write_bundle(
        args.root,
        args.output,
        expanded_train_bundle_paths(args.root),
    )
    result["ligand_role"] = "development_train_only"
    result["validation_rows"] = 0
    result["test_rows"] = 0
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
