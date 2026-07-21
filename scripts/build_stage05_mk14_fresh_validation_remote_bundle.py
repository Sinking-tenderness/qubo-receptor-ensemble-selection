"""Build the deterministic CPU bundle for fresh MAPK14 validation docking."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

try:
    from .build_stage05_mk14_remote_bundle import (
        file_sha256,
        manifest_paths,
        write_bundle,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import (
        file_sha256,
        manifest_paths,
        write_bundle,
    )


RECEPTOR_MANIFEST = (
    "data/processed/stage05_mk14_fresh_validation_receptor_manifest.csv"
)
LIGAND_MANIFEST = (
    "data/processed/stage05_mk14_fresh_validation_pdbqt_manifest.csv"
)
FIXED_PATHS = (
    "configs/stage05_mk14_fresh_validation_preregistration.json",
    "configs/stage05_mk14_fresh_validation_execution_amendment01.json",
    "configs/stage05_mk14_fresh_validation_e32_seed0_linux.json",
    "configs/stage05_mk14_fresh_validation_e32_seed1_linux.json",
    "configs/stage05_mk14_fresh_validation_e32_seed2_linux.json",
    "configs/stage05_mk14_fresh_validation_e32_seed_aggregation.json",
    "configs/stage05_mk14_expanded_train_e32_cpu2.txt",
    "data/processed/stage05_mk14_fresh_validation_panel.csv",
    RECEPTOR_MANIFEST,
    LIGAND_MANIFEST,
    "data/stage05_mk14_fresh_validation_panel_summary.json",
    "data/stage05_mk14_fresh_validation_frozen_model.json",
    "data/stage05_mk14_fresh_validation_preparation_summary.json",
    "environment/bin/vina_1.2.7_linux_x86_64",
    "scripts/aggregate_seed_replicates.py",
    "scripts/batch_vina_docking.py",
    "scripts/batch_vina_docking_parallel.py",
    "scripts/build_score_matrix.py",
    "scripts/evaluate_stage05_mk14_fresh_validation.py",
    "scripts/prepare_receptor.py",
    "scripts/run_md_receptor_ligand_benchmark.py",
    "scripts/run_stage05_mk14_fresh_validation_remote.sh",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def validate_manifests(root: Path) -> dict[str, object]:
    receptors = read_csv(root / RECEPTOR_MANIFEST)
    ligands = read_csv(root / LIGAND_MANIFEST)
    if len(receptors) != 5 or any(row["status"] != "ok" for row in receptors):
        raise ValueError("fresh validation receptor manifest differs")
    if len(ligands) != 1576:
        raise ValueError("fresh validation ligand count differs")
    if Counter(row["label"] for row in ligands) != Counter(
        {"active": 75, "decoy": 1501}
    ):
        raise ValueError("fresh validation label counts differ")
    if {row["split"] for row in ligands} != {"validation"}:
        raise ValueError("fresh validation manifest contains a non-validation row")
    if {row["selection_role"] for row in ligands} != {
        "fresh_validation_preregistered"
    }:
        raise ValueError("fresh validation selection role differs")
    if any(row["pdbqt_status"] != "ok" for row in ligands):
        raise ValueError("fresh validation manifest contains failed PDBQT")
    if len({row["ligand_id"] for row in ligands}) != len(ligands):
        raise ValueError("fresh validation manifest contains duplicate ligand IDs")
    for manifest, column, hash_column, identifier in (
        (receptors, "receptor_pdbqt", "receptor_pdbqt_sha256", "conformer_id"),
        (ligands, "pdbqt_path", "pdbqt_sha256", "ligand_id"),
    ):
        for row in manifest:
            path = root / row[column].replace("\\", "/")
            if not path.is_file():
                raise FileNotFoundError(path)
            if file_sha256(path).upper() != row[hash_column].upper():
                raise ValueError(f"prepared file hash differs: {row[identifier]}")
    return {
        "receptor_count": len(receptors),
        "ligand_count": len(ligands),
        "label_counts": dict(Counter(row["label"] for row in ligands)),
        "receptor_ligand_pairs_per_seed": len(receptors) * len(ligands),
        "seed_count": 3,
        "total_vina_jobs": len(receptors) * len(ligands) * 3,
        "test_rows": 0,
    }


def bundle_paths(root: Path) -> list[str]:
    validate_manifests(root)
    paths = list(FIXED_PATHS)
    paths.extend(manifest_paths(root, RECEPTOR_MANIFEST, "receptor_pdbqt"))
    paths.extend(manifest_paths(root, LIGAND_MANIFEST, "pdbqt_path"))
    return sorted(set(paths))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = validate_manifests(root)
    result = write_bundle(root, args.output, bundle_paths(root))
    result.update(audit)
    result["gpu_required"] = False
    result["validation_metrics_calculated"] = False
    result["test_scores_read"] = 0
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
