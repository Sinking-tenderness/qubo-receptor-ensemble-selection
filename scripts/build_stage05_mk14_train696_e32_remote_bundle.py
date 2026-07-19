"""Build the deterministic Linux bundle for new-536 docking and train-696 merge."""

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


NEW_MANIFEST = (
    "data/processed/stage05_mk14_expanded_train_new_268a268d_pdbqt_manifest.csv"
)
FULL_MANIFEST = "data/processed/stage05_mk14_expanded_train_348a348d_pdbqt_manifest.csv"
REUSED_MANIFEST = "data/processed/stage05_mk14_expanded_train_80a80d_pdbqt_manifest.csv"
RECEPTOR_MANIFEST = "data/processed/stage05_mk14_expanded8_receptor_manifest.csv"

FIXED_PATHS = (
    "configs/stage05_mk14_expanded_train696_preregistration.json",
    "configs/stage05_mk14_expanded_train_e32_cpu2.txt",
    "configs/stage05_mk14_expanded8_train536new_e32_seed0_linux.json",
    "configs/stage05_mk14_expanded8_train536new_e32_seed1_linux.json",
    "configs/stage05_mk14_expanded8_train536new_e32_seed2_linux.json",
    "configs/stage05_mk14_expanded8_train536new_e32_seed_aggregation.json",
    "configs/stage05_mk14_expanded_train696_e32_merge.json",
    "configs/stage05_mk14_expanded8_train160_e32_seed0_linux.json",
    "configs/stage05_mk14_expanded8_train160_e32_seed1_linux.json",
    "configs/stage05_mk14_expanded8_train160_e32_seed2_linux.json",
    "configs/stage05_mk14_expanded8_train160_e32_seed_aggregation.json",
    RECEPTOR_MANIFEST,
    REUSED_MANIFEST,
    NEW_MANIFEST,
    FULL_MANIFEST,
    "data/stage05_mk14_expanded_train696_panel_summary.json",
    "data/stage05_mk14_expanded_train696_preparation_summary.json",
    "environment/bin/vina_1.2.7_linux_x86_64",
    "reports/stage-05/mk14_expanded_train696_preparation_record.md",
    "scripts/aggregate_seed_replicates.py",
    "scripts/batch_vina_docking.py",
    "scripts/batch_vina_docking_parallel.py",
    "scripts/build_score_matrix.py",
    "scripts/merge_stage05_mk14_train696_e32.py",
    "scripts/prepare_receptor.py",
    "scripts/run_md_receptor_ligand_benchmark.py",
    "scripts/run_stage05_mk14_train696_e32_remote.sh",
    "results/runs/stage05_mk14_expanded8_train160_e32_seed0_linux/summary.json",
    "results/runs/stage05_mk14_expanded8_train160_e32_seed0_linux/representative_scores.csv",
    "results/runs/stage05_mk14_expanded8_train160_e32_seed1_linux/summary.json",
    "results/runs/stage05_mk14_expanded8_train160_e32_seed1_linux/representative_scores.csv",
    "results/runs/stage05_mk14_expanded8_train160_e32_seed2_linux/summary.json",
    "results/runs/stage05_mk14_expanded8_train160_e32_seed2_linux/representative_scores.csv",
    "results/runs/stage05_mk14_expanded8_train160_e32_aggregated/aggregated_seed_scores.csv",
    "results/runs/stage05_mk14_expanded8_train160_e32_aggregated/primary_median_score_matrix.csv",
    "results/runs/stage05_mk14_expanded8_train160_e32_aggregated/sensitivity_minimum_score_matrix.csv",
    "results/runs/stage05_mk14_expanded8_train160_e32_aggregated/summary.json",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def rows_by_id(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    output = {row["ligand_id"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{label} contains duplicate ligand IDs")
    return output


def validate_manifests(root: Path) -> dict[str, object]:
    manifests = {
        "reused": read_csv(root / REUSED_MANIFEST),
        "new": read_csv(root / NEW_MANIFEST),
        "full": read_csv(root / FULL_MANIFEST),
    }
    expected_counts = {"reused": 160, "new": 536, "full": 696}
    expected_roles = {
        "reused": "development_train",
        "new": "development_train_expanded",
        "full": "development_train_expanded",
    }
    indexed: dict[str, dict[str, dict[str, str]]] = {}
    for name, rows in manifests.items():
        if len(rows) != expected_counts[name]:
            raise ValueError(
                f"{name} manifest expected {expected_counts[name]} rows, got {len(rows)}"
            )
        if {row.get("split") for row in rows} != {"train"}:
            raise ValueError(f"{name} manifest is not train-only")
        if {row.get("selection_role") for row in rows} != {expected_roles[name]}:
            raise ValueError(f"{name} manifest selection role differs")
        if any(row.get("pdbqt_status") != "ok" for row in rows):
            raise ValueError(f"{name} manifest contains a failed PDBQT row")
        indexed[name] = rows_by_id(rows, f"{name} manifest")

    reused_ids = set(indexed["reused"])
    new_ids = set(indexed["new"])
    if reused_ids.intersection(new_ids):
        raise ValueError("reused and new ligand IDs overlap")
    if reused_ids.union(new_ids) != set(indexed["full"]):
        raise ValueError("reused and new manifests do not form the full panel")
    if Counter(row["label"] for row in manifests["full"]) != Counter(
        {"active": 348, "decoy": 348}
    ):
        raise ValueError("full manifest label counts differ")

    for name in ("reused", "new"):
        for ligand_id, row in indexed[name].items():
            full_row = indexed["full"][ligand_id]
            common = set(row).intersection(full_row).difference({"selection_role"})
            if any(row[field] != full_row[field] for field in common):
                raise ValueError(f"{name} manifest differs for {ligand_id}")

    for ligand_id, row in indexed["new"].items():
        path = root / row["pdbqt_path"].replace("\\", "/")
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path).upper() != row["pdbqt_sha256"].upper():
            raise ValueError(f"new ligand PDBQT SHA-256 differs: {ligand_id}")
    return {
        "reused_ligand_count": len(reused_ids),
        "new_ligand_count": len(new_ids),
        "full_ligand_count": len(indexed["full"]),
        "full_label_counts": dict(Counter(row["label"] for row in manifests["full"])),
        "validation_rows": 0,
        "test_rows": 0,
    }


def bundle_paths(root: Path) -> list[str]:
    validate_manifests(root)
    receptor_rows = read_csv(root / RECEPTOR_MANIFEST)
    if len(receptor_rows) != 8 or any(row.get("status") != "ok" for row in receptor_rows):
        raise ValueError("receptor manifest is not the admitted eight-receptor panel")
    paths = list(FIXED_PATHS)
    paths.extend(manifest_paths(root, RECEPTOR_MANIFEST, "receptor_pdbqt"))
    paths.extend(manifest_paths(root, NEW_MANIFEST, "pdbqt_path"))
    return sorted(set(paths))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = validate_manifests(root)
    result = write_bundle(root, args.output, bundle_paths(root))
    result.update(audit)
    result["receptor_count"] = 8
    result["seed_count"] = 3
    result["reused_vina_jobs"] = 3840
    result["new_vina_jobs"] = 12864
    result["complete_seed_cells"] = 16704
    result["gpu_required"] = False
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
