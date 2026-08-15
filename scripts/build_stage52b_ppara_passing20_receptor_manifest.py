"""Freeze the Stage 51-passing PPARA receptor manifest for Stage 52b."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PREREGISTRATION = Path(
    "configs/stage52_ppara_posthoc_exploratory_development_preregistration.json"
)
SOURCE_MANIFEST = Path(
    "data/processed/stage50_ppara_large_pool_prepared_receptor_manifest.csv"
)
GATE_RESULTS = Path(
    "data/processed/stage51_ppara_large_pool_receptor_gate_results.csv"
)
OUTPUT_MANIFEST = Path(
    "data/processed/stage52b_ppara_stage51_passing20_receptor_manifest.csv"
)
OUTPUT_SUMMARY = Path(
    "data/stage52b_ppara_stage51_passing20_receptor_manifest_summary.json"
)
SOURCE_MANIFEST_SHA256 = (
    "0F84A51286B7084C526F8824011637F55AD4D49C0FF0B040FB92182CFAB512A8"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def run(root: Path) -> dict[str, object]:
    root = root.resolve()
    prereg_path = root / PREREGISTRATION
    source_path = root / SOURCE_MANIFEST
    gate_path = root / GATE_RESULTS
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if file_sha256(source_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Stage 50 prepared receptor manifest hash differs")
    gate_descriptor = prereg["inputs"]["stage51_gate_results"]
    if file_sha256(gate_path) != gate_descriptor["sha256"]:
        raise ValueError("Stage 51 receptor gate hash differs")

    expected_ids = list(prereg["frozen_receptors"]["receptor_ids"])
    source_by_id = {row["conformer_id"]: row for row in read_csv(source_path)}
    gate_rows = read_csv(gate_path)
    passing_rows = [row for row in gate_rows if row["gate_pass"] == "True"]
    passing_ids = [row["conformer_id"] for row in passing_rows]
    if passing_ids != expected_ids:
        raise ValueError("Stage 51 passing receptor order differs from Stage 52 freeze")

    rows: list[dict[str, str]] = []
    for gate in passing_rows:
        receptor_id = gate["conformer_id"]
        if receptor_id not in source_by_id:
            raise ValueError(f"missing prepared receptor: {receptor_id}")
        source = dict(source_by_id[receptor_id])
        if source["status"] != "ok":
            raise ValueError(f"prepared receptor did not pass: {receptor_id}")
        receptor_path = root / source["receptor_pdbqt"]
        if not receptor_path.is_file():
            raise ValueError(f"prepared receptor file is missing: {receptor_id}")
        if file_sha256(receptor_path) != source["receptor_pdbqt_sha256"]:
            raise ValueError(f"prepared receptor hash differs: {receptor_id}")
        source.update(
            {
                "stage51_seed_count": gate["seed_count"],
                "stage51_successful_seed_count": gate["successful_seed_count"],
                "stage51_median_top_ranked_rmsd_angstrom": gate[
                    "median_top_ranked_rmsd_angstrom"
                ],
                "stage51_maximum_top_ranked_rmsd_angstrom": gate[
                    "maximum_top_ranked_rmsd_angstrom"
                ],
                "stage51_gate_pass": gate["gate_pass"],
            }
        )
        rows.append(source)

    if len(rows) != 20 or len({row["conformer_id"] for row in rows}) != 20:
        raise ValueError("Stage 52b receptor dimensions differ")
    output_path = root / OUTPUT_MANIFEST
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    stable = sum(int(row["stage51_successful_seed_count"]) == 3 for row in rows)
    two_of_three = sum(int(row["stage51_successful_seed_count"]) == 2 for row in rows)
    result = {
        "schema_version": "1.0",
        "status": "stage52b_ppara_passing20_receptor_manifest_ok",
        "operation": "deterministic filtering only; no docking or outcome optimization was run",
        "experiment_class": "post-hoc exploratory development-only",
        "inputs": {
            "preregistration": {
                "path": PREREGISTRATION.as_posix(),
                "sha256": file_sha256(prereg_path),
            },
            "source_receptor_manifest": {
                "path": SOURCE_MANIFEST.as_posix(),
                "sha256": file_sha256(source_path),
            },
            "stage51_gate_results": {
                "path": GATE_RESULTS.as_posix(),
                "sha256": file_sha256(gate_path),
            },
        },
        "receptor_count": len(rows),
        "stable_three_of_three_count": stable,
        "two_of_three_count": two_of_three,
        "receptor_ids": [row["conformer_id"] for row in rows],
        "output": {
            "path": relative_path(root, output_path),
            "sha256": file_sha256(output_path),
        },
        "data_boundary": {
            "train_ligand_rows_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "docking_jobs_started": 0,
        },
        "decision_boundary": (
            "This manifest freezes the outcome-informed Stage51-passing PPARA pool "
            "for post-hoc development only and does not change the failed Stage51 gate."
        ),
    }
    summary_path = root / OUTPUT_SUMMARY
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
