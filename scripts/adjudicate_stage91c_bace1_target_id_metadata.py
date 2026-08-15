from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SOURCE = Path(
    "analysis/stage91c_bace1_result_20260812/results/runs/"
    "stage91c_bace1_chembl365_unidock113_production/scores.csv"
)
OUTPUT = Path(
    "results/runs/stage92a_bace1_target_id_metadata_adjudication/"
    "scores_target_id_amended.csv"
)
RESULT = Path("data/stage92a_bace1_target_id_metadata_adjudication_result.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(root: Path) -> dict[str, object]:
    root = root.resolve()
    source_path = root / SOURCE
    output_path = root / OUTPUT
    source_rows = read_rows(source_path)
    if len(source_rows) != 37230:
        raise ValueError("Stage91c source row count differs")
    if {row["target_id"] for row in source_rows} != {"MK14"}:
        raise ValueError("Stage91c source target-id defect differs from adjudication")
    if {row["receptor_id"].split("_", 1)[0] for row in source_rows} != {"BACE1"}:
        raise ValueError("Stage91c receptor identities are not exclusively BACE1")
    if {row["ligand_id"].split("_", 1)[0] for row in source_rows} != {"BACE1"}:
        raise ValueError("Stage91c ligand identities are not exclusively BACE1")

    amended_rows = [{**row, "target_id": "BACE1"} for row in source_rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(amended_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(amended_rows)

    reread = read_rows(output_path)
    non_target_columns = [key for key in source_rows[0] if key != "target_id"]
    changed_non_target_cells = sum(
        source[key] != amended[key]
        for source, amended in zip(source_rows, reread, strict=True)
        for key in non_target_columns
    )
    changed_target_cells = sum(
        source["target_id"] != amended["target_id"]
        for source, amended in zip(source_rows, reread, strict=True)
    )
    checks = {
        "source_rows_are_all_bace1_by_receptor_id": True,
        "source_rows_are_all_bace1_by_ligand_id": True,
        "source_target_id_is_uniformly_mk14": True,
        "amended_target_id_is_uniformly_bace1": {
            row["target_id"] for row in reread
        }
        == {"BACE1"},
        "row_count_preserved": len(reread) == len(source_rows) == 37230,
        "row_order_preserved": all(
            source["ligand_id"] == amended["ligand_id"]
            and source["receptor_id"] == amended["receptor_id"]
            and source["seed_id"] == amended["seed_id"]
            for source, amended in zip(source_rows, reread, strict=True)
        ),
        "only_target_id_cells_changed": changed_non_target_cells == 0
        and changed_target_cells == 37230,
        "scores_preserved_exactly": all(
            source["gpu_score"] == amended["gpu_score"]
            for source, amended in zip(source_rows, reread, strict=True)
        ),
    }
    result = {
        "schema_version": "1.0",
        "status": (
            "stage92a_bace1_target_id_metadata_adjudication_ok"
            if all(checks.values())
            else "stage92a_bace1_target_id_metadata_adjudication_failed"
        ),
        "issue": "The shared Uni-Dock batch helper hard-coded target_id=MK14 while all Stage91c receptor, ligand, path, and experiment identities were BACE1.",
        "scientific_effect": "metadata-only; no score, pose, receptor, ligand, seed, order, or aggregate matrix value changed",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "row_count": len(reread),
        "changed_target_id_cell_count": changed_target_cells,
        "changed_non_target_cell_count": changed_non_target_cells,
        "source": {"path": SOURCE.as_posix(), "sha256": sha256(source_path)},
        "output": {"path": OUTPUT.as_posix(), "sha256": sha256(output_path)},
        "data_boundary": {
            "confirmation_scores_read": 0,
            "locked_test_scores_read": 0,
            "new_docking_jobs": 0,
            "quantum_jobs": 0,
        },
    }
    result_path = root / RESULT
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failed_checks"]:
        raise SystemExit(1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
