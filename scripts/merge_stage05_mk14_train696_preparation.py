"""Audit and merge reused and newly prepared MAPK14 train696 ligands."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

try:
    from .prepare_receptor import file_sha256
except ImportError:
    from prepare_receptor import file_sha256


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_prepared(
    row: dict[str, str], panel_row: dict[str, str], role: str
) -> dict[str, str]:
    ligand_id = panel_row["ligand_id"]
    if (
        row["ligand_id"] != ligand_id
        or row["label"] != panel_row["label"]
        or row.get("split") != "train"
        or panel_row["split"] != "train"
        or row.get("pdbqt_status") != "ok"
    ):
        raise ValueError(f"prepared ligand metadata differs: {ligand_id}")
    path = Path(row["pdbqt_path"].replace("\\", "/"))
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = file_sha256(path)
    if digest != row["pdbqt_sha256"].upper():
        raise ValueError(f"prepared ligand hash differs: {ligand_id}")
    if int(row.get("pdbqt_atom_count", 0)) <= 0:
        raise ValueError(f"prepared ligand has no PDBQT atoms: {ligand_id}")
    return {
        **row,
        "selection_role": role,
        "pdbqt_path": path.as_posix(),
        "pdbqt_sha256": digest,
    }


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = read_json(config_path)
    inputs = config["inputs"]
    outputs = config["outputs"]
    selection = config["panel_selection"]
    assert isinstance(inputs, dict)
    assert isinstance(outputs, dict)
    assert isinstance(selection, dict)
    panel_path = Path(str(outputs["expanded_panel_csv"]))
    new_panel_path = Path(str(outputs["new_ligands_csv"]))
    new_prepared_path = Path(str(outputs["new_pdbqt_manifest_csv"]))
    existing_prepared_path = Path(str(inputs["existing_prepared_manifest"]["path"]))
    for path in (panel_path, new_panel_path, new_prepared_path, existing_prepared_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_sha256(existing_prepared_path) != str(
        inputs["existing_prepared_manifest"]["sha256"]
    ).upper():
        raise ValueError("existing prepared-manifest SHA-256 differs")
    panel_rows = read_csv(panel_path)
    panel_by_id = {row["ligand_id"]: row for row in panel_rows}
    new_panel_rows = read_csv(new_panel_path)
    new_ids = {row["ligand_id"] for row in new_panel_rows}
    existing_rows = read_csv(existing_prepared_path)
    new_rows = read_csv(new_prepared_path)
    if len(panel_by_id) != int(selection["target_total_count"]):
        raise ValueError("expanded panel count differs")
    if len(new_ids) != int(selection["new_total_count"]):
        raise ValueError("new panel count differs")
    existing_by_id = {row["ligand_id"]: row for row in existing_rows}
    new_by_id = {row["ligand_id"]: row for row in new_rows}
    if set(existing_by_id) & new_ids:
        raise ValueError("a reused ligand was prepared again")
    if set(new_by_id) != new_ids:
        raise ValueError("new preparation and panel IDs differ")
    if set(existing_by_id) | new_ids != set(panel_by_id):
        raise ValueError("combined prepared IDs do not cover the panel")
    role = str(selection["selection_role"])
    combined = [
        validate_prepared(
            existing_by_id.get(ligand_id, new_by_id.get(ligand_id, {})),
            panel_by_id[ligand_id],
            role,
        )
        for ligand_id in sorted(panel_by_id)
    ]
    if Counter(row["label"] for row in combined) != Counter(
        {
            "active": int(selection["target_active_count"]),
            "decoy": int(selection["target_decoy_count"]),
        }
    ):
        raise ValueError("combined prepared label counts differ")
    if any(row["selection_role"] != role for row in combined):
        raise ValueError("combined prepared selection role differs")

    combined_path = Path(str(outputs["combined_pdbqt_manifest_csv"]))
    summary_path = Path(str(outputs["preparation_summary_json"]))
    if not overwrite and (combined_path.exists() or summary_path.exists()):
        raise FileExistsError("combined preparation outputs exist; use --overwrite")
    write_csv(combined_path, combined)
    summary = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": "expanded_train696_preparation_ok_remote_docking_pending",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "panel": {
            "ligand_count": len(combined),
            "label_counts": dict(sorted(Counter(row["label"] for row in combined).items())),
            "selection_role_counts": dict(
                sorted(Counter(row["selection_role"] for row in combined).items())
            ),
            "reused_prepared_ligands": len(existing_by_id),
            "new_prepared_ligands": len(new_by_id),
            "pdbqt_status_counts": dict(
                sorted(Counter(row["pdbqt_status"] for row in combined).items())
            ),
            "sdf_preparation_status_counts": dict(
                sorted(Counter(row["prep_status"] for row in combined).items())
            ),
            "nonzero_formal_charge_count": sum(
                int(row.get("formal_charge", "0") or 0) != 0 for row in combined
            ),
            "validation_rows": 0,
            "test_rows": 0,
        },
        "inputs": {
            "expanded_panel_csv": {
                "path": panel_path.as_posix(),
                "sha256": file_sha256(panel_path),
            },
            "existing_prepared_manifest": {
                "path": existing_prepared_path.as_posix(),
                "sha256": file_sha256(existing_prepared_path),
            },
            "new_prepared_manifest": {
                "path": new_prepared_path.as_posix(),
                "sha256": file_sha256(new_prepared_path),
            },
        },
        "output": {
            "combined_pdbqt_manifest_csv": {
                "path": combined_path.as_posix(),
                "sha256": file_sha256(combined_path),
            }
        },
        "next_gate": config["next_gate"],
        "interpretation_note": config["interpretation_boundary"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
