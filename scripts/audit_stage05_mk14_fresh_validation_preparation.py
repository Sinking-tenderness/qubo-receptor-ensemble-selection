"""Audit fresh MAPK14 validation ligand preparation before remote docking."""

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


def by_ligand_id(
    rows: list[dict[str, str]], label: str
) -> dict[str, dict[str, str]]:
    output = {row["ligand_id"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{label} contains duplicate ligand IDs")
    return output


def audit_prepared_rows(
    panel_rows: list[dict[str, str]],
    three_d_rows: list[dict[str, str]],
    pdbqt_rows: list[dict[str, str]],
) -> dict[str, object]:
    panel = by_ligand_id(panel_rows, "fresh panel")
    three_d = by_ligand_id(three_d_rows, "3D manifest")
    pdbqt = by_ligand_id(pdbqt_rows, "PDBQT manifest")
    if set(panel) != set(three_d) or set(panel) != set(pdbqt):
        raise ValueError("fresh panel and preparation ligand IDs differ")
    for ligand_id, source in panel.items():
        prepared_3d = three_d[ligand_id]
        prepared_pdbqt = pdbqt[ligand_id]
        for prepared in (prepared_3d, prepared_pdbqt):
            if (
                prepared["label"] != source["label"]
                or prepared["split"] != "validation"
                or prepared["split_group_id"] != source["split_group_id"]
                or prepared["selection_role"]
                != "fresh_validation_preregistered"
            ):
                raise ValueError(f"prepared metadata differs: {ligand_id}")
        if prepared_3d["prep_status"] not in {"ok", "warning"}:
            raise ValueError(f"3D preparation failed: {ligand_id}")
        if prepared_pdbqt["pdbqt_status"] != "ok":
            raise ValueError(f"PDBQT preparation failed: {ligand_id}")
        path = Path(prepared_pdbqt["pdbqt_path"].replace("\\", "/"))
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != prepared_pdbqt["pdbqt_sha256"].upper():
            raise ValueError(f"PDBQT SHA-256 differs: {ligand_id}")
        if int(prepared_pdbqt.get("pdbqt_atom_count", 0)) <= 0:
            raise ValueError(f"PDBQT contains no atoms: {ligand_id}")
    return {
        "ligand_count": len(panel),
        "label_counts": dict(
            sorted(Counter(row["label"] for row in panel_rows).items())
        ),
        "selection_role_counts": dict(
            sorted(
                Counter(row["selection_role"] for row in panel_rows).items()
            )
        ),
        "unique_split_group_count": len(
            {row["split_group_id"] for row in panel_rows}
        ),
        "three_d_status_counts": dict(
            sorted(Counter(row["prep_status"] for row in three_d_rows).items())
        ),
        "pdbqt_status_counts": dict(
            sorted(
                Counter(row["pdbqt_status"] for row in pdbqt_rows).items()
            )
        ),
        "nonzero_formal_charge_count": sum(
            int(row.get("formal_charge", "0") or 0) != 0
            for row in pdbqt_rows
        ),
        "test_rows": 0,
    }


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = read_json(config_path)
    outputs = config["outputs"]
    panel_summary_path = Path(str(outputs["fresh_panel_summary_json"]))
    panel_path = Path(str(outputs["fresh_panel_csv"]))
    model_path = Path(str(outputs["frozen_model_artifact_json"]))
    receptor_path = Path(str(outputs["fixed_receptor_manifest_csv"]))
    three_d_path = Path(str(outputs["three_d_manifest_csv"]))
    pdbqt_path = Path(str(outputs["pdbqt_manifest_csv"]))
    summary_path = Path(str(outputs["preparation_summary_json"]))
    for path in (
        panel_summary_path,
        panel_path,
        model_path,
        receptor_path,
        three_d_path,
        pdbqt_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    panel_summary = read_json(panel_summary_path)
    if panel_summary.get("status") != "fresh_validation_panel_ok_preparation_pending":
        raise ValueError("fresh validation panel did not pass")
    panel_record = panel_summary["outputs"]["fresh_panel_csv"]
    if file_sha256(panel_path) != str(panel_record["sha256"]).upper():
        raise ValueError("fresh panel SHA-256 differs")
    audit = audit_prepared_rows(
        read_csv(panel_path), read_csv(three_d_path), read_csv(pdbqt_path)
    )
    receptors = read_csv(receptor_path)
    expected_receptors = int(config["docking"]["receptor_count"])
    if len(receptors) != expected_receptors or any(
        row["status"] != "ok" for row in receptors
    ):
        raise ValueError("fixed validation receptor manifest differs")

    if summary_path.exists() and not overwrite:
        raise FileExistsError("preparation summary exists; use --overwrite")
    summary = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": "fresh_validation_preparation_ok_remote_docking_pending",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "audit": audit,
        "fixed_receptor_count": len(receptors),
        "expected_receptor_ligand_pairs_per_seed": len(receptors)
        * int(audit["ligand_count"]),
        "expected_three_seed_vina_jobs": len(receptors)
        * int(audit["ligand_count"])
        * len(config["docking"]["base_seeds"]),
        "inputs": {
            "fresh_panel_csv": {
                "path": panel_path.as_posix(),
                "sha256": file_sha256(panel_path),
            },
            "three_d_manifest_csv": {
                "path": three_d_path.as_posix(),
                "sha256": file_sha256(three_d_path),
            },
            "pdbqt_manifest_csv": {
                "path": pdbqt_path.as_posix(),
                "sha256": file_sha256(pdbqt_path),
            },
            "fixed_receptor_manifest_csv": {
                "path": receptor_path.as_posix(),
                "sha256": file_sha256(receptor_path),
            },
            "frozen_model_artifact_json": {
                "path": model_path.as_posix(),
                "sha256": file_sha256(model_path),
            },
        },
        "validation_scores_read": 0,
        "test_rows": 0,
        "test_scores_read": 0,
        "next_gate": "Build and independently audit the deterministic CPU bundle before any validation docking starts.",
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
