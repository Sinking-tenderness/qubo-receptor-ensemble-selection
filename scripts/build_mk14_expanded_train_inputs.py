"""Build audited eight-receptor and train-only MAPK14 docking manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object] | dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checked_record(record: dict[str, object]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def normalize_receptor_row(
    row: dict[str, str], source_pool: str
) -> dict[str, object]:
    status = row.get("status", row.get("preparation_status", ""))
    if status != "ok":
        raise ValueError(f"receptor preparation did not pass: {row.get('conformer_id')}")
    receptor_path = Path(row["receptor_pdbqt"].replace("\\", "/"))
    if not receptor_path.is_file():
        raise FileNotFoundError(receptor_path)
    receptor_hash = file_sha256(receptor_path)
    if receptor_hash != row["receptor_pdbqt_sha256"].upper():
        raise ValueError(f"receptor PDBQT hash differs: {row['conformer_id']}")
    input_path = row.get("input_pdb", row.get("aligned_pdb", "")).replace("\\", "/")
    input_hash = row.get("input_pdb_sha256", row.get("aligned_pdb_sha256", ""))
    return {
        "conformer_id": row["conformer_id"],
        "source_pool": source_pool,
        "input_structure": input_path,
        "input_structure_sha256": input_hash,
        "chain": row["chain"],
        "residue_count": row["residue_count"],
        "receptor_atom_count": row.get(
            "receptor_atom_count", row.get("pdbqt_atom_count", "")
        ),
        "hydrogen_like_atom_count": row["hydrogen_like_atom_count"],
        "autodock_atom_types": row["autodock_atom_types"],
        "charge_min": row["charge_min"],
        "charge_max": row["charge_max"],
        "receptor_pdbqt": receptor_path.as_posix(),
        "receptor_pdbqt_sha256": receptor_hash,
        "status": "ok",
    }


def run_build(config_path: Path, overwrite: bool) -> dict[str, object]:
    config = read_json(config_path)
    inputs = config.get("inputs")
    expected = config.get("expected")
    outputs = config.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(expected, dict) or not isinstance(outputs, dict):
        raise ValueError("input-build config is incomplete")
    paths = {
        key: checked_record(record)
        for key, record in inputs.items()
        if isinstance(record, dict)
    }
    required = {
        "existing_receptor_manifest",
        "new_receptor_manifest",
        "development_ligand_pdbqt_manifest",
        "frozen_train_manifest",
        "redocking_summary",
        "redocking_audit",
    }
    if set(paths) != required:
        raise ValueError("input-build inputs differ from the required set")
    redocking_summary = read_json(paths["redocking_summary"])
    redocking_audit = read_json(paths["redocking_audit"])
    if redocking_summary.get("status") != "expanded_redocking_gate_ok":
        raise ValueError("expanded redocking gate did not pass")
    if redocking_audit.get("status") != "independent_expanded_redocking_audit_ok":
        raise ValueError("expanded redocking audit did not pass")

    receptor_by_id: dict[str, dict[str, object]] = {}
    for source_pool, manifest_key in (
        ("existing_redocking_approved", "existing_receptor_manifest"),
        ("new_v3_structural_addition", "new_receptor_manifest"),
    ):
        for row in read_csv(paths[manifest_key]):
            normalized = normalize_receptor_row(row, source_pool)
            receptor_id = str(normalized["conformer_id"])
            if receptor_id in receptor_by_id:
                raise ValueError(f"duplicate receptor ID: {receptor_id}")
            receptor_by_id[receptor_id] = normalized
    expected_receptors = [str(value) for value in expected["receptor_ids"]]
    if set(receptor_by_id) != set(expected_receptors):
        raise ValueError("expanded receptor IDs differ from the frozen set")
    receptor_rows = [receptor_by_id[receptor_id] for receptor_id in expected_receptors]

    frozen_train_rows = read_csv(paths["frozen_train_manifest"])
    frozen_train_by_id = {row["ligand_id"]: row for row in frozen_train_rows}
    if len(frozen_train_by_id) != len(frozen_train_rows):
        raise ValueError("frozen train manifest contains duplicate ligand IDs")
    development_rows = read_csv(paths["development_ligand_pdbqt_manifest"])
    input_role_counts = Counter(row.get("selection_role", "") for row in development_rows)
    if input_role_counts != Counter(
        {
            "development_train": int(expected["train_ligand_count"]),
            "development_validation": int(expected["excluded_validation_ligand_count"]),
        }
    ):
        raise ValueError(f"development manifest role counts differ: {input_role_counts}")
    development_by_id = {row["ligand_id"]: row for row in development_rows}
    if len(development_by_id) != len(development_rows):
        raise ValueError("development PDBQT manifest contains duplicate ligand IDs")
    if set(frozen_train_by_id) - set(development_by_id):
        raise ValueError("a frozen train ligand is missing from the PDBQT manifest")

    train_rows: list[dict[str, str]] = []
    for ligand_id in sorted(frozen_train_by_id):
        frozen = frozen_train_by_id[ligand_id]
        prepared = development_by_id[ligand_id]
        if (
            frozen.get("split") != "train"
            or frozen.get("selection_role") != "development_train"
            or prepared.get("split") != "train"
            or prepared.get("selection_role") != "development_train"
            or frozen["label"] != prepared["label"]
        ):
            raise ValueError(f"train role or label differs: {ligand_id}")
        if prepared.get("pdbqt_status") != "ok":
            raise ValueError(f"ligand preparation did not pass: {ligand_id}")
        pdbqt_path = Path(prepared["pdbqt_path"].replace("\\", "/"))
        if not pdbqt_path.is_file():
            raise FileNotFoundError(pdbqt_path)
        pdbqt_hash = file_sha256(pdbqt_path)
        if pdbqt_hash != prepared["pdbqt_sha256"].upper():
            raise ValueError(f"ligand PDBQT hash differs: {ligand_id}")
        train_rows.append(
            {
                **prepared,
                "pdbqt_path": pdbqt_path.as_posix(),
                "pdbqt_sha256": pdbqt_hash,
            }
        )
    label_counts = Counter(row["label"] for row in train_rows)
    expected_labels = Counter(
        {key: int(value) for key, value in expected["train_label_counts"].items()}
    )
    if label_counts != expected_labels:
        raise ValueError(f"train label counts differ: {label_counts}")
    if any(row.get("selection_role") != "development_train" for row in train_rows):
        raise ValueError("a non-train role entered the output ligand manifest")
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("a non-train split entered the output ligand manifest")

    receptor_output = Path(str(outputs["receptor_manifest_csv"]))
    ligand_output = Path(str(outputs["train_ligand_manifest_csv"]))
    summary_output = Path(str(outputs["summary_json"]))
    if not overwrite and any(path.exists() for path in (receptor_output, ligand_output, summary_output)):
        raise FileExistsError("expanded train input outputs exist; use --overwrite")
    write_csv(receptor_output, receptor_rows)
    write_csv(ligand_output, train_rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "expanded_train_inputs_ok",
        "config": {"path": config_path.as_posix(), "sha256": file_sha256(config_path)},
        "receptor_count": len(receptor_rows),
        "receptor_ids": expected_receptors,
        "train_ligand_count": len(train_rows),
        "train_label_counts": dict(sorted(label_counts.items())),
        "output_selection_role_counts": {"development_train": len(train_rows)},
        "output_validation_rows": 0,
        "output_test_rows": 0,
        "source_validation_rows_excluded": input_role_counts[
            "development_validation"
        ],
        "outputs": {
            "receptor_manifest_csv": {
                "path": receptor_output.as_posix(),
                "sha256": file_sha256(receptor_output),
            },
            "train_ligand_manifest_csv": {
                "path": ligand_output.as_posix(),
                "sha256": file_sha256(ligand_output),
            },
        },
        "next_gate": "run three paired e16 seeds for 8 receptors x 160 development-train ligands using the revised common box",
        "interpretation_boundary": "This materializes docking inputs only. It does not read validation/test scores, calculate enrichment, or establish QUBO benefit.",
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_build(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
