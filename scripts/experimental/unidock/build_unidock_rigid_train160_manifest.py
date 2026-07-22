"""Build the fixed Train-160 manifest with four rigid-macrocycle replacements."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

try:
    from .run_unidock_gpu_equivalence import (
        file_sha256,
        macrocycle_closure_atom_types,
    )
except ImportError:
    from run_unidock_gpu_equivalence import (
        file_sha256,
        macrocycle_closure_atom_types,
    )


PDBQT_FIELDS = (
    "pdbqt_status",
    "pdbqt_message",
    "pdbqt_path",
    "pdbqt_atom_count",
    "pdbqt_atom_types",
    "pdbqt_charge_min",
    "pdbqt_charge_max",
    "torsdof",
    "pdbqt_sha256",
)


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def verified_csv(descriptor: dict[str, object]) -> tuple[Path, list[dict[str, str]]]:
    path = Path(str(descriptor["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if observed != str(descriptor["sha256"]).upper():
        raise ValueError(f"CSV SHA-256 differs: {path}")
    return path, read_csv(path)


def build_rows(
    source_rows: list[dict[str, str]],
    rigid_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    rigid_by_id = {row["ligand_id"]: row for row in rigid_rows}
    if len(rigid_by_id) != len(rigid_rows):
        raise ValueError("rigid-macrocycle manifest contains duplicate IDs")
    source_ids = {row["ligand_id"] for row in source_rows}
    if not set(rigid_by_id).issubset(source_ids):
        raise ValueError("rigid-macrocycle manifest contains an unknown ligand")
    output: list[dict[str, object]] = []
    for index, source in enumerate(source_rows):
        ligand_id = source["ligand_id"]
        row: dict[str, object] = {
            **source,
            "source_manifest_index": index,
            "seed_offset": index,
            "preparation_variant": "original_meeko_flexible",
            "source_pdbqt_path": source["pdbqt_path"],
            "source_pdbqt_sha256": source["pdbqt_sha256"],
        }
        if ligand_id in rigid_by_id:
            rigid = rigid_by_id[ligand_id]
            if int(rigid["source_manifest_index"]) != index:
                raise ValueError(f"rigid source index differs: {ligand_id}")
            if rigid["source_pdbqt_sha256"] != source["pdbqt_sha256"]:
                raise ValueError(f"rigid source hash differs: {ligand_id}")
            for field in PDBQT_FIELDS:
                row[field] = rigid[field]
            row["preparation_variant"] = "meeko_rigid_macrocycles"
            row["sdf_sha256"] = rigid["sdf_sha256"]
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    source_path, source_rows = verified_csv(config["source_manifest"])
    rigid_path, rigid_rows = verified_csv(config["rigid_macrocycle_manifest"])
    output_rows = build_rows(source_rows, rigid_rows)
    expected = config["expected"]
    if len(output_rows) != int(expected["ligand_count"]):
        raise ValueError("revised Train-160 ligand count differs")
    labels = Counter(str(row["label"]) for row in output_rows)
    expected_labels = Counter(
        {key: int(value) for key, value in expected["label_counts"].items()}
    )
    if labels != expected_labels:
        raise ValueError("revised Train-160 label counts differ")
    replacement_ids = {
        str(row["ligand_id"])
        for row in output_rows
        if row["preparation_variant"] == "meeko_rigid_macrocycles"
    }
    if replacement_ids != {row["ligand_id"] for row in rigid_rows}:
        raise ValueError("revised Train-160 replacement IDs differ")
    pseudoatom_ids: list[str] = []
    for row in output_rows:
        path = Path(str(row["pdbqt_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(row["pdbqt_sha256"]).upper():
            raise ValueError(f"PDBQT hash differs: {row['ligand_id']}")
        if macrocycle_closure_atom_types(path):
            pseudoatom_ids.append(str(row["ligand_id"]))
    if pseudoatom_ids:
        raise ValueError(
            f"revised Train-160 still contains closure pseudoatoms: {pseudoatom_ids}"
        )

    output_manifest = Path(str(config["outputs"]["manifest_csv"]))
    output_summary = Path(str(config["outputs"]["summary_json"]))
    if not args.overwrite and (output_manifest.exists() or output_summary.exists()):
        raise FileExistsError("revised Train-160 outputs exist; pass --overwrite")
    write_csv(output_manifest, output_rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "operation": "consumed-train manifest substitution only; four rigid-macrocycle PDBQTs replace their flexible-macrocycle counterparts",
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "source_manifest": {
            "path": source_path.as_posix(),
            "sha256": file_sha256(source_path),
        },
        "rigid_macrocycle_manifest": {
            "path": rigid_path.as_posix(),
            "sha256": file_sha256(rigid_path),
        },
        "ligand_count": len(output_rows),
        "label_counts": dict(sorted(labels.items())),
        "replacement_count": len(replacement_ids),
        "replacement_ligand_ids": sorted(replacement_ids),
        "closure_pseudoatom_ligand_count": 0,
        "order_preserved": [row["ligand_id"] for row in output_rows]
        == [row["ligand_id"] for row in source_rows],
        "output": {
            "path": output_manifest.as_posix(),
            "sha256": file_sha256(output_manifest),
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    write_json(output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
