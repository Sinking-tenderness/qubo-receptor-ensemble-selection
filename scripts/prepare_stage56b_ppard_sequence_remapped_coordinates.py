"""Create sequence-remapped PPARD mmCIF copies under the Stage 56a contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage56a_ppard_numbering_failure import (
    polymer_residues,
    sequence_mapping,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage 56b remapping input identity differs: {path}")
    return path


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def explicit_target_ligand_connections(
    structure: gemmi.Structure, chain_name: str, qualifying_ids: set[str]
) -> int:
    count = 0
    for connection in structure.connections:
        if connection.type != gemmi.ConnectionType.Covale:
            continue
        first, second = connection.partner1, connection.partner2
        for protein, ligand in ((first, second), (second, first)):
            if (
                protein.chain_name == chain_name
                and ligand.chain_name == chain_name
                and ligand.res_id.name in qualifying_ids
                and gemmi.find_tabulated_residue(protein.res_id.name).is_amino_acid()
            ):
                count += 1
                break
    return count


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    adjudication = read_json(inputs["adjudication_result"])
    if adjudication["status"] != "stage56a_ppard_author_numbering_failure_adjudicated":
        raise ValueError("Stage56a numbering adjudication is incomplete")
    if not adjudication["decision"]["sequence_correspondence_amendment_authorized"]:
        raise ValueError("Stage56a did not authorize sequence correspondence")
    if adjudication["decision"]["threshold_lowering_authorized"]:
        raise ValueError("Stage56a unexpectedly authorized threshold lowering")

    metadata = [
        row
        for row in read_csv(inputs["candidate_metadata_csv"])
        if row["target_id"] == "PPARD" and row["status"] == "metadata_eligible"
    ]
    if len(metadata) != int(config["expected_candidate_count"]):
        raise ValueError("Stage56b PPARD candidate count differs")
    reference_structure = gemmi.read_structure(str(inputs["reference_mmcif"]))
    reference = polymer_residues(reference_structure, "A")
    output_directory = root / config["outputs"]["remapped_mmcif_directory"]
    manifest_path = root / config["outputs"]["manifest_csv"]
    result_path = root / config["outputs"]["result_json"]
    if not overwrite and (manifest_path.exists() or result_path.exists()):
        raise FileExistsError("Stage56b remapping outputs exist; pass --overwrite")

    rows: list[dict[str, Any]] = []
    for candidate in sorted(metadata, key=lambda row: row["pdb_id"]):
        pdb_id = candidate["pdb_id"]
        source_path = root / str(config["raw_mmcif_path_template"]).format(
            pdb_id=pdb_id
        )
        structure = gemmi.read_structure(str(source_path))
        chain_name = candidate["selected_auth_chain"]
        candidate_residues = polymer_residues(structure, chain_name)
        mapping, metrics = sequence_mapping(reference, candidate_residues)
        if (
            int(metrics["sequence_mapped_residue_count"])
            < int(config["sequence_gate"]["minimum_sequence_mapped_residue_count"])
            or float(metrics["sequence_identity_fraction"])
            < float(config["sequence_gate"]["minimum_sequence_identity_fraction"])
        ):
            raise ValueError(f"Stage56b sequence gate failed: {pdb_id}")
        covalent_count = explicit_target_ligand_connections(
            structure,
            chain_name,
            {
                value
                for value in candidate["qualifying_ligand_ids"].split(";")
                if value
            },
        )
        remapped_count = 0
        unmapped_count = 0
        for index, (residue, _code) in enumerate(candidate_residues):
            key = (int(residue.seqid.num), str(residue.seqid.icode).strip())
            if key in mapping:
                target_number, target_icode = mapping[key]
                residue.seqid.num = int(target_number)
                residue.seqid.icode = target_icode or " "
                remapped_count += 1
            else:
                residue.seqid.num = -1000 - index
                residue.seqid.icode = " "
                unmapped_count += 1
        output_path = output_directory / f"{pdb_id}.cif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        structure.make_mmcif_document().write_file(str(output_path))
        rows.append(
            {
                "pdb_id": pdb_id,
                "chain": chain_name,
                "source_mmcif_path": source_path.relative_to(root).as_posix(),
                "source_mmcif_sha256": sha256(source_path),
                "remapped_mmcif_path": output_path.relative_to(root).as_posix(),
                "remapped_mmcif_sha256": sha256(output_path),
                **metrics,
                "remapped_polymer_residue_count": remapped_count,
                "unmapped_polymer_residue_count": unmapped_count,
                "original_explicit_target_ligand_covalent_connection_count": covalent_count,
            }
        )
    write_csv(manifest_path, rows)
    result = {
        "schema_version": "1.0",
        "status": "stage56b_ppard_sequence_remapped_coordinates_ready",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation),
        "candidate_count": len(rows),
        "sequence_mapping_pass_count": len(rows),
        "explicit_target_ligand_covalent_connection_count": sum(
            int(row["original_explicit_target_ligand_covalent_connection_count"])
            for row in rows
        ),
        "raw_coordinates_modified": False,
        "thresholds_changed": False,
        "outputs": {
            "manifest_csv": descriptor(root, manifest_path),
            "remapped_mmcif_directory": output_directory.relative_to(root).as_posix(),
        },
        "data_boundary": {
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
