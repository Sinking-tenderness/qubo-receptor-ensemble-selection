"""Audit a new target's DUD-E files and co-crystal structure inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_hash(path: Path, expected: str) -> str:
    observed = file_sha256(path)
    if observed != expected.upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return observed


def molblock_sha256(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        end = next(index for index, line in enumerate(lines) if line == "M  END")
    except StopIteration as exc:
        raise ValueError(f"SDF mol block has no M  END: {path}") from exc
    normalized = "\n".join(lines[: end + 1]) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest().upper()


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "target",
        "dude",
        "structures",
        "output_json",
        "interpretation_boundary",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("external-target intake config is incomplete")
    dude = config["dude"]
    required_dude = {
        "target_id",
        "target_url",
        "actives",
        "decoys",
        "pdb_blessed",
        "pdb_selection",
    }
    if not isinstance(dude, dict) or not required_dude.issubset(dude):
        raise ValueError("DUD-E intake specification is incomplete")
    structures = config["structures"]
    if (
        not isinstance(structures, list)
        or len(structures) < 2
        or len({str(row["structure_id"]) for row in structures})
        != len(structures)
    ):
        raise ValueError("structure intake list is invalid")
    for row in structures:
        if not {
            "structure_id",
            "role",
            "conformation_note",
            "pdb",
            "chain",
            "target_ligand",
        }.issubset(row):
            raise ValueError("structure intake row is incomplete")
    return config


def audit_ism(
    path: Path,
    expected_count: int,
    expected_unique_id_count: int | None = None,
    expected_duplicate_id_count: int = 0,
    expected_max_id_multiplicity: int = 1,
) -> dict[str, object]:
    ids: list[str] = []
    canonical: Counter[str] = Counter()
    parse_failures: list[dict[str, object]] = []
    charged_count = 0
    multi_fragment_count = 0
    heavy_counts: list[int] = []
    molecular_weights: list[float] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) < 2:
                parse_failures.append(
                    {"line_number": line_number, "reason": "fewer_than_two_fields"}
                )
                continue
            smiles, molecule_id = fields[0], fields[1]
            ids.append(molecule_id)
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                parse_failures.append(
                    {"line_number": line_number, "reason": "invalid_smiles"}
                )
                continue
            canonical[Chem.MolToSmiles(molecule, isomericSmiles=True)] += 1
            charged_count += int(Chem.GetFormalCharge(molecule) != 0)
            multi_fragment_count += int(len(Chem.GetMolFrags(molecule)) > 1)
            heavy_counts.append(molecule.GetNumHeavyAtoms())
            molecular_weights.append(float(Descriptors.MolWt(molecule)))
    if len(ids) != expected_count:
        raise ValueError(f"ISM row count differs: {path}")
    if parse_failures:
        raise ValueError(f"ISM parsing failed for {len(parse_failures)} rows: {path}")
    source_id_counts = Counter(ids)
    duplicate_source_ids = sorted(
        molecule_id
        for molecule_id, count in source_id_counts.items()
        if count > 1
    )
    unique_expected = (
        expected_count
        if expected_unique_id_count is None
        else expected_unique_id_count
    )
    if (
        len(source_id_counts) != unique_expected
        or len(duplicate_source_ids) != expected_duplicate_id_count
        or max(source_id_counts.values()) != expected_max_id_multiplicity
    ):
        raise ValueError(f"ISM source-ID multiplicity differs: {path}")
    duplicates = sum(count - 1 for count in canonical.values() if count > 1)
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "row_count": len(ids),
        "unique_source_molecule_id_count": len(source_id_counts),
        "duplicate_source_molecule_id_count": len(duplicate_source_ids),
        "duplicate_source_molecule_row_count": sum(
            source_id_counts[value] for value in duplicate_source_ids
        ),
        "maximum_source_id_multiplicity": max(source_id_counts.values()),
        "duplicate_source_molecule_ids": duplicate_source_ids,
        "recommended_internal_id_rule": (
            "target_label_plus_source_line_number; preserve source molecule ID as metadata"
        ),
        "rdkit_parsed_count": len(heavy_counts),
        "rdkit_failure_count": 0,
        "unique_canonical_smiles_count": len(canonical),
        "duplicate_canonical_row_count": duplicates,
        "charged_molecule_count": charged_count,
        "multi_fragment_molecule_count": multi_fragment_count,
        "heavy_atom_count_min": min(heavy_counts),
        "heavy_atom_count_max": max(heavy_counts),
        "molecular_weight_min": min(molecular_weights),
        "molecular_weight_max": max(molecular_weights),
    }


def parse_resolution(lines: list[str]) -> float | None:
    pattern = re.compile(r"^REMARK\s+2\s+RESOLUTION\.\s+([0-9.]+)\s+ANGSTROMS")
    for line in lines:
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    return None


def audit_pdb(
    path: Path,
    chain: str,
    ligand_name: str,
    ligand_resseq: int,
) -> tuple[dict[str, object], int]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    atom_lines = [line for line in lines if line.startswith("ATOM  ")]
    hetatm_lines = [line for line in lines if line.startswith("HETATM")]
    if not atom_lines:
        raise ValueError(f"PDB has no ATOM records: {path}")
    chains = sorted({line[21].strip() for line in atom_lines})
    if chains != [chain]:
        raise ValueError(f"PDB protein chains differ: {path}")
    protein_residues = {
        (line[21].strip(), int(line[22:26]), line[26].strip())
        for line in atom_lines
    }
    alternate_location_count = sum(
        bool(line[16].strip()) for line in atom_lines + hetatm_lines
    )
    water_count = 0
    hetero_groups: Counter[tuple[str, str, int]] = Counter()
    for line in hetatm_lines:
        name = line[17:20].strip()
        record_chain = line[21].strip()
        resseq = int(line[22:26])
        if name == "HOH":
            water_count += 1
        else:
            hetero_groups[(name, record_chain, resseq)] += 1
    ligand_key = (ligand_name, chain, ligand_resseq)
    ligand_atom_count = hetero_groups.get(ligand_key, 0)
    if ligand_atom_count <= 0:
        raise ValueError(f"target ligand is absent from PDB: {path}")
    return (
        {
            "path": path.as_posix(),
            "sha256": file_sha256(path),
            "experimental_method": next(
                (line[6:].strip() for line in lines if line.startswith("EXPDTA")),
                None,
            ),
            "resolution_angstrom": parse_resolution(lines),
            "protein_atom_count": len(atom_lines),
            "protein_residue_count": len(protein_residues),
            "protein_chains": chains,
            "water_atom_count": water_count,
            "alternate_location_coordinate_count": alternate_location_count,
            "nonwater_hetero_groups": [
                {
                    "residue_name": key[0],
                    "chain": key[1],
                    "residue_number": key[2],
                    "atom_count": count,
                    "is_target_ligand": key == ligand_key,
                }
                for key, count in sorted(hetero_groups.items())
            ],
            "remark_465_line_count": sum(
                line.startswith("REMARK 465") for line in lines
            ),
        },
        ligand_atom_count,
    )


def audit_sdf(path: Path) -> dict[str, object]:
    source_lines = path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    molecules = [molecule for molecule in supplier if molecule is not None]
    if len(molecules) != 1:
        raise ValueError(f"SDF must contain exactly one parsed molecule: {path}")
    molecule = molecules[0]
    centers = Chem.FindMolChiralCenters(
        molecule, includeUnassigned=True, useLegacyImplementation=False
    )
    conformer = molecule.GetConformer()
    coordinates = [conformer.GetAtomPosition(index) for index in range(molecule.GetNumAtoms())]
    spans = {
        axis: max(values) - min(values)
        for axis, values in {
            "x": [position.x for position in coordinates],
            "y": [position.y for position in coordinates],
            "z": [position.z for position in coordinates],
        }.items()
    }
    return {
        "path": path.as_posix(),
        "raw_sha256": file_sha256(path),
        "molblock_sha256_lf_normalized": molblock_sha256(path),
        "parsed_molecule_count": 1,
        "atom_count": molecule.GetNumAtoms(),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(molecule),
        "conformer_count": molecule.GetNumConformers(),
        "source_program_line": source_lines[1] if len(source_lines) > 1 else "",
        "source_dimension_tag": (
            "3D" if len(source_lines) > 1 and "3D" in source_lines[1] else "unspecified"
        ),
        "rdkit_conformer_is_3d": bool(conformer.Is3D()),
        "coordinate_span_angstrom": spans,
        "nonzero_z_span": spans["z"] > 1e-6,
        "assigned_chiral_centers": [
            f"{index}:{label}" for index, label in centers
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output_path = Path(str(config["output_json"]))
    if output_path.exists() and not args.overwrite:
        raise FileExistsError("external-target intake output already exists")

    dude = config["dude"]
    source_records: dict[str, object] = {}
    for key in ("pdb_blessed", "pdb_selection"):
        specification = dude[key]
        path = Path(specification["path"])
        source_records[key] = {
            "path": path.as_posix(),
            "source_url": specification["source_url"],
            "sha256": require_hash(path, specification["sha256"]),
        }
    active_spec = dude["actives"]
    decoy_spec = dude["decoys"]
    active_path = Path(active_spec["path"])
    decoy_path = Path(decoy_spec["path"])
    require_hash(active_path, active_spec["sha256"])
    require_hash(decoy_path, decoy_spec["sha256"])
    active_audit = audit_ism(
        active_path,
        int(active_spec["expected_count"]),
        int(active_spec["expected_unique_id_count"]),
        int(active_spec["expected_duplicate_id_count"]),
        int(active_spec["expected_max_id_multiplicity"]),
    )
    decoy_audit = audit_ism(
        decoy_path,
        int(decoy_spec["expected_count"]),
        int(decoy_spec["expected_unique_id_count"]),
        int(decoy_spec["expected_duplicate_id_count"]),
        int(decoy_spec["expected_max_id_multiplicity"]),
    )
    active_audit["source_url"] = active_spec["source_url"]
    decoy_audit["source_url"] = decoy_spec["source_url"]

    structure_records: list[dict[str, object]] = []
    for specification in config["structures"]:
        pdb_spec = specification["pdb"]
        ligand_spec = specification["target_ligand"]
        pdb_path = Path(pdb_spec["path"])
        sdf_path = Path(ligand_spec["sdf_path"])
        require_hash(pdb_path, pdb_spec["sha256"])
        require_hash(sdf_path, ligand_spec["sdf_sha256"])
        pdb_audit, pdb_ligand_atom_count = audit_pdb(
            pdb_path,
            str(specification["chain"]),
            str(ligand_spec["residue_name"]),
            int(ligand_spec["residue_number"]),
        )
        sdf_audit = audit_sdf(sdf_path)
        if pdb_ligand_atom_count != int(sdf_audit["heavy_atom_count"]):
            raise ValueError(
                f"PDB/SDF ligand heavy-atom count differs: {specification['structure_id']}"
            )
        structure_records.append(
            {
                "structure_id": specification["structure_id"],
                "role": specification["role"],
                "conformation_note": specification["conformation_note"],
                "chain": specification["chain"],
                "pdb_source_url": pdb_spec["source_url"],
                "pdb_audit": pdb_audit,
                "target_ligand": {
                    "residue_name": ligand_spec["residue_name"],
                    "residue_number": ligand_spec["residue_number"],
                    "source_url": ligand_spec["source_url"],
                    "pdb_coordinate_atom_count": pdb_ligand_atom_count,
                    "sdf_audit": sdf_audit,
                },
            }
        )

    implementation_path = Path(__file__)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "operation": "external target raw-data and co-crystal intake audit",
        "status": "ok",
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "implementation": {
            "path": f"scripts/{implementation_path.name}",
            "sha256": file_sha256(implementation_path),
        },
        "target": config["target"],
        "dude": {
            "target_id": dude["target_id"],
            "target_url": dude["target_url"],
            "actives": active_audit,
            "decoys": decoy_audit,
            "selection_records": source_records,
        },
        "structures": structure_records,
        "gate": {
            "all_raw_hashes_match": True,
            "all_smiles_parsed": True,
            "all_target_ligands_match_pdb_sdf_heavy_atoms": True,
            "structure_count": len(structure_records),
            "ready_for_alignment_and_redocking": True,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "target": config["target"]["target_id"],
                "active_count": active_audit["row_count"],
                "decoy_count": decoy_audit["row_count"],
                "structure_count": len(structure_records),
                "ready_for_alignment_and_redocking": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
