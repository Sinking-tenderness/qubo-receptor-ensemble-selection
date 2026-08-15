"""Audit PARP1 source inputs and allocate the frozen active panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolAlign, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from scripts.audit_external_target_intake import audit_ism, file_sha256
from scripts.audit_stage15_hivpr_source import (
    neutralized_heavy_molecule,
    pdb_atom_audit,
    read_json,
    verified,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def official_reference_audit(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    chain_a = [
        line for line in lines if line.startswith("ATOM  ") and line[21:22] == "A"
    ]
    residue_count = len(
        {(line[22:26].strip(), line[26:27].strip()) for line in chain_a}
    )
    ligand = [
        line
        for line in lines
        if line.startswith("HETATM")
        and line[17:20].strip() == "A92"
        and line[21:22] == "A"
        and line[22:26].strip() == "351"
    ]
    resolution = None
    for line in lines:
        if line.startswith("REMARK   2 RESOLUTION.") and "ANGSTROMS" in line:
            resolution = float(line.split()[3])
            break
    if residue_count != 348 or len(ligand) != 25:
        raise ValueError("3L3M chain A or A92 ligand differs")
    if resolution is None or not math.isclose(resolution, 2.5, abs_tol=1e-6):
        raise ValueError("3L3M resolution differs")
    return {
        "reference_auth_asym_id": "A",
        "protein_residue_count": residue_count,
        "cognate_ligand_comp_id": "A92",
        "cognate_ligand_auth_asym_id": "A",
        "cognate_ligand_auth_seq_id": 351,
        "cognate_ligand_heavy_atom_count": len(ligand),
        "resolution_angstrom": resolution,
    }


def ca_coordinates(path: Path, chain: str | None) -> dict[int, tuple[float, float, float]]:
    output: dict[int, tuple[float, float, float]] = {}
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        if chain is not None and line[21:22] != chain:
            continue
        output[int(line[22:26])] = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    return output


def receptor_frame_identity(dude_pdb: Path, official_pdb: Path) -> dict[str, object]:
    dude = ca_coordinates(dude_pdb, None)
    official = ca_coordinates(official_pdb, "A")
    if set(dude) != set(official) or len(dude) != 348:
        raise ValueError("DUD-E and RCSB PARP1 CA identities differ")
    squared = [
        sum((dude[key][axis] - official[key][axis]) ** 2 for axis in range(3))
        for key in sorted(dude)
    ]
    rmsd = math.sqrt(sum(squared) / len(squared))
    maximum = math.sqrt(max(squared))
    if not math.isclose(rmsd, 0.0, abs_tol=1e-9):
        raise ValueError("DUD-E and RCSB PARP1 CA coordinates differ")
    return {
        "matched_ca_count": len(squared),
        "fixed_frame_ca_rmsd_angstrom": rmsd,
        "maximum_ca_distance_angstrom": maximum,
        "coordinate_identity": True,
    }


def ligand_identity_audit(dude_mol2: Path, official_sdf: Path) -> dict[str, object]:
    dude = Chem.MolFromMol2Block(
        dude_mol2.read_text(encoding="utf-8", errors="replace"),
        removeHs=False,
        sanitize=True,
    )
    official = Chem.MolFromMolBlock(
        official_sdf.read_text(encoding="utf-8", errors="replace"),
        removeHs=False,
        sanitize=True,
    )
    if dude is None or official is None:
        raise ValueError("PARP1 reference ligand parsing failed")
    dude_heavy = neutralized_heavy_molecule(dude)
    official_heavy = neutralized_heavy_molecule(official)
    dude_elements = Counter(atom.GetSymbol() for atom in dude_heavy.GetAtoms())
    official_elements = Counter(atom.GetSymbol() for atom in official_heavy.GetAtoms())
    dude_bonds = Counter(str(bond.GetBondType()) for bond in dude_heavy.GetBonds())
    official_bonds = Counter(str(bond.GetBondType()) for bond in official_heavy.GetBonds())
    if (
        dude_heavy.GetNumAtoms() != 25
        or official_heavy.GetNumAtoms() != 25
        or dude_elements != official_elements
        or dude_bonds != official_bonds
    ):
        raise ValueError("PARP1 reference ligand heavy-atom graphs differ")
    rmsd = float(
        rdMolAlign.CalcRMS(
            dude_heavy,
            official_heavy,
            maxMatches=1_000_000,
            symmetrizeConjugatedTerminalGroups=True,
        )
    )
    if not math.isclose(rmsd, 0.0, abs_tol=1e-6):
        raise ValueError("DUD-E and RCSB PARP1 ligand coordinates differ")
    return {
        "heavy_atom_count": dude_heavy.GetNumAtoms(),
        "heavy_atom_elements": dict(sorted(dude_elements.items())),
        "heavy_atom_bond_types": dict(sorted(dude_bonds.items())),
        "dude_formula": rdMolDescriptors.CalcMolFormula(dude),
        "official_formula": rdMolDescriptors.CalcMolFormula(official),
        "dude_formal_charge": Chem.GetFormalCharge(dude),
        "official_formal_charge": Chem.GetFormalCharge(official),
        "protonation_difference_only": True,
        "charge_neutralized_fixed_frame_rmsd_angstrom": rmsd,
        "heavy_atom_coordinate_identity": True,
    }


def scaffold_for(molecule: Chem.Mol, canonical_smiles: str) -> str:
    copy = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(copy)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=copy, includeChirality=False
    )
    return scaffold or Chem.MolToSmiles(copy, canonical=True)


def allocate_active_panels(
    path: Path, allocation: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    seed = str(allocation["hash_seed"])
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(f"PARP1 active row has fewer than two fields: {line_number}")
            molecule = Chem.MolFromSmiles(fields[0])
            if molecule is None:
                raise ValueError(f"PARP1 active SMILES does not parse: {line_number}")
            canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
            source_id = fields[1]
            rank = hashlib.sha256(
                f"{seed}|{line_number}|{source_id}".encode("ascii")
            ).hexdigest()
            rows.append(
                {
                    "source_line_number": line_number,
                    "source_molecule_id": source_id,
                    "source_smiles": fields[0],
                    "canonical_smiles": canonical,
                    "scaffold_smiles": scaffold_for(molecule, canonical),
                    "allocation_rank_sha256": rank,
                }
            )
    if len(rows) != int(allocation["source_active_count"]):
        raise ValueError("PARP1 active source row count differs")
    if len({row["source_molecule_id"] for row in rows}) != len(rows):
        raise ValueError("PARP1 active source IDs are not unique")
    if len({row["canonical_smiles"] for row in rows}) != len(rows):
        raise ValueError("PARP1 active canonical SMILES are not unique")
    scaffold_counts = Counter(row["scaffold_smiles"] for row in rows)
    if max(scaffold_counts.values()) != int(allocation["maximum_scaffold_group_size"]):
        raise ValueError("PARP1 active scaffold grouping differs")

    ranked = sorted(
        rows,
        key=lambda row: (str(row["allocation_rank_sha256"]), int(row["source_line_number"])),
    )
    train_count = int(allocation["train_active_count"])
    fresh_count = int(allocation["fresh_validation_active_count"])
    test_count = int(allocation["locked_test_active_count"])
    selected_count = train_count + fresh_count + test_count
    if selected_count != int(allocation["selected_active_count"]):
        raise ValueError("PARP1 selected active count configuration differs")
    if len(ranked) - selected_count != int(allocation["unallocated_source_surplus_count"]):
        raise ValueError("PARP1 source surplus count differs")
    for index, row in enumerate(ranked):
        if index < train_count:
            row["selection_role"] = "development_train"
            row["split"] = "train"
        elif index < train_count + fresh_count:
            row["selection_role"] = "fresh_validation"
            row["split"] = "validation"
        elif index < selected_count:
            row["selection_role"] = "locked_test"
            row["split"] = "test"
        else:
            row["selection_role"] = "unallocated_source_surplus"
            row["split"] = ""
    split_scaffolds: dict[str, set[str]] = {}
    for row in ranked:
        if not row["split"]:
            continue
        split_scaffolds.setdefault(str(row["scaffold_smiles"]), set()).add(str(row["split"]))
    if any(len(splits) > 1 for splits in split_scaffolds.values()):
        raise ValueError("PARP1 active scaffold crosses a panel boundary")
    summary = {
        "source_active_count": len(rows),
        "selected_active_count": selected_count,
        "unallocated_source_surplus_count": len(rows) - selected_count,
        "unique_canonical_smiles_count": len({row["canonical_smiles"] for row in rows}),
        "scaffold_group_count": len(scaffold_counts),
        "maximum_scaffold_group_size": max(scaffold_counts.values()),
        "panel_counts": {
            "development_train": train_count,
            "fresh_validation": fresh_count,
            "locked_test": test_count,
        },
        "scaffold_disjoint": True,
    }
    return sorted(rows, key=lambda row: int(row["source_line_number"])), summary


def run(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if file_sha256(Path(__file__)) != str(implementation["sha256"]).upper():
        raise ValueError("Stage 17a implementation SHA-256 differs")
    prereg_path = verified(root, dict(config["preregistration"]))
    preregistration = read_json(prereg_path)
    if preregistration["preregistration_id"] != "stage17-parp1-replacement-exploratory-20260801-v1":
        raise ValueError("PARP1 preregistration differs")

    dude = dict(config["dude_source"])
    archive_path = verified(root, dict(dude["archive"]))
    active_spec = dict(dude["actives"])
    decoy_spec = dict(dude["decoys"])
    active_path = verified(root, active_spec)
    decoy_path = verified(root, decoy_spec)
    active_audit = audit_ism(
        active_path,
        int(active_spec["row_count"]),
        int(active_spec["unique_source_id_count"]),
        int(active_spec["duplicate_source_id_count"]),
        int(active_spec["maximum_source_id_multiplicity"]),
    )
    decoy_audit = audit_ism(
        decoy_path,
        int(decoy_spec["row_count"]),
        int(decoy_spec["unique_source_id_count"]),
        int(decoy_spec["duplicate_source_id_count"]),
        int(decoy_spec["maximum_source_id_multiplicity"]),
    )
    active_audit["path"] = active_path.relative_to(root).as_posix()
    decoy_audit["path"] = decoy_path.relative_to(root).as_posix()

    allocation_rows, allocation_summary = allocate_active_panels(
        active_path, dict(config["active_allocation"])
    )
    reference = dict(config["reference_structure"])
    dude_receptor = verified(root, dict(reference["dude_receptor_pdb"]))
    dude_ligand = verified(root, dict(reference["dude_crystal_ligand_mol2"]))
    official_pdb = verified(root, dict(reference["official_pdb"]))
    official_cif = verified(root, dict(reference["official_mmcif"]))
    official_sdf = verified(root, dict(reference["official_ligand_sdf"]))

    outputs = dict(config["outputs"])
    allocation_path = root / str(outputs["active_allocation_csv"])
    result_path = root / str(outputs["summary_json"])
    write_csv(allocation_path, allocation_rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage17a_parp1_source_and_active_allocation_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "target": config["target"],
        "dude_source": {
            "archive": {
                "path": archive_path.relative_to(root).as_posix(),
                "sha256": file_sha256(archive_path),
            },
            "actives": active_audit,
            "decoys": decoy_audit,
        },
        "active_allocation": {
            **allocation_summary,
            "path": allocation_path.relative_to(root).as_posix(),
            "sha256": file_sha256(allocation_path),
        },
        "reference_structure": {
            "pdb_id": "3L3M",
            "dude_receptor": {
                "path": dude_receptor.relative_to(root).as_posix(),
                "sha256": file_sha256(dude_receptor),
                **pdb_atom_audit(dude_receptor),
            },
            "official_pdb": {
                "path": official_pdb.relative_to(root).as_posix(),
                "sha256": file_sha256(official_pdb),
                **official_reference_audit(official_pdb),
            },
            "official_mmcif": {
                "path": official_cif.relative_to(root).as_posix(),
                "sha256": file_sha256(official_cif),
            },
            "protein_frame_identity": receptor_frame_identity(
                dude_receptor, official_pdb
            ),
            "ligand_identity": {
                "dude_mol2_path": dude_ligand.relative_to(root).as_posix(),
                "dude_mol2_sha256": file_sha256(dude_ligand),
                "official_sdf_path": official_sdf.relative_to(root).as_posix(),
                "official_sdf_sha256": file_sha256(official_sdf),
                **ligand_identity_audit(dude_ligand, official_sdf),
            },
        },
        "data_boundary": {
            "benchmark_docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "next_gate": "run the frozen P09874 X-ray metadata discovery and coordinate-only PARP1 receptor audit",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
