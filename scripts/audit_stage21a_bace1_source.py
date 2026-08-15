"""Audit the frozen BACE1 DUD-E source, panel allocation, and 3L5D identity."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

try:
    from scripts.audit_external_target_intake import audit_ism, file_sha256
    from scripts.audit_stage15_hivpr_source import pdb_atom_audit, read_json, verified
    from scripts.audit_stage20a_esr1_source import (
        allocate_active_panels,
        ca_coordinate_set,
        write_csv,
    )
except ModuleNotFoundError:
    from audit_external_target_intake import audit_ism, file_sha256
    from audit_stage15_hivpr_source import pdb_atom_audit, read_json, verified
    from audit_stage20a_esr1_source import (
        allocate_active_panels,
        ca_coordinate_set,
        write_csv,
    )


def protein_frame_provenance(dude_pdb: Path, official_pdb: Path) -> dict[str, object]:
    dude = ca_coordinate_set(dude_pdb, None)
    chain_a = ca_coordinate_set(official_pdb, "A")
    chain_b = ca_coordinate_set(official_pdb, "B")
    exact_a = dude & chain_a
    exact_b_only = (dude - chain_a) & chain_b
    unmatched = dude - chain_a - chain_b
    if (
        len(dude),
        len(chain_a),
        len(chain_b),
        len(exact_a),
        len(exact_b_only),
        len(unmatched),
    ) != (364, 386, 389, 352, 12, 0):
        raise ValueError("BACE1 DUD-E/RCSB CA-coordinate provenance differs")
    return {
        "dude_ca_count": len(dude),
        "official_chain_a_ca_count": len(chain_a),
        "official_chain_b_ca_count": len(chain_b),
        "exact_chain_a_coordinate_count": len(exact_a),
        "exact_chain_b_only_coordinate_count": len(exact_b_only),
        "unmatched_ca_count": len(unmatched),
        "interpretation": "DUD-E supplies a prepared 3L5D receptor assembled entirely from exact chain-A coordinates plus 12 exact chain-B donor coordinates.",
    }


def official_reference_audit(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    chain_a_ca = sum(
        line.startswith("ATOM  ")
        and line[21:22] == "A"
        and line[12:16].strip() == "CA"
        for line in lines
    )
    ligand_atoms = [
        line
        for line in lines
        if line.startswith("HETATM")
        and line[17:20].strip() == "BDV"
        and line[21:22] == "A"
        and line[22:26].strip() == "1"
    ]
    resolution = next(
        (
            float(line.split()[3])
            for line in lines
            if line.startswith("REMARK   2 RESOLUTION.") and "ANGSTROMS" in line
        ),
        None,
    )
    if chain_a_ca != 386 or len(ligand_atoms) != 28:
        raise ValueError("3L5D chain-A or BDV records differ")
    if resolution is None or not math.isclose(resolution, 1.75, abs_tol=1e-6):
        raise ValueError("3L5D resolution differs")
    return {
        "reference_auth_asym_id": "A",
        "reference_chain_a_ca_count": chain_a_ca,
        "cognate_ligand_comp_id": "BDV",
        "cognate_ligand_auth_asym_id": "A",
        "cognate_ligand_auth_seq_id": 1,
        "cognate_ligand_heavy_atom_count": len(ligand_atoms),
        "resolution_angstrom": resolution,
    }


def heavy_coordinate_signature(molecule: Chem.Mol) -> list[tuple[object, ...]]:
    conformer = molecule.GetConformer()
    output: list[tuple[object, ...]] = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        position = conformer.GetAtomPosition(atom.GetIdx())
        output.append(
            (
                atom.GetSymbol(),
                round(float(position.x), 6),
                round(float(position.y), 6),
                round(float(position.z), 6),
            )
        )
    return sorted(output)


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
        raise ValueError("BACE1 reference-ligand parsing failed")
    dude_heavy = [atom for atom in dude.GetAtoms() if atom.GetAtomicNum() > 1]
    official_heavy = [atom for atom in official.GetAtoms() if atom.GetAtomicNum() > 1]
    dude_elements = Counter(atom.GetSymbol() for atom in dude_heavy)
    official_elements = Counter(atom.GetSymbol() for atom in official_heavy)
    dude_bonds = Counter(
        str(bond.GetBondType())
        for bond in dude.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() > 1
        and bond.GetEndAtom().GetAtomicNum() > 1
    )
    official_bonds = Counter(
        str(bond.GetBondType())
        for bond in official.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() > 1
        and bond.GetEndAtom().GetAtomicNum() > 1
    )
    if (
        len(dude_heavy) != 28
        or len(official_heavy) != 28
        or dude_elements != official_elements
        or heavy_coordinate_signature(dude) != heavy_coordinate_signature(official)
    ):
        raise ValueError("BACE1 reference-ligand heavy atoms or coordinates differ")
    return {
        "heavy_atom_count": len(dude_heavy),
        "heavy_atom_elements": dict(sorted(dude_elements.items())),
        "dude_heavy_atom_bond_types": dict(sorted(dude_bonds.items())),
        "official_heavy_atom_bond_types": dict(sorted(official_bonds.items())),
        "dude_formula": rdMolDescriptors.CalcMolFormula(dude),
        "official_formula": rdMolDescriptors.CalcMolFormula(official),
        "dude_formal_charge": Chem.GetFormalCharge(dude),
        "official_formal_charge": Chem.GetFormalCharge(official),
        "protonation_or_tautomerization_difference_only": True,
        "heavy_atom_coordinate_identity": True,
        "maximum_nearest_heavy_atom_coordinate_distance_angstrom": 0.0,
    }


def run(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if file_sha256(Path(__file__)) != str(config["implementation"]["sha256"]).upper():
        raise ValueError("Stage 21a implementation SHA-256 differs")
    for dependency in config.get("dependencies", []):
        verified(root, dict(dependency))
    preregistration_path = verified(root, dict(config["preregistration"]))
    preregistration = read_json(preregistration_path)
    if (
        preregistration["preregistration_id"]
        != "stage21-bace1-independent-exploratory-20260801-v1"
    ):
        raise ValueError("BACE1 preregistration differs")

    dude = dict(config["dude_source"])
    archive_path = verified(root, dict(dude["archive"]))
    active_path = verified(root, dict(dude["actives"]))
    decoy_path = verified(root, dict(dude["decoys"]))
    active_audit = audit_ism(active_path, 283, 283, 0, 1)
    decoy_audit = audit_ism(decoy_path, 18100, 18080, 20, 2)
    if decoy_audit["unique_canonical_smiles_count"] != 18097:
        raise ValueError("BACE1 decoy canonical-SMILES count differs")
    allocation_rows, allocation_summary = allocate_active_panels(
        active_path, dict(config["active_allocation"])
    )

    reference = dict(config["reference_structure"])
    dude_receptor = verified(root, dict(reference["dude_receptor_pdb"]))
    dude_ligand = verified(root, dict(reference["dude_crystal_ligand_mol2"]))
    official_pdb = verified(root, dict(reference["official_pdb"]))
    official_cif = verified(root, dict(reference["official_mmcif"]))
    official_sdf = verified(root, dict(reference["official_ligand_sdf"]))
    allocation_path = root / str(config["outputs"]["active_allocation_csv"])
    result_path = root / str(config["outputs"]["summary_json"])
    write_csv(allocation_path, allocation_rows)

    for audit in (active_audit, decoy_audit):
        audit["path"] = (root / str(audit["path"])).relative_to(root).as_posix()
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage21a_bace1_source_and_active_allocation_ok",
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
            "pdb_id": "3L5D",
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
            "protein_frame_provenance": protein_frame_provenance(
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
            "ligand_identity_and_source_labels_read": True,
            "coordinate_pool_selected": False,
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "pparg_stage19c_enrichment_rows_read": 0,
        },
        "next_gate": "run frozen BACE1 RCSB metadata discovery, coordinate audit, and deterministic structural max-min selection",
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
