"""Audit the frozen HIVPR DUD-E source and 1XL2 reference identity."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolDescriptors

try:
    from .audit_external_target_intake import audit_ism, file_sha256
except ImportError:
    from audit_external_target_intake import audit_ism, file_sha256


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def verified(root: Path, record: dict[str, object]) -> Path:
    path = root / str(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def pdb_atom_audit(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    atoms = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    protein = [line for line in atoms if line.startswith("ATOM  ")]
    residues = {
        (line[21:22].strip(), line[22:26].strip(), line[26:27].strip())
        for line in protein
    }
    hydrogens = sum(
        (line[76:78].strip() == "H")
        or (not line[76:78].strip() and line[12:16].strip().startswith("H"))
        for line in atoms
    )
    return {
        "atom_record_count": len(atoms),
        "protein_atom_record_count": len(protein),
        "protein_residue_count": len(residues),
        "protein_chain_ids": sorted({line[21:22].strip() for line in protein}),
        "hydrogen_atom_count": hydrogens,
    }


def official_reference_audit(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    chain_atoms = {
        chain: [
            line
            for line in lines
            if line.startswith("ATOM  ") and line[21:22] == chain
        ]
        for chain in ("A", "B")
    }
    residue_counts = {
        chain: len(
            {(line[22:26].strip(), line[26:27].strip()) for line in atoms}
        )
        for chain, atoms in chain_atoms.items()
    }
    ligand = [
        line
        for line in lines
        if line.startswith("HETATM")
        and line[17:20].strip() == "189"
        and line[21:22] == "A"
        and line[22:26].strip() == "1001"
    ]
    resolution = None
    for line in lines:
        if line.startswith("REMARK   2 RESOLUTION.") and "ANGSTROMS" in line:
            resolution = float(line.split()[3])
            break
    if residue_counts != {"A": 99, "B": 99} or len(ligand) != 41:
        raise ValueError("1XL2 dimer chains or cognate ligand differ")
    if resolution is None or not math.isclose(resolution, 1.5, abs_tol=1e-6):
        raise ValueError("1XL2 resolution differs")
    return {
        "protein_auth_asym_ids": ["A", "B"],
        "protein_atom_counts": {
            chain: len(atoms) for chain, atoms in chain_atoms.items()
        },
        "protein_residue_counts": residue_counts,
        "biological_unit": "homodimer",
        "cognate_ligand_comp_id": "189",
        "cognate_ligand_auth_asym_id": "A",
        "cognate_ligand_auth_seq_id": 1001,
        "cognate_ligand_heavy_atom_count": len(ligand),
        "resolution_angstrom": resolution,
    }


def ca_coordinates(path: Path, dude_numbering: bool) -> dict[tuple[str, int], tuple[float, float, float]]:
    output: dict[tuple[str, int], tuple[float, float, float]] = {}
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        chain = line[21:22].strip()
        residue = int(line[22:26])
        if dude_numbering and chain == "B":
            residue -= 99
        output[(chain, residue)] = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    return output


def protein_frame_identity_audit(dude_pdb: Path, official_pdb: Path) -> dict[str, object]:
    dude = ca_coordinates(dude_pdb, dude_numbering=True)
    official = ca_coordinates(official_pdb, dude_numbering=False)
    if set(dude) != set(official) or len(dude) != 198:
        raise ValueError("DUD-E and RCSB HIVPR CA identities differ")
    squared = [
        sum((dude[key][axis] - official[key][axis]) ** 2 for axis in range(3))
        for key in sorted(dude)
    ]
    rmsd = math.sqrt(sum(squared) / len(squared))
    maximum = math.sqrt(max(squared))
    if not math.isclose(rmsd, 0.0, abs_tol=1e-9):
        raise ValueError("DUD-E and RCSB HIVPR CA coordinates differ")
    return {
        "matched_ca_count": len(squared),
        "fixed_frame_ca_rmsd_angstrom": rmsd,
        "maximum_ca_distance_angstrom": maximum,
        "dude_chain_b_residue_number_offset": 99,
        "coordinate_identity": True,
    }


def neutralized_heavy_molecule(molecule: Chem.Mol) -> Chem.Mol:
    heavy = Chem.RemoveHs(molecule)
    for atom in heavy.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
    heavy.UpdatePropertyCache(strict=False)
    return heavy


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
        raise ValueError("HIVPR reference ligand parsing failed")
    dude_heavy = neutralized_heavy_molecule(dude)
    official_heavy = neutralized_heavy_molecule(official)
    dude_elements = Counter(atom.GetSymbol() for atom in dude_heavy.GetAtoms())
    official_elements = Counter(atom.GetSymbol() for atom in official_heavy.GetAtoms())
    dude_bonds = Counter(str(bond.GetBondType()) for bond in dude_heavy.GetBonds())
    official_bonds = Counter(str(bond.GetBondType()) for bond in official_heavy.GetBonds())
    if (
        dude_heavy.GetNumAtoms() != 41
        or official_heavy.GetNumAtoms() != 41
        or dude_elements != official_elements
        or dude_bonds != official_bonds
    ):
        raise ValueError("HIVPR reference ligand heavy-atom graphs differ")
    rmsd = float(
        rdMolAlign.CalcRMS(
            dude_heavy,
            official_heavy,
            maxMatches=1_000_000,
            symmetrizeConjugatedTerminalGroups=True,
        )
    )
    if not math.isclose(rmsd, 0.0, abs_tol=1e-6):
        raise ValueError("DUD-E and RCSB HIVPR ligand coordinates differ")
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


def run(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if file_sha256(Path(__file__)) != str(implementation["sha256"]).upper():
        raise ValueError("Stage 15 implementation SHA-256 differs")

    verified(root, dict(config["master_preregistration"]))
    upstream_path = verified(root, dict(config["upstream_fa10_adjudication"]))
    if read_json(upstream_path)["status"] != "stage14e_fa10_confirmatory_technical_gate_closed":
        raise ValueError("FA10 upstream adjudication status differs")

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

    reference = dict(config["reference_structure"])
    dude_receptor = verified(root, dict(reference["dude_receptor_pdb"]))
    dude_ligand = verified(root, dict(reference["dude_crystal_ligand_mol2"]))
    official_pdb = verified(root, dict(reference["official_pdb"]))
    official_cif = verified(root, dict(reference["official_mmcif"]))
    official_sdf = verified(root, dict(reference["official_ligand_sdf"]))

    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage15_hivpr_source_audit_ok",
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
        "reference_structure": {
            "pdb_id": "1XL2",
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
            "protein_frame_identity": protein_frame_identity_audit(
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
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "next_gate": "run the frozen P03367 X-ray metadata discovery and dimer coordinate audit",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output_path = root / str(config["output_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
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
