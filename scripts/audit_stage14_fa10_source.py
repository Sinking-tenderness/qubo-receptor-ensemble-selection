"""Audit the frozen FA10 DUD-E source and 3KL6 reference identity."""

from __future__ import annotations

import argparse
import json
import math
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
    hydrogen_count = sum(
        (line[76:78].strip() == "H")
        or (not line[76:78].strip() and line[12:16].strip().startswith("H"))
        for line in atoms
    )
    return {
        "atom_record_count": len(atoms),
        "protein_atom_record_count": len(protein),
        "protein_residue_count": len(residues),
        "protein_chain_ids": sorted({line[21:22].strip() for line in protein}),
        "hydrogen_atom_count": hydrogen_count,
    }


def official_reference_audit(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    chain_a = [
        line
        for line in lines
        if line.startswith("ATOM  ") and line[21:22] == "A"
    ]
    ligand = [
        line
        for line in lines
        if line.startswith("HETATM")
        and line[17:20].strip() == "443"
        and line[21:22] == "A"
        and line[22:26].strip() == "1"
    ]
    resolution = None
    for line in lines:
        if line.startswith("REMARK   2 RESOLUTION.") and "ANGSTROMS" in line:
            resolution = float(line.split()[3])
            break
    if not chain_a or len(ligand) != 32 or resolution is None:
        raise ValueError("3KL6 reference chain, ligand, or resolution differs")
    residues = {(line[22:26].strip(), line[26:27].strip()) for line in chain_a}
    return {
        "heavy_chain_auth_asym_id": "A",
        "heavy_chain_atom_count": len(chain_a),
        "heavy_chain_residue_count": len(residues),
        "cognate_ligand_comp_id": "443",
        "cognate_ligand_auth_seq_id": 1,
        "cognate_ligand_heavy_atom_count": len(ligand),
        "resolution_angstrom": resolution,
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
        raise ValueError("FA10 reference ligand parsing failed")
    dude = Chem.RemoveHs(dude)
    official = Chem.RemoveHs(official)
    if dude.GetNumAtoms() != official.GetNumAtoms():
        raise ValueError("FA10 reference ligand heavy-atom counts differ")
    formula_dude = rdMolDescriptors.CalcMolFormula(dude)
    formula_official = rdMolDescriptors.CalcMolFormula(official)
    if formula_dude != formula_official:
        raise ValueError("FA10 reference ligand formulas differ")
    rmsd = float(
        rdMolAlign.CalcRMS(
            dude,
            official,
            maxMatches=1_000_000,
            symmetrizeConjugatedTerminalGroups=True,
        )
    )
    if not math.isclose(rmsd, 0.0, abs_tol=1e-6):
        raise ValueError("DUD-E and RCSB FA10 ligand coordinates differ")
    return {
        "heavy_atom_count": dude.GetNumAtoms(),
        "molecular_formula": formula_dude,
        "fixed_frame_symmetry_corrected_rmsd_angstrom": rmsd,
        "coordinate_identity": True,
    }


def run(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if file_sha256(Path(__file__)) != str(implementation["sha256"]).upper():
        raise ValueError("Stage 14 implementation SHA-256 differs")

    master = dict(config["master_preregistration"])
    verified(root, master)
    upstream = dict(config["upstream_egfr_adjudication"])
    upstream_path = verified(root, upstream)
    if read_json(upstream_path)["status"] != "stage13g_egfr_confirmatory_technical_gate_closed":
        raise ValueError("EGFR upstream adjudication status differs")

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
        "status": "stage14_fa10_source_audit_ok",
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
            "pdb_id": "3KL6",
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
        "next_gate": "run the frozen P00742 X-ray metadata discovery and coordinate-only receptor audit",
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
