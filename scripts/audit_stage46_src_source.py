"""Audit the frozen SRC DUD-E source and 3EL8 reference identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_external_target_intake import audit_ism


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def verified(root: Path, record: dict[str, Any]) -> Path:
    path = root / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"].upper():
        raise ValueError(f"Stage46 input identity differs: {path}")
    return path


def pdb_atom_audit(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    atoms = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    protein = [line for line in atoms if line.startswith("ATOM  ")]
    residues = {(line[21:22].strip(), line[22:26].strip(), line[26:27].strip()) for line in protein}
    hydrogens = sum(
        line[76:78].strip() == "H"
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


def official_reference_audit(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    chain_a = [line for line in lines if line.startswith("ATOM  ") and line[21:22] == "A"]
    ligand = [
        line
        for line in lines
        if line.startswith("HETATM")
        and line[17:20].strip() == "PD5"
        and line[21:22] == "A"
        and line[22:26].strip() == "601"
    ]
    resolution = next(
        (
            float(line.split()[3])
            for line in lines
            if line.startswith("REMARK   2 RESOLUTION.") and "ANGSTROMS" in line
        ),
        None,
    )
    if not chain_a or len(ligand) != 33 or resolution is None:
        raise ValueError("3EL8 reference chain, PD5 ligand, or resolution differs")
    return {
        "auth_asym_id": "A",
        "protein_atom_count": len(chain_a),
        "protein_residue_count": len({(line[22:26], line[26:27]) for line in chain_a}),
        "cognate_ligand_comp_id": "PD5",
        "cognate_ligand_auth_seq_id": 601,
        "cognate_ligand_heavy_atom_count": len(ligand),
        "resolution_angstrom": resolution,
    }


def ligand_identity_audit(dude_mol2: Path, official_sdf: Path) -> dict[str, Any]:
    dude = Chem.MolFromMol2Block(
        dude_mol2.read_text(encoding="utf-8", errors="replace"),
        removeHs=False,
        sanitize=True,
    )
    mol_block = official_sdf.read_text(encoding="utf-8", errors="replace").split("$$$$", 1)[0]
    official = Chem.MolFromMolBlock(mol_block, removeHs=False, sanitize=True)
    if dude is None or official is None:
        raise ValueError("SRC reference ligand parsing failed")
    dude_heavy = Chem.RemoveHs(dude)
    official_heavy = Chem.RemoveHs(official)
    if dude_heavy.GetNumAtoms() != official_heavy.GetNumAtoms():
        raise ValueError("SRC reference ligand heavy-atom counts differ")
    dude_formula = rdMolDescriptors.CalcMolFormula(dude_heavy)
    official_formula = rdMolDescriptors.CalcMolFormula(official_heavy)
    if dude_formula != official_formula:
        raise ValueError("SRC reference ligand formulas differ")
    rmsd = float(
        rdMolAlign.CalcRMS(
            dude_heavy,
            official_heavy,
            maxMatches=1_000_000,
            symmetrizeConjugatedTerminalGroups=True,
        )
    )
    if not math.isclose(rmsd, 0.0, abs_tol=1e-6):
        raise ValueError("DUD-E and RCSB SRC ligand coordinates differ")
    return {
        "heavy_atom_count": dude_heavy.GetNumAtoms(),
        "molecular_formula": dude_formula,
        "fixed_frame_symmetry_corrected_rmsd_angstrom": rmsd,
        "coordinate_identity": True,
    }


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage46 implementation path differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    stage45 = read_json(inputs["stage45_result"])
    stage45_audit = read_json(inputs["stage45_audit"])
    if stage45.get("status") != "stage45_pparg_md96_generalization_diagnosis_complete":
        raise ValueError("Stage45 result is incomplete")
    if stage45_audit.get("status") != "stage45_pparg_md96_generalization_diagnosis_independent_audit_ok":
        raise ValueError("Stage45 audit is incomplete")
    if not stage45["decision"]["k6_new_target_preregistration_authorized"]:
        raise ValueError("Stage45 did not authorize a new-target k=6 preregistration")

    active_spec = config["source_expectations"]["actives"]
    decoy_spec = config["source_expectations"]["decoys"]
    active_audit = audit_ism(
        inputs["actives"],
        active_spec["row_count"],
        active_spec["unique_source_id_count"],
        active_spec["duplicate_source_id_count"],
        active_spec["maximum_source_id_multiplicity"],
    )
    decoy_audit = audit_ism(
        inputs["decoys"],
        decoy_spec["row_count"],
        decoy_spec["unique_source_id_count"],
        decoy_spec["duplicate_source_id_count"],
        decoy_spec["maximum_source_id_multiplicity"],
    )
    for record, key in ((active_audit, "actives"), (decoy_audit, "decoys")):
        record["path"] = inputs[key].relative_to(root).as_posix()

    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage46_src_source_audit_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256(config_path),
        },
        "target": config["target"],
        "dude_source": {
            "archive": {
                "path": inputs["archive"].relative_to(root).as_posix(),
                "sha256": sha256(inputs["archive"]),
            },
            "actives": active_audit,
            "decoys": decoy_audit,
        },
        "reference_structure": {
            "pdb_id": "3EL8",
            "dude_receptor": {
                "path": inputs["dude_receptor"].relative_to(root).as_posix(),
                "sha256": sha256(inputs["dude_receptor"]),
                **pdb_atom_audit(inputs["dude_receptor"]),
            },
            "official_pdb": {
                "path": inputs["official_pdb"].relative_to(root).as_posix(),
                "sha256": sha256(inputs["official_pdb"]),
                **official_reference_audit(inputs["official_pdb"]),
            },
            "official_mmcif": {
                "path": inputs["official_mmcif"].relative_to(root).as_posix(),
                "sha256": sha256(inputs["official_mmcif"]),
            },
            "ligand_identity": {
                "dude_mol2_path": inputs["dude_ligand"].relative_to(root).as_posix(),
                "dude_mol2_sha256": sha256(inputs["dude_ligand"]),
                "official_sdf_path": inputs["official_ligand"].relative_to(root).as_posix(),
                "official_sdf_sha256": sha256(inputs["official_ligand"]),
                **ligand_identity_audit(inputs["dude_ligand"], inputs["official_ligand"]),
            },
        },
        "frozen_hypothesis": config["frozen_hypothesis"],
        "future_protocol": config["future_protocol"],
        "data_boundary": {
            "source_label_rows_read": active_audit["row_count"] + decoy_audit["row_count"],
            "structural_candidate_outcomes_read": 0,
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "decision": {
            "source_gate_passed": True,
            "structural_metadata_discovery_authorized": True,
            "production_docking_authorized": False,
            "fresh_validation_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "next_gate": "freeze and run label-independent RCSB metadata discovery for wild-type human SRC structures",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output = root / config["output_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.root / args.config
    run(config_path, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
