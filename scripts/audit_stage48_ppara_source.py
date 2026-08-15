"""Audit the frozen PPARA DUD-E source and 2P54 reference identity."""

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
        raise ValueError(f"Stage48 input identity differs: {path}")
    return path


def official_reference(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    chain = [line for line in lines if line.startswith("ATOM  ") and line[21:22] == "A"]
    ligand = [
        line
        for line in lines
        if line.startswith("HETATM")
        and line[17:20].strip() == "735"
        and line[21:22] == "A"
        and line[22:26].strip() == "469"
    ]
    ligand_heavy = [line for line in ligand if line[76:78].strip() != "H"]
    resolution = next(
        (float(line.split()[3]) for line in lines if line.startswith("REMARK   2 RESOLUTION.") and "ANGSTROMS" in line),
        None,
    )
    if not chain or len(ligand_heavy) != 33 or resolution is None:
        raise ValueError("2P54 chain A, ligand 735, or resolution differs")
    return {
        "auth_asym_id": "A",
        "protein_atom_count": len(chain),
        "protein_residue_count": len({(line[22:26], line[26:27]) for line in chain}),
        "cognate_ligand_comp_id": "735",
        "cognate_ligand_auth_seq_id": 469,
        "cognate_ligand_atom_count": len(ligand),
        "cognate_ligand_heavy_atom_count": len(ligand_heavy),
        "resolution_angstrom": resolution,
    }


def ligand_identity(dude_path: Path, official_path: Path) -> dict[str, Any]:
    dude = Chem.MolFromMol2Block(
        dude_path.read_text(encoding="utf-8", errors="replace"),
        removeHs=False,
        sanitize=True,
    )
    official_block = official_path.read_text(encoding="utf-8", errors="replace").split(chr(36) * 4, 1)[0]
    official = Chem.MolFromMolBlock(official_block, removeHs=False, sanitize=True)
    if dude is None or official is None:
        raise ValueError("PPARA reference ligand parsing failed")
    dude_heavy = Chem.RemoveHs(dude)
    official_heavy = Chem.RemoveHs(official)
    if dude_heavy.GetNumAtoms() != official_heavy.GetNumAtoms():
        raise ValueError("PPARA ligand heavy-atom counts differ")
    rmsd = float(
        rdMolAlign.CalcRMS(
            dude_heavy,
            official_heavy,
            maxMatches=1_000_000,
            symmetrizeConjugatedTerminalGroups=True,
        )
    )
    if not math.isclose(rmsd, 0.0, abs_tol=1e-6):
        raise ValueError("PPARA ligand heavy-atom coordinates differ")
    dude_formula = rdMolDescriptors.CalcMolFormula(dude_heavy)
    official_formula = rdMolDescriptors.CalcMolFormula(official_heavy)
    return {
        "heavy_atom_count": dude_heavy.GetNumAtoms(),
        "dude_formula": dude_formula,
        "official_formula": official_formula,
        "protonation_state_difference": dude_formula != official_formula,
        "fixed_frame_symmetry_corrected_rmsd_angstrom": rmsd,
        "heavy_atom_coordinate_identity": True,
    }


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage48 implementation path differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    screen = read_json(inputs["target_screen"])
    if screen.get("status") != "stage47_new_target_feasibility_screen_complete":
        raise ValueError("Stage47b screen is incomplete")
    if not screen.get("selected_target") or screen["selected_target"]["target_id"] != "PPARA":
        raise ValueError("Stage47b did not select PPARA")

    expectations = config["source_expectations"]
    active = audit_ism(
        inputs["actives"],
        expectations["actives"]["row_count"],
        expectations["actives"]["unique_source_id_count"],
        expectations["actives"]["duplicate_source_id_count"],
        expectations["actives"]["maximum_source_id_multiplicity"],
    )
    decoy = audit_ism(
        inputs["decoys"],
        expectations["decoys"]["row_count"],
        expectations["decoys"]["unique_source_id_count"],
        expectations["decoys"]["duplicate_source_id_count"],
        expectations["decoys"]["maximum_source_id_multiplicity"],
    )
    active["path"] = inputs["actives"].relative_to(root).as_posix()
    decoy["path"] = inputs["decoys"].relative_to(root).as_posix()
    identity = ligand_identity(inputs["dude_ligand"], inputs["official_ligand"])
    if not identity["protonation_state_difference"]:
        raise ValueError("expected PPARA DUD-E/RCSB protonation annotation difference is absent")

    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage48_ppara_source_audit_ok",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": sha256(config_path)},
        "target": config["target"],
        "dude_source": {
            "archive": {"path": inputs["archive"].relative_to(root).as_posix(), "sha256": sha256(inputs["archive"])},
            "actives": active,
            "decoys": decoy,
        },
        "reference_structure": {
            "pdb_id": "2P54",
            "official_pdb": {
                "path": inputs["official_pdb"].relative_to(root).as_posix(),
                "sha256": sha256(inputs["official_pdb"]),
                **official_reference(inputs["official_pdb"]),
            },
            "official_mmcif": {"path": inputs["official_mmcif"].relative_to(root).as_posix(), "sha256": sha256(inputs["official_mmcif"])},
            "ligand_identity": {
                "dude_mol2_path": inputs["dude_ligand"].relative_to(root).as_posix(),
                "dude_mol2_sha256": sha256(inputs["dude_ligand"]),
                "official_sdf_path": inputs["official_ligand"].relative_to(root).as_posix(),
                "official_sdf_sha256": sha256(inputs["official_ligand"]),
                **identity,
            },
        },
        "frozen_protocol": config["frozen_protocol"],
        "decision": {
            "source_gate_passed": True,
            "ligand_allocation_authorized": True,
            "coordinate_audit_authorized": True,
            "production_docking_authorized": False,
            "fresh_validation_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "data_boundary": {
            "source_label_rows_read": active["row_count"] + decoy["row_count"],
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "next_gate": "allocate molecule- and scaffold-disjoint ligand panels and audit the frozen 75-entry PPARA metadata pool at coordinate level",
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
