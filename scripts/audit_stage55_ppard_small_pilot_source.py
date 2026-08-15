"""Audit PPARD source identity and freeze the Stage 55 small-pilot protocol."""

from __future__ import annotations

import argparse
import csv
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage 55 input identity differs: {path}")
    return path


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def select_next_target(
    rows: list[dict[str, str]], completed_targets: set[str]
) -> tuple[list[str], dict[str, str]]:
    eligible = [row for row in rows if row["eligible_for_selection"] == "True"]
    eligible.sort(
        key=lambda row: (
            -int(row["metadata_eligible_count"]),
            -int(row["dude_clustered_active_count"]),
            row["target_id"],
        )
    )
    remaining = [row for row in eligible if row["target_id"] not in completed_targets]
    if not remaining:
        raise ValueError("Stage 55 has no remaining eligible target")
    return [row["target_id"] for row in eligible], remaining[0]


def official_reference(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    chain = [line for line in lines if line.startswith("ATOM  ") and line[21:22] == "A"]
    ligand = [
        line
        for line in lines
        if line.startswith("HETATM")
        and line[17:20].strip() == "K55"
        and line[21:22] == "A"
        and line[22:26].strip() == "922"
    ]
    ligand_heavy = [line for line in ligand if line[76:78].strip() != "H"]
    resolution = next(
        (
            float(line.split()[3])
            for line in lines
            if line.startswith("REMARK   2 RESOLUTION.") and "ANGSTROMS" in line
        ),
        None,
    )
    if not chain or len(ligand_heavy) != 33 or resolution is None:
        raise ValueError("2ZNP chain A, ligand K55, or resolution differs")
    return {
        "auth_asym_id": "A",
        "protein_atom_count": len(chain),
        "protein_residue_count": len(
            {(line[22:26], line[26:27]) for line in chain}
        ),
        "cognate_ligand_comp_id": "K55",
        "cognate_ligand_auth_seq_id": 922,
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
    official_block = official_path.read_text(
        encoding="utf-8", errors="replace"
    ).split(chr(36) * 4, 1)[0]
    official = Chem.MolFromMolBlock(
        official_block, removeHs=False, sanitize=True
    )
    if dude is None or official is None:
        raise ValueError("PPARD reference ligand parsing failed")
    dude_heavy = Chem.RemoveHs(dude)
    official_heavy = Chem.RemoveHs(official)
    if dude_heavy.GetNumAtoms() != official_heavy.GetNumAtoms():
        raise ValueError("PPARD ligand heavy-atom counts differ")
    rmsd = float(
        rdMolAlign.CalcRMS(
            dude_heavy,
            official_heavy,
            maxMatches=1_000_000,
            symmetrizeConjugatedTerminalGroups=True,
        )
    )
    if not math.isclose(rmsd, 0.0, abs_tol=1e-6):
        raise ValueError("PPARD ligand heavy-atom coordinates differ")
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
        raise ValueError("Stage 55 implementation identity differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}

    screen_result = read_json(inputs["target_screen_result"])
    if screen_result["status"] != "stage47_new_target_feasibility_screen_complete":
        raise ValueError("Stage47b target screen is incomplete")
    eligible_order, selected = select_next_target(
        read_csv(inputs["target_screen_csv"]),
        set(config["selection_rule"]["completed_targets"]),
    )
    if eligible_order != config["selection_rule"]["expected_eligible_order"]:
        raise ValueError("Stage 55 eligible target order differs")
    if selected["target_id"] != config["target"]["target_id"]:
        raise ValueError("Stage 55 did not select PPARD by the frozen rule")

    future_gate = read_json(inputs["stage54_future_intake_criteria"])
    if future_gate["status"] != "stage54_future_target_intake_criteria_frozen":
        raise ValueError("Stage54 future-target intake criteria are not frozen")
    if future_gate["criteria"] != config["frozen_protocol"]["functional_gate"]:
        raise ValueError("Stage 55 functional gate differs from Stage54")

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
        raise ValueError("expected PPARD DUD-E/RCSB protonation difference is absent")
    reference = official_reference(inputs["official_pdb"])

    protocol = config["frozen_protocol"]
    maximum_pair_jobs = (
        int(protocol["pilot_panel"]["active_count"])
        + int(protocol["pilot_panel"]["decoy_count"])
    ) * int(protocol["structural_pool"]["metadata_eligible_count_at_freeze"])
    maximum_seeded_jobs = maximum_pair_jobs * int(protocol["docking"]["seed_count"])
    if maximum_pair_jobs != 4896 or maximum_seeded_jobs != 14688:
        raise ValueError("Stage 55 pilot job budget differs")

    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage55_ppard_small_pilot_source_and_preregistration_ok",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation),
        "selection": {
            "eligible_target_order": eligible_order,
            "completed_targets_before_selection": config["selection_rule"][
                "completed_targets"
            ],
            "selected_target": selected["target_id"],
            "selection_was_outcome_blind": True,
            "metadata_eligible_structure_count": int(
                selected["metadata_eligible_count"]
            ),
            "dude_clustered_active_count": int(
                selected["dude_clustered_active_count"]
            ),
        },
        "target": config["target"],
        "dude_source": {
            "archive": descriptor(root, inputs["archive"]),
            "actives": active,
            "decoys": decoy,
        },
        "reference_structure": {
            "pdb_id": "2ZNP",
            "official_pdb": {
                **descriptor(root, inputs["official_pdb"]),
                **reference,
            },
            "official_mmcif": descriptor(root, inputs["official_mmcif"]),
            "ligand_identity": {
                "dude_mol2": descriptor(root, inputs["dude_ligand"]),
                "official_sdf": descriptor(root, inputs["official_ligand"]),
                **identity,
            },
        },
        "frozen_protocol": protocol,
        "pilot_budget": {
            "maximum_receptor_ligand_pairs": maximum_pair_jobs,
            "maximum_seeded_docking_jobs": maximum_seeded_jobs,
            "full_training_matrix_before_gate_permitted": False,
        },
        "decision": {
            "source_gate_passed": True,
            "outcome_blind_target_selection_confirmed": True,
            "ligand_panel_allocation_authorized": True,
            "coordinate_audit_authorized": True,
            "cognate_redocking_authorized": True,
            "pilot_production_docking_authorized": False,
            "full_training_matrix_authorized": False,
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
        "next_gate": (
            "allocate scaffold-disjoint PPARD panels, audit all 51 frozen metadata-eligible "
            "coordinates, and retain every structure passing the hard preparation and "
            "cognate-redocking gates"
        ),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output = root / config["output_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
