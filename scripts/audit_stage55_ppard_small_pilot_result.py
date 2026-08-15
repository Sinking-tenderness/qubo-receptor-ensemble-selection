"""Independently audit the Stage 55 PPARD source and pilot preregistration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def checked(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage 55 file identity differs: {path}")
    return path


def ism_identity(path: Path) -> dict[str, int]:
    rows = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if any(len(row) < 2 for row in rows):
        raise ValueError(f"Stage 55 malformed ISM row: {path}")
    ids = Counter(row[1] for row in rows)
    return {
        "row_count": len(rows),
        "unique_source_id_count": len(ids),
        "duplicate_source_id_count": sum(value > 1 for value in ids.values()),
        "maximum_source_id_multiplicity": max(ids.values()),
    }


def run(result_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    result_path = result_path.resolve()
    result = read_json(result_path)
    if result["status"] != "stage55_ppard_small_pilot_source_and_preregistration_ok":
        raise ValueError("Stage 55 source result did not pass")
    config_path = checked(root, result["config"])
    implementation_path = checked(root, result["implementation"])
    config = read_json(config_path)
    if implementation_path.resolve() != (root / config["implementation"]["path"]).resolve():
        raise ValueError("Stage 55 implementation path differs")
    inputs = {key: checked(root, value) for key, value in config["inputs"].items()}

    rows = read_csv(inputs["target_screen_csv"])
    eligible = [row for row in rows if row["eligible_for_selection"] == "True"]
    eligible.sort(
        key=lambda row: (
            -int(row["metadata_eligible_count"]),
            -int(row["dude_clustered_active_count"]),
            row["target_id"],
        )
    )
    eligible_order = [row["target_id"] for row in eligible]
    remaining = [
        value
        for value in eligible_order
        if value not in set(config["selection_rule"]["completed_targets"])
    ]
    if eligible_order != ["PPARA", "PPARD", "ESR2"] or remaining[0] != "PPARD":
        raise ValueError("Stage 55 outcome-blind target selection differs")
    if result["selection"]["selected_target"] != remaining[0]:
        raise ValueError("Stage 55 selected target ledger differs")

    for label in ("actives", "decoys"):
        observed = ism_identity(inputs[label])
        expected = config["source_expectations"][label]
        for key, value in observed.items():
            if value != int(expected[key]):
                raise ValueError(f"Stage 55 {label} identity differs: {key}")
    if result["dude_source"]["actives"]["row_count"] != 240:
        raise ValueError("Stage 55 active result count differs")
    if result["dude_source"]["decoys"]["row_count"] != 12250:
        raise ValueError("Stage 55 decoy result count differs")

    pdb_lines = inputs["official_pdb"].read_text(
        encoding="ascii", errors="replace"
    ).splitlines()
    ligand = [
        line
        for line in pdb_lines
        if line.startswith("HETATM")
        and line[17:20].strip() == "K55"
        and line[21:22] == "A"
        and line[22:26].strip() == "922"
    ]
    if len(ligand) != 33:
        raise ValueError("Stage 55 official K55 identity differs")
    ligand_result = result["reference_structure"]["ligand_identity"]
    if (
        ligand_result["heavy_atom_count"] != 33
        or ligand_result["fixed_frame_symmetry_corrected_rmsd_angstrom"] != 0.0
        or not ligand_result["heavy_atom_coordinate_identity"]
        or not ligand_result["protonation_state_difference"]
    ):
        raise ValueError("Stage 55 ligand coordinate identity differs")

    future = read_json(inputs["stage54_future_intake_criteria"])
    if config["frozen_protocol"]["functional_gate"] != future["criteria"]:
        raise ValueError("Stage 55 functional gate differs from Stage54")
    pilot = config["frozen_protocol"]["pilot_panel"]
    structures = config["frozen_protocol"]["structural_pool"]
    docking = config["frozen_protocol"]["docking"]
    pair_budget = (int(pilot["active_count"]) + int(pilot["decoy_count"])) * int(
        structures["metadata_eligible_count_at_freeze"]
    )
    seeded_budget = pair_budget * int(docking["seed_count"])
    if pair_budget != 4896 or seeded_budget != 14688:
        raise ValueError("Stage 55 pilot budget differs")
    if result["pilot_budget"] != {
        "maximum_receptor_ligand_pairs": pair_budget,
        "maximum_seeded_docking_jobs": seeded_budget,
        "full_training_matrix_before_gate_permitted": False,
    }:
        raise ValueError("Stage 55 pilot budget ledger differs")

    forbidden = (
        "pilot_production_docking_authorized",
        "full_training_matrix_authorized",
        "fresh_validation_authorized",
        "quantum_hardware_authorized",
    )
    if any(result["decision"][key] for key in forbidden):
        raise ValueError("Stage 55 authorization boundary differs")
    if result["data_boundary"] != {
        "source_label_rows_read": 12490,
        "docking_scores_read": 0,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
    }:
        raise ValueError("Stage 55 data boundary differs")

    audit = {
        "schema_version": "1.0",
        "status": "stage55_ppard_small_pilot_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation_path),
        "outcome_blind_target_selection_exact": True,
        "source_multiplicity_exact": True,
        "reference_ligand_identity_exact": True,
        "stage54_functional_gate_exact": True,
        "pilot_budget_exact": True,
        "selection": result["selection"],
        "pilot_budget": result["pilot_budget"],
        "decision": {
            "source_gate_passed": True,
            "ligand_panel_allocation_authorized": True,
            "coordinate_audit_authorized": True,
            "pilot_production_docking_authorized": False,
            "full_training_matrix_authorized": False,
            "fresh_validation_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "data_boundary": result["data_boundary"],
        "interpretation_boundary": result["interpretation_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
