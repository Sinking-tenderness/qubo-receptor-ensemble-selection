"""Independently audit the Stage48 PPARA source and frozen protocol result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def audit(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    result_path = root / "data/stage48_ppara_source_audit.json"
    result = read_json(result_path)
    if result.get("status") != "stage48_ppara_source_audit_ok":
        raise ValueError("Stage48 source result is incomplete")
    config_path = root / result["config"]["path"]
    if sha256(config_path) != result["config"]["sha256"]:
        raise ValueError("Stage48 config identity mismatch")
    config = read_json(config_path)
    implementation = root / config["implementation"]["path"]
    if sha256(implementation) != config["implementation"]["sha256"]:
        raise ValueError("Stage48 implementation identity mismatch")
    for value in config["inputs"].values():
        if sha256(root / value["path"]) != value["sha256"]:
            raise ValueError(f"Stage48 input identity mismatch: {value['path']}")

    screen = read_json(root / config["inputs"]["target_screen"]["path"])
    selected = screen["selected_target"]
    if selected["target_id"] != "PPARA" or int(selected["metadata_eligible_count"]) != 75:
        raise ValueError("Stage47b selected-target record differs")
    active = result["dude_source"]["actives"]
    decoy = result["dude_source"]["decoys"]
    if (active["row_count"], active["rdkit_failure_count"]) != (373, 0):
        raise ValueError("PPARA active source audit differs")
    if (decoy["row_count"], decoy["rdkit_failure_count"]) != (19399, 0):
        raise ValueError("PPARA decoy source audit differs")
    ligand = result["reference_structure"]["ligand_identity"]
    if (
        ligand["heavy_atom_count"] != 33
        or ligand["fixed_frame_symmetry_corrected_rmsd_angstrom"] != 0.0
        or not ligand["protonation_state_difference"]
    ):
        raise ValueError("PPARA reference ligand identity record differs")
    allocation = result["frozen_protocol"]["ligand_allocation"]
    active_total = sum(allocation[key]["active_count"] for key in ("development_train", "fresh_validation", "locked_test"))
    if active_total != 373:
        raise ValueError("PPARA frozen active allocation does not exhaust the source")
    if result["frozen_protocol"]["qubo"]["primary_subset_size"] != 6:
        raise ValueError("PPARA primary subset size differs")
    boundary = result["data_boundary"]
    protected = ("docking_scores_read", "fresh_validation_rows_read", "locked_test_rows_read", "new_docking_jobs", "quantum_hardware_jobs")
    if any(boundary[key] != 0 for key in protected):
        raise ValueError("Stage48 crossed a protected evidence boundary")

    audit_result = {
        "schema_version": "1.0",
        "status": "stage48_ppara_source_independent_audit_ok",
        "audited_result": {"path": result_path.relative_to(root).as_posix(), "sha256": sha256(result_path)},
        "selected_target": "PPARA",
        "metadata_eligible_count": 75,
        "source_row_counts": {"active": 373, "decoy": 19399},
        "reference_ligand_heavy_atom_identity": True,
        "protonation_difference_recorded": True,
        "frozen_active_allocation_total": active_total,
        "primary_subset_size": 6,
        "evidence_boundary_ok": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/stage48_ppara_source_audit_independent.json"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    audit(args.root, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
