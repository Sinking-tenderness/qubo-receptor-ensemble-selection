"""Download and hard-gate all frozen PPARD structural candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.select_stage13_egfr_coordinate_pool import (
    audit_structure,
    ca_map,
    derive_reference_residues,
    download_one,
    file_sha256,
    ligand_residue_map,
    read_csv,
    read_json,
    select_chain_atoms,
    write_csv,
    write_json,
)


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or file_sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage 56 structural input identity differs: {path}")
    return path


def conformer_id(pdb_id: str) -> str:
    return "PPARD_2ZNP_reference" if pdb_id == "2ZNP" else f"PPARD_{pdb_id}_aligned"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    lines = [
        "# Stage56 PPARD coordinate hard-gate",
        "",
        f"Frozen candidates audited: {counts['audited_count']}.",
        f"Coordinate eligible: {counts['coordinate_eligible_count']}.",
        f"Coordinate excluded: {counts['coordinate_excluded_count']}.",
        f"Required minimum: {counts['minimum_coordinate_eligible_count']}.",
        f"All eligible structures retained: {counts['redocking_pool_count']}.",
        "",
        f"Status: **{summary['status']}**.",
        "",
        summary["interpretation_boundary"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 56 structural implementation differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    stage55 = read_json(inputs["stage55_result"])
    stage55_audit = read_json(inputs["stage55_audit"])
    allocation = read_json(inputs["ligand_allocation_summary"])
    if stage55["status"] != "stage55_ppard_small_pilot_source_and_preregistration_ok":
        raise ValueError("Stage55 PPARD source gate did not pass")
    if stage55_audit["status"] != "stage55_ppard_small_pilot_independent_audit_ok":
        raise ValueError("Stage55 PPARD independent audit did not pass")
    if allocation["status"] != "stage56_ppard_ligand_panels_and_pilot_frozen":
        raise ValueError("Stage56 PPARD ligand allocation did not pass")

    candidates = [
        row
        for row in read_csv(inputs["candidate_metadata_csv"])
        if row.get("target_id") == "PPARD" and row.get("status") == "metadata_eligible"
    ]
    expected_count = int(config["structural_pool"]["metadata_eligible_count_at_freeze"])
    if len(candidates) != expected_count or len({row["pdb_id"] for row in candidates}) != expected_count:
        raise ValueError("PPARD frozen metadata-candidate count differs")
    reference_rows = [row for row in candidates if row["pdb_id"] == "2ZNP"]
    if len(reference_rows) != 1 or reference_rows[0]["selected_auth_chain"] != "A":
        raise ValueError("PPARD 2ZNP reference row is absent or ambiguous")

    outputs = {key: root / str(value) for key, value in config["outputs"].items()}
    protected = [
        path
        for key, path in outputs.items()
        if key not in {"raw_mmcif_directory", "aligned_protein_pdb_directory"}
    ]
    if any(path.exists() for path in protected) and not overwrite:
        raise FileExistsError("Stage56 structural outputs exist; pass --overwrite")

    reference_path = inputs["reference_mmcif"]
    raw_directory = outputs["raw_mmcif_directory"]
    raw_paths = {
        row["pdb_id"]: (
            reference_path
            if row["pdb_id"] == "2ZNP"
            else raw_directory / f"{row['pdb_id']}.cif"
        )
        for row in candidates
    }
    download = config["download"]
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=int(download["workers"])) as executor:
        futures = {
            executor.submit(
                download_one,
                pdb_id,
                path,
                str(download["url_template"]),
                float(download["timeout_seconds"]),
                int(download["maximum_retries"]),
                float(download["retry_backoff_seconds"]),
            ): pdb_id
            for pdb_id, path in raw_paths.items()
        }
        for future in as_completed(futures):
            pdb_id, error = future.result()
            if error:
                errors[pdb_id] = error

    reference = config["reference"]
    structure = gemmi.read_structure(str(reference_path))
    reference_atoms = select_chain_atoms(structure, str(reference["auth_chain"]))
    ligand_id = str(reference["ligand_comp_id"])
    pocket_numbers = [int(value) for value in reference["pocket_residue_numbers"]]
    anchor_numbers = {int(value) for value in reference["anchor_residue_numbers"]}
    if derive_reference_residues(reference_atoms, ligand_id, 6.0) != pocket_numbers:
        raise ValueError("PPARD reference pocket reconstruction differs")
    if derive_reference_residues(reference_atoms, ligand_id, 4.0) != sorted(anchor_numbers):
        raise ValueError("PPARD reference anchor reconstruction differs")
    if len(ca_map(reference_atoms)) != int(reference["visible_protein_ca_count"]):
        raise ValueError("PPARD reference C-alpha count differs")
    reference_ligands = ligand_residue_map(reference_atoms, {ligand_id})
    if len(reference_ligands) != 1:
        raise ValueError("PPARD reference ligand is ambiguous")
    reference_ligand_coords = np.vstack(
        [atom.coord for atom in next(iter(reference_ligands.values()))]
    )

    gate = config["coordinate_gate"]
    audit_rows: list[dict[str, Any]] = []
    aligned_directory = outputs["aligned_protein_pdb_directory"]
    for metadata in sorted(candidates, key=lambda row: row["pdb_id"]):
        pdb_id = metadata["pdb_id"]
        candidate_id = conformer_id(pdb_id)
        if pdb_id in errors:
            audit_rows.append(
                {
                    "conformer_id": candidate_id,
                    "pdb_id": pdb_id,
                    "chain": metadata["selected_auth_chain"],
                    "status": "coordinate_excluded",
                    "exclusion_reasons": "coordinate_file_unavailable",
                    "download_error": errors[pdb_id],
                }
            )
            continue
        aligned_path = aligned_directory / f"{candidate_id}_to_2ZNP_A.pdb"
        try:
            row, _vector, _names = audit_structure(
                pdb_id,
                metadata["selected_auth_chain"],
                {
                    value
                    for value in metadata["qualifying_ligand_ids"].split(";")
                    if value
                },
                raw_paths[pdb_id],
                aligned_path,
                reference_atoms,
                reference_ligand_coords,
                pocket_numbers,
                anchor_numbers,
                gate,
            )
            row["conformer_id"] = candidate_id
            row["title"] = metadata["title"]
            row["resolution_angstrom"] = metadata["resolution_angstrom"]
            row["qualifying_ligand_ids"] = metadata["qualifying_ligand_ids"]
            row["mmcif_path"] = raw_paths[pdb_id].relative_to(root).as_posix()
            if row["status"] == "coordinate_eligible":
                row["aligned_protein_pdb_path"] = aligned_path.relative_to(root).as_posix()
        except Exception as error:
            row = {
                "conformer_id": candidate_id,
                "pdb_id": pdb_id,
                "chain": metadata["selected_auth_chain"],
                "status": "coordinate_excluded",
                "exclusion_reasons": "coordinate_parse_or_audit_error",
                "audit_error": f"{type(error).__name__}: {error}",
                "mmcif_path": raw_paths[pdb_id].relative_to(root).as_posix(),
                "mmcif_sha256": file_sha256(raw_paths[pdb_id]),
            }
        audit_rows.append(row)

    eligible_rows = [row for row in audit_rows if row["status"] == "coordinate_eligible"]
    if len(audit_rows) != expected_count:
        raise ValueError("not every frozen PPARD candidate was audited")
    reference_id = "PPARD_2ZNP_reference"
    if not any(row["conformer_id"] == reference_id for row in eligible_rows):
        raise ValueError("PPARD reference failed coordinate eligibility")
    minimum_count = int(config["structural_pool"]["minimum_coordinate_eligible_count"])
    passed = len(eligible_rows) >= minimum_count
    redocking_rows: list[dict[str, Any]] = []
    if passed:
        ordered = sorted(
            eligible_rows,
            key=lambda row: (row["conformer_id"] != reference_id, row["conformer_id"]),
        )
        redocking_rows = [
            {
                "pool_role": "reference" if row["conformer_id"] == reference_id else "hard_gate_pass",
                "selection_rank": index,
                **row,
            }
            for index, row in enumerate(ordered, start=1)
        ]
    status = (
        "stage56_ppard_coordinate_pool_hard_gate_ok"
        if passed
        else "stage56_ppard_coordinate_pool_insufficient_stop"
    )
    write_csv(outputs["coordinate_audit_csv"], audit_rows)
    write_csv(outputs["eligible_pool_manifest_csv"], eligible_rows)
    if redocking_rows:
        write_csv(outputs["redocking_pool_manifest_csv"], redocking_rows)
    reasons = Counter(
        reason
        for row in audit_rows
        for reason in str(row.get("exclusion_reasons", "")).split(";")
        if reason
    )
    artifacts = {
        key: {
            "path": outputs[key].relative_to(root).as_posix(),
            "sha256": file_sha256(outputs[key]),
        }
        for key in (
            "coordinate_audit_csv",
            "eligible_pool_manifest_csv",
            "redocking_pool_manifest_csv",
        )
        if outputs[key].is_file()
    }
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "counts": {
            "audited_count": len(audit_rows),
            "download_failure_count": len(errors),
            "coordinate_eligible_count": len(eligible_rows),
            "coordinate_excluded_count": len(audit_rows) - len(eligible_rows),
            "minimum_coordinate_eligible_count": minimum_count,
            "redocking_pool_count": len(redocking_rows),
        },
        "coordinate_exclusion_reason_counts": dict(sorted(reasons.items())),
        "artifacts": artifacts,
        "reference": {
            "conformer_id": reference_id,
            "pocket_residue_numbers": pocket_numbers,
            "anchor_residue_numbers": sorted(anchor_numbers),
        },
        "redocking_receptor_ids": [row["conformer_id"] for row in redocking_rows],
        "selection_policy": {
            "hard_gate_only": True,
            "all_hard_gate_passing_structures_retained": passed,
            "max_min_or_outcome_informed_compression_used": False,
        },
        "data_boundary": {
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "qubo_outcomes_read": 0,
            "new_docking_jobs": 0,
        },
        "decision": {
            "coordinate_gate_passed": passed,
            "cognate_redocking_input_preparation_authorized": passed,
            "pilot_production_docking_authorized": False,
            "full_training_matrix_authorized": False,
            "fresh_validation_release_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "next_gate": (
            "prepare every hard-gate-passing PPARD receptor and cognate ligand, then run frozen Uni-Dock redocking"
            if passed
            else "stop PPARD before docking and proceed to the next outcome-unseen eligible target"
        ),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], summary)
    write_report(outputs["report_md"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
