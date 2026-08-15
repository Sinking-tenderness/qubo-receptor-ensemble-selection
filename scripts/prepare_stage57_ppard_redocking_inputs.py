"""Prepare every frozen PPARD hard-gate receptor and cognate redocking ligand."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import prepare_stage50_ppara_large_pool_redocking_inputs as base


def verify_descriptor(root: Path, value: dict[str, object]) -> Path:
    return base.verified(base.rooted(root, str(value["path"])), str(value["sha256"]))


def validate_inputs(
    config_path: Path, root: Path
) -> tuple[dict[str, object], dict[str, Path], list[dict[str, str]], dict[str, object]]:
    config = base.read_json(config_path)
    if base.file_sha256(Path(__file__)) != str(config["implementation"]["sha256"]).upper():
        raise ValueError("Stage57 preparation implementation SHA-256 differs")
    for dependency in config.get("dependencies", []):
        verify_descriptor(root, dict(dependency))
    inputs = {
        key: verify_descriptor(root, dict(value))
        for key, value in dict(config["inputs"]).items()
    }
    stage56b = base.read_json(inputs["stage56b_result"])
    stage56b_audit = base.read_json(inputs["stage56b_audit"])
    if stage56b.get("status") != "stage56_ppard_coordinate_pool_hard_gate_ok":
        raise ValueError("Stage56b PPARD coordinate pool did not pass")
    if (
        stage56b_audit.get("status")
        != "stage56b_ppard_allocation_and_coordinates_independent_audit_ok"
    ):
        raise ValueError("Stage56b independent audit did not pass")
    if stage56b["decision"]["cognate_redocking_input_preparation_authorized"] is not True:
        raise ValueError("Stage56b did not authorize cognate input preparation")
    rows = base.read_csv(inputs["selected_receptor_manifest"])
    expected = int(config["expected"]["frozen_receptor_count"])
    if len(rows) != expected or len({row["conformer_id"] for row in rows}) != expected:
        raise ValueError("Stage57 frozen receptor count or identity differs")
    if rows[0]["conformer_id"] != "PPARD_2ZNP_reference":
        raise ValueError("Stage57 reference receptor is not first")
    if [row["conformer_id"] for row in rows] != stage56b["redocking_receptor_ids"]:
        raise ValueError("Stage57 receptor order differs from Stage56b")
    if len(
        {
            (row["pdb_id"], row["selected_ligand_resname"], row["selected_ligand_resseq"])
            for row in rows
        }
    ) != expected:
        raise ValueError("Stage57 cognate cases are not unique")
    for row in rows:
        base.verified(base.rooted(root, row["mmcif_path"]), row["mmcif_sha256"])
        base.verified(
            base.rooted(root, row["aligned_protein_pdb_path"]),
            row["aligned_protein_pdb_sha256"],
        )
        if row["explicit_covalent_connections"].strip():
            raise ValueError("Stage57 pool contains an explicit target-ligand covalent connection")
    boundary = {
        "ligand_labels_read": 0,
        "benchmark_docking_scores_read": 0,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
    return config, inputs, rows, {
        "status": "audit_only_ok",
        "target_id": "PPARD",
        "frozen_receptor_count": len(rows),
        "cognate_ligand_count": len(rows),
        "receptor_ids": [row["conformer_id"] for row in rows],
        "data_boundary": boundary,
    }


def checkpoint(
    root: Path, path: Path
) -> tuple[dict[str, object], dict[str, object], np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        record = base.read_json(path)
        if record.get("status") != "stage57_ppard_case_preparation_ok":
            return None
        receptor = dict(record["receptor_row"])
        case = dict(record["case_row"])
        for path_key, hash_key in (
            ("receptor_pdbqt", "receptor_pdbqt_sha256"),
            ("completed_receptor_pdb", "completed_receptor_pdb_sha256"),
            ("receptor_preparation_summary", "receptor_preparation_summary_sha256"),
        ):
            base.verified(base.rooted(root, str(receptor[path_key])), str(receptor[hash_key]))
        for path_key, hash_key in (
            ("raw_sdf", "raw_sdf_sha256"),
            ("reference_sdf", "reference_sdf_sha256"),
            ("explicit_h_sdf", "explicit_h_sdf_sha256"),
            ("ligand_pdbqt", "ligand_pdbqt_sha256"),
            ("alignment_summary", "alignment_summary_sha256"),
        ):
            base.verified(base.rooted(root, str(case[path_key])), str(case[hash_key]))
        coordinates, _, _ = base.coordinates_and_elements_from_sdf_safe(
            base.rooted(root, str(case["reference_sdf"]))
        )
        return receptor, case, coordinates
    except (KeyError, OSError, TypeError, ValueError):
        return None


def box_overflow(coordinates: np.ndarray, box: dict[str, object]) -> float:
    center = np.asarray(
        [box["center_x"], box["center_y"], box["center_z"]], dtype=float
    )
    half_size = np.asarray(
        [box["size_x"], box["size_y"], box["size_z"]], dtype=float
    ) / 2.0
    return float(np.max(np.abs(coordinates - center) - half_size))


def alignment_audit_ppard(
    reference_mmcif: Path,
    raw_mmcif: Path,
    raw_chain: str,
    aligned_pdb: Path,
    required_anchor_numbers: list[int],
    residue_window: list[int],
) -> tuple[dict[str, object], list[object], np.ndarray, np.ndarray]:
    import gemmi

    from scripts.prepare_stage13e_egfr_redocking_inputs import (
        atom_identity,
        kabsch,
        select_chain_atoms,
        transform_atoms,
    )

    minimum_residue, maximum_residue = map(int, residue_window)
    reference_atoms = select_chain_atoms(
        gemmi.read_structure(str(reference_mmcif)), "A"
    )
    raw_atoms = select_chain_atoms(gemmi.read_structure(str(raw_mmcif)), raw_chain)
    aligned_atoms = select_chain_atoms(gemmi.read_structure(str(aligned_pdb)), "A")

    def corresponding_protein(atom: object) -> bool:
        return (
            atom.kind == "protein"
            and minimum_residue <= int(atom.resseq) <= maximum_residue
        )

    reference_ca = {
        (atom.resseq, atom.icode): atom
        for atom in reference_atoms
        if corresponding_protein(atom) and atom.atom_name == "CA"
    }
    raw_ca = {
        (atom.resseq, atom.icode): atom
        for atom in raw_atoms
        if corresponding_protein(atom) and atom.atom_name == "CA"
    }
    matched = sorted(
        key
        for key in set(reference_ca) & set(raw_ca)
        if reference_ca[key].resname == raw_ca[key].resname
    )
    anchor_keys = {(number, "") for number in required_anchor_numbers}
    if not anchor_keys.issubset(matched):
        raise ValueError("PPARD alignment is missing a frozen active-site anchor")
    reference_coordinates = np.vstack([reference_ca[key].coord for key in matched])
    raw_coordinates = np.vstack([raw_ca[key].coord for key in matched])
    rotation, translation = kabsch(raw_coordinates, reference_coordinates)
    transformed = transform_atoms(raw_atoms, rotation, translation)
    transformed_by_id = {
        atom_identity(atom): atom
        for atom in transformed
        if corresponding_protein(atom)
    }
    aligned_by_id = {
        atom_identity(atom): atom
        for atom in aligned_atoms
        if corresponding_protein(atom)
    }
    if set(transformed_by_id) != set(aligned_by_id):
        missing = len(set(transformed_by_id).difference(aligned_by_id))
        extra = len(set(aligned_by_id).difference(transformed_by_id))
        raise ValueError(
            "PPARD sequence-corresponding aligned receptor atom identities differ: "
            f"missing={missing}, extra={extra}"
        )
    maximum_error = max(
        float(
            np.max(
                np.abs(
                    transformed_by_id[identity].coord
                    - aligned_by_id[identity].coord
                )
            )
        )
        for identity in transformed_by_id
    )
    if maximum_error > 0.0011:
        raise ValueError("PPARD aligned receptor coordinates differ from recomputation")
    aligned_ca = raw_coordinates @ rotation + translation
    aligned_rmsd = float(
        np.sqrt(np.mean(np.sum((aligned_ca - reference_coordinates) ** 2, axis=1)))
    )
    return (
        {
            "method": (
                "proper-rotation Kabsch alignment over sequence-corresponding, "
                "residue-name-matched PPARD C-alpha atoms"
            ),
            "identity_audit_scope": (
                "protein atoms in the frozen 2ZNP sequence-corresponding residue "
                f"window {minimum_residue}-{maximum_residue}; synthetic negative "
                "unmapped tag/flank residues are excluded"
            ),
            "reference_mmcif": reference_mmcif.as_posix(),
            "reference_mmcif_sha256": base.file_sha256(reference_mmcif),
            "raw_mmcif": raw_mmcif.as_posix(),
            "raw_mmcif_sha256": base.file_sha256(raw_mmcif),
            "raw_chain": raw_chain,
            "aligned_protein_pdb": aligned_pdb.as_posix(),
            "aligned_protein_pdb_sha256": base.file_sha256(aligned_pdb),
            "matched_ca_count": len(matched),
            "required_anchor_residue_numbers": required_anchor_numbers,
            "target_sequence_residue_window": [minimum_residue, maximum_residue],
            "sequence_corresponding_atom_count": len(transformed_by_id),
            "aligned_global_ca_rmsd_angstrom": aligned_rmsd,
            "rotation_determinant": float(np.linalg.det(rotation)),
            "rotation_matrix_row_vector_convention": rotation.tolist(),
            "translation_vector_angstrom": translation.tolist(),
            "maximum_coordinate_difference_from_selected_aligned_pdb_angstrom": maximum_error,
        },
        raw_atoms,
        rotation,
        translation,
    )


def prepare_case(
    selected: dict[str, str],
    config: dict[str, object],
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    root: Path,
) -> tuple[dict[str, object], dict[str, object], np.ndarray]:
    conformer_id = selected["conformer_id"]
    pdb_id = selected["pdb_id"]
    ligand_resname = selected["selected_ligand_resname"]
    case_id = f"{pdb_id}_{ligand_resname}"
    case_root = outputs["run_directory"] / "preparation" / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    raw_mmcif = base.verified(
        base.rooted(root, selected["mmcif_path"]), selected["mmcif_sha256"]
    )
    aligned_pdb = base.verified(
        base.rooted(root, selected["aligned_protein_pdb_path"]),
        selected["aligned_protein_pdb_sha256"],
    )
    anchors = [int(value) for value in config["reference"]["anchor_residue_numbers"]]
    residue_window = list(
        dict(config["preparation_protocol"])["target_sequence_residue_window"]
    )
    alignment, raw_atoms, rotation, translation = alignment_audit_ppard(
        inputs["reference_mmcif"],
        raw_mmcif,
        selected["chain"],
        aligned_pdb,
        anchors,
        residue_window,
    )
    if int(alignment["matched_ca_count"]) != int(selected["matched_ca_count"]):
        raise ValueError(f"PPARD Stage56b matched-CA count changed: {conformer_id}")
    alignment["target"] = "PPARD"
    alignment["method"] = (
        "proper-rotation Kabsch alignment over sequence-corresponding, "
        "residue-name-matched PPARD C-alpha atoms"
    )
    alignment_path = case_root / "alignment_summary.json"
    base.write_json(alignment_path, alignment)

    receptor_root = case_root / "receptor"
    summary_path = receptor_root / "summary.json"
    target_sequence_pdb = receptor_root / f"{conformer_id}_target_sequence.pdb"
    completed_pdb = receptor_root / f"{conformer_id}_completed.pdb"
    protein_only = receptor_root / f"{conformer_id}_protein_only.pdb"
    prepared_pdb = receptor_root / f"{conformer_id}_prepared.pdb"
    receptor_pdbqt = receptor_root / f"{conformer_id}_receptor.pdbqt"
    disulfide = dict(config["preparation_protocol"])["disulfide_rule"]
    command = [
        sys.executable,
        str(inputs["prepare_receptor_script"]),
        "--input-pdb",
        str(aligned_pdb),
        "--chain",
        "A",
        "--minimum-residue-number",
        str(residue_window[0]),
        "--maximum-residue-number",
        str(residue_window[1]),
        "--minimum-disulfide-distance",
        str(disulfide["minimum_sg_distance_angstrom"]),
        "--maximum-disulfide-distance",
        str(disulfide["maximum_sg_distance_angstrom"]),
        "--target-sequence-pdb-output",
        str(target_sequence_pdb),
        "--completed-pdb-output",
        str(completed_pdb),
        "--protein-only-output",
        str(protein_only),
        "--prepared-pdb-output",
        str(prepared_pdb),
        "--pdbqt-output",
        str(receptor_pdbqt),
        "--summary-output",
        str(summary_path),
        "--overwrite",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"PPARD receptor preparation failed with return code {completed.returncode}")
    receptor_evidence = base.read_json(summary_path)
    if receptor_evidence.get("status") != "ok" or receptor_evidence.get("allow_bad_res") is not False:
        raise ValueError(f"PPARD receptor preparation evidence failed: {conformer_id}")
    maximum_displacement = float(
        receptor_evidence["maximum_existing_atom_displacement_angstrom"]
    )
    if maximum_displacement > float(
        config["preparation_protocol"]["maximum_existing_atom_displacement_angstrom"]
    ):
        raise ValueError(f"PPARD completion moved existing atoms: {conformer_id}")
    residue_change = dict(receptor_evidence["residue_count_change"])
    if len(set(residue_change.values())) != 1:
        raise ValueError(f"PPARD receptor residue count changed: {conformer_id}")

    download = dict(config["ligand_download"])
    source_url = base.build_modelserver_url(
        str(download["url_template"]),
        pdb_id,
        selected["chain"],
        int(selected["selected_ligand_resseq"]),
    )
    raw_sdf = case_root / f"{case_id}_raw.sdf"
    base.download_cached(
        source_url,
        raw_sdf,
        float(download["timeout_seconds"]),
        int(download["maximum_retries"]),
        float(download["retry_backoff_seconds"]),
    )
    raw_coordinates, raw_elements, raw_molecule = base.coordinates_and_elements_from_sdf_safe(raw_sdf)
    ligand_key = (
        ligand_resname,
        int(selected["selected_ligand_resseq"]),
        selected["selected_ligand_icode"],
    )
    raw_ligands = base.ligand_residue_map(raw_atoms, {ligand_resname})
    if ligand_key not in raw_ligands:
        raise ValueError(f"selected PPARD mmCIF ligand is missing: {case_id}")
    ligand_atoms = raw_ligands[ligand_key]
    mmcif_coordinates = np.vstack([atom.coord for atom in ligand_atoms])
    mmcif_elements = [atom.element for atom in ligand_atoms]
    if raw_molecule.GetNumHeavyAtoms() != int(selected["selected_ligand_heavy_atom_count"]):
        raise ValueError(f"PPARD ligand heavy-atom count differs: {case_id}")
    raw_rmsd, raw_maximum = base.point_set_rmsd_by_element(
        raw_coordinates, raw_elements, mmcif_coordinates, mmcif_elements
    )
    limit = float(
        config["preparation_protocol"]["maximum_element_matched_coordinate_error_angstrom"]
    )
    if raw_rmsd > limit or raw_maximum > limit:
        raise ValueError(f"PPARD ModelServer ligand coordinates differ: {case_id}")

    common_sdf = case_root / f"{case_id}_common_frame.sdf"
    explicit_h_sdf = case_root / f"{case_id}_common_frame_explicitH.sdf"
    base.transform_sdf_safe(
        raw_sdf,
        common_sdf,
        case_root / "common_frame_transform.json",
        rotation,
        translation,
        False,
    )
    base.transform_sdf_safe(
        raw_sdf,
        explicit_h_sdf,
        case_root / "explicit_h_transform.json",
        rotation,
        translation,
        True,
    )
    common_coordinates, common_elements, common_molecule = base.coordinates_and_elements_from_sdf_safe(common_sdf)
    transformed_mmcif = mmcif_coordinates @ rotation + translation
    common_rmsd, common_maximum = base.point_set_rmsd_by_element(
        common_coordinates, common_elements, transformed_mmcif, mmcif_elements
    )
    if common_rmsd > limit or common_maximum > limit:
        raise ValueError(f"PPARD common-frame ligand coordinates differ: {case_id}")
    overflow = box_overflow(common_coordinates, dict(config["frozen_common_box"]))
    if overflow > float(config["preparation_protocol"]["maximum_box_overflow_angstrom"]):
        raise ValueError(f"PPARD cognate pose lies outside the frozen common box: {case_id}")
    ligand_pdbqt = case_root / f"{case_id}_ligand.pdbqt"
    ligand_audit, ligand_variant = base.prepare_ligand_safe(explicit_h_sdf, ligand_pdbqt)

    receptor_row = {
        "conformer_id": conformer_id,
        "pdb_id": pdb_id,
        "chain": "A",
        "receptor_pdbqt": receptor_pdbqt.relative_to(root).as_posix(),
        "receptor_pdbqt_sha256": base.file_sha256(receptor_pdbqt),
        "completed_receptor_pdb": completed_pdb.relative_to(root).as_posix(),
        "completed_receptor_pdb_sha256": base.file_sha256(completed_pdb),
        "receptor_preparation_summary": summary_path.relative_to(root).as_posix(),
        "receptor_preparation_summary_sha256": base.file_sha256(summary_path),
        "completed_missing_heavy_atom_count": receptor_evidence["completed_heavy_atom_count"],
        "removed_noncorresponding_residue_count": receptor_evidence["target_sequence_filter"]["removed_noncorresponding_residue_count"],
        "cyx_template": receptor_evidence["cyx_template"],
        "status": "ok",
    }
    case_row = {
        "case_id": case_id,
        "conformer_id": conformer_id,
        "pdb_id": pdb_id,
        "source_chain": selected["chain"],
        "selected_ligand_resname": ligand_key[0],
        "selected_ligand_resseq": ligand_key[1],
        "selected_ligand_icode": ligand_key[2],
        "selected_ligand_heavy_atom_count": common_molecule.GetNumHeavyAtoms(),
        "source_url": source_url,
        "raw_sdf": raw_sdf.relative_to(root).as_posix(),
        "raw_sdf_sha256": base.file_sha256(raw_sdf),
        "reference_sdf": common_sdf.relative_to(root).as_posix(),
        "reference_sdf_sha256": base.file_sha256(common_sdf),
        "explicit_h_sdf": explicit_h_sdf.relative_to(root).as_posix(),
        "explicit_h_sdf_sha256": base.file_sha256(explicit_h_sdf),
        "ligand_pdbqt": ligand_pdbqt.relative_to(root).as_posix(),
        "ligand_pdbqt_sha256": base.file_sha256(ligand_pdbqt),
        "ligand_preparation_variant": ligand_variant,
        "ligand_pdbqt_atom_count": ligand_audit["pdbqt_atom_count"],
        "raw_sdf_to_mmcif_element_matched_rmsd_angstrom": raw_rmsd,
        "raw_sdf_to_mmcif_maximum_atom_distance_angstrom": raw_maximum,
        "common_sdf_to_transformed_mmcif_element_matched_rmsd_angstrom": common_rmsd,
        "common_sdf_to_transformed_mmcif_maximum_atom_distance_angstrom": common_maximum,
        "frozen_box_overflow_angstrom": overflow,
        "alignment_summary": alignment_path.relative_to(root).as_posix(),
        "alignment_summary_sha256": base.file_sha256(alignment_path),
        "status": "ok",
    }
    base.write_json(
        case_root / "case_preparation_summary.json",
        {
            "schema_version": "1.0",
            "status": "stage57_ppard_case_preparation_ok",
            "receptor_row": receptor_row,
            "case_row": case_row,
        },
    )
    return receptor_row, case_row, common_coordinates


def run(config_path: Path, root: Path, audit_only: bool, resume: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config, inputs, selected_rows, input_audit = validate_inputs(config_path, root)
    if audit_only:
        record = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "operation": "input audit only; no download, preparation, or docking was started",
            **input_audit,
        }
        print(json.dumps(record, indent=2, sort_keys=True))
        return record

    base.load_dependencies()
    outputs = {
        key: base.rooted(root, str(value))
        for key, value in dict(config["outputs"]).items()
    }
    receptor_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    resumed_count = 0
    prepared_count = 0
    for index, selected in enumerate(selected_rows, start=1):
        case_id = f"{selected['pdb_id']}_{selected['selected_ligand_resname']}"
        checkpoint_path = (
            outputs["run_directory"]
            / "preparation"
            / case_id
            / "case_preparation_summary.json"
        )
        cached = checkpoint(root, checkpoint_path) if resume else None
        if cached is None:
            print(f"preparing {index}/{len(selected_rows)}: {selected['conformer_id']}", flush=True)
            try:
                receptor, case, _coordinates = prepare_case(
                    selected, config, inputs, outputs, root
                )
                prepared_count += 1
            except Exception as error:
                receptor_summary = checkpoint_path.parent / "receptor" / "summary.json"
                evidence = base.read_json(receptor_summary) if receptor_summary.is_file() else {}
                failure = {
                    "case_id": case_id,
                    "conformer_id": selected["conformer_id"],
                    "pdb_id": selected["pdb_id"],
                    "error": f"{type(error).__name__}: {error}",
                    "receptor_preparation_error": evidence.get("error", ""),
                    "meeko_stderr": str(evidence.get("meeko_stderr", ""))[-4000:],
                    "status": "technical_preparation_failed",
                }
                base.write_json(checkpoint_path.parent / "case_preparation_failure.json", failure)
                failure_rows.append(failure)
                print(
                    f"preparation failed {index}/{len(selected_rows)}: "
                    f"{selected['conformer_id']}; continuing",
                    flush=True,
                )
                continue
        else:
            receptor, case, _coordinates = cached
            resumed_count += 1
            print(f"resume ok {index}/{len(selected_rows)}: {selected['conformer_id']}", flush=True)
        receptor_rows.append(receptor)
        case_rows.append(case)

    minimum_prepared = int(config["expected"]["minimum_prepared_receptor_count"])
    gate_ready = len(receptor_rows) >= minimum_prepared
    if len(receptor_rows) == len(selected_rows):
        status = "stage57_ppard_redocking_inputs_ok"
    elif gate_ready:
        status = "stage57_ppard_redocking_inputs_partial_gate_ready"
    else:
        status = "stage57_ppard_redocking_inputs_failed"
    common_box = {
        "schema_version": "1.0",
        "status": "stage57_ppard_frozen_common_box_ok",
        "derivation_rule": config["common_box_rule"],
        "receptor_count": len(receptor_rows),
        "center": {
            "x": config["frozen_common_box"]["center_x"],
            "y": config["frozen_common_box"]["center_y"],
            "z": config["frozen_common_box"]["center_z"],
        },
        "size": {
            "x": config["frozen_common_box"]["size_x"],
            "y": config["frozen_common_box"]["size_y"],
            "z": config["frozen_common_box"]["size_z"],
        },
        "box": config["frozen_common_box"],
    }
    if receptor_rows:
        base.write_csv(outputs["receptor_manifest_csv"], receptor_rows)
        base.write_csv(outputs["redocking_case_manifest_csv"], case_rows)
        base.write_json(outputs["common_box_json"], common_box)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": base.file_sha256(config_path),
        },
        "counts": {
            "frozen_receptor_count": len(selected_rows),
            "prepared_receptor_count": len(receptor_rows),
            "prepared_cognate_ligand_count": len(case_rows),
            "prepared_this_invocation": prepared_count,
            "resumed_this_invocation": resumed_count,
            "failed_case_count": len(failure_rows),
            "minimum_prepared_receptor_count": minimum_prepared,
            "completed_missing_heavy_atom_count": sum(
                int(row["completed_missing_heavy_atom_count"]) for row in receptor_rows
            ),
            "removed_noncorresponding_residue_count": sum(
                int(row["removed_noncorresponding_residue_count"]) for row in receptor_rows
            ),
            "cyx_typed_receptor_count": sum(bool(row["cyx_template"]) for row in receptor_rows),
        },
        "technical_gate_ready": gate_ready,
        "prepared_receptor_ids": [row["conformer_id"] for row in receptor_rows],
        "failed_cases": failure_rows,
        "common_box": common_box,
        "data_boundary": input_audit["data_boundary"],
        "outputs": {},
        "next_gate": (
            "run frozen three-seed PPARD Uni-Dock cognate redocking for every prepared receptor"
            if gate_ready
            else "stop PPARD because fewer than 24 frozen receptors were technically preparation-ready"
        ),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    if receptor_rows:
        result["outputs"] = {
            key: {
                "path": outputs[key].relative_to(root).as_posix(),
                "sha256": base.file_sha256(outputs[key]),
            }
            for key in (
                "receptor_manifest_csv",
                "redocking_case_manifest_csv",
                "common_box_json",
            )
        }
    base.write_json(outputs["summary_json"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.root, args.audit_only, args.resume)
    return 1 if str(result.get("status", "")).endswith("_failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
