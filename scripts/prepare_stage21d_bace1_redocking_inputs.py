"""Prepare and geometrically audit the BACE1 cognate-redocking inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scripts.prepare_stage18d_pparg_redocking_inputs import (
        alignment_audit_fa10,
        build_modelserver_url,
        coordinates_and_elements_from_sdf_safe,
        derive_common_box,
        download_cached,
        file_sha256,
        ligand_residue_map,
        point_set_rmsd_by_element,
        prepare_ligand_safe,
        read_csv,
        read_json,
        rooted,
        run_checked,
        transform_sdf_safe,
        verified,
        write_csv,
        write_json,
    )
except ModuleNotFoundError:
    from prepare_stage18d_pparg_redocking_inputs import (
        alignment_audit_fa10,
        build_modelserver_url,
        coordinates_and_elements_from_sdf_safe,
        derive_common_box,
        download_cached,
        file_sha256,
        ligand_residue_map,
        point_set_rmsd_by_element,
        prepare_ligand_safe,
        read_csv,
        read_json,
        rooted,
        run_checked,
        transform_sdf_safe,
        verified,
        write_csv,
        write_json,
    )


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if file_sha256(Path(__file__)) != str(config["implementation"]["sha256"]).upper():
        raise ValueError("Stage 21d implementation SHA-256 differs")
    for dependency in config.get("dependencies", []):
        record = dict(dependency)
        verified(rooted(root, str(record["path"])), str(record["sha256"]))
    inputs: dict[str, Path] = {}
    for key, descriptor in dict(config["inputs"]).items():
        record = dict(descriptor)
        inputs[key] = verified(rooted(root, str(record["path"])), str(record["sha256"]))
    selection = read_json(inputs["selection_summary"])
    if selection["status"] != "stage21c_bace1_preparation_ready_structural_pool_ok":
        raise ValueError("BACE1 preparation-ready structural selection did not pass")
    if any(int(value) != 0 for value in selection["data_boundary"].values()):
        raise ValueError("BACE1 structural selection crossed a protected boundary")
    selected_rows = read_csv(inputs["selected_receptor_manifest"])
    if len(selected_rows) != int(config["expected"]["receptor_count"]):
        raise ValueError("BACE1 selected receptor count differs")
    if [row["conformer_id"] for row in selected_rows] != selection["selected_receptor_ids"]:
        raise ValueError("BACE1 selected receptor order differs")
    if any(
        int(row["global_incomplete_standard_amino_acid_residue_count"])
        for row in selected_rows
    ):
        raise ValueError("BACE1 selected pool contains an incomplete residue")
    if any(row["explicit_covalent_connections"].strip() for row in selected_rows):
        raise ValueError("BACE1 selected pool contains an explicit covalent connection")

    outputs = {key: root / str(value) for key, value in dict(config["outputs"]).items()}
    file_keys = {
        "receptor_manifest_csv",
        "redocking_case_manifest_csv",
        "common_box_json",
        "summary_json",
    }
    if any(outputs[key].exists() for key in file_keys) and not overwrite:
        raise FileExistsError("Stage 21d outputs exist; pass --overwrite")

    anchors = [int(value) for value in config["reference"]["anchor_residue_numbers"]]
    protocol = dict(config["preparation_protocol"])
    download = dict(config["ligand_download"])
    receptor_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    common_coordinate_sets: list[np.ndarray] = []
    for selected in selected_rows:
        conformer_id = selected["conformer_id"]
        pdb_id = selected["pdb_id"]
        ligand_resname = selected["selected_ligand_resname"]
        case_id = f"{pdb_id}_{ligand_resname}"
        case_root = outputs["run_directory"] / "preparation" / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        raw_mmcif = verified(
            rooted(root, selected["mmcif_path"]), selected["mmcif_sha256"]
        )
        aligned_pdb = verified(
            rooted(root, selected["aligned_protein_pdb_path"]),
            selected["aligned_protein_pdb_sha256"],
        )
        alignment, raw_atoms, rotation, translation = alignment_audit_fa10(
            inputs["reference_mmcif"],
            raw_mmcif,
            selected["chain"],
            aligned_pdb,
            anchors,
        )
        alignment["target"] = "BACE1"
        alignment["method"] = (
            "proper-rotation Kabsch alignment over all shared, residue-name-matched "
            "BACE1 C-alpha atoms"
        )
        alignment_path = case_root / "alignment_summary.json"
        write_json(alignment_path, alignment)

        receptor_root = case_root / "receptor"
        summary_path = receptor_root / "summary.json"
        protein_only = receptor_root / f"{conformer_id}_protein_only.pdb"
        prepared_pdb = receptor_root / f"{conformer_id}_prepared.pdb"
        receptor_pdbqt = receptor_root / f"{conformer_id}_receptor.pdbqt"
        command = [
            sys.executable,
            str(inputs["prepare_receptor_script"]),
            "--input-pdb",
            str(aligned_pdb),
            "--chain",
            "A",
            "--protein-only-output",
            str(protein_only),
            "--prepared-pdb-output",
            str(prepared_pdb),
            "--pdbqt-output",
            str(receptor_pdbqt),
            "--summary-output",
            str(summary_path),
        ]
        if overwrite:
            command.append("--overwrite")
        run_checked(command, f"BACE1 receptor preparation for {conformer_id}")
        receptor_evidence = read_json(summary_path)
        if (
            receptor_evidence.get("status") != "ok"
            or receptor_evidence.get("allow_bad_res") is not False
        ):
            raise ValueError(f"BACE1 receptor preparation failed: {conformer_id}")
        residue_change = dict(receptor_evidence["residue_count_change"])
        if residue_change["input_protein_only"] != residue_change["output_pdbqt"]:
            raise ValueError(f"BACE1 receptor residue count changed: {conformer_id}")

        source_url = build_modelserver_url(
            str(download["url_template"]),
            pdb_id,
            selected["chain"],
            int(selected["selected_ligand_resseq"]),
        )
        raw_sdf = case_root / f"{case_id}_raw.sdf"
        download_cached(
            source_url,
            raw_sdf,
            float(download["timeout_seconds"]),
            int(download["maximum_retries"]),
            float(download["retry_backoff_seconds"]),
        )
        raw_coordinates, raw_elements, raw_molecule = (
            coordinates_and_elements_from_sdf_safe(raw_sdf)
        )
        ligand_key = (
            ligand_resname,
            int(selected["selected_ligand_resseq"]),
            selected["selected_ligand_icode"],
        )
        raw_ligands = ligand_residue_map(raw_atoms, {ligand_resname})
        if ligand_key not in raw_ligands:
            raise ValueError(f"selected BACE1 mmCIF ligand is missing: {case_id}")
        ligand_atoms = raw_ligands[ligand_key]
        mmcif_coordinates = np.vstack([atom.coord for atom in ligand_atoms])
        mmcif_elements = [atom.element for atom in ligand_atoms]
        if raw_molecule.GetNumHeavyAtoms() != int(
            selected["selected_ligand_heavy_atom_count"]
        ):
            raise ValueError(f"BACE1 ligand heavy-atom count differs: {case_id}")
        raw_rmsd, raw_maximum = point_set_rmsd_by_element(
            raw_coordinates, raw_elements, mmcif_coordinates, mmcif_elements
        )
        limit = float(protocol["maximum_element_matched_coordinate_error_angstrom"])
        if raw_rmsd > limit or raw_maximum > limit:
            raise ValueError(f"BACE1 ModelServer ligand coordinates differ: {case_id}")

        common_sdf = case_root / f"{case_id}_common_frame.sdf"
        common_summary = case_root / "common_frame_transform.json"
        explicit_h_sdf = case_root / f"{case_id}_common_frame_explicitH.sdf"
        explicit_h_summary = case_root / "explicit_h_transform.json"
        transform_sdf_safe(
            raw_sdf, common_sdf, common_summary, rotation, translation, False
        )
        transform_sdf_safe(
            raw_sdf,
            explicit_h_sdf,
            explicit_h_summary,
            rotation,
            translation,
            True,
        )
        common_coordinates, common_elements, common_molecule = (
            coordinates_and_elements_from_sdf_safe(common_sdf)
        )
        transformed_mmcif = mmcif_coordinates @ rotation + translation
        common_rmsd, common_maximum = point_set_rmsd_by_element(
            common_coordinates,
            common_elements,
            transformed_mmcif,
            mmcif_elements,
        )
        if common_rmsd > limit or common_maximum > limit:
            raise ValueError(f"BACE1 common-frame ligand coordinates differ: {case_id}")
        common_coordinate_sets.append(common_coordinates)
        ligand_pdbqt = case_root / f"{case_id}_ligand.pdbqt"
        ligand_audit, ligand_variant = prepare_ligand_safe(
            explicit_h_sdf, ligand_pdbqt
        )

        receptor_rows.append(
            {
                "conformer_id": conformer_id,
                "pdb_id": pdb_id,
                "chain": "A",
                "receptor_pdbqt": receptor_pdbqt.relative_to(root).as_posix(),
                "receptor_pdbqt_sha256": file_sha256(receptor_pdbqt),
                "receptor_preparation_summary": summary_path.relative_to(root).as_posix(),
                "receptor_preparation_summary_sha256": file_sha256(summary_path),
                "status": "ok",
            }
        )
        case_rows.append(
            {
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
                "raw_sdf_sha256": file_sha256(raw_sdf),
                "reference_sdf": common_sdf.relative_to(root).as_posix(),
                "reference_sdf_sha256": file_sha256(common_sdf),
                "explicit_h_sdf": explicit_h_sdf.relative_to(root).as_posix(),
                "explicit_h_sdf_sha256": file_sha256(explicit_h_sdf),
                "ligand_pdbqt": ligand_pdbqt.relative_to(root).as_posix(),
                "ligand_pdbqt_sha256": file_sha256(ligand_pdbqt),
                "ligand_preparation_variant": ligand_variant,
                "ligand_pdbqt_atom_count": ligand_audit["pdbqt_atom_count"],
                "raw_sdf_to_mmcif_element_matched_rmsd_angstrom": raw_rmsd,
                "raw_sdf_to_mmcif_maximum_atom_distance_angstrom": raw_maximum,
                "common_sdf_to_transformed_mmcif_element_matched_rmsd_angstrom": common_rmsd,
                "common_sdf_to_transformed_mmcif_maximum_atom_distance_angstrom": common_maximum,
                "alignment_summary": alignment_path.relative_to(root).as_posix(),
                "alignment_summary_sha256": file_sha256(alignment_path),
                "status": "ok",
            }
        )

    box_rule = dict(config["common_box_rule"])
    box = derive_common_box(
        common_coordinate_sets,
        float(box_rule["minimum_crystal_pose_margin_angstrom"]),
        float(box_rule["size_increment_angstrom"]),
        float(box_rule["minimum_axis_size_angstrom"]),
        int(box_rule["center_decimal_places"]),
    )
    if max(box["size"].values()) > float(box_rule["maximum_axis_size_angstrom"]):
        raise ValueError("BACE1 common box is too large")
    box_record = {
        "schema_version": "1.0",
        "status": "stage21d_bace1_common_box_ok",
        "derivation_rule": box_rule,
        "receptor_count": len(selected_rows),
        **box,
    }
    write_csv(outputs["receptor_manifest_csv"], receptor_rows)
    write_csv(outputs["redocking_case_manifest_csv"], case_rows)
    write_json(outputs["common_box_json"], box_record)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage21d_bace1_redocking_inputs_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "counts": {
            "selected_receptor_count": len(selected_rows),
            "prepared_receptor_count": len(receptor_rows),
            "prepared_cognate_ligand_count": len(case_rows),
            "failed_case_count": 0,
        },
        "common_box": box_record,
        "maximum_raw_ligand_coordinate_rmsd_angstrom": max(
            float(row["raw_sdf_to_mmcif_element_matched_rmsd_angstrom"])
            for row in case_rows
        ),
        "maximum_common_ligand_coordinate_rmsd_angstrom": max(
            float(row["common_sdf_to_transformed_mmcif_element_matched_rmsd_angstrom"])
            for row in case_rows
        ),
        "data_boundary": {
            "ligand_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "pparg_stage19c_enrichment_rows_read": 0,
        },
        "outputs": {
            key: {
                "path": outputs[key].relative_to(root).as_posix(),
                "sha256": file_sha256(outputs[key]),
            }
            for key in (
                "receptor_manifest_csv",
                "redocking_case_manifest_csv",
                "common_box_json",
            )
        },
        "next_gate": "run the 24-receptor by three-seed BACE1 Uni-Dock cognate-redocking bundle",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
