"""Prepare and geometrically audit the FA10 cognate-redocking inputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem

try:
    from .prepare_stage13e_egfr_redocking_inputs import (
        atom_identity,
        build_modelserver_url,
        checked_runtime,
        derive_common_box,
        download_cached,
        file_sha256,
        ligand_residue_map,
        kabsch,
        point_set_rmsd_by_element,
        prepare_ligand,
        read_csv,
        read_json,
        run_checked,
        select_chain_atoms,
        transform_atoms,
        verified,
        write_csv,
        write_json,
    )
except ImportError:
    from prepare_stage13e_egfr_redocking_inputs import (
        atom_identity,
        build_modelserver_url,
        checked_runtime,
        derive_common_box,
        download_cached,
        file_sha256,
        ligand_residue_map,
        kabsch,
        point_set_rmsd_by_element,
        prepare_ligand,
        read_csv,
        read_json,
        run_checked,
        select_chain_atoms,
        transform_atoms,
        verified,
        write_csv,
        write_json,
    )


def read_single_sdf(path: Path, remove_hydrogens: bool) -> Chem.Mol:
    text = path.read_text(encoding="utf-8", errors="replace")
    molblock = text.split("$$$$", 1)[0].rstrip()
    molecule = Chem.MolFromMolBlock(
        molblock,
        removeHs=remove_hydrogens,
        sanitize=True,
    )
    if molecule is None or molecule.GetNumConformers() != 1:
        raise ValueError(f"SDF must contain one parseable 3D molecule: {path}")
    return molecule


def coordinates_and_elements_from_sdf_safe(
    path: Path,
) -> tuple[np.ndarray, list[str], Chem.Mol]:
    molecule = read_single_sdf(path, remove_hydrogens=True)
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    elements = [atom.GetSymbol().upper() for atom in molecule.GetAtoms()]
    return coordinates, elements, molecule


def transform_sdf_safe(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    rotation: np.ndarray,
    translation: np.ndarray,
    add_hydrogens: bool,
) -> None:
    molecule = read_single_sdf(input_path, remove_hydrogens=False)
    before = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    after = before @ rotation + translation
    for atom_index, position in enumerate(after):
        molecule.GetConformer().SetAtomPosition(atom_index, tuple(map(float, position)))
    input_atom_count = molecule.GetNumAtoms()
    if add_hydrogens:
        molecule = Chem.AddHs(molecule, addCoords=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        Chem.MolToMolBlock(molecule, includeStereo=True) + "$$$$\n",
        encoding="ascii",
    )
    summary = {
        "schema_version": "1.0",
        "status": "ok",
        "operation": "Unicode-safe rigid-body SDF transform",
        "input_sdf": input_path.as_posix(),
        "input_sdf_sha256": file_sha256(input_path),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "translation_vector_angstrom": translation.tolist(),
        "input_atom_count": input_atom_count,
        "output_atom_count": molecule.GetNumAtoms(),
        "add_explicit_hydrogens": add_hydrogens,
        "output_sdf": output_path.as_posix(),
        "output_sdf_sha256": file_sha256(output_path),
    }
    write_json(summary_path, summary)


def prepare_ligand_safe(
    input_sdf: Path, output_pdbqt: Path
) -> tuple[dict[str, object], str]:
    with tempfile.TemporaryDirectory(prefix="fa10_ligand_") as temporary:
        temporary_root = Path(temporary)
        temporary_sdf = temporary_root / "ligand.sdf"
        temporary_pdbqt = temporary_root / "ligand.pdbqt"
        shutil.copy2(input_sdf, temporary_sdf)
        audit, variant = prepare_ligand(temporary_sdf, temporary_pdbqt)
        output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temporary_pdbqt, output_pdbqt)
    return audit, variant


def alignment_audit_fa10(
    reference_mmcif: Path,
    raw_mmcif: Path,
    raw_chain: str,
    aligned_pdb: Path,
    required_anchor_numbers: list[int],
) -> tuple[dict[str, object], list[object], np.ndarray, np.ndarray]:
    reference_atoms = select_chain_atoms(
        gemmi.read_structure(str(reference_mmcif)), "A"
    )
    raw_atoms = select_chain_atoms(gemmi.read_structure(str(raw_mmcif)), raw_chain)
    aligned_atoms = select_chain_atoms(gemmi.read_structure(str(aligned_pdb)), "A")
    reference_ca = {
        (atom.resseq, atom.icode): atom
        for atom in reference_atoms
        if atom.kind == "protein" and atom.atom_name == "CA"
    }
    raw_ca = {
        (atom.resseq, atom.icode): atom
        for atom in raw_atoms
        if atom.kind == "protein" and atom.atom_name == "CA"
    }
    matched = sorted(
        key
        for key in set(reference_ca) & set(raw_ca)
        if reference_ca[key].resname == raw_ca[key].resname
    )
    anchor_keys = {(number, "") for number in required_anchor_numbers}
    if not anchor_keys.issubset(matched):
        raise ValueError("FA10 alignment is missing a frozen active-site anchor")
    reference_coordinates = np.vstack([reference_ca[key].coord for key in matched])
    raw_coordinates = np.vstack([raw_ca[key].coord for key in matched])
    rotation, translation = kabsch(raw_coordinates, reference_coordinates)
    transformed = transform_atoms(raw_atoms, rotation, translation)
    transformed_by_id = {
        atom_identity(atom): atom for atom in transformed if atom.kind == "protein"
    }
    aligned_by_id = {
        atom_identity(atom): atom for atom in aligned_atoms if atom.kind == "protein"
    }
    if set(transformed_by_id) != set(aligned_by_id):
        raise ValueError("FA10 aligned receptor atom identities differ")
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
        raise ValueError("FA10 aligned receptor coordinates differ from recomputation")
    aligned_ca = raw_coordinates @ rotation + translation
    aligned_rmsd = float(
        np.sqrt(np.mean(np.sum((aligned_ca - reference_coordinates) ** 2, axis=1)))
    )
    return (
        {
            "method": (
                "proper-rotation Kabsch alignment over all shared, residue-name-matched "
                "FA10 heavy-chain C-alpha atoms"
            ),
            "reference_mmcif": reference_mmcif.as_posix(),
            "reference_mmcif_sha256": file_sha256(reference_mmcif),
            "raw_mmcif": raw_mmcif.as_posix(),
            "raw_mmcif_sha256": file_sha256(raw_mmcif),
            "raw_chain": raw_chain,
            "aligned_protein_pdb": aligned_pdb.as_posix(),
            "aligned_protein_pdb_sha256": file_sha256(aligned_pdb),
            "matched_ca_count": len(matched),
            "required_anchor_residue_numbers": required_anchor_numbers,
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


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if file_sha256(Path(__file__)) != str(implementation["sha256"]).upper():
        raise ValueError("Stage 14c implementation SHA-256 differs")
    for dependency in implementation["dependencies"]:
        record = dict(dependency)
        verified(root / str(record["path"]), str(record["sha256"]))
    runtime = checked_runtime(config)

    inputs: dict[str, Path] = {}
    for key, value in dict(config["inputs"]).items():
        record = dict(value)
        inputs[key] = verified(root / str(record["path"]), str(record["sha256"]))
    selection = read_json(inputs["selection_summary"])
    if selection["status"] != "stage14b_fa10_structural_pool_ok":
        raise ValueError("Stage 14b FA10 structural pool did not pass")
    if any(int(value) != 0 for value in dict(selection["data_boundary"]).values()):
        raise ValueError("Stage 14b crossed a protected data boundary")
    selected_rows = read_csv(inputs["selected_receptor_manifest"])
    expected_count = int(dict(config["expected"])["receptor_count"])
    if len(selected_rows) != expected_count:
        raise ValueError("Stage 14c selected receptor count differs")
    if [row["conformer_id"] for row in selected_rows] != selection["selected_receptor_ids"]:
        raise ValueError("Stage 14c selected receptor order differs")
    if any(row["explicit_covalent_connections"].strip() for row in selected_rows):
        raise ValueError("Stage 14c selected pool contains an explicit covalent connection")
    if any(
        int(row["global_incomplete_standard_amino_acid_residue_count"]) != 0
        for row in selected_rows
    ):
        raise ValueError("Stage 14c selected pool contains an incomplete residue")

    outputs = {
        key: root / str(value) for key, value in dict(config["outputs"]).items()
    }
    file_keys = {
        "receptor_manifest_csv",
        "redocking_case_manifest_csv",
        "common_box_json",
        "summary_json",
    }
    if any(outputs[key].exists() for key in file_keys) and not overwrite:
        raise FileExistsError("Stage 14c outputs exist; pass --overwrite")

    reference = dict(config["reference"])
    anchor_numbers = [int(value) for value in reference["anchor_residue_numbers"]]
    protocol = dict(config["preparation_protocol"])
    ligand_download = dict(config["ligand_download"])
    run_directory = outputs["run_directory"]
    receptor_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    common_coordinate_sets: list[np.ndarray] = []
    for selected in selected_rows:
        conformer_id = selected["conformer_id"]
        pdb_id = selected["pdb_id"]
        ligand_resname = selected["selected_ligand_resname"]
        case_id = f"{pdb_id}_{ligand_resname}"
        case_root = run_directory / "preparation" / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        raw_mmcif = verified(root / selected["mmcif_path"], selected["mmcif_sha256"])
        aligned_pdb = verified(
            root / selected["aligned_protein_pdb_path"],
            selected["aligned_protein_pdb_sha256"],
        )
        alignment, raw_atoms, rotation, translation = alignment_audit_fa10(
            inputs["reference_mmcif"],
            raw_mmcif,
            selected["chain"],
            aligned_pdb,
            anchor_numbers,
        )
        alignment_path = case_root / "alignment_summary.json"
        write_json(alignment_path, alignment)

        receptor_root = case_root / "receptor"
        receptor_summary = receptor_root / "summary.json"
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
            str(receptor_summary),
        ]
        if overwrite:
            command.append("--overwrite")
        run_checked(command, f"FA10 receptor preparation for {conformer_id}")
        receptor_evidence = read_json(receptor_summary)
        if (
            receptor_evidence.get("status") != "ok"
            or receptor_evidence.get("allow_bad_res") is not False
        ):
            raise ValueError(f"FA10 receptor preparation failed: {conformer_id}")
        residue_change = dict(receptor_evidence["residue_count_change"])
        if residue_change["input_protein_only"] != residue_change["output_pdbqt"]:
            raise ValueError(f"FA10 receptor residue count changed: {conformer_id}")

        source_url = build_modelserver_url(
            str(ligand_download["url_template"]),
            pdb_id,
            selected["chain"],
            int(selected["selected_ligand_resseq"]),
        )
        raw_sdf = case_root / f"{case_id}_raw.sdf"
        download_cached(
            source_url,
            raw_sdf,
            float(ligand_download["timeout_seconds"]),
            int(ligand_download["maximum_retries"]),
            float(ligand_download["retry_backoff_seconds"]),
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
            raise ValueError(f"selected FA10 mmCIF ligand is missing: {case_id}")
        ligand_atoms = raw_ligands[ligand_key]
        mmcif_coordinates = np.vstack([atom.coord for atom in ligand_atoms])
        mmcif_elements = [atom.element for atom in ligand_atoms]
        if raw_molecule.GetNumHeavyAtoms() != int(selected["selected_ligand_heavy_atom_count"]):
            raise ValueError(f"FA10 ligand heavy-atom count differs: {case_id}")
        raw_rmsd, raw_maximum = point_set_rmsd_by_element(
            raw_coordinates, raw_elements, mmcif_coordinates, mmcif_elements
        )
        limit = float(protocol["maximum_element_matched_coordinate_error_angstrom"])
        if raw_rmsd > limit or raw_maximum > limit:
            raise ValueError(f"FA10 ModelServer ligand coordinates differ: {case_id}")

        common_sdf = case_root / f"{case_id}_common_frame.sdf"
        common_summary = case_root / "common_frame_transform.json"
        explicit_h_sdf = case_root / f"{case_id}_common_frame_explicitH.sdf"
        explicit_h_summary = case_root / "explicit_h_transform.json"
        for output_sdf, output_summary, add_hydrogens in (
            (common_sdf, common_summary, False),
            (explicit_h_sdf, explicit_h_summary, True),
        ):
            transform_sdf_safe(
                raw_sdf,
                output_sdf,
                output_summary,
                rotation,
                translation,
                add_hydrogens,
            )
        common_coordinates, common_elements, common_molecule = (
            coordinates_and_elements_from_sdf_safe(common_sdf)
        )
        transformed_mmcif = mmcif_coordinates @ rotation + translation
        common_rmsd, common_maximum = point_set_rmsd_by_element(
            common_coordinates, common_elements, transformed_mmcif, mmcif_elements
        )
        if common_rmsd > limit or common_maximum > limit:
            raise ValueError(f"FA10 common-frame ligand coordinates differ: {case_id}")
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
                "receptor_preparation_summary": receptor_summary.relative_to(root).as_posix(),
                "receptor_preparation_summary_sha256": file_sha256(receptor_summary),
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
    if box["minimum_observed_margin_angstrom"] < float(
        box_rule["minimum_crystal_pose_margin_angstrom"]
    ) - 1e-9:
        raise ValueError("FA10 common box margin is too small")
    if max(box["size"].values()) > float(box_rule["maximum_axis_size_angstrom"]):
        raise ValueError("FA10 common box is too large")
    box_record = {
        "schema_version": "1.0",
        "status": "stage14c_fa10_common_box_ok",
        "derivation_rule": box_rule,
        "receptor_count": len(selected_rows),
        **box,
    }
    write_csv(outputs["receptor_manifest_csv"], receptor_rows)
    write_csv(outputs["redocking_case_manifest_csv"], case_rows)
    write_json(outputs["common_box_json"], box_record)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage14c_fa10_redocking_inputs_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "runtime": runtime,
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
        "next_gate": "build and run the 16-receptor by three-seed FA10 Uni-Dock cognate-redocking bundle",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


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
