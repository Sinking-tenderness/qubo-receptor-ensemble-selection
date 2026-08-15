"""Prepare and geometrically audit EGFR cognate-redocking inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
from pathlib import Path

import gemmi
import numpy as np

try:
    from .batch_prepare_ligand_pdbqt import find_meeko_script, parse_pdbqt
    from .run_mk14_expanded_redocking_gate import (
        build_modelserver_url,
        coordinates_and_elements_from_sdf,
        download_cached,
        point_set_rmsd_by_element,
        run_checked,
    )
    from .select_stage13_egfr_coordinate_pool import (
        file_sha256,
        kabsch,
        ligand_residue_map,
        read_csv,
        read_json,
        select_chain_atoms,
        transform_atoms,
        verified,
        write_csv,
        write_json,
    )
    from .select_stage13c_egfr_local_pocket_pool import ca_atom_map
    from .experimental.unidock.run_unidock_gpu_equivalence import (
        macrocycle_closure_atom_types,
    )
except ImportError:
    from batch_prepare_ligand_pdbqt import find_meeko_script, parse_pdbqt
    from run_mk14_expanded_redocking_gate import (
        build_modelserver_url,
        coordinates_and_elements_from_sdf,
        download_cached,
        point_set_rmsd_by_element,
        run_checked,
    )
    from select_stage13_egfr_coordinate_pool import (
        file_sha256,
        kabsch,
        ligand_residue_map,
        read_csv,
        read_json,
        select_chain_atoms,
        transform_atoms,
        verified,
        write_csv,
        write_json,
    )
    from select_stage13c_egfr_local_pocket_pool import ca_atom_map
    from experimental.unidock.run_unidock_gpu_equivalence import (
        macrocycle_closure_atom_types,
    )


def derive_common_box(
    coordinate_sets: list[np.ndarray],
    minimum_margin_angstrom: float,
    size_increment_angstrom: float,
    minimum_axis_size_angstrom: float,
    center_decimals: int,
) -> dict[str, object]:
    if not coordinate_sets:
        raise ValueError("cannot derive a box without ligand coordinates")
    coordinates = np.vstack(coordinate_sets)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    center = np.round((minimum + maximum) / 2.0, center_decimals)
    sizes: list[float] = []
    margins: list[float] = []
    for axis in range(3):
        half_needed = max(
            float(center[axis] - minimum[axis]),
            float(maximum[axis] - center[axis]),
        ) + minimum_margin_angstrom
        raw_size = 2.0 * half_needed
        size = math.ceil(raw_size / size_increment_angstrom) * size_increment_angstrom
        size = max(size, minimum_axis_size_angstrom)
        sizes.append(float(size))
        margins.extend(
            [
                float(minimum[axis] - (center[axis] - size / 2.0)),
                float((center[axis] + size / 2.0) - maximum[axis]),
            ]
        )
    return {
        "center": {
            axis: float(center[index]) for index, axis in enumerate(("x", "y", "z"))
        },
        "size": {
            axis: sizes[index] for index, axis in enumerate(("x", "y", "z"))
        },
        "ligand_minimum_angstrom": minimum.tolist(),
        "ligand_maximum_angstrom": maximum.tolist(),
        "minimum_observed_margin_angstrom": min(margins),
    }


def citation_intent_matches(title: str, patterns: list[str]) -> list[str]:
    """Return diagnostic citation patterns without changing receptor admission."""
    return [pattern for pattern in patterns if re.search(pattern, title, re.IGNORECASE)]


def atom_identity(atom: object) -> tuple[object, ...]:
    return (
        atom.resseq,
        atom.icode,
        atom.resname,
        atom.atom_name,
        atom.element,
    )


def alignment_audit(
    reference_mmcif: Path,
    raw_mmcif: Path,
    raw_chain: str,
    aligned_pdb: Path,
    anchor_numbers: list[int],
) -> tuple[dict[str, object], list[object], np.ndarray, np.ndarray]:
    reference_structure = gemmi.read_structure(str(reference_mmcif))
    raw_structure = gemmi.read_structure(str(raw_mmcif))
    aligned_structure = gemmi.read_structure(str(aligned_pdb))
    reference_atoms = select_chain_atoms(reference_structure, "A")
    raw_atoms = select_chain_atoms(raw_structure, raw_chain)
    aligned_atoms = select_chain_atoms(aligned_structure, "A")
    reference_ca = ca_atom_map(reference_atoms)
    raw_ca = ca_atom_map(raw_atoms)
    keys = [(number, "") for number in anchor_numbers]
    if any(key not in reference_ca or key not in raw_ca for key in keys):
        raise ValueError("redocking alignment is missing a frozen anchor C-alpha")
    if any(reference_ca[key].resname != raw_ca[key].resname for key in keys):
        raise ValueError("redocking alignment anchor residue names differ")
    reference_coordinates = np.vstack([reference_ca[key].coord for key in keys])
    raw_coordinates = np.vstack([raw_ca[key].coord for key in keys])
    rotation, translation = kabsch(raw_coordinates, reference_coordinates)
    transformed = transform_atoms(raw_atoms, rotation, translation)
    transformed_by_id = {
        atom_identity(atom): atom
        for atom in transformed
        if atom.kind == "protein"
    }
    aligned_by_id = {
        atom_identity(atom): atom
        for atom in aligned_atoms
        if atom.kind == "protein"
    }
    if set(transformed_by_id) != set(aligned_by_id):
        raise ValueError("aligned receptor atom identities differ from recomputation")
    coordinate_error = max(
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
    if coordinate_error > 0.0011:
        raise ValueError("aligned receptor coordinates differ from recomputation")
    aligned_anchor = raw_coordinates @ rotation + translation
    anchor_rmsd = float(
        np.sqrt(
            np.mean(np.sum((aligned_anchor - reference_coordinates) ** 2, axis=1))
        )
    )
    return (
        {
            "method": "proper-rotation Kabsch alignment over the 20 frozen ATP-pocket anchor C-alpha atoms",
            "reference_mmcif": reference_mmcif.as_posix(),
            "reference_mmcif_sha256": file_sha256(reference_mmcif),
            "raw_mmcif": raw_mmcif.as_posix(),
            "raw_mmcif_sha256": file_sha256(raw_mmcif),
            "raw_chain": raw_chain,
            "aligned_protein_pdb": aligned_pdb.as_posix(),
            "aligned_protein_pdb_sha256": file_sha256(aligned_pdb),
            "anchor_residue_numbers": anchor_numbers,
            "aligned_anchor_ca_rmsd_angstrom": anchor_rmsd,
            "rotation_determinant": float(np.linalg.det(rotation)),
            "rotation_matrix_row_vector_convention": rotation.tolist(),
            "translation_vector_angstrom": translation.tolist(),
            "maximum_coordinate_difference_from_selected_aligned_pdb_angstrom": coordinate_error,
        },
        raw_atoms,
        rotation,
        translation,
    )


def prepare_ligand(explicit_h_sdf: Path, output_path: Path) -> tuple[dict[str, object], str]:
    command = [
        sys.executable,
        str(find_meeko_script()),
        "-i",
        str(explicit_h_sdf),
        "-o",
        str(output_path),
    ]
    run_checked(command, f"ligand preparation for {output_path.stem}")
    variant = "meeko_flexible"
    pseudoatoms = macrocycle_closure_atom_types(output_path)
    if pseudoatoms:
        output_path.unlink()
        run_checked(
            command + ["--rigid_macrocycles"],
            f"rigid-macrocycle preparation for {output_path.stem}",
        )
        variant = "meeko_rigid_macrocycles"
    remaining = macrocycle_closure_atom_types(output_path)
    if remaining:
        raise ValueError(f"Uni-Dock-incompatible macrocycle pseudoatoms remain: {remaining}")
    return parse_pdbqt(output_path), variant


def checked_runtime(config: dict[str, object]) -> dict[str, str]:
    actual = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": importlib.metadata.version("scipy"),
        "rdkit_version": importlib.metadata.version("rdkit"),
        "meeko_version": importlib.metadata.version("meeko"),
        "prody_version": importlib.metadata.version("prody"),
    }
    expected = {key: str(value) for key, value in dict(config["runtime"]).items()}
    if actual != expected:
        raise RuntimeError(f"Stage 13e runtime differs: {actual} != {expected}")
    return actual


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    script_path = verified(
        Path(str(implementation["path"])), str(implementation["sha256"])
    )
    if script_path.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 13e implementation path differs")
    for dependency in implementation["dependencies"]:
        record = dict(dependency)
        verified(Path(str(record["path"])), str(record["sha256"]))
    runtime = checked_runtime(config)

    inputs: dict[str, Path] = {}
    for key, value in dict(config["inputs"]).items():
        record = dict(value)
        inputs[key] = verified(Path(str(record["path"])), str(record["sha256"]))
    selection_summary = read_json(inputs["selection_summary"])
    if selection_summary.get("status") != "stage13d_egfr_preparation_ready_pool_ok":
        raise ValueError("Stage 13e selection input did not pass")
    if any(int(value) != 0 for value in selection_summary["data_boundary"].values()):
        raise ValueError("Stage 13e selection crossed a data boundary")
    selected_rows = read_csv(inputs["selected_receptor_manifest"])
    if len(selected_rows) != int(config["expected"]["receptor_count"]):
        raise ValueError("Stage 13e selected receptor count differs")
    if [row["conformer_id"] for row in selected_rows] != selection_summary["selected_receptor_ids"]:
        raise ValueError("Stage 13e selected receptor order differs")

    protocol = dict(config["preparation_protocol"])
    citation_policy = dict(protocol["citation_intent_diagnostic"])
    citation_patterns = [str(value) for value in citation_policy["patterns"]]
    citation_diagnostics = [
        {
            "conformer_id": row["conformer_id"],
            "pdb_id": row["pdb_id"],
            "primary_citation_title": row["primary_citation_title"],
            "primary_citation_doi": row["primary_citation_doi"],
            "matched_patterns": citation_intent_matches(
                row["primary_citation_title"], citation_patterns
            ),
            "explicit_covalent_connections": row["explicit_covalent_connections"],
        }
        for row in selected_rows
        if citation_intent_matches(row["primary_citation_title"], citation_patterns)
    ]
    if any(row["explicit_covalent_connections"].strip() for row in selected_rows):
        raise ValueError("Stage 13e selected pool contains an explicit covalent connection")
    if citation_policy["policy"] != "record_only_without_explicit_coordinate_bond":
        raise ValueError("Stage 13e citation-intent policy differs")

    outputs = {key: Path(str(value)) for key, value in dict(config["outputs"]).items()}
    run_directory = outputs["run_directory"]
    file_keys = {"receptor_manifest_csv", "redocking_case_manifest_csv", "common_box_json", "summary_json"}
    existing = [outputs[key] for key in file_keys if outputs[key].exists()]
    if existing and not overwrite:
        raise FileExistsError("Stage 13e outputs exist; pass --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()

    coordinate_amendment = read_json(inputs["coordinate_amendment"])
    anchor_numbers = [
        int(value)
        for value in coordinate_amendment["reference"]["required_anchor_residue_numbers"]
    ]
    ligand_download = dict(config["ligand_download"])
    receptor_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    common_coordinate_sets: list[np.ndarray] = []
    for selected in selected_rows:
        conformer_id = selected["conformer_id"]
        pdb_id = selected["pdb_id"]
        case_id = f"{pdb_id}_{selected['selected_ligand_resname']}"
        case_root = run_directory / "preparation" / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        raw_mmcif = verified(
            Path(selected["mmcif_path"]), selected["mmcif_sha256"]
        )
        aligned_pdb = verified(
            Path(selected["aligned_protein_pdb_path"]),
            selected["aligned_protein_pdb_sha256"],
        )
        alignment, raw_atoms, rotation, translation = alignment_audit(
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
        run_checked(command, f"receptor preparation for {conformer_id}")
        receptor_evidence = read_json(receptor_summary)
        if receptor_evidence.get("status") != "ok" or receptor_evidence.get("allow_bad_res") is not False:
            raise ValueError(f"receptor preparation did not pass: {conformer_id}")
        residue_change = dict(receptor_evidence["residue_count_change"])
        if residue_change["input_protein_only"] != residue_change["output_pdbqt"]:
            raise ValueError(f"receptor residue count changed: {conformer_id}")

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
        raw_sdf_coordinates, raw_sdf_elements, raw_molecule = coordinates_and_elements_from_sdf(raw_sdf)
        ligand_key = (
            selected["selected_ligand_resname"],
            int(selected["selected_ligand_resseq"]),
            selected["selected_ligand_icode"],
        )
        raw_ligands = ligand_residue_map(raw_atoms, {ligand_key[0]})
        if ligand_key not in raw_ligands:
            raise ValueError(f"selected mmCIF ligand is missing: {case_id}")
        raw_ligand_atoms = raw_ligands[ligand_key]
        mmcif_coordinates = np.vstack([atom.coord for atom in raw_ligand_atoms])
        mmcif_elements = [atom.element for atom in raw_ligand_atoms]
        if raw_molecule.GetNumHeavyAtoms() != int(selected["selected_ligand_heavy_atom_count"]):
            raise ValueError(f"selected ligand heavy-atom count differs: {case_id}")
        raw_rmsd, raw_maximum = point_set_rmsd_by_element(
            raw_sdf_coordinates,
            raw_sdf_elements,
            mmcif_coordinates,
            mmcif_elements,
        )
        coordinate_limit = float(protocol["maximum_element_matched_coordinate_error_angstrom"])
        if raw_rmsd > coordinate_limit or raw_maximum > coordinate_limit:
            raise ValueError(f"ModelServer ligand coordinates differ: {case_id}")

        common_sdf = case_root / f"{case_id}_common_frame.sdf"
        common_summary = case_root / "common_frame_transform.json"
        explicit_h_sdf = case_root / f"{case_id}_common_frame_explicitH.sdf"
        explicit_h_summary = case_root / "explicit_h_transform.json"
        for output_sdf, output_summary, add_hydrogens in (
            (common_sdf, common_summary, False),
            (explicit_h_sdf, explicit_h_summary, True),
        ):
            transform_command = [
                sys.executable,
                str(inputs["transform_sdf_script"]),
                "--input-sdf",
                str(raw_sdf),
                "--alignment-summary",
                str(alignment_path),
                "--output-sdf",
                str(output_sdf),
                "--summary-output",
                str(output_summary),
            ]
            if add_hydrogens:
                transform_command.append("--add-explicit-hydrogens")
            if overwrite:
                transform_command.append("--overwrite")
            run_checked(transform_command, f"ligand common-frame transform for {case_id}")
        common_coordinates, common_elements, common_molecule = coordinates_and_elements_from_sdf(common_sdf)
        transformed_mmcif = mmcif_coordinates @ rotation + translation
        common_rmsd, common_maximum = point_set_rmsd_by_element(
            common_coordinates,
            common_elements,
            transformed_mmcif,
            mmcif_elements,
        )
        if common_rmsd > coordinate_limit or common_maximum > coordinate_limit:
            raise ValueError(f"common-frame ligand coordinates differ: {case_id}")
        common_coordinate_sets.append(common_coordinates)

        ligand_pdbqt = case_root / f"{case_id}_ligand.pdbqt"
        ligand_audit, ligand_variant = prepare_ligand(explicit_h_sdf, ligand_pdbqt)
        receptor_rows.append(
            {
                "conformer_id": conformer_id,
                "pdb_id": pdb_id,
                "chain": "A",
                "receptor_pdbqt": receptor_pdbqt.as_posix(),
                "receptor_pdbqt_sha256": file_sha256(receptor_pdbqt),
                "receptor_preparation_summary": receptor_summary.as_posix(),
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
                "raw_sdf": raw_sdf.as_posix(),
                "raw_sdf_sha256": file_sha256(raw_sdf),
                "reference_sdf": common_sdf.as_posix(),
                "reference_sdf_sha256": file_sha256(common_sdf),
                "explicit_h_sdf": explicit_h_sdf.as_posix(),
                "explicit_h_sdf_sha256": file_sha256(explicit_h_sdf),
                "ligand_pdbqt": ligand_pdbqt.as_posix(),
                "ligand_pdbqt_sha256": file_sha256(ligand_pdbqt),
                "ligand_preparation_variant": ligand_variant,
                "ligand_pdbqt_atom_count": ligand_audit["pdbqt_atom_count"],
                "raw_sdf_to_mmcif_element_matched_rmsd_angstrom": raw_rmsd,
                "raw_sdf_to_mmcif_maximum_atom_distance_angstrom": raw_maximum,
                "common_sdf_to_transformed_mmcif_element_matched_rmsd_angstrom": common_rmsd,
                "common_sdf_to_transformed_mmcif_maximum_atom_distance_angstrom": common_maximum,
                "alignment_summary": alignment_path.as_posix(),
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
        raise ValueError("derived common box margin is too small")
    if max(box["size"].values()) > float(box_rule["maximum_axis_size_angstrom"]):
        raise ValueError("derived common box is too large")
    box_record = {
        "schema_version": "1.0",
        "status": "stage13e_common_box_ok",
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
        "status": "stage13e_egfr_redocking_inputs_ok",
        "config": {"path": config_path.as_posix(), "sha256": file_sha256(config_path)},
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
        "citation_intent_diagnostic": {
            "policy": citation_policy["policy"],
            "matched_receptor_count": len(citation_diagnostics),
            "explicit_covalent_connection_count": 0,
            "records": citation_diagnostics,
            "interpretation": (
                "Citation wording is diagnostic only. Every admitted coordinate case "
                "has no explicit protein-ligand covalent connection and must still pass "
                "the unchanged noncovalent three-seed redocking gate."
            ),
        },
        "data_boundary": {
            "ligand_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "MAPK14_stage11_rows_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            key: {"path": outputs[key].as_posix(), "sha256": file_sha256(outputs[key])}
            for key in ("receptor_manifest_csv", "redocking_case_manifest_csv", "common_box_json")
        },
        "next_gate": "build the GPU execution bundle and run 16 receptors x 3 seeds of Uni-Dock cognate redocking",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
