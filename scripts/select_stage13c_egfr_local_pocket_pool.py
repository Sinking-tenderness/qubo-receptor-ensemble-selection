"""Re-audit EGFR candidates with local ATP-pocket alignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np

try:
    from .select_stage13_egfr_coordinate_pool import (
        AtomRecord,
        explicit_covalent_connections,
        file_sha256,
        incomplete_standard_residues,
        is_heavy,
        kabsch,
        ligand_residue_map,
        load_config,
        maxmin_select,
        minimum_distance,
        pairwise_feature_vector,
        protein_residue_map,
        read_csv,
        read_json,
        residue_point,
        rmsd,
        select_chain_atoms,
        transform_atoms,
        verified,
        write_aligned_protein_pdb,
        write_csv,
        write_json,
        download_one,
    )
    from .select_stage13b_egfr_expanded_coordinate_pool import (
        select_expanded_metadata_candidates,
    )
except ImportError:
    from select_stage13_egfr_coordinate_pool import (
        AtomRecord,
        explicit_covalent_connections,
        file_sha256,
        incomplete_standard_residues,
        is_heavy,
        kabsch,
        ligand_residue_map,
        load_config,
        maxmin_select,
        minimum_distance,
        pairwise_feature_vector,
        protein_residue_map,
        read_csv,
        read_json,
        residue_point,
        rmsd,
        select_chain_atoms,
        transform_atoms,
        verified,
        write_aligned_protein_pdb,
        write_csv,
        write_json,
        download_one,
    )
    from select_stage13b_egfr_expanded_coordinate_pool import (
        select_expanded_metadata_candidates,
    )


def clean_cif_text(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    if text.startswith(";") and text.endswith(";"):
        text = text[1:-1]
    return " ".join(text.split())


def primary_citation(path: Path) -> tuple[str, str]:
    block = gemmi.cif.read_file(str(path)).sole_block()
    identifiers = list(block.find_values("_citation.id"))
    titles = list(block.find_values("_citation.title"))
    dois = list(block.find_values("_citation.pdbx_database_id_DOI"))
    for index, identifier in enumerate(identifiers):
        if str(identifier).lower() == "primary":
            title = clean_cif_text(str(titles[index])) if index < len(titles) else ""
            doi = clean_cif_text(str(dois[index])) if index < len(dois) else ""
            return title, doi
    return "", ""


def citation_is_excluded(
    title: str, doi: str, pattern: str, excluded_doi: str
) -> bool:
    return bool(re.search(pattern, title, flags=re.IGNORECASE)) or (
        doi.lower() == excluded_doi.lower()
    )


def ca_atom_map(atoms: list[AtomRecord]) -> dict[tuple[int, str], AtomRecord]:
    output: dict[tuple[int, str], AtomRecord] = {}
    for key, values in protein_residue_map(atoms).items():
        ca = [atom for atom in values if atom.atom_name == "CA"]
        if ca:
            output[key] = ca[0]
    return output


def audit_structure(
    pdb_id: str,
    chain_name: str,
    qualifying_ids: set[str],
    path: Path,
    aligned_path: Path,
    reference_atoms: list[AtomRecord],
    reference_ligand_coords: np.ndarray,
    pocket_numbers: list[int],
    anchor_numbers: list[int],
    original_gate: dict[str, object],
    correction: dict[str, object],
) -> tuple[dict[str, object], np.ndarray | None, list[str] | None]:
    structure = gemmi.read_structure(str(path))
    atoms = select_chain_atoms(structure, chain_name)
    reference_ca = ca_atom_map(reference_atoms)
    mobile_ca = ca_atom_map(atoms)
    shared = sorted(set(reference_ca) & set(mobile_ca))
    matched = [
        key
        for key in shared
        if reference_ca[key].resname == mobile_ca[key].resname
    ]
    mismatched = [
        key
        for key in shared
        if reference_ca[key].resname != mobile_ca[key].resname
    ]
    if len(matched) < 3:
        raise ValueError("fewer than three same-number C-alpha matches")

    anchor_keys = [(number, "") for number in anchor_numbers]
    missing_anchor_ca = [
        number for number in anchor_numbers if (number, "") not in mobile_ca
    ]
    mismatched_anchor_names = [
        number
        for number in anchor_numbers
        if (number, "") in mobile_ca
        and mobile_ca[(number, "")].resname
        != reference_ca[(number, "")].resname
    ]
    usable_anchor_keys = [
        key
        for key in anchor_keys
        if key in mobile_ca
        and mobile_ca[key].resname == reference_ca[key].resname
    ]
    alignment_keys = usable_anchor_keys if len(usable_anchor_keys) >= 3 else matched
    reference_alignment = np.vstack(
        [reference_ca[key].coord for key in alignment_keys]
    )
    mobile_alignment = np.vstack([mobile_ca[key].coord for key in alignment_keys])
    rotation, translation = kabsch(mobile_alignment, reference_alignment)
    transformed = transform_atoms(atoms, rotation, translation)
    determinant = float(np.linalg.det(rotation))
    aligned_anchor_rmsd = rmsd(
        mobile_alignment @ rotation + translation, reference_alignment
    )
    reference_global = np.vstack([reference_ca[key].coord for key in matched])
    mobile_global = np.vstack([mobile_ca[key].coord for key in matched])
    aligned_global_rmsd = rmsd(
        mobile_global @ rotation + translation, reference_global
    )

    reference_residues = protein_residue_map(reference_atoms)
    candidate_residues = protein_residue_map(transformed)
    present: list[int] = []
    missing_pocket: list[int] = []
    mismatched_pocket: list[int] = []
    incomplete_anchors: list[int] = []
    expected_pocket_atoms: set[tuple[int, str]] = set()
    observed_pocket_atoms: set[tuple[int, str]] = set()
    ca_points: list[np.ndarray] = []
    sidechain_points: list[np.ndarray] = []
    anchor_set = set(anchor_numbers)
    for number in pocket_numbers:
        reference_values = reference_residues[(number, "")]
        reference_name = reference_values[0].resname
        expected_names = {atom.atom_name for atom in reference_values}
        expected_pocket_atoms.update((number, name) for name in expected_names)
        candidate_values = candidate_residues.get((number, ""), [])
        candidate_name = candidate_values[0].resname if candidate_values else ""
        point = residue_point(candidate_values) if candidate_values else None
        if point is not None and candidate_name == reference_name:
            present.append(number)
            observed_names = {atom.atom_name for atom in candidate_values}
            observed_pocket_atoms.update((number, name) for name in observed_names)
            if number in anchor_set and not expected_names.issubset(observed_names):
                incomplete_anchors.append(number)
            ca_point, sidechain_point = point
        else:
            if candidate_values and candidate_name != reference_name:
                mismatched_pocket.append(number)
            else:
                missing_pocket.append(number)
            reference_point = residue_point(reference_values)
            assert reference_point is not None
            ca_point, sidechain_point = reference_point
        ca_points.append(ca_point)
        sidechain_points.append(sidechain_point)
    pocket_fraction = len(present) / len(pocket_numbers)
    pocket_heavy_fraction = len(
        expected_pocket_atoms & observed_pocket_atoms
    ) / len(expected_pocket_atoms)

    ligand_records: list[
        tuple[float, tuple[str, int, str], list[AtomRecord]]
    ] = []
    for key, values in ligand_residue_map(transformed, qualifying_ids).items():
        coordinates = np.vstack([atom.coord for atom in values])
        ligand_records.append(
            (minimum_distance(coordinates, reference_ligand_coords), key, values)
        )
    ligand_records.sort(key=lambda value: (value[0], value[1]))
    closest = ligand_records[0] if ligand_records else None
    selected_ligand = closest[1] if closest else ("", 0, "")
    ligand_distance = closest[0] if closest else math.nan
    ligand_coordinates = (
        np.vstack([atom.coord for atom in closest[2]]) if closest else None
    )
    ligand_centroid_distance = (
        float(
            np.linalg.norm(
                ligand_coordinates.mean(axis=0)
                - reference_ligand_coords.mean(axis=0)
            )
        )
        if ligand_coordinates is not None
        else math.nan
    )
    site_gate = dict(correction["strengthened_atp_site_gate"])
    hinge_number = int(site_gate["hinge_residue_number"])
    hinge_names = {str(value) for value in site_gate["hinge_backbone_atom_names"]}
    hinge_coordinates = [
        atom.coord
        for atom in candidate_residues.get((hinge_number, ""), [])
        if atom.atom_name in hinge_names
    ]
    ligand_hinge_distance = (
        minimum_distance(ligand_coordinates, np.vstack(hinge_coordinates))
        if ligand_coordinates is not None and hinge_coordinates
        else math.nan
    )
    protein_heavy_coordinates = [
        atom.coord
        for atom in transformed
        if atom.kind == "protein" and is_heavy(atom)
    ]
    protein_ligand_distance = (
        minimum_distance(
            np.vstack(protein_heavy_coordinates), ligand_coordinates
        )
        if protein_heavy_coordinates and ligand_coordinates is not None
        else math.nan
    )
    explicit_connections = (
        explicit_covalent_connections(
            structure,
            chain_name,
            selected_ligand,
            {key[0] for key in candidate_residues},
        )
        if closest
        else []
    )
    citation_title, citation_doi = primary_citation(path)
    covalent_gate = dict(correction["strengthened_covalent_intent_gate"])
    citation_excluded = citation_is_excluded(
        citation_title,
        citation_doi,
        str(covalent_gate["excluded_primary_citation_title_pattern"]),
        str(covalent_gate["excluded_primary_citation_doi"]),
    )
    incomplete = incomplete_standard_residues(transformed)
    ca_count = len(mobile_ca)
    heavy_count = len(protein_heavy_coordinates)

    replacement = dict(correction["replacement_alignment_gate"])
    reasons: list[str] = []
    if len(matched) < int(replacement["minimum_same-number_same-residue_global_ca_count"]):
        reasons.append("too_few_matched_ca")
    if missing_anchor_ca:
        reasons.append("missing_required_anchor_residue")
    if mismatched_anchor_names:
        reasons.append("anchor_residue_name_mismatch")
    if incomplete_anchors:
        reasons.append("incomplete_anchor_heavy_atom_template")
    if aligned_anchor_rmsd > float(
        replacement["maximum_aligned_anchor_ca_rmsd_angstrom"]
    ):
        reasons.append("aligned_anchor_ca_rmsd_above_limit")
    if determinant < 0.999999:
        reasons.append("rotation_is_not_proper")
    if pocket_fraction < float(original_gate["minimum_pocket_residue_fraction"]):
        reasons.append("pocket_residue_fraction_below_limit")
    if mismatched_pocket:
        reasons.append("pocket_residue_name_mismatch")
    if pocket_heavy_fraction < float(
        original_gate["minimum_pocket_heavy_atom_completeness_fraction"]
    ):
        reasons.append("pocket_heavy_atom_completeness_below_limit")
    if not math.isfinite(ligand_distance) or ligand_distance > float(
        original_gate["require_qualifying_ligand_within_reference_ligand_angstrom"]
    ):
        reasons.append("no_qualifying_ligand_in_reference_atp_site")
    if not math.isfinite(ligand_centroid_distance) or ligand_centroid_distance > float(
        site_gate[
            "maximum_selected_ligand_centroid_distance_to_reference_hyz_centroid_angstrom"
        ]
    ):
        reasons.append("selected_ligand_centroid_outside_atp_site")
    if not math.isfinite(ligand_hinge_distance) or ligand_hinge_distance > float(
        site_gate[
            "maximum_selected_ligand_heavy_atom_distance_to_hinge_backbone_angstrom"
        ]
    ):
        reasons.append("selected_ligand_lacks_hinge_contact")
    if explicit_connections:
        reasons.append("explicit_target_ligand_covalent_connection")
    geometric_covalent_limit = float(
        dict(original_gate["hidden_covalency"])[
            "exclude_if_minimum_protein_ligand_heavy_atom_distance_at_or_below_angstrom"
        ]
    )
    if math.isfinite(protein_ligand_distance) and protein_ligand_distance <= geometric_covalent_limit:
        reasons.append("protein_ligand_distance_indicates_covalency")
    if citation_excluded:
        reasons.append("primary_citation_indicates_targeted_covalent_ligand")

    if not reasons:
        write_aligned_protein_pdb(aligned_path, transformed)
        vector, feature_names = pairwise_feature_vector(
            ca_points, sidechain_points
        )
    else:
        vector, feature_names = None, None
    conformer_id = (
        "EGFR_2RGP_reference" if pdb_id == "2RGP" else f"EGFR_{pdb_id}_aligned"
    )
    row = {
        "conformer_id": conformer_id,
        "pdb_id": pdb_id,
        "chain": chain_name,
        "status": "coordinate_eligible" if not reasons else "coordinate_excluded",
        "exclusion_reasons": ";".join(reasons),
        "mmcif_path": path.as_posix(),
        "mmcif_sha256": file_sha256(path),
        "aligned_protein_pdb_path": aligned_path.as_posix() if not reasons else "",
        "aligned_protein_pdb_sha256": file_sha256(aligned_path) if not reasons else "",
        "matched_global_ca_count": len(matched),
        "global_residue_name_mismatch_count": len(mismatched),
        "aligned_anchor_ca_rmsd_angstrom": aligned_anchor_rmsd,
        "aligned_global_ca_rmsd_diagnostic_angstrom": aligned_global_rmsd,
        "rotation_determinant": determinant,
        "missing_anchor_residues": ";".join(map(str, missing_anchor_ca)),
        "mismatched_anchor_residues": ";".join(map(str, mismatched_anchor_names)),
        "incomplete_anchor_residues": ";".join(map(str, incomplete_anchors)),
        "pocket_present_count": len(present),
        "pocket_residue_fraction": pocket_fraction,
        "pocket_heavy_atom_completeness_fraction": pocket_heavy_fraction,
        "missing_pocket_residues": ";".join(map(str, missing_pocket)),
        "mismatched_pocket_residues": ";".join(map(str, mismatched_pocket)),
        "selected_ligand_resname": selected_ligand[0],
        "selected_ligand_resseq": selected_ligand[1],
        "selected_ligand_icode": selected_ligand[2],
        "selected_ligand_heavy_atom_count": len(closest[2]) if closest else 0,
        "selected_ligand_min_distance_to_reference_hyz_angstrom": ligand_distance,
        "selected_ligand_centroid_distance_to_reference_hyz_angstrom": ligand_centroid_distance,
        "selected_ligand_min_distance_to_met793_backbone_angstrom": ligand_hinge_distance,
        "minimum_protein_ligand_heavy_atom_distance_angstrom": protein_ligand_distance,
        "explicit_covalent_connections": ";".join(explicit_connections),
        "primary_citation_title": citation_title,
        "primary_citation_doi": citation_doi,
        "primary_citation_covalent_exclusion": citation_excluded,
        "protein_ca_count": ca_count,
        "protein_heavy_atom_count": heavy_count,
        "protein_heavy_atoms_per_ca": heavy_count / ca_count if ca_count else 0.0,
        "global_incomplete_standard_amino_acid_residue_count": len(incomplete),
        "global_incomplete_standard_amino_acid_residues": ";".join(incomplete),
        "imputed_pocket_feature_count": len(pocket_numbers) - len(present),
    }
    return row, vector, feature_names


def write_report(path: Path, summary: dict[str, object]) -> None:
    counts = dict(summary["counts"])
    lines = [
        "# Stage 13c EGFR Local-Pocket Structural Pool",
        "",
        "## Result",
        "",
        f"- Audited structures: {counts['audited_count']}",
        f"- Corrected coordinate-eligible structures: {counts['coordinate_eligible_count']}",
        f"- Targeted-covalent citation exclusions: {counts['citation_covalent_exclusion_count']}",
        f"- Non-ATP-site ligand exclusions: {counts['non_atp_site_exclusion_count']}",
        f"- Selected receptors: {counts['selected_receptor_count']}",
        f"- Status: {summary['status']}",
        "",
        "This is a documented post-coordinate structural-method correction.",
        "No activity label, docking score, fresh-validation row, or test row was read.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    implementation = dict(config["implementation"])
    script_path = verified(
        Path(str(implementation["path"])), str(implementation["sha256"])
    )
    if script_path.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 13c implementation path differs")
    for prefix in ("base_dependency", "expansion_dependency", "math_dependency"):
        verified(
            Path(str(implementation[f"{prefix}_path"])),
            str(implementation[f"{prefix}_sha256"]),
        )
    prereg_record = dict(config["preregistration"])
    coordinate_record = dict(config["preregistration_amendment"])
    expansion_record = dict(config["pool_expansion_amendment"])
    correction_record = dict(config["method_correction_amendment"])
    prereg_path = verified(
        Path(str(prereg_record["path"])), str(prereg_record["sha256"])
    )
    coordinate_path = verified(
        Path(str(coordinate_record["path"])), str(coordinate_record["sha256"])
    )
    expansion_path = verified(
        Path(str(expansion_record["path"])), str(expansion_record["sha256"])
    )
    correction_path = verified(
        Path(str(correction_record["path"])), str(correction_record["sha256"])
    )
    coordinate = read_json(coordinate_path)
    expansion = read_json(expansion_path)
    correction = read_json(correction_path)
    if correction["coordinate_gate_amendment"]["sha256"] != file_sha256(coordinate_path):
        raise ValueError("Stage 13c coordinate amendment differs")
    if correction["metadata_pool_expansion_amendment"]["sha256"] != file_sha256(expansion_path):
        raise ValueError("Stage 13c pool expansion amendment differs")
    boundary = dict(correction["data_boundary"])
    for key in (
        "ligand_labels_read",
        "docking_scores_read",
        "MAPK14_stage11_rows_read",
        "fresh_validation_rows_read",
        "test_rows_read",
    ):
        if int(boundary[key]) != 0:
            raise ValueError("Stage 13c data boundary differs")

    expected_runtime = {
        key: str(value) for key, value in dict(config["runtime"]).items()
    }
    runtime = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "gemmi_version": gemmi.__version__,
    }
    if runtime != expected_runtime:
        raise RuntimeError(f"Stage 13c runtime differs: {runtime} != {expected_runtime}")

    inputs: dict[str, Path] = {}
    for key, value in dict(config["inputs"]).items():
        record = dict(value)
        inputs[key] = verified(Path(str(record["path"])), str(record["sha256"]))
    trigger_summary = read_json(inputs["stage13b_summary"])
    if trigger_summary.get("status") != correction["trigger"]["required_status"]:
        raise ValueError("Stage 13c trigger status differs")
    if any(int(value) != 0 for value in trigger_summary["data_boundary"].values()):
        raise ValueError("Stage 13c trigger data boundary differs")

    outputs = {key: Path(str(value)) for key, value in dict(config["outputs"]).items()}
    directory_keys = {"raw_mmcif_directory", "aligned_protein_pdb_directory"}
    existing = [
        path
        for key, path in outputs.items()
        if key not in directory_keys and path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError("Stage 13c outputs exist; pass --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()

    all_metadata = read_csv(inputs["candidate_metadata_csv"])
    candidates, _ = select_expanded_metadata_candidates(all_metadata, expansion)
    expected_count = int(correction["selection"]["expanded_metadata_candidate_count"])
    if len(candidates) != expected_count:
        raise ValueError("Stage 13c candidate count differs")
    raw_directory = outputs["raw_mmcif_directory"]
    raw_paths = {
        row["pdb_id"]: raw_directory / f"{row['pdb_id']}.cif" for row in candidates
    }
    download = dict(config["download"])
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
        download_errors: dict[str, str] = {}
        for future in as_completed(futures):
            pdb_id, error = future.result()
            if error:
                download_errors[pdb_id] = error

    reference = dict(coordinate["reference"])
    reference_structure = gemmi.read_structure(str(inputs["reference_mmcif"]))
    reference_atoms = select_chain_atoms(
        reference_structure, str(reference["auth_chain"])
    )
    reference_ligands = ligand_residue_map(
        reference_atoms, {str(reference["ligand_comp_id"])}
    )
    reference_ligand_coords = np.vstack(
        [atom.coord for atom in next(iter(reference_ligands.values()))]
    )
    pocket_numbers = [
        int(value) for value in reference["reference_pocket_residue_numbers"]
    ]
    anchor_numbers = [
        int(value) for value in reference["required_anchor_residue_numbers"]
    ]
    original_gate = dict(coordinate["coordinate_gate"])

    audit_rows: list[dict[str, object]] = []
    vectors: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    aligned_directory = outputs["aligned_protein_pdb_directory"]
    for metadata in candidates:
        pdb_id = metadata["pdb_id"]
        conformer_id = (
            "EGFR_2RGP_reference" if pdb_id == "2RGP" else f"EGFR_{pdb_id}_aligned"
        )
        if pdb_id in download_errors:
            row, vector, names = (
                {
                    "conformer_id": conformer_id,
                    "pdb_id": pdb_id,
                    "chain": metadata["selected_auth_chain"],
                    "status": "coordinate_excluded",
                    "exclusion_reasons": "coordinate_file_unavailable",
                    "download_error": download_errors[pdb_id],
                },
                None,
                None,
            )
        else:
            try:
                row, vector, names = audit_structure(
                    pdb_id,
                    metadata["selected_auth_chain"],
                    {
                        value
                        for value in metadata["qualifying_ligand_ids"].split(";")
                        if value
                    },
                    raw_paths[pdb_id],
                    aligned_directory / f"{conformer_id}_to_2RGP_A.pdb",
                    reference_atoms,
                    reference_ligand_coords,
                    pocket_numbers,
                    anchor_numbers,
                    original_gate,
                    correction,
                )
            except Exception as error:
                row, vector, names = (
                    {
                        "conformer_id": conformer_id,
                        "pdb_id": pdb_id,
                        "chain": metadata["selected_auth_chain"],
                        "status": "coordinate_excluded",
                        "exclusion_reasons": "coordinate_parse_or_audit_error",
                        "audit_error": f"{type(error).__name__}: {error}",
                        "mmcif_path": raw_paths[pdb_id].as_posix(),
                        "mmcif_sha256": file_sha256(raw_paths[pdb_id]),
                    },
                    None,
                    None,
                )
        audit_rows.append(row)
        if vector is not None:
            vectors[conformer_id] = vector
            feature_names = names
    audit_rows.sort(key=lambda row: str(row["conformer_id"]))
    eligible_rows = [row for row in audit_rows if row["status"] == "coordinate_eligible"]
    eligible_by_id = {str(row["conformer_id"]): row for row in eligible_rows}
    reference_id = "EGFR_2RGP_reference"
    if reference_id not in eligible_by_id:
        raise ValueError("Stage 13c reference failed")
    assert feature_names is not None
    ordered_ids = sorted(vectors)
    matrix = np.vstack([vectors[conformer_id] for conformer_id in ordered_ids])
    standard_deviations = matrix.std(axis=0)
    keep = standard_deviations >= float(
        coordinate["structural_selection"]["minimum_variable_feature_sd_angstrom"]
    )
    variable_feature_count = int(keep.sum())
    if variable_feature_count < 3:
        raise ValueError("too few variable Stage 13c structural features")
    means = matrix.mean(axis=0)
    standardized = (matrix[:, keep] - means[keep]) / standard_deviations[keep]
    standardized /= math.sqrt(variable_feature_count)
    distance_by_pair: dict[tuple[str, str], float] = {}
    distance_rows: list[dict[str, object]] = []
    for first_index, second_index in combinations(range(len(ordered_ids)), 2):
        first = ordered_ids[first_index]
        second = ordered_ids[second_index]
        distance = float(
            np.linalg.norm(standardized[first_index] - standardized[second_index])
        )
        distance_by_pair[(first, second)] = distance
        distance_rows.append(
            {
                "conformer_id_a": first,
                "conformer_id_b": second,
                "standardized_pocket_distance": distance,
            }
        )
    target_count = int(correction["selection"]["target_receptor_count"])
    selected_rows: list[dict[str, object]] = []
    if len(ordered_ids) >= target_count:
        additions = maxmin_select(
            ordered_ids, [reference_id], distance_by_pair, target_count - 1
        )
        selected_rows.append(
            {
                "pool_role": "reference_seed",
                "selection_rank": 1,
                "minimum_standardized_distance_to_selected_pool": "",
                **eligible_by_id[reference_id],
            }
        )
        for addition in additions:
            conformer_id = str(addition["conformer_id"])
            selected_rows.append(
                {
                    "pool_role": "maxmin_addition",
                    "selection_rank": int(addition["selection_rank"]) + 1,
                    "minimum_standardized_distance_to_selected_pool": addition[
                        "minimum_standardized_distance_to_selected_pool"
                    ],
                    **eligible_by_id[conformer_id],
                }
            )
    feature_rows = [
        {
            "conformer_id": conformer_id,
            **{
                name: float(value)
                for name, value in zip(feature_names, vectors[conformer_id])
            },
        }
        for conformer_id in ordered_ids
    ]
    eligible_manifest = [
        {
            "conformer_id": row["conformer_id"],
            "pdb_id": row["pdb_id"],
            "chain": row["chain"],
            "mmcif_path": row["mmcif_path"],
            "mmcif_sha256": row["mmcif_sha256"],
            "aligned_protein_pdb_path": row["aligned_protein_pdb_path"],
            "aligned_protein_pdb_sha256": row["aligned_protein_pdb_sha256"],
            "selected_ligand_resname": row["selected_ligand_resname"],
            "selected_ligand_resseq": row["selected_ligand_resseq"],
            "selected_ligand_icode": row["selected_ligand_icode"],
        }
        for row in eligible_rows
    ]
    write_csv(outputs["coordinate_audit_csv"], audit_rows)
    write_csv(outputs["eligible_pool_manifest_csv"], eligible_manifest)
    write_csv(outputs["feature_matrix_csv"], feature_rows)
    write_csv(outputs["pairwise_distances_csv"], distance_rows)
    if selected_rows:
        write_csv(outputs["selected16_manifest_csv"], selected_rows)

    reasons = Counter(
        reason
        for row in audit_rows
        for reason in str(row.get("exclusion_reasons", "")).split(";")
        if reason
    )
    citation_count = sum(
        "primary_citation_indicates_targeted_covalent_ligand"
        in str(row.get("exclusion_reasons", ""))
        for row in audit_rows
    )
    non_atp_count = sum(
        "selected_ligand_centroid_outside_atp_site" in str(row.get("exclusion_reasons", ""))
        or "selected_ligand_lacks_hinge_contact" in str(row.get("exclusion_reasons", ""))
        for row in audit_rows
    )
    status = (
        "stage13c_egfr_local_pocket_structural_selection_ok"
        if len(selected_rows) == target_count
        else "stage13c_egfr_local_pocket_pool_insufficient_stop"
    )
    output_records = {}
    for key, path in outputs.items():
        if key in directory_keys or key in {"summary_json", "report_md"} or not path.is_file():
            continue
        output_records[key] = {"path": path.as_posix(), "sha256": file_sha256(path)}
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {"path": config_path.as_posix(), "sha256": file_sha256(config_path)},
        "method_correction_amendment": {"path": correction_path.as_posix(), "sha256": file_sha256(correction_path)},
        "runtime": runtime,
        "counts": {
            "audited_count": len(audit_rows),
            "coordinate_eligible_count": len(eligible_rows),
            "coordinate_excluded_count": len(audit_rows) - len(eligible_rows),
            "citation_covalent_exclusion_count": citation_count,
            "non_atp_site_exclusion_count": non_atp_count,
            "selected_receptor_count": len(selected_rows),
            "target_receptor_count": target_count,
            "raw_feature_count": len(feature_names),
            "variable_feature_count": variable_feature_count,
            "pairwise_distance_count": len(distance_rows),
        },
        "coordinate_exclusion_reason_counts": dict(sorted(reasons.items())),
        "selected_receptor_ids": [str(row["conformer_id"]) for row in selected_rows],
        "data_boundary": {
            "ligand_labels_read": 0,
            "docking_scores_read": 0,
            "MAPK14_stage11_rows_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": output_records,
        "next_gate": (
            "freeze deterministic heavy-atom completion and receptor/native-ligand preparation, then require co-crystal redocking RMSD at or below 2.0 A"
            if status == "stage13c_egfr_local_pocket_structural_selection_ok"
            else "stop without docking and move the external-target pilot to a different protein"
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
