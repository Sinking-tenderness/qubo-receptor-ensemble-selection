"""Prepare and redock the four structurally selected MAPK14 additions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from .batch_prepare_ligand_pdbqt import find_meeko_script, parse_pdbqt
    from .prepare_receptor import file_sha256
    from .select_mk14_rcsb_coordinate_pool import (
        is_heavy,
        kabsch,
        match_ca,
        parse_pdb,
        rmsd,
        select_chain_atoms,
        transform_atoms,
    )
except ImportError:
    from batch_prepare_ligand_pdbqt import find_meeko_script, parse_pdbqt
    from prepare_receptor import file_sha256
    from select_mk14_rcsb_coordinate_pool import (
        is_heavy,
        kabsch,
        match_ca,
        parse_pdb,
        rmsd,
        select_chain_atoms,
        transform_atoms,
    )


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checked_record(record: dict[str, object]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def run_checked(command: list[str], operation: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        combined = "\n".join(
            value.strip()
            for value in (completed.stdout, completed.stderr)
            if value.strip()
        )
        raise RuntimeError(
            f"{operation} failed with return code {completed.returncode}: {combined[-2000:]}"
        )
    return completed


def build_modelserver_url(template: str, pdb_id: str, chain: str, resseq: int) -> str:
    query = urllib.parse.urlencode(
        {"auth_asym_id": chain, "auth_seq_id": resseq, "encoding": "sdf"}
    )
    return f"{template.format(pdb_id=pdb_id)}?{query}"


def download_cached(
    url: str,
    path: Path,
    timeout_seconds: float,
    maximum_retries: int,
    retry_backoff_seconds: float,
) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(maximum_retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "qubo-receptor-ensemble-selection/1.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
            if not payload or b"M  END" not in payload:
                raise ValueError("downloaded ligand response is not an SDF mol block")
            path.write_bytes(payload)
            return
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = error
            if attempt + 1 < maximum_retries:
                time.sleep(retry_backoff_seconds * (2**attempt))
    assert last_error is not None
    raise RuntimeError(f"ligand download failed: {last_error}")


def molblock_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    marker = "M  END"
    if marker not in text:
        raise ValueError(f"SDF mol block terminator is missing: {path}")
    normalized = text[: text.index(marker) + len(marker)].rstrip() + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest().upper()


def atom_identity(atom: object) -> tuple[object, ...]:
    return (
        atom.record,
        atom.resseq,
        atom.icode,
        atom.resname,
        atom.atom_name,
    )


def alignment_transform_audit(
    reference_pdb: Path,
    raw_pdb: Path,
    aligned_pdb: Path,
    chain: str,
) -> dict[str, object]:
    reference_atoms = select_chain_atoms(parse_pdb(reference_pdb), "A")
    mobile_atoms = select_chain_atoms(parse_pdb(raw_pdb), chain)
    reference_coords, mobile_coords, mismatches = match_ca(
        reference_atoms, mobile_atoms
    )
    rotation, translation = kabsch(mobile_coords, reference_coords)
    transformed = transform_atoms(mobile_atoms, rotation, translation)
    aligned_atoms = select_chain_atoms(parse_pdb(aligned_pdb), chain)
    if [atom_identity(atom) for atom in transformed] != [
        atom_identity(atom) for atom in aligned_atoms
    ]:
        raise ValueError("recomputed and selected aligned atom identities differ")
    coordinate_error = float(
        np.max(
            np.abs(
                np.vstack([atom.coord for atom in transformed])
                - np.vstack([atom.coord for atom in aligned_atoms])
            )
        )
    )
    if coordinate_error > 0.0011:
        raise ValueError(
            f"recomputed alignment differs from selected PDB by {coordinate_error} A"
        )
    aligned_ca = mobile_coords @ rotation + translation
    return {
        "method": "sequence-matched C-alpha Kabsch rigid-body alignment using the coordinate-selector implementation",
        "reference_path": reference_pdb.as_posix(),
        "reference_sha256": file_sha256(reference_pdb),
        "reference_chain": "A",
        "mobile_path": raw_pdb.as_posix(),
        "mobile_sha256": file_sha256(raw_pdb),
        "mobile_chain": chain,
        "selected_aligned_path": aligned_pdb.as_posix(),
        "selected_aligned_sha256": file_sha256(aligned_pdb),
        "matched_ca_count": len(reference_coords),
        "residue_name_mismatches_excluded": mismatches,
        "aligned_global_ca_rmsd_angstrom": rmsd(aligned_ca, reference_coords),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "rotation_matrix_row_vector_convention": rotation.tolist(),
        "translation_vector_angstrom": translation.tolist(),
        "maximum_coordinate_difference_from_selected_aligned_pdb_angstrom": coordinate_error,
    }


def coordinates_and_elements_from_sdf(path: Path) -> tuple[np.ndarray, list[str], object]:
    from rdkit import Chem

    molecules = list(Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True))
    if len(molecules) != 1 or molecules[0] is None:
        raise ValueError(f"SDF must contain one parseable molecule: {path}")
    molecule = molecules[0]
    if molecule.GetNumConformers() != 1:
        raise ValueError(f"SDF must contain one conformer: {path}")
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    elements = [atom.GetSymbol().upper() for atom in molecule.GetAtoms()]
    return coordinates, elements, molecule


def point_set_rmsd_by_element(
    first_coords: np.ndarray,
    first_elements: list[str],
    second_coords: np.ndarray,
    second_elements: list[str],
) -> tuple[float, float]:
    if sorted(first_elements) != sorted(second_elements):
        raise ValueError("element counts differ between SDF and aligned PDB ligand")
    squared_distances: list[float] = []
    assigned_distances: list[float] = []
    for element in sorted(set(first_elements)):
        first_indices = [index for index, value in enumerate(first_elements) if value == element]
        second_indices = [index for index, value in enumerate(second_elements) if value == element]
        cost = np.linalg.norm(
            first_coords[first_indices, None, :] - second_coords[None, second_indices, :],
            axis=2,
        )
        rows, columns = linear_sum_assignment(cost)
        values = cost[rows, columns]
        assigned_distances.extend(float(value) for value in values)
        squared_distances.extend(float(value * value) for value in values)
    return math.sqrt(sum(squared_distances) / len(squared_distances)), max(
        assigned_distances
    )


def aligned_ligand_coordinates(
    aligned_pdb: Path, chain: str, resname: str, resseq: int, icode: str
) -> tuple[np.ndarray, list[str]]:
    atoms = [
        atom
        for atom in select_chain_atoms(parse_pdb(aligned_pdb), chain)
        if atom.record == "HETATM"
        and atom.resname == resname
        and atom.resseq == resseq
        and atom.icode == icode
        and is_heavy(atom)
    ]
    if not atoms:
        raise ValueError("selected co-crystal ligand was not found in aligned PDB")
    elements = [
        (atom.element or atom.atom_name.lstrip("0123456789")[:1]).upper()
        for atom in atoms
    ]
    return np.vstack([atom.coord for atom in atoms]), elements


def box_audit(
    coordinates: np.ndarray,
    center: dict[str, float],
    size: dict[str, float],
) -> dict[str, object]:
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    axes = ["x", "y", "z"]
    margins: dict[str, float] = {}
    for index, axis in enumerate(axes):
        lower = float(center[axis]) - float(size[axis]) / 2.0
        upper = float(center[axis]) + float(size[axis]) / 2.0
        margins[f"{axis}_lower"] = float(minimum[index] - lower)
        margins[f"{axis}_upper"] = float(upper - maximum[index])
    return {
        "ligand_minimum_angstrom": minimum.tolist(),
        "ligand_maximum_angstrom": maximum.tolist(),
        "margins_angstrom": margins,
        "minimum_margin_angstrom": min(margins.values()),
    }


def write_vina_config(
    path: Path,
    receptor: Path,
    ligand: Path,
    center: dict[str, float],
    size: dict[str, float],
    parameters: dict[str, int | float],
) -> None:
    lines = [
        f"receptor = {receptor.as_posix()}",
        f"ligand = {ligand.as_posix()}",
        "",
        *(f"center_{axis} = {center[axis]}" for axis in ("x", "y", "z")),
        "",
        *(f"size_{axis} = {size[axis]}" for axis in ("x", "y", "z")),
        "",
        f"exhaustiveness = {parameters['exhaustiveness']}",
        f"cpu = {parameters['cpu']}",
        f"num_modes = {parameters['num_modes']}",
        f"energy_range = {parameters['energy_range']}",
        f"seed = {parameters['seed']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def check_runtime(config: dict[str, object]) -> dict[str, str]:
    actual = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": importlib.metadata.version("scipy"),
        "rdkit_version": importlib.metadata.version("rdkit"),
        "meeko_version": importlib.metadata.version("meeko"),
        "prody_version": importlib.metadata.version("prody"),
    }
    expected = {key: str(value) for key, value in config["runtime"].items()}
    if actual != expected:
        raise RuntimeError(f"runtime differs: {actual} != {expected}")
    return actual


def run_gate(config_path: Path, overwrite: bool) -> dict[str, object]:
    config = read_json(config_path)
    runtime = check_runtime(config)
    inputs = config["inputs"]
    assert isinstance(inputs, dict)
    input_paths = {
        key: checked_record(record)
        for key, record in inputs.items()
        if isinstance(record, dict)
    }
    required_inputs = {
        "selection_summary",
        "selection_audit",
        "selected_manifest",
        "reference_pdb",
        "prepare_receptor_script",
        "transform_sdf_script",
        "evaluate_rmsd_script",
        "vina_executable",
    }
    if set(input_paths) != required_inputs:
        raise ValueError("redocking gate inputs differ from the required set")
    selection_summary = read_json(input_paths["selection_summary"])
    selection_audit = read_json(input_paths["selection_audit"])
    if selection_summary.get("status") != "expanded8_structural_selection_ok":
        raise ValueError("coordinate selection did not pass")
    if selection_audit.get("status") != "independent_coordinate_selection_audit_ok":
        raise ValueError("independent coordinate-selection audit did not pass")
    protocol_amendment_path: Path | None = None
    protocol_amendment_record = config.get("protocol_amendment")
    if protocol_amendment_record is not None:
        if not isinstance(protocol_amendment_record, dict):
            raise ValueError("protocol amendment record is invalid")
        protocol_amendment_path = checked_record(protocol_amendment_record)
        protocol_amendment = read_json(protocol_amendment_path)
        boundary = protocol_amendment.get("data_boundary")
        if not isinstance(boundary, dict) or any(
            int(boundary.get(key, -1)) != 0
            for key in (
                "ligand_labels_read",
                "benchmark_docking_scores_read",
                "previous_validation_rows_read",
                "test_rows_read",
                "vina_runs_completed_in_expansion_gate",
            )
        ):
            raise ValueError("protocol amendment crossed a frozen data boundary")

    output = config["outputs"]
    assert isinstance(output, dict)
    run_root = Path(str(output["run_root"]))
    receptor_manifest_path = Path(str(output["receptor_manifest_csv"]))
    redocking_results_path = Path(str(output["redocking_results_csv"]))
    summary_path = Path(str(output["summary_json"]))
    protected_outputs = [receptor_manifest_path, redocking_results_path, summary_path]
    if any(path.exists() for path in protected_outputs) and not overwrite:
        raise FileExistsError("redocking gate outputs exist; review or use --overwrite")

    selected_rows = {
        row["conformer_id"]: row
        for row in read_csv(input_paths["selected_manifest"])
        if row["pool_role"] == "new_maxmin_addition"
    }
    cases = config["cases"]
    assert isinstance(cases, list)
    if set(selected_rows) != {str(case["conformer_id"]) for case in cases}:
        raise ValueError("configured redocking cases differ from v2 additions")

    download = config["ligand_download"]
    box = config["common_box"]
    vina_parameters = config["vina_parameters"]
    assert isinstance(download, dict)
    assert isinstance(box, dict)
    assert isinstance(vina_parameters, dict)
    center = {axis: float(box[f"center_{axis}"]) for axis in ("x", "y", "z")}
    size = {axis: float(box[f"size_{axis}"]) for axis in ("x", "y", "z")}
    minimum_box_margin = float(box["minimum_crystal_pose_margin_angstrom"])
    maximum_reference_rmsd = float(
        config["reference_coordinate_gate"]["maximum_element_matched_rmsd_angstrom"]
    )

    receptor_rows: list[dict[str, object]] = []
    prepared_cases: list[dict[str, object]] = []
    for case in cases:
        conformer_id = str(case["conformer_id"])
        selected = selected_rows[conformer_id]
        for key in ("pdb_id", "chain", "selected_ligand_resname", "selected_ligand_resseq"):
            expected = str(case[key])
            observed = str(selected[key])
            if expected != observed:
                raise ValueError(f"selected case field differs for {conformer_id}: {key}")
        raw_pdb = Path(selected["pdb_path"])
        aligned_pdb = Path(selected["aligned_pdb_path"])
        if file_sha256(raw_pdb) != selected["pdb_sha256"] or file_sha256(
            aligned_pdb
        ) != selected["aligned_pdb_sha256"]:
            raise ValueError(f"selected coordinate hash differs: {conformer_id}")

        case_id = f"{case['pdb_id']}_{case['selected_ligand_resname']}"
        case_root = run_root / case_id
        alignment_summary_path = case_root / "alignment_summary.json"
        alignment = alignment_transform_audit(
            input_paths["reference_pdb"], raw_pdb, aligned_pdb, str(case["chain"])
        )
        write_json(alignment_summary_path, alignment)

        raw_sdf = case_root / f"{case_id}_raw.sdf"
        source_url = build_modelserver_url(
            str(download["url_template"]),
            str(case["pdb_id"]),
            str(case["chain"]),
            int(case["selected_ligand_resseq"]),
        )
        download_cached(
            source_url,
            raw_sdf,
            float(download["timeout_seconds"]),
            int(download["maximum_retries"]),
            float(download["retry_backoff_seconds"]),
        )

        common_sdf = case_root / f"{case_id}_common_frame.sdf"
        common_transform_summary = case_root / "common_frame_transform.json"
        run_checked(
            [
                sys.executable,
                str(input_paths["transform_sdf_script"]),
                "--input-sdf",
                str(raw_sdf),
                "--alignment-summary",
                str(alignment_summary_path),
                "--output-sdf",
                str(common_sdf),
                "--summary-output",
                str(common_transform_summary),
                "--overwrite",
            ],
            f"common-frame ligand transform for {case_id}",
        )
        explicit_h_sdf = case_root / f"{case_id}_common_frame_explicitH.sdf"
        explicit_h_summary = case_root / "explicit_h_transform.json"
        run_checked(
            [
                sys.executable,
                str(input_paths["transform_sdf_script"]),
                "--input-sdf",
                str(raw_sdf),
                "--alignment-summary",
                str(alignment_summary_path),
                "--output-sdf",
                str(explicit_h_sdf),
                "--summary-output",
                str(explicit_h_summary),
                "--add-explicit-hydrogens",
                "--overwrite",
            ],
            f"explicit-H ligand transform for {case_id}",
        )

        sdf_coords, sdf_elements, molecule = coordinates_and_elements_from_sdf(
            common_sdf
        )
        if molecule.GetNumHeavyAtoms() != int(case["selected_ligand_heavy_atom_count"]):
            raise ValueError(f"ligand heavy-atom count differs for {case_id}")
        pdb_coords, pdb_elements = aligned_ligand_coordinates(
            aligned_pdb,
            str(case["chain"]),
            str(case["selected_ligand_resname"]),
            int(case["selected_ligand_resseq"]),
            str(case.get("selected_ligand_icode", "")),
        )
        reference_rmsd, reference_max_distance = point_set_rmsd_by_element(
            sdf_coords, sdf_elements, pdb_coords, pdb_elements
        )
        if reference_rmsd > maximum_reference_rmsd:
            raise ValueError(
                f"transformed SDF differs from aligned crystal ligand for {case_id}: {reference_rmsd} A"
            )
        current_box_audit = box_audit(sdf_coords, center, size)

        receptor_root = case_root / "receptor"
        receptor_pdbqt = receptor_root / f"{conformer_id}_receptor.pdbqt"
        receptor_summary_path = receptor_root / "preparation_summary.json"
        run_checked(
            [
                sys.executable,
                str(input_paths["prepare_receptor_script"]),
                "--input-pdb",
                str(aligned_pdb),
                "--chain",
                str(case["chain"]),
                "--protein-only-output",
                str(receptor_root / f"{conformer_id}_protein_only.pdb"),
                "--prepared-pdb-output",
                str(receptor_root / f"{conformer_id}_prepared.pdb"),
                "--pdbqt-output",
                str(receptor_pdbqt),
                "--summary-output",
                str(receptor_summary_path),
                "--charge-model",
                "gasteiger",
                "--overwrite",
            ],
            f"receptor preparation for {case_id}",
        )
        receptor_summary = read_json(receptor_summary_path)
        if receptor_summary.get("status") != "ok":
            raise ValueError(f"receptor preparation did not pass for {case_id}")

        ligand_pdbqt = case_root / f"{case_id}.pdbqt"
        meeko_script = find_meeko_script()
        run_checked(
            [
                sys.executable,
                str(meeko_script),
                "-i",
                str(explicit_h_sdf),
                "-o",
                str(ligand_pdbqt),
            ],
            f"ligand preparation for {case_id}",
        )
        ligand_pdbqt_audit = parse_pdbqt(ligand_pdbqt)

        receptor_audit = receptor_summary["outputs"]["receptor_pdbqt"]["audit"]
        receptor_rows.append(
            {
                "conformer_id": conformer_id,
                "pdb_id": case["pdb_id"],
                "chain": case["chain"],
                "aligned_pdb": aligned_pdb.as_posix(),
                "aligned_pdb_sha256": file_sha256(aligned_pdb),
                "receptor_pdbqt": receptor_pdbqt.as_posix(),
                "receptor_pdbqt_sha256": file_sha256(receptor_pdbqt),
                "residue_count": receptor_audit["residue_count"],
                "pdbqt_atom_count": receptor_audit["coordinate_record_count"],
                "hydrogen_like_atom_count": receptor_audit[
                    "hydrogen_like_atom_count"
                ],
                "autodock_atom_types": ";".join(
                    receptor_audit["autodock_atom_types"]
                ),
                "charge_min": receptor_audit["charge_min"],
                "charge_max": receptor_audit["charge_max"],
                "status": "ok",
            }
        )
        prepared_cases.append(
            {
                "case_id": case_id,
                "conformer_id": conformer_id,
                "pdb_id": case["pdb_id"],
                "chain": case["chain"],
                "ligand_resname": case["selected_ligand_resname"],
                "ligand_resseq": case["selected_ligand_resseq"],
                "ligand_source_url": source_url,
                "raw_sdf": raw_sdf,
                "raw_sdf_sha256": file_sha256(raw_sdf),
                "raw_sdf_molblock_sha256": molblock_sha256(raw_sdf),
                "common_sdf": common_sdf,
                "common_sdf_sha256": file_sha256(common_sdf),
                "explicit_h_sdf": explicit_h_sdf,
                "explicit_h_sdf_sha256": file_sha256(explicit_h_sdf),
                "ligand_pdbqt": ligand_pdbqt,
                "ligand_pdbqt_sha256": file_sha256(ligand_pdbqt),
                "ligand_pdbqt_audit": ligand_pdbqt_audit,
                "receptor_pdbqt": receptor_pdbqt,
                "receptor_pdbqt_sha256": file_sha256(receptor_pdbqt),
                "alignment_summary": alignment_summary_path,
                "alignment_summary_sha256": file_sha256(alignment_summary_path),
                "reference_coordinate_rmsd_angstrom": reference_rmsd,
                "reference_coordinate_max_distance_angstrom": reference_max_distance,
                "box_audit": current_box_audit,
            }
        )

    write_csv(receptor_manifest_path, receptor_rows)
    existing_revalidation_cases: list[dict[str, object]] = []
    configured_existing_cases = config.get("existing_revalidation_cases", [])
    if not isinstance(configured_existing_cases, list):
        raise ValueError("existing revalidation cases must be a list")
    for configured_case in configured_existing_cases:
        if not isinstance(configured_case, dict):
            raise ValueError("existing revalidation case is invalid")
        reference_sdf = checked_record(configured_case["reference_sdf"])
        ligand_pdbqt = checked_record(configured_case["ligand_pdbqt"])
        receptor_pdbqt = checked_record(configured_case["receptor_pdbqt"])
        coordinates, _, _ = coordinates_and_elements_from_sdf(reference_sdf)
        existing_revalidation_cases.append(
            {
                "case_id": str(configured_case["case_id"]),
                "conformer_id": str(configured_case["conformer_id"]),
                "ligand_resname": str(configured_case["reference_ligand"]),
                "common_sdf": reference_sdf,
                "common_sdf_sha256": file_sha256(reference_sdf),
                "ligand_pdbqt": ligand_pdbqt,
                "ligand_pdbqt_sha256": file_sha256(ligand_pdbqt),
                "receptor_pdbqt": receptor_pdbqt,
                "receptor_pdbqt_sha256": file_sha256(receptor_pdbqt),
                "box_audit": box_audit(coordinates, center, size),
                "source_role": "previously_redocking_approved_revalidation",
            }
        )
    all_redocking_cases = prepared_cases + existing_revalidation_cases
    box_failures = [
        case
        for case in all_redocking_cases
        if float(case["box_audit"]["minimum_margin_angstrom"]) < minimum_box_margin
    ]
    base_summary: dict[str, object] = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "config": {"path": config_path.as_posix(), "sha256": file_sha256(config_path)},
        "runtime": runtime,
        "selection_summary": {
            "path": input_paths["selection_summary"].as_posix(),
            "sha256": file_sha256(input_paths["selection_summary"]),
        },
        "selection_audit": {
            "path": input_paths["selection_audit"].as_posix(),
            "sha256": file_sha256(input_paths["selection_audit"]),
        },
        "protocol_amendment": (
            {
                "path": protocol_amendment_path.as_posix(),
                "sha256": file_sha256(protocol_amendment_path),
            }
            if protocol_amendment_path is not None
            else None
        ),
        "receptor_count": len(prepared_cases),
        "existing_revalidation_case_count": len(existing_revalidation_cases),
        "total_redocking_case_count": len(all_redocking_cases),
        "common_box": box,
        "case_preparation": [
            {
                **{
                    key: value
                    for key, value in case.items()
                    if not isinstance(value, Path)
                    and key not in {"ligand_pdbqt_audit"}
                },
                "raw_sdf": case["raw_sdf"].as_posix(),
                "common_sdf": case["common_sdf"].as_posix(),
                "explicit_h_sdf": case["explicit_h_sdf"].as_posix(),
                "ligand_pdbqt": case["ligand_pdbqt"].as_posix(),
                "receptor_pdbqt": case["receptor_pdbqt"].as_posix(),
                "alignment_summary": case["alignment_summary"].as_posix(),
                "ligand_pdbqt_audit": case["ligand_pdbqt_audit"],
            }
            for case in prepared_cases
        ],
        "receptor_manifest": {
            "path": receptor_manifest_path.as_posix(),
            "sha256": file_sha256(receptor_manifest_path),
        },
        "existing_case_revalidation_inputs": [
            {
                "case_id": case["case_id"],
                "conformer_id": case["conformer_id"],
                "reference_ligand": case["ligand_resname"],
                "reference_sdf": case["common_sdf"].as_posix(),
                "reference_sdf_sha256": case["common_sdf_sha256"],
                "ligand_pdbqt": case["ligand_pdbqt"].as_posix(),
                "ligand_pdbqt_sha256": case["ligand_pdbqt_sha256"],
                "receptor_pdbqt": case["receptor_pdbqt"].as_posix(),
                "receptor_pdbqt_sha256": case["receptor_pdbqt_sha256"],
                "box_audit": case["box_audit"],
            }
            for case in existing_revalidation_cases
        ],
        "data_boundary": {
            "ligand_labels_read": 0,
            "docking_scores_from_benchmark_read": 0,
            "previous_validation_rows_read": 0,
            "test_rows_read": 0,
        },
    }
    if box_failures:
        base_summary["status"] = "expanded_redocking_box_gate_failed"
        base_summary["box_failure_case_ids"] = [case["case_id"] for case in box_failures]
        base_summary["next_gate"] = "preregister a revised common box before any redocking"
        write_json(summary_path, base_summary)
        print(json.dumps(base_summary, indent=2, ensure_ascii=True))
        return base_summary

    redocking_rows: list[dict[str, object]] = []
    for case in all_redocking_cases:
        case_id = str(case["case_id"])
        case_root = run_root / case_id
        vina_config = case_root / "redocking_vina.txt"
        write_vina_config(
            vina_config,
            case["receptor_pdbqt"],
            case["ligand_pdbqt"],
            center,
            size,
            vina_parameters,
        )
        docked_pdbqt = case_root / f"{case_id}_redocked.pdbqt"
        vina_log = case_root / "redocking_vina.log"
        completed = run_checked(
            [
                str(input_paths["vina_executable"]),
                "--config",
                str(vina_config),
                "--out",
                str(docked_pdbqt),
            ],
            f"Vina redocking for {case_id}",
        )
        vina_log.write_text(
            "\n".join(
                value.rstrip()
                for value in (completed.stdout, completed.stderr)
                if value.strip()
            )
            + "\n",
            encoding="ascii",
            errors="replace",
        )
        pose_table = case_root / "pose_rmsd.csv"
        evaluation_summary_path = case_root / "redocking_evaluation.json"
        run_checked(
            [
                sys.executable,
                str(input_paths["evaluate_rmsd_script"]),
                "--case-id",
                case_id,
                "--reference-sdf",
                str(case["common_sdf"]),
                "--docked-pdbqt",
                str(docked_pdbqt),
                "--pose-table-output",
                str(pose_table),
                "--summary-output",
                str(evaluation_summary_path),
                "--success-threshold",
                str(config["redocking_gate"]["maximum_top_ranked_rmsd_angstrom"]),
            ],
            f"redocking RMSD evaluation for {case_id}",
        )
        evaluation = read_json(evaluation_summary_path)
        redocking_rows.append(
            {
                "case_id": case_id,
                "conformer_id": case["conformer_id"],
                "reference_ligand": case["ligand_resname"],
                "reference_sdf": case["common_sdf"].as_posix(),
                "reference_sdf_sha256": case["common_sdf_sha256"],
                "ligand_pdbqt": case["ligand_pdbqt"].as_posix(),
                "ligand_pdbqt_sha256": case["ligand_pdbqt_sha256"],
                "receptor_pdbqt": case["receptor_pdbqt"].as_posix(),
                "receptor_pdbqt_sha256": case["receptor_pdbqt_sha256"],
                "vina_config": vina_config.as_posix(),
                "vina_config_sha256": file_sha256(vina_config),
                "docked_pdbqt": docked_pdbqt.as_posix(),
                "docked_pdbqt_sha256": file_sha256(docked_pdbqt),
                "pose_count": evaluation["pose_count"],
                "top_ranked_affinity_kcal_per_mol": evaluation[
                    "top_ranked_affinity_kcal_per_mol"
                ],
                "top_ranked_rmsd_angstrom": evaluation[
                    "top_ranked_rmsd_angstrom"
                ],
                "top_ranked_pose_success": evaluation[
                    "top_ranked_pose_success"
                ],
                "best_rmsd_pose_rank": evaluation["best_rmsd_pose_rank"],
                "best_rmsd_angstrom": evaluation["best_rmsd_angstrom"],
                "any_pose_success": evaluation["any_pose_success"],
            }
        )
    write_csv(redocking_results_path, redocking_rows)
    all_top_ranked_pass = all(
        str(row["top_ranked_pose_success"]).lower() == "true"
        or row["top_ranked_pose_success"] is True
        for row in redocking_rows
    )
    base_summary.update(
        {
            "status": (
                "expanded_redocking_gate_ok"
                if all_top_ranked_pass
                else "expanded_redocking_gate_failed"
            ),
            "vina": {
                "executable": input_paths["vina_executable"].as_posix(),
                "executable_sha256": file_sha256(input_paths["vina_executable"]),
                "parameters": vina_parameters,
            },
            "redocking_results": redocking_rows,
            "redocking_results_csv": {
                "path": redocking_results_path.as_posix(),
                "sha256": file_sha256(redocking_results_path),
            },
            "all_top_ranked_poses_within_threshold": all_top_ranked_pass,
            "next_gate": (
                "dock the frozen development-train ligands against only the four new receptors and reuse the existing four-receptor scores"
                if all_top_ranked_pass
                else "stop before expanded benchmark docking and review failed redocking cases"
            ),
            "interpretation_boundary": "Passing redocking establishes pose reproduction for the configured co-crystal cases only. It does not establish affinity accuracy, virtual-screening enrichment, receptor complementarity, or QUBO benefit.",
        }
    )
    write_json(summary_path, base_summary)
    print(json.dumps(base_summary, indent=2, ensure_ascii=True))
    return base_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        summary = run_gate(args.config, args.overwrite)
    except Exception as error:
        config = read_json(args.config)
        summary_path = Path(str(config["outputs"]["summary_json"]))
        failure = {
            "schema_version": "1.0",
            "experiment_id": config.get("experiment_id", "unknown"),
            "status": "expanded_redocking_execution_failed",
            "error": f"{type(error).__name__}: {error}",
            "data_boundary": {
                "ligand_labels_read": 0,
                "docking_scores_from_benchmark_read": 0,
                "previous_validation_rows_read": 0,
                "test_rows_read": 0,
            },
        }
        write_json(summary_path, failure)
        raise
    return 0 if summary["status"] == "expanded_redocking_gate_ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
