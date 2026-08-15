"""Audit and select the preregistered EGFR structural receptor pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np

try:
    from .select_mk14_rcsb_coordinate_pool import (
        BACKBONE_NAMES,
        STANDARD_AMINO_ACID_HEAVY_ATOMS,
        kabsch,
        maxmin_select,
        pairwise_feature_vector,
        rmsd,
    )
except ImportError:
    from select_mk14_rcsb_coordinate_pool import (
        BACKBONE_NAMES,
        STANDARD_AMINO_ACID_HEAVY_ATOMS,
        kabsch,
        maxmin_select,
        pairwise_feature_vector,
        rmsd,
    )


@dataclass(frozen=True)
class AtomRecord:
    kind: str
    atom_name: str
    altloc: str
    resname: str
    resseq: int
    icode: str
    coord: np.ndarray
    occupancy: float
    b_iso: float
    element: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified(path: Path, expected_sha256: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != expected_sha256.upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def load_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "implementation",
        "preregistration",
        "preregistration_amendment",
        "runtime",
        "inputs",
        "download",
        "outputs",
        "interpretation_boundary",
    }
    if not required.issubset(config):
        raise ValueError("Stage 13 coordinate-selection config is incomplete")
    return config


def normalized_char(value: object) -> str:
    text = str(value)
    return "" if text in {"", " ", ".", "?", "\x00"} else text


def altloc_rank(atom: AtomRecord) -> tuple[float, int, int, str]:
    occupancy = atom.occupancy if math.isfinite(atom.occupancy) else 0.0
    return (
        -occupancy,
        0 if atom.altloc == "" else 1,
        0 if atom.altloc == "A" else 1,
        atom.altloc,
    )


def select_chain_atoms(structure: gemmi.Structure, chain_name: str) -> list[AtomRecord]:
    if not structure or not structure[0]:
        raise ValueError("mmCIF contains no coordinate model")
    chains = [chain for chain in structure[0] if chain.name == chain_name]
    if not chains:
        raise ValueError(f"author chain is absent: {chain_name}")
    grouped: dict[tuple[object, ...], list[AtomRecord]] = defaultdict(list)
    for chain in chains:
        for residue in chain:
            if residue.entity_type == gemmi.EntityType.Water:
                continue
            is_protein = (
                residue.entity_type == gemmi.EntityType.Polymer
                and residue.name in STANDARD_AMINO_ACID_HEAVY_ATOMS
            )
            kind = "protein" if is_protein else "hetero"
            resseq = int(residue.seqid.num)
            icode = normalized_char(residue.seqid.icode)
            for atom in residue:
                record = AtomRecord(
                    kind=kind,
                    atom_name=str(atom.name).strip(),
                    altloc=normalized_char(atom.altloc),
                    resname=str(residue.name),
                    resseq=resseq,
                    icode=icode,
                    coord=np.array(
                        [atom.pos.x, atom.pos.y, atom.pos.z], dtype=float
                    ),
                    occupancy=float(atom.occ),
                    b_iso=float(atom.b_iso),
                    element=str(atom.element.name).upper(),
                )
                key = (kind, resseq, icode, residue.name, record.atom_name)
                grouped[key].append(record)
    selected = [min(values, key=altloc_rank) for values in grouped.values()]
    return sorted(
        selected,
        key=lambda atom: (
            0 if atom.kind == "protein" else 1,
            atom.resseq,
            atom.icode,
            atom.resname,
            atom.atom_name,
        ),
    )


def is_heavy(atom: AtomRecord) -> bool:
    return atom.element not in {"H", "D"}


def protein_residue_map(
    atoms: list[AtomRecord],
) -> dict[tuple[int, str], list[AtomRecord]]:
    output: dict[tuple[int, str], list[AtomRecord]] = defaultdict(list)
    for atom in atoms:
        if atom.kind == "protein" and is_heavy(atom):
            output[(atom.resseq, atom.icode)].append(atom)
    return output


def ligand_residue_map(
    atoms: list[AtomRecord], qualifying_ids: set[str]
) -> dict[tuple[str, int, str], list[AtomRecord]]:
    output: dict[tuple[str, int, str], list[AtomRecord]] = defaultdict(list)
    for atom in atoms:
        if atom.kind == "hetero" and atom.resname in qualifying_ids and is_heavy(atom):
            output[(atom.resname, atom.resseq, atom.icode)].append(atom)
    return output


def ca_map(atoms: list[AtomRecord]) -> dict[tuple[int, str], AtomRecord]:
    return {
        key: next(atom for atom in values if atom.atom_name == "CA")
        for key, values in protein_residue_map(atoms).items()
        if any(atom.atom_name == "CA" for atom in values)
    }


def residue_point(
    values: list[AtomRecord],
) -> tuple[np.ndarray, np.ndarray] | None:
    ca = [atom.coord for atom in values if atom.atom_name == "CA"]
    if not ca:
        return None
    sidechain = [
        atom.coord for atom in values if atom.atom_name not in BACKBONE_NAMES
    ]
    geometry = sidechain or [atom.coord for atom in values]
    return ca[0], np.vstack(geometry).mean(axis=0)


def minimum_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2).min()
    )


def transform_atoms(
    atoms: list[AtomRecord], rotation: np.ndarray, translation: np.ndarray
) -> list[AtomRecord]:
    return [
        replace(atom, coord=atom.coord @ rotation + translation) for atom in atoms
    ]


def derive_reference_residues(
    atoms: list[AtomRecord], ligand_comp_id: str, cutoff: float
) -> list[int]:
    ligands = ligand_residue_map(atoms, {ligand_comp_id})
    if len(ligands) != 1:
        raise ValueError("reference ligand is missing or ambiguous")
    ligand_coords = np.vstack(
        [atom.coord for atom in next(iter(ligands.values()))]
    )
    selected: list[int] = []
    for (number, icode), values in protein_residue_map(atoms).items():
        if icode:
            continue
        coordinates = np.vstack([atom.coord for atom in values])
        if minimum_distance(coordinates, ligand_coords) <= cutoff:
            selected.append(number)
    return sorted(selected)


def incomplete_standard_residues(atoms: list[AtomRecord]) -> list[str]:
    output: list[str] = []
    for (number, icode), values in sorted(protein_residue_map(atoms).items()):
        expected = STANDARD_AMINO_ACID_HEAVY_ATOMS[values[0].resname]
        observed = {atom.atom_name for atom in values}
        missing = sorted(expected - observed)
        if missing:
            output.append(
                f"{values[0].resname}:{number}{icode}[{','.join(missing)}]"
            )
    return output


def explicit_covalent_connections(
    structure: gemmi.Structure,
    chain_name: str,
    ligand_key: tuple[str, int, str],
    protein_numbers: set[int],
) -> list[str]:
    ligand_name, ligand_number, _ = ligand_key
    matches: list[str] = []
    for connection in structure.connections:
        if connection.type != gemmi.ConnectionType.Covale:
            continue
        first, second = connection.partner1, connection.partner2
        for ligand, protein in ((first, second), (second, first)):
            ligand_seq = int(ligand.res_id.seqid.num)
            protein_seq = int(protein.res_id.seqid.num)
            if (
                ligand.res_id.name == ligand_name
                and ligand_seq == ligand_number
                and protein.res_id.name in STANDARD_AMINO_ACID_HEAVY_ATOMS
                and protein_seq in protein_numbers
                and protein.chain_name == chain_name
            ):
                matches.append(
                    f"{protein.res_id.name}:{protein_seq}:{protein.atom_name}-"
                    f"{ligand.res_id.name}:{ligand_seq}:{ligand.atom_name}"
                )
    return sorted(set(matches))


def write_aligned_protein_pdb(path: Path, atoms: list[AtomRecord]) -> None:
    lines = ["REMARK 900 LABEL-INDEPENDENT EGFR STRUCTURAL POOL ALIGNMENT"]
    serial = 1
    for atom in atoms:
        if atom.kind != "protein":
            continue
        name = atom.atom_name[:4]
        atom_field = f" {name:<3}" if len(name) < 4 and len(atom.element) == 1 else f"{name:>4}"
        x, y, z = atom.coord
        occupancy = atom.occupancy if math.isfinite(atom.occupancy) else 1.0
        b_iso = atom.b_iso if math.isfinite(atom.b_iso) else 0.0
        lines.append(
            f"ATOM  {serial:5d} {atom_field} {atom.resname:>3} A"
            f"{atom.resseq:4d}{atom.icode[:1] or ' '}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_iso:6.2f}"
            f"          {atom.element:>2}"
        )
        serial += 1
    lines.extend(["TER", "END"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def download_one(
    pdb_id: str,
    path: Path,
    url_template: str,
    timeout_seconds: float,
    maximum_retries: int,
    retry_backoff_seconds: float,
) -> tuple[str, str]:
    if path.is_file() and path.stat().st_size > 0:
        return pdb_id, ""
    last_error: Exception | None = None
    for attempt in range(maximum_retries):
        try:
            request = urllib.request.Request(
                url_template.format(pdb_id=pdb_id),
                headers={"User-Agent": "qubo-receptor-ensemble-selection/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = response.read()
            if b"_atom_site." not in data:
                raise ValueError("downloaded mmCIF has no atom_site loop")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return pdb_id, ""
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = error
            if attempt + 1 < maximum_retries:
                time.sleep(retry_backoff_seconds * (2**attempt))
    assert last_error is not None
    return pdb_id, f"{type(last_error).__name__}: {last_error}"


def audit_structure(
    pdb_id: str,
    chain_name: str,
    qualifying_ids: set[str],
    path: Path,
    aligned_path: Path,
    reference_atoms: list[AtomRecord],
    reference_ligand_coords: np.ndarray,
    pocket_numbers: list[int],
    anchor_numbers: set[int],
    gate: dict[str, object],
) -> tuple[dict[str, object], np.ndarray | None, list[str] | None]:
    structure = gemmi.read_structure(str(path))
    atoms = select_chain_atoms(structure, chain_name)
    reference_ca = ca_map(reference_atoms)
    mobile_ca = ca_map(atoms)
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
        raise ValueError("fewer than three matched C-alpha atoms")
    reference_coordinates = np.vstack(
        [reference_ca[key].coord for key in matched]
    )
    mobile_coordinates = np.vstack([mobile_ca[key].coord for key in matched])
    rotation, translation = kabsch(mobile_coordinates, reference_coordinates)
    transformed = transform_atoms(atoms, rotation, translation)
    aligned_mobile = mobile_coordinates @ rotation + translation
    aligned_rmsd = rmsd(aligned_mobile, reference_coordinates)
    determinant = float(np.linalg.det(rotation))

    reference_residues = protein_residue_map(reference_atoms)
    candidate_residues = protein_residue_map(transformed)
    present: list[int] = []
    missing: list[int] = []
    mismatched_pocket: list[int] = []
    incomplete_anchors: list[int] = []
    expected_pocket_atoms: set[tuple[int, str]] = set()
    observed_pocket_atoms: set[tuple[int, str]] = set()
    ca_points: list[np.ndarray] = []
    sidechain_points: list[np.ndarray] = []
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
            if number in anchor_numbers and not expected_names.issubset(observed_names):
                incomplete_anchors.append(number)
            ca_point, sidechain_point = point
        else:
            if candidate_values and candidate_name != reference_name:
                mismatched_pocket.append(number)
            else:
                missing.append(number)
            reference_point = residue_point(reference_values)
            assert reference_point is not None
            ca_point, sidechain_point = reference_point
        ca_points.append(ca_point)
        sidechain_points.append(sidechain_point)
    pocket_fraction = len(present) / len(pocket_numbers)
    missing_anchors = sorted(anchor_numbers - set(present))
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
    ligand_distance = closest[0] if closest else math.nan
    selected_ligand = closest[1] if closest else ("", 0, "")
    protein_heavy = [
        atom.coord for atom in transformed if atom.kind == "protein" and is_heavy(atom)
    ]
    selected_ligand_heavy = (
        [atom.coord for atom in closest[2]] if closest else []
    )
    protein_ligand_distance = (
        minimum_distance(
            np.vstack(protein_heavy), np.vstack(selected_ligand_heavy)
        )
        if protein_heavy and selected_ligand_heavy
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
    incomplete = incomplete_standard_residues(transformed)
    ca_count = len(ca_map(transformed))
    heavy_count = len(protein_heavy)

    reasons: list[str] = []
    if len(matched) < int(gate["minimum_matched_ca_count"]):
        reasons.append("too_few_matched_ca")
    if aligned_rmsd > float(gate["maximum_aligned_global_ca_rmsd_angstrom"]):
        reasons.append("aligned_global_ca_rmsd_above_limit")
    if determinant < 0.999999:
        reasons.append("rotation_is_not_proper")
    if pocket_fraction < float(gate["minimum_pocket_residue_fraction"]):
        reasons.append("pocket_residue_fraction_below_limit")
    if mismatched_pocket:
        reasons.append("pocket_residue_name_mismatch")
    if missing_anchors:
        reasons.append("missing_required_anchor_residue")
    if incomplete_anchors:
        reasons.append("incomplete_anchor_heavy_atom_template")
    if pocket_heavy_fraction < float(
        gate["minimum_pocket_heavy_atom_completeness_fraction"]
    ):
        reasons.append("pocket_heavy_atom_completeness_below_limit")
    if not math.isfinite(ligand_distance) or ligand_distance > float(
        gate["require_qualifying_ligand_within_reference_ligand_angstrom"]
    ):
        reasons.append("no_qualifying_ligand_in_reference_atp_site")
    covalent_gate = dict(gate["hidden_covalency"])
    if explicit_connections:
        reasons.append("explicit_target_ligand_covalent_connection")
    if math.isfinite(protein_ligand_distance) and protein_ligand_distance <= float(
        covalent_gate[
            "exclude_if_minimum_protein_ligand_heavy_atom_distance_at_or_below_angstrom"
        ]
    ):
        reasons.append("protein_ligand_distance_indicates_covalency")

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
        "matched_ca_count": len(matched),
        "aligned_global_ca_rmsd_angstrom": aligned_rmsd,
        "rotation_determinant": determinant,
        "global_residue_name_mismatch_count": len(mismatched),
        "protein_ca_count": ca_count,
        "protein_heavy_atom_count": heavy_count,
        "protein_heavy_atoms_per_ca": heavy_count / ca_count if ca_count else 0.0,
        "pocket_present_count": len(present),
        "pocket_residue_fraction": pocket_fraction,
        "pocket_heavy_atom_completeness_fraction": pocket_heavy_fraction,
        "missing_pocket_residues": ";".join(map(str, missing)),
        "mismatched_pocket_residues": ";".join(map(str, mismatched_pocket)),
        "missing_anchor_residues": ";".join(map(str, missing_anchors)),
        "incomplete_anchor_residues": ";".join(map(str, incomplete_anchors)),
        "selected_ligand_resname": selected_ligand[0],
        "selected_ligand_resseq": selected_ligand[1],
        "selected_ligand_icode": selected_ligand[2],
        "selected_ligand_heavy_atom_count": len(closest[2]) if closest else 0,
        "selected_ligand_min_distance_to_reference_ligand_angstrom": ligand_distance,
        "minimum_protein_ligand_heavy_atom_distance_angstrom": protein_ligand_distance,
        "explicit_covalent_connections": ";".join(explicit_connections),
        "global_incomplete_standard_amino_acid_residue_count": len(incomplete),
        "global_incomplete_standard_amino_acid_residues": ";".join(incomplete),
        "imputed_pocket_feature_count": len(pocket_numbers) - len(present),
    }
    return row, vector, feature_names


def write_report(path: Path, summary: dict[str, object]) -> None:
    counts = dict(summary["counts"])
    lines = [
        "# Stage 13 EGFR Coordinate Pool",
        "",
        "## Result",
        "",
        f"- Audited structures: {counts['audited_count']}",
        f"- Coordinate-eligible structures: {counts['coordinate_eligible_count']}",
        f"- Hidden-covalent exclusions: {counts['hidden_covalent_exclusion_count']}",
        f"- Selected receptors: {counts['selected_receptor_count']}",
        f"- Status: {summary['status']}",
        "",
        "No ligand activity label, docking score, MAPK14 Stage 11 row, fresh-validation row, or test row was read.",
        "Receptor preparation and co-crystal redocking remain mandatory before benchmark docking.",
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
        raise ValueError("Stage 13 coordinate implementation path differs")
    verified(
        Path(str(implementation["dependency_path"])),
        str(implementation["dependency_sha256"]),
    )
    prereg_record = dict(config["preregistration"])
    amendment_record = dict(config["preregistration_amendment"])
    prereg_path = verified(
        Path(str(prereg_record["path"])), str(prereg_record["sha256"])
    )
    amendment_path = verified(
        Path(str(amendment_record["path"])), str(amendment_record["sha256"])
    )
    preregistration = read_json(prereg_path)
    amendment = read_json(amendment_path)
    if amendment["original_preregistration"]["sha256"] != file_sha256(prereg_path):
        raise ValueError("Stage 13 amendment does not match preregistration")
    boundary = dict(amendment["data_boundary"])
    if any(int(value) != 0 for value in boundary.values()):
        raise ValueError("Stage 13 coordinate data boundary differs")

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
        raise RuntimeError(f"Stage 13 coordinate runtime differs: {runtime} != {expected_runtime}")

    inputs: dict[str, Path] = {}
    for key, record_value in dict(config["inputs"]).items():
        record = dict(record_value)
        inputs[key] = verified(Path(str(record["path"])), str(record["sha256"]))
    discovery = read_json(inputs["discovery_summary"])
    if (
        discovery.get("status") != "stage13_egfr_metadata_discovery_ok"
        or any(int(value) != 0 for value in discovery["data_boundary"].values())
    ):
        raise ValueError("Stage 13 metadata discovery is not admissible")

    outputs = {key: Path(str(value)) for key, value in dict(config["outputs"]).items()}
    directory_keys = {"raw_mmcif_directory", "aligned_protein_pdb_directory"}
    file_outputs = [path for key, path in outputs.items() if key not in directory_keys]
    existing = [path for path in file_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Stage 13 coordinate outputs exist; pass --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()

    candidates = read_csv(inputs["eligible_candidates_csv"])
    expected_count = int(amendment["upstream_metadata_discovery"]["eligible_candidate_count"])
    if len(candidates) != expected_count:
        raise ValueError("Stage 13 eligible candidate count differs")
    if len({row["pdb_id"] for row in candidates}) != len(candidates):
        raise ValueError("Stage 13 candidate PDB IDs are not unique")

    raw_directory = outputs["raw_mmcif_directory"]
    raw_paths = {
        row["pdb_id"]: raw_directory / f"{row['pdb_id']}.cif" for row in candidates
    }
    reference_record = dict(amendment["reference"])
    configured_reference = inputs["reference_mmcif"]
    reference_destination = raw_paths[str(reference_record["pdb_id"])]
    if configured_reference.resolve() != reference_destination.resolve():
        raise ValueError("Stage 13 reference mmCIF path differs")
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

    reference_structure = gemmi.read_structure(str(configured_reference))
    reference_atoms = select_chain_atoms(
        reference_structure, str(reference_record["auth_chain"])
    )
    reference_ligand_id = str(reference_record["ligand_comp_id"])
    pocket_numbers = [int(value) for value in reference_record["reference_pocket_residue_numbers"]]
    anchor_numbers = {int(value) for value in reference_record["required_anchor_residue_numbers"]}
    if derive_reference_residues(reference_atoms, reference_ligand_id, 6.0) != pocket_numbers:
        raise ValueError("Stage 13 reference pocket reconstruction differs")
    if derive_reference_residues(reference_atoms, reference_ligand_id, 4.0) != sorted(anchor_numbers):
        raise ValueError("Stage 13 reference anchor reconstruction differs")
    if len(ca_map(reference_atoms)) != int(reference_record["visible_protein_ca_count"]):
        raise ValueError("Stage 13 reference C-alpha count differs")
    reference_ligands = ligand_residue_map(reference_atoms, {reference_ligand_id})
    reference_ligand_coords = np.vstack(
        [atom.coord for atom in next(iter(reference_ligands.values()))]
    )
    gate = dict(amendment["coordinate_gate"])

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
            audit_rows.append(
                {
                    "conformer_id": conformer_id,
                    "pdb_id": pdb_id,
                    "chain": metadata["selected_auth_chain"],
                    "status": "coordinate_excluded",
                    "exclusion_reasons": "coordinate_file_unavailable",
                    "download_error": download_errors[pdb_id],
                    "mmcif_path": "",
                    "mmcif_sha256": "",
                    "aligned_protein_pdb_path": "",
                    "aligned_protein_pdb_sha256": "",
                }
            )
            continue
        aligned_path = aligned_directory / f"{conformer_id}_to_2RGP_A.pdb"
        try:
            row, vector, names = audit_structure(
                pdb_id,
                metadata["selected_auth_chain"],
                {value for value in metadata["qualifying_ligand_ids"].split(";") if value},
                raw_paths[pdb_id],
                aligned_path,
                reference_atoms,
                reference_ligand_coords,
                pocket_numbers,
                anchor_numbers,
                gate,
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
                    "aligned_protein_pdb_path": "",
                    "aligned_protein_pdb_sha256": "",
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
        raise ValueError("Stage 13 reference failed coordinate eligibility")

    feature_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    variable_feature_count = 0
    if feature_names is not None and vectors:
        ordered_ids = sorted(vectors)
        matrix = np.vstack([vectors[conformer_id] for conformer_id in ordered_ids])
        standard_deviations = matrix.std(axis=0)
        keep = standard_deviations >= float(
            amendment["structural_selection"]["minimum_variable_feature_sd_angstrom"]
        )
        variable_feature_count = int(keep.sum())
        if variable_feature_count < 3:
            raise ValueError("too few variable Stage 13 structural features")
        means = matrix.mean(axis=0)
        standardized = (matrix[:, keep] - means[keep]) / standard_deviations[keep]
        standardized /= math.sqrt(variable_feature_count)
        distance_by_pair: dict[tuple[str, str], float] = {}
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
        target_count = int(amendment["structural_selection"]["target_receptor_count"])
        if len(ordered_ids) >= target_count:
            additions = maxmin_select(
                ordered_ids,
                [reference_id],
                distance_by_pair,
                target_count - 1,
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
    if feature_rows:
        write_csv(outputs["feature_matrix_csv"], feature_rows)
    if distance_rows:
        write_csv(outputs["pairwise_distances_csv"], distance_rows)
    if selected_rows:
        write_csv(outputs["selected16_manifest_csv"], selected_rows)

    reason_counts = Counter(
        reason
        for row in audit_rows
        for reason in str(row.get("exclusion_reasons", "")).split(";")
        if reason
    )
    hidden_covalent_count = sum(
        "explicit_target_ligand_covalent_connection" in str(row.get("exclusion_reasons", ""))
        or "protein_ligand_distance_indicates_covalency" in str(row.get("exclusion_reasons", ""))
        for row in audit_rows
    )
    target_count = int(amendment["structural_selection"]["target_receptor_count"])
    status = (
        "stage13_egfr_structural_selection_ok"
        if len(selected_rows) == target_count
        else "stage13_egfr_coordinate_pool_insufficient_stop"
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
        "preregistration": {"path": prereg_path.as_posix(), "sha256": file_sha256(prereg_path)},
        "preregistration_amendment": {"path": amendment_path.as_posix(), "sha256": file_sha256(amendment_path)},
        "runtime": runtime,
        "counts": {
            "audited_count": len(audit_rows),
            "coordinate_eligible_count": len(eligible_rows),
            "coordinate_excluded_count": len(audit_rows) - len(eligible_rows),
            "hidden_covalent_exclusion_count": hidden_covalent_count,
            "selected_receptor_count": len(selected_rows),
            "target_receptor_count": target_count,
            "raw_feature_count": len(feature_names or []),
            "variable_feature_count": variable_feature_count,
            "pairwise_distance_count": len(distance_rows),
        },
        "coordinate_exclusion_reason_counts": dict(sorted(reason_counts.items())),
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
            "freeze one deterministic missing-heavy-atom completion protocol, prepare the selected receptors and native ligands, and require co-crystal redocking RMSD at or below 2.0 A"
            if status == "stage13_egfr_structural_selection_ok"
            else "do not dock; adjudicate the frozen coordinate exclusions and revise the receptor-pool design prospectively"
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
