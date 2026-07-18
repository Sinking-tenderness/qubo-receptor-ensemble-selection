"""Audit and structurally select an expanded MAPK14 receptor pool."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    from .prepare_receptor import file_sha256
except ImportError:
    from prepare_receptor import file_sha256


BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}
REQUIRED_INPUT_KEYS = {
    "discovery_summary",
    "candidate_metadata",
    "eligible_new_candidates",
    "existing_aligned_manifest",
    "reference_pdb",
}
REQUIRED_OUTPUT_KEYS = {
    "raw_pdb_directory",
    "aligned_pdb_directory",
    "coordinate_audit_csv",
    "eligible_pool_manifest_csv",
    "feature_matrix_csv",
    "pairwise_distances_csv",
    "selected_expansion_manifest_csv",
    "summary_json",
}
PREPARATION_READINESS_KEYS = {
    "minimum_chain_heavy_atoms_per_ca",
    "minimum_pocket_heavy_atom_completeness_fraction",
    "exclude_polymer_like_hetero_residues",
    "polymer_backbone_atom_names",
}
OPTIONAL_PREPARATION_READINESS_KEYS = {
    "exclude_incomplete_standard_amino_acid_residues",
}
STANDARD_AMINO_ACID_HEAVY_ATOMS = {
    "ALA": {"N", "CA", "C", "O", "CB"},
    "ARG": {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    "ASP": {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"},
    "CYS": {"N", "CA", "C", "O", "CB", "SG"},
    "GLN": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    "GLY": {"N", "CA", "C", "O"},
    "HIS": {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
    "LEU": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"},
    "LYS": {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    "MET": {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
    "PHE": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"N", "CA", "C", "O", "CB", "CG", "CD"},
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
    "THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    "TRP": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"N", "CA", "C", "O", "CB", "CG1", "CG2"},
}


@dataclass(frozen=True)
class Atom:
    line: str
    record: str
    serial: int
    atom_name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    coord: np.ndarray
    occupancy: float
    element: str


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
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
        raise ValueError(f"cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "preregistration",
        "runtime",
        "inputs",
        "download",
        "coordinate_gate",
        "outputs",
        "interpretation_boundary",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"coordinate-selection config is missing: {', '.join(missing)}")
    preregistration = config["preregistration"]
    runtime = config["runtime"]
    inputs = config["inputs"]
    outputs = config["outputs"]
    if not isinstance(preregistration, dict) or set(preregistration) != {
        "path",
        "sha256",
    }:
        raise ValueError("preregistration record is invalid")
    if not isinstance(runtime, dict) or set(runtime) != {
        "conda_environment",
        "python_version",
        "numpy_version",
    }:
        raise ValueError("runtime lock is incomplete")
    if not isinstance(inputs, dict) or set(inputs) != REQUIRED_INPUT_KEYS:
        raise ValueError("coordinate-selection inputs differ from required set")
    for key, record in inputs.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"invalid input record: {key}")
    if not isinstance(outputs, dict) or set(outputs) != REQUIRED_OUTPUT_KEYS:
        raise ValueError("coordinate-selection outputs differ from required set")
    download = config["download"]
    gate = config["coordinate_gate"]
    if not isinstance(download, dict) or int(download.get("workers", 0)) < 1:
        raise ValueError("download settings are invalid")
    if not isinstance(gate, dict):
        raise ValueError("coordinate_gate is missing")
    if int(gate.get("minimum_matched_ca_count", 0)) < 3:
        raise ValueError("minimum_matched_ca_count is too small")
    if float(gate.get("maximum_aligned_global_ca_rmsd_angstrom", 0.0)) <= 0.0:
        raise ValueError("maximum aligned RMSD must be positive")
    if gate.get("missing_pocket_feature_imputation") != (
        "reference_aligned_coordinates"
    ):
        raise ValueError("missing pocket imputation rule changed")
    readiness = config.get("preparation_readiness_gate")
    if readiness is not None:
        if (
            not isinstance(readiness, dict)
            or not PREPARATION_READINESS_KEYS.issubset(readiness)
            or not set(readiness).issubset(
                PREPARATION_READINESS_KEYS | OPTIONAL_PREPARATION_READINESS_KEYS
            )
        ):
            raise ValueError("preparation readiness gate is invalid")
        if float(readiness["minimum_chain_heavy_atoms_per_ca"]) <= 1.0:
            raise ValueError("chain heavy-atom-per-CA threshold is too small")
        pocket_threshold = float(
            readiness["minimum_pocket_heavy_atom_completeness_fraction"]
        )
        if not 0.0 < pocket_threshold <= 1.0:
            raise ValueError("pocket heavy-atom completeness threshold is invalid")
        backbone_names = readiness["polymer_backbone_atom_names"]
        if not isinstance(backbone_names, list) or set(backbone_names) != {
            "N",
            "CA",
            "C",
            "O",
        }:
            raise ValueError("polymer backbone atom-name rule changed")
        amendment = config.get("preregistration_amendment")
        if not isinstance(amendment, dict) or set(amendment) != {"path", "sha256"}:
            raise ValueError("preparation readiness gate requires an amendment record")
    return config


def check_runtime(config: dict[str, object]) -> dict[str, str]:
    expected = config["runtime"]
    assert isinstance(expected, dict)
    actual = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    if actual != {key: str(value) for key, value in expected.items()}:
        raise RuntimeError(f"runtime differs: {actual} != {expected}")
    return actual


def checked_inputs(config: dict[str, object]) -> dict[str, Path]:
    inputs = config["inputs"]
    assert isinstance(inputs, dict)
    paths: dict[str, Path] = {}
    for key, record in inputs.items():
        assert isinstance(record, dict)
        path = Path(str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(record["sha256"]).upper():
            raise ValueError(f"input SHA-256 differs: {key}")
        paths[key] = path
    return paths


def validate_preregistration(preregistration: dict[str, object]) -> None:
    boundary = preregistration.get("data_boundary")
    expansion = preregistration.get("pool_expansion")
    coordinate = preregistration.get("coordinate_eligibility")
    if not isinstance(boundary, dict) or boundary.get("test") != "locked_unreleased":
        raise ValueError("test boundary changed")
    if (
        boundary.get("labels_allowed_during_structural_selection") is not False
        or boundary.get("docking_scores_allowed_during_structural_selection")
        is not False
    ):
        raise ValueError("structural selection boundary changed")
    if not isinstance(expansion, dict) or (
        int(expansion.get("new_receptor_count", 0)) != 4
        or int(expansion.get("final_receptor_count", 0)) != 8
    ):
        raise ValueError("pool expansion size changed")
    if not isinstance(coordinate, dict):
        raise ValueError("coordinate eligibility rules are missing")
    if coordinate.get("alignment") != (
        "proper-rotation Kabsch alignment using sequence-matched protein C-alpha atoms"
    ):
        raise ValueError("alignment rule changed")


def parse_pdb(path: Path) -> list[Atom]:
    atoms: list[Atom] = []
    active_model = True
    saw_model = False
    for line_number, line in enumerate(
        path.read_text(encoding="ascii").splitlines(), start=1
    ):
        if line.startswith("MODEL "):
            saw_model = True
            model_text = line[10:14].strip() or line[6:].strip()
            active_model = model_text == "1"
            continue
        if line.startswith("ENDMDL"):
            if saw_model and active_model:
                break
            active_model = False
            continue
        if not active_model or not line.startswith(("ATOM  ", "HETATM")):
            continue
        if len(line) < 54:
            raise ValueError(f"invalid PDB coordinate line {line_number}: {path}")
        occupancy_text = line[54:60].strip()
        occupancy = float(occupancy_text) if occupancy_text else 0.0
        atom_name = line[12:16].strip()
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not element:
            element = atom_name.lstrip("0123456789")[:1].upper()
        atoms.append(
            Atom(
                line=line,
                record=line[0:6].strip(),
                serial=int(line[6:11]),
                atom_name=atom_name,
                altloc=line[16:17].strip(),
                resname=line[17:20].strip(),
                chain=line[21:22].strip(),
                resseq=int(line[22:26]),
                icode=line[26:27].strip(),
                coord=np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=float,
                ),
                occupancy=occupancy,
                element=element,
            )
        )
    if not atoms:
        raise ValueError(f"no PDB atoms found: {path}")
    return atoms


def altloc_rank(atom: Atom) -> tuple[float, int, int, str]:
    return (
        -atom.occupancy,
        0 if atom.altloc == "" else 1,
        0 if atom.altloc == "A" else 1,
        atom.altloc,
    )


def select_chain_atoms(atoms: list[Atom], chain: str) -> list[Atom]:
    grouped: dict[tuple[object, ...], list[Atom]] = defaultdict(list)
    for atom in atoms:
        if atom.chain != chain or atom.resname == "HOH":
            continue
        key = (
            atom.record,
            atom.resseq,
            atom.icode,
            atom.resname,
            atom.atom_name,
        )
        grouped[key].append(atom)
    selected = [min(values, key=altloc_rank) for values in grouped.values()]
    return sorted(selected, key=lambda atom: (atom.serial, atom.record))


def is_heavy(atom: Atom) -> bool:
    return atom.element != "H" and not atom.atom_name.upper().startswith("H")


def ca_map(atoms: list[Atom]) -> dict[tuple[int, str], Atom]:
    return {
        (atom.resseq, atom.icode): atom
        for atom in atoms
        if atom.record == "ATOM" and atom.atom_name == "CA"
    }


def match_ca(
    reference: list[Atom], mobile: list[Atom]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    reference_map = ca_map(reference)
    mobile_map = ca_map(mobile)
    keys = sorted(set(reference_map) & set(mobile_map))
    matched: list[tuple[int, str]] = []
    mismatches: list[str] = []
    for key in keys:
        if reference_map[key].resname == mobile_map[key].resname:
            matched.append(key)
        else:
            mismatches.append(
                f"{key[0]}{key[1]}:{reference_map[key].resname}!={mobile_map[key].resname}"
            )
    if len(matched) < 3:
        raise ValueError("fewer than three sequence-number matched C-alpha atoms")
    return (
        np.vstack([reference_map[key].coord for key in matched]),
        np.vstack([mobile_map[key].coord for key in matched]),
        mismatches,
    )


def kabsch(
    mobile: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (reference - reference_center)
    left, _, right_transposed = np.linalg.svd(covariance)
    rotation = left @ right_transposed
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transposed
    translation = reference_center - mobile_center @ rotation
    return rotation, translation


def transform_atoms(
    atoms: list[Atom], rotation: np.ndarray, translation: np.ndarray
) -> list[Atom]:
    return [
        Atom(
            **{
                **atom.__dict__,
                "coord": atom.coord @ rotation + translation,
            }
        )
        for atom in atoms
    ]


def rmsd(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def write_aligned_pdb(path: Path, atoms: list[Atom]) -> None:
    output = ["REMARK 900 LABEL-INDEPENDENT MAPK14 STRUCTURAL POOL ALIGNMENT"]
    for atom in atoms:
        x, y, z = atom.coord
        original = atom.line
        normalized_altloc = f"{original[:16]} {original[17:]}"
        output.append(
            f"{normalized_altloc[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{normalized_altloc[54:]}"
        )
    output.extend(["TER", "END"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(output) + "\n", encoding="ascii")


def residue_atoms(atoms: list[Atom]) -> dict[tuple[int, str], list[Atom]]:
    output: dict[tuple[int, str], list[Atom]] = defaultdict(list)
    for atom in atoms:
        if atom.record == "ATOM" and is_heavy(atom):
            output[(atom.resseq, atom.icode)].append(atom)
    return output


def residue_point(values: list[Atom]) -> tuple[np.ndarray, np.ndarray] | None:
    ca = [atom.coord for atom in values if atom.atom_name == "CA"]
    if not ca:
        return None
    sidechain = [
        atom.coord for atom in values if atom.atom_name not in BACKBONE_NAMES
    ]
    geometry = sidechain or [atom.coord for atom in values]
    return ca[0], np.vstack(geometry).mean(axis=0)


def preparation_readiness_metrics(
    atoms: list[Atom],
    reference_atoms: list[Atom],
    pocket_numbers: list[int],
    polymer_backbone_atom_names: set[str],
) -> dict[str, object]:
    protein_heavy = [
        atom for atom in atoms if atom.record == "ATOM" and is_heavy(atom)
    ]
    ca_count = sum(atom.atom_name == "CA" for atom in protein_heavy)
    heavy_atoms_per_ca = len(protein_heavy) / ca_count if ca_count else 0.0

    reference_by_residue = residue_atoms(reference_atoms)
    candidate_by_residue = residue_atoms(atoms)
    expected_pocket_atoms: set[tuple[int, str]] = set()
    observed_pocket_atoms: set[tuple[int, str]] = set()
    for number in pocket_numbers:
        reference_values = reference_by_residue[(number, "")]
        expected_pocket_atoms.update(
            (number, atom.atom_name) for atom in reference_values
        )
        candidate_values = candidate_by_residue.get((number, ""), [])
        if candidate_values and candidate_values[0].resname == reference_values[0].resname:
            observed_pocket_atoms.update(
                (number, atom.atom_name) for atom in candidate_values
            )
    matched_pocket_atoms = expected_pocket_atoms & observed_pocket_atoms
    pocket_completeness = (
        len(matched_pocket_atoms) / len(expected_pocket_atoms)
        if expected_pocket_atoms
        else 0.0
    )

    hetero_groups: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for atom in atoms:
        if atom.record == "HETATM" and is_heavy(atom):
            hetero_groups[(atom.resname, atom.resseq, atom.icode)].add(
                atom.atom_name
            )
    polymer_like = sorted(
        key
        for key, atom_names in hetero_groups.items()
        if polymer_backbone_atom_names.issubset(atom_names)
    )
    standard_residue_groups: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for atom in protein_heavy:
        standard_residue_groups[(atom.resname, atom.resseq, atom.icode)].add(
            atom.atom_name
        )
    incomplete_standard_residues: list[tuple[str, int, str, list[str]]] = []
    for (resname, resseq, icode), atom_names in sorted(standard_residue_groups.items()):
        expected = STANDARD_AMINO_ACID_HEAVY_ATOMS.get(resname)
        if expected is None:
            continue
        missing_atoms = sorted(expected - atom_names)
        if missing_atoms:
            incomplete_standard_residues.append(
                (resname, resseq, icode, missing_atoms)
            )
    return {
        "chain_protein_heavy_atom_count": len(protein_heavy),
        "chain_ca_count": ca_count,
        "chain_heavy_atoms_per_ca": heavy_atoms_per_ca,
        "reference_pocket_heavy_atom_count": len(expected_pocket_atoms),
        "matched_pocket_heavy_atom_count": len(matched_pocket_atoms),
        "pocket_heavy_atom_completeness_fraction": pocket_completeness,
        "polymer_like_hetero_residues": ";".join(
            f"{resname}:{resseq}{icode}" for resname, resseq, icode in polymer_like
        ),
        "polymer_like_hetero_residue_count": len(polymer_like),
        "incomplete_standard_amino_acid_residues": ";".join(
            f"{resname}:{resseq}{icode}[{','.join(missing_atoms)}]"
            for resname, resseq, icode, missing_atoms in incomplete_standard_residues
        ),
        "incomplete_standard_amino_acid_residue_count": len(
            incomplete_standard_residues
        ),
    }


def ligand_groups(
    atoms: list[Atom], qualifying_ids: set[str]
) -> dict[tuple[str, int, str], list[Atom]]:
    output: dict[tuple[str, int, str], list[Atom]] = defaultdict(list)
    for atom in atoms:
        if atom.record == "HETATM" and atom.resname in qualifying_ids and is_heavy(atom):
            output[(atom.resname, atom.resseq, atom.icode)].append(atom)
    return output


def minimum_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2).min()
    )


def pairwise_feature_vector(
    ca_points: list[np.ndarray], sidechain_points: list[np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    names: list[str] = []
    for prefix, points in (("ca", ca_points), ("sidechain", sidechain_points)):
        for first, second in combinations(range(len(points)), 2):
            values.append(float(np.linalg.norm(points[first] - points[second])))
            names.append(f"{prefix}_{first:02d}_{second:02d}")
    return np.array(values, dtype=float), names


def download_one(
    pdb_id: str,
    path: Path,
    template: str,
    timeout_seconds: float,
    maximum_retries: int,
    retry_backoff_seconds: float,
) -> tuple[str, Path, str]:
    if path.is_file() and path.stat().st_size > 0:
        return pdb_id, path, ""
    url = template.format(pdb_id=pdb_id)
    last_error: Exception | None = None
    for attempt in range(maximum_retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                data = response.read()
            if b"ATOM  " not in data:
                raise ValueError(f"downloaded PDB has no ATOM records: {pdb_id}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return pdb_id, path, ""
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = error
            if attempt + 1 < maximum_retries:
                time.sleep(retry_backoff_seconds * (2**attempt))
    assert last_error is not None
    return pdb_id, path, f"{type(last_error).__name__}: {last_error}"


def maxmin_select(
    ids: list[str],
    existing_ids: list[str],
    distance_by_pair: dict[tuple[str, str], float],
    count: int,
) -> list[dict[str, object]]:
    selected = list(existing_ids)
    remaining = sorted(set(ids) - set(existing_ids))
    output: list[dict[str, object]] = []
    for rank in range(1, count + 1):
        candidates = []
        for candidate in remaining:
            minimum = min(
                distance_by_pair[tuple(sorted((candidate, chosen)))]
                for chosen in selected
            )
            candidates.append((minimum, candidate))
        if not candidates:
            raise ValueError("too few candidates for max-min selection")
        minimum_distance_to_pool, chosen = min(
            candidates, key=lambda value: (-value[0], value[1])
        )
        output.append(
            {
                "selection_rank": rank,
                "conformer_id": chosen,
                "minimum_standardized_distance_to_selected_pool": (
                    minimum_distance_to_pool
                ),
            }
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return output


def audit_structure(
    conformer_id: str,
    pdb_id: str,
    path: Path,
    chain: str,
    qualifying_ligand_ids: set[str],
    reference_atoms: list[Atom],
    reference_ligand_coords: np.ndarray,
    pocket_numbers: list[int],
    anchor_numbers: set[int],
    minimum_pocket_fraction: float,
    ligand_cutoff: float,
    minimum_matched_ca: int,
    maximum_rmsd: float,
    aligned_path: Path,
    preparation_readiness_gate: dict[str, object] | None = None,
) -> tuple[dict[str, object], np.ndarray | None, list[str] | None]:
    raw_atoms = parse_pdb(path)
    mobile_atoms = select_chain_atoms(raw_atoms, chain)
    reference_coords, mobile_coords, mismatches = match_ca(
        reference_atoms, mobile_atoms
    )
    rotation, translation = kabsch(mobile_coords, reference_coords)
    transformed = transform_atoms(mobile_atoms, rotation, translation)
    aligned_mobile = mobile_coords @ rotation + translation
    aligned_rmsd = rmsd(aligned_mobile, reference_coords)
    determinant = float(np.linalg.det(rotation))
    by_residue = residue_atoms(transformed)
    reference_by_residue = residue_atoms(reference_atoms)
    reference_names = {
        number: reference_by_residue[(number, "")][0].resname
        for number in pocket_numbers
    }
    present: list[int] = []
    mismatched_pocket: list[int] = []
    missing: list[int] = []
    ca_points: list[np.ndarray] = []
    sidechain_points: list[np.ndarray] = []
    for number in pocket_numbers:
        candidate_atoms = by_residue.get((number, ""), [])
        point = residue_point(candidate_atoms) if candidate_atoms else None
        candidate_name = candidate_atoms[0].resname if candidate_atoms else ""
        if point is not None and candidate_name == reference_names[number]:
            present.append(number)
            ca_point, sidechain_point = point
        else:
            if candidate_atoms and candidate_name != reference_names[number]:
                mismatched_pocket.append(number)
            else:
                missing.append(number)
            reference_point = residue_point(reference_by_residue[(number, "")])
            assert reference_point is not None
            ca_point, sidechain_point = reference_point
        ca_points.append(ca_point)
        sidechain_points.append(sidechain_point)
    pocket_fraction = len(present) / len(pocket_numbers)
    missing_anchors = sorted(anchor_numbers - set(present))

    groups = ligand_groups(transformed, qualifying_ligand_ids)
    ligand_records: list[tuple[float, tuple[str, int, str], list[Atom]]] = []
    for key, values in groups.items():
        coordinates = np.vstack([atom.coord for atom in values])
        ligand_records.append(
            (minimum_distance(coordinates, reference_ligand_coords), key, values)
        )
    ligand_records.sort(key=lambda value: (value[0], value[1]))
    closest_ligand = ligand_records[0] if ligand_records else None
    ligand_distance = closest_ligand[0] if closest_ligand else math.nan

    readiness = preparation_readiness_metrics(
        transformed,
        reference_atoms,
        pocket_numbers,
        set(
            preparation_readiness_gate.get(
                "polymer_backbone_atom_names", ["N", "CA", "C", "O"]
            )
            if preparation_readiness_gate
            else ["N", "CA", "C", "O"]
        ),
    )

    reasons: list[str] = []
    if len(reference_coords) < minimum_matched_ca:
        reasons.append("too_few_matched_ca")
    if aligned_rmsd > maximum_rmsd:
        reasons.append("aligned_global_ca_rmsd_above_limit")
    if determinant < 0.999999:
        reasons.append("rotation_is_not_proper")
    if pocket_fraction < minimum_pocket_fraction:
        reasons.append("pocket_residue_fraction_below_limit")
    if mismatched_pocket:
        reasons.append("pocket_residue_name_mismatch")
    if missing_anchors:
        reasons.append("missing_required_anchor_residue")
    if not math.isfinite(ligand_distance) or ligand_distance > ligand_cutoff:
        reasons.append("no_qualifying_ligand_in_reference_pocket")
    if preparation_readiness_gate:
        if float(readiness["chain_heavy_atoms_per_ca"]) < float(
            preparation_readiness_gate["minimum_chain_heavy_atoms_per_ca"]
        ):
            reasons.append("protein_atom_to_ca_ratio_below_limit")
        if float(readiness["pocket_heavy_atom_completeness_fraction"]) < float(
            preparation_readiness_gate[
                "minimum_pocket_heavy_atom_completeness_fraction"
            ]
        ):
            reasons.append("pocket_heavy_atom_completeness_below_limit")
        if (
            preparation_readiness_gate["exclude_polymer_like_hetero_residues"]
            and int(readiness["polymer_like_hetero_residue_count"]) > 0
        ):
            reasons.append("polymer_like_hetero_residue_in_chain")
        if (
            preparation_readiness_gate.get(
                "exclude_incomplete_standard_amino_acid_residues", False
            )
            and int(readiness["incomplete_standard_amino_acid_residue_count"]) > 0
        ):
            reasons.append("incomplete_standard_amino_acid_residue")

    if not reasons:
        write_aligned_pdb(aligned_path, transformed)
        vector, feature_names = pairwise_feature_vector(
            ca_points, sidechain_points
        )
    else:
        vector, feature_names = None, None
    selected_ligand = closest_ligand[1] if closest_ligand else ("", 0, "")
    row = {
        "conformer_id": conformer_id,
        "pdb_id": pdb_id,
        "chain": chain,
        "status": "coordinate_eligible" if not reasons else "coordinate_excluded",
        "exclusion_reasons": ";".join(reasons),
        "pdb_path": path.as_posix(),
        "pdb_sha256": file_sha256(path),
        "aligned_pdb_path": aligned_path.as_posix() if not reasons else "",
        "aligned_pdb_sha256": file_sha256(aligned_path) if not reasons else "",
        "matched_ca_count": len(reference_coords),
        "aligned_global_ca_rmsd_angstrom": aligned_rmsd,
        "rotation_determinant": determinant,
        "global_residue_name_mismatch_count": len(mismatches),
        "pocket_present_count": len(present),
        "pocket_residue_fraction": pocket_fraction,
        "missing_pocket_residues": ";".join(map(str, missing)),
        "mismatched_pocket_residues": ";".join(map(str, mismatched_pocket)),
        "missing_anchor_residues": ";".join(map(str, missing_anchors)),
        "selected_ligand_resname": selected_ligand[0],
        "selected_ligand_resseq": selected_ligand[1],
        "selected_ligand_icode": selected_ligand[2],
        "selected_ligand_heavy_atom_count": (
            len(closest_ligand[2]) if closest_ligand else 0
        ),
        "selected_ligand_min_distance_to_reference_ligand_angstrom": (
            ligand_distance
        ),
        "qualifying_ligand_ids": ";".join(sorted(qualifying_ligand_ids)),
        "imputed_pocket_feature_count": len(pocket_numbers) - len(present),
        **readiness,
    }
    return row, vector, feature_names


def run_selection(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    runtime = check_runtime(config)
    paths = checked_inputs(config)
    prereg_record = config["preregistration"]
    assert isinstance(prereg_record, dict)
    prereg_path = Path(str(prereg_record["path"]))
    if file_sha256(prereg_path) != str(prereg_record["sha256"]).upper():
        raise ValueError("preregistration SHA-256 differs")
    preregistration = read_json(prereg_path)
    validate_preregistration(preregistration)
    amendment_record = config.get("preregistration_amendment")
    amendment_path: Path | None = None
    if amendment_record is not None:
        assert isinstance(amendment_record, dict)
        amendment_path = Path(str(amendment_record["path"]))
        if not amendment_path.is_file():
            raise FileNotFoundError(amendment_path)
        if file_sha256(amendment_path) != str(amendment_record["sha256"]).upper():
            raise ValueError("preregistration amendment SHA-256 differs")
        amendment = read_json(amendment_path)
        if (
            amendment.get("original_preregistration_sha256")
            != str(prereg_record["sha256"]).upper()
            or amendment.get("data_boundary", {}).get("ligand_labels_read") != 0
            or amendment.get("data_boundary", {}).get("docking_scores_read") != 0
        ):
            raise ValueError("preregistration amendment is not admissible")
    discovery_summary = read_json(paths["discovery_summary"])
    if (
        discovery_summary.get("status") != "metadata_discovery_ok"
        or int(discovery_summary["data_boundary"]["ligand_labels_read"]) != 0
        or int(discovery_summary["data_boundary"]["docking_scores_read"]) != 0
    ):
        raise ValueError("metadata discovery is not admissible")

    outputs = config["outputs"]
    assert isinstance(outputs, dict)
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    file_output_keys = tuple(
        sorted(
            REQUIRED_OUTPUT_KEYS
            - {"raw_pdb_directory", "aligned_pdb_directory"}
        )
    )
    existing_outputs = [
        output_paths[key] for key in file_output_keys if output_paths[key].exists()
    ]
    if existing_outputs and not overwrite:
        raise FileExistsError("coordinate-selection outputs exist; review first")
    if overwrite:
        for path in existing_outputs:
            path.unlink()

    new_rows = read_csv(paths["eligible_new_candidates"])
    metadata_by_id = {
        row["pdb_id"]: row for row in read_csv(paths["candidate_metadata"])
    }
    existing_manifest = read_csv(paths["existing_aligned_manifest"])
    raw_directory = output_paths["raw_pdb_directory"]
    aligned_directory = output_paths["aligned_pdb_directory"]
    download = config["download"]
    assert isinstance(download, dict)
    raw_paths = {
        row["pdb_id"]: raw_directory / f"{row['pdb_id']}.pdb" for row in new_rows
    }
    download_arguments = {
        "template": str(download["url_template"]),
        "timeout_seconds": float(download["timeout_seconds"]),
        "maximum_retries": int(download["maximum_retries"]),
        "retry_backoff_seconds": float(download["retry_backoff_seconds"]),
    }
    with ThreadPoolExecutor(max_workers=int(download["workers"])) as executor:
        futures = {
            executor.submit(
                download_one, pdb_id, path, **download_arguments
            ): pdb_id
            for pdb_id, path in raw_paths.items()
        }
        download_errors: dict[str, str] = {}
        for future in as_completed(futures):
            pdb_id, _, error = future.result()
            if error:
                download_errors[pdb_id] = error

    coordinate = preregistration["coordinate_eligibility"]
    expansion = preregistration["pool_expansion"]
    gate = config["coordinate_gate"]
    readiness_gate = config.get("preparation_readiness_gate")
    assert isinstance(coordinate, dict)
    assert isinstance(expansion, dict)
    assert isinstance(gate, dict)
    pocket_numbers = [
        int(value) for value in coordinate["reference_pocket_residue_numbers"]
    ]
    anchor_numbers = {
        int(value) for value in coordinate["required_anchor_residue_numbers"]
    }
    reference_atoms = select_chain_atoms(parse_pdb(paths["reference_pdb"]), "A")
    reference_ligand = [
        atom.coord
        for atom in reference_atoms
        if atom.record == "HETATM" and atom.resname == "LGF" and is_heavy(atom)
    ]
    if not reference_ligand:
        raise ValueError("reference LGF ligand was not found")
    reference_ligand_coords = np.vstack(reference_ligand)

    audit_rows: list[dict[str, object]] = []
    vectors: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    existing_ids = [str(value) for value in expansion["existing_receptor_ids"]]
    for manifest in existing_manifest:
        conformer_id = manifest["conformer_id"]
        pdb_id = conformer_id.split("_")[1]
        metadata = metadata_by_id[pdb_id]
        source_path = Path(manifest["pdb_path"])
        aligned_path = aligned_directory / f"{conformer_id}_structural_gate.pdb"
        row, vector, names = audit_structure(
            conformer_id,
            pdb_id,
            source_path,
            manifest["chain"],
            set(metadata["qualifying_ligand_ids"].split(";")),
            reference_atoms,
            reference_ligand_coords,
            pocket_numbers,
            anchor_numbers,
            float(coordinate["minimum_pocket_residue_fraction"]),
            float(
                coordinate[
                    "require_drug_like_ligand_within_reference_pocket_angstrom"
                ]
            ),
            int(gate["minimum_matched_ca_count"]),
            float(gate["maximum_aligned_global_ca_rmsd_angstrom"]),
            aligned_path,
            readiness_gate if isinstance(readiness_gate, dict) else None,
        )
        row["source_role"] = "existing_seed"
        audit_rows.append(row)
        if vector is not None:
            vectors[conformer_id] = vector
            feature_names = names
    for metadata in new_rows:
        pdb_id = metadata["pdb_id"]
        conformer_id = f"MK14_{pdb_id}_aligned"
        if pdb_id in download_errors:
            audit_rows.append(
                {
                    "conformer_id": conformer_id,
                    "pdb_id": pdb_id,
                    "chain": metadata["selected_auth_chain"],
                    "status": "coordinate_excluded",
                    "exclusion_reasons": "coordinate_file_unavailable",
                    "download_error": download_errors[pdb_id],
                    "pdb_path": "",
                    "pdb_sha256": "",
                    "aligned_pdb_path": "",
                    "aligned_pdb_sha256": "",
                    "source_role": "new_rcsb_candidate",
                }
            )
            continue
        aligned_path = aligned_directory / f"{pdb_id}_{metadata['selected_auth_chain']}_to_2QD9_A.pdb"
        row, vector, names = audit_structure(
            conformer_id,
            pdb_id,
            raw_paths[pdb_id],
            metadata["selected_auth_chain"],
            set(metadata["qualifying_ligand_ids"].split(";")),
            reference_atoms,
            reference_ligand_coords,
            pocket_numbers,
            anchor_numbers,
            float(coordinate["minimum_pocket_residue_fraction"]),
            float(
                coordinate[
                    "require_drug_like_ligand_within_reference_pocket_angstrom"
                ]
            ),
            int(gate["minimum_matched_ca_count"]),
            float(gate["maximum_aligned_global_ca_rmsd_angstrom"]),
            aligned_path,
            readiness_gate if isinstance(readiness_gate, dict) else None,
        )
        row["source_role"] = "new_rcsb_candidate"
        audit_rows.append(row)
        if vector is not None:
            vectors[conformer_id] = vector
            feature_names = names
    audit_rows.sort(key=lambda value: str(value["conformer_id"]))
    if any(
        row["status"] != "coordinate_eligible"
        for row in audit_rows
        if row["conformer_id"] in existing_ids
    ):
        raise ValueError("an existing seed receptor failed coordinate eligibility")
    eligible_rows = [
        row for row in audit_rows if row["status"] == "coordinate_eligible"
    ]
    eligible_by_id = {str(row["conformer_id"]): row for row in eligible_rows}
    new_eligible_ids = sorted(set(eligible_by_id) - set(existing_ids))
    if len(new_eligible_ids) < int(expansion["minimum_eligible_new_candidate_count"]):
        raise ValueError("too few new structures passed coordinate eligibility")
    assert feature_names is not None
    ordered_ids = sorted(vectors)
    matrix = np.vstack([vectors[value] for value in ordered_ids])
    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0)
    keep = standard_deviations >= float(
        gate["minimum_variable_feature_sd_angstrom"]
    )
    if int(keep.sum()) < 3:
        raise ValueError("too few variable structural features")
    standardized = (matrix[:, keep] - means[keep]) / standard_deviations[keep]
    standardized /= math.sqrt(int(keep.sum()))
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
    selected_new = maxmin_select(
        ordered_ids,
        existing_ids,
        distance_by_pair,
        int(expansion["new_receptor_count"]),
    )
    selection_rows: list[dict[str, object]] = []
    for rank, conformer_id in enumerate(existing_ids, start=1):
        selection_rows.append(
            {
                "pool_role": "existing_seed",
                "selection_rank": rank,
                **eligible_by_id[conformer_id],
                "minimum_standardized_distance_to_selected_pool": "",
            }
        )
    for selected in selected_new:
        conformer_id = str(selected["conformer_id"])
        selection_rows.append(
            {
                "pool_role": "new_maxmin_addition",
                **selected,
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
    pool_rows = [
        {
            "conformer_id": row["conformer_id"],
            "pdb_id": row["pdb_id"],
            "chain": row["chain"],
            "pdb_path": row["pdb_path"],
            "pdb_sha256": row["pdb_sha256"],
            "aligned_pdb_path": row["aligned_pdb_path"],
            "aligned_pdb_sha256": row["aligned_pdb_sha256"],
            "selected_ligand_resname": row["selected_ligand_resname"],
            "selected_ligand_resseq": row["selected_ligand_resseq"],
            "selected_ligand_icode": row["selected_ligand_icode"],
            "source_role": row["source_role"],
        }
        for row in eligible_rows
    ]
    write_csv(output_paths["coordinate_audit_csv"], audit_rows)
    write_csv(output_paths["eligible_pool_manifest_csv"], pool_rows)
    write_csv(output_paths["feature_matrix_csv"], feature_rows)
    write_csv(output_paths["pairwise_distances_csv"], distance_rows)
    write_csv(output_paths["selected_expansion_manifest_csv"], selection_rows)

    reason_counts = Counter(
        reason
        for row in audit_rows
        for reason in str(row["exclusion_reasons"]).split(";")
        if reason
    )
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "expanded8_structural_selection_ok",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "preregistration": {
            "path": prereg_path.as_posix(),
            "sha256": file_sha256(prereg_path),
        },
        "preregistration_amendment": (
            {
                "path": amendment_path.as_posix(),
                "sha256": file_sha256(amendment_path),
            }
            if amendment_path is not None
            else None
        ),
        "preparation_readiness_gate": readiness_gate,
        "runtime": runtime,
        "counts": {
            "existing_seed_count": len(existing_ids),
            "new_metadata_candidate_count": len(new_rows),
            "new_coordinate_eligible_count": len(new_eligible_ids),
            "new_coordinate_excluded_count": len(new_rows) - len(new_eligible_ids),
            "eligible_pool_count": len(eligible_rows),
            "selected_new_count": len(selected_new),
            "final_selected_pool_count": len(selection_rows),
            "raw_feature_count": matrix.shape[1],
            "variable_feature_count": int(keep.sum()),
            "pairwise_distance_count": len(distance_rows),
        },
        "coordinate_exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "selected_new_receptors": selected_new,
        "selected_final_receptor_ids": [
            str(value["conformer_id"]) for value in selection_rows
        ],
        "data_boundary": {
            "ligand_labels_read": 0,
            "docking_scores_read": 0,
            "previous_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            key: {"path": output_paths[key].as_posix(), "sha256": file_sha256(output_paths[key])}
            for key in file_output_keys
            if key != "summary_json"
        },
        "next_gate": "prepare the four new receptors and redock each selected co-crystal ligand before any benchmark docking",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(output_paths["summary_json"], summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_selection(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
