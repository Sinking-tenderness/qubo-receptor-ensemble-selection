"""Build a label-independent expanded CDK2 receptor structure pool."""

from __future__ import annotations

import argparse
import csv
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import pdist, squareform

try:
    from .align_receptor_structure import PDBAtom, file_sha256, parse_pdb
except ImportError:
    from align_receptor_structure import PDBAtom, file_sha256, parse_pdb


BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "OXT"}
REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "inputs",
    "input_sha256",
    "initial_candidates",
    "reference_conformer_id",
    "pocket_residue_numbers",
    "eligibility",
    "features",
    "cluster_counts",
    "expected",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_INPUT_KEYS = {"md_alignment_manifest", "md_preparation_manifest"}
REQUIRED_ELIGIBILITY_KEYS = {
    "require_all_reference_pocket_residues",
    "require_reference_residue_name_match",
    "require_receptor_pdbqt",
    "require_zero_pdbqt_hetatm",
    "expected_autodock_atom_types",
}
REQUIRED_FEATURE_KEYS = {
    "include_ca_pairwise_distances",
    "include_sidechain_heavy_centroid_pairwise_distances",
    "minimum_feature_sd_angstrom",
}
REQUIRED_EXPECTED_KEYS = {
    "initial_candidate_count",
    "md_candidate_count",
    "total_candidate_count",
    "eligible_candidate_count",
    "excluded_candidate_ids",
    "non_md_e32_candidate_count",
}
OUTPUT_KEYS = (
    "candidate_audit_csv",
    "eligible_pool_manifest_csv",
    "non_md_e32_receptor_manifest_csv",
    "feature_matrix_csv",
    "pairwise_distances_csv",
    "cluster_assignments_csv",
    "summary_json",
)
CANDIDATE_KEYS = {
    "conformer_id",
    "source_type",
    "pdb_path",
    "pdb_sha256",
    "chain",
    "selected_altloc",
    "source_identifier",
    "receptor_pdbqt_path",
    "receptor_pdbqt_sha256",
}
SHA256_PATTERN = re.compile(r"^[0-9A-Fa-f]{64}$")


def portable_path(value: str) -> Path:
    """Interpret repository-relative manifest paths on Windows or POSIX."""
    return Path(value.replace("\\", "/"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_sha256(value: object, label: str) -> None:
    if not SHA256_PATTERN.fullmatch(str(value)):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("expanded structural pool configuration must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"expanded structural pool configuration is missing keys: {', '.join(missing)}")

    inputs = config["inputs"]
    hashes = config["input_sha256"]
    eligibility = config["eligibility"]
    features = config["features"]
    expected = config["expected"]
    outputs = config["outputs"]
    candidates = config["initial_candidates"]
    if not isinstance(inputs, dict) or not REQUIRED_INPUT_KEYS.issubset(inputs):
        raise ValueError("inputs is missing one or more MD manifest paths")
    if not isinstance(hashes, dict) or not REQUIRED_INPUT_KEYS.issubset(hashes):
        raise ValueError("input_sha256 is missing one or more MD manifest hashes")
    for key in REQUIRED_INPUT_KEYS:
        validate_sha256(hashes[key], f"input_sha256.{key}")
    if not isinstance(eligibility, dict) or not REQUIRED_ELIGIBILITY_KEYS.issubset(eligibility):
        raise ValueError("eligibility is missing one or more required rules")
    for key in (
        "require_all_reference_pocket_residues",
        "require_reference_residue_name_match",
        "require_receptor_pdbqt",
        "require_zero_pdbqt_hetatm",
    ):
        if not isinstance(eligibility[key], bool):
            raise ValueError(f"eligibility.{key} must be a JSON boolean")
    atom_types = eligibility["expected_autodock_atom_types"]
    if (
        not isinstance(atom_types, list)
        or not atom_types
        or any(not isinstance(value, str) or not value for value in atom_types)
        or len(set(atom_types)) != len(atom_types)
    ):
        raise ValueError("expected_autodock_atom_types must contain unique non-empty strings")
    if not isinstance(features, dict) or not REQUIRED_FEATURE_KEYS.issubset(features):
        raise ValueError("features is missing one or more required settings")
    feature_flags = (
        features["include_ca_pairwise_distances"],
        features["include_sidechain_heavy_centroid_pairwise_distances"],
    )
    if any(not isinstance(value, bool) for value in feature_flags) or not any(feature_flags):
        raise ValueError("at least one pocket feature flag must be true and both must be booleans")
    if float(features["minimum_feature_sd_angstrom"]) < 0.0:
        raise ValueError("minimum_feature_sd_angstrom must be non-negative")
    pocket = config["pocket_residue_numbers"]
    if (
        not isinstance(pocket, list)
        or len(pocket) < 3
        or any(not isinstance(value, int) or value <= 0 for value in pocket)
        or len(set(pocket)) != len(pocket)
    ):
        raise ValueError("pocket_residue_numbers must contain at least three unique positive integers")
    clusters = config["cluster_counts"]
    if (
        not isinstance(clusters, list)
        or not clusters
        or any(not isinstance(value, int) or value <= 0 for value in clusters)
        or len(set(clusters)) != len(clusters)
    ):
        raise ValueError("cluster_counts must contain unique positive integers")
    if not isinstance(expected, dict) or not REQUIRED_EXPECTED_KEYS.issubset(expected):
        raise ValueError("expected is missing one or more preregistered counts")
    numeric_expected = [key for key in REQUIRED_EXPECTED_KEYS if key != "excluded_candidate_ids"]
    if any(int(expected[key]) < 0 for key in numeric_expected):
        raise ValueError("expected candidate counts must be non-negative")
    if max(clusters) > int(expected["eligible_candidate_count"]):
        raise ValueError("cluster_counts cannot exceed expected eligible candidate count")
    excluded_ids = expected["excluded_candidate_ids"]
    if not isinstance(excluded_ids, list) or len(set(excluded_ids)) != len(excluded_ids):
        raise ValueError("excluded_candidate_ids must be a unique JSON list")
    if not isinstance(outputs, dict) or not set(OUTPUT_KEYS).issubset(outputs):
        raise ValueError("outputs is missing one or more structural pool paths")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("initial_candidates must be a non-empty JSON list")
    conformer_ids: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or not CANDIDATE_KEYS.issubset(candidate):
            raise ValueError(f"initial candidate {index} is missing one or more required fields")
        if len(str(candidate["chain"])) != 1:
            raise ValueError(f"initial candidate {index} chain must be one character")
        if len(str(candidate["selected_altloc"])) != 1:
            raise ValueError(f"initial candidate {index} selected_altloc must be one character")
        validate_sha256(candidate["pdb_sha256"], f"initial candidate {index} pdb_sha256")
        validate_sha256(
            candidate["receptor_pdbqt_sha256"],
            f"initial candidate {index} receptor_pdbqt_sha256",
        )
        conformer_ids.append(str(candidate["conformer_id"]))
    if len(conformer_ids) != len(set(conformer_ids)):
        raise ValueError("initial_candidates contains duplicate conformer IDs")
    if str(config["reference_conformer_id"]) not in conformer_ids:
        raise ValueError("reference_conformer_id is not present in initial_candidates")
    if len(candidates) != int(expected["initial_candidate_count"]):
        raise ValueError("initial candidate count differs from expected.initial_candidate_count")
    if int(expected["total_candidate_count"]) != (
        int(expected["initial_candidate_count"]) + int(expected["md_candidate_count"])
    ):
        raise ValueError("expected total candidate count is internally inconsistent")
    return config


def verify_file(path: Path, expected_hash: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != expected_hash.upper():
        raise ValueError(f"{label} SHA-256 differs: expected {expected_hash.upper()}, got {actual}")
    return actual


def is_hydrogen(atom: PDBAtom, original_line: str) -> bool:
    element = original_line[76:78].strip().upper() if len(original_line) >= 78 else ""
    name = atom.atom_name.upper().lstrip("0123456789")
    return element == "H" or (not element and name.startswith("H"))


def audit_pocket(
    path: Path,
    chain: str,
    pocket_numbers: list[int],
    reference_residue_names: dict[int, str] | None = None,
    selected_altloc: str = "A",
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    lines, atoms = parse_pdb(path)
    selected: dict[tuple[int, str], PDBAtom] = {}
    for atom in atoms:
        if (
            atom.record != "ATOM"
            or atom.chain != chain
            or atom.icode
            or atom.resseq not in pocket_numbers
            or atom.altloc not in {"", selected_altloc}
        ):
            continue
        key = (atom.resseq, atom.atom_name)
        previous = selected.get(key)
        if previous is None or (previous.altloc == selected_altloc and atom.altloc == ""):
            selected[key] = atom

    residue_names: dict[int, str] = {}
    missing: list[int] = []
    ca_issues: list[int] = []
    mismatches: list[str] = []
    ca_coords: list[np.ndarray] = []
    sidechain_coords: list[np.ndarray] = []
    sidechain_counts: list[int] = []
    for number in pocket_numbers:
        residue_atoms = [atom for (residue, _), atom in selected.items() if residue == number]
        if not residue_atoms:
            missing.append(number)
            continue
        names = sorted({atom.resname for atom in residue_atoms})
        if len(names) != 1:
            ca_issues.append(number)
            continue
        residue_name = names[0]
        residue_names[number] = residue_name
        if reference_residue_names is not None and reference_residue_names[number] != residue_name:
            mismatches.append(f"{number}:{reference_residue_names[number]}!={residue_name}")
        ca_atoms = [atom for atom in residue_atoms if atom.atom_name == "CA"]
        if len(ca_atoms) != 1:
            ca_issues.append(number)
            continue
        ca = ca_atoms[0]
        sidechain_heavy = [
            atom
            for atom in residue_atoms
            if atom.atom_name not in BACKBONE_ATOM_NAMES
            and not is_hydrogen(atom, lines[atom.line_index])
        ]
        ca_coords.append(ca.coord)
        sidechain_coords.append(
            np.vstack([atom.coord for atom in sidechain_heavy]).mean(axis=0)
            if sidechain_heavy
            else ca.coord
        )
        sidechain_counts.append(len(sidechain_heavy))

    complete = not missing and not ca_issues and len(ca_coords) == len(pocket_numbers)
    audit = {
        "pocket_residue_count": len(residue_names),
        "missing_pocket_residues": missing,
        "ca_issue_residues": sorted(set(ca_issues)),
        "residue_name_mismatches": mismatches,
        "residue_names": residue_names,
        "sidechain_heavy_atom_counts": sidechain_counts if complete else [],
    }
    geometry = None
    if complete:
        geometry = {
            "ca": np.vstack(ca_coords),
            "sidechain_centroid": np.vstack(sidechain_coords),
        }
    return audit, geometry


def audit_pdbqt(path: Path) -> dict[str, object]:
    atom_count = 0
    atom_record_count = 0
    hetatm_count = 0
    charges: list[float] = []
    atom_types: set[str] = set()
    hydrogen_like_count = 0
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        record = line[0:6].strip()
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"invalid PDBQT coordinate record at {path}:{line_number}")
        try:
            charge = float(fields[-2])
        except ValueError as exc:
            raise ValueError(f"invalid PDBQT charge at {path}:{line_number}") from exc
        atom_type = fields[-1]
        atom_count += 1
        atom_record_count += record == "ATOM"
        hetatm_count += record == "HETATM"
        charges.append(charge)
        atom_types.add(atom_type)
        hydrogen_like_count += atom_type.upper().startswith("H")
    if not atom_count:
        raise ValueError(f"PDBQT contains no ATOM or HETATM records: {path}")
    return {
        "atom_count": atom_count,
        "atom_record_count": atom_record_count,
        "hetatm_count": hetatm_count,
        "hydrogen_like_atom_count": hydrogen_like_count,
        "charge_min": min(charges),
        "charge_max": max(charges),
        "atom_types": sorted(atom_types),
    }


def detect_single_chain(path: Path) -> str:
    _, atoms = parse_pdb(path)
    chains = sorted({atom.chain for atom in atoms if atom.record == "ATOM"})
    if len(chains) != 1 or len(chains[0]) != 1:
        raise ValueError(f"expected one non-empty ATOM chain in {path}, got {chains}")
    return chains[0]


def load_md_candidates(
    alignment_path: Path,
    preparation_path: Path,
    expected_count: int,
) -> list[dict[str, object]]:
    alignment_rows = read_csv(alignment_path)
    preparation_rows = read_csv(preparation_path)
    alignment_by_id = {row["conformer_id"]: row for row in alignment_rows}
    preparation_by_id = {row["conformer_id"]: row for row in preparation_rows}
    if len(alignment_by_id) != len(alignment_rows) or len(preparation_by_id) != len(preparation_rows):
        raise ValueError("MD alignment or preparation manifest contains duplicate conformer IDs")
    if set(alignment_by_id) != set(preparation_by_id):
        raise ValueError("MD alignment and preparation manifests contain different conformer IDs")
    if len(alignment_by_id) != expected_count:
        raise ValueError(f"expected {expected_count} MD candidates, got {len(alignment_by_id)}")

    candidates: list[dict[str, object]] = []
    for conformer_id in sorted(alignment_by_id):
        alignment = alignment_by_id[conformer_id]
        preparation = preparation_by_id[conformer_id]
        if alignment.get("alignment_status") != "ok":
            raise ValueError(f"MD alignment did not pass for {conformer_id}")
        if preparation.get("preparation_status") != "ok":
            raise ValueError(f"MD receptor preparation did not pass for {conformer_id}")
        if alignment["aligned_heavy_pdb_sha256"].upper() != preparation["aligned_heavy_pdb_sha256"].upper():
            raise ValueError(f"aligned-heavy hash differs between MD manifests for {conformer_id}")
        pdb_path = portable_path(alignment["aligned_heavy_pdb_path"])
        receptor_path = portable_path(preparation["receptor_pdbqt_path"])
        candidates.append({
            "conformer_id": conformer_id,
            "source_type": "md_cluster_medoid",
            "source_identifier": conformer_id,
            "pdb_path": pdb_path.as_posix(),
            "pdb_sha256": alignment["aligned_heavy_pdb_sha256"].upper(),
            "chain": detect_single_chain(pdb_path),
            "selected_altloc": "A",
            "receptor_pdbqt_path": receptor_path.as_posix(),
            "receptor_pdbqt_sha256": preparation["receptor_pdbqt_sha256"].upper(),
            "md_cluster_id": alignment.get("cluster_id", ""),
            "md_frame_index": alignment.get("frame_index", ""),
            "md_time_ps": alignment.get("time_ps", ""),
            "temporal_support_role": alignment.get("temporal_support_role", ""),
        })
    return candidates


def point_distance_features(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        raise ValueError("point coordinates must have shape (points, 3)")
    first, second = np.triu_indices(points.shape[0], k=1)
    return np.linalg.norm(points[first] - points[second], axis=1)


def feature_names(prefix: str, residue_numbers: list[int]) -> list[str]:
    return [f"{prefix}_{first}_{second}" for first, second in combinations(residue_numbers, 2)]


def standardize_features(
    values: np.ndarray, minimum_sd: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("feature matrix must have at least two rows and one column")
    if not np.all(np.isfinite(values)):
        raise ValueError("feature matrix contains non-finite values")
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0, ddof=0)
    kept = (standard_deviations >= minimum_sd) & (standard_deviations > 0.0)
    if not np.any(kept):
        raise ValueError("all pocket geometry features were removed by the SD threshold")
    scaled = (values[:, kept] - means[kept]) / standard_deviations[kept]
    return scaled, kept, means, standard_deviations


def kabsch_rmsd(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise ValueError("Kabsch RMSD requires matching coordinate arrays with shape (points, 3)")
    first_centered = first - first.mean(axis=0)
    second_centered = second - second.mean(axis=0)
    left, _, right_transposed = np.linalg.svd(first_centered.T @ second_centered)
    rotation = left @ right_transposed
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transposed
    aligned = first_centered @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - second_centered) ** 2, axis=1))))


def medoid_indices_by_id(
    labels: np.ndarray,
    distances: np.ndarray,
    conformer_ids: list[str],
) -> dict[int, int]:
    if labels.ndim != 1 or distances.shape != (len(labels), len(labels)):
        raise ValueError("cluster labels and distance matrix dimensions differ")
    medoids: dict[int, int] = {}
    for label in sorted(set(int(value) for value in labels)):
        indices = np.flatnonzero(labels == label)
        totals = distances[np.ix_(indices, indices)].sum(axis=1)
        medoids[label] = min(
            (int(index) for index in indices),
            key=lambda index: (float(totals[np.flatnonzero(indices == index)[0]]), conformer_ids[index]),
        )
    return medoids


def deterministic_cluster_labels(
    scaled_features: np.ndarray,
    distances: np.ndarray,
    conformer_ids: list[str],
    cluster_count: int,
    linkage_tree: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    if not 1 <= cluster_count <= len(conformer_ids):
        raise ValueError("cluster_count must be between one and the number of conformers")
    if cluster_count == 1:
        preliminary = np.zeros(len(conformer_ids), dtype=int)
    else:
        tree = linkage_tree if linkage_tree is not None else linkage(scaled_features, method="ward")
        preliminary = cut_tree(tree, n_clusters=[cluster_count]).reshape(-1).astype(int)
    preliminary_medoids = medoid_indices_by_id(preliminary, distances, conformer_ids)
    old_labels = sorted(preliminary_medoids, key=lambda label: conformer_ids[preliminary_medoids[label]])
    mapping = {old: new for new, old in enumerate(old_labels)}
    labels = np.array([mapping[int(label)] for label in preliminary], dtype=int)
    return labels, medoid_indices_by_id(labels, distances, conformer_ids)


def silhouette_from_distances(labels: np.ndarray, distances: np.ndarray) -> float | None:
    unique = sorted(set(int(value) for value in labels))
    if len(unique) < 2:
        return None
    values = np.zeros(len(labels), dtype=float)
    for index, label in enumerate(labels):
        same = np.flatnonzero(labels == label)
        same = same[same != index]
        if not len(same):
            continue
        within = float(distances[index, same].mean())
        nearest = min(
            float(distances[index, labels == other].mean())
            for other in unique
            if other != int(label)
        )
        denominator = max(within, nearest)
        values[index] = 0.0 if denominator == 0.0 else (nearest - within) / denominator
    return float(values.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    input_hashes = config["input_sha256"]
    outputs = config["outputs"]
    expected = config["expected"]
    eligibility = config["eligibility"]
    feature_config = config["features"]
    assert isinstance(inputs, dict)
    assert isinstance(input_hashes, dict)
    assert isinstance(outputs, dict)
    assert isinstance(expected, dict)
    assert isinstance(eligibility, dict)
    assert isinstance(feature_config, dict)

    output_paths = {key: portable_path(str(outputs[key])) for key in OUTPUT_KEYS}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("expanded structural pool outputs exist; use --overwrite after review")
    if args.overwrite:
        for path in existing:
            path.unlink()

    input_paths = {key: portable_path(str(inputs[key])) for key in REQUIRED_INPUT_KEYS}
    for key, path in input_paths.items():
        verify_file(path, str(input_hashes[key]), key)
    md_candidates = load_md_candidates(
        input_paths["md_alignment_manifest"],
        input_paths["md_preparation_manifest"],
        int(expected["md_candidate_count"]),
    )
    initial_candidates = [dict(row) for row in config["initial_candidates"]]
    candidates = sorted([*initial_candidates, *md_candidates], key=lambda row: str(row["conformer_id"]))
    if len(candidates) != int(expected["total_candidate_count"]):
        raise ValueError("observed total candidate count differs from preregistered expectation")
    candidate_ids = [str(row["conformer_id"]) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("expanded candidate pool contains duplicate conformer IDs")

    reference_id = str(config["reference_conformer_id"])
    reference = next(row for row in candidates if row["conformer_id"] == reference_id)
    reference_path = portable_path(str(reference["pdb_path"]))
    verify_file(reference_path, str(reference["pdb_sha256"]), f"{reference_id} PDB")
    pocket_numbers = [int(value) for value in config["pocket_residue_numbers"]]
    reference_audit, reference_geometry = audit_pocket(
        reference_path,
        str(reference["chain"]),
        pocket_numbers,
        selected_altloc=str(reference["selected_altloc"]),
    )
    if reference_geometry is None or reference_audit["missing_pocket_residues"]:
        raise ValueError("reference structure does not contain the complete configured pocket")
    reference_residue_names = {
        int(number): str(name)
        for number, name in dict(reference_audit["residue_names"]).items()
    }

    expected_atom_types = sorted(str(value) for value in eligibility["expected_autodock_atom_types"])
    audit_rows: list[dict[str, object]] = []
    geometry_by_id: dict[str, dict[str, np.ndarray]] = {}
    eligible_candidates: list[dict[str, object]] = []
    for candidate in candidates:
        conformer_id = str(candidate["conformer_id"])
        pdb_path = portable_path(str(candidate["pdb_path"]))
        pdbqt_path = portable_path(str(candidate["receptor_pdbqt_path"]))
        pdb_hash = verify_file(pdb_path, str(candidate["pdb_sha256"]), f"{conformer_id} PDB")
        pdbqt_hash = verify_file(
            pdbqt_path,
            str(candidate["receptor_pdbqt_sha256"]),
            f"{conformer_id} PDBQT",
        )
        pocket_audit, geometry = audit_pocket(
            pdb_path,
            str(candidate["chain"]),
            pocket_numbers,
            reference_residue_names,
            selected_altloc=str(candidate["selected_altloc"]),
        )
        pdbqt_audit = audit_pdbqt(pdbqt_path)
        reasons: list[str] = []
        if eligibility["require_all_reference_pocket_residues"]:
            if pocket_audit["missing_pocket_residues"]:
                reasons.append("missing_reference_pocket_residues")
            if pocket_audit["ca_issue_residues"]:
                reasons.append("invalid_or_missing_pocket_ca")
        if eligibility["require_reference_residue_name_match"] and pocket_audit["residue_name_mismatches"]:
            reasons.append("reference_pocket_residue_name_mismatch")
        if eligibility["require_zero_pdbqt_hetatm"] and int(pdbqt_audit["hetatm_count"]) != 0:
            reasons.append("pdbqt_contains_hetatm")
        if list(pdbqt_audit["atom_types"]) != expected_atom_types:
            reasons.append("unexpected_pdbqt_atom_types")
        eligible = not reasons and geometry is not None
        if eligible:
            geometry_by_id[conformer_id] = geometry
            eligible_candidates.append({
                "conformer_id": conformer_id,
                "source_type": candidate["source_type"],
                "source_identifier": candidate["source_identifier"],
                "chain": candidate["chain"],
                "selected_altloc": candidate["selected_altloc"],
                "preparation_status": "ok",
                "pdb_path": pdb_path.as_posix(),
                "pdb_sha256": pdb_hash,
                "receptor_pdbqt_path": pdbqt_path.as_posix(),
                "receptor_pdbqt_sha256": pdbqt_hash,
                "pocket_residue_count": pocket_audit["pocket_residue_count"],
                "pdbqt_atom_count": pdbqt_audit["atom_count"],
                "pdbqt_hetatm_count": pdbqt_audit["hetatm_count"],
                "pdbqt_hydrogen_like_atom_count": pdbqt_audit["hydrogen_like_atom_count"],
                "pdbqt_charge_min": round(float(pdbqt_audit["charge_min"]), 6),
                "pdbqt_charge_max": round(float(pdbqt_audit["charge_max"]), 6),
                "pdbqt_autodock_atom_types": ";".join(pdbqt_audit["atom_types"]),
            })
        audit_rows.append({
            "conformer_id": conformer_id,
            "source_type": candidate["source_type"],
            "source_identifier": candidate["source_identifier"],
            "chain": candidate["chain"],
            "selected_altloc": candidate["selected_altloc"],
            "pdb_path": pdb_path.as_posix(),
            "pdb_sha256": pdb_hash,
            "receptor_pdbqt_path": pdbqt_path.as_posix(),
            "receptor_pdbqt_sha256": pdbqt_hash,
            "pocket_residue_count": pocket_audit["pocket_residue_count"],
            "missing_pocket_residues": ";".join(map(str, pocket_audit["missing_pocket_residues"])),
            "ca_issue_residues": ";".join(map(str, pocket_audit["ca_issue_residues"])),
            "residue_name_mismatches": ";".join(pocket_audit["residue_name_mismatches"]),
            "pdbqt_atom_count": pdbqt_audit["atom_count"],
            "pdbqt_atom_record_count": pdbqt_audit["atom_record_count"],
            "pdbqt_hetatm_count": pdbqt_audit["hetatm_count"],
            "pdbqt_hydrogen_like_atom_count": pdbqt_audit["hydrogen_like_atom_count"],
            "pdbqt_charge_min": round(float(pdbqt_audit["charge_min"]), 6),
            "pdbqt_charge_max": round(float(pdbqt_audit["charge_max"]), 6),
            "pdbqt_autodock_atom_types": ";".join(pdbqt_audit["atom_types"]),
            "eligibility_status": "eligible" if eligible else "excluded",
            "exclusion_reasons": ";".join(reasons),
        })

    observed_excluded = sorted(
        str(row["conformer_id"]) for row in audit_rows if row["eligibility_status"] == "excluded"
    )
    if len(eligible_candidates) != int(expected["eligible_candidate_count"]):
        exclusion_details = {
            str(row["conformer_id"]): {
                "reasons": row["exclusion_reasons"],
                "missing_pocket_residues": row["missing_pocket_residues"],
                "ca_issue_residues": row["ca_issue_residues"],
                "residue_name_mismatches": row["residue_name_mismatches"],
                "pdbqt_hetatm_count": row["pdbqt_hetatm_count"],
                "pdbqt_autodock_atom_types": row["pdbqt_autodock_atom_types"],
            }
            for row in audit_rows
            if row["eligibility_status"] == "excluded"
        }
        raise ValueError(
            f"eligible candidate count differs: expected {expected['eligible_candidate_count']}, "
            f"got {len(eligible_candidates)}; exclusions={json.dumps(exclusion_details, sort_keys=True)}"
        )
    if observed_excluded != sorted(str(value) for value in expected["excluded_candidate_ids"]):
        raise ValueError(
            f"excluded candidate IDs differ: expected {expected['excluded_candidate_ids']}, "
            f"got {observed_excluded}"
        )
    eligible_candidates.sort(key=lambda row: str(row["conformer_id"]))
    non_md_e32_candidates = [
        row for row in eligible_candidates if row["source_type"] != "md_cluster_medoid"
    ]
    if len(non_md_e32_candidates) != int(expected["non_md_e32_candidate_count"]):
        raise ValueError("non-MD e32 candidate count differs from preregistered expectation")

    conformer_ids = [str(row["conformer_id"]) for row in eligible_candidates]
    source_types = [str(row["source_type"]) for row in eligible_candidates]
    names: list[str] = []
    feature_blocks: list[np.ndarray] = []
    if feature_config["include_ca_pairwise_distances"]:
        names.extend(feature_names("ca_distance", pocket_numbers))
        feature_blocks.append(
            np.vstack([point_distance_features(geometry_by_id[value]["ca"]) for value in conformer_ids])
        )
    if feature_config["include_sidechain_heavy_centroid_pairwise_distances"]:
        names.extend(feature_names("sidechain_centroid_distance", pocket_numbers))
        feature_blocks.append(
            np.vstack([
                point_distance_features(geometry_by_id[value]["sidechain_centroid"])
                for value in conformer_ids
            ])
        )
    raw_features = np.hstack(feature_blocks)
    scaled_features, kept, feature_means, feature_sds = standardize_features(
        raw_features,
        float(feature_config["minimum_feature_sd_angstrom"]),
    )
    kept_names = [name for name, keep in zip(names, kept, strict=True) if keep]
    dropped_names = [name for name, keep in zip(names, kept, strict=True) if not keep]
    structural_distances = squareform(pdist(scaled_features, metric="euclidean"))

    feature_rows: list[dict[str, object]] = []
    for row_index, conformer_id in enumerate(conformer_ids):
        row: dict[str, object] = {
            "conformer_id": conformer_id,
            "source_type": source_types[row_index],
        }
        for feature_index, name in enumerate(names):
            row[f"raw__{name}"] = round(float(raw_features[row_index, feature_index]), 6)
            if kept[feature_index]:
                scaled_index = int(np.flatnonzero(np.flatnonzero(kept) == feature_index)[0])
                row[f"z__{name}"] = round(float(scaled_features[row_index, scaled_index]), 6)
        feature_rows.append(row)

    pairwise_rows: list[dict[str, object]] = []
    ca_rmsd_values: list[float] = []
    standardized_distance_values: list[float] = []
    for first, second in combinations(range(len(conformer_ids)), 2):
        ca_rmsd = kabsch_rmsd(
            geometry_by_id[conformer_ids[first]]["ca"],
            geometry_by_id[conformer_ids[second]]["ca"],
        )
        standardized_distance = float(structural_distances[first, second])
        ca_rmsd_values.append(ca_rmsd)
        standardized_distance_values.append(standardized_distance)
        pairwise_rows.append({
            "conformer_id_1": conformer_ids[first],
            "source_type_1": source_types[first],
            "conformer_id_2": conformer_ids[second],
            "source_type_2": source_types[second],
            "standardized_euclidean_distance": round(standardized_distance, 6),
            "pocket_ca_kabsch_rmsd_angstrom": round(ca_rmsd, 6),
        })

    linkage_tree = linkage(scaled_features, method="ward")
    cluster_rows: list[dict[str, object]] = []
    cluster_summaries: list[dict[str, object]] = []
    for cluster_count in [int(value) for value in config["cluster_counts"]]:
        labels, medoids = deterministic_cluster_labels(
            scaled_features,
            structural_distances,
            conformer_ids,
            cluster_count,
            linkage_tree,
        )
        if len(set(labels)) != cluster_count:
            raise RuntimeError(f"Ward clustering returned {len(set(labels))} clusters for k={cluster_count}")
        cluster_sizes = {
            label: int(np.sum(labels == label)) for label in sorted(set(int(value) for value in labels))
        }
        for index, conformer_id in enumerate(conformer_ids):
            label = int(labels[index])
            medoid_index = medoids[label]
            cluster_rows.append({
                "cluster_count": cluster_count,
                "cluster_id": label,
                "conformer_id": conformer_id,
                "source_type": source_types[index],
                "cluster_size": cluster_sizes[label],
                "is_cluster_medoid": index == medoid_index,
                "medoid_conformer_id": conformer_ids[medoid_index],
                "distance_to_medoid_standardized_euclidean": round(
                    float(structural_distances[index, medoid_index]), 6
                ),
            })
        silhouette = silhouette_from_distances(labels, structural_distances)
        cluster_summaries.append({
            "cluster_count": cluster_count,
            "cluster_sizes": [cluster_sizes[label] for label in sorted(cluster_sizes)],
            "medoid_conformer_ids": [conformer_ids[medoids[label]] for label in sorted(medoids)],
            "silhouette_standardized_euclidean": None if silhouette is None else round(silhouette, 6),
        })

    write_csv(output_paths["candidate_audit_csv"], audit_rows)
    write_csv(output_paths["eligible_pool_manifest_csv"], eligible_candidates)
    write_csv(output_paths["non_md_e32_receptor_manifest_csv"], non_md_e32_candidates)
    write_csv(output_paths["feature_matrix_csv"], feature_rows)
    write_csv(output_paths["pairwise_distances_csv"], pairwise_rows)
    write_csv(output_paths["cluster_assignments_csv"], cluster_rows)
    csv_output_keys = [key for key in OUTPUT_KEYS if key != "summary_json"]
    output_hashes = {key: file_sha256(output_paths[key]) for key in csv_output_keys}
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "operation": "label-independent receptor integrity gate, invariant pocket features, Ward clustering, and medoid baselines",
        "config_path": args.config.as_posix(),
        "config_sha256": file_sha256(args.config),
        "input_sha256": {key: file_sha256(path) for key, path in sorted(input_paths.items())},
        "candidate_counts": {
            "initial": len(initial_candidates),
            "md": len(md_candidates),
            "total": len(candidates),
            "eligible": len(eligible_candidates),
            "excluded": len(observed_excluded),
            "non_md_e32": len(non_md_e32_candidates),
        },
        "excluded_candidates": [
            {
                "conformer_id": row["conformer_id"],
                "reasons": str(row["exclusion_reasons"]).split(";"),
                "missing_pocket_residues": [
                    int(value) for value in str(row["missing_pocket_residues"]).split(";") if value
                ],
            }
            for row in audit_rows
            if row["eligibility_status"] == "excluded"
        ],
        "eligible_conformer_ids": conformer_ids,
        "non_md_e32_conformer_ids": [
            str(row["conformer_id"]) for row in non_md_e32_candidates
        ],
        "reference": {
            "conformer_id": reference_id,
            "pocket_residue_numbers": pocket_numbers,
            "pocket_residue_names": [reference_residue_names[number] for number in pocket_numbers],
            "sidechain_centroid_fallback_to_ca_residues": [
                number
                for number, count in zip(
                    pocket_numbers,
                    reference_audit["sidechain_heavy_atom_counts"],
                    strict=True,
                )
                if count == 0
            ],
        },
        "features": {
            "raw_feature_count": len(names),
            "retained_feature_count": len(kept_names),
            "dropped_feature_count": len(dropped_names),
            "minimum_sd_angstrom": float(feature_config["minimum_feature_sd_angstrom"]),
            "retained_feature_names": kept_names,
            "dropped_feature_names": dropped_names,
            "raw_feature_means_angstrom": [round(float(value), 8) for value in feature_means],
            "raw_feature_sds_angstrom": [round(float(value), 8) for value in feature_sds],
        },
        "pairwise_structural_diversity": {
            "pair_count": len(pairwise_rows),
            "pocket_ca_kabsch_rmsd_angstrom": {
                "minimum": round(float(min(ca_rmsd_values)), 6),
                "median": round(float(np.median(ca_rmsd_values)), 6),
                "maximum": round(float(max(ca_rmsd_values)), 6),
            },
            "standardized_euclidean_distance": {
                "minimum": round(float(min(standardized_distance_values)), 6),
                "median": round(float(np.median(standardized_distance_values)), 6),
                "maximum": round(float(max(standardized_distance_values)), 6),
            },
        },
        "clustering": {
            "method": "Ward agglomerative clustering on standardized invariant pocket-distance features",
            "scipy_version": scipy.__version__,
            "cluster_summaries": cluster_summaries,
        },
        "label_independence_audit": {
            "ligand_manifests_read": 0,
            "docking_score_files_read": 0,
            "activity_labels_read": 0,
            "selection_inputs": [
                "fixed receptor file hashes",
                "pocket residue completeness and residue identity",
                "prepared PDBQT integrity",
                "invariant pocket geometry",
            ],
        },
        "outputs": {key: output_paths[key].as_posix() for key in OUTPUT_KEYS},
        "output_sha256": output_hashes,
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output_paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
