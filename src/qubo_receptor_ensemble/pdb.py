"""PDB parsing, rigid alignment, and receptor auditing helpers.

Consolidated from ``scripts/align_receptor_structure.py`` and
``scripts/prepare_receptor.py``; behavior is identical to the originals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PDBAtom:
    line_index: int
    record: str
    atom_name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    coord: np.ndarray


def parse_pdb(path: Path) -> tuple[list[str], list[PDBAtom]]:
    """Parse an ASCII PDB file into raw lines and coordinate records."""
    if not path.is_file():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding="ascii").splitlines()
    if any(line.startswith("ANISOU") for line in lines):
        raise ValueError("ANISOU records are not supported by this rigid-coordinate writer")

    atoms: list[PDBAtom] = []
    for index, line in enumerate(lines):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if len(line) < 54:
            raise ValueError(f"invalid PDB coordinate line {index + 1} in {path}")
        atoms.append(
            PDBAtom(
                line_index=index,
                record=line[0:6].strip(),
                atom_name=line[12:16].strip(),
                altloc=line[16:17].strip(),
                resname=line[17:20].strip(),
                chain=line[21:22].strip(),
                resseq=int(line[22:26]),
                icode=line[26:27].strip(),
                coord=np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=float,
                ),
            )
        )
    if not atoms:
        raise ValueError(f"no ATOM or HETATM coordinates found in {path}")
    return lines, atoms


def collect_ca_atoms(atoms: list[PDBAtom], chain: str) -> dict[tuple[int, str], PDBAtom]:
    anchors: dict[tuple[int, str], PDBAtom] = {}
    for atom in atoms:
        if (
            atom.record == "ATOM"
            and atom.chain == chain
            and atom.atom_name == "CA"
            and atom.altloc in {"", "A"}
        ):
            anchors.setdefault((atom.resseq, atom.icode), atom)
    if not anchors:
        raise ValueError(f"no C-alpha atoms found for chain {chain!r}")
    return anchors


def match_ca_coordinates(
    reference_atoms: list[PDBAtom],
    mobile_atoms: list[PDBAtom],
    reference_chain: str,
    mobile_chain: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    reference = collect_ca_atoms(reference_atoms, reference_chain)
    mobile = collect_ca_atoms(mobile_atoms, mobile_chain)
    reference_entries = sorted(reference.items(), key=lambda item: item[0])
    mobile_entries = sorted(mobile.items(), key=lambda item: item[0])

    # Global alignment keeps coordinates paired when construct numbering has
    # offsets or insertions. Only equal residue names become Kabsch anchors.
    gap_penalty = -2
    match_score = 2
    mismatch_score = -1
    scores = np.zeros((len(reference_entries) + 1, len(mobile_entries) + 1), dtype=int)
    traceback = np.empty(scores.shape, dtype="U1")
    traceback[0, 0] = ""
    for i in range(1, len(reference_entries) + 1):
        scores[i, 0] = scores[i - 1, 0] + gap_penalty
        traceback[i, 0] = "U"
    for j in range(1, len(mobile_entries) + 1):
        scores[0, j] = scores[0, j - 1] + gap_penalty
        traceback[0, j] = "L"
    for i, (_, reference_atom) in enumerate(reference_entries, start=1):
        for j, (_, mobile_atom) in enumerate(mobile_entries, start=1):
            diagonal = scores[i - 1, j - 1] + (
                match_score if reference_atom.resname == mobile_atom.resname else mismatch_score
            )
            up = scores[i - 1, j] + gap_penalty
            left = scores[i, j - 1] + gap_penalty
            best = max(diagonal, up, left)
            scores[i, j] = best
            traceback[i, j] = "D" if diagonal == best else ("U" if up == best else "L")

    matched_reference: list[np.ndarray] = []
    matched_mobile: list[np.ndarray] = []
    residue_mismatches: list[str] = []
    i, j = len(reference_entries), len(mobile_entries)
    while i or j:
        direction = traceback[i, j]
        if direction == "D":
            reference_key, reference_atom = reference_entries[i - 1]
            mobile_key, mobile_atom = mobile_entries[j - 1]
            if reference_atom.resname == mobile_atom.resname:
                matched_reference.append(reference_atom.coord)
                matched_mobile.append(mobile_atom.coord)
            else:
                residue_mismatches.append(
                    f"{reference_key[0]}{reference_key[1]}/{mobile_key[0]}{mobile_key[1]}:"
                    f"{reference_atom.resname}!={mobile_atom.resname}"
                )
            i -= 1
            j -= 1
        elif direction == "U":
            i -= 1
        elif direction == "L":
            j -= 1
        else:  # pragma: no cover - defensive guard for malformed traceback state
            raise RuntimeError("invalid residue sequence alignment traceback")

    if len(matched_reference) < 3:
        raise ValueError(
            "fewer than three sequence-matched C-alpha atoms are available"
        )

    matched_reference.reverse()
    matched_mobile.reverse()
    residue_mismatches.reverse()
    return np.vstack(matched_reference), np.vstack(matched_mobile), residue_mismatches


def rmsd(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def calculate_kabsch_transform(
    mobile_coords: np.ndarray, reference_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mobile_center = mobile_coords.mean(axis=0)
    reference_center = reference_coords.mean(axis=0)
    mobile_centered = mobile_coords - mobile_center
    reference_centered = reference_coords - reference_center

    covariance = mobile_centered.T @ reference_centered
    left, _, right_transposed = np.linalg.svd(covariance)
    rotation = left @ right_transposed
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transposed
    translation = reference_center - mobile_center @ rotation
    return rotation, translation


def transform_coordinates(
    coordinates: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return coordinates @ rotation + translation


def write_transformed_pdb(
    output_path: Path,
    lines: list[str],
    atoms: list[PDBAtom],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> None:
    output_lines = list(lines)
    for atom in atoms:
        x, y, z = transform_coordinates(atom.coord, rotation, translation)
        original = output_lines[atom.line_index]
        output_lines[atom.line_index] = (
            f"{original[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{original[54:]}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="ascii")


def count_records(atoms: list[PDBAtom]) -> dict[str, int]:
    return {
        "atom_records": sum(atom.record == "ATOM" for atom in atoms),
        "hetatm_records": sum(atom.record == "HETATM" for atom in atoms),
        "total_coordinate_records": len(atoms),
    }


def coordinate_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="ascii").splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]


def residue_count(lines: list[str]) -> int:
    residues = {
        (line[21:22], line[22:26], line[26:27])
        for line in lines
        if len(line) >= 27
    }
    return len(residues)


def audit_pdb(path: Path) -> dict[str, object]:
    lines = coordinate_lines(path)
    hydrogen_count = 0
    for line in lines:
        element = line[76:78].strip() if len(line) >= 78 else ""
        atom_name = line[12:16].strip() if len(line) >= 16 else ""
        if element == "H" or (not element and atom_name.startswith("H")):
            hydrogen_count += 1
    return {
        "coordinate_record_count": len(lines),
        "atom_record_count": sum(line.startswith("ATOM  ") for line in lines),
        "hetatm_record_count": sum(line.startswith("HETATM") for line in lines),
        "residue_count": residue_count(lines),
        "hydrogen_count": hydrogen_count,
    }


def audit_pdbqt(path: Path) -> dict[str, object]:
    lines = coordinate_lines(path)
    charges: list[float] = []
    atom_types: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if len(line) < 78:
            raise ValueError(f"PDBQT coordinate line {line_number} is too short")
        try:
            charges.append(float(line[70:76].strip()))
        except ValueError as exc:
            raise ValueError(
                f"invalid PDBQT charge on coordinate line {line_number}: {line[70:76]!r}"
            ) from exc
        atom_types.add(line[77:].strip())

    if not lines:
        raise ValueError(f"no PDBQT coordinate records found in {path}")
    return {
        "coordinate_record_count": len(lines),
        "atom_record_count": sum(line.startswith("ATOM  ") for line in lines),
        "hetatm_record_count": sum(line.startswith("HETATM") for line in lines),
        "residue_count": residue_count(lines),
        "hydrogen_like_atom_count": sum(
            line[77:].strip().startswith("H") for line in lines
        ),
        "charge_min": min(charges),
        "charge_max": max(charges),
        "autodock_atom_types": sorted(atom_types),
    }
