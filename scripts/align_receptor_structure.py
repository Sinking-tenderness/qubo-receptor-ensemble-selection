"""Rigidly align a receptor structure to a reference PDB coordinate frame."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_pdb(path: Path) -> tuple[list[str], list[PDBAtom]]:
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
    common_keys = sorted(set(reference).intersection(mobile))

    matched_keys: list[tuple[int, str]] = []
    residue_mismatches: list[str] = []
    for key in common_keys:
        reference_resname = reference[key].resname
        mobile_resname = mobile[key].resname
        if reference_resname != mobile_resname:
            residue_mismatches.append(
                f"{key[0]}{key[1]}:{reference_resname}!={mobile_resname}"
            )
            continue
        matched_keys.append(key)

    if len(matched_keys) < 3:
        raise ValueError("fewer than three sequence-matched C-alpha atoms are available")

    reference_coords = np.vstack([reference[key].coord for key in matched_keys])
    mobile_coords = np.vstack([mobile[key].coord for key in matched_keys])
    return reference_coords, mobile_coords, residue_mismatches


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mobile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--reference-chain", default="A")
    parser.add_argument("--mobile-chain", default="A")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reference_lines, reference_atoms = parse_pdb(args.reference)
    mobile_lines, mobile_atoms = parse_pdb(args.mobile)
    del reference_lines

    reference_coords, mobile_coords, residue_mismatches = match_ca_coordinates(
        reference_atoms,
        mobile_atoms,
        args.reference_chain,
        args.mobile_chain,
    )
    rotation, translation = calculate_kabsch_transform(mobile_coords, reference_coords)
    aligned_mobile_coords = transform_coordinates(mobile_coords, rotation, translation)

    write_transformed_pdb(
        args.output,
        mobile_lines,
        mobile_atoms,
        rotation,
        translation,
    )
    _, output_atoms = parse_pdb(args.output)
    if count_records(output_atoms) != count_records(mobile_atoms):
        raise RuntimeError("coordinate record counts changed during alignment")

    summary = {
        "reference_path": str(args.reference),
        "reference_sha256": file_sha256(args.reference),
        "reference_chain": args.reference_chain,
        "mobile_path": str(args.mobile),
        "mobile_sha256": file_sha256(args.mobile),
        "mobile_chain": args.mobile_chain,
        "output_path": str(args.output),
        "output_sha256": file_sha256(args.output),
        "matched_ca_count": int(len(reference_coords)),
        "residue_name_mismatches_excluded": residue_mismatches,
        "rmsd_before_angstrom": rmsd(mobile_coords, reference_coords),
        "rmsd_after_angstrom": rmsd(aligned_mobile_coords, reference_coords),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "rotation_matrix_row_vector_convention": rotation.tolist(),
        "translation_vector_angstrom": translation.tolist(),
        "mobile_coordinate_record_counts": count_records(mobile_atoms),
        "output_coordinate_record_counts": count_records(output_atoms),
        "method": "sequence-matched C-alpha Kabsch rigid-body alignment",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
