"""Extract aligned pocket geometry features relative to a reference ligand."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BACKBONE = {"N", "CA", "C", "O", "OXT"}


@dataclass(frozen=True)
class Atom:
    record: str
    atom_name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    coord: np.ndarray


def parse_pdb(path: Path) -> list[Atom]:
    atoms: list[Atom] = []
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if len(line) < 54:
            raise ValueError(f"invalid coordinate line {line_number} in {path}")
        atoms.append(
            Atom(
                record=line[:6].strip(), atom_name=line[12:16].strip(),
                resname=line[17:20].strip(), chain=line[21:22].strip(),
                resseq=int(line[22:26]), icode=line[26:27].strip(),
                coord=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            )
        )
    if not atoms:
        raise ValueError(f"no PDB coordinates found in {path}")
    return atoms


def is_hydrogen(atom: Atom) -> bool:
    return atom.atom_name.startswith("H")


def min_distance(atoms: list[Atom], ligand_coords: np.ndarray) -> float:
    coordinates = np.vstack([atom.coord for atom in atoms])
    return float(np.linalg.norm(coordinates[:, None, :] - ligand_coords[None, :, :], axis=2).min())


def residue_key(atom: Atom) -> tuple[str, int, str, str]:
    return atom.chain, atom.resseq, atom.icode, atom.resname


def reference_pocket(
    atoms: list[Atom], chain: str, ligand_resname: str, ligand_chain: str, cutoff: float
) -> tuple[list[tuple[str, int, str, str]], np.ndarray]:
    ligand = [
        atom for atom in atoms
        if atom.record == "HETATM" and atom.resname == ligand_resname and atom.chain == ligand_chain
    ]
    if not ligand:
        raise ValueError(f"reference ligand {ligand_resname} was not found in chain {ligand_chain}")
    ligand_coords = np.vstack([atom.coord for atom in ligand if not is_hydrogen(atom)])
    by_residue: dict[tuple[str, int, str, str], list[Atom]] = defaultdict(list)
    for atom in atoms:
        if atom.record == "ATOM" and atom.chain == chain and not is_hydrogen(atom):
            by_residue[residue_key(atom)].append(atom)
    pocket = sorted(
        key for key, residue_atoms in by_residue.items()
        if min_distance(residue_atoms, ligand_coords) <= cutoff
    )
    if not pocket:
        raise ValueError("reference pocket contains no residues")
    return pocket, ligand_coords


def centroid(atoms: list[Atom]) -> np.ndarray:
    return np.vstack([atom.coord for atom in atoms]).mean(axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-pdb", type=Path, required=True)
    parser.add_argument("--reference-chain", default="A")
    parser.add_argument("--ligand-resname", default="STU")
    parser.add_argument("--ligand-chain", default="A")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--conformer-pdb", type=Path, required=True)
    parser.add_argument("--conformer-id", required=True)
    parser.add_argument("--conformer-chain", default="A")
    parser.add_argument("--residue-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    if args.cutoff <= 0:
        raise ValueError("--cutoff must be positive")

    reference_atoms = parse_pdb(args.reference_pdb)
    pocket_keys, ligand_coords = reference_pocket(
        reference_atoms, args.reference_chain, args.ligand_resname, args.ligand_chain, args.cutoff
    )
    conformer_atoms = parse_pdb(args.conformer_pdb)
    by_residue: dict[tuple[str, int, str, str], list[Atom]] = defaultdict(list)
    for atom in conformer_atoms:
        if atom.record == "ATOM" and atom.chain == args.conformer_chain and not is_hydrogen(atom):
            by_residue[residue_key(atom)].append(atom)

    reference_ca = {
        residue_key(atom): atom.coord for atom in reference_atoms
        if atom.record == "ATOM" and atom.chain == args.reference_chain and atom.atom_name == "CA"
    }
    rows: list[dict[str, object]] = []
    matched_ca_reference: list[np.ndarray] = []
    matched_ca_conformer: list[np.ndarray] = []
    for key in pocket_keys:
        _, resseq, icode, resname = key
        target_key = (args.conformer_chain, resseq, icode, resname)
        atoms = by_residue.get(target_key, [])
        sidechain = [atom for atom in atoms if atom.atom_name not in BACKBONE]
        geometry_atoms = sidechain or atoms
        ca_atoms = [atom for atom in atoms if atom.atom_name == "CA"]
        present = bool(geometry_atoms)
        row: dict[str, object] = {
            "conformer_id": args.conformer_id,
            "residue_chain": args.conformer_chain,
            "residue_number": resseq,
            "insertion_code": icode,
            "residue_name": resname,
            "present": present,
            "sidechain_atom_count": len(sidechain),
            "min_distance_to_reference_ligand_angstrom": "",
            "sidechain_centroid_x": "",
            "sidechain_centroid_y": "",
            "sidechain_centroid_z": "",
        }
        if present:
            point = centroid(geometry_atoms)
            row.update({
                "min_distance_to_reference_ligand_angstrom": round(min_distance(geometry_atoms, ligand_coords), 4),
                "sidechain_centroid_x": round(float(point[0]), 4),
                "sidechain_centroid_y": round(float(point[1]), 4),
                "sidechain_centroid_z": round(float(point[2]), 4),
            })
        if ca_atoms and key in reference_ca:
            matched_ca_reference.append(reference_ca[key])
            matched_ca_conformer.append(ca_atoms[0].coord)
        rows.append(row)

    if len(matched_ca_reference) >= 3:
        ref = np.vstack(matched_ca_reference)
        mobile = np.vstack(matched_ca_conformer)
        ca_rmsd = float(np.sqrt(np.mean(np.sum((mobile - ref) ** 2, axis=1))))
    else:
        ca_rmsd = math.nan
    args.residue_output.parent.mkdir(parents=True, exist_ok=True)
    with args.residue_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "conformer_id": args.conformer_id,
        "conformer_pdb": args.conformer_pdb.as_posix(),
        "reference_pdb": args.reference_pdb.as_posix(),
        "reference_ligand": f"{args.ligand_resname}:{args.ligand_chain}",
        "pocket_cutoff_angstrom": args.cutoff,
        "reference_pocket_residue_count": len(pocket_keys),
        "present_residue_count": sum(bool(row["present"]) for row in rows),
        "pocket_ca_rmsd_to_reference_angstrom": None if math.isnan(ca_rmsd) else round(ca_rmsd, 4),
        "interpretation_note": "Distances are geometry proxies in a shared coordinate frame, not interaction energies or experimental binding evidence.",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
