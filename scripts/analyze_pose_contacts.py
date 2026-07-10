"""Analyze simple receptor-ligand contact geometry for selected docked poses."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


POLAR_ELEMENTS = {"N", "O", "S"}
HYDROPHOBIC_ELEMENTS = {"C"}


@dataclass(frozen=True)
class Atom:
    serial: int
    name: str
    resname: str
    chain: str
    resseq: str
    x: float
    y: float
    z: float
    element: str
    record: str

    @property
    def residue_id(self) -> str:
        return f"{self.chain}:{self.resname}{self.resseq}"


def infer_element(line: str) -> str:
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element.title()
    atom_name = line[12:16].strip()
    letters = "".join(char for char in atom_name if char.isalpha())
    return letters[:1].title() or "C"


def parse_pdb_atoms(path: Path, protein_only: bool = False) -> list[Atom]:
    atoms: list[Atom] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            record = line[0:6].strip()
            resname = line[17:20].strip()
            if protein_only and record != "ATOM":
                continue
            if protein_only and resname in {"HOH", "WAT"}:
                continue
            atoms.append(
                Atom(
                    serial=int(line[6:11]),
                    name=line[12:16].strip(),
                    resname=resname,
                    chain=line[21:22].strip() or "_",
                    resseq=line[22:27].strip(),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    element=infer_element(line),
                    record=record,
                )
            )
    if not atoms:
        raise ValueError(f"no atoms parsed from {path}")
    return atoms


def distance(a: Atom, b: Atom) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def classify_contact(receptor_atom: Atom, ligand_atom: Atom, dist: float, polar_cutoff: float, hydrophobic_cutoff: float) -> str:
    if receptor_atom.element in POLAR_ELEMENTS and ligand_atom.element in POLAR_ELEMENTS and dist <= polar_cutoff:
        return "polar_candidate"
    if receptor_atom.element in HYDROPHOBIC_ELEMENTS and ligand_atom.element in HYDROPHOBIC_ELEMENTS and dist <= hydrophobic_cutoff:
        return "hydrophobic_candidate"
    return "close_contact"


def analyze_ligand(
    ligand_id: str,
    ligand_path: Path,
    receptor_atoms: list[Atom],
    contact_cutoff: float,
    polar_cutoff: float,
    hydrophobic_cutoff: float,
    clash_cutoff: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    ligand_atoms = parse_pdb_atoms(ligand_path)
    contact_rows: list[dict[str, object]] = []
    residue_min_distance: dict[str, float] = {}
    residue_contact_types: dict[str, set[str]] = {}
    polar_count = 0
    hydrophobic_count = 0
    clash_count = 0

    for ligand_atom in ligand_atoms:
        if ligand_atom.element == "H":
            continue
        for receptor_atom in receptor_atoms:
            if receptor_atom.element == "H":
                continue
            dist = distance(receptor_atom, ligand_atom)
            if dist > contact_cutoff:
                continue
            contact_type = classify_contact(receptor_atom, ligand_atom, dist, polar_cutoff, hydrophobic_cutoff)
            if contact_type == "polar_candidate":
                polar_count += 1
            elif contact_type == "hydrophobic_candidate":
                hydrophobic_count += 1
            if dist < clash_cutoff:
                clash_count += 1
            residue_id = receptor_atom.residue_id
            residue_min_distance[residue_id] = min(dist, residue_min_distance.get(residue_id, dist))
            residue_contact_types.setdefault(residue_id, set()).add(contact_type)
            contact_rows.append(
                {
                    "ligand_id": ligand_id,
                    "receptor_residue": residue_id,
                    "receptor_atom": receptor_atom.name,
                    "receptor_element": receptor_atom.element,
                    "ligand_atom": ligand_atom.name,
                    "ligand_element": ligand_atom.element,
                    "distance_angstrom": round(dist, 3),
                    "contact_type": contact_type,
                    "possible_clash": dist < clash_cutoff,
                }
            )

    sorted_residues = sorted(residue_min_distance, key=lambda residue: residue_min_distance[residue])
    hinge_residues = [residue for residue in sorted_residues if residue.endswith("GLU81") or residue.endswith("LEU83")]
    summary = {
        "ligand_id": ligand_id,
        "ligand_path": ligand_path.as_posix(),
        "ligand_heavy_atom_count": sum(1 for atom in ligand_atoms if atom.element != "H"),
        "contact_row_count": len(contact_rows),
        "contact_residue_count": len(residue_min_distance),
        "polar_candidate_count": polar_count,
        "hydrophobic_candidate_count": hydrophobic_count,
        "possible_clash_count": clash_count,
        "closest_residues": ";".join(
            f"{residue}:{residue_min_distance[residue]:.2f}" for residue in sorted_residues[:10]
        ),
        "hinge_residue_contacts": ";".join(hinge_residues),
        "contact_residue_list": ";".join(sorted_residues),
    }
    return contact_rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor-pdb", type=Path, required=True)
    parser.add_argument("--ligand", nargs=2, action="append", metavar=("LIGAND_ID", "PDB"), required=True)
    parser.add_argument("--contacts-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--polar-cutoff", type=float, default=3.5)
    parser.add_argument("--hydrophobic-cutoff", type=float, default=4.0)
    parser.add_argument("--clash-cutoff", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    receptor_atoms = parse_pdb_atoms(args.receptor_pdb, protein_only=True)
    all_contacts: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for ligand_id, ligand_path_text in args.ligand:
        ligand_path = Path(ligand_path_text)
        contacts, summary = analyze_ligand(
            ligand_id=ligand_id,
            ligand_path=ligand_path,
            receptor_atoms=receptor_atoms,
            contact_cutoff=args.contact_cutoff,
            polar_cutoff=args.polar_cutoff,
            hydrophobic_cutoff=args.hydrophobic_cutoff,
            clash_cutoff=args.clash_cutoff,
        )
        all_contacts.extend(contacts)
        summaries.append(summary)

    write_csv(args.contacts_output, all_contacts)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"contacts_output={args.contacts_output}")
    print(f"summary_output={args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
