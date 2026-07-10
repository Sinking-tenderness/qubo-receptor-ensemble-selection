"""Prepare PyMOL inputs for visual inspection of selected docked poses."""

from __future__ import annotations

import argparse
import csv
import locale
from pathlib import Path


RANKING_REQUIRED = {"ligand_id", "label", "rank", "docking_score", "pose_path"}
AUTODOCK_ELEMENT_MAP = {
    "A": "C",
    "C": "C",
    "N": "N",
    "NA": "N",
    "OA": "O",
    "O": "O",
    "HD": "H",
    "H": "H",
    "S": "S",
    "SA": "S",
    "BR": "Br",
    "CL": "Cl",
    "F": "F",
    "I": "I",
}


def validate_columns(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("ranking CSV has no header")
    missing = RANKING_REQUIRED.difference(fieldnames)
    if missing:
        raise ValueError(f"ranking CSV is missing required columns: {sorted(missing)}")


def read_ranking(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames)
        return list(reader)


def ligand_short_name(ligand_id: str) -> str:
    return "LIG"


def autodock_atom_type_to_element(atom_type: str) -> str:
    normalized = atom_type.strip().upper()
    return AUTODOCK_ELEMENT_MAP.get(normalized, normalized[:1].title() or "C")


def pdb_atom_line_from_pdbqt(line: str, atom_index: int, residue_name: str) -> str:
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    atom_type = line[77:].strip() if len(line) >= 78 else line[12:16].strip()
    element = autodock_atom_type_to_element(atom_type)
    atom_name = f"{element}{atom_index % 1000:03d}"[:4]
    return (
        f"HETATM{atom_index:5d} {atom_name:<4} {residue_name:>3} X{1:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2}\n"
    )


def extract_first_model_pdb(pdbqt_path: Path, output_pdb: Path, residue_name: str) -> int:
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    atom_count = 0
    in_first_model = False
    seen_model = False
    with pdbqt_path.open("r", encoding="utf-8", errors="ignore") as source, output_pdb.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        target.write(f"REMARK Extracted first docking model from {pdbqt_path.as_posix()}\n")
        for line in source:
            if line.startswith("MODEL"):
                if seen_model:
                    break
                seen_model = True
                in_first_model = True
                continue
            if line.startswith("ENDMDL") and in_first_model:
                break
            if not in_first_model and seen_model:
                continue
            if line.startswith(("ATOM", "HETATM")):
                atom_count += 1
                target.write(pdb_atom_line_from_pdbqt(line, atom_count, residue_name))
        target.write("END\n")
    if atom_count == 0:
        raise ValueError(f"no atoms extracted from {pdbqt_path}")
    return atom_count


def write_pymol_script(
    output_pml: Path,
    receptor_pdb: Path,
    extracted: list[dict[str, object]],
) -> None:
    output_pml.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "reinitialize",
        f"load {receptor_pdb.resolve().as_posix()}, receptor_1AQ1",
        "hide everything",
        "show cartoon, receptor_1AQ1 and polymer.protein",
        "color gray80, receptor_1AQ1 and polymer.protein",
        "show sticks, receptor_1AQ1 and resn STU",
        "color yellow, receptor_1AQ1 and resn STU",
        "select pocket5A, byres (receptor_1AQ1 and polymer.protein within 5 of (receptor_1AQ1 and resn STU))",
        "show sticks, pocket5A",
        "color cyan, pocket5A",
        "set stick_radius, 0.16",
        "set sphere_scale, 0.25",
    ]

    active_colors = ["limegreen", "forest", "chartreuse"]
    decoy_colors = ["magenta", "orange", "hotpink", "tv_red"]
    active_index = 0
    decoy_index = 0
    for item in extracted:
        object_name = str(item["object_name"])
        pose_path = Path(str(item["pose_pdb"]))
        label = str(item["label"])
        color = active_colors[active_index % len(active_colors)] if label == "active" else decoy_colors[decoy_index % len(decoy_colors)]
        if label == "active":
            active_index += 1
        else:
            decoy_index += 1
        lines.extend(
            [
                f"load {pose_path.resolve().as_posix()}, {object_name}",
                f"show sticks, {object_name}",
                f"color {color}, {object_name}",
                f"set stick_radius, 0.22, {object_name}",
            ]
        )

    lines.extend(
        [
            "orient receptor_1AQ1 and resn STU",
            "zoom receptor_1AQ1 and resn STU, 8",
            "set transparency, 0.25, receptor_1AQ1 and cartoon",
            "bg_color white",
        ]
    )
    # PyMOL on Windows may read .pml files using the system ANSI code page.
    # Use the preferred locale encoding so non-ASCII local paths are readable.
    encoding = locale.getpreferredencoding(False)
    output_pml.write_text("\n".join(lines) + "\n", encoding=encoding)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--receptor-pdb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pymol-script", type=Path, required=True)
    parser.add_argument("--ligand-ids", nargs="+", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ranking_rows = read_ranking(args.ranking)
    ranking_by_ligand = {row["ligand_id"]: row for row in ranking_rows}
    extracted: list[dict[str, object]] = []

    for ligand_id in args.ligand_ids:
        if ligand_id not in ranking_by_ligand:
            raise ValueError(f"ligand {ligand_id} is not present in ranking table")
        row = ranking_by_ligand[ligand_id]
        pose_path = Path(row["pose_path"])
        if not pose_path.exists():
            raise FileNotFoundError(f"missing pose file for {ligand_id}: {pose_path}")
        residue_name = ligand_short_name(ligand_id)
        output_pdb = args.output_dir / f"{ligand_id}_pose1.pdb"
        atom_count = extract_first_model_pdb(pose_path, output_pdb, residue_name)
        extracted.append(
            {
                "ligand_id": ligand_id,
                "label": row["label"],
                "rank": row["rank"],
                "docking_score": row["docking_score"],
                "pose_pdb": output_pdb,
                "object_name": f"{ligand_id}_{row['label']}_rank{row['rank']}",
                "atom_count": atom_count,
            }
        )

    write_pymol_script(args.pymol_script, args.receptor_pdb, extracted)
    print("prepared_pose_count=" + str(len(extracted)))
    for item in extracted:
        print(
            f"{item['ligand_id']}\t{item['label']}\trank={item['rank']}\t"
            f"score={item['docking_score']}\tatoms={item['atom_count']}\t{item['pose_pdb']}"
        )
    print(f"pymol_script={args.pymol_script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
