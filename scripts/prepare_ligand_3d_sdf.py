"""Generate 3D SDF files from a ligand manifest using RDKit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


REQUIRED_COLUMNS = {"ligand_id", "smiles", "label", "target_id"}


def validate_columns(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("input CSV has no header")
    missing = REQUIRED_COLUMNS.difference(fieldnames)
    if missing:
        raise ValueError(f"input CSV is missing required columns: {sorted(missing)}")


def safe_filename(text: str) -> str:
    keep = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def build_3d_mol(smiles: str, seed: int) -> tuple[Chem.Mol | None, str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "failed", "rdkit_parse_failed"

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status != 0:
        return None, "failed", f"embed_failed_code_{embed_status}"

    if AllChem.MMFFHasAllMoleculeParams(mol):
        optimize_status = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        method = "MMFF94"
    else:
        optimize_status = AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        method = "UFF"

    if optimize_status < 0:
        return None, "failed", f"{method}_optimize_failed_code_{optimize_status}"
    if optimize_status > 0:
        status = "warning"
        message = f"{method}_not_converged_code_{optimize_status}"
    else:
        status = "ok"
        message = f"{method}_converged"

    return mol, status, message


def read_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames)
        return list(reader)


def write_manifest(output_manifest: Path, rows: list[dict[str, object]]) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input ligand manifest or QC CSV")
    parser.add_argument("--sdf-dir", type=Path, required=True, help="Directory for per-ligand SDF files")
    parser.add_argument("--manifest", type=Path, required=True, help="Output preparation manifest CSV")
    parser.add_argument("--seed", type=int, default=20260709)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_rows = read_rows(args.input)
    args.sdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for index, row in enumerate(input_rows):
        ligand_id = row["ligand_id"]
        smiles = row["canonical_smiles"] if row.get("canonical_smiles") else row["smiles"]
        ligand_seed = args.seed + index
        sdf_path = args.sdf_dir / f"{safe_filename(ligand_id)}.sdf"
        mol, status, message = build_3d_mol(smiles, ligand_seed)

        if mol is not None:
            mol.SetProp("_Name", ligand_id)
            mol.SetProp("ligand_id", ligand_id)
            mol.SetProp("source_smiles", row["smiles"])
            mol.SetProp("preparation_smiles", smiles)
            mol.SetProp("label", row["label"])
            mol.SetProp("target_id", row["target_id"])
            mol.SetProp("rdkit_embed_seed", str(ligand_seed))
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol)
            writer.close()
            output_path = sdf_path.as_posix()
            atom_count = mol.GetNumAtoms()
            heavy_atom_count = mol.GetNumHeavyAtoms()
        else:
            output_path = ""
            atom_count = ""
            heavy_atom_count = ""

        manifest_rows.append(
            {
                **row,
                "prep_status": status,
                "prep_message": message,
                "rdkit_embed_seed": ligand_seed,
                "sdf_path": output_path,
                "sdf_atom_count": atom_count,
                "sdf_heavy_atom_count": heavy_atom_count,
            }
        )

    write_manifest(args.manifest, manifest_rows)
    counts: dict[str, int] = {}
    for row in manifest_rows:
        status = str(row["prep_status"])
        counts[status] = counts.get(status, 0) + 1
    print(f"input_rows={len(input_rows)}")
    for status, count in sorted(counts.items()):
        print(f"{status}={count}")
    print(f"sdf_dir={args.sdf_dir}")
    print(f"manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
