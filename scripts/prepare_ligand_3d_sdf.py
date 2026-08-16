"""Generate 3D SDF files from a ligand manifest using RDKit.

Thin CLI wrapper; the core logic lives in ``qubo_receptor_ensemble.preparation``.
"""



from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import argparse
from pathlib import Path

from rdkit import Chem

from qubo_receptor_ensemble.io import safe_filename
from qubo_receptor_ensemble.preparation import (
    PREP_3D_REQUIRED_COLUMNS,
    build_3d_mol,
    read_rows as _read_rows,
    validate_columns as _validate_columns,
    write_manifest,
)

REQUIRED_COLUMNS = PREP_3D_REQUIRED_COLUMNS

__all__ = [
    "REQUIRED_COLUMNS",
    "validate_columns",
    "safe_filename",
    "build_3d_mol",
    "read_rows",
    "write_manifest",
]


def validate_columns(fieldnames: list[str] | None) -> None:
    _validate_columns(fieldnames, PREP_3D_REQUIRED_COLUMNS, "CSV")


def read_rows(input_csv: Path) -> list[dict[str, str]]:
    return _read_rows(input_csv, PREP_3D_REQUIRED_COLUMNS, "CSV")


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
