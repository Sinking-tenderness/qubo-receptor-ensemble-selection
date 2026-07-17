"""Audit a ligand manifest with RDKit before 3D ligand preparation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


REQUIRED_COLUMNS = {"ligand_id", "smiles", "label", "source", "target_id"}


def validate_columns(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("input CSV has no header")
    missing = REQUIRED_COLUMNS.difference(fieldnames)
    if missing:
        raise ValueError(f"input CSV is missing required columns: {sorted(missing)}")


def audit_row(row: dict[str, str]) -> dict[str, str | int | float | bool]:
    smiles = row["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            **row,
            "rdkit_status": "invalid_smiles",
            "canonical_smiles": "",
            "formula": "",
            "heavy_atom_count": "",
            "mol_weight": "",
            "formal_charge": "",
            "rotatable_bonds": "",
            "hbd": "",
            "hba": "",
            "clogp": "",
            "fragment_count": "",
            "has_multiple_fragments": "",
        }

    fragments = Chem.GetMolFrags(mol)
    return {
        **row,
        "rdkit_status": "ok",
        "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "mol_weight": round(Descriptors.MolWt(mol), 3),
        "formal_charge": Chem.GetFormalCharge(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "clogp": round(Crippen.MolLogP(mol), 3),
        "fragment_count": len(fragments),
        "has_multiple_fragments": len(fragments) > 1,
    }


def read_rows(input_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames)
        return list(reader), list(reader.fieldnames or [])


def write_csv(output_csv: Path, rows: list[dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    label_counts = Counter(str(row["label"]) for row in rows)
    status_counts = Counter(str(row["rdkit_status"]) for row in rows)
    canonical = [str(row["canonical_smiles"]) for row in rows if row["rdkit_status"] == "ok"]
    canonical_counts = Counter(canonical)
    duplicate_canonical = sorted(smiles for smiles, count in canonical_counts.items() if count > 1)
    ok_rows = [row for row in rows if row["rdkit_status"] == "ok"]
    multi_fragment = [row["ligand_id"] for row in ok_rows if row["has_multiple_fragments"]]
    charged = [row["ligand_id"] for row in ok_rows if row["formal_charge"] != 0]
    heavy_atom_counts = [int(row["heavy_atom_count"]) for row in ok_rows]
    mol_weights = [float(row["mol_weight"]) for row in ok_rows]
    return {
        "row_count": len(rows),
        "label_counts": dict(label_counts),
        "rdkit_status_counts": dict(status_counts),
        "unique_canonical_smiles": len(canonical_counts),
        "duplicate_canonical_smiles_count": len(duplicate_canonical),
        "duplicate_canonical_smiles": duplicate_canonical,
        "multi_fragment_ligand_count": len(multi_fragment),
        "multi_fragment_ligand_ids_preview": multi_fragment[:100],
        "charged_ligand_count": len(charged),
        "charged_ligand_ids_preview": charged[:100],
        "preview_limit": 100,
        "heavy_atom_count_min": min(heavy_atom_counts) if heavy_atom_counts else None,
        "heavy_atom_count_max": max(heavy_atom_counts) if heavy_atom_counts else None,
        "mol_weight_min": min(mol_weights) if mol_weights else None,
        "mol_weight_max": max(mol_weights) if mol_weights else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input ligand manifest CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output audited CSV")
    parser.add_argument("--summary", type=Path, required=True, help="Output JSON summary")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_rows, _ = read_rows(args.input)
    audited_rows = [audit_row(row) for row in input_rows]
    write_csv(args.output, audited_rows)
    summary = build_summary(audited_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"output={args.output}")
    print(f"summary={args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
