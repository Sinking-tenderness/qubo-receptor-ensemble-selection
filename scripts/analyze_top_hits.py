"""Summarize top-ranked docking hits with ligand properties and inspection flags."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RANKING_REQUIRED = {"ligand_id", "label", "rank", "docking_score", "pose_path"}
MANIFEST_REQUIRED = {
    "ligand_id",
    "mol_weight",
    "heavy_atom_count",
    "formal_charge",
    "rotatable_bonds",
    "hbd",
    "hba",
    "clogp",
    "prep_status",
    "prep_message",
    "torsdof",
}


def validate_columns(fieldnames: list[str] | None, required: set[str], table_name: str) -> None:
    if fieldnames is None:
        raise ValueError(f"{table_name} has no header")
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {sorted(missing)}")


def read_csv(path: Path, required: set[str], table_name: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames, required, table_name)
        return list(reader)


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def inspection_flags(row: dict[str, object]) -> list[str]:
    flags: list[str] = []
    mol_weight = row.get("mol_weight")
    heavy_atom_count = row.get("heavy_atom_count")
    formal_charge = row.get("formal_charge")
    rotatable_bonds = row.get("rotatable_bonds")
    clogp = row.get("clogp")
    torsdof = row.get("torsdof")
    prep_status = row.get("prep_status")
    prep_message = str(row.get("prep_message") or "")

    if isinstance(mol_weight, float) and mol_weight >= 500:
        flags.append("high_mw")
    if isinstance(heavy_atom_count, int) and heavy_atom_count >= 35:
        flags.append("large_ligand")
    if isinstance(formal_charge, int) and formal_charge != 0:
        flags.append("charged")
    if isinstance(rotatable_bonds, int) and rotatable_bonds >= 8:
        flags.append("many_rotatable_bonds")
    if isinstance(torsdof, int) and torsdof >= 8:
        flags.append("high_torsdof")
    if isinstance(clogp, float) and clogp >= 4:
        flags.append("high_clogp")
    if prep_status == "warning" or "not_converged" in prep_message:
        flags.append("prep_warning")

    return flags


def merge_rows(
    ranking_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    top_n: int,
) -> list[dict[str, object]]:
    manifest_by_ligand = {row["ligand_id"]: row for row in manifest_rows}
    ranked = sorted(ranking_rows, key=lambda row: int(float(row["rank"])))
    merged: list[dict[str, object]] = []

    for row in ranked[:top_n]:
        ligand_id = row["ligand_id"]
        manifest = manifest_by_ligand.get(ligand_id)
        if manifest is None:
            raise ValueError(f"ligand {ligand_id} is missing from manifest")

        output_row: dict[str, object] = {
            "rank": to_int(row["rank"]),
            "ligand_id": ligand_id,
            "label": row["label"],
            "docking_score": to_float(row["docking_score"]),
            "pose_path": row["pose_path"],
            "mol_weight": to_float(manifest["mol_weight"]),
            "heavy_atom_count": to_int(manifest["heavy_atom_count"]),
            "formal_charge": to_int(manifest["formal_charge"]),
            "rotatable_bonds": to_int(manifest["rotatable_bonds"]),
            "torsdof": to_int(manifest["torsdof"]),
            "hbd": to_int(manifest["hbd"]),
            "hba": to_int(manifest["hba"]),
            "clogp": to_float(manifest["clogp"]),
            "formula": manifest.get("formula", ""),
            "prep_status": manifest["prep_status"],
            "prep_message": manifest["prep_message"],
            "source_molecule_id": manifest.get("source_molecule_id", ""),
            "source_extra_id": manifest.get("source_extra_id", ""),
        }
        flags = inspection_flags(output_row)
        output_row["inspection_flags"] = ";".join(flags)
        output_row["needs_inspection"] = bool(flags) or row["label"] != "active"
        merged.append(output_row)
    return merged


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    label_counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}
    for row in rows:
        label = str(row["label"])
        label_counts[label] = label_counts.get(label, 0) + 1
        flags = str(row["inspection_flags"])
        for flag in flags.split(";"):
            if not flag:
                continue
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    decoys = [row for row in rows if row["label"] != "active"]
    actives = [row for row in rows if row["label"] == "active"]
    return {
        "top_n": len(rows),
        "label_counts": label_counts,
        "decoy_ligand_ids": [row["ligand_id"] for row in decoys],
        "active_ligand_ids": [row["ligand_id"] for row in actives],
        "inspection_flag_counts": dict(sorted(flag_counts.items())),
    }


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
    parser.add_argument("--ranking", type=Path, required=True, help="Per-ligand ranking CSV")
    parser.add_argument("--manifest", type=Path, required=True, help="Ligand preparation/QC manifest CSV")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True, help="Output top-hit analysis CSV")
    parser.add_argument("--summary", type=Path, required=True, help="Output summary JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ranking_rows = read_csv(args.ranking, RANKING_REQUIRED, "ranking table")
    manifest_rows = read_csv(args.manifest, MANIFEST_REQUIRED, "manifest table")
    merged = merge_rows(ranking_rows, manifest_rows, args.top_n)
    summary = build_summary(merged)

    write_csv(args.output, merged)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"output={args.output}")
    print(f"summary={args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
