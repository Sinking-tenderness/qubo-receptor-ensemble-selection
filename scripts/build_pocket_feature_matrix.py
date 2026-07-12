"""Combine per-conformer pocket residue features into a wide feature matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def residue_label(row: dict[str, str]) -> str:
    insertion = row["insertion_code"] or "_"
    # Conformer chains may differ (for example, PDB 1JVP uses chain P), while
    # residue numbering and residue identity define the aligned reference schema.
    return f"{row['residue_number']}_{insertion}_{row['residue_name']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_csv(args.feature_manifest)
    ok = [row for row in manifest if row.get("status") == "ok"]
    if not ok:
        raise ValueError("feature manifest has no successful conformers")

    per_conformer = {row["conformer_id"]: read_csv(Path(row["residue_csv"])) for row in ok}
    labels = [residue_label(row) for row in per_conformer[ok[0]["conformer_id"]]]
    if len(labels) != len(set(labels)):
        raise ValueError("reference conformer has duplicate residue labels")
    expected = set(labels)
    output_rows: list[dict[str, object]] = []
    for manifest_row in ok:
        conformer_id = manifest_row["conformer_id"]
        rows = per_conformer[conformer_id]
        by_label = {residue_label(row): row for row in rows}
        if set(by_label) != expected:
            raise ValueError(f"residue feature schema differs for {conformer_id}")
        output: dict[str, object] = {
            "conformer_id": conformer_id,
            "source_type": manifest_row["source_type"],
            "pdb_path": manifest_row["pdb_path"],
            "chain": manifest_row["chain"],
        }
        for label in labels:
            row = by_label[label]
            for name in ("present", "min_distance_to_reference_ligand_angstrom", "sidechain_centroid_x", "sidechain_centroid_y", "sidechain_centroid_z"):
                output[f"{label}__{name}"] = row[name]
        output_rows.append(output)

    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with args.matrix_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "conformer_count": len(output_rows),
        "pocket_residue_count": len(labels),
        "feature_count_excluding_metadata": len(fields) - 4,
        "feature_schema": ["present", "min_distance_to_reference_ligand_angstrom", "sidechain_centroid_x", "sidechain_centroid_y", "sidechain_centroid_z"],
        "interpretation_note": "This matrix contains pocket geometry proxies. It is not a ligand interaction matrix and must not be used as an activity label.",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
