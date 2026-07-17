"""Evaluate symmetry-corrected heavy-atom redocking RMSD without pose alignment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


VINA_RESULT_PATTERN = re.compile(
    r"^REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", re.MULTILINE
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_vina_affinities(text: str) -> list[float]:
    return [float(match.group(1)) for match in VINA_RESULT_PATTERN.finditer(text)]


def calculate_pose_rmsds(reference_sdf: Path, docked_pdbqt: Path) -> list[float]:
    from meeko import PDBQTMolecule, RDKitMolCreate
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    reference = Chem.SDMolSupplier(
        str(reference_sdf), removeHs=True, sanitize=True
    )[0]
    if reference is None or reference.GetNumConformers() != 1:
        raise ValueError("reference SDF must contain one parseable 3D molecule")
    pdbqt_molecule = PDBQTMolecule.from_file(str(docked_pdbqt), skip_typing=False)
    converted = RDKitMolCreate.from_pdbqt_mol(pdbqt_molecule)
    if len(converted) != 1 or converted[0] is None:
        raise ValueError("docked PDBQT did not convert to exactly one RDKit molecule")
    predicted = Chem.RemoveHs(converted[0])
    if predicted.GetNumAtoms() != reference.GetNumAtoms():
        raise ValueError(
            "reference and predicted heavy-atom counts differ: "
            f"{reference.GetNumAtoms()} versus {predicted.GetNumAtoms()}"
        )
    return [
        float(
            rdMolAlign.CalcRMS(
                predicted,
                reference,
                prbId=pose_index,
                refId=0,
                maxMatches=1_000_000,
                symmetrizeConjugatedTerminalGroups=True,
            )
        )
        for pose_index in range(predicted.GetNumConformers())
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--reference-sdf", type=Path, required=True)
    parser.add_argument("--docked-pdbqt", type=Path, required=True)
    parser.add_argument("--pose-table-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--success-threshold", type=float, default=2.0)
    args = parser.parse_args()

    for path in (args.reference_sdf, args.docked_pdbqt):
        if not path.is_file():
            raise FileNotFoundError(path)
    affinities = parse_vina_affinities(
        args.docked_pdbqt.read_text(encoding="ascii", errors="replace")
    )
    rmsds = calculate_pose_rmsds(args.reference_sdf, args.docked_pdbqt)
    if len(affinities) != len(rmsds):
        raise ValueError(
            f"affinity count {len(affinities)} differs from pose count {len(rmsds)}"
        )
    rows = [
        {
            "case_id": args.case_id,
            "pose_rank": rank,
            "affinity_kcal_per_mol": affinity,
            "symmetry_corrected_heavy_atom_rmsd_angstrom": round(rmsd, 6),
            "within_success_threshold": rmsd <= args.success_threshold,
        }
        for rank, (affinity, rmsd) in enumerate(zip(affinities, rmsds), start=1)
    ]
    write_csv(args.pose_table_output, rows)
    best_rmsd_row = min(rows, key=lambda row: float(row["symmetry_corrected_heavy_atom_rmsd_angstrom"]))
    summary = {
        "schema_version": "1.0",
        "case_id": args.case_id,
        "status": "ok",
        "rmsd_definition": (
            "symmetry-corrected heavy-atom RMSD in the fixed receptor coordinate "
            "frame; no post-docking rigid-body alignment"
        ),
        "reference_sdf": {
            "path": args.reference_sdf.as_posix(),
            "sha256": file_sha256(args.reference_sdf),
        },
        "docked_pdbqt": {
            "path": args.docked_pdbqt.as_posix(),
            "sha256": file_sha256(args.docked_pdbqt),
        },
        "pose_count": len(rows),
        "success_threshold_angstrom": args.success_threshold,
        "top_ranked_affinity_kcal_per_mol": rows[0]["affinity_kcal_per_mol"],
        "top_ranked_rmsd_angstrom": rows[0]["symmetry_corrected_heavy_atom_rmsd_angstrom"],
        "top_ranked_pose_success": rows[0]["within_success_threshold"],
        "best_rmsd_pose_rank": best_rmsd_row["pose_rank"],
        "best_rmsd_angstrom": best_rmsd_row[
            "symmetry_corrected_heavy_atom_rmsd_angstrom"
        ],
        "any_pose_success": any(bool(row["within_success_threshold"]) for row in rows),
        "pose_table": {
            "path": args.pose_table_output.as_posix(),
            "sha256": file_sha256(args.pose_table_output),
        },
        "interpretation_note": (
            "A successful redocking pose supports pose reproduction for this prepared "
            "receptor-ligand pair. It does not validate affinity prediction or screening enrichment."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
