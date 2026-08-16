"""Evaluate symmetry-corrected heavy-atom redocking RMSD without pose alignment.

Thin CLI wrapper; the core logic lives in ``qubo_receptor_ensemble.docking``.
"""



from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import argparse
import csv
import json
from pathlib import Path

from qubo_receptor_ensemble.docking import (
    VINA_RESULT_PATTERN,
    calculate_pose_rmsds,
    parse_vina_affinities,
)
from qubo_receptor_ensemble.io import file_sha256

__all__ = [
    "VINA_RESULT_PATTERN",
    "file_sha256",
    "parse_vina_affinities",
    "calculate_pose_rmsds",
    "write_csv",
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
