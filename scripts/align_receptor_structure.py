"""Rigidly align a receptor structure to a reference PDB coordinate frame.

Thin CLI wrapper; the core logic lives in ``qubo_receptor_ensemble.pdb``.
"""



from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import argparse
import json
from pathlib import Path

import numpy as np

from qubo_receptor_ensemble.io import file_sha256
from qubo_receptor_ensemble.pdb import (
    PDBAtom,
    calculate_kabsch_transform,
    collect_ca_atoms,
    count_records,
    match_ca_coordinates,
    parse_pdb,
    rmsd,
    transform_coordinates,
    write_transformed_pdb,
)

__all__ = [
    "PDBAtom",
    "file_sha256",
    "parse_pdb",
    "collect_ca_atoms",
    "match_ca_coordinates",
    "rmsd",
    "calculate_kabsch_transform",
    "transform_coordinates",
    "write_transformed_pdb",
    "count_records",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mobile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--reference-chain", default="A")
    parser.add_argument("--mobile-chain", default="A")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reference_lines, reference_atoms = parse_pdb(args.reference)
    mobile_lines, mobile_atoms = parse_pdb(args.mobile)
    del reference_lines

    reference_coords, mobile_coords, residue_mismatches = match_ca_coordinates(
        reference_atoms,
        mobile_atoms,
        args.reference_chain,
        args.mobile_chain,
    )
    rotation, translation = calculate_kabsch_transform(mobile_coords, reference_coords)
    aligned_mobile_coords = transform_coordinates(mobile_coords, rotation, translation)

    write_transformed_pdb(
        args.output,
        mobile_lines,
        mobile_atoms,
        rotation,
        translation,
    )
    _, output_atoms = parse_pdb(args.output)
    if count_records(output_atoms) != count_records(mobile_atoms):
        raise RuntimeError("coordinate record counts changed during alignment")

    summary = {
        "reference_path": str(args.reference),
        "reference_sha256": file_sha256(args.reference),
        "reference_chain": args.reference_chain,
        "mobile_path": str(args.mobile),
        "mobile_sha256": file_sha256(args.mobile),
        "mobile_chain": args.mobile_chain,
        "output_path": str(args.output),
        "output_sha256": file_sha256(args.output),
        "matched_ca_count": int(len(reference_coords)),
        "residue_name_mismatches_excluded": residue_mismatches,
        "rmsd_before_angstrom": rmsd(mobile_coords, reference_coords),
        "rmsd_after_angstrom": rmsd(aligned_mobile_coords, reference_coords),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "rotation_matrix_row_vector_convention": rotation.tolist(),
        "translation_vector_angstrom": translation.tolist(),
        "mobile_coordinate_record_counts": count_records(mobile_atoms),
        "output_coordinate_record_counts": count_records(output_atoms),
        "method": "sequence-matched C-alpha Kabsch rigid-body alignment",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
