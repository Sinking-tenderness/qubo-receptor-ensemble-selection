"""Apply an audited receptor-alignment transform to every conformer in an SDF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_transform(path: Path | None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if path is None:
        return np.eye(3), np.zeros(3), {"method": "identity"}
    summary = json.loads(path.read_text(encoding="ascii"))
    try:
        rotation = np.asarray(summary["rotation_matrix_row_vector_convention"], dtype=float)
        translation = np.asarray(summary["translation_vector_angstrom"], dtype=float)
    except KeyError as exc:
        raise ValueError(f"alignment summary is missing transform field: {exc.args[0]}") from exc
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("alignment transform must contain a 3x3 rotation and 3-vector translation")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-6):
        raise ValueError(f"alignment rotation determinant is not +1: {determinant}")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("alignment rotation matrix is not orthonormal")
    return rotation, translation, summary


def transform_coordinates(
    coordinates: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (n, 3)")
    return coordinates @ rotation + translation


def geometry_audit(coordinates: np.ndarray) -> dict[str, list[float]]:
    return {
        "centroid_angstrom": coordinates.mean(axis=0).tolist(),
        "minimum_angstrom": coordinates.min(axis=0).tolist(),
        "maximum_angstrom": coordinates.max(axis=0).tolist(),
        "span_angstrom": np.ptp(coordinates, axis=0).tolist(),
    }


def maximum_distance_error(before: np.ndarray, after: np.ndarray) -> float:
    before_distances = np.linalg.norm(before[:, None, :] - before[None, :, :], axis=2)
    after_distances = np.linalg.norm(after[:, None, :] - after[None, :, :], axis=2)
    return float(np.max(np.abs(before_distances - after_distances)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sdf", type=Path, required=True)
    parser.add_argument(
        "--alignment-summary",
        type=Path,
        help="Alignment JSON written by align_receptor_structure.py; omit for identity.",
    )
    parser.add_argument("--output-sdf", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--add-explicit-hydrogens",
        action="store_true",
        help="Add coordinate-bearing explicit hydrogens after transforming heavy atoms.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    from rdkit import Chem

    args = build_parser().parse_args()
    if not args.input_sdf.is_file():
        raise FileNotFoundError(args.input_sdf)
    if args.alignment_summary is not None and not args.alignment_summary.is_file():
        raise FileNotFoundError(args.alignment_summary)
    existing = [path for path in (args.output_sdf, args.summary_output) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("outputs exist; use --overwrite after review")

    rotation, translation, alignment = load_transform(args.alignment_summary)
    supplier = Chem.SDMolSupplier(str(args.input_sdf), removeHs=False, sanitize=True)
    molecules = list(supplier)
    if not molecules or any(molecule is None for molecule in molecules):
        raise ValueError("input SDF contains an unparseable molecule")

    maximum_error = 0.0
    molecule_audits: list[dict[str, object]] = []
    for molecule_index, molecule in enumerate(molecules):
        if molecule.GetNumConformers() == 0:
            raise ValueError(f"molecule {molecule_index} has no coordinates")
        input_atom_count = molecule.GetNumAtoms()
        input_heavy_atom_count = molecule.GetNumHeavyAtoms()
        before = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
        after = transform_coordinates(before, rotation, translation)
        distance_error = maximum_distance_error(before, after)
        maximum_error = max(maximum_error, distance_error)
        conformer = molecule.GetConformer()
        for atom_index, (x, y, z) in enumerate(after):
            conformer.SetAtomPosition(atom_index, (float(x), float(y), float(z)))
        if args.add_explicit_hydrogens:
            molecule = Chem.AddHs(molecule, addCoords=True)
            molecules[molecule_index] = molecule
        output_coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
        heavy_coordinate_error = float(
            np.max(np.abs(output_coordinates[:input_atom_count] - after))
        )
        if heavy_coordinate_error > 1e-6:
            raise RuntimeError(
                "adding explicit hydrogens changed transformed input-atom coordinates"
            )
        molecule_audits.append(
            {
                "molecule_index": molecule_index,
                "input_atom_count": input_atom_count,
                "input_heavy_atom_count": input_heavy_atom_count,
                "output_atom_count": molecule.GetNumAtoms(),
                "output_heavy_atom_count": molecule.GetNumHeavyAtoms(),
                "explicit_hydrogens_added": molecule.GetNumAtoms() - input_atom_count,
                "formal_charge": Chem.GetFormalCharge(molecule),
                "before": geometry_audit(before),
                "after": geometry_audit(after),
                "maximum_pairwise_distance_error_angstrom": distance_error,
                "maximum_input_atom_coordinate_error_after_hydrogen_addition_angstrom": (
                    heavy_coordinate_error
                ),
            }
        )
    if maximum_error > 1e-6:
        raise RuntimeError(f"rigid transform changed internal distances by {maximum_error}")

    args.output_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(args.output_sdf))
    try:
        for molecule in molecules:
            writer.write(molecule)
    finally:
        writer.close()
    if not args.output_sdf.is_file() or args.output_sdf.stat().st_size == 0:
        raise RuntimeError("transformed SDF was not written")

    summary = {
        "schema_version": "1.0",
        "status": "ok",
        "operation": "rigid-body SDF coordinate transform using row-vector convention",
        "input_sdf": args.input_sdf.as_posix(),
        "input_sdf_sha256": file_sha256(args.input_sdf),
        "alignment_summary": (
            args.alignment_summary.as_posix() if args.alignment_summary is not None else None
        ),
        "alignment_summary_sha256": (
            file_sha256(args.alignment_summary) if args.alignment_summary is not None else None
        ),
        "alignment_method": alignment.get("method", "unknown"),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "translation_vector_angstrom": translation.tolist(),
        "add_explicit_hydrogens": args.add_explicit_hydrogens,
        "molecule_count": len(molecules),
        "molecule_audits": molecule_audits,
        "maximum_pairwise_distance_error_angstrom": maximum_error,
        "output_sdf": args.output_sdf.as_posix(),
        "output_sdf_sha256": file_sha256(args.output_sdf),
        "interpretation_note": (
            "Input-atom coordinates were rigidly transformed without changing internal "
            "geometry. Explicit hydrogens were added only when requested; transformed "
            "heavy-atom coordinates were retained."
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
