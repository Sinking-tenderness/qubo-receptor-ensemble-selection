"""Align MD medoid receptors to 1AQ1 and export protein-only heavy-atom PDBs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .align_receptor_structure import (
        PDBAtom,
        calculate_kabsch_transform,
        collect_ca_atoms,
        file_sha256,
        match_ca_coordinates,
        parse_pdb,
        rmsd,
        transform_coordinates,
    )
except ImportError:
    from align_receptor_structure import (
        PDBAtom,
        calculate_kabsch_transform,
        collect_ca_atoms,
        file_sha256,
        match_ca_coordinates,
        parse_pdb,
        rmsd,
        transform_coordinates,
    )


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "clustering_experiment_id",
    "purpose",
    "inputs",
    "reference_pdb_sha256",
    "reference_chain",
    "mobile_chain",
    "expected_medoid_count",
    "expected_mobile_residue_count",
    "pocket_residue_numbers",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_INPUT_KEYS = {"clustering_summary", "medoid_manifest", "reference_pdb"}
REQUIRED_OUTPUT_KEYS = {"aligned_heavy_directory", "alignment_manifest_csv", "summary_json"}


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("medoid alignment configuration must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"medoid alignment configuration is missing keys: {', '.join(missing)}")
    inputs = config["inputs"]
    outputs = config["outputs"]
    if not isinstance(inputs, dict) or not REQUIRED_INPUT_KEYS.issubset(inputs):
        raise ValueError("inputs is missing one or more medoid alignment paths")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUT_KEYS.issubset(outputs):
        raise ValueError("outputs is missing one or more medoid alignment paths")
    if len(str(config["reference_chain"])) != 1 or len(str(config["mobile_chain"])) != 1:
        raise ValueError("reference_chain and mobile_chain must be one-character PDB IDs")
    if int(config["expected_medoid_count"]) <= 0 or int(config["expected_mobile_residue_count"]) <= 0:
        raise ValueError("expected medoid and residue counts must be positive")
    pocket = config["pocket_residue_numbers"]
    if (
        not isinstance(pocket, list)
        or len(pocket) < 3
        or any(not isinstance(value, int) or value <= 0 for value in pocket)
        or len(set(pocket)) != len(pocket)
    ):
        raise ValueError("pocket_residue_numbers must contain unique positive integers")
    return config


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty alignment manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_hydrogen_pdb_atom(atom: PDBAtom, original_line: str) -> bool:
    element = original_line[76:78].strip().upper() if len(original_line) >= 78 else ""
    normalized_name = atom.atom_name.upper().lstrip("0123456789")
    return element == "H" or (not element and normalized_name.startswith("H"))


def write_aligned_heavy_protein(
    output_path: Path,
    lines: list[str],
    atoms: list[PDBAtom],
    chain: str,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> None:
    output_lines: list[str] = []
    for atom in atoms:
        original = lines[atom.line_index]
        if (
            atom.record != "ATOM"
            or atom.chain != chain
            or atom.altloc not in {"", "A"}
            or is_hydrogen_pdb_atom(atom, original)
        ):
            continue
        x, y, z = transform_coordinates(atom.coord, rotation, translation)
        output_lines.append(f"{original[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{original[54:]}")
    if not output_lines:
        raise ValueError("heavy-atom protein selection produced no coordinate records")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join([*output_lines, "TER", "END"]) + "\n", encoding="ascii")


def audit_heavy_protein(path: Path, expected_chain: str) -> dict[str, object]:
    lines, atoms = parse_pdb(path)
    hydrogen_count = sum(
        is_hydrogen_pdb_atom(atom, lines[atom.line_index]) for atom in atoms
    )
    residues = {(atom.chain, atom.resseq, atom.icode) for atom in atoms if atom.record == "ATOM"}
    chains = sorted({atom.chain for atom in atoms if atom.record == "ATOM"})
    audit = {
        "coordinate_record_count": len(atoms),
        "atom_record_count": sum(atom.record == "ATOM" for atom in atoms),
        "hetatm_record_count": sum(atom.record == "HETATM" for atom in atoms),
        "hydrogen_count": hydrogen_count,
        "residue_count": len(residues),
        "chains": chains,
    }
    if audit["hetatm_record_count"] != 0 or hydrogen_count != 0:
        raise RuntimeError("aligned heavy receptor contains HETATM or hydrogen records")
    if chains != [expected_chain]:
        raise RuntimeError(f"aligned heavy receptor chains differ from {expected_chain!r}: {chains}")
    return audit


def pocket_ca_rmsd(
    reference_atoms: list[PDBAtom],
    mobile_atoms: list[PDBAtom],
    reference_chain: str,
    mobile_chain: str,
    pocket_numbers: list[int],
) -> float:
    reference = collect_ca_atoms(reference_atoms, reference_chain)
    mobile = collect_ca_atoms(mobile_atoms, mobile_chain)
    reference_by_number = {
        number: atom.coord for (number, icode), atom in reference.items()
        if not icode and number in pocket_numbers
    }
    mobile_by_number = {
        number: atom.coord for (number, icode), atom in mobile.items()
        if not icode and number in pocket_numbers
    }
    expected = set(pocket_numbers)
    if set(reference_by_number) != expected or set(mobile_by_number) != expected:
        raise ValueError("reference or mobile structure is missing a requested pocket CA atom")
    reference_coords = np.vstack([reference_by_number[number] for number in pocket_numbers])
    mobile_coords = np.vstack([mobile_by_number[number] for number in pocket_numbers])
    return rmsd(mobile_coords, reference_coords)


def temporal_support_role(row: dict[str, str]) -> str:
    revisited = row.get("revisited_after_exit", "").strip().lower() == "true"
    cluster_size = int(row["cluster_size"])
    if not revisited and cluster_size < 5:
        return "exploratory_low_temporal_support"
    return "revisited_primary_candidate" if revisited else "single_visit_candidate"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    outputs = config["outputs"]
    assert isinstance(inputs, dict)
    assert isinstance(outputs, dict)
    input_paths = {key: Path(str(value)) for key, value in inputs.items()}
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_reference_hash = str(config["reference_pdb_sha256"]).upper()
    if file_sha256(input_paths["reference_pdb"]) != expected_reference_hash:
        raise ValueError("reference PDB SHA-256 differs from the configured value")

    clustering_summary = json.loads(
        input_paths["clustering_summary"].read_text(encoding="ascii")
    )
    if clustering_summary.get("status") != "ok":
        raise ValueError("clustering summary does not have status=ok")
    if clustering_summary.get("experiment_id") != config["clustering_experiment_id"]:
        raise ValueError("clustering experiment ID differs from alignment configuration")
    recorded_manifest_hash = (
        clustering_summary.get("output_sha256", {}).get("medoid_manifest_csv")
        if isinstance(clustering_summary.get("output_sha256"), dict)
        else None
    )
    actual_manifest_hash = file_sha256(input_paths["medoid_manifest"])
    if str(recorded_manifest_hash).upper() != actual_manifest_hash:
        raise ValueError("medoid manifest SHA-256 differs from the clustering summary")

    medoids = read_csv(input_paths["medoid_manifest"])
    expected_medoid_count = int(config["expected_medoid_count"])
    if len(medoids) != expected_medoid_count:
        raise ValueError(
            f"expected {expected_medoid_count} medoids but manifest contains {len(medoids)}"
        )
    conformer_ids = [row["conformer_id"] for row in medoids]
    if len(set(conformer_ids)) != len(conformer_ids):
        raise ValueError("medoid manifest contains duplicate conformer IDs")

    output_directory = Path(str(outputs["aligned_heavy_directory"]))
    manifest_path = Path(str(outputs["alignment_manifest_csv"]))
    summary_path = Path(str(outputs["summary_json"]))
    existing = [path for path in (manifest_path, summary_path) if path.exists()]
    existing_pdbs = list(output_directory.glob("*.pdb")) if output_directory.exists() else []
    if (existing or existing_pdbs) and not args.overwrite:
        raise FileExistsError("medoid alignment outputs exist; use --overwrite after review")
    if args.overwrite:
        for path in [*existing, *existing_pdbs]:
            path.unlink()
    output_directory.mkdir(parents=True, exist_ok=True)

    reference_lines, reference_atoms = parse_pdb(input_paths["reference_pdb"])
    del reference_lines
    reference_chain = str(config["reference_chain"])
    mobile_chain = str(config["mobile_chain"])
    pocket_numbers = [int(value) for value in config["pocket_residue_numbers"]]
    expected_residue_count = int(config["expected_mobile_residue_count"])
    output_rows: list[dict[str, object]] = []
    for row in medoids:
        source_path = Path(row["pdb_path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_hash = file_sha256(source_path)
        if source_hash != row["pdb_sha256"].upper():
            raise ValueError(f"source medoid hash differs for {row['conformer_id']}")
        mobile_lines, mobile_atoms = parse_pdb(source_path)
        reference_coords, mobile_coords, residue_mismatches = match_ca_coordinates(
            reference_atoms,
            mobile_atoms,
            reference_chain,
            mobile_chain,
        )
        rotation, translation = calculate_kabsch_transform(mobile_coords, reference_coords)
        aligned_matched = transform_coordinates(mobile_coords, rotation, translation)
        output_path = output_directory / f"{row['conformer_id']}_to_1AQ1_A_heavy.pdb"
        write_aligned_heavy_protein(
            output_path, mobile_lines, mobile_atoms, mobile_chain, rotation, translation
        )
        _, aligned_atoms = parse_pdb(output_path)
        audit = audit_heavy_protein(output_path, mobile_chain)
        if int(audit["residue_count"]) != expected_residue_count:
            raise RuntimeError(
                f"aligned residue count differs for {row['conformer_id']}: {audit['residue_count']}"
            )
        output_rows.append({
            **row,
            "temporal_support_role": temporal_support_role(row),
            "source_medoid_pdb_sha256": source_hash,
            "aligned_heavy_pdb_path": output_path.as_posix(),
            "aligned_heavy_pdb_sha256": file_sha256(output_path),
            "matched_ca_count": len(reference_coords),
            "residue_name_mismatch_count": len(residue_mismatches),
            "rmsd_before_alignment_angstrom": round(rmsd(mobile_coords, reference_coords), 6),
            "rmsd_after_alignment_angstrom": round(rmsd(aligned_matched, reference_coords), 6),
            "pocket_ca_rmsd_to_1AQ1_angstrom": round(
                pocket_ca_rmsd(
                    reference_atoms,
                    aligned_atoms,
                    reference_chain,
                    mobile_chain,
                    pocket_numbers,
                ),
                6,
            ),
            "rotation_determinant": round(float(np.linalg.det(rotation)), 12),
            "aligned_heavy_atom_count": audit["atom_record_count"],
            "aligned_residue_count": audit["residue_count"],
            "aligned_hydrogen_count": audit["hydrogen_count"],
            "aligned_hetatm_count": audit["hetatm_record_count"],
            "alignment_status": "ok",
        })

    write_csv(manifest_path, output_rows)
    after_values = np.array(
        [float(row["rmsd_after_alignment_angstrom"]) for row in output_rows]
    )
    pocket_values = np.array(
        [float(row["pocket_ca_rmsd_to_1AQ1_angstrom"]) for row in output_rows]
    )
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "clustering_experiment_id": config["clustering_experiment_id"],
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "inputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in input_paths.items()
        },
        "medoid_count": len(output_rows),
        "primary_revisited_candidate_count": sum(
            row["temporal_support_role"] == "revisited_primary_candidate"
            for row in output_rows
        ),
        "exploratory_low_temporal_support_count": sum(
            row["temporal_support_role"] == "exploratory_low_temporal_support"
            for row in output_rows
        ),
        "matched_ca_count_values": sorted(
            set(int(row["matched_ca_count"]) for row in output_rows)
        ),
        "aligned_heavy_atom_count_values": sorted(
            set(int(row["aligned_heavy_atom_count"]) for row in output_rows)
        ),
        "aligned_residue_count_values": sorted(
            set(int(row["aligned_residue_count"]) for row in output_rows)
        ),
        "rmsd_after_alignment_angstrom": {
            "minimum": round(float(after_values.min()), 6),
            "maximum": round(float(after_values.max()), 6),
        },
        "pocket_ca_rmsd_to_1AQ1_angstrom": {
            "minimum": round(float(pocket_values.min()), 6),
            "maximum": round(float(pocket_values.max()), 6),
        },
        "outputs": {
            "aligned_heavy_directory": output_directory.as_posix(),
            "alignment_manifest_csv": manifest_path.as_posix(),
            "alignment_manifest_sha256": file_sha256(manifest_path),
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
