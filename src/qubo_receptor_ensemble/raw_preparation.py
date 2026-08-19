"""Raw-data preparation for source-to-result receptor ensemble experiments.

The module deliberately owns only artifacts derived during the current run.
It never reads a prepared receptor or ligand manifest as an input source.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .io import file_sha256
from .pdb import (
    calculate_kabsch_transform,
    count_records,
    match_ca_coordinates,
    parse_pdb,
    rmsd,
    transform_coordinates,
    write_transformed_pdb,
)


def discover_structure_files(directory: Path) -> list[Path]:
    """Discover deterministic CIF/PDB candidates, preferring CIF over duplicate PDB."""
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    by_id: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".cif", ".pdb"}:
            continue
        structure_id = path.stem.upper()
        existing = by_id.get(structure_id)
        if existing is None or (
            path.suffix.lower() == ".cif" and existing.suffix.lower() != ".cif"
        ):
            by_id[structure_id] = path
    return [by_id[key] for key in sorted(by_id)]


def _require_gemmi():
    try:
        import gemmi
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "CIF preparation requires Gemmi in the active qubo-unidock environment"
        ) from exc
    return gemmi


def convert_structure_to_pdb(source: Path, output: Path) -> dict[str, object]:
    """Convert a raw CIF/PDB to a current-run PDB artifact."""
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".pdb":
        shutil.copyfile(source, output)
    elif source.suffix.lower() == ".cif":
        gemmi = _require_gemmi()
        structure = gemmi.read_structure(str(source))
        if len(structure) == 0:
            raise ValueError(f"CIF contains no models: {source}")
        structure[0].remove_alternative_conformations()
        structure.write_pdb(str(output))
    else:
        raise ValueError(f"unsupported receptor structure format: {source}")
    lines = output.read_text(encoding="ascii").splitlines()
    filtered_lines = [line for line in lines if not line.startswith("ANISOU")]
    anisou_removed_count = len(lines) - len(filtered_lines)
    if anisou_removed_count:
        output.write_text("\n".join(filtered_lines) + "\n", encoding="ascii")
    return {
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "output_path": str(output),
        "output_sha256": file_sha256(output),
        "format": source.suffix.lower().lstrip("."),
        "anisou_removed_count": anisou_removed_count,
    }


def _chain_names(atoms: list[object]) -> list[str]:
    return sorted({str(atom.chain) for atom in atoms if getattr(atom, "record") == "ATOM"})


def _matching_ca_count(
    reference_atoms: list[object], mobile_atoms: list[object], reference_chain: str, mobile_chain: str
) -> int:
    try:
        reference_coords, _, _ = match_ca_coordinates(
            reference_atoms,
            mobile_atoms,
            reference_chain,
            mobile_chain,
        )
    except ValueError:
        return 0
    return int(len(reference_coords))


def _select_alignment_chains(
    reference_atoms: list[object],
    mobile_atoms: list[object],
    reference_chain: str,
    mobile_chain: str,
) -> tuple[str, str, int]:
    reference_chains = (
        [reference_chain]
        if reference_chain not in {"", "auto"}
        else _chain_names(reference_atoms)
    )
    mobile_chains = (
        [mobile_chain] if mobile_chain not in {"", "auto"} else _chain_names(mobile_atoms)
    )
    candidates = [
        (count, ref, mob)
        for ref in reference_chains
        for mob in mobile_chains
        for count in [_matching_ca_count(reference_atoms, mobile_atoms, ref, mob)]
    ]
    if not candidates:
        raise ValueError("no protein chains are available for alignment")
    count, selected_reference, selected_mobile = max(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    if count < 3:
        raise ValueError(
            f"fewer than three sequence-matched C-alpha atoms are available: {count}"
        )
    return selected_reference, selected_mobile, count


def align_pdb_file(
    reference: Path,
    mobile: Path,
    output: Path,
    *,
    reference_chain: str = "auto",
    mobile_chain: str = "auto",
) -> dict[str, object]:
    """Align a mobile PDB to a reference and return an auditable summary."""
    reference_lines, reference_atoms = parse_pdb(reference)
    mobile_lines, mobile_atoms = parse_pdb(mobile)
    del reference_lines
    selected_reference_chain, selected_mobile_chain, matched_count = _select_alignment_chains(
        reference_atoms, mobile_atoms, reference_chain, mobile_chain
    )
    reference_coords, mobile_coords, residue_mismatches = match_ca_coordinates(
        reference_atoms,
        mobile_atoms,
        selected_reference_chain,
        selected_mobile_chain,
    )
    rotation, translation = calculate_kabsch_transform(mobile_coords, reference_coords)
    aligned_mobile_coords = transform_coordinates(mobile_coords, rotation, translation)
    write_transformed_pdb(output, mobile_lines, mobile_atoms, rotation, translation)
    _, output_atoms = parse_pdb(output)
    if count_records(output_atoms) != count_records(mobile_atoms):
        raise RuntimeError("coordinate record counts changed during alignment")
    return {
        "reference_path": str(reference),
        "reference_sha256": file_sha256(reference),
        "reference_chain": selected_reference_chain,
        "mobile_path": str(mobile),
        "mobile_sha256": file_sha256(mobile),
        "mobile_chain": selected_mobile_chain,
        "output_path": str(output),
        "output_sha256": file_sha256(output),
        "matched_ca_count": int(matched_count),
        "residue_name_mismatches_excluded": residue_mismatches,
        "rmsd_before_angstrom": rmsd(mobile_coords, reference_coords),
        "rmsd_after_angstrom": rmsd(aligned_mobile_coords, reference_coords),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "translation_vector_angstrom": translation.tolist(),
        "mobile_coordinate_record_counts": count_records(mobile_atoms),
        "output_coordinate_record_counts": count_records(output_atoms),
        "method": "sequence-matched C-alpha Kabsch rigid-body alignment",
    }


def _read_ligand_coordinates(path: Path) -> np.ndarray:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "box calculation requires RDKit in the active qubo-unidock environment"
        ) from exc
    suffix = path.suffix.lower()
    if suffix == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    elif suffix in {".sdf", ".mol"}:
        molecule = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
    else:
        raise ValueError(f"unsupported crystal ligand format: {path}")
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"crystal ligand has no usable coordinates: {path}")
    return np.asarray(molecule.GetConformer().GetPositions(), dtype=float)


def calculate_ligand_box(
    ligand_path: Path,
    *,
    padding: float,
    minimum_size: Sequence[float],
) -> dict[str, float]:
    """Compute a docking box from crystal-ligand bounds in its current frame."""
    if padding < 0:
        raise ValueError("box padding must be non-negative")
    if len(minimum_size) != 3 or any(float(value) <= 0 for value in minimum_size):
        raise ValueError("box minimum_size must contain three positive values")
    coordinates = _read_ligand_coordinates(ligand_path)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    center = (lower + upper) / 2.0
    size = np.maximum(upper - lower + 2.0 * float(padding), np.asarray(minimum_size, dtype=float))
    return {
        "center_x": round(float(center[0]), 6),
        "center_y": round(float(center[1]), 6),
        "center_z": round(float(center[2]), 6),
        "size_x": round(float(size[0]), 6),
        "size_y": round(float(size[1]), 6),
        "size_z": round(float(size[2]), 6),
    }


def _run_receptor_preparation(
    input_pdb: Path,
    output_directory: Path,
    receptor_id: str,
    chain: str,
    *,
    script_path: Path | None = None,
    allow_bad_res: bool = False,
) -> dict[str, object]:
    script = script_path or Path(__file__).resolve().parents[2] / "scripts" / "prepare_receptor.py"
    if not script.is_file():
        raise FileNotFoundError(f"receptor preparation script not found: {script}")
    output_directory.mkdir(parents=True, exist_ok=True)
    protein_only = output_directory / f"{receptor_id}_protein.pdb"
    prepared_pdb = output_directory / f"{receptor_id}_prepared.pdb"
    pdbqt = output_directory / f"{receptor_id}.pdbqt"
    summary_path = output_directory / f"{receptor_id}_prepare_summary.json"
    command = [
        sys.executable,
        str(script),
        "--input-pdb",
        str(input_pdb),
        "--chain",
        chain,
        "--protein-only-output",
        str(protein_only),
        "--prepared-pdb-output",
        str(prepared_pdb),
        "--pdbqt-output",
        str(pdbqt),
        "--summary-output",
        str(summary_path),
        "--overwrite",
    ]
    if allow_bad_res:
        command.append("--allow-bad-res")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        details = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
        raise RuntimeError(f"receptor preparation failed for {receptor_id}: {details[-1000:]}")
    if not pdbqt.is_file() or not prepared_pdb.is_file():
        raise RuntimeError(f"receptor preparation returned no PDB/PDBQT for {receptor_id}")
    summary: dict[str, object] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="ascii"))
    return {
        "prepared_pdb": prepared_pdb,
        "receptor_pdbqt": pdbqt,
        "preparation_summary": summary_path,
        "preparation": summary,
    }


def prepare_raw_receptors(
    *,
    reference_pdb: Path,
    rcsb_directory: Path,
    output_directory: Path,
    receptor_count: int,
    reference_chain: str = "auto",
    mobile_chain: str = "auto",
    minimum_alignment_ca_count: int = 50,
    allow_bad_res: bool = False,
    candidate_ids: Sequence[str] | None = None,
    prepare_receptor: Callable[..., dict[str, object]] | None = None,
) -> dict[str, object]:
    """Create and select current-run receptor PDB/PDBQT artifacts from raw structures."""
    if receptor_count <= 0:
        raise ValueError("receptor_count must be positive")
    if minimum_alignment_ca_count < 3:
        raise ValueError("minimum_alignment_ca_count must be at least three")
    candidates = discover_structure_files(rcsb_directory)
    requested = {str(value).upper() for value in (candidate_ids or [])}
    if requested:
        candidates = [path for path in candidates if path.stem.upper() in requested]
        found = {path.stem.upper() for path in candidates}
        missing = sorted(requested.difference(found))
        if missing:
            raise FileNotFoundError(f"configured RCSB structures are missing: {missing}")
    if not candidates:
        raise ValueError(f"no raw RCSB structures found in {rcsb_directory}")

    source_directory = output_directory / "receptors" / "source_pdb"
    aligned_directory = output_directory / "receptors" / "aligned_pdb"
    prepared_directory = output_directory / "receptors" / "prepared"
    source_directory.mkdir(parents=True, exist_ok=True)
    aligned_directory.mkdir(parents=True, exist_ok=True)
    prepared_directory.mkdir(parents=True, exist_ok=True)
    prepare_fn = prepare_receptor or _run_receptor_preparation
    records: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for source in candidates:
        receptor_id = source.stem.upper()
        source_pdb = source_directory / f"{receptor_id}.pdb"
        aligned_pdb = aligned_directory / f"{receptor_id}.pdb"
        record: dict[str, object] = {
            "conformer_id": receptor_id,
            "rcsb_id": receptor_id,
            "source_structure": str(source),
            "source_sha256": file_sha256(source),
            "status": "candidate",
        }
        try:
            conversion = convert_structure_to_pdb(source, source_pdb)
            record["source_pdb"] = str(source_pdb)
            record["conversion"] = conversion
            alignment = align_pdb_file(
                reference_pdb,
                source_pdb,
                aligned_pdb,
                reference_chain=reference_chain,
                mobile_chain=mobile_chain,
            )
            record["alignment"] = alignment
            if int(alignment["matched_ca_count"]) < minimum_alignment_ca_count:
                raise ValueError(
                    f"matched C-alpha count {alignment['matched_ca_count']} is below "
                    f"minimum {minimum_alignment_ca_count}"
                )
            prepared = prepare_fn(
                aligned_pdb,
                prepared_directory,
                receptor_id,
                str(alignment["mobile_chain"]),
                allow_bad_res=allow_bad_res,
            )
            record["receptor_pdb"] = str(aligned_pdb)
            record["receptor_pdbqt"] = str(prepared["receptor_pdbqt"])
            record["preparation"] = prepared.get("preparation", {})
            record["status"] = "ok"
            selected.append(record)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        if len(selected) == receptor_count:
            break
    if len(selected) < receptor_count:
        raise ValueError(
            f"raw RCSB preparation produced {len(selected)} usable receptors; "
            f"requested {receptor_count}. Candidate audit: {records}"
        )
    return {
        "selected": selected,
        "candidates": records,
        "candidate_count": len(records),
        "selected_count": len(selected),
    }
