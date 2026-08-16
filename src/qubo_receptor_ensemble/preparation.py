"""Ligand 3D SDF and PDBQT preparation helpers.

Consolidated from ``scripts/prepare_ligand_3d_sdf.py`` and
``scripts/batch_prepare_ligand_pdbqt.py``; behavior is identical to the
originals.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from qubo_receptor_ensemble.io import file_sha256, safe_filename

PREP_3D_REQUIRED_COLUMNS = {"ligand_id", "smiles", "label", "target_id"}
PDBQT_REQUIRED_COLUMNS = {"ligand_id", "label", "sdf_path", "prep_status"}


def validate_columns(
    fieldnames: list[str] | None, required: set[str], label: str
) -> None:
    if fieldnames is None:
        raise ValueError("input CSV has no header")
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"input {label} is missing required columns: {sorted(missing)}")


def read_rows(input_csv: Path, required: set[str], label: str) -> list[dict[str, str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames, required, label)
        return list(reader)


def write_manifest(output_manifest: Path, rows: list[dict[str, object]]) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_3d_mol(smiles: str, seed: int) -> tuple[Chem.Mol | None, str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "failed", "rdkit_parse_failed"

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status != 0:
        return None, "failed", f"embed_failed_code_{embed_status}"

    if AllChem.MMFFHasAllMoleculeParams(mol):
        optimize_status = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        method = "MMFF94"
    else:
        optimize_status = AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        method = "UFF"

    if optimize_status < 0:
        return None, "failed", f"{method}_optimize_failed_code_{optimize_status}"
    if optimize_status > 0:
        status = "warning"
        message = f"{method}_not_converged_code_{optimize_status}"
    else:
        status = "ok"
        message = f"{method}_converged"

    return mol, status, message


def parse_pdbqt(pdbqt_path: Path) -> dict[str, object]:
    atom_count = 0
    charges: list[float] = []
    atom_types: set[str] = set()
    torsdof = ""
    with pdbqt_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                atom_count += 1
                if len(line) >= 76:
                    try:
                        charges.append(float(line[70:76].strip()))
                    except ValueError:
                        pass
                if len(line) >= 78:
                    atom_types.add(line[77:].strip())
            elif line.startswith("TORSDOF"):
                parts = line.split()
                torsdof = parts[1] if len(parts) > 1 else ""
    return {
        "pdbqt_atom_count": atom_count,
        "pdbqt_atom_types": ";".join(sorted(atom_types)),
        "pdbqt_charge_min": min(charges) if charges else "",
        "pdbqt_charge_max": max(charges) if charges else "",
        "torsdof": torsdof,
    }


def validated_existing_pdbqt(pdbqt_path: Path) -> dict[str, object] | None:
    if not pdbqt_path.is_file():
        return None
    parsed = parse_pdbqt(pdbqt_path)
    if int(parsed["pdbqt_atom_count"]) <= 0 or parsed["torsdof"] == "":
        return None
    return {
        "pdbqt_status": "ok",
        "pdbqt_message": "meeko_existing_validated",
        "pdbqt_path": pdbqt_path.as_posix(),
        "pdbqt_sha256": file_sha256(pdbqt_path),
        **parsed,
    }


def find_meeko_script() -> Path:
    candidates = [
        Path(sys.prefix) / "Scripts" / "mk_prepare_ligand.py",
        Path(sys.prefix) / "bin" / "mk_prepare_ligand.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find mk_prepare_ligand.py under the active Python environment"
    )


def run_meeko(
    meeko_script: Path,
    sdf_path: Path,
    pdbqt_path: Path,
    rigid_macrocycles: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(meeko_script),
        "-i",
        str(sdf_path),
        "-o",
        str(pdbqt_path),
    ]
    if rigid_macrocycles:
        cmd.append("--rigid_macrocycles")
    return subprocess.run(cmd, text=True, capture_output=True, check=False)
