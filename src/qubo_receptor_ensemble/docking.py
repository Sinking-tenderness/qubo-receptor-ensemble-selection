"""Docking output parsing and redocking RMSD evaluation.

Consolidated from ``scripts/evaluate_redocking_rmsd.py``,
``scripts/batch_vina_docking.py``, and
``scripts/batch_vina_docking_parallel.py``; behavior is identical to the
originals.
"""

from __future__ import annotations

import csv
import random
import re
import subprocess
from pathlib import Path

from qubo_receptor_ensemble.io import safe_filename

VINA_RESULT_PATTERN = re.compile(
    r"^REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", re.MULTILINE
)

REQUIRED_COLUMNS = {"ligand_id", "label", "target_id", "pdbqt_status", "pdbqt_path"}
VINA_CONFIG_KEYS = {
    "center_x",
    "center_y",
    "center_z",
    "size_x",
    "size_y",
    "size_z",
    "exhaustiveness",
    "num_modes",
}


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


def validate_columns(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("input manifest has no header")
    missing = REQUIRED_COLUMNS.difference(fieldnames)
    if missing:
        raise ValueError(f"input manifest is missing required columns: {sorted(missing)}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames)
        return list(reader)


def read_vina_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", maxsplit=1)]
            values[key] = value
    missing = VINA_CONFIG_KEYS.difference(values)
    if missing:
        raise ValueError(f"Vina config is missing required keys: {sorted(missing)}")
    return values


def select_rows(
    rows: list[dict[str, str]],
    max_ligands: int | None,
    sample_per_label: int | None,
    sample_seed: int,
) -> list[dict[str, str]]:
    ok_rows = [row for row in rows if row["pdbqt_status"] == "ok"]
    if sample_per_label is not None:
        rng = random.Random(sample_seed)
        selected: list[dict[str, str]] = []
        for label in sorted({row["label"] for row in ok_rows}):
            label_rows = [row for row in ok_rows if row["label"] == label]
            if sample_per_label > len(label_rows):
                raise ValueError(
                    f"requested {sample_per_label} {label} rows, but only {len(label_rows)} are available"
                )
            selected.extend(rng.sample(label_rows, sample_per_label))
        return sorted(selected, key=lambda row: row["ligand_id"])
    if max_ligands is not None:
        return ok_rows[:max_ligands]
    return ok_rows


def get_vina_version(vina_exe: Path) -> str:
    completed = subprocess.run(
        [str(vina_exe), "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    version = (completed.stdout or completed.stderr).strip()
    return version.replace("\n", " ")


def parse_vina_modes(stdout: str) -> list[dict[str, object]]:
    modes: list[dict[str, object]] = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            pose_rank = int(parts[0])
            docking_score = float(parts[1])
            rmsd_lb = float(parts[2])
            rmsd_ub = float(parts[3])
        except ValueError:
            continue
        modes.append(
            {
                "pose_rank": pose_rank,
                "docking_score": docking_score,
                "vina_rmsd_lb_from_best": rmsd_lb,
                "vina_rmsd_ub_from_best": rmsd_ub,
            }
        )
    return modes


def result_rows_for_modes(
    row: dict[str, str],
    receptor_id: str,
    modes: list[dict[str, object]],
    status: str,
    message: str,
    runtime_seconds: float | str,
    seed: int,
    software_version: str,
    output_pose: Path,
    log_path: Path,
) -> list[dict[str, object]]:
    return [
        {
            "target_id": row["target_id"],
            "receptor_id": receptor_id,
            "ligand_id": row["ligand_id"],
            "label": row["label"],
            **mode,
            "status": status,
            "message": message,
            "runtime_seconds": runtime_seconds,
            "seed": seed,
            "software_version": software_version,
            "pose_path": output_pose.as_posix(),
            "log_path": log_path.as_posix(),
        }
        for mode in modes
    ]


def build_vina_command(
    vina_exe: Path,
    receptor: Path,
    ligand: Path,
    output_pose: Path,
    config: dict[str, str],
    seed: int,
) -> list[str]:
    return [
        str(vina_exe),
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
        "--center_x",
        config["center_x"],
        "--center_y",
        config["center_y"],
        "--center_z",
        config["center_z"],
        "--size_x",
        config["size_x"],
        "--size_y",
        config["size_y"],
        "--size_z",
        config["size_z"],
        "--exhaustiveness",
        config["exhaustiveness"],
        "--num_modes",
        config["num_modes"],
        *( ["--cpu", config["cpu"]] if config.get("cpu") else [] ),
        "--seed",
        str(seed),
        "--out",
        str(output_pose),
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows as CSV; unlike ``qubo_receptor_ensemble.io.write_csv`` this
    variant accepts an empty row list (header-only output)."""
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


def replace_ligand_rows(
    rows: list[dict[str, object]], ligand_id: str, replacement: list[dict[str, object]]
) -> list[dict[str, object]]:
    return [row for row in rows if row.get("ligand_id") != ligand_id] + replacement


def ligand_seed(row: dict[str, str], index: int, base_seed: int) -> int:
    value = row.get("seed_offset", "").strip()
    offset = int(value) if value else index
    if offset < 0:
        raise ValueError(f"negative seed offset for {row['ligand_id']}")
    return base_seed + offset
