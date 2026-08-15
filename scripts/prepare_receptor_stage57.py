"""Prepare a PPARD receptor with strict completion and geometry-backed CYX typing."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from openmm import unit
from openmm.app import PDBFile
from pdbfixer import PDBFixer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_receptor import audit_pdb, audit_pdbqt, file_sha256, write_summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input-pdb", type=Path, required=True)
    value.add_argument("--chain", default="A")
    value.add_argument("--minimum-residue-number", type=int, required=True)
    value.add_argument("--maximum-residue-number", type=int, required=True)
    value.add_argument("--minimum-disulfide-distance", type=float, default=1.8)
    value.add_argument("--maximum-disulfide-distance", type=float, default=2.3)
    value.add_argument("--target-sequence-pdb-output", type=Path, required=True)
    value.add_argument("--completed-pdb-output", type=Path, required=True)
    value.add_argument("--protein-only-output", type=Path, required=True)
    value.add_argument("--prepared-pdb-output", type=Path, required=True)
    value.add_argument("--pdbqt-output", type=Path, required=True)
    value.add_argument("--summary-output", type=Path, required=True)
    value.add_argument("--overwrite", action="store_true")
    return value


def residue_number(line: str) -> int:
    return int(line[22:26].strip())


def filter_target_sequence(
    input_path: Path,
    output_path: Path,
    chain: str,
    minimum_residue: int,
    maximum_residue: int,
) -> dict[str, object]:
    kept: list[str] = []
    kept_residues: set[tuple[str, int, str]] = set()
    removed_residues: set[tuple[str, str, str]] = set()
    for line in input_path.read_text(encoding="ascii").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if not line.startswith("ATOM  ") or line[21:22] != chain:
            continue
        try:
            number = residue_number(line)
        except ValueError:
            removed_residues.add((line[21:22], line[22:26].strip(), line[26:27]))
            continue
        if minimum_residue <= number <= maximum_residue:
            kept.append(line)
            kept_residues.add((chain, number, line[26:27]))
        else:
            removed_residues.add((chain, str(number), line[26:27]))
    if not kept:
        raise ValueError("target-sequence filtering removed every receptor atom")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(kept + ["END"]) + "\n", encoding="ascii")
    return {
        "kept_atom_count": len(kept),
        "kept_residue_count": len(kept_residues),
        "removed_noncorresponding_residue_count": len(removed_residues),
        "removed_noncorresponding_residues": [
            f"{chain_id}:{number}{icode.strip()}"
            for chain_id, number, icode in sorted(removed_residues)
        ],
    }


def coordinate_map(path: Path) -> dict[tuple[str, str, str, str], np.ndarray]:
    pdb = PDBFile(str(path))
    coordinates = np.asarray(pdb.positions.value_in_unit(unit.angstrom), dtype=float)
    output: dict[tuple[str, str, str, str], np.ndarray] = {}
    for atom, coordinate in zip(pdb.topology.atoms(), coordinates):
        key = (atom.residue.chain.id, atom.residue.id, atom.residue.name, atom.name)
        if key in output:
            raise ValueError(f"duplicate atom identity prevents coordinate audit: {key}")
        output[key] = coordinate
    return output


def preexisting_disulfides(
    coordinates: dict[tuple[str, str, str, str], np.ndarray],
    minimum_distance: float,
    maximum_distance: float,
) -> list[dict[str, object]]:
    sulfurs = [
        (key, value)
        for key, value in coordinates.items()
        if key[2] == "CYS" and key[3] == "SG"
    ]
    candidates: list[tuple[float, tuple[str, str, str, str], tuple[str, str, str, str]]] = []
    for index, (first_key, first_coordinate) in enumerate(sulfurs):
        for second_key, second_coordinate in sulfurs[index + 1 :]:
            if first_key[0] != second_key[0]:
                continue
            distance = float(np.linalg.norm(first_coordinate - second_coordinate))
            if minimum_distance <= distance <= maximum_distance:
                candidates.append((distance, first_key, second_key))
    candidates.sort()
    used: set[tuple[str, str, str, str]] = set()
    output: list[dict[str, object]] = []
    for distance, first_key, second_key in candidates:
        if first_key in used or second_key in used:
            raise ValueError("ambiguous sulfur geometry prevents deterministic CYX typing")
        used.update((first_key, second_key))
        output.append(
            {
                "chain": first_key[0],
                "first_residue_id": first_key[1],
                "second_residue_id": second_key[1],
                "sg_distance_angstrom": distance,
            }
        )
    return output


def cyx_template(disulfides: list[dict[str, object]]) -> str:
    return ",".join(
        f"{value['chain']}:{value['first_residue_id']},{value['second_residue_id']}=CYX"
        for value in disulfides
    )


def missing_atom_name(value: object) -> str:
    return str(getattr(value, "name", value))


def run(args: argparse.Namespace) -> int:
    outputs = (
        args.target_sequence_pdb_output,
        args.completed_pdb_output,
        args.protein_only_output,
        args.prepared_pdb_output,
        args.pdbqt_output,
        args.summary_output,
    )
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Stage57 receptor outputs exist; pass --overwrite")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    evidence: dict[str, object] = {
        "schema_version": "1.0",
        "status": "running",
        "protocol": (
            "retain sequence-corresponding PPARD residues, complete missing standard-residue "
            "heavy atoms with PDBFixer, then run strict Meeko preparation"
        ),
        "input_pdb": str(args.input_pdb),
        "input_sha256": file_sha256(args.input_pdb),
        "chain": args.chain,
        "residue_number_window": [
            args.minimum_residue_number,
            args.maximum_residue_number,
        ],
        "missing_residue_addition_permitted": False,
        "nonstandard_residue_replacement_permitted": False,
        "missing_heavy_atom_completion_permitted": True,
        "hydrogen_addition_by_pdbfixer_permitted": False,
        "allow_bad_res": False,
    }
    try:
        filtering = filter_target_sequence(
            args.input_pdb,
            args.target_sequence_pdb_output,
            args.chain,
            args.minimum_residue_number,
            args.maximum_residue_number,
        )
        before = audit_pdb(args.target_sequence_pdb_output)
        before_coordinates = coordinate_map(args.target_sequence_pdb_output)
        disulfides = preexisting_disulfides(
            before_coordinates,
            args.minimum_disulfide_distance,
            args.maximum_disulfide_distance,
        )
        fixer = PDBFixer(filename=str(args.target_sequence_pdb_output))
        fixer.findMissingResidues()
        detected_missing_residues = {
            f"{chain_index}:{residue_index}": list(names)
            for (chain_index, residue_index), names in fixer.missingResidues.items()
        }
        fixer.missingResidues = {}
        fixer.findNonstandardResidues()
        nonstandard = [
            {
                "residue": f"{residue.chain.id}:{residue.id}:{residue.name}",
                "suggested_replacement": replacement,
            }
            for residue, replacement in fixer.nonstandardResidues
        ]
        if nonstandard:
            raise ValueError("nonstandard protein residues require an unregistered correction")
        fixer.findMissingAtoms()
        missing_atoms = {
            f"{residue.chain.id}:{residue.id}:{residue.name}": sorted(
                missing_atom_name(atom) for atom in atoms
            )
            for residue, atoms in fixer.missingAtoms.items()
        }
        missing_terminals = {
            f"{residue.chain.id}:{residue.id}:{residue.name}": sorted(
                missing_atom_name(atom) for atom in atoms
            )
            for residue, atoms in fixer.missingTerminals.items()
        }
        fixer.addMissingAtoms()
        with args.completed_pdb_output.open("w", encoding="ascii", newline="\n") as handle:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
        after = audit_pdb(args.completed_pdb_output)
        after_coordinates = coordinate_map(args.completed_pdb_output)
        if after["residue_count"] != before["residue_count"]:
            raise ValueError("PDBFixer changed the target-sequence residue count")
        if not set(before_coordinates).issubset(after_coordinates):
            raise ValueError("PDBFixer removed or renamed an existing receptor atom")
        maximum_displacement = max(
            float(np.linalg.norm(before_coordinates[key] - after_coordinates[key]))
            for key in before_coordinates
        )
        if maximum_displacement > 0.001:
            raise ValueError("PDBFixer moved an existing receptor atom above 0.001 A")

        args.protein_only_output.write_bytes(args.completed_pdb_output.read_bytes())
        output_basename = args.pdbqt_output.with_suffix("")
        command = [
            sys.executable,
            "-m",
            "meeko.cli.mk_prepare_receptor",
            "-i",
            str(args.protein_only_output),
            "-o",
            str(output_basename),
            "-p",
            "--write_pdb",
            str(args.prepared_pdb_output),
            "--charge_model",
            "gasteiger",
        ]
        template = cyx_template(disulfides)
        if template:
            command.extend(["--set_template", template])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        evidence.update(
            {
                "meeko_version": importlib.metadata.version("meeko"),
                "meeko_command": command,
                "meeko_return_code": completed.returncode,
                "meeko_stdout": completed.stdout,
                "meeko_stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(f"strict Meeko preparation failed with return code {completed.returncode}")
        pdbqt = audit_pdbqt(args.pdbqt_output)
        if pdbqt["residue_count"] != after["residue_count"]:
            raise ValueError("Meeko changed the completed receptor residue count")
        evidence.update(
            {
                "status": "ok",
                "target_sequence_filter": filtering,
                "detected_missing_residues_not_added": detected_missing_residues,
                "nonstandard_residues": nonstandard,
                "completed_missing_atoms": missing_atoms,
                "completed_missing_terminal_atoms": missing_terminals,
                "completed_heavy_atom_count": sum(map(len, missing_atoms.values()))
                + sum(map(len, missing_terminals.values())),
                "maximum_existing_atom_displacement_angstrom": maximum_displacement,
                "preexisting_disulfides": disulfides,
                "cyx_template": template,
                "residue_count_change": {
                    "target_sequence_input": before["residue_count"],
                    "after_completion": after["residue_count"],
                    "output_pdbqt": pdbqt["residue_count"],
                },
                "outputs": {
                    "target_sequence_pdb": {
                        "path": str(args.target_sequence_pdb_output),
                        "sha256": file_sha256(args.target_sequence_pdb_output),
                    },
                    "completed_pdb": {
                        "path": str(args.completed_pdb_output),
                        "sha256": file_sha256(args.completed_pdb_output),
                    },
                    "receptor_pdbqt": {
                        "path": str(args.pdbqt_output),
                        "sha256": file_sha256(args.pdbqt_output),
                    },
                },
            }
        )
        write_summary(args.summary_output, evidence)
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        evidence["status"] = "failed"
        evidence["error"] = f"{type(error).__name__}: {error}"
        write_summary(args.summary_output, evidence)
        raise


def main() -> int:
    args = parser().parse_args()
    if not args.input_pdb.is_file():
        raise FileNotFoundError(args.input_pdb)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
