"""Complete missing standard-residue heavy atoms before strict Meeko preparation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from openmm import unit
from openmm.app import PDBFile
from pdbfixer import PDBFixer

try:
    from scripts.prepare_receptor import audit_pdb, file_sha256, write_summary
except ModuleNotFoundError:
    from prepare_receptor import audit_pdb, file_sha256, write_summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input-pdb", type=Path, required=True)
    value.add_argument("--chain", default="A")
    value.add_argument("--completed-pdb-output", type=Path, required=True)
    value.add_argument("--protein-only-output", type=Path, required=True)
    value.add_argument("--prepared-pdb-output", type=Path, required=True)
    value.add_argument("--pdbqt-output", type=Path, required=True)
    value.add_argument("--summary-output", type=Path, required=True)
    value.add_argument("--overwrite", action="store_true")
    return value


def residue_key(residue) -> str:
    return f"{residue.chain.id}:{residue.id}:{residue.name}"


def missing_atom_name(value: object) -> str:
    name = getattr(value, "name", value)
    return str(name)


def atom_coordinate_map(path: Path) -> dict[tuple[str, str, str, str], np.ndarray]:
    pdb = PDBFile(str(path))
    coordinates = np.asarray(pdb.positions.value_in_unit(unit.angstrom), dtype=float)
    output: dict[tuple[str, str, str, str], np.ndarray] = {}
    for atom, coordinate in zip(pdb.topology.atoms(), coordinates):
        key = (atom.residue.chain.id, atom.residue.id, atom.residue.name, atom.name)
        if key in output:
            raise ValueError(f"duplicate atom identity prevents coordinate audit: {key}")
        output[key] = coordinate
    return output


def run(args: argparse.Namespace) -> int:
    outputs = (
        args.completed_pdb_output,
        args.protein_only_output,
        args.prepared_pdb_output,
        args.pdbqt_output,
        args.summary_output,
    )
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Stage50 receptor outputs exist; pass --overwrite")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    evidence: dict[str, object] = {
        "schema_version": "1.0",
        "status": "running",
        "protocol": "PDBFixer missing-heavy-atom completion followed by strict Meeko preparation",
        "input_pdb": str(args.input_pdb),
        "input_sha256": file_sha256(args.input_pdb),
        "chain": args.chain,
        "missing_residue_addition_permitted": False,
        "nonstandard_residue_replacement_permitted": False,
        "missing_heavy_atom_completion_permitted": True,
        "hydrogen_addition_by_pdbfixer_permitted": False,
        "residue_deletion_permitted": False,
        "allow_bad_res": False,
    }
    try:
        before = audit_pdb(args.input_pdb)
        before_coordinates = atom_coordinate_map(args.input_pdb)
        fixer = PDBFixer(filename=str(args.input_pdb))
        fixer.findMissingResidues()
        detected_missing_residues = {
            f"{chain_index}:{residue_index}": list(names)
            for (chain_index, residue_index), names in fixer.missingResidues.items()
        }
        fixer.missingResidues = {}
        fixer.findNonstandardResidues()
        nonstandard = [
            {"residue": residue_key(residue), "suggested_replacement": replacement}
            for residue, replacement in fixer.nonstandardResidues
        ]
        if nonstandard:
            raise ValueError("nonstandard protein residues require an unregistered correction")
        fixer.findMissingAtoms()
        missing_atoms = {
            residue_key(residue): sorted(missing_atom_name(atom) for atom in atoms)
            for residue, atoms in fixer.missingAtoms.items()
        }
        missing_terminals = {
            residue_key(residue): sorted(missing_atom_name(atom) for atom in atoms)
            for residue, atoms in fixer.missingTerminals.items()
        }
        fixer.addMissingAtoms()
        with args.completed_pdb_output.open("w", encoding="ascii", newline="\n") as handle:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
        after_completion = audit_pdb(args.completed_pdb_output)
        after_coordinates = atom_coordinate_map(args.completed_pdb_output)
        if after_completion["residue_count"] != before["residue_count"]:
            raise ValueError("PDBFixer changed the receptor residue count")
        if after_completion["hetatm_record_count"] != 0:
            raise ValueError("completed receptor contains HETATM records")
        if not set(before_coordinates).issubset(after_coordinates):
            raise ValueError("PDBFixer removed or renamed an existing receptor atom")
        maximum_existing_atom_displacement = max(
            float(np.linalg.norm(before_coordinates[key] - after_coordinates[key]))
            for key in before_coordinates
        )
        if maximum_existing_atom_displacement > 0.001:
            raise ValueError("PDBFixer moved an existing receptor atom above 0.001 A")

        parameterization_summary = args.summary_output.with_name(
            "strict_meeko_parameterization_summary.json"
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name("prepare_receptor.py")),
            "--input-pdb",
            str(args.completed_pdb_output),
            "--chain",
            args.chain,
            "--protein-only-output",
            str(args.protein_only_output),
            "--prepared-pdb-output",
            str(args.prepared_pdb_output),
            "--pdbqt-output",
            str(args.pdbqt_output),
            "--summary-output",
            str(parameterization_summary),
            "--overwrite",
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        evidence["meeko_wrapper_command"] = command
        evidence["meeko_wrapper_return_code"] = completed.returncode
        evidence["meeko_wrapper_stdout"] = completed.stdout
        evidence["meeko_wrapper_stderr"] = completed.stderr
        if completed.returncode != 0:
            raise RuntimeError(f"strict Meeko preparation failed with return code {completed.returncode}")
        parameterization = json.loads(parameterization_summary.read_text(encoding="ascii"))
        if parameterization.get("status") != "ok" or parameterization.get("allow_bad_res") is not False:
            raise ValueError("strict Meeko preparation evidence did not pass")
        residue_change = dict(parameterization["residue_count_change"])
        if residue_change["input_protein_only"] != residue_change["output_pdbqt"]:
            raise ValueError("Meeko changed the completed receptor residue count")

        evidence.update(
            {
                "status": "ok",
                "detected_missing_residues_not_added": detected_missing_residues,
                "nonstandard_residues": nonstandard,
                "completed_missing_atoms": missing_atoms,
                "completed_missing_terminal_atoms": missing_terminals,
                "completed_heavy_atom_count": sum(map(len, missing_atoms.values()))
                + sum(map(len, missing_terminals.values())),
                "maximum_existing_atom_displacement_angstrom": maximum_existing_atom_displacement,
                "residue_count_change": {
                    "input": before["residue_count"],
                    "after_completion": after_completion["residue_count"],
                    "output_pdbqt": residue_change["output_pdbqt"],
                },
                "strict_meeko_parameterization": {
                    "path": str(parameterization_summary),
                    "sha256": file_sha256(parameterization_summary),
                },
                "outputs": {
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
