"""Clean, parameterize, and audit a rigid receptor for AutoDock Vina."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def coordinate_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="ascii").splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]


def residue_count(lines: list[str]) -> int:
    residues = {
        (line[21:22], line[22:26], line[26:27])
        for line in lines
        if len(line) >= 27
    }
    return len(residues)


def audit_pdb(path: Path) -> dict[str, object]:
    lines = coordinate_lines(path)
    hydrogen_count = 0
    for line in lines:
        element = line[76:78].strip() if len(line) >= 78 else ""
        atom_name = line[12:16].strip() if len(line) >= 16 else ""
        if element == "H" or (not element and atom_name.startswith("H")):
            hydrogen_count += 1
    return {
        "coordinate_record_count": len(lines),
        "atom_record_count": sum(line.startswith("ATOM  ") for line in lines),
        "hetatm_record_count": sum(line.startswith("HETATM") for line in lines),
        "residue_count": residue_count(lines),
        "hydrogen_count": hydrogen_count,
    }


def audit_pdbqt(path: Path) -> dict[str, object]:
    lines = coordinate_lines(path)
    charges: list[float] = []
    atom_types: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if len(line) < 78:
            raise ValueError(f"PDBQT coordinate line {line_number} is too short")
        try:
            charges.append(float(line[70:76].strip()))
        except ValueError as exc:
            raise ValueError(
                f"invalid PDBQT charge on coordinate line {line_number}: {line[70:76]!r}"
            ) from exc
        atom_types.add(line[77:].strip())

    if not lines:
        raise ValueError(f"no PDBQT coordinate records found in {path}")
    return {
        "coordinate_record_count": len(lines),
        "atom_record_count": sum(line.startswith("ATOM  ") for line in lines),
        "hetatm_record_count": sum(line.startswith("HETATM") for line in lines),
        "residue_count": residue_count(lines),
        "hydrogen_like_atom_count": sum(
            line[77:].strip().startswith("H") for line in lines
        ),
        "charge_min": min(charges),
        "charge_max": max(charges),
        "autodock_atom_types": sorted(atom_types),
    }


def clean_protein_with_prody(
    input_pdb: Path,
    output_pdb: Path,
    chain: str,
    prody_altloc: str,
) -> dict[str, object]:
    from prody import parsePDB, writePDB

    atoms = parsePDB(str(input_pdb), altloc=prody_altloc)
    if atoms is None:
        raise ValueError(f"ProDy could not parse {input_pdb}")
    protein = atoms.select(f"chain {chain} and not water and not hetero")
    if protein is None or protein.numAtoms() == 0:
        raise ValueError(f"no protein atoms selected for chain {chain}")

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    writePDB(str(output_pdb), protein)
    selected_residues = sum(1 for _ in protein.getHierView().iterResidues())
    return {
        "parsed_atom_count": int(atoms.numAtoms()),
        "selected_protein_atom_count": int(protein.numAtoms()),
        "selected_residue_count": selected_residues,
        "selected_chains": sorted(set(protein.getChids())),
    }


def check_output_paths(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output files already exist; use --overwrite: {formatted}")
    if overwrite:
        for path in existing:
            path.unlink()


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pdb", type=Path, required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--prody-altloc", default="A")
    parser.add_argument(
        "--wanted-altloc",
        help="Explicit Meeko choices such as A:131=A,A:264=A",
    )
    parser.add_argument(
        "--meeko-default-altloc",
        help="Default alternate location passed to Meeko, e.g. A or 1",
    )
    parser.add_argument(
        "--allow-bad-res",
        action="store_true",
        help="Ask Meeko to delete residues with incomplete templates; audit before use",
    )
    parser.add_argument("--protein-only-output", type=Path, required=True)
    parser.add_argument("--prepared-pdb-output", type=Path, required=True)
    parser.add_argument("--pdbqt-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--charge-model",
        choices=["gasteiger", "espaloma", "zero"],
        default="gasteiger",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input_pdb.is_file():
        raise FileNotFoundError(args.input_pdb)
    if len(args.chain) != 1:
        raise ValueError("--chain must contain exactly one PDB chain identifier")

    output_paths = [
        args.protein_only_output,
        args.prepared_pdb_output,
        args.pdbqt_output,
        args.summary_output,
    ]
    check_output_paths(output_paths, args.overwrite)

    summary: dict[str, object] = {
        "status": "running",
        "input_pdb": str(args.input_pdb),
        "input_sha256": file_sha256(args.input_pdb),
        "chain": args.chain,
        "prody_altloc": args.prody_altloc,
        "wanted_altloc": args.wanted_altloc,
        "meeko_default_altloc": args.meeko_default_altloc,
        "allow_bad_res": args.allow_bad_res,
        "charge_model": args.charge_model,
        "python_version": sys.version.split()[0],
        "meeko_version": importlib.metadata.version("meeko"),
        "prody_version": importlib.metadata.version("prody"),
    }

    try:
        summary["cleaning"] = clean_protein_with_prody(
            args.input_pdb,
            args.protein_only_output,
            args.chain,
            args.prody_altloc,
        )

        output_basename = args.pdbqt_output.with_suffix("")
        args.prepared_pdb_output.parent.mkdir(parents=True, exist_ok=True)
        args.pdbqt_output.parent.mkdir(parents=True, exist_ok=True)
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
            args.charge_model,
        ]
        if args.wanted_altloc:
            command.extend(["--wanted_altloc", args.wanted_altloc])
        if args.meeko_default_altloc:
            command.extend(["--default_altloc", args.meeko_default_altloc])
        if args.allow_bad_res:
            command.append("--allow_bad_res")
        summary["meeko_command"] = command

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        summary["meeko_return_code"] = completed.returncode
        summary["meeko_stdout"] = completed.stdout
        summary["meeko_stderr"] = completed.stderr
        if completed.returncode != 0:
            raise RuntimeError(f"Meeko failed with return code {completed.returncode}")
        if not args.prepared_pdb_output.is_file() or not args.pdbqt_output.is_file():
            raise RuntimeError("Meeko returned success but required output files are missing")

        protein_only_audit = audit_pdb(args.protein_only_output)
        prepared_pdb_audit = audit_pdb(args.prepared_pdb_output)
        pdbqt_audit = audit_pdbqt(args.pdbqt_output)
        if protein_only_audit["hetatm_record_count"] != 0:
            raise RuntimeError("protein-only PDB still contains HETATM records")
        if pdbqt_audit["hetatm_record_count"] != 0:
            raise RuntimeError("receptor PDBQT contains HETATM records")
        if (
            pdbqt_audit["residue_count"] != protein_only_audit["residue_count"]
            and not args.allow_bad_res
        ):
            raise RuntimeError("residue count changed during receptor parameterization")
        summary["residue_count_change"] = {
            "input_protein_only": protein_only_audit["residue_count"],
            "output_pdbqt": pdbqt_audit["residue_count"],
            "allowed_by_allow_bad_res": args.allow_bad_res,
        }

        summary["outputs"] = {
            "protein_only_pdb": {
                "path": str(args.protein_only_output),
                "sha256": file_sha256(args.protein_only_output),
                "audit": protein_only_audit,
            },
            "prepared_pdb": {
                "path": str(args.prepared_pdb_output),
                "sha256": file_sha256(args.prepared_pdb_output),
                "audit": prepared_pdb_audit,
            },
            "receptor_pdbqt": {
                "path": str(args.pdbqt_output),
                "sha256": file_sha256(args.pdbqt_output),
                "audit": pdbqt_audit,
            },
        }
        summary["status"] = "ok"
        write_summary(args.summary_output, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        write_summary(args.summary_output, summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
