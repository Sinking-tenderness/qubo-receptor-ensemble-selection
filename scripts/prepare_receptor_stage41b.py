"""Prepare a Stage41b receptor with an optional frozen residue template override."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_receptor import (
    audit_pdb,
    audit_pdbqt,
    check_output_paths,
    clean_protein_with_prody,
    file_sha256,
    write_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pdb", type=Path, required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--protein-only-output", type=Path, required=True)
    parser.add_argument("--prepared-pdb-output", type=Path, required=True)
    parser.add_argument("--pdbqt-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--charge-model", default="gasteiger")
    parser.add_argument("--set-template")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def delegate(args: argparse.Namespace) -> int:
    script = Path(__file__).with_name("prepare_receptor.py")
    command = [
        sys.executable,
        str(script),
        "--input-pdb",
        str(args.input_pdb),
        "--chain",
        args.chain,
        "--protein-only-output",
        str(args.protein_only_output),
        "--prepared-pdb-output",
        str(args.prepared_pdb_output),
        "--pdbqt-output",
        str(args.pdbqt_output),
        "--summary-output",
        str(args.summary_output),
        "--charge-model",
        args.charge_model,
    ]
    if args.overwrite:
        command.append("--overwrite")
    return subprocess.run(command, check=False).returncode


def prepare_with_template(args: argparse.Namespace) -> int:
    outputs = [
        args.protein_only_output,
        args.prepared_pdb_output,
        args.pdbqt_output,
        args.summary_output,
    ]
    check_output_paths(outputs, args.overwrite)
    summary: dict[str, object] = {
        "status": "running",
        "input_pdb": str(args.input_pdb),
        "input_sha256": file_sha256(args.input_pdb),
        "chain": args.chain,
        "prody_altloc": "A",
        "wanted_altloc": None,
        "meeko_default_altloc": None,
        "allow_bad_res": False,
        "charge_model": args.charge_model,
        "set_template": args.set_template,
        "python_version": sys.version.split()[0],
        "meeko_version": importlib.metadata.version("meeko"),
        "prody_version": importlib.metadata.version("prody"),
    }
    try:
        summary["cleaning"] = clean_protein_with_prody(
            args.input_pdb, args.protein_only_output, args.chain, "A"
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
            "--set_template",
            args.set_template,
        ]
        summary["meeko_command"] = command
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        summary["meeko_return_code"] = completed.returncode
        summary["meeko_stdout"] = completed.stdout
        summary["meeko_stderr"] = completed.stderr
        if completed.returncode != 0:
            raise RuntimeError(f"Meeko failed with return code {completed.returncode}")
        if not args.prepared_pdb_output.is_file() or not args.pdbqt_output.is_file():
            raise RuntimeError("Meeko returned success but required output files are missing")

        protein_audit = audit_pdb(args.protein_only_output)
        prepared_audit = audit_pdb(args.prepared_pdb_output)
        pdbqt_audit = audit_pdbqt(args.pdbqt_output)
        if protein_audit["hetatm_record_count"] != 0 or pdbqt_audit["hetatm_record_count"] != 0:
            raise RuntimeError("template-prepared receptor contains HETATM records")
        if pdbqt_audit["residue_count"] != protein_audit["residue_count"]:
            raise RuntimeError("residue count changed during template parameterization")
        summary["residue_count_change"] = {
            "input_protein_only": protein_audit["residue_count"],
            "output_pdbqt": pdbqt_audit["residue_count"],
            "allowed_by_allow_bad_res": False,
        }
        summary["outputs"] = {
            "protein_only_pdb": {
                "path": str(args.protein_only_output),
                "sha256": file_sha256(args.protein_only_output),
                "audit": protein_audit,
            },
            "prepared_pdb": {
                "path": str(args.prepared_pdb_output),
                "sha256": file_sha256(args.prepared_pdb_output),
                "audit": prepared_audit,
            },
            "receptor_pdbqt": {
                "path": str(args.pdbqt_output),
                "sha256": file_sha256(args.pdbqt_output),
                "audit": pdbqt_audit,
            },
        }
        summary["status"] = "ok"
        write_summary(args.summary_output, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        write_summary(args.summary_output, summary)
        raise


def main() -> int:
    args = build_parser().parse_args()
    if not args.input_pdb.is_file():
        raise FileNotFoundError(args.input_pdb)
    if args.set_template is None:
        return delegate(args)
    return prepare_with_template(args)


if __name__ == "__main__":
    raise SystemExit(main())
