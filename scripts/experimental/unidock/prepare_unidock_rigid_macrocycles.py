"""Re-prepare fixed consumed-train macrocycles for a Uni-Dock diagnostic."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import sys
from pathlib import Path

try:
    from scripts.batch_prepare_ligand_pdbqt import (
        file_sha256,
        find_meeko_script,
        parse_pdbqt,
        run_meeko,
        safe_filename,
        write_manifest,
    )
    from .run_unidock_gpu_equivalence import macrocycle_closure_atom_types
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.batch_prepare_ligand_pdbqt import (
        file_sha256,
        find_meeko_script,
        parse_pdbqt,
        run_meeko,
        safe_filename,
        write_manifest,
    )
    from run_unidock_gpu_equivalence import macrocycle_closure_atom_types


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def select_source_rows(
    rows: list[dict[str, str]], expected: list[dict[str, object]]
) -> list[tuple[int, dict[str, str], dict[str, object]]]:
    by_id = {row["ligand_id"]: (index, row) for index, row in enumerate(rows)}
    if len(by_id) != len(rows):
        raise ValueError("source ligand manifest contains duplicate IDs")
    selected: list[tuple[int, dict[str, str], dict[str, object]]] = []
    for descriptor in expected:
        ligand_id = str(descriptor["ligand_id"])
        if ligand_id not in by_id:
            raise ValueError(f"expected ligand is absent: {ligand_id}")
        index, row = by_id[ligand_id]
        if index != int(descriptor["source_manifest_index"]):
            raise ValueError(f"source manifest index differs: {ligand_id}")
        if row["label"] != descriptor["label"]:
            raise ValueError(f"source label differs: {ligand_id}")
        if row["pdbqt_sha256"].upper() != str(
            descriptor["source_pdbqt_sha256"]
        ).upper():
            raise ValueError(f"source PDBQT hash differs: {ligand_id}")
        selected.append((index, row, descriptor))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    source = config["input_manifest"]
    source_path = Path(str(source["path"]))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if file_sha256(source_path) != str(source["sha256"]).upper():
        raise ValueError("source ligand manifest SHA-256 differs")
    meeko_version = importlib.metadata.version("meeko")
    required_version = str(config["preparation"]["meeko_version"])
    if meeko_version != required_version:
        raise ValueError(
            f"Meeko version differs: expected {required_version}, got {meeko_version}"
        )
    selected = select_source_rows(read_csv(source_path), config["ligands"])
    output_manifest = Path(str(config["outputs"]["manifest_csv"]))
    output_summary = Path(str(config["outputs"]["summary_json"]))
    output_directory = Path(str(config["outputs"]["pdbqt_directory"]))
    if not args.overwrite and (output_manifest.exists() or output_summary.exists()):
        raise FileExistsError("rigid-macrocycle outputs exist; pass --overwrite")
    output_directory.mkdir(parents=True, exist_ok=True)
    meeko_script = find_meeko_script()
    output_rows: list[dict[str, object]] = []

    for source_index, row, _ in selected:
        ligand_id = row["ligand_id"]
        sdf_path = Path(row["sdf_path"])
        if not sdf_path.is_file():
            raise FileNotFoundError(sdf_path)
        output_path = output_directory / f"{safe_filename(ligand_id)}.pdbqt"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(output_path)
        completed = run_meeko(
            meeko_script,
            sdf_path,
            output_path,
            rigid_macrocycles=True,
        )
        if completed.returncode != 0 or not output_path.is_file():
            message = "\n".join(
                part.strip()
                for part in (completed.stdout, completed.stderr)
                if part.strip()
            )
            raise RuntimeError(f"Meeko failed for {ligand_id}: {message[-500:]}")
        pseudoatom_types = macrocycle_closure_atom_types(output_path)
        if pseudoatom_types:
            raise ValueError(
                f"rigid preparation retained closure pseudoatoms for {ligand_id}: "
                f"{pseudoatom_types}"
            )
        parsed = parse_pdbqt(output_path)
        output_rows.append(
            {
                **row,
                "source_manifest_index": source_index,
                "seed_offset": source_index,
                "source_pdbqt_path": row["pdbqt_path"],
                "source_pdbqt_sha256": row["pdbqt_sha256"],
                "sdf_sha256": file_sha256(sdf_path),
                "pdbqt_status": "ok",
                "pdbqt_message": "meeko_rigid_macrocycles_ok",
                "pdbqt_path": output_path.as_posix(),
                "pdbqt_sha256": file_sha256(output_path),
                **parsed,
                "macrocycle_closure_pseudoatom_types": "",
            }
        )

    write_manifest(output_manifest, output_rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "operation": "consumed-train Meeko rigid-macrocycle re-preparation only",
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "input_manifest": {
            "path": source_path.as_posix(),
            "sha256": file_sha256(source_path),
        },
        "meeko": {
            "version": meeko_version,
            "script": meeko_script.as_posix(),
            "rigid_macrocycles": True,
        },
        "ligand_count": len(output_rows),
        "label_counts": {"decoy": len(output_rows)},
        "seed_offsets": {
            str(row["ligand_id"]): int(row["seed_offset"])
            for row in output_rows
        },
        "closure_pseudoatom_ligand_count": 0,
        "outputs": {
            "manifest_csv": {
                "path": output_manifest.as_posix(),
                "sha256": file_sha256(output_manifest),
            },
            "pdbqt_files": [
                {
                    "ligand_id": row["ligand_id"],
                    "path": row["pdbqt_path"],
                    "sha256": row["pdbqt_sha256"],
                    "atom_types": row["pdbqt_atom_types"],
                    "torsdof": row["torsdof"],
                }
                for row in output_rows
            ],
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    write_json(output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
