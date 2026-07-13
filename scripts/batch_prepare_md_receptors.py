"""Apply one audited Meeko receptor-preparation protocol to all MD medoids."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    from .prepare_receptor import file_sha256
except ImportError:
    from prepare_receptor import file_sha256


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "alignment_experiment_id",
    "purpose",
    "inputs",
    "alignment_summary_sha256",
    "alignment_manifest_sha256",
    "pilot_conformer_id",
    "expected_receptor_count",
    "expected_residue_count",
    "chain",
    "charge_model",
    "expected_autodock_atom_types",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_INPUT_KEYS = {
    "alignment_summary",
    "alignment_manifest",
    "pilot_summary",
    "prepare_receptor_script",
}
REQUIRED_OUTPUT_KEYS = {"prepared_directory", "preparation_manifest_csv", "summary_json"}


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("batch receptor preparation config must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"batch receptor preparation config is missing keys: {', '.join(missing)}")
    inputs = config["inputs"]
    outputs = config["outputs"]
    if not isinstance(inputs, dict) or not REQUIRED_INPUT_KEYS.issubset(inputs):
        raise ValueError("inputs is missing one or more receptor preparation paths")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUT_KEYS.issubset(outputs):
        raise ValueError("outputs is missing one or more receptor preparation paths")
    if len(str(config["chain"])) != 1:
        raise ValueError("chain must be a one-character PDB chain ID")
    if int(config["expected_receptor_count"]) <= 0 or int(config["expected_residue_count"]) <= 0:
        raise ValueError("expected receptor and residue counts must be positive")
    atom_types = config["expected_autodock_atom_types"]
    if not isinstance(atom_types, list) or not atom_types or len(set(atom_types)) != len(atom_types):
        raise ValueError("expected_autodock_atom_types must be a non-empty unique list")
    if str(config["charge_model"]) not in {"gasteiger", "espaloma", "zero"}:
        raise ValueError("unsupported charge model")
    return config


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty receptor preparation manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def receptor_output_paths(directory: Path, conformer_id: str) -> dict[str, Path]:
    item_directory = directory / conformer_id
    return {
        "item_directory": item_directory,
        "protein_only_pdb": item_directory / f"{conformer_id}_protein_only.pdb",
        "prepared_pdb": item_directory / f"{conformer_id}_prepared.pdb",
        "receptor_pdbqt": item_directory / f"{conformer_id}_receptor.pdbqt",
        "item_summary": item_directory / "preparation_summary.json",
    }


def successful_manifest_row(
    alignment_row: dict[str, str],
    item_summary: dict[str, object],
    runtime_seconds: float,
    expected_residue_count: int,
    expected_atom_types: list[str],
) -> dict[str, object]:
    if item_summary.get("status") != "ok":
        raise ValueError("item preparation summary does not have status=ok")
    outputs = item_summary.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("item preparation summary is missing outputs")
    protein = outputs["protein_only_pdb"]
    prepared = outputs["prepared_pdb"]
    receptor = outputs["receptor_pdbqt"]
    if not all(isinstance(value, dict) for value in (protein, prepared, receptor)):
        raise ValueError("item output records must be JSON objects")
    protein_audit = protein["audit"]
    prepared_audit = prepared["audit"]
    receptor_audit = receptor["audit"]
    if not all(isinstance(value, dict) for value in (protein_audit, prepared_audit, receptor_audit)):
        raise ValueError("item output audits must be JSON objects")
    residue_count = int(receptor_audit["residue_count"])
    atom_types = sorted(str(value) for value in receptor_audit["autodock_atom_types"])
    if residue_count != expected_residue_count:
        raise ValueError(f"prepared receptor residue count differs: {residue_count}")
    if int(receptor_audit["hetatm_record_count"]) != 0:
        raise ValueError("prepared receptor PDBQT contains HETATM records")
    if atom_types != sorted(expected_atom_types):
        raise ValueError(f"AutoDock atom types differ: {atom_types}")
    return {
        "conformer_id": alignment_row["conformer_id"],
        "cluster_id": alignment_row["cluster_id"],
        "temporal_support_role": alignment_row["temporal_support_role"],
        "aligned_heavy_pdb_path": alignment_row["aligned_heavy_pdb_path"],
        "aligned_heavy_pdb_sha256": alignment_row["aligned_heavy_pdb_sha256"],
        "preparation_status": "ok",
        "preparation_message": "meeko_receptor_preparation_ok",
        "runtime_seconds": round(runtime_seconds, 3),
        "charge_model": item_summary["charge_model"],
        "meeko_version": item_summary["meeko_version"],
        "prody_version": item_summary["prody_version"],
        "protein_only_pdb_path": protein["path"],
        "protein_only_pdb_sha256": protein["sha256"],
        "protein_only_atom_count": protein_audit["atom_record_count"],
        "protein_only_hydrogen_count": protein_audit["hydrogen_count"],
        "prepared_pdb_path": prepared["path"],
        "prepared_pdb_sha256": prepared["sha256"],
        "prepared_pdb_atom_count": prepared_audit["atom_record_count"],
        "prepared_pdb_hydrogen_count": prepared_audit["hydrogen_count"],
        "receptor_pdbqt_path": receptor["path"],
        "receptor_pdbqt_sha256": receptor["sha256"],
        "receptor_pdbqt_atom_count": receptor_audit["atom_record_count"],
        "receptor_residue_count": residue_count,
        "receptor_hydrogen_like_atom_count": receptor_audit["hydrogen_like_atom_count"],
        "receptor_charge_min": receptor_audit["charge_min"],
        "receptor_charge_max": receptor_audit["charge_max"],
        "receptor_autodock_atom_types": ";".join(atom_types),
        "item_summary_path": str(outputs.get("summary_path", "")),
        "item_summary_sha256": "",
    }


def failure_manifest_row(
    alignment_row: dict[str, str], runtime_seconds: float, message: str, item_summary: Path
) -> dict[str, object]:
    return {
        "conformer_id": alignment_row["conformer_id"],
        "cluster_id": alignment_row["cluster_id"],
        "temporal_support_role": alignment_row["temporal_support_role"],
        "aligned_heavy_pdb_path": alignment_row["aligned_heavy_pdb_path"],
        "aligned_heavy_pdb_sha256": alignment_row["aligned_heavy_pdb_sha256"],
        "preparation_status": "failed",
        "preparation_message": message,
        "runtime_seconds": round(runtime_seconds, 3),
        "charge_model": "",
        "meeko_version": "",
        "prody_version": "",
        "protein_only_pdb_path": "",
        "protein_only_pdb_sha256": "",
        "protein_only_atom_count": "",
        "protein_only_hydrogen_count": "",
        "prepared_pdb_path": "",
        "prepared_pdb_sha256": "",
        "prepared_pdb_atom_count": "",
        "prepared_pdb_hydrogen_count": "",
        "receptor_pdbqt_path": "",
        "receptor_pdbqt_sha256": "",
        "receptor_pdbqt_atom_count": "",
        "receptor_residue_count": "",
        "receptor_hydrogen_like_atom_count": "",
        "receptor_charge_min": "",
        "receptor_charge_max": "",
        "receptor_autodock_atom_types": "",
        "item_summary_path": item_summary.as_posix() if item_summary.exists() else "",
        "item_summary_sha256": file_sha256(item_summary) if item_summary.exists() else "",
    }


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
    if file_sha256(input_paths["alignment_summary"]) != str(config["alignment_summary_sha256"]).upper():
        raise ValueError("alignment summary SHA-256 differs from the configured value")
    if file_sha256(input_paths["alignment_manifest"]) != str(config["alignment_manifest_sha256"]).upper():
        raise ValueError("alignment manifest SHA-256 differs from the configured value")

    alignment_summary = json.loads(input_paths["alignment_summary"].read_text(encoding="ascii"))
    if alignment_summary.get("status") != "ok":
        raise ValueError("alignment summary does not have status=ok")
    if alignment_summary.get("experiment_id") != config["alignment_experiment_id"]:
        raise ValueError("alignment experiment ID differs from preparation configuration")
    alignment_rows = read_csv(input_paths["alignment_manifest"])
    expected_count = int(config["expected_receptor_count"])
    if len(alignment_rows) != expected_count:
        raise ValueError(f"expected {expected_count} aligned receptors, got {len(alignment_rows)}")
    if any(row.get("alignment_status") != "ok" for row in alignment_rows):
        raise ValueError("alignment manifest contains a non-ok receptor")
    for row in alignment_rows:
        aligned_path = Path(row["aligned_heavy_pdb_path"])
        if not aligned_path.is_file():
            raise FileNotFoundError(aligned_path)
        if file_sha256(aligned_path) != row["aligned_heavy_pdb_sha256"].upper():
            raise ValueError(f"aligned receptor hash differs for {row['conformer_id']}")

    pilot_summary = json.loads(input_paths["pilot_summary"].read_text(encoding="ascii"))
    if pilot_summary.get("status") != "ok":
        raise ValueError("pilot receptor preparation does not have status=ok")
    pilot_id = str(config["pilot_conformer_id"])
    pilot_alignment_row = next(
        (row for row in alignment_rows if row["conformer_id"] == pilot_id), None
    )
    if pilot_alignment_row is None:
        raise ValueError("pilot conformer is absent from the alignment manifest")
    if pilot_summary.get("input_sha256") != pilot_alignment_row["aligned_heavy_pdb_sha256"]:
        raise ValueError("pilot input hash differs from the aligned receptor manifest")
    if pilot_summary.get("charge_model") != config["charge_model"]:
        raise ValueError("pilot charge model differs from batch configuration")

    prepared_directory = Path(str(outputs["prepared_directory"]))
    manifest_path = Path(str(outputs["preparation_manifest_csv"]))
    summary_path = Path(str(outputs["summary_json"]))
    expected_paths: list[Path] = [manifest_path, summary_path]
    for row in alignment_rows:
        paths = receptor_output_paths(prepared_directory, row["conformer_id"])
        expected_paths.extend(path for key, path in paths.items() if key != "item_directory")
    existing = [path for path in expected_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("batch receptor outputs exist; use --overwrite after review")

    expected_residue_count = int(config["expected_residue_count"])
    expected_atom_types = [str(value) for value in config["expected_autodock_atom_types"]]
    result_rows: list[dict[str, object]] = []
    for index, alignment_row in enumerate(alignment_rows, start=1):
        conformer_id = alignment_row["conformer_id"]
        paths = receptor_output_paths(prepared_directory, conformer_id)
        paths["item_directory"].mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(input_paths["prepare_receptor_script"]),
            "--input-pdb", alignment_row["aligned_heavy_pdb_path"],
            "--chain", str(config["chain"]),
            "--protein-only-output", str(paths["protein_only_pdb"]),
            "--prepared-pdb-output", str(paths["prepared_pdb"]),
            "--pdbqt-output", str(paths["receptor_pdbqt"]),
            "--summary-output", str(paths["item_summary"]),
            "--charge-model", str(config["charge_model"]),
        ]
        if args.overwrite:
            command.append("--overwrite")
        print(f"preparing receptor {index}/{len(alignment_rows)}: {conformer_id}", flush=True)
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        runtime_seconds = time.perf_counter() - started
        try:
            if completed.returncode != 0:
                raise RuntimeError(
                    f"prepare_receptor_return_code_{completed.returncode}: "
                    f"{completed.stderr.strip()[-500:]}"
                )
            item_summary = json.loads(paths["item_summary"].read_text(encoding="ascii"))
            if item_summary.get("input_sha256") != alignment_row["aligned_heavy_pdb_sha256"]:
                raise ValueError("item summary input hash differs from alignment manifest")
            if item_summary.get("charge_model") != config["charge_model"]:
                raise ValueError("item summary charge model differs from batch configuration")
            row = successful_manifest_row(
                alignment_row,
                item_summary,
                runtime_seconds,
                expected_residue_count,
                expected_atom_types,
            )
            row["item_summary_path"] = paths["item_summary"].as_posix()
            row["item_summary_sha256"] = file_sha256(paths["item_summary"])
            result_rows.append(row)
        except Exception as exc:
            result_rows.append(
                failure_manifest_row(
                    alignment_row,
                    runtime_seconds,
                    f"{type(exc).__name__}: {exc}",
                    paths["item_summary"],
                )
            )

    write_csv(manifest_path, result_rows)
    failed = [row for row in result_rows if row["preparation_status"] != "ok"]
    pilot_outputs = pilot_summary["outputs"]
    assert isinstance(pilot_outputs, dict)
    pilot_pdbqt_hash = pilot_outputs["receptor_pdbqt"]["sha256"]
    batch_pilot = next(row for row in result_rows if row["conformer_id"] == pilot_id)
    pilot_hash_matches = batch_pilot["receptor_pdbqt_sha256"] == pilot_pdbqt_hash
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "ok" if not failed and pilot_hash_matches
            else "partial_failure" if failed
            else "pilot_hash_mismatch"
        ),
        "alignment_experiment_id": config["alignment_experiment_id"],
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "inputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in input_paths.items()
        },
        "requested_receptor_count": len(alignment_rows),
        "successful_receptor_count": len(result_rows) - len(failed),
        "failed_receptor_count": len(failed),
        "failed_conformer_ids": [row["conformer_id"] for row in failed],
        "charge_model": config["charge_model"],
        "expected_residue_count": expected_residue_count,
        "expected_autodock_atom_types": sorted(expected_atom_types),
        "pilot_conformer_id": pilot_id,
        "pilot_pdbqt_sha256": pilot_pdbqt_hash,
        "batch_pilot_pdbqt_sha256": batch_pilot["receptor_pdbqt_sha256"],
        "batch_pilot_matches_pilot": pilot_hash_matches,
        "outputs": {
            "prepared_directory": prepared_directory.as_posix(),
            "preparation_manifest_csv": manifest_path.as_posix(),
            "preparation_manifest_sha256": file_sha256(manifest_path),
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if not failed and pilot_hash_matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
