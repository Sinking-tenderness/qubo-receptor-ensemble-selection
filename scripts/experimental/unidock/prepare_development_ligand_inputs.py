"""Prepare a frozen development ligand panel for Uni-Dock with checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem

from scripts.batch_prepare_ligand_pdbqt import (
    file_sha256,
    find_meeko_script,
    parse_pdbqt,
    run_meeko,
    safe_filename,
)
from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
    macrocycle_closure_atom_types,
)
from scripts.prepare_ligand_3d_sdf import build_3d_mol


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def verified(root: Path, descriptor: dict[str, Any]) -> Path:
    path = Path(str(descriptor["path"]))
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def row_signature(row: dict[str, str], config_sha256: str) -> str:
    payload = {
        "canonical_smiles": row["canonical_smiles"],
        "config_sha256": config_sha256,
        "label": row["label"],
        "ligand_id": row["ligand_id"],
        "source_smiles": row["source_smiles"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()


def validate_checkpoint(
    checkpoint_path: Path,
    row: dict[str, str],
    root: Path,
    config_sha256: str,
) -> dict[str, Any] | None:
    if not checkpoint_path.is_file():
        return None
    try:
        checkpoint = read_json(checkpoint_path)
        if checkpoint["row_signature"] != row_signature(row, config_sha256):
            return None
        result = dict(checkpoint["result"])
        if result["ligand_id"] != row["ligand_id"]:
            return None
        for path_key, hash_key in (
            ("sdf_path", "sdf_sha256"),
            ("pdbqt_path", "pdbqt_sha256"),
        ):
            path = root / str(result[path_key])
            if not path.is_file() or file_sha256(path) != str(result[hash_key]).upper():
                return None
        pdbqt = root / str(result["pdbqt_path"])
        audit = parse_pdbqt(pdbqt)
        if int(audit["pdbqt_atom_count"]) <= 0 or audit["torsdof"] == "":
            return None
        if macrocycle_closure_atom_types(pdbqt):
            return None
        result["resume_status"] = "validated_checkpoint"
        return result
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def error_message(stdout: str, stderr: str) -> str:
    return "\n".join(
        part.strip() for part in (stdout, stderr) if part.strip()
    )[-1000:]


def prepare_one(task: dict[str, Any]) -> dict[str, Any]:
    row = dict(task["row"])
    root = Path(str(task["root"]))
    sdf_directory = Path(str(task["sdf_directory"]))
    pdbqt_directory = Path(str(task["pdbqt_directory"]))
    checkpoint_directory = Path(str(task["checkpoint_directory"]))
    meeko_script = Path(str(task["meeko_script"]))
    index = int(task["index"])
    resume = bool(task["resume"])
    overwrite = bool(task["overwrite"])
    base_seed = int(task["base_seed"])
    offsets = [int(value) for value in task["seed_offsets"]]
    target_id = str(task["target_id"])
    config_sha256 = str(task["config_sha256"])
    ligand_id = str(row["ligand_id"])
    smiles = str(row["canonical_smiles"])
    safe_id = safe_filename(ligand_id)
    sdf_path = sdf_directory / f"{safe_id}.sdf"
    pdbqt_path = pdbqt_directory / f"{safe_id}.pdbqt"
    checkpoint_path = checkpoint_directory / f"{safe_id}.json"
    for directory in (sdf_directory, pdbqt_directory, checkpoint_directory):
        directory.mkdir(parents=True, exist_ok=True)

    if resume:
        checkpoint = validate_checkpoint(
            checkpoint_path, row, root, config_sha256
        )
        if checkpoint is not None:
            return checkpoint
    elif not overwrite and any(
        path.exists() for path in (sdf_path, pdbqt_path, checkpoint_path)
    ):
        raise FileExistsError(f"existing output for {ligand_id}; use --resume")
    for path in (sdf_path, pdbqt_path, checkpoint_path):
        if path.exists():
            path.unlink()

    molecule = None
    sdf_status = ""
    sdf_message = ""
    selected_seed = 0
    attempt_count = 0
    for offset in offsets:
        attempt_count += 1
        selected_seed = base_seed + index + offset
        molecule, sdf_status, sdf_message = build_3d_mol(smiles, selected_seed)
        if molecule is not None:
            break
    if molecule is None:
        raise RuntimeError(
            f"RDKit 3D preparation failed for {ligand_id}: {sdf_message}"
        )
    molecule.SetProp("_Name", ligand_id)
    molecule.SetProp("ligand_id", ligand_id)
    molecule.SetProp("source_smiles", str(row["source_smiles"]))
    molecule.SetProp("preparation_smiles", smiles)
    molecule.SetProp("label", str(row["label"]))
    molecule.SetProp("target_id", target_id)
    molecule.SetProp("rdkit_embed_seed", str(selected_seed))

    with tempfile.TemporaryDirectory(
        prefix=f"{target_id.lower()}_development_"
    ) as temporary:
        staging = Path(temporary)
        staged_sdf = staging / "ligand.sdf"
        staged_pdbqt = staging / "ligand.pdbqt"
        writer = Chem.SDWriter(str(staged_sdf))
        writer.write(molecule)
        writer.close()
        if not staged_sdf.is_file():
            raise RuntimeError(f"RDKit did not write SDF for {ligand_id}")

        flexible = run_meeko(
            meeko_script, staged_sdf, staged_pdbqt, rigid_macrocycles=False
        )
        flexible_ok = flexible.returncode == 0 and staged_pdbqt.is_file()
        pseudoatoms = (
            macrocycle_closure_atom_types(staged_pdbqt) if flexible_ok else []
        )
        variant = "meeko_flexible"
        message = "meeko_flexible_ok"
        if not flexible_ok or pseudoatoms:
            if staged_pdbqt.exists():
                staged_pdbqt.unlink()
            rigid = run_meeko(
                meeko_script, staged_sdf, staged_pdbqt, rigid_macrocycles=True
            )
            if rigid.returncode != 0 or not staged_pdbqt.is_file():
                details = error_message(
                    flexible.stdout + "\n" + rigid.stdout,
                    flexible.stderr + "\n" + rigid.stderr,
                )
                raise RuntimeError(
                    f"Meeko preparation failed for {ligand_id}: {details}"
                )
            variant = "meeko_rigid_macrocycles"
            message = (
                "meeko_rigid_after_closure_pseudoatom_detection"
                if pseudoatoms
                else "meeko_rigid_after_flexible_failure"
            )
        remaining = macrocycle_closure_atom_types(staged_pdbqt)
        if remaining:
            raise ValueError(
                f"closure pseudoatoms remain for {ligand_id}: {remaining}"
            )
        pdbqt_audit = parse_pdbqt(staged_pdbqt)
        shutil.copy2(staged_sdf, sdf_path)
        shutil.copy2(staged_pdbqt, pdbqt_path)

    if int(pdbqt_audit["pdbqt_atom_count"]) <= 0 or pdbqt_audit["torsdof"] == "":
        raise ValueError(f"invalid PDBQT for {ligand_id}")
    result = {
        **row,
        "source_manifest_index": index,
        "seed_offset": index,
        "prep_status": sdf_status,
        "prep_message": sdf_message,
        "rdkit_embed_seed": selected_seed,
        "rdkit_embed_attempt_count": attempt_count,
        "sdf_path": sdf_path.relative_to(root).as_posix(),
        "sdf_atom_count": molecule.GetNumAtoms(),
        "sdf_heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "sdf_sha256": file_sha256(sdf_path),
        "pdbqt_status": "ok",
        "pdbqt_message": message,
        "pdbqt_path": pdbqt_path.relative_to(root).as_posix(),
        "pdbqt_sha256": file_sha256(pdbqt_path),
        "preparation_variant": variant,
        "resume_status": "prepared_this_invocation",
        **pdbqt_audit,
    }
    write_json(
        checkpoint_path,
        {
            "schema_version": "1.0",
            "row_signature": row_signature(row, config_sha256),
            "result": result,
        },
    )
    return result


def validate_source(
    root: Path, config: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Path], Counter[str]]:
    inputs = {
        key: verified(root, dict(value))
        for key, value in dict(config["inputs"]).items()
    }
    allocation = read_json(inputs["panel_allocation_summary"])
    source = dict(config["source"])
    if allocation["status"] != source["required_allocation_status"]:
        raise ValueError("development panel allocation did not pass")
    realized = dict(allocation["outputs"])["train_manifest_csv"]
    if realized["path"] != inputs["train_manifest"].relative_to(root).as_posix():
        raise ValueError("realized train-manifest path differs")
    if realized["sha256"].upper() != file_sha256(inputs["train_manifest"]):
        raise ValueError("realized train-manifest hash differs")
    for key in source["required_zero_allocation_boundaries"]:
        if int(allocation["data_boundary"][key]) != 0:
            raise ValueError(f"allocation crossed score boundary: {key}")
    rows = read_csv(inputs["train_manifest"])
    expected = dict(config["expected"])
    if len(rows) != int(expected["ligand_count"]):
        raise ValueError("development ligand count differs")
    labels = Counter(row["label"] for row in rows)
    frozen_labels = Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    )
    if labels != frozen_labels:
        raise ValueError("development label counts differ")
    if {row["split"] for row in rows} != {source["required_split"]}:
        raise ValueError("development manifest exposed another split")
    if {row["selection_role"] for row in rows} != {
        source["required_selection_role"]
    }:
        raise ValueError("development selection role differs")
    if len({row["ligand_id"] for row in rows}) != len(rows):
        raise ValueError("development manifest contains duplicate ligand IDs")
    return rows, inputs, labels


def run(
    config_path: Path,
    root: Path,
    audit_only: bool,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if file_sha256(Path(__file__)) != str(
        implementation["preparer"]["sha256"]
    ).upper():
        raise ValueError("development ligand preparer SHA-256 differs")
    for key in ("three_d_helper", "meeko_helper", "pseudoatom_audit_helper"):
        verified(root, dict(implementation[key]))
    source_rows, inputs, labels = validate_source(root, config)

    protocol = dict(config["preparation"])
    meeko_version = importlib.metadata.version("meeko")
    rdkit_version = importlib.metadata.version("rdkit")
    if meeko_version != str(protocol["meeko_version"]):
        raise ValueError(f"Meeko version differs: {meeko_version}")
    if rdkit_version != str(protocol["rdkit_version"]):
        raise ValueError(f"RDKit version differs: {rdkit_version}")
    meeko_script = find_meeko_script()
    expected = dict(config["expected"])
    if audit_only:
        result = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "status": "audit_only_ok",
            "target_id": config["target_id"],
            "ligand_count": len(source_rows),
            "label_counts": dict(sorted(labels.items())),
            "future_receptor_count": int(expected["future_receptor_count"]),
            "future_seed_count": int(expected["future_seed_count"]),
            "future_pair_count": int(expected["future_pair_count"]),
            "rdkit_version": rdkit_version,
            "meeko_version": meeko_version,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "docking_scores_read": 0,
            "operation": "input audit only; no ligand structure was prepared and no docking was started",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    outputs = dict(config["outputs"])
    run_directory = root / str(outputs["run_directory"])
    sdf_directory = run_directory / "sdf"
    pdbqt_directory = run_directory / "pdbqt"
    checkpoint_directory = run_directory / "checkpoints"
    manifest_path = root / str(outputs["manifest_csv"])
    summary_path = root / str(outputs["summary_json"])
    if not resume and not overwrite and (
        run_directory.exists() or manifest_path.exists() or summary_path.exists()
    ):
        raise FileExistsError("development outputs exist; pass --resume or --overwrite")
    if overwrite:
        for path in (manifest_path, summary_path):
            if path.exists():
                path.unlink()
    for directory in (sdf_directory, pdbqt_directory, checkpoint_directory):
        directory.mkdir(parents=True, exist_ok=True)

    config_sha256 = file_sha256(config_path)
    tasks = [
        {
            "row": row,
            "root": str(root),
            "sdf_directory": str(sdf_directory),
            "pdbqt_directory": str(pdbqt_directory),
            "checkpoint_directory": str(checkpoint_directory),
            "meeko_script": str(meeko_script),
            "index": index,
            "resume": resume,
            "overwrite": overwrite,
            "base_seed": int(protocol["rdkit_embed_base_seed"]),
            "seed_offsets": list(protocol["deterministic_retry_seed_offsets"]),
            "target_id": config["target_id"],
            "config_sha256": config_sha256,
        }
        for index, row in enumerate(source_rows)
    ]
    prepared_by_index: dict[int, dict[str, Any]] = {}
    workers = int(protocol["local_worker_count"])
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(prepare_one, task): int(task["index"]) for task in tasks
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            index = futures[future]
            prepared_by_index[index] = future.result()
            if completed % 25 == 0 or completed == len(tasks):
                print(f"prepared_or_resumed {completed}/{len(tasks)}", flush=True)
    prepared_rows = [prepared_by_index[index] for index in range(len(source_rows))]
    if [row["ligand_id"] for row in prepared_rows] != [
        row["ligand_id"] for row in source_rows
    ]:
        raise ValueError("development output order differs")
    for row in prepared_rows:
        for path_key, hash_key in (
            ("sdf_path", "sdf_sha256"),
            ("pdbqt_path", "pdbqt_sha256"),
        ):
            verified(root, {"path": row[path_key], "sha256": row[hash_key]})
        if macrocycle_closure_atom_types(root / str(row["pdbqt_path"])):
            raise ValueError(f"closure pseudoatom remains: {row['ligand_id']}")
    variants = Counter(str(row["preparation_variant"]) for row in prepared_rows)
    sdf_statuses = Counter(str(row["prep_status"]) for row in prepared_rows)
    resumed_count = sum(
        row["resume_status"] == "validated_checkpoint" for row in prepared_rows
    )
    if any(row["pdbqt_status"] != "ok" for row in prepared_rows):
        raise ValueError("development manifest contains a failed PDBQT")

    write_csv(manifest_path, prepared_rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": config["success_status"],
        "target_id": config["target_id"],
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": config_sha256,
        },
        "source_manifest": {
            "path": inputs["train_manifest"].relative_to(root).as_posix(),
            "sha256": file_sha256(inputs["train_manifest"]),
        },
        "ligand_count": len(prepared_rows),
        "label_counts": dict(sorted(labels.items())),
        "sdf_status_counts": dict(sorted(sdf_statuses.items())),
        "preparation_variant_counts": dict(sorted(variants.items())),
        "closure_pseudoatom_ligand_count": 0,
        "failed_ligand_count": 0,
        "order_preserved": True,
        "resumed_ligand_count": resumed_count,
        "prepared_ligand_count_this_invocation": len(prepared_rows) - resumed_count,
        "preparation": {
            "rdkit_version": rdkit_version,
            "meeko_version": meeko_version,
            "rdkit_method": protocol["rdkit_method"],
            "rdkit_embed_base_seed": protocol["rdkit_embed_base_seed"],
            "deterministic_retry_seed_offsets": protocol[
                "deterministic_retry_seed_offsets"
            ],
            "macrocycle_policy": protocol["macrocycle_policy"],
        },
        "future_production": {
            "receptor_count": int(expected["future_receptor_count"]),
            "seed_count": int(expected["future_seed_count"]),
            "pair_count": int(expected["future_pair_count"]),
        },
        "data_boundary": {
            "train_rows_read": len(prepared_rows),
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "docking_scores_read": 0,
        },
        "output": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": file_sha256(manifest_path),
        },
        "next_gate": config["next_gate"],
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.audit_only, args.resume, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
