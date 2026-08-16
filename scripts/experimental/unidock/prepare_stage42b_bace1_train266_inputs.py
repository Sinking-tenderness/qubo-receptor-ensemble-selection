"""Prepare the frozen BACE1 Train-266 ligands for Uni-Dock production."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.metadata
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from rdkit import Chem

try:
    from scripts.batch_prepare_ligand_pdbqt import (
        file_sha256,
        find_meeko_script,
        parse_pdbqt,
        run_meeko,
        safe_filename,
    )
    from scripts.prepare_ligand_3d_sdf import build_3d_mol
    from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
        macrocycle_closure_atom_types,
    )
except ModuleNotFoundError:
    from batch_prepare_ligand_pdbqt import (
        file_sha256,
        find_meeko_script,
        parse_pdbqt,
        run_meeko,
        safe_filename,
    )
    from prepare_ligand_3d_sdf import build_3d_mol
    from run_unidock_gpu_equivalence import macrocycle_closure_atom_types


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def verified(root: Path, descriptor: dict[str, object]) -> Path:
    path = Path(str(descriptor["path"]))
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def error_message(stdout: str, stderr: str) -> str:
    return "\n".join(part.strip() for part in (stdout, stderr) if part.strip())[-1000:]


def prepare_one(task: dict[str, object]) -> dict[str, object]:
    row = dict(task["row"])
    root = Path(str(task["root"]))
    sdf_directory = Path(str(task["sdf_directory"]))
    pdbqt_directory = Path(str(task["pdbqt_directory"]))
    meeko_script = Path(str(task["meeko_script"]))
    index = int(task["index"])
    overwrite = bool(task["overwrite"])
    base_seed = int(task["base_seed"])
    offsets = [int(value) for value in task["seed_offsets"]]
    ligand_id = str(row["ligand_id"])
    smiles = str(row["canonical_smiles"])
    sdf_path = sdf_directory / f"{safe_filename(ligand_id)}.sdf"
    pdbqt_path = pdbqt_directory / f"{safe_filename(ligand_id)}.pdbqt"
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (sdf_path, pdbqt_path):
        if path.exists():
            if not overwrite:
                raise FileExistsError(path)
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
        raise RuntimeError(f"RDKit 3D preparation failed for {ligand_id}: {sdf_message}")
    molecule.SetProp("_Name", ligand_id)
    molecule.SetProp("ligand_id", ligand_id)
    molecule.SetProp("source_smiles", str(row["source_smiles"]))
    molecule.SetProp("preparation_smiles", smiles)
    molecule.SetProp("label", str(row["label"]))
    molecule.SetProp("target_id", "BACE1")
    molecule.SetProp("rdkit_embed_seed", str(selected_seed))
    with tempfile.TemporaryDirectory(prefix="bace1_train266_") as temporary:
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
                raise RuntimeError(f"Meeko preparation failed for {ligand_id}: {details}")
            variant = "meeko_rigid_macrocycles"
            message = (
                "meeko_rigid_after_closure_pseudoatom_detection"
                if pseudoatoms
                else "meeko_rigid_after_flexible_failure"
            )
        remaining = macrocycle_closure_atom_types(staged_pdbqt)
        if remaining:
            raise ValueError(f"closure pseudoatoms remain for {ligand_id}: {remaining}")
        pdbqt_audit = parse_pdbqt(staged_pdbqt)
        shutil.copy2(staged_sdf, sdf_path)
        shutil.copy2(staged_pdbqt, pdbqt_path)
    if int(pdbqt_audit["pdbqt_atom_count"]) <= 0 or pdbqt_audit["torsdof"] == "":
        raise ValueError(f"invalid PDBQT for {ligand_id}")
    return {
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
        **pdbqt_audit,
    }


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if file_sha256(Path(__file__)) != str(implementation["preparer"]["sha256"]).upper():
        raise ValueError("Stage 42b preparer SHA-256 differs")
    for key in ("three_d_helper", "meeko_helper", "pseudoatom_audit_helper"):
        verified(root, dict(implementation[key]))
    inputs_config = dict(config["inputs"])
    allocation_config = verified(root, dict(inputs_config["panel_allocation_config"]))
    allocation_summary_path = root / str(inputs_config["panel_allocation_summary_path"])
    train_manifest_path = root / str(inputs_config["train_manifest_path"])
    if not allocation_summary_path.is_file() or not train_manifest_path.is_file():
        raise FileNotFoundError("Stage 42a realized outputs are missing")
    allocation = read_json(allocation_summary_path)
    if allocation["status"] != "stage42a_bace1_ligand_panels_frozen":
        raise ValueError("Stage 42a BACE1 ligand allocation did not pass")
    if str(allocation["config"]["sha256"]).upper() != file_sha256(allocation_config):
        raise ValueError("Stage 42a realized config identity differs")
    realized_manifest = dict(allocation["outputs"])["train_manifest_csv"]
    if str(realized_manifest["path"]) != train_manifest_path.relative_to(root).as_posix():
        raise ValueError("Stage 42a realized train-manifest path differs")
    if str(realized_manifest["sha256"]).upper() != file_sha256(train_manifest_path):
        raise ValueError("Stage 42a realized train-manifest hash differs")
    if any(
        int(allocation["data_boundary"][key]) != 0
        for key in ("docking_scores_read", "fresh_validation_docking_scores_read", "test_docking_scores_read")
    ):
        raise ValueError("Stage 42a allocation crossed a score boundary")
    source_rows = read_csv(train_manifest_path)
    expected = dict(config["expected"])
    if len(source_rows) != int(expected["ligand_count"]):
        raise ValueError("Stage 42b Train-266 count differs")
    labels = Counter(row["label"] for row in source_rows)
    if labels != Counter({key: int(value) for key, value in dict(expected["label_counts"]).items()}):
        raise ValueError("Stage 42b Train-266 labels differ")
    if {row["split"] for row in source_rows} != {"train"}:
        raise ValueError("Stage 42b exposed a non-train split")
    if {row["selection_role"] for row in source_rows} != {"development_train"}:
        raise ValueError("Stage 42b selection role differs")
    if len({row["ligand_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("Stage 42b source contains duplicate ligand IDs")

    protocol = dict(config["preparation"])
    meeko_version = importlib.metadata.version("meeko")
    rdkit_version = importlib.metadata.version("rdkit")
    if meeko_version != str(protocol["meeko_version"]):
        raise ValueError(f"Meeko version differs: {meeko_version}")
    if rdkit_version != str(protocol["rdkit_version"]):
        raise ValueError(f"RDKit version differs: {rdkit_version}")
    meeko_script = find_meeko_script()
    outputs = dict(config["outputs"])
    run_directory = root / str(outputs["run_directory"])
    sdf_directory = run_directory / "sdf"
    pdbqt_directory = run_directory / "pdbqt"
    manifest_path = root / str(outputs["manifest_csv"])
    summary_path = root / str(outputs["summary_json"])
    if not overwrite and (manifest_path.exists() or summary_path.exists()):
        raise FileExistsError("Stage 42b outputs exist; pass --overwrite")
    sdf_directory.mkdir(parents=True, exist_ok=True)
    pdbqt_directory.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "row": row,
            "root": str(root),
            "sdf_directory": str(sdf_directory),
            "pdbqt_directory": str(pdbqt_directory),
            "meeko_script": str(meeko_script),
            "index": index,
            "overwrite": overwrite,
            "base_seed": int(protocol["rdkit_embed_base_seed"]),
            "seed_offsets": list(protocol["deterministic_retry_seed_offsets"]),
        }
        for index, row in enumerate(source_rows)
    ]
    prepared_by_index: dict[int, dict[str, object]] = {}
    workers = int(protocol["local_worker_count"])
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(prepare_one, task): int(task["index"]) for task in tasks}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            index = futures[future]
            prepared_by_index[index] = future.result()
            if completed % 25 == 0 or completed == len(tasks):
                print(f"prepared {completed}/{len(tasks)}", flush=True)
    prepared_rows = [prepared_by_index[index] for index in range(len(source_rows))]
    if [row["ligand_id"] for row in prepared_rows] != [row["ligand_id"] for row in source_rows]:
        raise ValueError("Stage 42b output order differs")
    for row in prepared_rows:
        for path_key, hash_key in (("sdf_path", "sdf_sha256"), ("pdbqt_path", "pdbqt_sha256")):
            verified(root, {"path": row[path_key], "sha256": row[hash_key]})
        if macrocycle_closure_atom_types(root / str(row["pdbqt_path"])):
            raise ValueError(f"Stage 42b closure pseudoatom remains: {row['ligand_id']}")
    variants = Counter(str(row["preparation_variant"]) for row in prepared_rows)
    sdf_statuses = Counter(str(row["prep_status"]) for row in prepared_rows)
    if any(row["pdbqt_status"] != "ok" for row in prepared_rows):
        raise ValueError("Stage 42b contains a failed PDBQT")

    write_csv(manifest_path, prepared_rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage42b_bace1_train266_unidock_inputs_ok",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "source_manifest": {"path": train_manifest_path.relative_to(root).as_posix(), "sha256": file_sha256(train_manifest_path)},
        "ligand_count": len(prepared_rows),
        "label_counts": dict(sorted(labels.items())),
        "sdf_status_counts": dict(sorted(sdf_statuses.items())),
        "preparation_variant_counts": dict(sorted(variants.items())),
        "closure_pseudoatom_ligand_count": 0,
        "failed_ligand_count": 0,
        "order_preserved": True,
        "preparation": {
            "rdkit_version": rdkit_version,
            "meeko_version": meeko_version,
            "rdkit_method": protocol["rdkit_method"],
            "rdkit_embed_base_seed": protocol["rdkit_embed_base_seed"],
            "deterministic_retry_seed_offsets": protocol["deterministic_retry_seed_offsets"],
            "macrocycle_policy": protocol["macrocycle_policy"],
        },
        "data_boundary": {
            "train_rows_read": len(prepared_rows),
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "docking_scores_read": 0,
        },
        "output": {"path": manifest_path.relative_to(root).as_posix(), "sha256": file_sha256(manifest_path)},
        "next_gate": "generate and independently audit the 34-receptor by Train-266 by three-seed Uni-Dock development matrix",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
