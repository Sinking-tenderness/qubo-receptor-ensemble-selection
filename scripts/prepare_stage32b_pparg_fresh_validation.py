"""Prepare Stage32b fresh-validation ligands and reuse two Stage32 receptors."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.batch_prepare_ligand_pdbqt import find_meeko_script, parse_pdbqt, run_meeko, safe_filename
from scripts.experimental.unidock.run_unidock_gpu_equivalence import macrocycle_closure_atom_types
from scripts.prepare_ligand_3d_sdf import build_3d_mol
from scripts.stage32b_common import descriptor, read_csv, read_json, sha256, write_csv


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
    molecule.SetProp("target_id", "PPARG")
    molecule.SetProp("rdkit_embed_seed", str(selected_seed))
    with tempfile.TemporaryDirectory(prefix="pparg_stage32b_") as temporary:
        staging = Path(temporary)
        staged_sdf = staging / "ligand.sdf"
        staged_pdbqt = staging / "ligand.pdbqt"
        writer = Chem.SDWriter(str(staged_sdf))
        writer.write(molecule)
        writer.close()
        flexible = run_meeko(meeko_script, staged_sdf, staged_pdbqt, rigid_macrocycles=False)
        flexible_ok = flexible.returncode == 0 and staged_pdbqt.is_file()
        pseudoatoms = macrocycle_closure_atom_types(staged_pdbqt) if flexible_ok else []
        variant = "meeko_flexible"
        message = "meeko_flexible_ok"
        if not flexible_ok or pseudoatoms:
            if staged_pdbqt.exists():
                staged_pdbqt.unlink()
            rigid = run_meeko(meeko_script, staged_sdf, staged_pdbqt, rigid_macrocycles=True)
            if rigid.returncode != 0 or not staged_pdbqt.is_file():
                details = error_message(flexible.stdout + "\n" + rigid.stdout, flexible.stderr + "\n" + rigid.stderr)
                raise RuntimeError(f"Meeko preparation failed for {ligand_id}: {details}")
            variant = "meeko_rigid_macrocycles"
            message = "meeko_rigid_after_closure_pseudoatom_detection" if pseudoatoms else "meeko_rigid_after_flexible_failure"
        if macrocycle_closure_atom_types(staged_pdbqt):
            raise ValueError(f"closure pseudoatoms remain for {ligand_id}")
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
        "sdf_sha256": sha256(sdf_path),
        "pdbqt_status": "ok",
        "pdbqt_message": message,
        "pdbqt_path": pdbqt_path.relative_to(root).as_posix(),
        "pdbqt_sha256": sha256(pdbqt_path),
        "preparation_variant": variant,
        **pdbqt_audit,
    }


def checkpoint_valid(root: Path, checkpoint: Path) -> dict[str, Any] | None:
    if not checkpoint.is_file():
        return None
    try:
        row = read_json(checkpoint)
        sdf = root / row["sdf_path"]
        pdbqt = root / row["pdbqt_path"]
        if not sdf.is_file() or not pdbqt.is_file():
            return None
        if sha256(sdf) != row["sdf_sha256"] or sha256(pdbqt) != row["pdbqt_sha256"]:
            return None
        if macrocycle_closure_atom_types(pdbqt):
            return None
        return row
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return None


def prepare(config_path: Path, root: Path, stage32_workspace: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    stage32_workspace = stage32_workspace.resolve()
    config = read_json(config_path)
    selection = read_json(root / config["outputs"]["train_selection_json"])
    if selection.get("status") != "stage32b_pparg_md_pair_train_selection_frozen":
        raise ValueError("Stage32b train selection gate differs")
    source_ligands = read_csv(root / config["outputs"]["fresh_validation_source_manifest"])
    if len(source_ligands) != 1576 or Counter(row["label"] for row in source_ligands) != Counter({"active": 75, "decoy": 1501}):
        raise ValueError("Stage32b source ligand panel differs")
    source_receptors = read_csv(root / config["outputs"]["selected_receptor_manifest"])
    if {row["conformer_id"] for row in source_receptors} != set(selection["selected_pair"]["receptor_ids"]):
        raise ValueError("Stage32b receptor source panel differs")

    protocol = config["ligand_preparation"]
    rdkit_version = importlib.metadata.version("rdkit")
    meeko_version = importlib.metadata.version("meeko")
    if rdkit_version != protocol["rdkit_version"] or meeko_version != protocol["meeko_version"]:
        raise ValueError(f"Stage32b preparation environment differs: rdkit={rdkit_version}, meeko={meeko_version}")
    meeko_script = find_meeko_script()
    input_root = root / config["outputs"]["run_directory"] / "inputs"
    sdf_directory = input_root / "sdf"
    pdbqt_directory = input_root / "pdbqt"
    checkpoint_directory = input_root / "checkpoints"
    receptor_directory = input_root / "receptors"
    for directory in (sdf_directory, pdbqt_directory, checkpoint_directory, receptor_directory):
        directory.mkdir(parents=True, exist_ok=True)

    prepared_receptors = []
    for row in source_receptors:
        source = stage32_workspace / row["receptor_pdbqt"]
        if not source.is_file() or sha256(source) != row["receptor_pdbqt_sha256"]:
            raise ValueError(f"Stage32 receptor source differs: {row['conformer_id']}")
        target = receptor_directory / source.name
        if overwrite or not target.is_file() or sha256(target) != row["receptor_pdbqt_sha256"]:
            shutil.copy2(source, target)
        record = dict(row)
        record["receptor_pdbqt"] = target.relative_to(root).as_posix()
        record["receptor_pdbqt_sha256"] = sha256(target)
        record["status"] = "ok"
        prepared_receptors.append(record)

    prepared_by_index: dict[int, dict[str, Any]] = {}
    tasks = []
    for index, row in enumerate(source_ligands):
        checkpoint = checkpoint_directory / f"{index:05d}.json"
        existing = None if overwrite else checkpoint_valid(root, checkpoint)
        if existing is not None:
            prepared_by_index[index] = existing
            continue
        partial = (sdf_directory / f"{safe_filename(row['ligand_id'])}.sdf").exists() or (pdbqt_directory / f"{safe_filename(row['ligand_id'])}.pdbqt").exists()
        tasks.append({
            "row": row,
            "root": str(root),
            "sdf_directory": str(sdf_directory),
            "pdbqt_directory": str(pdbqt_directory),
            "meeko_script": str(meeko_script),
            "index": index,
            "overwrite": overwrite or partial,
            "base_seed": int(protocol["rdkit_embed_base_seed"]),
            "seed_offsets": list(protocol["deterministic_retry_seed_offsets"]),
        })
    if tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(protocol["worker_count"])) as executor:
            futures = {executor.submit(prepare_one, task): int(task["index"]) for task in tasks}
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                index = futures[future]
                row = future.result()
                prepared_by_index[index] = row
                checkpoint = checkpoint_directory / f"{index:05d}.json"
                checkpoint.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="ascii")
                if completed % 25 == 0 or completed == len(tasks):
                    print(f"prepared this run {completed}/{len(tasks)}; total {len(prepared_by_index)}/{len(source_ligands)}", flush=True)
    prepared = [prepared_by_index[index] for index in range(len(source_ligands))]
    if any(row["pdbqt_status"] != "ok" for row in prepared):
        raise ValueError("Stage32b contains a failed ligand PDBQT")
    ligand_manifest = root / config["outputs"]["prepared_ligand_manifest"]
    receptor_manifest = root / config["outputs"]["selected_receptor_manifest"]
    write_csv(ligand_manifest, prepared)
    write_csv(receptor_manifest, prepared_receptors)
    result = {
        "schema_version": "1.0",
        "status": "stage32b_pparg_md_pair_fresh_validation_inputs_ok",
        "experiment_id": config["experiment_id"],
        "config": descriptor(root, config_path),
        "stage32_workspace": str(stage32_workspace),
        "prepared_ligand_manifest": descriptor(root, ligand_manifest),
        "prepared_receptor_manifest": descriptor(root, receptor_manifest),
        "counts": {"receptors": 2, "ligands": len(prepared), "active": 75, "decoy": 1501, "checkpoint_count": len(prepared), "locked_test_rows": 0},
        "environment": {"rdkit": rdkit_version, "meeko": meeko_version},
        "data_boundary": {"fresh_validation_identity_rows_read": 1576, "fresh_validation_score_rows_read": 0, "locked_test_rows_read": 0, "new_docking_jobs": 0},
        "next_gate": "run Stage32b audit-only, then execute six Uni-Dock batches",
        "decision_boundary": config["decision_boundary"],
    }
    output = root / config["outputs"]["preparation_result"]
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32b_pparg_md_pair_fresh_validation.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--stage32-workspace", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args.config, args.root, args.stage32_workspace, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
