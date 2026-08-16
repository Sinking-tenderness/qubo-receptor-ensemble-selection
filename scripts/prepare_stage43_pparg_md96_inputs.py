"""Freeze the Stage43 PPARG MD-96 panel and materialize its 80 new receptors."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import write_json  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)




def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def rooted(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def frozen_frames(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    candidates = read_csv(rooted(root, config["inputs"]["stage31_candidate_manifest"]))
    frame_manifest = read_csv(rooted(root, config["inputs"]["stage28b_frame_manifest"]))
    historical = read_csv(rooted(root, config["inputs"]["stage32_prepared_receptor_manifest"]))
    by_frame = {row["frame_id"]: row for row in frame_manifest}
    historical_ids = {row["frame_id"] for row in historical}
    candidates.sort(key=lambda row: (int(row["group_index"]), int(row["temporal_maximin_rank"])))
    if len(candidates) != 96 or Counter(int(row["group_index"]) for row in candidates) != Counter({index: 12 for index in range(8)}):
        raise ValueError("Stage43 candidate panel is not the frozen 12-per-start Stage31 pool")
    if len(historical_ids) != 16:
        raise ValueError("Stage43 historical Stage32 receptor count differs")

    output: list[dict[str, Any]] = []
    for panel_rank, row in enumerate(candidates, start=1):
        source = by_frame[row["frame_id"]]
        if int(source["global_frame_index"]) != int(row["global_frame_index"]):
            raise ValueError(f"Stage43 frame provenance differs: {row['frame_id']}")
        suffix = row["conformer_id"].replace("PPARG_", "").replace("_aligned", "")
        receptor_id = f"{row['frame_id']}_{suffix}"
        directory = f"results/runs/stage43_pparg_md96_input_preparation/receptors/{receptor_id}"
        historical_reuse = row["frame_id"] in historical_ids
        output.append({
            "panel_rank": panel_rank,
            "conformer_id": receptor_id,
            "frame_id": row["frame_id"],
            "start_index": row["group_index"],
            "source_conformer_id": row["conformer_id"],
            "cohort_id": row["cohort_id"],
            "temporal_maximin_rank": row["temporal_maximin_rank"],
            "global_frame_index": row["global_frame_index"],
            "local_frame_index": row["local_frame_index"],
            "time_ps": row["time_ps"],
            "aligned_protein_dcd": source["aligned_protein_dcd"],
            "aligned_protein_pdb": source["aligned_protein_pdb"],
            "evidence_role": "historical_stage32_reuse" if historical_reuse else "new_stage43_docking",
            "snapshot_pdb": f"{directory}/snapshot_heavy.pdb",
            "protein_only_pdb": f"{directory}/protein_only.pdb",
            "prepared_pdb": f"{directory}/prepared.pdb",
            "receptor_pdbqt": f"{directory}/{receptor_id}_receptor.pdbqt",
            "receptor_preparation_summary": f"{directory}/summary.json",
            "status": "historical_stage32_reuse" if historical_reuse else "selection_frozen",
            "snapshot_pdb_sha256": "",
            "receptor_pdbqt_sha256": "",
            "receptor_preparation_summary_sha256": "",
        })
    if Counter(row["evidence_role"] for row in output) != Counter({"historical_stage32_reuse": 16, "new_stage43_docking": 80}):
        raise ValueError("Stage43 historical/new partition differs")
    return output


def materialize_new_receptors(
    config: dict[str, Any],
    root: Path,
    frames: list[dict[str, Any]],
    trajectory_root: Path,
    preparation_python: Path,
    overwrite: bool,
) -> None:
    try:
        import mdtraj as md
    except ImportError as error:
        raise RuntimeError("mdtraj is required only for Stage43 receptor materialization") from error
    prepare_script = root / "scripts/prepare_receptor.py"
    for row in frames:
        if row["evidence_role"] == "historical_stage32_reuse":
            continue
        topology = rooted(trajectory_root, row["aligned_protein_pdb"])
        trajectory = rooted(trajectory_root, row["aligned_protein_dcd"])
        if not topology.is_file() or not trajectory.is_file():
            raise FileNotFoundError(f"missing Stage28b trajectory input for {row['conformer_id']}")
        outputs = [
            rooted(root, row["snapshot_pdb"]),
            rooted(root, row["protein_only_pdb"]),
            rooted(root, row["prepared_pdb"]),
            rooted(root, row["receptor_pdbqt"]),
            rooted(root, row["receptor_preparation_summary"]),
        ]
        if all(path.is_file() for path in outputs) and not overwrite:
            summary = read_json(outputs[-1])
            if summary.get("status") != "ok":
                raise ValueError(f"existing Stage43 preparation is invalid: {row['conformer_id']}")
        else:
            outputs[0].parent.mkdir(parents=True, exist_ok=True)
            frame = md.load_frame(str(trajectory), int(row["local_frame_index"]), top=str(topology))
            atoms = frame.topology.select(config["receptor_panel"]["snapshot_atom_selection"])
            if len(atoms) == 0:
                raise ValueError(f"empty Stage43 atom selection: {row['conformer_id']}")
            frame.atom_slice(atoms).save_pdb(str(outputs[0]))
            command = [
                str(preparation_python), str(prepare_script),
                "--input-pdb", str(outputs[0]),
                "--chain", config["receptor_panel"]["chain_id"],
                "--protein-only-output", str(outputs[1]),
                "--prepared-pdb-output", str(outputs[2]),
                "--pdbqt-output", str(outputs[3]),
                "--summary-output", str(outputs[4]),
                "--charge-model", config["receptor_panel"]["charge_model"],
            ]
            if overwrite or any(path.exists() for path in outputs[1:]):
                command.append("--overwrite")
            subprocess.run(command, cwd=root, check=True)
        row["status"] = "ok"
        row["snapshot_pdb_sha256"] = sha256(outputs[0])
        row["receptor_pdbqt_sha256"] = sha256(outputs[3])
        row["receptor_preparation_summary_sha256"] = sha256(outputs[4])


def prepare(
    config_path: Path,
    root: Path,
    materialize: bool,
    trajectory_root: Path | None,
    receptor_preparation_python: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    frames = frozen_frames(config, root)
    ligands = read_csv(rooted(root, config["inputs"]["stage32_ligand_manifest"]))
    if len(ligands) != 160 or Counter(row["label"] for row in ligands) != Counter({"active": 80, "decoy": 80}):
        raise ValueError("Stage43 ligand panel differs from Stage32 Train-160")
    if materialize:
        if trajectory_root is None or receptor_preparation_python is None:
            raise ValueError("materialization requires trajectory root and receptor preparation Python")
        materialize_new_receptors(
            config, root, frames, trajectory_root.resolve(),
            receptor_preparation_python.resolve(), overwrite,
        )
    outputs = config["outputs"]
    frame_path = rooted(root, outputs["frame_manifest"])
    receptor_path = rooted(root, outputs["prepared_receptor_manifest"])
    result_path = rooted(root, outputs["preparation_result"])
    write_csv(frame_path, frames)
    write_csv(receptor_path, frames)
    complete = materialize and all(
        row["status"] in {"ok", "historical_stage32_reuse"} for row in frames
    )
    result = {
        "schema_version": "1.0",
        "status": "stage43_pparg_md96_inputs_ok" if complete else "stage43_pparg_md96_selection_frozen",
        "experiment_id": config["experiment_id"],
        "config": descriptor(root, config_path),
        "frame_manifest": descriptor(root, frame_path),
        "prepared_receptor_manifest": descriptor(root, receptor_path),
        "counts": {
            "receptors": 96, "historical_receptors": 16, "new_receptors": 80,
            "ligands": 160, "seed_count": 3,
            "combined_pair_count": 46080, "new_pair_count": 38400,
        },
        "data_boundary": {
            "train_rows_read": 160, "fresh_validation_rows_read": 0,
            "test_rows_read": 0, "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "next_gate": "run Stage43 audit-only before the 240 new Uni-Dock batches" if complete else "materialize the 80 new MD receptors on the preserved Stage28b workspace",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage43_pparg_md96_rank_sensitive_replication.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--materialize-receptors", action="store_true")
    parser.add_argument("--trajectory-root", type=Path)
    parser.add_argument("--receptor-preparation-python", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    prepare(config_path, root, args.materialize_receptors, args.trajectory_root, args.receptor_preparation_python, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
