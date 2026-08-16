"""Freeze Stage32 panels and optionally materialize the selected MD receptors."""

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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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
    stage31 = read_csv(rooted(root, config["inputs"]["stage31_candidate_manifest"]))
    frames = read_csv(rooted(root, config["inputs"]["stage28b_frame_manifest"]))
    by_id = {row["frame_id"]: row for row in frames}
    panel = config["receptor_panel"]
    ranks = {int(value) for value in panel["within_start_temporal_maximin_ranks"]}
    selected = [
        row for row in stage31
        if row["cohort_id"] == panel["source_cohort_id"]
        and int(row["temporal_maximin_rank"]) in ranks
    ]
    selected.sort(key=lambda row: (int(row["group_index"]), int(row["temporal_maximin_rank"])))
    if len(selected) != int(panel["receptor_count"]):
        raise ValueError("Stage32 selected frame count differs")
    if Counter(int(row["group_index"]) for row in selected) != Counter({value: 2 for value in range(8)}):
        raise ValueError("Stage32 frame panel is not two-per-start balanced")
    output = []
    for rank, row in enumerate(selected, start=1):
        source = by_id[row["frame_id"]]
        if int(source["global_frame_index"]) != int(row["global_frame_index"]):
            raise ValueError("Stage32 frame provenance differs")
        receptor_id = f"{row['frame_id']}_{row['conformer_id'].replace('PPARG_', '').replace('_aligned', '')}"
        directory = f"results/runs/stage32_pparg_md_functional_pilot_input_preparation/receptors/{receptor_id}"
        output.append({
            "panel_rank": rank,
            "conformer_id": receptor_id,
            "frame_id": row["frame_id"],
            "start_index": row["group_index"],
            "source_conformer_id": row["conformer_id"],
            "temporal_maximin_rank": row["temporal_maximin_rank"],
            "global_frame_index": row["global_frame_index"],
            "local_frame_index": row["local_frame_index"],
            "time_ps": row["time_ps"],
            "aligned_protein_dcd": source["aligned_protein_dcd"],
            "aligned_protein_pdb": source["aligned_protein_pdb"],
            "snapshot_pdb": f"{directory}/snapshot_heavy.pdb",
            "protein_only_pdb": f"{directory}/protein_only.pdb",
            "prepared_pdb": f"{directory}/prepared.pdb",
            "receptor_pdbqt": f"{directory}/{receptor_id}_receptor.pdbqt",
            "receptor_preparation_summary": f"{directory}/summary.json",
            "status": "selection_frozen",
        })
    return output


def frozen_ligands(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    source = read_csv(rooted(root, config["inputs"]["stage19b_train_ligand_manifest"]))
    timing = config["evidence_timing"]
    valid = [
        row for row in source
        if row["split"] == timing["allowed_split"]
        and row["selection_role"] == timing["allowed_selection_role"]
        and row["pdbqt_status"] == "ok"
    ]
    panel = config["ligand_panel"]
    selected: list[dict[str, Any]] = []
    for label, required in (("active", int(panel["active_count"])), ("decoy", int(panel["decoy_count"]))):
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in valid:
            if row["label"] == label:
                groups[row["split_group_id"]].append(row)
        ordered_groups = sorted(
            groups,
            key=lambda group: hashlib.sha256(f"{panel['selection_seed']}|{label}|{group}".encode("ascii")).hexdigest(),
        )
        if len(ordered_groups) < required:
            raise ValueError(f"Stage32 has too few {label} scaffold groups")
        for class_rank, group in enumerate(ordered_groups[:required], start=1):
            row = min(groups[group], key=lambda value: (value["allocation_rank_sha256"], value["ligand_id"]))
            selected.append({
                "panel_label_rank": class_rank,
                "panel_selection_hash": hashlib.sha256(f"{panel['selection_seed']}|{label}|{group}".encode("ascii")).hexdigest(),
                **row,
            })
    selected.sort(key=lambda row: (0 if row["label"] == "active" else 1, int(row["panel_label_rank"])))
    for index, row in enumerate(selected, start=1):
        row["panel_rank"] = index
    if Counter(row["label"] for row in selected) != Counter({"active": 80, "decoy": 80}):
        raise ValueError("Stage32 label balance differs")
    for label in ("active", "decoy"):
        values = [row["split_group_id"] for row in selected if row["label"] == label]
        if len(values) != len(set(values)):
            raise ValueError(f"Stage32 {label} scaffold groups are not unique")
    return selected


def materialize_receptors(
    config: dict[str, Any],
    root: Path,
    frames: list[dict[str, Any]],
    overwrite: bool,
    trajectory_root: Path | None,
    receptor_preparation_python: Path | None,
) -> None:
    try:
        import mdtraj as md
    except ImportError as error:
        raise RuntimeError("mdtraj is required only for --materialize-receptors") from error
    prepare_script = root / "scripts/prepare_receptor.py"
    preparation_python = (receptor_preparation_python or Path(sys.executable)).resolve()
    if not preparation_python.is_file():
        raise FileNotFoundError(f"receptor-preparation Python is missing: {preparation_python}")
    source_root = (trajectory_root or root).resolve()
    atom_selection = config["receptor_panel"]["snapshot_atom_selection"]
    for row in frames:
        topology = rooted(source_root, row["aligned_protein_pdb"])
        trajectory = rooted(source_root, row["aligned_protein_dcd"])
        if not topology.is_file() or not trajectory.is_file():
            raise FileNotFoundError(f"missing Stage28b trajectory input for {row['conformer_id']}")
        snapshot = rooted(root, row["snapshot_pdb"])
        outputs = [
            snapshot,
            rooted(root, row["protein_only_pdb"]),
            rooted(root, row["prepared_pdb"]),
            rooted(root, row["receptor_pdbqt"]),
            rooted(root, row["receptor_preparation_summary"]),
        ]
        if all(path.is_file() for path in outputs) and not overwrite:
            summary = read_json(outputs[4])
            if summary.get("status") != "ok":
                raise ValueError(f"existing Stage32 preparation is invalid: {row['conformer_id']}")
            row["snapshot_pdb_sha256"] = sha256(snapshot)
            row["receptor_pdbqt_sha256"] = sha256(outputs[3])
            row["receptor_preparation_summary_sha256"] = sha256(outputs[4])
            row["status"] = "ok"
            continue
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        frame = md.load_frame(str(trajectory), int(row["local_frame_index"]), top=str(topology))
        atom_indices = frame.topology.select(str(atom_selection))
        if len(atom_indices) == 0:
            raise ValueError(f"empty Stage32 atom selection for {row['conformer_id']}")
        frame.atom_slice(atom_indices).save_pdb(str(snapshot))
        command = [
            str(preparation_python),
            str(prepare_script),
            "--input-pdb", str(snapshot),
            "--chain", str(config["receptor_panel"]["chain_id"]),
            "--protein-only-output", str(outputs[1]),
            "--prepared-pdb-output", str(outputs[2]),
            "--pdbqt-output", str(outputs[3]),
            "--summary-output", str(outputs[4]),
            "--charge-model", str(config["receptor_panel"]["charge_model"]),
        ]
        if overwrite or any(path.exists() for path in outputs[1:]):
            command.append("--overwrite")
        subprocess.run(command, cwd=root, check=True)
        summary = read_json(outputs[4])
        if summary.get("status") != "ok":
            raise ValueError(f"Stage32 receptor preparation failed: {row['conformer_id']}")
        row["snapshot_pdb_sha256"] = sha256(snapshot)
        row["receptor_pdbqt_sha256"] = sha256(outputs[3])
        row["receptor_preparation_summary_sha256"] = sha256(outputs[4])
        row["status"] = "ok"


def prepare(
    config_path: Path,
    root: Path,
    materialize: bool,
    overwrite: bool,
    trajectory_root: Path | None = None,
    receptor_preparation_python: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if read_json(rooted(root, config["inputs"]["stage31_audit"])).get("status") != "stage31_pparg_objective_landscape_screen_audit_ok":
        raise ValueError("Stage31 audit gate differs")
    if read_json(rooted(root, config["inputs"]["stage19b_preparation_summary"])).get("status") != "stage19b_pparg_train668_unidock_inputs_ok":
        raise ValueError("Stage19b ligand preparation gate differs")
    frames = frozen_frames(config, root)
    ligands = frozen_ligands(config, root)
    if materialize:
        materialize_receptors(
            config,
            root,
            frames,
            overwrite,
            trajectory_root,
            receptor_preparation_python,
        )
    outputs = config["outputs"]
    frame_path = rooted(root, outputs["selected_frame_manifest"])
    ligand_path = rooted(root, outputs["selected_ligand_manifest"])
    receptor_path = rooted(root, outputs["prepared_receptor_manifest"])
    result_path = rooted(root, outputs["preparation_result"])
    write_csv(frame_path, frames)
    write_csv(ligand_path, ligands)
    if materialize:
        write_csv(receptor_path, frames)
    status = "stage32_inputs_ok" if materialize and all(row["status"] == "ok" for row in frames) else "stage32_selection_frozen_awaiting_remote_receptor_materialization"
    result = {
        "schema_version": "1.0",
        "status": status,
        "experiment_id": config["experiment_id"],
        "config": descriptor(root, config_path),
        "selected_frame_manifest": descriptor(root, frame_path),
        "selected_ligand_manifest": descriptor(root, ligand_path),
        "prepared_receptor_manifest": descriptor(root, receptor_path) if receptor_path.is_file() else None,
        "counts": {"receptors": len(frames), "starts": len({row["start_index"] for row in frames}), "ligands": len(ligands), "labels": dict(sorted(Counter(row["label"] for row in ligands).items())), "seed_count": len(config["seeds"]), "expected_pairs": len(frames) * len(ligands) * len(config["seeds"])},
        "data_boundary": {"train_rows_read": len(read_csv(rooted(root, config["inputs"]["stage19b_train_ligand_manifest"]))), "fresh_validation_rows_read": 0, "test_rows_read": 0, "docking_scores_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "next_gate": "materialize and audit all 16 receptor PDBQT files before Stage32 docking" if not materialize else "run Stage32 audit-only before GPU docking",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32_pparg_md_functional_complementarity_pilot.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--materialize-receptors", action="store_true")
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        help="Existing Stage28b workspace containing the aligned DCD/topology files",
    )
    parser.add_argument(
        "--receptor-preparation-python",
        type=Path,
        help="Python executable from the frozen ProDy/Meeko Uni-Dock environment",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(
        args.config,
        args.root,
        args.materialize_receptors,
        args.overwrite,
        args.trajectory_root,
        args.receptor_preparation_python,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
