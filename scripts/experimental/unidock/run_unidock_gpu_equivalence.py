"""Run the fixed train-only MAPK14 Uni-Dock GPU equivalence experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


VINA_RESULT = re.compile(
    r"^REMARK\s+VINA\s+RESULT:\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
)
MACROCYCLE_CLOSURE_ATOM_TYPE = re.compile(r"^(?:CG|G)\d+$")
DEFAULT_MAXIMUM_ABSOLUTE_GPU_SCORE = 100.0
OUTPUT_CONTAINER_WARNING = "WARNING: in add_to_output_container"
COORDINATE_SIZE_MISMATCH = "t.coords.size()="


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


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


def pdbqt_atom_types(path: Path) -> set[str]:
    atom_types: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                fields = line.split()
                if fields:
                    atom_types.add(fields[-1])
    return atom_types


def macrocycle_closure_atom_types(path: Path) -> list[str]:
    return sorted(
        atom_type
        for atom_type in pdbqt_atom_types(path)
        if MACROCYCLE_CLOSURE_ATOM_TYPE.fullmatch(atom_type)
    )


def unidock_input_compatibility(
    config: dict[str, object], audit: dict[str, object]
) -> dict[str, object]:
    protocol = config["unidock"]
    policy = str(
        protocol.get("macrocycle_closure_pseudoatom_policy", "reject")
    )
    allowed_policies = {"reject", "allow_train_diagnostic"}
    if policy not in allowed_policies:
        raise ValueError(
            "unknown Uni-Dock macrocycle pseudoatom policy: " f"{policy}"
        )
    count = int(audit["macrocycle_closure_pseudoatom_ligand_count"])
    train_only = (
        int(audit["validation_rows"]) == 0
        and int(audit["test_rows"]) == 0
        and config["data_boundary"]["allowed_split"] == "train"
    )
    compatible = count == 0 or (
        policy == "allow_train_diagnostic" and train_only
    )
    return {
        "status": "compatible" if compatible else "blocked",
        "compatible": compatible,
        "macrocycle_closure_pseudoatom_policy": policy,
        "macrocycle_closure_pseudoatom_ligand_count": count,
        "train_only": train_only,
        "reason": (
            "no Meeko macrocycle closure pseudoatoms detected"
            if count == 0
            else (
                "unsupported Meeko macrocycle closure pseudoatoms are "
                "explicitly allowed for a consumed-train diagnostic only"
                if compatible
                else "Meeko CG*/G* macrocycle closure pseudoatoms require "
                "rigid-macrocycle re-preparation or an explicit consumed-train "
                "diagnostic route before Uni-Dock execution"
            )
        ),
    }


def validate_gpu_score(
    score: float,
    ligand_id: str,
    maximum_absolute_score: float = DEFAULT_MAXIMUM_ABSOLUTE_GPU_SCORE,
) -> None:
    if not math.isfinite(score):
        raise ValueError(f"non-finite Uni-Dock score for {ligand_id}: {score}")
    if maximum_absolute_score <= 0.0:
        raise ValueError("maximum absolute GPU score guard must be positive")
    if abs(score) > maximum_absolute_score:
        raise ValueError(
            f"nonphysical Uni-Dock score for {ligand_id}: {score} kcal/mol; "
            f"absolute guard={maximum_absolute_score}"
        )


def unidock_log_warnings(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    output_container = text.count(OUTPUT_CONTAINER_WARNING)
    coordinate_mismatch = text.count(COORDINATE_SIZE_MISMATCH)
    return {
        "output_container_warning_count": output_container,
        "coordinate_size_mismatch_count": coordinate_mismatch,
        "total_count": output_container + coordinate_mismatch,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def rooted_path(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"configured path leaves bundle root: {value}") from error
    return path


def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def verified_path(
    root: Path,
    descriptor: dict[str, object],
    path_key: str = "path",
    hash_key: str = "sha256",
) -> Path:
    path = rooted_path(root, str(descriptor[path_key]))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(descriptor[hash_key])
    observed = file_sha256(path)
    if observed != expected.upper():
        raise ValueError(f"SHA-256 differs for {path}: {observed}")
    return path


def validate_inputs(
    root: Path, config: dict[str, object]
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    inputs = config["inputs"]
    expected = config["expected"]
    boundary = config["data_boundary"]
    receptor_path = verified_path(root, inputs["receptor_manifest"])
    ligand_path = verified_path(root, inputs["ligand_manifest"])
    receptors = read_csv(receptor_path)
    ligands = read_csv(ligand_path)
    receptor_ids = [row["conformer_id"] for row in receptors]

    if receptor_ids != list(expected["receptor_ids"]):
        raise ValueError("GPU-equivalence receptor IDs or order differ")
    if len(receptors) != int(expected["receptor_count"]):
        raise ValueError("GPU-equivalence receptor count differs")
    if any(row["status"] != "ok" for row in receptors):
        raise ValueError("receptor manifest contains a non-ok row")
    if len(ligands) != int(expected["ligand_count"]):
        raise ValueError("GPU-equivalence ligand count differs")
    if Counter(row["label"] for row in ligands) != Counter(
        boundary["label_counts"]
    ):
        raise ValueError("GPU-equivalence ligand labels differ")
    if {row["split"] for row in ligands} != {boundary["allowed_split"]}:
        raise ValueError("GPU-equivalence manifest contains a non-train row")
    if {row["selection_role"] for row in ligands} != {
        boundary["allowed_selection_role"]
    }:
        raise ValueError("GPU-equivalence selection role differs")
    if any(row["pdbqt_status"] != "ok" for row in ligands):
        raise ValueError("ligand manifest contains a failed PDBQT row")
    if len({row["ligand_id"] for row in ligands}) != len(ligands):
        raise ValueError("ligand manifest contains duplicate IDs")

    for rows, path_column, hash_column, id_column in (
        (
            receptors,
            "receptor_pdbqt",
            "receptor_pdbqt_sha256",
            "conformer_id",
        ),
        (ligands, "pdbqt_path", "pdbqt_sha256", "ligand_id"),
    ):
        for row in rows:
            path = rooted_path(root, row[path_column])
            if not path.is_file():
                raise FileNotFoundError(path)
            if file_sha256(path) != row[hash_column].upper():
                raise ValueError(f"prepared PDBQT hash differs: {row[id_column]}")

    macrocycle_ligands: list[dict[str, object]] = []
    for row in ligands:
        path = rooted_path(root, row["pdbqt_path"])
        pseudoatom_types = macrocycle_closure_atom_types(path)
        if pseudoatom_types:
            macrocycle_ligands.append(
                {
                    "ligand_id": row["ligand_id"],
                    "label": row["label"],
                    "pdbqt_path": relative_path(root, path),
                    "pdbqt_sha256": file_sha256(path),
                    "pseudoatom_types": pseudoatom_types,
                }
            )

    cpu_audit: list[dict[str, object]] = []
    for seed in inputs["cpu_seed_runs"]:
        summary_path = verified_path(
            root, seed, "summary_path", "summary_sha256"
        )
        scores_path = verified_path(root, seed, "scores_path", "scores_sha256")
        summary = read_json(summary_path)
        scores = read_csv(scores_path)
        seed_id = str(seed["seed_id"])
        base_seed = int(seed["base_seed"])
        if int(summary["docking_parameters"]["base_seed"]) != base_seed:
            raise ValueError(f"CPU base seed differs: {seed_id}")
        if len(scores) != int(expected["cpu_reference_pair_count_per_seed"]):
            raise ValueError(f"CPU reference pair count differs: {seed_id}")
        if any(row["status"] != "ok" for row in scores):
            raise ValueError(f"CPU reference contains a failed pair: {seed_id}")
        keys = {(row["ligand_id"], row["receptor_id"]) for row in scores}
        if len(keys) != len(scores):
            raise ValueError(f"CPU reference contains duplicate pairs: {seed_id}")
        selected = [row for row in scores if row["receptor_id"] in receptor_ids]
        if len(selected) != int(expected["pair_count_per_seed"]):
            raise ValueError(f"selected CPU pair count differs: {seed_id}")
        cpu_audit.append(
            {
                "seed_id": seed_id,
                "base_seed": base_seed,
                "source_pair_count": len(scores),
                "selected_pair_count": len(selected),
                "measured_wall_runtime_seconds": float(
                    summary["measured_wall_runtime_seconds"]
                ),
            }
        )

    return receptors, ligands, {
        "status": "audit_only_ok",
        "receptor_count": len(receptors),
        "receptor_ids": receptor_ids,
        "ligand_count": len(ligands),
        "label_counts": dict(sorted(Counter(row["label"] for row in ligands).items())),
        "seed_count": len(cpu_audit),
        "gpu_pair_count": len(receptors) * len(ligands) * len(cpu_audit),
        "macrocycle_closure_pseudoatom_ligand_count": len(
            macrocycle_ligands
        ),
        "macrocycle_closure_pseudoatom_ligands": macrocycle_ligands,
        "cpu_reference": cpu_audit,
        "validation_rows": 0,
        "test_rows": 0,
        "enrichment_metrics_calculated": False,
    }


def conda_package_version(package: str) -> str | None:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None
    metadata_dir = Path(prefix) / "conda-meta"
    candidates = sorted(metadata_dir.glob(f"{package}-*.json"))
    versions: set[str] = set()
    for path in candidates:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError, UnicodeError):
            continue
        if metadata.get("name") == package and metadata.get("version"):
            versions.add(str(metadata["version"]))
    if len(versions) > 1:
        raise ValueError(f"multiple installed {package} versions found: {versions}")
    return next(iter(versions), None)


def executable_evidence(executable: str, required_version: str) -> dict[str, object]:
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(f"Uni-Dock executable is not on PATH: {executable}")
    binary = Path(resolved).resolve()
    package_version = conda_package_version("unidock")
    probe = subprocess.run(
        [str(binary), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    probe_output = probe.stdout.strip()
    version_verified = package_version == required_version or (
        required_version in probe_output
    )
    if not version_verified:
        raise ValueError(
            "cannot verify required Uni-Dock version "
            f"{required_version}; conda={package_version!r}, probe={probe_output!r}"
        )
    return {
        "requested_executable": executable,
        "resolved_executable": str(binary),
        "binary_sha256": file_sha256(binary),
        "required_package_version": required_version,
        "conda_package_version": package_version,
        "version_probe_returncode": probe.returncode,
        "version_probe_output": probe_output,
        "version_verified": True,
    }


def parse_vina_pose(path: Path) -> tuple[float, int]:
    scores: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            match = VINA_RESULT.match(raw_line.strip())
            if match:
                score = float(match.group(1))
                if not math.isfinite(score):
                    raise ValueError(f"non-finite Vina score in {path}")
                scores.append(score)
    if not scores:
        raise ValueError(f"no Vina result found in {path}")
    return scores[0], len(scores)


def map_pose_outputs(
    pose_dir: Path, ligands: list[dict[str, str]]
) -> dict[str, Path]:
    expected_by_stem = {
        Path(row["pdbqt_path"]).stem: row["ligand_id"] for row in ligands
    }
    mapped: dict[str, Path] = {}
    for path in sorted(pose_dir.glob("*.pdbqt")):
        stem = path.stem[:-4] if path.stem.endswith("_out") else path.stem
        ligand_id = expected_by_stem.get(stem)
        if ligand_id is None:
            continue
        if ligand_id in mapped:
            raise ValueError(f"multiple pose outputs found for {ligand_id}")
        mapped[ligand_id] = path
    missing = sorted(set(row["ligand_id"] for row in ligands) - set(mapped))
    if missing:
        raise ValueError(f"missing Uni-Dock outputs: {missing[:10]}")
    return mapped


def protocol_signature(
    config_sha256: str,
    seed_id: str,
    base_seed: int,
    receptor: dict[str, str],
    ligands: list[dict[str, str]],
    protocol: dict[str, object],
) -> str:
    value = {
        "config_sha256": config_sha256,
        "seed_id": seed_id,
        "base_seed": base_seed,
        "receptor_id": receptor["conformer_id"],
        "receptor_sha256": receptor["receptor_pdbqt_sha256"],
        "ligands": [
            (row["ligand_id"], row["pdbqt_sha256"]) for row in ligands
        ],
        "protocol": protocol,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest().upper()


def batch_paths(
    run_directory: Path, seed_id: str, receptor_id: str
) -> dict[str, Path]:
    directory = run_directory / "batches" / seed_id / receptor_id
    return {
        "directory": directory,
        "pose_directory": directory / "poses",
        "ligand_index": directory / "ligands.index",
        "log": directory / "unidock.log",
        "scores": directory / "scores.csv",
        "summary": directory / "batch_summary.json",
    }


def validate_checkpoint(
    root: Path,
    paths: dict[str, Path],
    signature: str,
    ligand_ids: set[str],
) -> tuple[list[dict[str, str]], dict[str, object]] | None:
    if not paths["summary"].is_file() or not paths["scores"].is_file():
        return None
    try:
        summary = read_json(paths["summary"])
        rows = read_csv(paths["scores"])
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if summary.get("status") != "ok" or summary.get("signature") != signature:
        return None
    if {row["ligand_id"] for row in rows} != ligand_ids:
        return None
    if len(rows) != len(ligand_ids):
        return None
    for row in rows:
        output = rooted_path(root, row["output_pose_path"])
        if not output.is_file() or file_sha256(output) != row["output_pose_sha256"]:
            return None
    if file_sha256(paths["scores"]) != summary.get("scores_sha256"):
        return None
    return rows, summary


def unidock_command(
    executable: str,
    receptor_path: Path,
    ligand_index: Path,
    pose_directory: Path,
    protocol: dict[str, object],
    base_seed: int,
) -> list[str]:
    box = protocol["box"]
    return [
        executable,
        "--receptor",
        str(receptor_path),
        "--ligand_index",
        str(ligand_index),
        "--scoring",
        str(protocol["scoring"]),
        "--center_x",
        str(box["center_x"]),
        "--center_y",
        str(box["center_y"]),
        "--center_z",
        str(box["center_z"]),
        "--size_x",
        str(box["size_x"]),
        "--size_y",
        str(box["size_y"]),
        "--size_z",
        str(box["size_z"]),
        "--exhaustiveness",
        str(protocol["exhaustiveness"]),
        "--max_step",
        str(protocol["max_step"]),
        "--refine_step",
        str(protocol["refine_step"]),
        "--num_modes",
        str(protocol["num_modes"]),
        "--energy_range",
        str(protocol["energy_range"]),
        "--verbosity",
        str(protocol["verbosity"]),
        "--seed",
        str(base_seed),
        "--dir",
        str(pose_directory),
    ]


def run_batch(
    root: Path,
    paths: dict[str, Path],
    executable: str,
    receptor: dict[str, str],
    ligands: list[dict[str, str]],
    protocol: dict[str, object],
    seed_id: str,
    base_seed: int,
    signature: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    paths["pose_directory"].mkdir(parents=True, exist_ok=True)
    ligand_paths = [rooted_path(root, row["pdbqt_path"]) for row in ligands]
    paths["ligand_index"].write_text(
        "\n".join(str(path) for path in ligand_paths) + "\n",
        encoding="utf-8",
    )
    receptor_path = rooted_path(root, receptor["receptor_pdbqt"])
    command = unidock_command(
        executable,
        receptor_path,
        paths["ligand_index"],
        paths["pose_directory"],
        protocol,
        base_seed,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(protocol["cuda_visible_devices"])
    environment["OMP_NUM_THREADS"] = "1"
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    with paths["log"].open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Uni-Dock failed for {seed_id}/{receptor['conformer_id']} "
            f"with exit code {completed.returncode}; see {paths['log']}"
        )
    engine_warnings = unidock_log_warnings(paths["log"])

    mapped = map_pose_outputs(paths["pose_directory"], ligands)
    rows: list[dict[str, object]] = []
    score_guard = float(
        protocol.get(
            "maximum_absolute_score_kcal_per_mol",
            DEFAULT_MAXIMUM_ABSOLUTE_GPU_SCORE,
        )
    )
    for ligand in ligands:
        output = mapped[ligand["ligand_id"]]
        score, pose_count = parse_vina_pose(output)
        validate_gpu_score(score, ligand["ligand_id"], score_guard)
        rows.append(
            {
                "target_id": "MK14",
                "seed_id": seed_id,
                "base_seed": base_seed,
                "receptor_id": receptor["conformer_id"],
                "ligand_id": ligand["ligand_id"],
                "label": ligand["label"],
                "selection_role": ligand["selection_role"],
                "gpu_score": score,
                "pose_count": pose_count,
                "status": "ok",
                "output_pose_path": relative_path(root, output),
                "output_pose_sha256": file_sha256(output),
            }
        )
    write_csv(paths["scores"], rows)
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "status": "ok",
        "signature": signature,
        "seed_id": seed_id,
        "base_seed": base_seed,
        "receptor_id": receptor["conformer_id"],
        "ligand_count": len(rows),
        "elapsed_seconds": elapsed,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        "omp_num_threads": environment["OMP_NUM_THREADS"],
        "maximum_absolute_score_kcal_per_mol": score_guard,
        "engine_log_warnings": engine_warnings,
        "log_path": relative_path(root, paths["log"]),
        "log_sha256": file_sha256(paths["log"]),
        "scores_path": relative_path(root, paths["scores"]),
        "scores_sha256": file_sha256(paths["scores"]),
        "score_minimum": min(float(row["gpu_score"]) for row in rows),
        "score_maximum": max(float(row["gpu_score"]) for row in rows),
    }
    write_json(paths["summary"], summary)
    return rows, summary


def output_descriptor(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": relative_path(root, path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def flatten(rows: Iterable[list[dict[str, object]]]) -> list[dict[str, object]]:
    return [row for group in rows for row in group]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--unidock", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    config_path = args.config.resolve()
    config = read_json(config_path)
    config_sha256 = file_sha256(config_path)
    receptors, ligands, input_audit = validate_inputs(root, config)
    audit_output = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "config": {
            "path": relative_path(root, config_path),
            "sha256": config_sha256,
        },
        **input_audit,
        "operation": "input audit only; no GPU docking or equivalence metric was run",
    }
    compatibility = unidock_input_compatibility(config, input_audit)
    audit_output["input_compatibility"] = compatibility
    audit_output["status"] = (
        "audit_only_ok"
        if compatibility["compatible"]
        else "input_compatibility_failed"
    )
    if args.audit_only:
        print(json.dumps(audit_output, indent=2, sort_keys=True))
        return 0 if compatibility["compatible"] else 2
    if not compatibility["compatible"]:
        raise ValueError(str(compatibility["reason"]))

    protocol = config["unidock"]
    executable = args.unidock or str(protocol["executable"])
    executable_info = executable_evidence(
        executable, str(protocol["required_package_version"])
    )
    resolved_executable = str(executable_info["resolved_executable"])
    outputs = config["outputs"]
    run_directory = rooted_path(root, str(outputs["run_directory"]))
    run_directory.mkdir(parents=True, exist_ok=True)
    invocation_started = time.perf_counter()
    all_rows: list[list[dict[str, object]]] = []
    batch_summaries: list[dict[str, object]] = []
    executed_batches = 0
    resumed_batches = 0
    ligand_ids = {row["ligand_id"] for row in ligands}

    for seed in config["inputs"]["cpu_seed_runs"]:
        seed_id = str(seed["seed_id"])
        base_seed = int(seed["base_seed"])
        for receptor in receptors:
            receptor_id = receptor["conformer_id"]
            paths = batch_paths(run_directory, seed_id, receptor_id)
            paths["directory"].mkdir(parents=True, exist_ok=True)
            signature = protocol_signature(
                config_sha256,
                seed_id,
                base_seed,
                receptor,
                ligands,
                protocol,
            )
            checkpoint = (
                validate_checkpoint(root, paths, signature, ligand_ids)
                if args.resume
                else None
            )
            if checkpoint is not None:
                rows, summary = checkpoint
                all_rows.append([dict(row) for row in rows])
                batch_summaries.append(summary)
                resumed_batches += 1
                print(f"resume ok: {seed_id}/{receptor_id}", flush=True)
                continue
            print(f"running: {seed_id}/{receptor_id}", flush=True)
            rows, summary = run_batch(
                root,
                paths,
                resolved_executable,
                receptor,
                ligands,
                protocol,
                seed_id,
                base_seed,
                signature,
            )
            all_rows.append(rows)
            batch_summaries.append(summary)
            executed_batches += 1
            print(
                f"completed: {seed_id}/{receptor_id} "
                f"in {summary['elapsed_seconds']:.3f} s",
                flush=True,
            )

    gpu_rows = flatten(all_rows)
    gpu_rows.sort(
        key=lambda row: (
            str(row["seed_id"]),
            str(row["receptor_id"]),
            str(row["ligand_id"]),
        )
    )
    if len(gpu_rows) != int(config["expected"]["total_gpu_pair_count"]):
        raise ValueError("complete GPU score count differs")
    if len(
        {
            (row["seed_id"], row["receptor_id"], row["ligand_id"])
            for row in gpu_rows
        }
    ) != len(gpu_rows):
        raise ValueError("complete GPU scores contain duplicate keys")

    scores_path = rooted_path(root, str(outputs["gpu_scores_csv"]))
    batches_path = rooted_path(root, str(outputs["gpu_batch_runs_csv"]))
    summary_path = rooted_path(root, str(outputs["gpu_summary_json"]))
    write_csv(scores_path, gpu_rows)
    batch_rows = [
        {
            "seed_id": summary["seed_id"],
            "base_seed": summary["base_seed"],
            "receptor_id": summary["receptor_id"],
            "ligand_count": summary["ligand_count"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "score_minimum": summary["score_minimum"],
            "score_maximum": summary["score_maximum"],
            "log_path": summary["log_path"],
            "log_sha256": summary["log_sha256"],
            "scores_path": summary["scores_path"],
            "scores_sha256": summary["scores_sha256"],
            "signature": summary["signature"],
            "engine_warning_count": summary.get(
                "engine_log_warnings", {}
            ).get("total_count", 0),
            "status": summary["status"],
        }
        for summary in batch_summaries
    ]
    write_csv(batches_path, batch_rows)
    gpu_elapsed = sum(float(row["elapsed_seconds"]) for row in batch_summaries)
    engine_warning_count = sum(
        int(summary.get("engine_log_warnings", {}).get("total_count", 0))
        for summary in batch_summaries
    )
    run_summary: dict[str, object] = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "operation": "train-only Uni-Dock GPU score generation; no enrichment or validation metric was calculated",
        "config": {
            "path": relative_path(root, config_path),
            "sha256": config_sha256,
        },
        "unidock_executable": executable_info,
        "protocol": protocol,
        "receptor_count": len(receptors),
        "ligand_count": len(ligands),
        "seed_count": len(config["inputs"]["cpu_seed_runs"]),
        "gpu_pair_count": len(gpu_rows),
        "batch_count": len(batch_summaries),
        "executed_batches_this_invocation": executed_batches,
        "resumed_batches_this_invocation": resumed_batches,
        "gpu_batch_elapsed_seconds_total": gpu_elapsed,
        "gpu_pairs_per_second": len(gpu_rows) / gpu_elapsed,
        "engine_warning_count": engine_warning_count,
        "engine_warning_batch_count": sum(
            int(summary.get("engine_log_warnings", {}).get("total_count", 0))
            > 0
            for summary in batch_summaries
        ),
        "current_invocation_elapsed_seconds": time.perf_counter()
        - invocation_started,
        "score_range_kcal_per_mol": {
            "minimum": min(float(row["gpu_score"]) for row in gpu_rows),
            "maximum": max(float(row["gpu_score"]) for row in gpu_rows),
        },
        "outputs": {
            "gpu_scores_csv": output_descriptor(root, scores_path),
            "gpu_batch_runs_csv": output_descriptor(root, batches_path),
        },
        "validation_rows": 0,
        "test_rows": 0,
        "enrichment_metrics_calculated": False,
        "interpretation_note": config["decision_boundary"],
    }
    write_json(summary_path, run_summary)
    print(json.dumps(run_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
