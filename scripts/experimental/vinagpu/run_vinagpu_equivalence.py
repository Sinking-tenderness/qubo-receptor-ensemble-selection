"""Run the frozen train-only MAPK14 AutoDock Vina-GPU 2.1 pilot."""

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


VINA_RESULT = re.compile(
    r"^REMARK\s+VINA\s+RESULT:\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
)
MACROCYCLE_CLOSURE_ATOM_TYPE = re.compile(r"^(?:CG|G)\d+$")
DEFAULT_MAXIMUM_ABSOLUTE_SCORE = 100.0
BLOCKED_LOG_PATTERNS = (
    "CL_OUT_OF_HOST_MEMORY",
    "pocket too large!",
    "Relation too large!",
    "Grid too large!",
    "Build kernel 1 from source",
    "Build kernel 2 from source",
)


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


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


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


def rooted_path(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"configured path leaves repository root: {value}") from error
    return path


def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def output_descriptor(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": relative_path(root, path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified_path(
    root: Path,
    descriptor: dict[str, object],
    path_key: str = "path",
    hash_key: str = "sha256",
) -> Path:
    path = rooted_path(root, str(descriptor[path_key]))
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if observed != str(descriptor[hash_key]).upper():
        raise ValueError(f"SHA-256 differs for {path}: {observed}")
    return path


def pdbqt_atom_types(path: Path) -> set[str]:
    atom_types: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                fields = line.split()
                if fields:
                    atom_types.add(fields[-1])
    return atom_types


def validate_inputs(
    root: Path, config: dict[str, object]
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    inputs = config["inputs"]
    expected = config["expected"]
    boundary = config["data_boundary"]
    receptor_manifest = verified_path(root, inputs["receptor_manifest"])
    ligand_manifest = verified_path(root, inputs["ligand_manifest"])
    receptors = read_csv(receptor_manifest)
    ligands = read_csv(ligand_manifest)

    receptor_ids = [row["conformer_id"] for row in receptors]
    if receptor_ids != list(expected["receptor_ids"]):
        raise ValueError("receptor IDs or order differ from the frozen protocol")
    if len(receptors) != int(expected["receptor_count"]):
        raise ValueError("receptor count differs from the frozen protocol")
    if any(row["status"] != "ok" for row in receptors):
        raise ValueError("receptor manifest contains a non-ok row")
    if len(ligands) != int(expected["ligand_count"]):
        raise ValueError("ligand count differs from the frozen protocol")
    if Counter(row["label"] for row in ligands) != Counter(
        boundary["label_counts"]
    ):
        raise ValueError("ligand label counts differ from the frozen protocol")
    if {row["split"] for row in ligands} != {boundary["allowed_split"]}:
        raise ValueError("ligand manifest contains a non-train row")
    if {row["selection_role"] for row in ligands} != {
        boundary["allowed_selection_role"]
    }:
        raise ValueError("ligand manifest contains an unauthorized selection role")
    if any(row["pdbqt_status"] != "ok" for row in ligands):
        raise ValueError("ligand manifest contains a failed PDBQT row")
    if len({row["ligand_id"] for row in ligands}) != len(ligands):
        raise ValueError("ligand manifest contains duplicate IDs")

    offsets = [int(row["seed_offset"]) for row in ligands]
    if offsets != list(range(len(ligands))):
        raise ValueError("ligand seed offsets must be ordered contiguous integers")

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

    closure_ligands: list[str] = []
    for ligand in ligands:
        path = rooted_path(root, ligand["pdbqt_path"])
        atom_types = pdbqt_atom_types(path)
        if any(MACROCYCLE_CLOSURE_ATOM_TYPE.fullmatch(x) for x in atom_types):
            closure_ligands.append(ligand["ligand_id"])
    if closure_ligands:
        raise ValueError(
            "Vina-GPU input contains unsupported macrocycle closure pseudoatoms: "
            f"{closure_ligands[:5]}"
        )

    cpu_audit: list[dict[str, object]] = []
    expected_pairs = int(expected["cpu_reference_pair_count_per_seed"])
    receptor_set = set(receptor_ids)
    expected_pair_keys = {
        (ligand["ligand_id"], receptor_id)
        for ligand in ligands
        for receptor_id in receptor_ids
    }
    for seed in inputs["cpu_seed_runs"]:
        summary_path = verified_path(root, seed, "summary_path", "summary_sha256")
        scores_path = verified_path(root, seed, "scores_path", "scores_sha256")
        summary = read_json(summary_path)
        scores = read_csv(scores_path)
        seed_id = str(seed["seed_id"])
        base_seed = int(seed["base_seed"])
        if int(summary["docking_parameters"]["base_seed"]) != base_seed:
            raise ValueError(f"CPU base seed differs: {seed_id}")
        selected = [row for row in scores if row["receptor_id"] in receptor_set]
        keys = {(row["ligand_id"], row["receptor_id"]) for row in selected}
        if (
            len(selected) != expected_pairs
            or len(keys) != expected_pairs
            or keys != expected_pair_keys
        ):
            raise ValueError(f"CPU reference pair set differs: {seed_id}")
        if any(row["status"] != "ok" for row in selected):
            raise ValueError(f"CPU reference contains a failed pair: {seed_id}")
        cpu_audit.append(
            {
                "seed_id": seed_id,
                "base_seed": base_seed,
                "selected_pair_count": len(selected),
            }
        )

    total_pairs = len(receptors) * len(ligands) * len(cpu_audit)
    if total_pairs != int(expected["total_gpu_pair_count"]):
        raise ValueError("derived GPU pair count differs from the frozen protocol")
    if tuple(config["compatibility_smoke"]["blocked_log_patterns"]) != BLOCKED_LOG_PATTERNS:
        raise ValueError("compatibility smoke blocked log patterns differ")
    if not bool(config["compatibility_smoke"]["require_precompiled_kernel_mode"]):
        raise ValueError("compatibility smoke must require precompiled kernels")
    return receptors, ligands, {
        "status": "audit_only_ok",
        "receptor_count": len(receptors),
        "receptor_ids": receptor_ids,
        "ligand_count": len(ligands),
        "label_counts": dict(sorted(Counter(x["label"] for x in ligands).items())),
        "seed_count": len(cpu_audit),
        "gpu_pair_count": total_pairs,
        "seed_policy": config["seed_policy"],
        "macrocycle_closure_pseudoatom_ligand_count": 0,
        "cpu_reference": cpu_audit,
        "validation_rows": 0,
        "test_rows": 0,
        "enrichment_metrics_calculated": False,
    }


def resolve_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    located = shutil.which(value)
    if located is None:
        raise FileNotFoundError(f"Vina-GPU executable not found: {value}")
    return Path(located).resolve()


def git_head(source_tree: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_tree), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot read Vina-GPU source commit: {result.stdout.strip()}")
    return result.stdout.strip()


def makefile_settings(path: Path) -> dict[str, str]:
    wanted = {
        "GPU_PLATFORM",
        "OPENCL_VERSION",
        "DOCKING_BOX_SIZE",
    }
    observed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() in wanted:
            observed[key.strip()] = value.strip()
    if set(observed) != wanted:
        raise ValueError(f"Vina-GPU Makefile lacks frozen build settings: {path}")
    return observed


def runtime_evidence(
    executable_value: str,
    kernel_directory: Path,
    source_tree: Path,
    protocol: dict[str, object],
) -> dict[str, object]:
    executable = resolve_executable(executable_value)
    kernel_directory = kernel_directory.expanduser().resolve()
    source_tree = source_tree.expanduser().resolve()
    kernel1 = kernel_directory / "Kernel1_Opt.bin"
    kernel2 = kernel_directory / "Kernel2_Opt.bin"
    makefile = kernel_directory / "Makefile"
    for path in (kernel1, kernel2, makefile):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    version = subprocess.run(
        [str(executable), "--version"],
        cwd=executable.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    version_output = version.stdout.strip()
    required_version = str(protocol["required_version_probe"])
    if version.returncode != 0 or required_version not in version_output:
        raise ValueError(
            f"Vina-GPU version probe failed: return={version.returncode}, "
            f"output={version_output!r}"
        )
    observed_head = git_head(source_tree)
    if observed_head != str(protocol["source_commit"]):
        raise ValueError(f"Vina-GPU source commit differs: {observed_head}")

    settings = makefile_settings(makefile)
    expected_settings = {
        "GPU_PLATFORM": str(protocol["build_profile"]["gpu_platform"]),
        "OPENCL_VERSION": str(protocol["build_profile"]["opencl_version"]),
        "DOCKING_BOX_SIZE": str(
            protocol["build_profile"]["docking_box_size"]
        ),
    }
    if settings != expected_settings:
        raise ValueError(
            f"Vina-GPU build settings differ: {settings} != {expected_settings}"
        )
    return {
        "schema_version": "1.0",
        "status": "locked",
        "source_repository": protocol["source_repository"],
        "source_commit": observed_head,
        "method": protocol["method"],
        "version_probe": version_output,
        "executable_path": str(executable),
        "executable_sha256": file_sha256(executable),
        "opencl_binary_path": str(kernel_directory),
        "kernel1_sha256": file_sha256(kernel1),
        "kernel2_sha256": file_sha256(kernel2),
        "makefile_sha256": file_sha256(makefile),
        "build_settings": settings,
        "kernel_mode": protocol["build_profile"]["kernel_mode"],
    }


def ensure_runtime_lock(path: Path, evidence: dict[str, object]) -> None:
    if path.is_file():
        existing = read_json(path)
        if existing != evidence:
            raise ValueError(
                "runtime identity differs from the existing immutable lock: "
                f"{path}"
            )
        return
    write_json(path, evidence)


def pair_seed(base_seed: int, ligand: dict[str, str]) -> int:
    offset = int(ligand["seed_offset"])
    if offset < 0:
        raise ValueError(f"negative seed offset: {ligand['ligand_id']}")
    return base_seed + offset


def vinagpu_command(
    executable: Path,
    kernel_directory: Path,
    receptor_path: Path,
    ligand_path: Path,
    output_path: Path,
    protocol: dict[str, object],
    seed: int,
) -> list[str]:
    box = protocol["box"]
    command = [
        str(executable),
        "--receptor",
        str(receptor_path),
        "--ligand",
        str(ligand_path),
        "--out",
        str(output_path),
        "--opencl_binary_path",
        str(kernel_directory),
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
        "--thread",
        str(protocol["thread"]),
        "--rilc_bfgs",
        str(protocol["rilc_bfgs"]),
        "--num_modes",
        str(protocol["num_modes"]),
        "--energy_range",
        str(protocol["energy_range"]),
        "--seed",
        str(seed),
    ]
    search_depth = protocol["search_depth"]
    if search_depth != "heuristic":
        command.extend(["--search_depth", str(search_depth)])
    return command


def parse_vina_pose(path: Path) -> tuple[float, int]:
    scores: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = VINA_RESULT.match(line.strip())
            if match:
                scores.append(float(match.group(1)))
    if not scores:
        raise ValueError(f"no Vina score found in {path}")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError(f"non-finite Vina score found in {path}")
    return scores[0], len(scores)


def pair_signature(
    config_sha256: str,
    runtime_lock: dict[str, object],
    seed_id: str,
    seed: int,
    receptor: dict[str, str],
    ligand: dict[str, str],
    protocol: dict[str, object],
) -> str:
    value = {
        "config_sha256": config_sha256,
        "runtime_identity": runtime_lock,
        "seed_id": seed_id,
        "pair_seed": seed,
        "receptor_id": receptor["conformer_id"],
        "receptor_sha256": receptor["receptor_pdbqt_sha256"],
        "ligand_id": ligand["ligand_id"],
        "ligand_sha256": ligand["pdbqt_sha256"],
        "protocol": protocol,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest().upper()


def pair_paths(
    run_directory: Path,
    seed_id: str,
    receptor_id: str,
    ligand_id: str,
) -> dict[str, Path]:
    directory = run_directory / "pairs" / seed_id / receptor_id / ligand_id
    return {
        "directory": directory,
        "log": directory / "vinagpu.log",
        "pose": directory / "pose_out.pdbqt",
        "summary": directory / "pair_summary.json",
    }


def checkpoint_row(
    root: Path,
    paths: dict[str, Path],
    signature: str,
) -> tuple[dict[str, object], dict[str, object]] | None:
    if not paths["summary"].is_file():
        return None
    try:
        summary = read_json(paths["summary"])
        pose = rooted_path(root, str(summary["output_pose_path"]))
        log = rooted_path(root, str(summary["log_path"]))
        if summary.get("status") != "ok" or summary.get("signature") != signature:
            return None
        if file_sha256(pose) != summary["output_pose_sha256"]:
            return None
        if file_sha256(log) != summary["log_sha256"]:
            return None
        score = float(summary["gpu_score"])
        if not math.isfinite(score):
            return None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    row = {
        "target_id": "MK14",
        "seed_id": summary["seed_id"],
        "base_seed": summary["base_seed"],
        "seed_offset": summary["seed_offset"],
        "pair_seed": summary["pair_seed"],
        "receptor_id": summary["receptor_id"],
        "ligand_id": summary["ligand_id"],
        "label": summary["label"],
        "selection_role": summary["selection_role"],
        "gpu_vinagpu21_score": summary["gpu_score"],
        "pose_count": summary["pose_count"],
        "status": "ok",
        "output_pose_path": summary["output_pose_path"],
        "output_pose_sha256": summary["output_pose_sha256"],
    }
    return row, summary


def run_pair(
    root: Path,
    paths: dict[str, Path],
    executable: Path,
    kernel_directory: Path,
    runtime_lock: dict[str, object],
    config_sha256: str,
    receptor: dict[str, str],
    ligand: dict[str, str],
    protocol: dict[str, object],
    seed_id: str,
    base_seed: int,
    resume: bool,
) -> tuple[dict[str, object], dict[str, object], bool]:
    seed = pair_seed(base_seed, ligand)
    signature = pair_signature(
        config_sha256,
        runtime_lock,
        seed_id,
        seed,
        receptor,
        ligand,
        protocol,
    )
    if resume:
        checkpoint = checkpoint_row(root, paths, signature)
        if checkpoint is not None:
            return checkpoint[0], checkpoint[1], True

    paths["directory"].mkdir(parents=True, exist_ok=True)
    receptor_path = rooted_path(root, receptor["receptor_pdbqt"])
    ligand_path = rooted_path(root, ligand["pdbqt_path"])
    command = vinagpu_command(
        executable,
        kernel_directory,
        receptor_path,
        ligand_path,
        paths["pose"],
        protocol,
        seed,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(protocol["cuda_visible_devices"])
    environment["OMP_NUM_THREADS"] = "1"
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    with paths["log"].open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            cwd=executable.parent,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    log_text = paths["log"].read_text(encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            f"Vina-GPU failed for {seed_id}/{receptor['conformer_id']}/"
            f"{ligand['ligand_id']} with exit code {result.returncode}; "
            f"see {paths['log']}"
        )
    blocked = [pattern for pattern in BLOCKED_LOG_PATTERNS if pattern in log_text]
    if blocked:
        raise RuntimeError(f"blocked Vina-GPU log pattern(s): {blocked}")
    if not paths["pose"].is_file() or paths["pose"].stat().st_size == 0:
        raise RuntimeError(f"Vina-GPU did not create a pose: {paths['pose']}")

    score, pose_count = parse_vina_pose(paths["pose"])
    guard = float(
        protocol.get(
            "maximum_absolute_score_kcal_per_mol",
            DEFAULT_MAXIMUM_ABSOLUTE_SCORE,
        )
    )
    if abs(score) > guard:
        raise ValueError(
            f"nonphysical Vina-GPU score for {ligand['ligand_id']}: {score}"
        )
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "status": "ok",
        "signature": signature,
        "seed_id": seed_id,
        "base_seed": base_seed,
        "seed_offset": int(ligand["seed_offset"]),
        "pair_seed": seed,
        "receptor_id": receptor["conformer_id"],
        "ligand_id": ligand["ligand_id"],
        "label": ligand["label"],
        "selection_role": ligand["selection_role"],
        "gpu_score": score,
        "pose_count": pose_count,
        "elapsed_seconds": elapsed,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        "omp_num_threads": environment["OMP_NUM_THREADS"],
        "runtime_executable_sha256": runtime_lock["executable_sha256"],
        "output_pose_path": relative_path(root, paths["pose"]),
        "output_pose_sha256": file_sha256(paths["pose"]),
        "log_path": relative_path(root, paths["log"]),
        "log_sha256": file_sha256(paths["log"]),
    }
    write_json(paths["summary"], summary)
    row = {
        "target_id": "MK14",
        "seed_id": seed_id,
        "base_seed": base_seed,
        "seed_offset": int(ligand["seed_offset"]),
        "pair_seed": seed,
        "receptor_id": receptor["conformer_id"],
        "ligand_id": ligand["ligand_id"],
        "label": ligand["label"],
        "selection_role": ligand["selection_role"],
        "gpu_vinagpu21_score": score,
        "pose_count": pose_count,
        "status": "ok",
        "output_pose_path": summary["output_pose_path"],
        "output_pose_sha256": summary["output_pose_sha256"],
    }
    return row, summary, False


def smoke_paths(run_directory: Path) -> dict[str, Path]:
    directory = run_directory / "compatibility_smoke"
    return {
        "directory": directory,
        "log": directory / "vinagpu.log",
        "pose": directory / "pose_out.pdbqt",
        "summary": directory / "pair_summary.json",
    }


def run_smoke(
    root: Path,
    config: dict[str, object],
    config_sha256: str,
    receptors: list[dict[str, str]],
    ligands: list[dict[str, str]],
    executable: Path,
    kernel_directory: Path,
    runtime_lock: dict[str, object],
    run_directory: Path,
    resume: bool,
) -> dict[str, object]:
    smoke = config["compatibility_smoke"]
    receptor = next(
        row for row in receptors if row["conformer_id"] == smoke["receptor_id"]
    )
    ligand = next(row for row in ligands if row["ligand_id"] == smoke["ligand_id"])
    seed_descriptor = next(
        row
        for row in config["inputs"]["cpu_seed_runs"]
        if row["seed_id"] == smoke["seed_id"]
    )
    row, pair_summary, resumed = run_pair(
        root,
        smoke_paths(run_directory),
        executable,
        kernel_directory,
        runtime_lock,
        config_sha256,
        receptor,
        ligand,
        config["vinagpu"],
        str(seed_descriptor["seed_id"]),
        int(seed_descriptor["base_seed"]),
        resume,
    )
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "compatibility_smoke_passed",
        "operation": "one real MAPK14 receptor-ligand pair using the frozen box",
        "resumed": resumed,
        "seed_id": row["seed_id"],
        "pair_seed": row["pair_seed"],
        "receptor_id": row["receptor_id"],
        "ligand_id": row["ligand_id"],
        "gpu_vinagpu21_score": row["gpu_vinagpu21_score"],
        "elapsed_seconds": pair_summary["elapsed_seconds"],
        "output_pose_path": row["output_pose_path"],
        "output_pose_sha256": row["output_pose_sha256"],
        "pair_summary_path": relative_path(
            root, smoke_paths(run_directory)["summary"]
        ),
        "pair_summary_sha256": file_sha256(
            smoke_paths(run_directory)["summary"]
        ),
        "blocked_log_patterns_absent": True,
        "precompiled_kernel_mode_verified": True,
    }
    write_json(run_directory / "compatibility_smoke_summary.json", summary)
    return summary


def validate_smoke(
    root: Path,
    run_directory: Path,
    config: dict[str, object],
    config_sha256: str,
    runtime_lock: dict[str, object],
    receptors: list[dict[str, str]],
    ligands: list[dict[str, str]],
) -> None:
    path = run_directory / "compatibility_smoke_summary.json"
    if not path.is_file() or read_json(path).get("status") != "compatibility_smoke_passed":
        raise RuntimeError("compatibility smoke is missing; run --smoke-only first")
    smoke = config["compatibility_smoke"]
    receptor = next(x for x in receptors if x["conformer_id"] == smoke["receptor_id"])
    ligand = next(x for x in ligands if x["ligand_id"] == smoke["ligand_id"])
    seed_descriptor = next(
        x
        for x in config["inputs"]["cpu_seed_runs"]
        if x["seed_id"] == smoke["seed_id"]
    )
    seed = pair_seed(int(seed_descriptor["base_seed"]), ligand)
    signature = pair_signature(
        config_sha256,
        runtime_lock,
        str(seed_descriptor["seed_id"]),
        seed,
        receptor,
        ligand,
        config["vinagpu"],
    )
    if checkpoint_row(root, smoke_paths(run_directory), signature) is None:
        raise RuntimeError("compatibility smoke checkpoint no longer validates")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--vinagpu")
    parser.add_argument("--opencl-binary-path", type=Path)
    parser.add_argument("--source-tree", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--lock-runtime-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    mode_count = sum(
        int(flag)
        for flag in (args.audit_only, args.lock_runtime_only, args.smoke_only)
    )
    if mode_count > 1:
        parser.error("choose at most one audit/lock/smoke mode")

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
    if args.audit_only:
        print(json.dumps(audit_output, indent=2, sort_keys=True))
        return 0

    if not args.vinagpu or args.opencl_binary_path is None:
        parser.error("--vinagpu and --opencl-binary-path are required for runtime modes")
    executable = resolve_executable(args.vinagpu)
    source_tree = (
        args.source_tree.resolve()
        if args.source_tree is not None
        else executable.parent.parent.resolve()
    )
    runtime_lock = runtime_evidence(
        str(executable),
        args.opencl_binary_path,
        source_tree,
        config["vinagpu"],
    )
    outputs = config["outputs"]
    run_directory = rooted_path(root, str(outputs["run_directory"]))
    run_directory.mkdir(parents=True, exist_ok=True)
    lock_path = rooted_path(root, str(outputs["runtime_lock_json"]))
    ensure_runtime_lock(lock_path, runtime_lock)
    if args.lock_runtime_only:
        print(json.dumps(runtime_lock, indent=2, sort_keys=True))
        return 0

    if args.smoke_only:
        summary = run_smoke(
            root,
            config,
            config_sha256,
            receptors,
            ligands,
            executable,
            args.opencl_binary_path.resolve(),
            runtime_lock,
            run_directory,
            args.resume,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    validate_smoke(
        root,
        run_directory,
        config,
        config_sha256,
        runtime_lock,
        receptors,
        ligands,
    )

    rows: list[dict[str, object]] = []
    pair_summaries: list[dict[str, object]] = []
    resumed_count = 0
    executed_count = 0
    expected_total = int(config["expected"]["total_gpu_pair_count"])
    invocation_started = time.perf_counter()
    completed_count = 0
    for seed_descriptor in config["inputs"]["cpu_seed_runs"]:
        seed_id = str(seed_descriptor["seed_id"])
        base_seed = int(seed_descriptor["base_seed"])
        for receptor in receptors:
            print(f"running {seed_id}/{receptor['conformer_id']}", flush=True)
            for ligand in ligands:
                paths = pair_paths(
                    run_directory,
                    seed_id,
                    receptor["conformer_id"],
                    ligand["ligand_id"],
                )
                row, summary, resumed = run_pair(
                    root,
                    paths,
                    executable,
                    args.opencl_binary_path.resolve(),
                    runtime_lock,
                    config_sha256,
                    receptor,
                    ligand,
                    config["vinagpu"],
                    seed_id,
                    base_seed,
                    args.resume,
                )
                rows.append(row)
                pair_summaries.append(summary)
                resumed_count += int(resumed)
                executed_count += int(not resumed)
                completed_count += 1
                if completed_count % 25 == 0 or completed_count == expected_total:
                    print(
                        f"progress={completed_count}/{expected_total} "
                        f"executed={executed_count} resumed={resumed_count}",
                        flush=True,
                    )

    keys = {
        (str(row["seed_id"]), str(row["receptor_id"]), str(row["ligand_id"]))
        for row in rows
    }
    if len(rows) != expected_total or len(keys) != expected_total:
        raise RuntimeError("completed Vina-GPU pair set is incomplete or duplicated")

    scores_path = rooted_path(root, str(outputs["gpu_scores_csv"]))
    pair_runs_path = rooted_path(root, str(outputs["gpu_pair_runs_csv"]))
    summary_path = rooted_path(root, str(outputs["gpu_summary_json"]))
    write_csv(scores_path, rows)
    pair_run_rows = [
        {
            "seed_id": row["seed_id"],
            "receptor_id": row["receptor_id"],
            "ligand_id": row["ligand_id"],
            "pair_seed": row["pair_seed"],
            "elapsed_seconds": row["elapsed_seconds"],
            "pair_summary_path": relative_path(
                root,
                pair_paths(
                    run_directory,
                    str(row["seed_id"]),
                    str(row["receptor_id"]),
                    str(row["ligand_id"]),
                )["summary"],
            ),
            "log_path": row["log_path"],
            "log_sha256": row["log_sha256"],
        }
        for row in pair_summaries
    ]
    write_csv(pair_runs_path, pair_run_rows)
    pair_seconds = sum(float(row["elapsed_seconds"]) for row in pair_summaries)
    invocation_seconds = time.perf_counter() - invocation_started
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "operation": "consumed Train-160 single-pair AutoDock Vina-GPU 2.1 execution",
        "config": {
            "path": relative_path(root, config_path),
            "sha256": config_sha256,
        },
        "runtime_lock": output_descriptor(root, lock_path),
        "seed_policy": config["seed_policy"],
        "gpu_pair_count": len(rows),
        "receptor_count": len(receptors),
        "ligand_count": len(ligands),
        "seed_count": len(config["inputs"]["cpu_seed_runs"]),
        "executed_pair_count_this_invocation": executed_count,
        "resumed_pair_count_this_invocation": resumed_count,
        "gpu_pair_elapsed_seconds_total": pair_seconds,
        "invocation_wall_seconds": invocation_seconds,
        "pairs_per_second": len(rows) / pair_seconds,
        "score_minimum": min(float(row["gpu_vinagpu21_score"]) for row in rows),
        "score_maximum": max(float(row["gpu_vinagpu21_score"]) for row in rows),
        "outputs": {
            "gpu_scores_csv": output_descriptor(root, scores_path),
            "gpu_pair_runs_csv": output_descriptor(root, pair_runs_path),
        },
        "validation_rows": 0,
        "test_rows": 0,
        "enrichment_metrics_calculated": False,
        "interpretation_note": config["decision_boundary"],
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
