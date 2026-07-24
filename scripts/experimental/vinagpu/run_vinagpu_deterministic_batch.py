"""Run the audited deterministic-batch Vina-GPU 2.1 bridge diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .run_vinagpu_equivalence import (
        BLOCKED_LOG_PATTERNS,
        ensure_runtime_lock,
        file_sha256,
        output_descriptor,
        parse_vina_pose,
        read_csv,
        read_json,
        relative_path,
        resolve_executable,
        rooted_path,
        runtime_evidence,
        validate_inputs,
        write_csv,
        write_json,
    )
except ImportError:
    from run_vinagpu_equivalence import (
        BLOCKED_LOG_PATTERNS,
        ensure_runtime_lock,
        file_sha256,
        output_descriptor,
        parse_vina_pose,
        read_csv,
        read_json,
        relative_path,
        resolve_executable,
        rooted_path,
        runtime_evidence,
        validate_inputs,
        write_csv,
        write_json,
    )


def verified_descriptor(root: Path, descriptor: dict[str, object]) -> Path:
    path = rooted_path(root, str(descriptor["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"configured SHA-256 differs: {path}")
    return path


def validate_bridge_inputs(
    root: Path, config: dict[str, object]
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[tuple[str, str, str], dict[str, str]],
    dict[str, object],
]:
    receptors, ligands, audit = validate_inputs(root, config)
    reference_path = verified_descriptor(
        root, config["inputs"]["single_pair_gpu_reference"]
    )
    result_summary_path = verified_descriptor(
        root, config["inputs"]["single_pair_result_summary"]
    )
    result_summary = read_json(result_summary_path)
    if result_summary.get("status") != "gpu_equivalence_gate_failed":
        raise ValueError("single-pair reference must retain its frozen failed status")
    reference_rows = read_csv(reference_path)
    expected_count = int(config["expected"]["total_gpu_pair_count"])
    if len(reference_rows) != expected_count:
        raise ValueError("single-pair GPU reference row count differs")

    reference: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in reference_rows:
        key = (row["seed_id"], row["receptor_id"], row["ligand_id"])
        if key in reference:
            raise ValueError(f"duplicate single-pair GPU reference: {key}")
        score = float(row["gpu_vinagpu21_score"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite single-pair GPU reference: {key}")
        if int(row["pair_seed"]) != int(row["base_seed"]) + int(
            row["seed_offset"]
        ):
            raise ValueError(f"single-pair GPU seed policy differs: {key}")
        if len(row["output_pose_sha256"]) != 64:
            raise ValueError(f"single-pair pose hash is invalid: {key}")
        reference[key] = row

    expected_keys = {
        (str(seed["seed_id"]), receptor["conformer_id"], ligand["ligand_id"])
        for seed in config["inputs"]["cpu_seed_runs"]
        for receptor in receptors
        for ligand in ligands
    }
    if set(reference) != expected_keys:
        raise ValueError("single-pair GPU reference keys differ")
    chunk_size = int(config["expected"]["chunk_size"])
    if chunk_size <= 1 or len(ligands) % chunk_size != 0:
        raise ValueError("chunk size must evenly divide the ligand count")
    chunk_count = (
        len(config["inputs"]["cpu_seed_runs"])
        * len(receptors)
        * (len(ligands) // chunk_size)
    )
    if chunk_count != int(config["expected"]["chunk_count"]):
        raise ValueError("derived chunk count differs")
    audit.update(
        {
            "single_pair_reference_path": relative_path(root, reference_path),
            "single_pair_reference_sha256": file_sha256(reference_path),
            "single_pair_reference_rows": len(reference),
            "frozen_v1_status": result_summary["status"],
            "chunk_size": chunk_size,
            "chunk_count": chunk_count,
        }
    )
    return receptors, ligands, reference, audit


def verify_patch_manifest(
    path: Path, patch_config: dict[str, object]
) -> dict[str, object]:
    manifest = read_json(path)
    if manifest.get("status") != "ok":
        raise ValueError("deterministic batch patch manifest is not ok")
    for key in (
        "patch_id",
        "source_commit",
        "source_files",
        "opencl_kernel_sources_modified",
        "scoring_or_search_code_modified",
    ):
        if manifest.get(key) != patch_config.get(key):
            raise ValueError(f"deterministic batch patch differs: {key}")
    return manifest


def locked_runtime_evidence(
    executable_value: str,
    kernel_directory: Path,
    source_tree: Path,
    patch_manifest_path: Path,
    config: dict[str, object],
) -> dict[str, object]:
    evidence = runtime_evidence(
        executable_value,
        kernel_directory,
        source_tree,
        config["vinagpu"],
    )
    protocol = config["vinagpu"]
    if evidence["kernel1_sha256"] != protocol["expected_kernel1_sha256"]:
        raise ValueError("Kernel1 differs from the frozen single-pair runtime")
    if evidence["kernel2_sha256"] != protocol["expected_kernel2_sha256"]:
        raise ValueError("Kernel2 differs from the frozen single-pair runtime")
    patch_manifest_path = patch_manifest_path.resolve()
    patch_manifest = verify_patch_manifest(
        patch_manifest_path, config["patch"]
    )
    evidence["deterministic_batch_patch"] = {
        **patch_manifest,
        "manifest_path": str(patch_manifest_path),
        "manifest_sha256": file_sha256(patch_manifest_path),
    }
    return evidence


def ligand_chunks(
    ligands: list[dict[str, str]], chunk_size: int
) -> list[list[dict[str, str]]]:
    return [
        ligands[start : start + chunk_size]
        for start in range(0, len(ligands), chunk_size)
    ]


def staged_ligand_name(ligand: dict[str, str]) -> str:
    return f"{int(ligand['seed_offset']):06d}__{ligand['ligand_id']}.pdbqt"


def chunk_paths(
    run_directory: Path,
    seed_id: str,
    receptor_id: str,
    chunk_index: int,
) -> dict[str, Path]:
    directory = (
        run_directory
        / "chunks"
        / seed_id
        / receptor_id
        / f"chunk_{chunk_index:03d}"
    )
    return {
        "directory": directory,
        "input_directory": directory / "inputs",
        "output_directory": directory / "outputs",
        "log": directory / "vinagpu.log",
        "scores": directory / "scores.csv",
        "summary": directory / "chunk_summary.json",
    }


def stage_ligands(
    root: Path,
    directory: Path,
    ligands: list[dict[str, str]],
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    expected_names: set[str] = set()
    for ligand in ligands:
        name = staged_ligand_name(ligand)
        expected_names.add(name)
        source = rooted_path(root, ligand["pdbqt_path"])
        destination = directory / name
        if os.path.lexists(destination):
            if not destination.is_symlink() or destination.resolve() != source:
                raise ValueError(f"staged ligand path differs: {destination}")
        else:
            destination.symlink_to(source)
        staged[ligand["ligand_id"]] = destination
    observed_names = {path.name for path in directory.iterdir()}
    if observed_names != expected_names:
        raise ValueError(
            f"staging directory contains unexpected entries: {directory}"
        )
    return staged


def batch_command(
    executable: Path,
    kernel_directory: Path,
    receptor_path: Path,
    input_directory: Path,
    output_directory: Path,
    protocol: dict[str, object],
    seed: int,
) -> list[str]:
    box = protocol["box"]
    command = [
        str(executable),
        "--receptor",
        str(receptor_path),
        "--ligand_directory",
        str(input_directory),
        "--output_directory",
        str(output_directory),
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
    if protocol["search_depth"] != "heuristic":
        command.extend(["--search_depth", str(protocol["search_depth"])])
    return command


def chunk_signature(
    config_sha256: str,
    runtime_lock: dict[str, object],
    seed_id: str,
    receptor: dict[str, str],
    ligands: list[dict[str, str]],
    reference: dict[tuple[str, str, str], dict[str, str]],
    protocol: dict[str, object],
) -> str:
    value = {
        "config_sha256": config_sha256,
        "runtime_identity": runtime_lock,
        "seed_id": seed_id,
        "receptor_id": receptor["conformer_id"],
        "receptor_sha256": receptor["receptor_pdbqt_sha256"],
        "ligands": [
            {
                "ligand_id": ligand["ligand_id"],
                "seed_offset": int(ligand["seed_offset"]),
                "sha256": ligand["pdbqt_sha256"],
                "reference_score": reference[
                    (seed_id, receptor["conformer_id"], ligand["ligand_id"])
                ]["gpu_vinagpu21_score"],
                "reference_pose_sha256": reference[
                    (seed_id, receptor["conformer_id"], ligand["ligand_id"])
                ]["output_pose_sha256"],
            }
            for ligand in ligands
        ],
        "protocol": protocol,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest().upper()


def checkpoint(
    root: Path,
    paths: dict[str, Path],
    signature: str,
    expected_ligand_ids: set[str],
) -> tuple[list[dict[str, str]], dict[str, object]] | None:
    if not paths["summary"].is_file() or not paths["scores"].is_file():
        return None
    try:
        summary = read_json(paths["summary"])
        rows = read_csv(paths["scores"])
        if summary.get("status") != "ok" or summary.get("signature") != signature:
            return None
        if len(rows) != len(expected_ligand_ids):
            return None
        if {row["ligand_id"] for row in rows} != expected_ligand_ids:
            return None
        if any(row["exact_score_match"] != "True" for row in rows):
            return None
        if any(row["exact_pose_hash_match"] != "True" for row in rows):
            return None
        if file_sha256(paths["scores"]) != summary["scores_sha256"]:
            return None
        for row in rows:
            pose = rooted_path(root, row["output_pose_path"])
            if file_sha256(pose) != row["output_pose_sha256"]:
                return None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return rows, summary


def run_chunk(
    root: Path,
    paths: dict[str, Path],
    executable: Path,
    kernel_directory: Path,
    runtime_lock: dict[str, object],
    config_sha256: str,
    receptor: dict[str, str],
    ligands: list[dict[str, str]],
    reference: dict[tuple[str, str, str], dict[str, str]],
    protocol: dict[str, object],
    seed_id: str,
    base_seed: int,
    chunk_index: int,
    resume: bool,
) -> tuple[list[dict[str, object]], dict[str, object], bool]:
    signature = chunk_signature(
        config_sha256,
        runtime_lock,
        seed_id,
        receptor,
        ligands,
        reference,
        protocol,
    )
    ligand_ids = {ligand["ligand_id"] for ligand in ligands}
    if resume:
        saved = checkpoint(root, paths, signature, ligand_ids)
        if saved is not None:
            return saved[0], saved[1], True

    staged = stage_ligands(root, paths["input_directory"], ligands)
    paths["output_directory"].mkdir(parents=True, exist_ok=True)
    receptor_path = rooted_path(root, receptor["receptor_pdbqt"])
    first_offset = int(ligands[0]["seed_offset"])
    offsets = [int(ligand["seed_offset"]) for ligand in ligands]
    if offsets != list(range(first_offset, first_offset + len(ligands))):
        raise ValueError("chunk seed offsets are not contiguous")
    chunk_seed = base_seed + first_offset
    command = batch_command(
        executable,
        kernel_directory,
        receptor_path,
        paths["input_directory"],
        paths["output_directory"],
        protocol,
        chunk_seed,
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
            f"Vina-GPU batch failed for {seed_id}/{receptor['conformer_id']}/"
            f"chunk{chunk_index} with exit code {result.returncode}; see {paths['log']}"
        )
    blocked = [pattern for pattern in BLOCKED_LOG_PATTERNS if pattern in log_text]
    if blocked:
        raise RuntimeError(f"blocked Vina-GPU batch log pattern(s): {blocked}")

    expected_output_names = {
        f"{path.stem}_out.pdbqt" for path in staged.values()
    }
    observed_output_names = {
        path.name for path in paths["output_directory"].iterdir()
    }
    if observed_output_names != expected_output_names:
        raise RuntimeError(
            "Vina-GPU batch output set differs; "
            f"expected={sorted(expected_output_names)}, "
            f"observed={sorted(observed_output_names)}"
        )

    rows: list[dict[str, object]] = []
    maximum_score_delta = 0.0
    for local_index, ligand in enumerate(ligands):
        input_path = staged[ligand["ligand_id"]]
        output_name = f"{input_path.stem}_out.pdbqt"
        output_path = paths["output_directory"] / output_name
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"missing Vina-GPU batch pose: {output_path}")
        score, pose_count = parse_vina_pose(output_path)
        guard = float(protocol["maximum_absolute_score_kcal_per_mol"])
        if abs(score) > guard:
            raise ValueError(f"nonphysical Vina-GPU score: {score}")
        key = (seed_id, receptor["conformer_id"], ligand["ligand_id"])
        expected = reference[key]
        expected_score = float(expected["gpu_vinagpu21_score"])
        score_delta = score - expected_score
        output_hash = file_sha256(output_path)
        score_match = score == expected_score
        pose_match = output_hash == expected["output_pose_sha256"]
        maximum_score_delta = max(maximum_score_delta, abs(score_delta))
        rows.append(
            {
                "target_id": "MK14",
                "seed_id": seed_id,
                "base_seed": base_seed,
                "seed_offset": int(ligand["seed_offset"]),
                "pair_seed": chunk_seed + local_index,
                "receptor_id": receptor["conformer_id"],
                "ligand_id": ligand["ligand_id"],
                "label": ligand["label"],
                "batch_chunk_index": chunk_index,
                "batch_local_index": local_index,
                "batch_score": score,
                "single_pair_reference_score": expected_score,
                "score_delta": score_delta,
                "exact_score_match": score_match,
                "pose_count": pose_count,
                "output_pose_path": relative_path(root, output_path),
                "output_pose_sha256": output_hash,
                "single_pair_reference_pose_sha256": expected[
                    "output_pose_sha256"
                ],
                "exact_pose_hash_match": pose_match,
                "status": "ok" if score_match and pose_match else "bridge_mismatch",
            }
        )
    write_csv(paths["scores"], rows)
    exact_scores = sum(bool(row["exact_score_match"]) for row in rows)
    exact_poses = sum(bool(row["exact_pose_hash_match"]) for row in rows)
    bridge_ok = exact_scores == len(rows) and exact_poses == len(rows)
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "status": "ok" if bridge_ok else "bridge_mismatch",
        "signature": signature,
        "seed_id": seed_id,
        "base_seed": base_seed,
        "chunk_seed": chunk_seed,
        "receptor_id": receptor["conformer_id"],
        "chunk_index": chunk_index,
        "first_seed_offset": first_offset,
        "ligand_count": len(rows),
        "exact_score_match_count": exact_scores,
        "exact_pose_hash_match_count": exact_poses,
        "maximum_absolute_score_delta": maximum_score_delta,
        "elapsed_seconds": elapsed,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "log_path": relative_path(root, paths["log"]),
        "log_sha256": file_sha256(paths["log"]),
        "scores_path": relative_path(root, paths["scores"]),
        "scores_sha256": file_sha256(paths["scores"]),
    }
    write_json(paths["summary"], summary)
    return rows, summary, False


def validate_preflight(
    root: Path,
    config: dict[str, object],
    run_directory: Path,
    config_sha256: str,
    runtime_lock: dict[str, object],
) -> None:
    path = rooted_path(root, str(config["outputs"]["preflight_summary_json"]))
    if not path.is_file():
        raise RuntimeError("deterministic batch preflight is missing")
    summary = read_json(path)
    if summary.get("status") != "exact_bridge_preflight_passed":
        raise RuntimeError("deterministic batch preflight did not pass")
    if summary.get("config_sha256") != config_sha256:
        raise RuntimeError("deterministic batch preflight config differs")
    if summary.get("runtime_identity_sha256") != hashlib.sha256(
        json.dumps(
            runtime_lock, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest().upper():
        raise RuntimeError("deterministic batch preflight runtime differs")
    if not str(summary["chunk_summary_path"]).startswith(
        relative_path(root, run_directory)
    ):
        raise RuntimeError("preflight summary points outside the run directory")


def build_bridge_summary(
    config: dict[str, object],
    pair_count: int,
    score_matches: int,
    pose_matches: int,
    maximum_delta: float,
    elapsed_seconds: float,
) -> dict[str, object]:
    references = config["throughput_references"]
    cpu = references["cpu_32vcpu"]
    single = references["single_pair_gpu"]
    batch_rate = pair_count / elapsed_seconds
    cpu_rate = float(cpu["pair_count"]) / float(cpu["elapsed_seconds"])
    single_rate = float(single["pair_count"]) / float(single["elapsed_seconds"])
    speedup_cpu = batch_rate / cpu_rate
    speedup_single = batch_rate / single_rate
    gate = config["bridge_gate"]
    checks = {
        "complete_pairs": {
            "observed": pair_count,
            "threshold": int(config["expected"]["total_gpu_pair_count"]),
            "passed": pair_count == int(config["expected"]["total_gpu_pair_count"]),
        },
        "exact_score_matches": {
            "observed": score_matches,
            "threshold": pair_count,
            "passed": score_matches == pair_count,
        },
        "exact_pose_hash_matches": {
            "observed": pose_matches,
            "threshold": pair_count,
            "passed": pose_matches == pair_count,
        },
        "maximum_absolute_score_delta": {
            "observed": maximum_delta,
            "threshold": float(gate["maximum_absolute_score_delta_kcal_per_mol"]),
            "passed": maximum_delta
            <= float(gate["maximum_absolute_score_delta_kcal_per_mol"]),
        },
        "speedup_vs_recorded_32vcpu": {
            "observed": speedup_cpu,
            "threshold": float(gate["minimum_throughput_speedup_vs_recorded_32vcpu"]),
            "passed": speedup_cpu
            >= float(gate["minimum_throughput_speedup_vs_recorded_32vcpu"]),
        },
        "speedup_vs_single_pair_gpu": {
            "observed": speedup_single,
            "threshold": float(gate["minimum_throughput_speedup_vs_single_pair_gpu"]),
            "passed": speedup_single
            >= float(gate["minimum_throughput_speedup_vs_single_pair_gpu"]),
        },
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "deterministic_batch_bridge_passed"
            if passed
            else "deterministic_batch_bridge_failed"
        ),
        "all_gate_checks_passed": passed,
        "gate_checks": checks,
        "throughput": {
            "batch_pair_count": pair_count,
            "batch_elapsed_seconds": elapsed_seconds,
            "batch_pairs_per_second": batch_rate,
            "speedup_vs_recorded_32vcpu": speedup_cpu,
            "speedup_vs_single_pair_gpu": speedup_single,
        },
        "validation_rows": 0,
        "test_rows": 0,
        "frozen_v1_gate_status_unchanged": "gpu_equivalence_gate_failed",
        "interpretation_note": config["decision_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--vinagpu")
    parser.add_argument("--opencl-binary-path", type=Path)
    parser.add_argument("--source-tree", type=Path)
    parser.add_argument("--patch-manifest", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--lock-runtime-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    modes = sum(
        int(flag)
        for flag in (args.audit_only, args.lock_runtime_only, args.preflight_only)
    )
    if modes > 1:
        parser.error("choose at most one audit/lock/preflight mode")
    root = args.root.resolve()
    config_path = args.config.resolve()
    config = read_json(config_path)
    config_sha256 = file_sha256(config_path)
    receptors, ligands, reference, input_audit = validate_bridge_inputs(root, config)
    audit = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "audit_only_ok",
        "config": {
            "path": relative_path(root, config_path),
            "sha256": config_sha256,
        },
        **input_audit,
        "operation": "input audit only; no GPU batch was run",
    }
    if args.audit_only:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0

    required_runtime = (
        args.vinagpu,
        args.opencl_binary_path,
        args.source_tree,
        args.patch_manifest,
    )
    if any(value is None for value in required_runtime):
        parser.error(
            "--vinagpu, --opencl-binary-path, --source-tree, and "
            "--patch-manifest are required for runtime modes"
        )
    executable = resolve_executable(str(args.vinagpu))
    runtime_lock = locked_runtime_evidence(
        str(executable),
        args.opencl_binary_path.resolve(),
        args.source_tree.resolve(),
        args.patch_manifest.resolve(),
        config,
    )
    outputs = config["outputs"]
    run_directory = rooted_path(root, str(outputs["run_directory"]))
    run_directory.mkdir(parents=True, exist_ok=True)
    lock_path = rooted_path(root, str(outputs["runtime_lock_json"]))
    ensure_runtime_lock(lock_path, runtime_lock)
    if args.lock_runtime_only:
        print(json.dumps(runtime_lock, indent=2, sort_keys=True))
        return 0

    chunk_size = int(config["expected"]["chunk_size"])
    chunks = ligand_chunks(ligands, chunk_size)
    preflight = config["preflight"]
    if args.preflight_only:
        seed_descriptor = next(
            seed
            for seed in config["inputs"]["cpu_seed_runs"]
            if seed["seed_id"] == preflight["seed_id"]
        )
        receptor = next(
            row
            for row in receptors
            if row["conformer_id"] == preflight["receptor_id"]
        )
        chunk_index = int(preflight["chunk_index"])
        rows, summary, resumed = run_chunk(
            root,
            chunk_paths(
                run_directory,
                str(seed_descriptor["seed_id"]),
                receptor["conformer_id"],
                chunk_index,
            ),
            executable,
            args.opencl_binary_path.resolve(),
            runtime_lock,
            config_sha256,
            receptor,
            chunks[chunk_index],
            reference,
            config["vinagpu"],
            str(seed_descriptor["seed_id"]),
            int(seed_descriptor["base_seed"]),
            chunk_index,
            args.resume,
        )
        passed = (
            summary["status"] == "ok"
            and len(rows) == int(preflight["expected_pair_count"])
        )
        preflight_summary = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "status": (
                "exact_bridge_preflight_passed"
                if passed
                else "exact_bridge_preflight_failed"
            ),
            "config_sha256": config_sha256,
            "runtime_identity_sha256": hashlib.sha256(
                json.dumps(
                    runtime_lock, sort_keys=True, separators=(",", ":")
                ).encode("ascii")
            ).hexdigest().upper(),
            "resumed": resumed,
            "pair_count": len(rows),
            "exact_score_match_count": summary["exact_score_match_count"],
            "exact_pose_hash_match_count": summary[
                "exact_pose_hash_match_count"
            ],
            "maximum_absolute_score_delta": summary[
                "maximum_absolute_score_delta"
            ],
            "elapsed_seconds": summary["elapsed_seconds"],
            "chunk_summary_path": relative_path(
                root,
                chunk_paths(
                    run_directory,
                    str(seed_descriptor["seed_id"]),
                    receptor["conformer_id"],
                    chunk_index,
                )["summary"],
            ),
        }
        preflight_path = rooted_path(
            root, str(outputs["preflight_summary_json"])
        )
        write_json(preflight_path, preflight_summary)
        print(json.dumps(preflight_summary, indent=2, sort_keys=True))
        return 0 if passed else 2

    validate_preflight(
        root,
        config,
        run_directory,
        config_sha256,
        runtime_lock,
    )
    all_rows: list[dict[str, object]] = []
    all_summaries: list[dict[str, object]] = []
    resumed_count = 0
    executed_count = 0
    completed_chunks = 0
    expected_chunks = int(config["expected"]["chunk_count"])
    invocation_started = time.perf_counter()
    for seed in config["inputs"]["cpu_seed_runs"]:
        seed_id = str(seed["seed_id"])
        base_seed = int(seed["base_seed"])
        for receptor in receptors:
            print(f"running {seed_id}/{receptor['conformer_id']}", flush=True)
            for chunk_index, ligand_chunk in enumerate(chunks):
                rows, summary, resumed = run_chunk(
                    root,
                    chunk_paths(
                        run_directory,
                        seed_id,
                        receptor["conformer_id"],
                        chunk_index,
                    ),
                    executable,
                    args.opencl_binary_path.resolve(),
                    runtime_lock,
                    config_sha256,
                    receptor,
                    ligand_chunk,
                    reference,
                    config["vinagpu"],
                    seed_id,
                    base_seed,
                    chunk_index,
                    args.resume,
                )
                if summary["status"] != "ok":
                    raise RuntimeError(
                        f"deterministic bridge mismatch: {summary['scores_path']}"
                    )
                all_rows.extend(rows)
                all_summaries.append(summary)
                resumed_count += int(resumed)
                executed_count += int(not resumed)
                completed_chunks += 1
                if completed_chunks % 10 == 0 or completed_chunks == expected_chunks:
                    print(
                        f"chunks={completed_chunks}/{expected_chunks} "
                        f"executed={executed_count} resumed={resumed_count}",
                        flush=True,
                    )

    expected_pairs = int(config["expected"]["total_gpu_pair_count"])
    keys = {
        (str(row["seed_id"]), str(row["receptor_id"]), str(row["ligand_id"]))
        for row in all_rows
    }
    if len(all_rows) != expected_pairs or len(keys) != expected_pairs:
        raise RuntimeError("deterministic batch pair set is incomplete or duplicated")
    scores_path = rooted_path(root, str(outputs["batch_scores_csv"]))
    runs_path = rooted_path(root, str(outputs["batch_runs_csv"]))
    run_summary_path = rooted_path(root, str(outputs["batch_summary_json"]))
    bridge_summary_path = rooted_path(root, str(outputs["bridge_summary_json"]))
    write_csv(scores_path, all_rows)
    run_rows = [
        {
            "seed_id": summary["seed_id"],
            "receptor_id": summary["receptor_id"],
            "chunk_index": summary["chunk_index"],
            "chunk_seed": summary["chunk_seed"],
            "ligand_count": summary["ligand_count"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "scores_path": summary["scores_path"],
            "scores_sha256": summary["scores_sha256"],
            "log_path": summary["log_path"],
            "log_sha256": summary["log_sha256"],
        }
        for summary in all_summaries
    ]
    write_csv(runs_path, run_rows)
    elapsed_total = sum(float(summary["elapsed_seconds"]) for summary in all_summaries)
    score_matches = sum(bool(row["exact_score_match"]) for row in all_rows)
    pose_matches = sum(bool(row["exact_pose_hash_match"]) for row in all_rows)
    maximum_delta = max(abs(float(row["score_delta"])) for row in all_rows)
    run_summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "pair_count": len(all_rows),
        "chunk_count": len(all_summaries),
        "chunk_size": chunk_size,
        "executed_chunk_count_this_invocation": executed_count,
        "resumed_chunk_count_this_invocation": resumed_count,
        "batch_elapsed_seconds_total": elapsed_total,
        "invocation_wall_seconds": time.perf_counter() - invocation_started,
        "exact_score_match_count": score_matches,
        "exact_pose_hash_match_count": pose_matches,
        "maximum_absolute_score_delta": maximum_delta,
        "runtime_lock": output_descriptor(root, lock_path),
        "outputs": {
            "batch_scores_csv": output_descriptor(root, scores_path),
            "batch_runs_csv": output_descriptor(root, runs_path),
        },
        "validation_rows": 0,
        "test_rows": 0,
    }
    write_json(run_summary_path, run_summary)
    bridge_summary = build_bridge_summary(
        config,
        len(all_rows),
        score_matches,
        pose_matches,
        maximum_delta,
        elapsed_total,
    )
    bridge_summary["config"] = {
        "path": relative_path(root, config_path),
        "sha256": config_sha256,
    }
    bridge_summary["batch_run_summary"] = output_descriptor(
        root, run_summary_path
    )
    write_json(bridge_summary_path, bridge_summary)
    print(json.dumps(bridge_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
