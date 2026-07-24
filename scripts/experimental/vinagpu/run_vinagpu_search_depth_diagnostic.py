"""Run the targeted consumed-train Vina-GPU search-depth diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .audit_vinagpu_equivalence import group_metrics, quantile
    from .run_vinagpu_deterministic_batch import (
        batch_command,
        ligand_chunks,
        locked_runtime_evidence,
        stage_ligands,
        validate_bridge_inputs,
    )
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
        write_csv,
        write_json,
    )
except ImportError:
    from audit_vinagpu_equivalence import group_metrics, quantile
    from run_vinagpu_deterministic_batch import (
        batch_command,
        ligand_chunks,
        locked_runtime_evidence,
        stage_ligands,
        validate_bridge_inputs,
    )
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
        write_csv,
        write_json,
    )


def verified_json(root: Path, descriptor: dict[str, object]) -> dict[str, object]:
    path = rooted_path(root, str(descriptor["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"configured SHA-256 differs: {path}")
    return read_json(path)


def runtime_projection(runtime: dict[str, object]) -> dict[str, object]:
    patch = dict(runtime["deterministic_batch_patch"])
    patch.pop("manifest_path", None)
    return {
        key: runtime[key]
        for key in (
            "status",
            "source_repository",
            "source_commit",
            "method",
            "version_probe",
            "executable_sha256",
            "kernel1_sha256",
            "kernel2_sha256",
            "makefile_sha256",
            "build_settings",
            "kernel_mode",
        )
    } | {"deterministic_batch_patch": patch}


def validate_approved_runtime(
    observed: dict[str, object], approved: dict[str, object]
) -> None:
    if runtime_projection(observed) != runtime_projection(approved):
        raise ValueError("runtime differs from the passed deterministic bridge")


def heuristic_depth(ligand: dict[str, str]) -> int:
    value = (
        0.24 * int(ligand["pdbqt_atom_count"]) + 0.29 * int(ligand["torsdof"]) - 3.41
    ) * 1.5
    return int(max(1.0, value))


def read_target_cpu_scores(
    root: Path,
    config: dict[str, object],
    target_keys: set[tuple[str, str]],
) -> dict[tuple[str, str, str], float]:
    scores: dict[tuple[str, str, str], float] = {}
    for seed in config["inputs"]["cpu_seed_runs"]:
        seed_id = str(seed["seed_id"])
        receptors = {
            receptor_id
            for target_seed, receptor_id in target_keys
            if target_seed == seed_id
        }
        if not receptors:
            continue
        for row in read_csv(rooted_path(root, str(seed["scores_path"]))):
            if row["receptor_id"] not in receptors:
                continue
            key = (seed_id, row["receptor_id"], row["ligand_id"])
            if key in scores:
                raise ValueError(f"duplicate CPU target score: {key}")
            value = float(row["representative_score"])
            if not math.isfinite(value) or row["status"] != "ok":
                raise ValueError(f"invalid CPU target score: {key}")
            scores[key] = value
    return scores


def comparison_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            **row,
            "cpu_vina_e32_score": float(row["cpu_vina_e32_score"]),
            "gpu_vinagpu21_score": float(row["gpu_vinagpu21_score"]),
            "score_delta_gpu_minus_cpu": float(row["score_delta_gpu_minus_cpu"]),
        }
        for row in rows
    ]


def baseline_rows(
    target_keys: set[tuple[str, str]],
    ligands: list[dict[str, str]],
    reference: dict[tuple[str, str, str], dict[str, str]],
    cpu_scores: dict[tuple[str, str, str], float],
) -> list[dict[str, object]]:
    ligand_map = {row["ligand_id"]: row for row in ligands}
    rows: list[dict[str, object]] = []
    for key in sorted(cpu_scores):
        seed_id, receptor_id, ligand_id = key
        if (seed_id, receptor_id) not in target_keys:
            continue
        gpu = float(reference[key]["gpu_vinagpu21_score"])
        cpu = cpu_scores[key]
        rows.append(
            {
                "seed_id": seed_id,
                "receptor_id": receptor_id,
                "ligand_id": ligand_id,
                "label": ligand_map[ligand_id]["label"],
                "cpu_vina_e32_score": cpu,
                "gpu_vinagpu21_score": gpu,
                "score_delta_gpu_minus_cpu": gpu - cpu,
            }
        )
    return rows


def validate_diagnostic_inputs(root: Path, config: dict[str, object]) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[tuple[str, str, str], dict[str, str]],
    dict[tuple[str, str, str], float],
    dict[str, object],
]:
    receptors, ligands, reference, audit = validate_bridge_inputs(root, config)
    bridge = verified_json(root, config["inputs"]["deterministic_batch_bridge_result"])
    if (
        bridge.get("status") != "deterministic_batch_bridge_passed"
        or not bridge.get("all_gate_checks_passed")
        or int(bridge["batch_pair_count"]) != 2400
        or int(bridge["exact_score_match_count"]) != 2400
        or int(bridge["exact_pose_hash_match_count"]) != 2400
        or int(bridge["validation_rows"]) != 0
        or int(bridge["test_rows"]) != 0
    ):
        raise ValueError("deterministic batch bridge evidence is not an exact pass")
    approved_runtime = verified_json(
        root, config["inputs"]["approved_deterministic_batch_runtime"]
    )
    if approved_runtime.get("status") != "locked":
        raise ValueError("approved deterministic runtime is not locked")

    diagnostic = config["search_depth_diagnostic"]
    target_groups = list(diagnostic["target_groups"])
    target_keys = {
        (str(row["seed_id"]), str(row["receptor_id"])) for row in target_groups
    }
    expected_group_count = int(config["expected"]["diagnostic_group_count"])
    if (
        len(target_groups) != expected_group_count
        or len(target_keys) != expected_group_count
    ):
        raise ValueError("diagnostic target groups differ or are duplicated")
    available_seeds = {str(row["seed_id"]) for row in config["inputs"]["cpu_seed_runs"]}
    available_receptors = {row["conformer_id"] for row in receptors}
    if any(
        seed_id not in available_seeds or receptor_id not in available_receptors
        for seed_id, receptor_id in target_keys
    ):
        raise ValueError("diagnostic target group is not available")

    profiles = list(diagnostic["profiles"])
    depths = [int(row["search_depth"]) for row in profiles]
    profile_ids = [str(row["profile_id"]) for row in profiles]
    if (
        not profiles
        or depths != sorted(set(depths))
        or len(profile_ids) != len(set(profile_ids))
        or any(depth < 1 for depth in depths)
    ):
        raise ValueError("search-depth profiles must be unique and increasing")
    observed_depths = sorted(heuristic_depth(ligand) for ligand in ligands)
    heuristic = diagnostic["heuristic_reference"]
    expected_depth_summary = {
        "observed_ligand_count": len(observed_depths),
        "observed_minimum": observed_depths[0],
        "observed_median": int(statistics.median(observed_depths)),
        "observed_maximum": observed_depths[-1],
    }
    for key, observed in expected_depth_summary.items():
        if int(heuristic[key]) != observed:
            raise ValueError(f"heuristic search-depth evidence differs: {key}")
    if bool(heuristic["first_fixed_depth_must_cover_all_heuristic_depths"]):
        if depths[0] < observed_depths[-1]:
            raise ValueError("first fixed depth does not cover the heuristic maximum")

    chunk_size = int(config["expected"]["chunk_size"])
    chunks = ligand_chunks(ligands, chunk_size)
    derived_pairs = len(target_keys) * len(ligands)
    derived_chunks = len(target_keys) * len(chunks)
    if derived_pairs != int(config["expected"]["diagnostic_pair_count_per_profile"]):
        raise ValueError("diagnostic pair count differs")
    if derived_chunks != int(config["expected"]["diagnostic_chunk_count_per_profile"]):
        raise ValueError("diagnostic chunk count differs")

    cpu_scores = read_target_cpu_scores(root, config, target_keys)
    expected_cpu_keys = {
        (seed_id, receptor_id, ligand["ligand_id"])
        for seed_id, receptor_id in target_keys
        for ligand in ligands
    }
    if set(cpu_scores) != expected_cpu_keys:
        raise ValueError("target CPU score keys differ")
    frozen_rows = baseline_rows(target_keys, ligands, reference, cpu_scores)
    frozen_groups = group_metrics(frozen_rows)
    frozen_by_key = {
        (str(row["seed_id"]), str(row["receptor_id"])): row for row in frozen_groups
    }
    for target in target_groups:
        key = (str(target["seed_id"]), str(target["receptor_id"]))
        if not math.isclose(
            float(target["frozen_heuristic_spearman"]),
            float(frozen_by_key[key]["spearman"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"frozen heuristic Spearman differs: {key}")

    audit.update(
        {
            "operation": "targeted consumed-train search-depth input audit",
            "bridge_status": bridge["status"],
            "approved_runtime_sha256": config["inputs"][
                "approved_deterministic_batch_runtime"
            ]["sha256"],
            "diagnostic_target_groups": [
                {"seed_id": seed_id, "receptor_id": receptor_id}
                for seed_id, receptor_id in sorted(target_keys)
            ],
            "diagnostic_pair_count_per_profile": derived_pairs,
            "diagnostic_chunk_count_per_profile": derived_chunks,
            "profile_depths": depths,
            "heuristic_depth_distribution": {
                **expected_depth_summary,
                "observed_p25": observed_depths[
                    math.floor(0.25 * (len(observed_depths) - 1))
                ],
                "observed_p75": observed_depths[
                    math.floor(0.75 * (len(observed_depths) - 1))
                ],
                "observed_p95": observed_depths[
                    math.floor(0.95 * (len(observed_depths) - 1))
                ],
            },
            "frozen_heuristic_group_metrics": frozen_groups,
            "labels_used_for_profile_selection": False,
        }
    )
    return receptors, ligands, reference, cpu_scores, audit


def search_chunk_paths(
    profile_directory: Path,
    seed_id: str,
    receptor_id: str,
    chunk_index: int,
) -> dict[str, Path]:
    directory = (
        profile_directory
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


def search_chunk_signature(
    config_sha256: str,
    runtime_lock: dict[str, object],
    profile: dict[str, object],
    seed_id: str,
    receptor: dict[str, str],
    ligands: list[dict[str, str]],
    cpu_scores: dict[tuple[str, str, str], float],
    protocol: dict[str, object],
) -> str:
    value = {
        "config_sha256": config_sha256,
        "runtime_identity": runtime_projection(runtime_lock),
        "profile": profile,
        "seed_id": seed_id,
        "receptor_id": receptor["conformer_id"],
        "receptor_sha256": receptor["receptor_pdbqt_sha256"],
        "ligands": [
            {
                "ligand_id": ligand["ligand_id"],
                "seed_offset": int(ligand["seed_offset"]),
                "sha256": ligand["pdbqt_sha256"],
                "cpu_score": cpu_scores[
                    (seed_id, receptor["conformer_id"], ligand["ligand_id"])
                ],
            }
            for ligand in ligands
        ],
        "protocol": protocol,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def search_checkpoint(
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
        if any(row["status"] != "ok" for row in rows):
            return None
        if file_sha256(paths["scores"]) != summary["scores_sha256"]:
            return None
        if file_sha256(paths["log"]) != summary["log_sha256"]:
            return None
        for row in rows:
            pose = rooted_path(root, row["output_pose_path"])
            if file_sha256(pose) != row["output_pose_sha256"]:
                return None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return rows, summary


def clear_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            raise ValueError(f"unexpected directory in Vina-GPU output: {child}")
        child.unlink()


def run_search_chunk(
    root: Path,
    paths: dict[str, Path],
    executable: Path,
    kernel_directory: Path,
    runtime_lock: dict[str, object],
    config_sha256: str,
    profile: dict[str, object],
    receptor: dict[str, str],
    ligands: list[dict[str, str]],
    cpu_scores: dict[tuple[str, str, str], float],
    protocol: dict[str, object],
    seed_id: str,
    base_seed: int,
    chunk_index: int,
    resume: bool,
) -> tuple[list[dict[str, object]], dict[str, object], bool]:
    signature = search_chunk_signature(
        config_sha256,
        runtime_lock,
        profile,
        seed_id,
        receptor,
        ligands,
        cpu_scores,
        protocol,
    )
    ligand_ids = {ligand["ligand_id"] for ligand in ligands}
    if resume:
        saved = search_checkpoint(root, paths, signature, ligand_ids)
        if saved is not None:
            return saved[0], saved[1], True

    staged = stage_ligands(root, paths["input_directory"], ligands)
    clear_output_directory(paths["output_directory"])
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
            f"Vina-GPU search-depth batch failed for {seed_id}/"
            f"{receptor['conformer_id']}/chunk{chunk_index} with exit code "
            f"{result.returncode}; see {paths['log']}"
        )
    blocked = [pattern for pattern in BLOCKED_LOG_PATTERNS if pattern in log_text]
    if blocked:
        raise RuntimeError(f"blocked Vina-GPU batch log pattern(s): {blocked}")
    depth = int(profile["search_depth"])
    if f"Search_depth is fixed to {depth}" not in log_text:
        raise RuntimeError("Vina-GPU log does not confirm the fixed search depth")

    expected_output_names = {f"{path.stem}_out.pdbqt" for path in staged.values()}
    observed_output_names = {path.name for path in paths["output_directory"].iterdir()}
    if observed_output_names != expected_output_names:
        raise RuntimeError(
            "Vina-GPU batch output set differs; "
            f"expected={sorted(expected_output_names)}, "
            f"observed={sorted(observed_output_names)}"
        )

    rows: list[dict[str, object]] = []
    for local_index, ligand in enumerate(ligands):
        input_path = staged[ligand["ligand_id"]]
        output_path = paths["output_directory"] / f"{input_path.stem}_out.pdbqt"
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"missing Vina-GPU batch pose: {output_path}")
        score, pose_count = parse_vina_pose(output_path)
        guard = float(protocol["maximum_absolute_score_kcal_per_mol"])
        if abs(score) > guard:
            raise ValueError(f"nonphysical Vina-GPU score: {score}")
        key = (seed_id, receptor["conformer_id"], ligand["ligand_id"])
        cpu_score = cpu_scores[key]
        rows.append(
            {
                "target_id": "MK14",
                "profile_id": profile["profile_id"],
                "search_depth": depth,
                "seed_id": seed_id,
                "base_seed": base_seed,
                "seed_offset": int(ligand["seed_offset"]),
                "pair_seed": chunk_seed + local_index,
                "receptor_id": receptor["conformer_id"],
                "ligand_id": ligand["ligand_id"],
                "label": ligand["label"],
                "selection_role": ligand["selection_role"],
                "batch_chunk_index": chunk_index,
                "batch_local_index": local_index,
                "gpu_vinagpu21_score": score,
                "cpu_vina_e32_score": cpu_score,
                "score_delta_gpu_minus_cpu": score - cpu_score,
                "absolute_score_delta": abs(score - cpu_score),
                "pose_count": pose_count,
                "output_pose_path": relative_path(root, output_path),
                "output_pose_sha256": file_sha256(output_path),
                "status": "ok",
            }
        )
    write_csv(paths["scores"], rows)
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "status": "ok",
        "signature": signature,
        "profile_id": profile["profile_id"],
        "search_depth": depth,
        "seed_id": seed_id,
        "base_seed": base_seed,
        "chunk_seed": chunk_seed,
        "receptor_id": receptor["conformer_id"],
        "chunk_index": chunk_index,
        "first_seed_offset": first_offset,
        "ligand_count": len(rows),
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


def gate_check(observed: float, threshold: float, comparison: str) -> dict[str, object]:
    if comparison == "equal":
        passed = observed == threshold
    elif comparison == "maximum":
        passed = observed <= threshold
    else:
        passed = observed >= threshold
    return {
        "observed": observed,
        "threshold": threshold,
        "comparison": comparison,
        "passed": passed,
    }


def assess_profile(
    config: dict[str, object],
    profile: dict[str, object],
    rows: list[dict[str, object]],
    elapsed_seconds: float,
    frozen_groups: dict[tuple[str, str], dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    expected_pairs = int(config["expected"]["diagnostic_pair_count_per_profile"])
    groups = group_metrics(comparison_rows(rows))
    expected_groups = int(config["expected"]["diagnostic_group_count"])
    if len(groups) != expected_groups:
        raise ValueError("profile group count differs")
    absolute = [abs(float(row["score_delta_gpu_minus_cpu"])) for row in rows]
    thresholds = config["diagnostic_gate"]
    throughput = config["throughput_reference"]
    cpu_rate = float(throughput["pair_count"]) / float(throughput["elapsed_seconds"])
    gpu_rate = len(rows) / elapsed_seconds
    speedup = gpu_rate / cpu_rate
    checks = {
        "complete_pairs": gate_check(len(rows), expected_pairs, "equal"),
        "overall_median_absolute_score_delta": gate_check(
            statistics.median(absolute),
            float(
                thresholds["maximum_overall_median_absolute_score_delta_kcal_per_mol"]
            ),
            "maximum",
        ),
        "overall_p95_absolute_score_delta": gate_check(
            quantile(absolute, 0.95),
            float(thresholds["maximum_overall_p95_absolute_score_delta_kcal_per_mol"]),
            "maximum",
        ),
        "minimum_each_group_spearman": gate_check(
            min(float(row["spearman"]) for row in groups),
            float(thresholds["minimum_each_group_spearman"]),
            "minimum",
        ),
        "minimum_each_group_top5pct_overlap": gate_check(
            min(float(row["top5pct_overlap"]) for row in groups),
            float(thresholds["minimum_each_group_top5pct_overlap"]),
            "minimum",
        ),
        "throughput_speedup_vs_recorded_32vcpu": gate_check(
            speedup,
            float(thresholds["minimum_throughput_speedup_vs_recorded_32vcpu"]),
            "minimum",
        ),
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    speed_passed = bool(checks["throughput_speedup_vs_recorded_32vcpu"]["passed"])
    group_rows: list[dict[str, object]] = []
    for row in groups:
        key = (str(row["seed_id"]), str(row["receptor_id"]))
        baseline = frozen_groups[key]
        group_rows.append(
            {
                "profile_id": profile["profile_id"],
                "search_depth": int(profile["search_depth"]),
                **row,
                "frozen_heuristic_spearman": baseline["spearman"],
                "spearman_change_vs_heuristic": float(row["spearman"])
                - float(baseline["spearman"]),
                "frozen_heuristic_top5pct_overlap": baseline["top5pct_overlap"],
            }
        )
    summary = {
        "schema_version": "1.0",
        "status": (
            "search_depth_candidate_selected"
            if passed
            else "search_depth_profile_failed"
        ),
        "profile_id": profile["profile_id"],
        "search_depth": int(profile["search_depth"]),
        "pair_count": len(rows),
        "group_count": len(groups),
        "elapsed_seconds": elapsed_seconds,
        "pairs_per_second": gpu_rate,
        "speedup_vs_recorded_32vcpu": speedup,
        "gate_checks": checks,
        "all_gate_checks_passed": passed,
        "continue_to_next_profile": not passed and speed_passed,
        "labels_used_for_profile_selection": False,
        "group_metrics": group_rows,
    }
    return summary, group_rows


def profile_paths(run_directory: Path, profile_id: str) -> dict[str, Path]:
    directory = run_directory / "profiles" / profile_id
    return {
        "directory": directory,
        "scores": directory / "profile_scores.csv",
        "runs": directory / "chunk_runs.csv",
        "groups": directory / "group_metrics.csv",
        "summary": directory / "profile_summary.json",
    }


def run_profile(
    root: Path,
    config: dict[str, object],
    config_sha256: str,
    run_directory: Path,
    executable: Path,
    kernel_directory: Path,
    runtime_lock: dict[str, object],
    profile: dict[str, object],
    receptors: list[dict[str, str]],
    ligands: list[dict[str, str]],
    cpu_scores: dict[tuple[str, str, str], float],
    frozen_groups: dict[tuple[str, str], dict[str, object]],
    resume: bool,
) -> dict[str, object]:
    paths = profile_paths(run_directory, str(profile["profile_id"]))
    protocol = {**config["vinagpu"], "search_depth": int(profile["search_depth"])}
    chunks = ligand_chunks(ligands, int(config["expected"]["chunk_size"]))
    receptor_map = {row["conformer_id"]: row for row in receptors}
    seed_map = {str(row["seed_id"]): row for row in config["inputs"]["cpu_seed_runs"]}
    targets = [
        (str(row["seed_id"]), str(row["receptor_id"]))
        for row in config["search_depth_diagnostic"]["target_groups"]
    ]
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    resumed_count = 0
    executed_count = 0
    expected_chunks = int(config["expected"]["diagnostic_chunk_count_per_profile"])
    completed = 0
    for seed_id, receptor_id in targets:
        seed = seed_map[seed_id]
        receptor = receptor_map[receptor_id]
        print(
            f"running {profile['profile_id']} {seed_id}/{receptor_id}",
            flush=True,
        )
        for chunk_index, ligand_chunk in enumerate(chunks):
            rows, summary, resumed = run_search_chunk(
                root,
                search_chunk_paths(
                    paths["directory"], seed_id, receptor_id, chunk_index
                ),
                executable,
                kernel_directory,
                runtime_lock,
                config_sha256,
                profile,
                receptor,
                ligand_chunk,
                cpu_scores,
                protocol,
                seed_id,
                int(seed["base_seed"]),
                chunk_index,
                resume,
            )
            all_rows.extend(rows)
            summaries.append(summary)
            resumed_count += int(resumed)
            executed_count += int(not resumed)
            completed += 1
            if completed % 10 == 0 or completed == expected_chunks:
                print(
                    f"profile={profile['profile_id']} chunks={completed}/"
                    f"{expected_chunks} executed={executed_count} "
                    f"resumed={resumed_count}",
                    flush=True,
                )

    expected_pairs = int(config["expected"]["diagnostic_pair_count_per_profile"])
    keys = {
        (str(row["seed_id"]), str(row["receptor_id"]), str(row["ligand_id"]))
        for row in all_rows
    }
    if len(all_rows) != expected_pairs or len(keys) != expected_pairs:
        raise RuntimeError("search-depth profile pair set is incomplete or duplicated")
    elapsed = sum(float(summary["elapsed_seconds"]) for summary in summaries)
    profile_summary, group_rows = assess_profile(
        config, profile, all_rows, elapsed, frozen_groups
    )
    write_csv(paths["scores"], all_rows)
    write_csv(
        paths["runs"],
        [
            {
                "profile_id": summary["profile_id"],
                "search_depth": summary["search_depth"],
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
            for summary in summaries
        ],
    )
    write_csv(paths["groups"], group_rows)
    profile_summary.update(
        {
            "experiment_id": config["experiment_id"],
            "executed_chunk_count_this_invocation": executed_count,
            "resumed_chunk_count_this_invocation": resumed_count,
            "chunk_count": len(summaries),
            "outputs": {
                "profile_scores_csv": output_descriptor(root, paths["scores"]),
                "chunk_runs_csv": output_descriptor(root, paths["runs"]),
                "group_metrics_csv": output_descriptor(root, paths["groups"]),
            },
            "validation_rows": 0,
            "test_rows": 0,
            "enrichment_metrics_calculated": False,
        }
    )
    write_json(paths["summary"], profile_summary)
    return profile_summary


def runtime_identity_sha256(runtime_lock: dict[str, object]) -> str:
    return (
        hashlib.sha256(
            json.dumps(runtime_lock, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        )
        .hexdigest()
        .upper()
    )


def validate_preflight(
    root: Path,
    config: dict[str, object],
    config_sha256: str,
    runtime_lock: dict[str, object],
) -> None:
    path = rooted_path(root, str(config["outputs"]["preflight_summary_json"]))
    if not path.is_file():
        raise RuntimeError("search-depth preflight is missing")
    summary = read_json(path)
    preflight = config["preflight"]
    if summary.get("status") != "search_depth_preflight_passed":
        raise RuntimeError("search-depth preflight did not pass")
    if summary.get("config_sha256") != config_sha256:
        raise RuntimeError("search-depth preflight config differs")
    if summary.get("runtime_identity_sha256") != runtime_identity_sha256(runtime_lock):
        raise RuntimeError("search-depth preflight runtime differs")
    for key in ("profile_id", "search_depth", "seed_id", "receptor_id"):
        if summary.get(key) != preflight[key]:
            raise RuntimeError(f"search-depth preflight differs: {key}")


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
    receptors, ligands, reference, cpu_scores, audit = validate_diagnostic_inputs(
        root, config
    )
    audit = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "audit_only_ok",
        "config": {
            "path": relative_path(root, config_path),
            "sha256": config_sha256,
        },
        **audit,
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
    approved_runtime = verified_json(
        root, config["inputs"]["approved_deterministic_batch_runtime"]
    )
    validate_approved_runtime(runtime_lock, approved_runtime)
    outputs = config["outputs"]
    run_directory = rooted_path(root, str(outputs["run_directory"]))
    run_directory.mkdir(parents=True, exist_ok=True)
    lock_path = rooted_path(root, str(outputs["runtime_lock_json"]))
    ensure_runtime_lock(lock_path, runtime_lock)
    if args.lock_runtime_only:
        print(json.dumps(runtime_lock, indent=2, sort_keys=True))
        return 0

    target_keys = {
        (str(row["seed_id"]), str(row["receptor_id"]))
        for row in config["search_depth_diagnostic"]["target_groups"]
    }
    frozen_rows = baseline_rows(target_keys, ligands, reference, cpu_scores)
    frozen_groups = {
        (str(row["seed_id"]), str(row["receptor_id"])): row
        for row in group_metrics(frozen_rows)
    }

    preflight = config["preflight"]
    profiles = list(config["search_depth_diagnostic"]["profiles"])
    if args.preflight_only:
        profile = next(
            row for row in profiles if row["profile_id"] == preflight["profile_id"]
        )
        seed = next(
            row
            for row in config["inputs"]["cpu_seed_runs"]
            if row["seed_id"] == preflight["seed_id"]
        )
        receptor = next(
            row for row in receptors if row["conformer_id"] == preflight["receptor_id"]
        )
        chunks = ligand_chunks(ligands, int(config["expected"]["chunk_size"]))
        chunk_index = int(preflight["chunk_index"])
        paths = profile_paths(run_directory, str(profile["profile_id"]))
        protocol = {
            **config["vinagpu"],
            "search_depth": int(profile["search_depth"]),
        }
        rows, summary, resumed = run_search_chunk(
            root,
            search_chunk_paths(
                paths["directory"],
                str(seed["seed_id"]),
                receptor["conformer_id"],
                chunk_index,
            ),
            executable,
            args.opencl_binary_path.resolve(),
            runtime_lock,
            config_sha256,
            profile,
            receptor,
            chunks[chunk_index],
            cpu_scores,
            protocol,
            str(seed["seed_id"]),
            int(seed["base_seed"]),
            chunk_index,
            args.resume,
        )
        passed = summary["status"] == "ok" and len(rows) == int(
            preflight["expected_pair_count"]
        )
        preflight_summary = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "status": (
                "search_depth_preflight_passed"
                if passed
                else "search_depth_preflight_failed"
            ),
            "config_sha256": config_sha256,
            "runtime_identity_sha256": runtime_identity_sha256(runtime_lock),
            "profile_id": profile["profile_id"],
            "search_depth": int(profile["search_depth"]),
            "seed_id": seed["seed_id"],
            "receptor_id": receptor["conformer_id"],
            "chunk_index": chunk_index,
            "pair_count": len(rows),
            "resumed": resumed,
            "elapsed_seconds": summary["elapsed_seconds"],
            "chunk_summary_path": relative_path(
                root,
                search_chunk_paths(
                    paths["directory"],
                    str(seed["seed_id"]),
                    receptor["conformer_id"],
                    chunk_index,
                )["summary"],
            ),
            "validation_rows": 0,
            "test_rows": 0,
        }
        write_json(
            rooted_path(root, str(outputs["preflight_summary_json"])),
            preflight_summary,
        )
        print(json.dumps(preflight_summary, indent=2, sort_keys=True))
        return 0 if passed else 2

    validate_preflight(root, config, config_sha256, runtime_lock)
    executed_profiles: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    stop_reason = "profile_ladder_exhausted_without_candidate"
    for profile in profiles:
        summary = run_profile(
            root,
            config,
            config_sha256,
            run_directory,
            executable,
            args.opencl_binary_path.resolve(),
            runtime_lock,
            profile,
            receptors,
            ligands,
            cpu_scores,
            frozen_groups,
            args.resume,
        )
        summary_path = profile_paths(run_directory, str(profile["profile_id"]))[
            "summary"
        ]
        executed_profiles.append(
            {
                "profile_id": profile["profile_id"],
                "search_depth": int(profile["search_depth"]),
                "status": summary["status"],
                "all_gate_checks_passed": summary["all_gate_checks_passed"],
                "pair_count": summary["pair_count"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "speedup_vs_recorded_32vcpu": summary["speedup_vs_recorded_32vcpu"],
                "minimum_group_spearman": summary["gate_checks"][
                    "minimum_each_group_spearman"
                ]["observed"],
                "profile_summary": output_descriptor(root, summary_path),
            }
        )
        if bool(summary["all_gate_checks_passed"]):
            selected = {
                "profile_id": profile["profile_id"],
                "search_depth": int(profile["search_depth"]),
            }
            stop_reason = "first_profile_passing_every_gate_selected"
            break
        if not bool(summary["continue_to_next_profile"]):
            stop_reason = "speed_gate_failed_so_higher_depths_were_not_run"
            break

    diagnostic_summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "search_depth_candidate_selected"
            if selected is not None
            else "search_depth_diagnostic_failed"
        ),
        "config": {
            "path": relative_path(root, config_path),
            "sha256": config_sha256,
        },
        "runtime_lock": output_descriptor(root, lock_path),
        "bridge_result": config["inputs"]["deterministic_batch_bridge_result"],
        "frozen_v1_gate_status_unchanged": "gpu_equivalence_gate_failed",
        "planned_profiles": profiles,
        "executed_profiles": executed_profiles,
        "selected_profile": selected,
        "stop_reason": stop_reason,
        "next_action": (
            "Freeze and run a complete five-receptor, three-seed Train-160 confirmation using only the selected fixed depth."
            if selected is not None
            else "Retain official CPU Vina; do not run larger or fresh-data Vina-GPU docking with this protocol."
        ),
        "validation_rows": 0,
        "test_rows": 0,
        "enrichment_metrics_calculated": False,
        "labels_used_for_profile_selection": False,
        "interpretation_note": config["decision_boundary"],
    }
    diagnostic_path = rooted_path(root, str(outputs["diagnostic_summary_json"]))
    write_json(diagnostic_path, diagnostic_summary)
    print(json.dumps(diagnostic_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
