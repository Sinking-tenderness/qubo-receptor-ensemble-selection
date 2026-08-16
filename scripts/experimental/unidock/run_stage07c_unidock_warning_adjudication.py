"""Run the Stage 07c new-seed and warning-replay confirmation."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

try:
    from .run_stage07b_unidock_enhanced_confirmation import (
        audit_batch_poses,
        validate_inputs as validate_stage07b_inputs,
    )
    from .run_unidock_gpu_equivalence import (
        COORDINATE_SIZE_MISMATCH,
        OUTPUT_CONTAINER_WARNING,
        batch_paths,
        executable_evidence,
        file_sha256,
        output_descriptor,
        protocol_signature,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        run_batch,
        validate_checkpoint,
        verified_path,
        write_csv,
        write_json,
    )
except ImportError:
    from run_stage07b_unidock_enhanced_confirmation import (
        audit_batch_poses,
        validate_inputs as validate_stage07b_inputs,
    )
    from run_unidock_gpu_equivalence import (
        COORDINATE_SIZE_MISMATCH,
        OUTPUT_CONTAINER_WARNING,
        batch_paths,
        executable_evidence,
        file_sha256,
        output_descriptor,
        protocol_signature,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        run_batch,
        validate_checkpoint,
        verified_path,
        write_csv,
        write_json,
    )


NEW_SEED_ID = "seed3"
NEW_BASE_SEED = 20260804
REPLAY_RUN_ID = "seed2_replay"
REPLAY_BASE_SEED = 20260803
REPLAY_RECEPTOR_ID = "MK14_3MPT_aligned"
PROFILE_ID = "enhanced"
COORDINATE_WARNING = re.compile(
    r"^t\.coords\.size\(\)=(\d+), out\[0\]\.coords\.size\(\)=(\d+)$"
)


def validate_config(config: dict[str, object]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "implementation",
        "data_boundary",
        "inputs",
        "expected",
        "unidock",
        "warning_adjudication",
        "profile_gate",
        "outputs",
        "decision_boundary",
    }
    if set(config) != required:
        raise ValueError("Stage 07c config keys differ")
    expected = dict(config["expected"])
    fixed_counts = {
        "receptor_count": 4,
        "ligand_count": 160,
        "prior_seed_count": 3,
        "combined_seed_count": 4,
        "prior_score_count": 1920,
        "new_seed_pair_count": 640,
        "warning_replay_pair_count": 160,
        "total_gpu_pair_count": 800,
        "batch_count": 5,
        "validation_rows": 0,
        "test_rows": 0,
    }
    for key, value in fixed_counts.items():
        if int(expected[key]) != value:
            raise ValueError(f"Stage 07c expected count differs: {key}")
    seeds = list(config["inputs"]["seeds"])
    expected_seeds = (
        ("seed0", 20260801),
        ("seed1", 20260802),
        ("seed2", 20260803),
        (NEW_SEED_ID, NEW_BASE_SEED),
    )
    observed_seeds = tuple(
        (str(row["seed_id"]), int(row["base_seed"])) for row in seeds
    )
    if observed_seeds != expected_seeds:
        raise ValueError("Stage 07c seed ledger differs")
    protocol = dict(config["unidock"])
    if int(protocol["exhaustiveness"]) != 1024:
        raise ValueError("Stage 07c exhaustiveness must be 1024")
    if int(protocol["max_step"]) != 80:
        raise ValueError("Stage 07c max_step must be 80")
    if str(protocol["profile_id"]) != PROFILE_ID:
        raise ValueError("Stage 07c profile ID differs")
    replay = dict(config["warning_adjudication"])["replay"]
    if str(replay["run_id"]) != REPLAY_RUN_ID:
        raise ValueError("Stage 07c replay run ID differs")
    if int(replay["base_seed"]) != REPLAY_BASE_SEED:
        raise ValueError("Stage 07c replay seed differs")
    if str(replay["receptor_id"]) != REPLAY_RECEPTOR_ID:
        raise ValueError("Stage 07c replay receptor differs")
    boundary = dict(config["data_boundary"])
    if int(boundary["fresh_validation_rows_permitted"]) != 0:
        raise ValueError("fresh validation must remain unavailable")
    if int(boundary["test_rows_permitted"]) != 0:
        raise ValueError("locked test must remain unavailable")


def verify_implementation(config: dict[str, object], key: str, path: Path) -> None:
    descriptor = dict(config["implementation"])[key]
    if Path(str(descriptor["path"])).resolve() != path.resolve():
        raise ValueError(f"implementation path differs: {key}")
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"implementation SHA-256 differs: {key}")


def validate_inputs(
    root: Path, config: dict[str, object]
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, object],
]:
    receptors, ligands, audit = validate_stage07b_inputs(root, config)
    prior_path = verified_path(root, config["inputs"]["prior_scores"])
    replay_path = verified_path(root, config["inputs"]["replay_reference"])
    verified_path(root, config["inputs"]["evidence_provenance"])
    prior_rows = read_csv(prior_path)
    replay_rows = read_csv(replay_path)
    expected = dict(config["expected"])
    if len(prior_rows) != int(expected["prior_score_count"]):
        raise ValueError("Stage 07c prior score count differs")
    if {row["profile_id"] for row in prior_rows} != {PROFILE_ID}:
        raise ValueError("Stage 07c prior profile differs")
    if {row["seed_id"] for row in prior_rows} != {"seed0", "seed1", "seed2"}:
        raise ValueError("Stage 07c prior seeds differ")
    if any(row["pose_integrity_status"] != "ok" for row in prior_rows):
        raise ValueError("Stage 07c prior evidence contains a failed pose")
    if len(replay_rows) != int(expected["warning_replay_pair_count"]):
        raise ValueError("Stage 07c replay reference count differs")
    if {row["seed_id"] for row in replay_rows} != {"seed2"}:
        raise ValueError("Stage 07c replay reference seed differs")
    if {row["receptor_id"] for row in replay_rows} != {REPLAY_RECEPTOR_ID}:
        raise ValueError("Stage 07c replay reference receptor differs")
    if any(row["pose_integrity_status"] != "ok" for row in replay_rows):
        raise ValueError("Stage 07c replay reference contains a failed pose")
    audit.update(
        {
            "profile_count": 1,
            "seed_count": 4,
            "prior_score_count": len(prior_rows),
            "new_seed_pair_count": int(expected["new_seed_pair_count"]),
            "warning_replay_pair_count": len(replay_rows),
            "expected_pair_count": int(expected["total_gpu_pair_count"]),
        }
    )
    return receptors, ligands, prior_rows, replay_rows, audit


def classify_warning_log(
    log_path: Path, pose_audit: dict[str, object]
) -> dict[str, object]:
    lines = [
        line.strip()
        for line in log_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    ]
    output_lines = [line for line in lines if line.startswith(OUTPUT_CONTAINER_WARNING)]
    coordinate_lines = [line for line in lines if line.startswith(COORDINATE_SIZE_MISMATCH)]
    other_warning_lines = [
        line
        for line in lines
        if "WARNING" in line and not line.startswith(OUTPUT_CONTAINER_WARNING)
    ]
    coordinate_pairs = []
    invalid_coordinate_lines = []
    for line in coordinate_lines:
        match = COORDINATE_WARNING.fullmatch(line)
        if match is None:
            invalid_coordinate_lines.append(line)
            continue
        first = int(match.group(1))
        second = int(match.group(2))
        coordinate_pairs.append((first, second))
        if second - first != 1:
            invalid_coordinate_lines.append(line)
    event_count = max(len(output_lines), len(coordinate_lines))
    any_warning = bool(
        event_count or other_warning_lines or invalid_coordinate_lines
    )
    approved_shape = (
        event_count > 0
        and len(output_lines) == len(coordinate_lines)
        and not other_warning_lines
        and not invalid_coordinate_lines
    )
    pose_safe = int(pose_audit["failure_count"]) == 0
    resolved = not any_warning or (approved_shape and pose_safe)
    return {
        "known_warning_event_count": event_count if approved_shape else 0,
        "unresolved_warning_event_count": 0 if resolved else max(1, event_count),
        "output_container_warning_count": len(output_lines),
        "coordinate_size_warning_count": len(coordinate_lines),
        "coordinate_size_pairs": [list(value) for value in coordinate_pairs],
        "other_warning_lines": other_warning_lines,
        "invalid_coordinate_lines": invalid_coordinate_lines,
        "pose_integrity_failure_count": int(pose_audit["failure_count"]),
        "status": "resolved" if resolved else "unresolved",
    }


def run(
    config_path: Path,
    root: Path,
    unidock: str | None,
    audit_only: bool,
    resume: bool,
) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    validate_config(config)
    verify_implementation(config, "runner", Path(__file__))
    verify_implementation(
        config,
        "unidock_batch_helper",
        Path(__file__).with_name("run_unidock_gpu_equivalence.py"),
    )
    verify_implementation(
        config,
        "pose_audit_helper",
        Path(__file__).with_name(
            "run_stage07b_unidock_enhanced_confirmation.py"
        ),
    )
    receptors, ligands, _, _, audit = validate_inputs(root, config)
    if audit_only:
        result = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "config": {
                "path": relative_path(root, config_path),
                "sha256": file_sha256(config_path),
            },
            **audit,
            "operation": "input audit only; no GPU docking was started",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    protocol = dict(config["unidock"])
    executable = unidock or str(protocol["executable"])
    executable_info = executable_evidence(
        executable, str(protocol["required_package_version"])
    )
    outputs = dict(config["outputs"])
    run_directory = rooted_path(root, str(outputs["run_directory"]))
    run_directory.mkdir(parents=True, exist_ok=True)
    config_sha256 = file_sha256(config_path)
    ligand_ids = {row["ligand_id"] for row in ligands}
    receptor_by_id = {row["conformer_id"]: row for row in receptors}
    units = [
        {
            "run_role": "new_seed",
            "run_id": NEW_SEED_ID,
            "seed_id": NEW_SEED_ID,
            "base_seed": NEW_BASE_SEED,
            "receptor": receptor,
        }
        for receptor in receptors
    ]
    units.append(
        {
            "run_role": "warning_replay",
            "run_id": REPLAY_RUN_ID,
            "seed_id": REPLAY_RUN_ID,
            "base_seed": REPLAY_BASE_SEED,
            "receptor": receptor_by_id[REPLAY_RECEPTOR_ID],
        }
    )
    all_rows: list[dict[str, object]] = []
    batch_summaries: list[dict[str, object]] = []
    executed_batches = 0
    resumed_batches = 0
    invocation_started = time.perf_counter()
    for unit in units:
        role = str(unit["run_role"])
        run_id = str(unit["run_id"])
        seed_id = str(unit["seed_id"])
        base_seed = int(unit["base_seed"])
        receptor = dict(unit["receptor"])
        receptor_id = receptor["conformer_id"]
        paths = batch_paths(run_directory / role, run_id, receptor_id)
        paths["directory"].mkdir(parents=True, exist_ok=True)
        signature = protocol_signature(
            config_sha256,
            run_id,
            base_seed,
            receptor,
            ligands,
            protocol,
        )
        checkpoint = (
            validate_checkpoint(root, paths, signature, ligand_ids)
            if resume
            else None
        )
        if checkpoint is not None:
            rows, summary = checkpoint
            if not all("pose_integrity_status" in row for row in rows):
                checkpoint = None
            if "warning_adjudication" not in summary:
                checkpoint = None
        if checkpoint is not None:
            rows, summary = checkpoint
            resumed_batches += 1
            print(f"resume ok: {role}/{run_id}/{receptor_id}", flush=True)
        else:
            print(f"running: {role}/{run_id}/{receptor_id}", flush=True)
            rows, summary = run_batch(
                root,
                paths,
                str(executable_info["resolved_executable"]),
                receptor,
                ligands,
                protocol,
                seed_id,
                base_seed,
                signature,
            )
            rows, pose_audit = audit_batch_poses(root, ligands, rows)
            warning_audit = classify_warning_log(paths["log"], pose_audit)
            write_csv(paths["scores"], rows)
            summary["scores_sha256"] = file_sha256(paths["scores"])
            summary["pose_integrity_audit"] = pose_audit
            summary["warning_adjudication"] = warning_audit
            summary["status"] = (
                "ok"
                if pose_audit["status"] == "ok"
                and warning_audit["status"] == "resolved"
                else "audit_failed"
            )
            write_json(paths["summary"], summary)
            executed_batches += 1
            print(
                f"completed: {role}/{run_id}/{receptor_id} "
                f"in {float(summary['elapsed_seconds']):.3f} s",
                flush=True,
            )
        for row in rows:
            all_rows.append(
                {
                    "run_role": role,
                    "run_id": run_id,
                    "comparison_seed_id": (
                        "seed2" if role == "warning_replay" else seed_id
                    ),
                    **dict(row),
                }
            )
        batch_summaries.append(
            {"run_role": role, "run_id": run_id, **dict(summary)}
        )

    all_rows.sort(
        key=lambda row: (
            str(row["run_role"]),
            str(row["run_id"]),
            str(row["receptor_id"]),
            str(row["ligand_id"]),
        )
    )
    if len(all_rows) != int(config["expected"]["total_gpu_pair_count"]):
        raise ValueError("Stage 07c GPU score count differs")
    keys = {
        (row["run_role"], row["run_id"], row["receptor_id"], row["ligand_id"])
        for row in all_rows
    }
    if len(keys) != len(all_rows):
        raise ValueError("Stage 07c GPU scores contain duplicate keys")

    scores_path = rooted_path(root, str(outputs["scores_csv"]))
    batches_path = rooted_path(root, str(outputs["batch_runs_csv"]))
    summary_path = rooted_path(root, str(outputs["run_summary_json"]))
    write_csv(scores_path, all_rows)
    batch_rows = [
        {
            "run_role": summary["run_role"],
            "run_id": summary["run_id"],
            "seed_id": summary["seed_id"],
            "base_seed": summary["base_seed"],
            "receptor_id": summary["receptor_id"],
            "ligand_count": summary["ligand_count"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "known_warning_event_count": summary["warning_adjudication"][
                "known_warning_event_count"
            ],
            "unresolved_warning_event_count": summary[
                "warning_adjudication"
            ]["unresolved_warning_event_count"],
            "pose_integrity_failure_count": summary["pose_integrity_audit"][
                "failure_count"
            ],
            "signature": summary["signature"],
            "status": summary["status"],
        }
        for summary in batch_summaries
    ]
    write_csv(batches_path, batch_rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "operation": "Stage 07c new-seed and known-warning replay",
        "config": {
            "path": relative_path(root, config_path),
            "sha256": config_sha256,
        },
        "unidock_executable": executable_info,
        "input_audit": audit,
        "protocol": protocol,
        "pair_count": len(all_rows),
        "batch_count": len(batch_rows),
        "executed_batches_this_invocation": executed_batches,
        "resumed_batches_this_invocation": resumed_batches,
        "known_warning_event_count": sum(
            int(row["known_warning_event_count"]) for row in batch_rows
        ),
        "unresolved_warning_event_count": sum(
            int(row["unresolved_warning_event_count"]) for row in batch_rows
        ),
        "pose_integrity_failure_count": sum(
            int(row["pose_integrity_failure_count"]) for row in batch_rows
        ),
        "elapsed_seconds": sum(
            float(row["elapsed_seconds"]) for row in batch_rows
        ),
        "current_invocation_elapsed_seconds": time.perf_counter()
        - invocation_started,
        "outputs": {
            "scores_csv": output_descriptor(root, scores_path),
            "batch_runs_csv": output_descriptor(root, batches_path),
        },
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
        "interpretation_note": config["decision_boundary"],
    }
    write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--unidock", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.unidock, args.audit_only, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
