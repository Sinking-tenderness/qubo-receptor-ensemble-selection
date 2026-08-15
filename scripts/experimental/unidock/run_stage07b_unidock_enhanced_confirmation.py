"""Run the Stage 07b train-only Uni-Dock enhanced-profile confirmation."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

try:
    from .run_unidock_gpu_equivalence import (
        batch_paths,
        executable_evidence,
        file_sha256,
        macrocycle_closure_atom_types,
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
    from run_unidock_gpu_equivalence import (
        batch_paths,
        executable_evidence,
        file_sha256,
        macrocycle_closure_atom_types,
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


PROFILE_ORDER = (
    "detail_recheck",
    "step_extended",
    "depth_extended",
    "enhanced",
)
CANDIDATE_PROFILES = (
    "step_extended",
    "depth_extended",
    "enhanced",
)
PROFILE_SETTINGS = {
    "detail_recheck": (512, 40),
    "step_extended": (512, 80),
    "depth_extended": (1024, 40),
    "enhanced": (1024, 80),
}


def validate_config(config: dict[str, object]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "implementation",
        "data_boundary",
        "inputs",
        "expected",
        "unidock_common",
        "profiles",
        "profile_gate",
        "outputs",
        "decision_boundary",
    }
    if set(config) != required:
        raise ValueError("Stage 07b confirmation config keys differ")

    profiles = dict(config["profiles"])
    if tuple(profiles) != PROFILE_ORDER:
        raise ValueError("Stage 07b profile order differs")
    for profile_id, expected_setting in PROFILE_SETTINGS.items():
        profile = dict(profiles[profile_id])
        observed = (
            int(profile["exhaustiveness"]),
            int(profile["max_step"]),
        )
        if observed != expected_setting:
            raise ValueError(f"official Stage 07b profile differs: {profile_id}")

    seeds = list(config["inputs"]["seeds"])
    expected_seeds = (
        ("seed0", 20260801),
        ("seed1", 20260802),
        ("seed2", 20260803),
    )
    observed_seeds = tuple(
        (str(row["seed_id"]), int(row["base_seed"])) for row in seeds
    )
    if observed_seeds != expected_seeds:
        raise ValueError("Stage 07b paired seeds differ")

    expected = dict(config["expected"])
    count_checks = {
        "receptor_count": 4,
        "ligand_count": 160,
        "seed_count": 3,
        "profile_count": 4,
        "batch_count_per_profile": 12,
        "pair_count_per_profile": 1920,
        "total_batch_count": 48,
        "total_pair_count": 7680,
        "validation_rows": 0,
        "test_rows": 0,
    }
    for key, value in count_checks.items():
        if int(expected[key]) != value:
            raise ValueError(f"Stage 07b expected count differs: {key}")

    gate = dict(config["profile_gate"])
    if str(gate["reference_profile_id"]) != "enhanced":
        raise ValueError("Stage 07b reference profile must be enhanced")
    if tuple(gate["candidate_profile_ids"]) != CANDIDATE_PROFILES:
        raise ValueError("Stage 07b candidate profiles differ")
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
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    expected = dict(config["expected"])
    boundary = dict(config["data_boundary"])
    inputs = dict(config["inputs"])
    receptor_rows = read_csv(verified_path(root, inputs["receptor_manifest"]))
    ligand_rows = read_csv(verified_path(root, inputs["ligand_manifest"]))

    selected_ids = [str(value) for value in expected["receptor_ids"]]
    receptors_by_id = {row["conformer_id"]: row for row in receptor_rows}
    if len(receptors_by_id) != len(receptor_rows):
        raise ValueError("receptor manifest contains duplicate IDs")
    if any(receptor_id not in receptors_by_id for receptor_id in selected_ids):
        raise ValueError("a selected receptor is absent from the manifest")
    receptors = [receptors_by_id[receptor_id] for receptor_id in selected_ids]
    if len(receptors) != int(expected["receptor_count"]):
        raise ValueError("selected receptor count differs")
    if any(row["status"] != "ok" for row in receptors):
        raise ValueError("selected receptor manifest contains a failed row")

    if len(ligand_rows) != int(expected["ligand_count"]):
        raise ValueError("ligand count differs")
    if len({row["ligand_id"] for row in ligand_rows}) != len(ligand_rows):
        raise ValueError("ligand manifest contains duplicate IDs")
    labels = Counter(row["label"] for row in ligand_rows)
    expected_labels = Counter(
        {key: int(value) for key, value in expected["label_counts"].items()}
    )
    if labels != expected_labels:
        raise ValueError("ligand labels differ")
    if {row["split"] for row in ligand_rows} != {boundary["allowed_split"]}:
        raise ValueError("a non-train ligand is visible")
    if {row["selection_role"] for row in ligand_rows} != {
        boundary["allowed_selection_role"]
    }:
        raise ValueError("ligand selection role differs")
    if any(row["pdbqt_status"] != "ok" for row in ligand_rows):
        raise ValueError("ligand manifest contains a failed PDBQT")

    for rows, path_column, hash_column, id_column in (
        (
            receptors,
            "receptor_pdbqt",
            "receptor_pdbqt_sha256",
            "conformer_id",
        ),
        (ligand_rows, "pdbqt_path", "pdbqt_sha256", "ligand_id"),
    ):
        for row in rows:
            path = rooted_path(root, row[path_column])
            if not path.is_file():
                raise FileNotFoundError(path)
            if file_sha256(path) != row[hash_column].upper():
                raise ValueError(f"prepared input hash differs: {row[id_column]}")

    closure_ligands = []
    for row in ligand_rows:
        atom_types = macrocycle_closure_atom_types(
            rooted_path(root, row["pdbqt_path"])
        )
        if atom_types:
            closure_ligands.append(
                {"ligand_id": row["ligand_id"], "atom_types": atom_types}
            )
    if closure_ligands:
        raise ValueError("Stage 07b ligand inputs retain closure pseudoatoms")
    variants = Counter(row["preparation_variant"] for row in ligand_rows)
    if variants != Counter(
        {
            "original_meeko_flexible": 156,
            "meeko_rigid_macrocycles": 4,
        }
    ):
        raise ValueError("rigid-macrocycle repair composition differs")

    return receptors, ligand_rows, {
        "status": "audit_only_ok",
        "receptor_ids": selected_ids,
        "receptor_count": len(receptors),
        "ligand_count": len(ligand_rows),
        "label_counts": dict(sorted(labels.items())),
        "preparation_variant_counts": dict(sorted(variants.items())),
        "macrocycle_closure_pseudoatom_ligand_count": 0,
        "profile_count": len(PROFILE_ORDER),
        "seed_count": len(inputs["seeds"]),
        "expected_pair_count": int(expected["total_pair_count"]),
        "validation_rows": 0,
        "test_rows": 0,
    }


def pdbqt_atom_signature(path: Path) -> tuple[int, dict[str, int]]:
    atom_types: Counter[str] = Counter()
    atom_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if not raw_line.startswith(("ATOM  ", "HETATM")):
                continue
            fields = raw_line.split()
            if not fields:
                continue
            atom_count += 1
            atom_types[fields[-1]] += 1
    if atom_count == 0:
        raise ValueError(f"PDBQT contains no atom records: {path}")
    return atom_count, dict(sorted(atom_types.items()))


def audit_batch_poses(
    root: Path,
    ligands: list[dict[str, str]],
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    ligands_by_id = {row["ligand_id"]: row for row in ligands}
    mismatches: list[dict[str, object]] = []
    audited_rows: list[dict[str, object]] = []
    for raw_row in rows:
        row = dict(raw_row)
        ligand_id = str(row["ligand_id"])
        ligand = ligands_by_id[ligand_id]
        input_path = rooted_path(root, ligand["pdbqt_path"])
        output_path = rooted_path(root, str(row["output_pose_path"]))
        input_count, input_types = pdbqt_atom_signature(input_path)
        output_count, output_types = pdbqt_atom_signature(output_path)
        count_match = input_count == output_count
        types_match = input_types == output_types
        one_pose = int(row["pose_count"]) == 1
        status = "ok" if count_match and types_match and one_pose else "failed"
        row.update(
            {
                "input_atom_count": input_count,
                "output_atom_count": output_count,
                "atom_count_match": count_match,
                "atom_types_match": types_match,
                "single_pose_match": one_pose,
                "pose_integrity_status": status,
            }
        )
        if status != "ok":
            mismatches.append(
                {
                    "ligand_id": ligand_id,
                    "input_atom_count": input_count,
                    "output_atom_count": output_count,
                    "input_atom_types": input_types,
                    "output_atom_types": output_types,
                    "pose_count": int(row["pose_count"]),
                }
            )
        audited_rows.append(row)
    return audited_rows, {
        "audited_pose_count": len(audited_rows),
        "failure_count": len(mismatches),
        "mismatches": mismatches,
        "status": "ok" if not mismatches else "failed",
    }


def merged_protocol(
    config: dict[str, object], profile_id: str
) -> dict[str, object]:
    protocol = dict(config["unidock_common"])
    protocol.update(dict(config["profiles"])[profile_id])
    protocol["profile_id"] = profile_id
    return protocol


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
    receptors, ligands, audit = validate_inputs(root, config)
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

    executable = unidock or str(config["unidock_common"]["executable"])
    executable_info = executable_evidence(
        executable,
        str(config["unidock_common"]["required_package_version"]),
    )
    outputs = dict(config["outputs"])
    run_directory = rooted_path(root, str(outputs["run_directory"]))
    run_directory.mkdir(parents=True, exist_ok=True)
    config_sha256 = file_sha256(config_path)
    ligand_ids = {row["ligand_id"] for row in ligands}
    all_rows: list[dict[str, object]] = []
    batch_summaries: list[dict[str, object]] = []
    executed_batches = 0
    resumed_batches = 0
    invocation_started = time.perf_counter()

    for profile_id in PROFILE_ORDER:
        protocol = merged_protocol(config, profile_id)
        profile_directory = run_directory / profile_id
        for seed in config["inputs"]["seeds"]:
            seed_id = str(seed["seed_id"])
            base_seed = int(seed["base_seed"])
            for receptor in receptors:
                receptor_id = receptor["conformer_id"]
                paths = batch_paths(profile_directory, seed_id, receptor_id)
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
                    if resume
                    else None
                )
                if checkpoint is not None:
                    rows, summary = checkpoint
                    if not all(
                        "pose_integrity_status" in row for row in rows
                    ) or "pose_integrity_audit" not in summary:
                        checkpoint = None
                if checkpoint is not None:
                    rows, summary = checkpoint
                    resumed_batches += 1
                    print(
                        f"resume ok: {profile_id}/{seed_id}/{receptor_id}",
                        flush=True,
                    )
                else:
                    print(
                        f"running: {profile_id}/{seed_id}/{receptor_id}",
                        flush=True,
                    )
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
                    write_csv(paths["scores"], rows)
                    summary["scores_sha256"] = file_sha256(paths["scores"])
                    summary["pose_integrity_audit"] = pose_audit
                    summary["status"] = (
                        "ok" if pose_audit["status"] == "ok"
                        else "pose_integrity_failed"
                    )
                    write_json(paths["summary"], summary)
                    executed_batches += 1
                    print(
                        f"completed: {profile_id}/{seed_id}/{receptor_id} "
                        f"in {float(summary['elapsed_seconds']):.3f} s",
                        flush=True,
                    )
                for row in rows:
                    all_rows.append({"profile_id": profile_id, **dict(row)})
                batch_summaries.append(
                    {"profile_id": profile_id, **dict(summary)}
                )

    all_rows.sort(
        key=lambda row: (
            str(row["profile_id"]),
            str(row["seed_id"]),
            str(row["receptor_id"]),
            str(row["ligand_id"]),
        )
    )
    expected_count = int(config["expected"]["total_pair_count"])
    if len(all_rows) != expected_count:
        raise ValueError("complete Stage 07b score count differs")
    unique_keys = {
        (
            row["profile_id"],
            row["seed_id"],
            row["receptor_id"],
            row["ligand_id"],
        )
        for row in all_rows
    }
    if len(unique_keys) != len(all_rows):
        raise ValueError("complete Stage 07b scores contain duplicate keys")

    scores_path = rooted_path(root, str(outputs["scores_csv"]))
    batches_path = rooted_path(root, str(outputs["batch_runs_csv"]))
    summary_path = rooted_path(root, str(outputs["run_summary_json"]))
    write_csv(scores_path, all_rows)
    batch_rows = [
        {
            "profile_id": summary["profile_id"],
            "seed_id": summary["seed_id"],
            "base_seed": summary["base_seed"],
            "receptor_id": summary["receptor_id"],
            "ligand_count": summary["ligand_count"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "score_minimum": summary["score_minimum"],
            "score_maximum": summary["score_maximum"],
            "engine_warning_count": summary.get(
                "engine_log_warnings", {}
            ).get("total_count", 0),
            "pose_integrity_failure_count": summary.get(
                "pose_integrity_audit", {}
            ).get("failure_count", len(ligands)),
            "signature": summary["signature"],
            "status": summary["status"],
        }
        for summary in batch_summaries
    ]
    write_csv(batches_path, batch_rows)
    profile_runtime = {
        profile_id: sum(
            float(row["elapsed_seconds"])
            for row in batch_rows
            if row["profile_id"] == profile_id
        )
        for profile_id in PROFILE_ORDER
    }
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "operation": "consumed Train-160 Uni-Dock enhanced-profile confirmation",
        "config": {
            "path": relative_path(root, config_path),
            "sha256": config_sha256,
        },
        "unidock_executable": executable_info,
        "input_audit": audit,
        "profiles": {
            profile_id: merged_protocol(config, profile_id)
            for profile_id in PROFILE_ORDER
        },
        "pair_count": len(all_rows),
        "batch_count": len(batch_rows),
        "executed_batches_this_invocation": executed_batches,
        "resumed_batches_this_invocation": resumed_batches,
        "profile_batch_elapsed_seconds": profile_runtime,
        "profile_pairs_per_second": {
            profile_id: (
                int(config["expected"]["pair_count_per_profile"])
                / profile_runtime[profile_id]
            )
            for profile_id in PROFILE_ORDER
        },
        "engine_warning_count": sum(
            int(row["engine_warning_count"]) for row in batch_rows
        ),
        "pose_integrity_failure_count": sum(
            int(row["pose_integrity_failure_count"]) for row in batch_rows
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
    run(
        args.config,
        args.root,
        args.unidock,
        args.audit_only,
        args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
