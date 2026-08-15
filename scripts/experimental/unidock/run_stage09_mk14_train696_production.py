"""Run the frozen MAPK14 Train-696 x 16 x 3 Uni-Dock production matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

try:
    from .run_stage07b_unidock_enhanced_confirmation import audit_batch_poses
    from .run_stage07c_unidock_warning_adjudication import classify_warning_log
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
    from run_stage07b_unidock_enhanced_confirmation import audit_batch_poses
    from run_stage07c_unidock_warning_adjudication import classify_warning_log
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


FROZEN_SEEDS = (
    ("seed0", 20260801),
    ("seed1", 20260802),
    ("seed2", 20260803),
)
FROZEN_PROFILE = ("enhanced", 1024, 80)


def verify_implementation(
    root: Path, config: dict[str, object], key: str, expected_path: Path
) -> None:
    descriptor = dict(config["implementation"])[key]
    path = rooted_path(root, str(descriptor["path"]))
    if path.resolve() != expected_path.resolve():
        raise ValueError(f"Stage 09 implementation path differs: {key}")
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage 09 implementation hash differs: {key}")


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
        "execution",
        "outputs",
        "decision_boundary",
    }
    if set(config) != required:
        raise ValueError("Stage 09 config keys differ")
    boundary = dict(config["data_boundary"])
    if int(boundary["validation_rows_permitted"]) != 0 or int(
        boundary["test_rows_permitted"]
    ) != 0:
        raise ValueError("Stage 09 crossed a frozen data boundary")
    seeds = tuple(
        (str(row["seed_id"]), int(row["base_seed"]))
        for row in dict(config["inputs"])["seeds"]
    )
    if seeds != FROZEN_SEEDS:
        raise ValueError("Stage 09 seed ledger differs")
    expected = dict(config["expected"])
    fixed_counts = {
        "receptor_count": 16,
        "ligand_count": 696,
        "seed_count": 3,
        "batch_count": 48,
        "pair_count": 33408,
        "validation_rows": 0,
        "test_rows": 0,
    }
    for key, value in fixed_counts.items():
        if int(expected[key]) != value:
            raise ValueError(f"Stage 09 expected count differs: {key}")
    protocol = dict(config["unidock"])
    profile = (
        str(protocol["profile_id"]),
        int(protocol["exhaustiveness"]),
        int(protocol["max_step"]),
    )
    if profile != FROZEN_PROFILE:
        raise ValueError("Stage 09 frozen Uni-Dock profile differs")
    if str(protocol["required_package_version"]) != "1.1.3":
        raise ValueError("Stage 09 Uni-Dock version differs")
    if int(protocol["num_modes"]) != 1:
        raise ValueError("Stage 09 must retain exactly one pose")


def validate_inputs(
    root: Path, config: dict[str, object]
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    inputs = dict(config["inputs"])
    expected = dict(config["expected"])
    boundary = dict(config["data_boundary"])
    receptor_path = verified_path(root, dict(inputs["receptor_manifest"]))
    ligand_path = verified_path(root, dict(inputs["ligand_manifest"]))
    admission_path = verified_path(root, dict(inputs["receptor_admission_audit"]))
    preparation_path = verified_path(root, dict(inputs["ligand_preparation_summary"]))
    profile_path = verified_path(root, dict(inputs["profile_freeze_result"]))
    admission = read_json(admission_path)
    preparation = read_json(preparation_path)
    profile = read_json(profile_path)
    if admission.get("status") != "independent_stage08c_final_replacement_audit_ok":
        raise ValueError("the final 16-receptor admission audit did not pass")
    if int(admission.get("final_receptor_count", 0)) != 16:
        raise ValueError("the admitted receptor count differs")
    if preparation.get("status") != "stage09_train696_unidock_inputs_ok":
        raise ValueError("the Stage 09 ligand preparation did not pass")
    if profile.get("status") != "unidock_profile_frozen_train_only":
        raise ValueError("the train-only Uni-Dock profile was not frozen")
    if profile.get("selected_profile_id") != "enhanced":
        raise ValueError("the selected Uni-Dock profile differs")

    receptors = read_csv(receptor_path)
    ligands = read_csv(ligand_path)
    receptor_ids = [row["conformer_id"] for row in receptors]
    if receptor_ids != [str(value) for value in expected["receptor_ids"]]:
        raise ValueError("Stage 09 receptor order differs")
    if len(receptors) != int(expected["receptor_count"]):
        raise ValueError("Stage 09 receptor count differs")
    if any(row["status"] != "ok" for row in receptors):
        raise ValueError("Stage 09 receptor manifest contains a failed row")
    if len(ligands) != int(expected["ligand_count"]):
        raise ValueError("Stage 09 ligand count differs")
    if len({row["ligand_id"] for row in ligands}) != len(ligands):
        raise ValueError("Stage 09 ligand manifest contains duplicate IDs")
    labels = Counter(row["label"] for row in ligands)
    expected_labels = Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    )
    if labels != expected_labels:
        raise ValueError("Stage 09 ligand labels differ")
    if {row["split"] for row in ligands} != {boundary["allowed_split"]}:
        raise ValueError("a non-train ligand is visible")
    if {row["selection_role"] for row in ligands} != {
        boundary["allowed_selection_role"]
    }:
        raise ValueError("Stage 09 ligand selection role differs")
    if any(row["pdbqt_status"] != "ok" for row in ligands):
        raise ValueError("Stage 09 ligand manifest contains a failed PDBQT")

    for rows, path_column, hash_column, id_column in (
        (receptors, "receptor_pdbqt", "receptor_pdbqt_sha256", "conformer_id"),
        (ligands, "pdbqt_path", "pdbqt_sha256", "ligand_id"),
    ):
        for row in rows:
            path = rooted_path(root, row[path_column])
            if not path.is_file() or file_sha256(path) != row[hash_column].upper():
                raise ValueError(f"prepared input identity differs: {row[id_column]}")
    pseudoatom_ids = [
        row["ligand_id"]
        for row in ligands
        if macrocycle_closure_atom_types(rooted_path(root, row["pdbqt_path"]))
    ]
    if pseudoatom_ids:
        raise ValueError(f"Stage 09 ligand inputs retain pseudoatoms: {pseudoatom_ids}")
    variants = Counter(row["preparation_variant"] for row in ligands)
    expected_variants = Counter(
        {"original_meeko_flexible": 681, "meeko_rigid_macrocycles": 15}
    )
    if variants != expected_variants:
        raise ValueError("Stage 09 preparation variants differ")
    audit = {
        "status": "audit_only_ok",
        "receptor_count": len(receptors),
        "receptor_ids": receptor_ids,
        "ligand_count": len(ligands),
        "label_counts": dict(sorted(labels.items())),
        "preparation_variant_counts": dict(sorted(variants.items())),
        "macrocycle_closure_pseudoatom_ligand_count": 0,
        "seed_count": len(FROZEN_SEEDS),
        "expected_batch_count": int(expected["batch_count"]),
        "expected_pair_count": int(expected["pair_count"]),
        "validation_rows": 0,
        "test_rows": 0,
    }
    return receptors, ligands, audit


def selected_records(
    records: list[dict[str, object]],
    key: str,
    requested: list[str] | None,
) -> list[dict[str, object]]:
    if not requested:
        return records
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate Stage 09 filter: {key}")
    allowed = {str(row[key]) for row in records}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"unknown Stage 09 {key} filter: {unknown}")
    requested_set = set(requested)
    return [row for row in records if str(row[key]) in requested_set]


def checkpoint(
    root: Path,
    paths: dict[str, Path],
    signature: str,
    ligand_ids: set[str],
) -> tuple[list[dict[str, str]], dict[str, object]] | None:
    value = validate_checkpoint(root, paths, signature, ligand_ids)
    if value is None:
        return None
    rows, summary = value
    pose_audit = dict(summary.get("pose_integrity_audit", {}))
    warning = dict(summary.get("warning_adjudication", {}))
    if not all(row.get("pose_integrity_status") == "ok" for row in rows):
        return None
    if int(pose_audit.get("failure_count", -1)) != 0:
        return None
    if int(warning.get("unresolved_warning_event_count", -1)) != 0:
        return None
    return rows, summary


def matrix_rows(
    rows: list[dict[str, object]],
    ligands: list[dict[str, str]],
    receptor_ids: list[str],
    aggregation: str,
) -> list[dict[str, object]]:
    scores: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["ligand_id"]), str(row["receptor_id"]))
        scores.setdefault(key, []).append(float(row["gpu_score"]))
    output: list[dict[str, object]] = []
    for ligand in ligands:
        ligand_id = ligand["ligand_id"]
        result: dict[str, object] = {
            "ligand_id": ligand_id,
            "label": ligand["label"],
            "selection_role": ligand["selection_role"],
        }
        for receptor_id in receptor_ids:
            values = scores.get((ligand_id, receptor_id), [])
            if len(values) != len(FROZEN_SEEDS):
                raise ValueError(f"incomplete seed values: {ligand_id}/{receptor_id}")
            if aggregation == "median":
                result[receptor_id] = statistics.median(values)
            elif aggregation == "minimum":
                result[receptor_id] = min(values)
            else:
                raise ValueError(f"unknown Stage 09 aggregation: {aggregation}")
        output.append(result)
    return output


def collect_batches(
    root: Path,
    config: dict[str, object],
    receptors: list[dict[str, str]],
    ligands: list[dict[str, str]],
    config_sha256: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, str]],
]:
    run_directory = rooted_path(root, str(config["outputs"]["run_directory"]))
    protocol = dict(config["unidock"])
    ligand_ids = {row["ligand_id"] for row in ligands}
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    for seed in dict(config["inputs"])["seeds"]:
        seed_id = str(seed["seed_id"])
        base_seed = int(seed["base_seed"])
        for receptor in receptors:
            receptor_id = receptor["conformer_id"]
            paths = batch_paths(run_directory, seed_id, receptor_id)
            signature = protocol_signature(
                config_sha256,
                seed_id,
                base_seed,
                receptor,
                ligands,
                protocol,
            )
            value = checkpoint(root, paths, signature, ligand_ids)
            if value is None:
                missing.append({"seed_id": seed_id, "receptor_id": receptor_id})
                continue
            rows, summary = value
            all_rows.extend(dict(row) for row in rows)
            summaries.append(dict(summary))
    return all_rows, summaries, missing


def finalize(
    root: Path,
    config_path: Path,
    config: dict[str, object],
    receptors: list[dict[str, str]],
    ligands: list[dict[str, str]],
    input_audit: dict[str, object],
    executable_info: dict[str, object] | None,
    executed_batches: int,
    resumed_batches: int,
    invocation_elapsed: float,
    selected_seed_ids: list[str],
    selected_receptor_ids: list[str],
) -> dict[str, object]:
    outputs = dict(config["outputs"])
    config_sha256 = file_sha256(config_path)
    rows, summaries, missing = collect_batches(
        root, config, receptors, ligands, config_sha256
    )
    progress_path = rooted_path(root, str(outputs["progress_json"]))
    progress = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage09_production_complete" if not missing else "stage09_partial_ok",
        "selected_seed_ids": selected_seed_ids,
        "selected_receptor_ids": selected_receptor_ids,
        "completed_batch_count": len(summaries),
        "missing_batch_count": len(missing),
        "missing_batches": missing,
        "completed_pair_count": len(rows),
        "expected_batch_count": int(config["expected"]["batch_count"]),
        "expected_pair_count": int(config["expected"]["pair_count"]),
        "executed_batches_this_invocation": executed_batches,
        "resumed_batches_this_invocation": resumed_batches,
        "current_invocation_elapsed_seconds": invocation_elapsed,
        "validation_rows_read": 0,
        "test_rows_read": 0,
    }
    write_json(progress_path, progress)
    if missing:
        print(json.dumps(progress, indent=2, sort_keys=True))
        return progress

    expected_pairs = int(config["expected"]["pair_count"])
    if len(rows) != expected_pairs:
        raise ValueError("complete Stage 09 pair count differs")
    unique = {
        (str(row["seed_id"]), str(row["receptor_id"]), str(row["ligand_id"]))
        for row in rows
    }
    if len(unique) != expected_pairs:
        raise ValueError("complete Stage 09 keys are not unique")
    seed_order = {seed_id: index for index, (seed_id, _) in enumerate(FROZEN_SEEDS)}
    receptor_ids = [row["conformer_id"] for row in receptors]
    receptor_order = {value: index for index, value in enumerate(receptor_ids)}
    ligand_order = {row["ligand_id"]: index for index, row in enumerate(ligands)}
    rows.sort(
        key=lambda row: (
            seed_order[str(row["seed_id"])],
            receptor_order[str(row["receptor_id"])],
            ligand_order[str(row["ligand_id"])],
        )
    )
    for row in rows:
        score = float(row["gpu_score"])
        if not math.isfinite(score) or abs(score) > float(
            config["unidock"]["maximum_absolute_score_kcal_per_mol"]
        ):
            raise ValueError(f"invalid Stage 09 score: {row['ligand_id']}")

    scores_path = rooted_path(root, str(outputs["scores_csv"]))
    batches_path = rooted_path(root, str(outputs["batch_runs_csv"]))
    median_path = rooted_path(root, str(outputs["median_matrix_csv"]))
    minimum_path = rooted_path(root, str(outputs["minimum_matrix_csv"]))
    summary_path = rooted_path(root, str(outputs["summary_json"]))
    write_csv(scores_path, rows)
    batch_rows = [
        {
            "seed_id": summary["seed_id"],
            "base_seed": summary["base_seed"],
            "receptor_id": summary["receptor_id"],
            "ligand_count": summary["ligand_count"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "score_minimum": summary["score_minimum"],
            "score_maximum": summary["score_maximum"],
            "known_warning_event_count": dict(
                summary["warning_adjudication"]
            )["known_warning_event_count"],
            "unresolved_warning_event_count": dict(
                summary["warning_adjudication"]
            )["unresolved_warning_event_count"],
            "pose_integrity_failure_count": dict(
                summary["pose_integrity_audit"]
            )["failure_count"],
            "signature": summary["signature"],
            "status": summary["status"],
        }
        for summary in summaries
    ]
    batch_rows.sort(
        key=lambda row: (
            seed_order[str(row["seed_id"])],
            receptor_order[str(row["receptor_id"])],
        )
    )
    write_csv(batches_path, batch_rows)
    write_csv(median_path, matrix_rows(rows, ligands, receptor_ids, "median"))
    write_csv(minimum_path, matrix_rows(rows, ligands, receptor_ids, "minimum"))
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage09_train696_unidock_matrix_ok",
        "operation": "train-only Uni-Dock production score generation",
        "config": {
            "path": relative_path(root, config_path),
            "sha256": config_sha256,
        },
        "unidock_executable": executable_info,
        "input_audit": input_audit,
        "frozen_protocol": dict(config["unidock"]),
        "batch_count": len(batch_rows),
        "pair_count": len(rows),
        "known_warning_event_count": sum(
            int(row["known_warning_event_count"]) for row in batch_rows
        ),
        "unresolved_warning_event_count": 0,
        "pose_integrity_failure_count": 0,
        "batch_elapsed_seconds": sum(
            float(row["elapsed_seconds"]) for row in batch_rows
        ),
        "pairs_per_batch_second": len(rows)
        / sum(float(row["elapsed_seconds"]) for row in batch_rows),
        "executed_batches_this_invocation": executed_batches,
        "resumed_batches_this_invocation": resumed_batches,
        "current_invocation_elapsed_seconds": invocation_elapsed,
        "aggregation": {
            "primary": "median across the three paired seeds",
            "sensitivity": "minimum across the three paired seeds",
            "score_direction": "more negative is more favorable",
        },
        "data_boundary": {
            "validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            "scores_csv": output_descriptor(root, scores_path),
            "batch_runs_csv": output_descriptor(root, batches_path),
            "median_matrix_csv": output_descriptor(root, median_path),
            "minimum_matrix_csv": output_descriptor(root, minimum_path),
            "progress_json": output_descriptor(root, progress_path),
        },
        "next_gate": "run the independent Stage 09 matrix audit before any train-only receptor-selection analysis",
        "interpretation_note": config["decision_boundary"],
    }
    write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run(
    config_path: Path,
    root: Path,
    unidock: str | None,
    audit_only: bool,
    resume: bool,
    seed_ids: list[str] | None,
    receptor_ids: list[str] | None,
    finalize_only: bool,
) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    validate_config(config)
    implementation_paths = {
        "runner": Path(__file__),
        "unidock_batch_helper": Path(__file__).with_name(
            "run_unidock_gpu_equivalence.py"
        ),
        "pose_audit_helper": Path(__file__).with_name(
            "run_stage07b_unidock_enhanced_confirmation.py"
        ),
        "warning_adjudication_helper": Path(__file__).with_name(
            "run_stage07c_unidock_warning_adjudication.py"
        ),
    }
    for key, path in implementation_paths.items():
        verify_implementation(root, config, key, path)
    receptors, ligands, input_audit = validate_inputs(root, config)
    seeds = [dict(row) for row in dict(config["inputs"])["seeds"]]
    selected_seeds = selected_records(seeds, "seed_id", seed_ids)
    selected_receptors = selected_records(
        [dict(row) for row in receptors], "conformer_id", receptor_ids
    )
    selected_seed_ids = [str(row["seed_id"]) for row in selected_seeds]
    selected_receptor_ids = [str(row["conformer_id"]) for row in selected_receptors]
    if audit_only:
        result = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "config": {
                "path": relative_path(root, config_path),
                "sha256": file_sha256(config_path),
            },
            **input_audit,
            "selected_seed_ids": selected_seed_ids,
            "selected_receptor_ids": selected_receptor_ids,
            "selected_batch_count": len(selected_seeds) * len(selected_receptors),
            "selected_pair_count": len(selected_seeds)
            * len(selected_receptors)
            * len(ligands),
            "operation": "input audit only; no GPU docking was started",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    executable_info: dict[str, object] | None = None
    executed_batches = 0
    resumed_batches = 0
    invocation_started = time.perf_counter()
    if not finalize_only:
        protocol = dict(config["unidock"])
        executable_info = executable_evidence(
            unidock or str(protocol["executable"]),
            str(protocol["required_package_version"]),
        )
        run_directory = rooted_path(root, str(config["outputs"]["run_directory"]))
        config_sha256 = file_sha256(config_path)
        ligand_ids = {row["ligand_id"] for row in ligands}
        for seed in selected_seeds:
            seed_id = str(seed["seed_id"])
            base_seed = int(seed["base_seed"])
            for receptor in selected_receptors:
                receptor_id = str(receptor["conformer_id"])
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
                value = (
                    checkpoint(root, paths, signature, ligand_ids) if resume else None
                )
                if value is not None:
                    resumed_batches += 1
                    print(f"resume ok: {seed_id}/{receptor_id}", flush=True)
                    continue
                print(f"running: {seed_id}/{receptor_id}", flush=True)
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
                warning = classify_warning_log(paths["log"], pose_audit)
                summary["pose_integrity_audit"] = pose_audit
                summary["warning_adjudication"] = warning
                summary["status"] = (
                    "ok"
                    if int(pose_audit["failure_count"]) == 0
                    and int(warning["unresolved_warning_event_count"]) == 0
                    else "technical_integrity_failed"
                )
                write_csv(paths["scores"], rows)
                summary["scores_sha256"] = file_sha256(paths["scores"])
                write_json(paths["summary"], summary)
                if summary["status"] != "ok":
                    raise ValueError(f"Stage 09 technical gate failed: {seed_id}/{receptor_id}")
                executed_batches += 1
                print(
                    f"completed: {seed_id}/{receptor_id} "
                    f"in {float(summary['elapsed_seconds']):.3f} s",
                    flush=True,
                )
    return finalize(
        root,
        config_path,
        config,
        receptors,
        ligands,
        input_audit,
        executable_info,
        executed_batches,
        resumed_batches,
        time.perf_counter() - invocation_started,
        selected_seed_ids,
        selected_receptor_ids,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--unidock", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-id", action="append", default=None)
    parser.add_argument("--receptor-id", action="append", default=None)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only and args.finalize_only:
        parser.error("--audit-only and --finalize-only are mutually exclusive")
    result = run(
        args.config,
        args.root,
        args.unidock,
        args.audit_only,
        args.resume,
        args.seed_id,
        args.receptor_id,
        args.finalize_only,
    )
    return 0 if result["status"] in {
        "audit_only_ok",
        "stage09_partial_ok",
        "stage09_train696_unidock_matrix_ok",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
