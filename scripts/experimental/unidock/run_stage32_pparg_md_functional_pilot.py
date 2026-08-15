"""Run the frozen Stage32 PPARG MD-frame Train-160 Uni-Dock pilot."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experimental.unidock import run_stage09_mk14_train696_production as common
from scripts.experimental.unidock.run_unidock_batch_targeted import run_batch


FROZEN_SEEDS = (("seed0", 20260801), ("seed1", 20260802), ("seed2", 20260803))
FROZEN_PROFILE = ("enhanced", 1024, 80)


def validate_config(config: dict[str, Any]) -> None:
    expected = config["expected"]
    fixed = {"receptor_count": 16, "ligand_count": 160, "seed_count": 3, "batch_count": 48, "pair_count": 7680, "fresh_validation_rows": 0, "test_rows": 0}
    for key, value in fixed.items():
        if int(expected[key]) != value:
            raise ValueError(f"Stage32 expected count differs: {key}")
    seeds = tuple((str(row["seed_id"]), int(row["base_seed"])) for row in config["seeds"])
    if seeds != FROZEN_SEEDS:
        raise ValueError("Stage32 seed ledger differs")
    protocol = config["unidock"]
    if (str(protocol["profile_id"]), int(protocol["exhaustiveness"]), int(protocol["max_step"])) != FROZEN_PROFILE:
        raise ValueError("Stage32 Uni-Dock profile differs")
    timing = config["evidence_timing"]
    if timing["fresh_validation_rows_permitted"] or timing["test_rows_permitted"]:
        raise ValueError("Stage32 protected-data boundary differs")


def validate_inputs(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    outputs = config["outputs"]
    preparation_path = common.rooted_path(root, outputs["preparation_result"])
    preparation = common.read_json(preparation_path)
    if preparation.get("status") != "stage32_inputs_ok":
        raise ValueError("Stage32 receptors have not passed materialization")
    receptors = common.read_csv(common.rooted_path(root, outputs["prepared_receptor_manifest"]))
    ligands = common.read_csv(common.rooted_path(root, outputs["selected_ligand_manifest"]))
    if len(receptors) != 16 or len({row["conformer_id"] for row in receptors}) != 16:
        raise ValueError("Stage32 receptor manifest differs")
    if Counter(int(row["start_index"]) for row in receptors) != Counter({value: 2 for value in range(8)}):
        raise ValueError("Stage32 receptors are not two-per-start balanced")
    if any(row["status"] != "ok" for row in receptors):
        raise ValueError("Stage32 receptor preparation contains a failure")
    if len(ligands) != 160 or Counter(row["label"] for row in ligands) != Counter({"active": 80, "decoy": 80}):
        raise ValueError("Stage32 ligand panel differs")
    timing = config["evidence_timing"]
    if {row["split"] for row in ligands} != {timing["allowed_split"]} or {row["selection_role"] for row in ligands} != {timing["allowed_selection_role"]}:
        raise ValueError("Stage32 ligand boundary differs")
    if any(row["target_id"] != "PPARG" or row["pdbqt_status"] != "ok" for row in ligands):
        raise ValueError("Stage32 ligand target or preparation differs")
    for row in receptors:
        path = common.rooted_path(root, row["receptor_pdbqt"])
        if not path.is_file() or common.file_sha256(path) != row["receptor_pdbqt_sha256"].upper():
            raise ValueError(f"Stage32 receptor PDBQT differs: {row['conformer_id']}")
    for row in ligands:
        path = common.rooted_path(root, row["pdbqt_path"])
        if not path.is_file() or common.file_sha256(path) != row["pdbqt_sha256"].upper():
            raise ValueError(f"Stage32 ligand PDBQT differs: {row['ligand_id']}")
        if common.macrocycle_closure_atom_types(path):
            raise ValueError(f"Stage32 ligand retains closure pseudoatoms: {row['ligand_id']}")
    return receptors, ligands, {
        "status": "audit_only_ok",
        "target_id": "PPARG",
        "experiment_class": config["evidence_timing"]["experiment_class"],
        "receptor_count": len(receptors),
        "represented_start_count": len({row["start_index"] for row in receptors}),
        "ligand_count": len(ligands),
        "label_counts": dict(sorted(Counter(row["label"] for row in ligands).items())),
        "seed_count": len(config["seeds"]),
        "expected_batch_count": config["expected"]["batch_count"],
        "expected_pair_count": config["expected"]["pair_count"],
        "fresh_validation_rows": 0,
        "test_rows": 0,
    }


def selected_records(records: list[dict[str, Any]], key: str, requested: list[str] | None) -> list[dict[str, Any]]:
    if requested is None:
        return records
    unknown = sorted(set(requested) - {str(row[key]) for row in records})
    if unknown:
        raise ValueError(f"unknown Stage32 {key}: {unknown}")
    requested_set = set(requested)
    return [row for row in records if str(row[key]) in requested_set]


def collect_batches(root: Path, config: dict[str, Any], receptors: list[dict[str, str]], ligands: list[dict[str, str]], config_sha256: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    run_directory = common.rooted_path(root, config["outputs"]["run_directory"])
    protocol = config["unidock"]
    ligand_ids = {row["ligand_id"] for row in ligands}
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for seed in config["seeds"]:
        for receptor in receptors:
            paths = common.batch_paths(run_directory, seed["seed_id"], receptor["conformer_id"])
            signature = common.protocol_signature(config_sha256, seed["seed_id"], int(seed["base_seed"]), receptor, ligands, protocol)
            value = common.checkpoint(root, paths, signature, ligand_ids)
            if value is None:
                missing.append({"seed_id": seed["seed_id"], "receptor_id": receptor["conformer_id"]})
            else:
                rows, summary = value
                all_rows.extend(dict(row) for row in rows)
                summaries.append(dict(summary))
    return all_rows, summaries, missing


def matrix_rows(rows: list[dict[str, Any]], ligands: list[dict[str, str]], receptor_ids: list[str], aggregation: str) -> list[dict[str, Any]]:
    scores: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        scores.setdefault((str(row["ligand_id"]), str(row["receptor_id"])), []).append(float(row["gpu_score"]))
    output = []
    for ligand in ligands:
        record: dict[str, Any] = {"ligand_id": ligand["ligand_id"], "label": ligand["label"], "selection_role": ligand["selection_role"], "split_group_id": ligand["split_group_id"]}
        for receptor_id in receptor_ids:
            values = scores.get((ligand["ligand_id"], receptor_id), [])
            if len(values) != 3:
                raise ValueError(f"incomplete Stage32 seed values: {ligand['ligand_id']}/{receptor_id}")
            record[receptor_id] = statistics.median(values) if aggregation == "median" else min(values)
        output.append(record)
    return output


def finalize(root: Path, config_path: Path, config: dict[str, Any], receptors: list[dict[str, str]], ligands: list[dict[str, str]], input_audit: dict[str, Any], executable_info: dict[str, Any] | None, executed: int, resumed: int, elapsed: float, selected_seed_ids: list[str], selected_receptor_ids: list[str]) -> dict[str, Any]:
    config_hash = common.file_sha256(config_path)
    rows, summaries, missing = collect_batches(root, config, receptors, ligands, config_hash)
    outputs = config["outputs"]
    progress_path = common.rooted_path(root, outputs["progress_json"])
    progress = {
        "schema_version": "1.0", "experiment_id": config["experiment_id"],
        "status": "stage32_production_complete" if not missing else "stage32_partial_ok",
        "selected_seed_ids": selected_seed_ids, "selected_receptor_ids": selected_receptor_ids,
        "completed_batch_count": len(summaries), "missing_batch_count": len(missing), "missing_batches": missing,
        "completed_pair_count": len(rows), "expected_batch_count": 48, "expected_pair_count": 7680,
        "executed_batches_this_invocation": executed, "resumed_batches_this_invocation": resumed,
        "current_invocation_elapsed_seconds": elapsed, "fresh_validation_rows_read": 0, "test_rows_read": 0,
    }
    common.write_json(progress_path, progress)
    if missing:
        print(json.dumps(progress, indent=2, sort_keys=True))
        return progress
    if len(rows) != 7680 or len({(row["seed_id"], row["receptor_id"], row["ligand_id"]) for row in rows}) != 7680:
        raise ValueError("complete Stage32 pair coverage differs")
    seed_order = {seed_id: index for index, (seed_id, _) in enumerate(FROZEN_SEEDS)}
    receptor_ids = [row["conformer_id"] for row in receptors]
    receptor_order = {value: index for index, value in enumerate(receptor_ids)}
    ligand_order = {row["ligand_id"]: index for index, row in enumerate(ligands)}
    rows.sort(key=lambda row: (seed_order[str(row["seed_id"])], receptor_order[str(row["receptor_id"])], ligand_order[str(row["ligand_id"])]))
    if any(not math.isfinite(float(row["gpu_score"])) or abs(float(row["gpu_score"])) > float(config["unidock"]["maximum_absolute_score_kcal_per_mol"]) for row in rows):
        raise ValueError("Stage32 contains an invalid score")
    scores_path = common.rooted_path(root, outputs["scores_csv"])
    batches_path = common.rooted_path(root, outputs["batch_runs_csv"])
    median_path = common.rooted_path(root, outputs["median_matrix_csv"])
    minimum_path = common.rooted_path(root, outputs["minimum_matrix_csv"])
    summary_path = common.rooted_path(root, outputs["summary_json"])
    common.write_csv(scores_path, rows)
    batch_rows = [{
        "seed_id": value["seed_id"], "base_seed": value["base_seed"], "receptor_id": value["receptor_id"],
        "ligand_count": value["ligand_count"], "elapsed_seconds": value["elapsed_seconds"],
        "score_minimum": value["score_minimum"], "score_maximum": value["score_maximum"],
        "known_warning_event_count": value["warning_adjudication"]["known_warning_event_count"],
        "unresolved_warning_event_count": value["warning_adjudication"]["unresolved_warning_event_count"],
        "pose_integrity_failure_count": value["pose_integrity_audit"]["failure_count"], "signature": value["signature"], "status": value["status"],
    } for value in summaries]
    batch_rows.sort(key=lambda row: (seed_order[str(row["seed_id"])], receptor_order[str(row["receptor_id"])]))
    common.write_csv(batches_path, batch_rows)
    common.write_csv(median_path, matrix_rows(rows, ligands, receptor_ids, "median"))
    common.write_csv(minimum_path, matrix_rows(rows, ligands, receptor_ids, "minimum"))
    result = {
        "schema_version": "1.0", "experiment_id": config["experiment_id"], "status": "stage32_pparg_md_functional_pilot_matrix_ok",
        "operation": "prospective train-only PPARG MD-frame Uni-Dock pilot matrix generation",
        "config": {"path": common.relative_path(root, config_path), "sha256": config_hash},
        "unidock_executable": executable_info, "input_audit": input_audit, "frozen_protocol": config["unidock"],
        "batch_count": len(batch_rows), "pair_count": len(rows),
        "known_warning_event_count": sum(int(row["known_warning_event_count"]) for row in batch_rows),
        "unresolved_warning_event_count": 0, "pose_integrity_failure_count": 0,
        "batch_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in batch_rows),
        "executed_batches_this_invocation": executed, "resumed_batches_this_invocation": resumed,
        "current_invocation_elapsed_seconds": elapsed,
        "aggregation": {"primary": "median across three paired seeds", "sensitivity": "minimum across three paired seeds", "score_direction": "more negative is more favorable"},
        "data_boundary": {"train_rows_read": 160, "fresh_validation_rows_read": 0, "test_rows_read": 0},
        "outputs": {key: common.output_descriptor(root, path) for key, path in {"scores_csv": scores_path, "batch_runs_csv": batches_path, "median_matrix_csv": median_path, "minimum_matrix_csv": minimum_path, "progress_json": progress_path}.items()},
        "next_gate": "independently audit the complete matrix, then run the frozen functional-landscape analysis",
        "interpretation_note": config["interpretation_boundary"],
    }
    common.write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run(config_path: Path, root: Path, unidock: str | None, audit_only: bool, resume: bool, seed_ids: list[str] | None, receptor_ids: list[str] | None, finalize_only: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = common.read_json(config_path)
    validate_config(config)
    receptors, ligands, input_audit = validate_inputs(root, config)
    selected_seeds = selected_records([dict(row) for row in config["seeds"]], "seed_id", seed_ids)
    selected_receptors = selected_records([dict(row) for row in receptors], "conformer_id", receptor_ids)
    selected_seed_ids = [row["seed_id"] for row in selected_seeds]
    selected_receptor_ids = [row["conformer_id"] for row in selected_receptors]
    if audit_only:
        result = {"schema_version": "1.0", "experiment_id": config["experiment_id"], **input_audit, "selected_seed_ids": selected_seed_ids, "selected_receptor_ids": selected_receptor_ids, "selected_batch_count": len(selected_seeds) * len(selected_receptors), "selected_pair_count": len(selected_seeds) * len(selected_receptors) * len(ligands), "operation": "input audit only; no GPU docking was started"}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    executable_info = None
    executed = 0
    resumed = 0
    started = time.perf_counter()
    if not finalize_only:
        protocol = config["unidock"]
        executable_info = common.executable_evidence(unidock or protocol["executable"], protocol["required_package_version"])
        run_directory = common.rooted_path(root, config["outputs"]["run_directory"])
        config_hash = common.file_sha256(config_path)
        ligand_ids = {row["ligand_id"] for row in ligands}
        for seed in selected_seeds:
            for receptor in selected_receptors:
                paths = common.batch_paths(run_directory, seed["seed_id"], receptor["conformer_id"])
                paths["directory"].mkdir(parents=True, exist_ok=True)
                signature = common.protocol_signature(config_hash, seed["seed_id"], int(seed["base_seed"]), receptor, ligands, protocol)
                value = common.checkpoint(root, paths, signature, ligand_ids) if resume else None
                if value is not None:
                    resumed += 1
                    print(f"resume ok: {seed['seed_id']}/{receptor['conformer_id']}", flush=True)
                    continue
                print(f"running: {seed['seed_id']}/{receptor['conformer_id']}", flush=True)
                rows, summary = run_batch(root, paths, executable_info["resolved_executable"], receptor, ligands, protocol, seed["seed_id"], int(seed["base_seed"]), signature)
                rows, pose_audit = common.audit_batch_poses(root, ligands, rows)
                warning = common.classify_warning_log(paths["log"], pose_audit)
                summary["pose_integrity_audit"] = pose_audit
                summary["warning_adjudication"] = warning
                summary["status"] = "ok" if int(pose_audit["failure_count"]) == 0 and int(warning["unresolved_warning_event_count"]) == 0 else "technical_integrity_failed"
                common.write_csv(paths["scores"], rows)
                summary["scores_sha256"] = common.file_sha256(paths["scores"])
                common.write_json(paths["summary"], summary)
                if summary["status"] != "ok":
                    raise ValueError(f"Stage32 technical gate failed: {seed['seed_id']}/{receptor['conformer_id']}")
                executed += 1
                print(f"completed: {seed['seed_id']}/{receptor['conformer_id']} in {float(summary['elapsed_seconds']):.3f} s", flush=True)
    return finalize(root, config_path, config, receptors, ligands, input_audit, executable_info, executed, resumed, time.perf_counter() - started, selected_seed_ids, selected_receptor_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32_pparg_md_functional_complementarity_pilot.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--unidock")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-id", action="append")
    parser.add_argument("--receptor-id", action="append")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only and args.finalize_only:
        parser.error("--audit-only and --finalize-only are mutually exclusive")
    result = run(args.config, args.root, args.unidock, args.audit_only, args.resume, args.seed_id, args.receptor_id, args.finalize_only)
    return 0 if result["status"] in {"audit_only_ok", "stage32_partial_ok", "stage32_pparg_md_functional_pilot_matrix_ok"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
