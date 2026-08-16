"""Run the frozen Stage32b PPARG MD-pair fresh-validation confirmation."""

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
    fixed = {"receptor_count": 2, "ligand_count": 1576, "seed_count": 3, "batch_count": 6, "score_row_count": 9456, "locked_test_rows": 0}
    if any(int(expected[key]) != value for key, value in fixed.items()):
        raise ValueError("Stage32b expected counts differ")
    seeds = tuple((str(row["seed_id"]), int(row["base_seed"])) for row in config["seeds"])
    if seeds != FROZEN_SEEDS:
        raise ValueError("Stage32b seed ledger differs")
    protocol = config["unidock"]
    if (protocol["profile_id"], int(protocol["exhaustiveness"]), int(protocol["max_step"])) != FROZEN_PROFILE:
        raise ValueError("Stage32b Uni-Dock profile differs")
    if config["evidence_timing"]["locked_test_rows_permitted"]:
        raise ValueError("Stage32b locked-test boundary differs")


def validate_inputs(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    preparation = common.read_json(common.rooted_path(root, config["outputs"]["preparation_result"]))
    if preparation.get("status") != "stage32b_pparg_md_pair_fresh_validation_inputs_ok":
        raise ValueError("Stage32b inputs are not ready")
    selection = common.read_json(common.rooted_path(root, config["outputs"]["train_selection_json"]))
    receptors = common.read_csv(common.rooted_path(root, config["outputs"]["selected_receptor_manifest"]))
    ligands = common.read_csv(common.rooted_path(root, config["outputs"]["prepared_ligand_manifest"]))
    if len(receptors) != 2 or {row["conformer_id"] for row in receptors} != set(selection["selected_pair"]["receptor_ids"]):
        raise ValueError("Stage32b receptor panel differs")
    if len(ligands) != 1576 or Counter(row["label"] for row in ligands) != Counter({"active": 75, "decoy": 1501}):
        raise ValueError("Stage32b ligand panel differs")
    if {row["selection_role"] for row in ligands} != {"fresh_validation"} or {row["split"] for row in ligands} != {"validation"}:
        raise ValueError("Stage32b protected split differs")
    for row in receptors:
        path = common.rooted_path(root, row["receptor_pdbqt"])
        if not path.is_file() or common.file_sha256(path) != row["receptor_pdbqt_sha256"].upper():
            raise ValueError(f"Stage32b receptor differs: {row['conformer_id']}")
    for row in ligands:
        path = common.rooted_path(root, row["pdbqt_path"])
        if not path.is_file() or common.file_sha256(path) != row["pdbqt_sha256"].upper():
            raise ValueError(f"Stage32b ligand differs: {row['ligand_id']}")
        if common.macrocycle_closure_atom_types(path):
            raise ValueError(f"Stage32b ligand retains closure pseudoatoms: {row['ligand_id']}")
    return receptors, ligands, {
        "status": "audit_only_ok",
        "target_id": "PPARG",
        "receptor_count": 2,
        "ligand_count": 1576,
        "label_counts": {"active": 75, "decoy": 1501},
        "seed_count": 3,
        "expected_batch_count": 6,
        "expected_score_row_count": 9456,
        "locked_test_rows": 0,
    }


def collect(root: Path, config: dict[str, Any], receptors: list[dict[str, str]], ligands: list[dict[str, str]], config_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing = []
    run_directory = common.rooted_path(root, config["outputs"]["run_directory"])
    ligand_ids = {row["ligand_id"] for row in ligands}
    for seed in config["seeds"]:
        for receptor in receptors:
            paths = common.batch_paths(run_directory, seed["seed_id"], receptor["conformer_id"])
            signature = common.protocol_signature(config_hash, seed["seed_id"], int(seed["base_seed"]), receptor, ligands, config["unidock"])
            checkpoint = common.checkpoint(root, paths, signature, ligand_ids)
            if checkpoint is None:
                missing.append({"seed_id": seed["seed_id"], "receptor_id": receptor["conformer_id"]})
            else:
                batch_rows, summary = checkpoint
                rows.extend(dict(row) for row in batch_rows)
                summaries.append(dict(summary))
    return rows, summaries, missing


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
                raise ValueError(f"incomplete Stage32b seed values: {ligand['ligand_id']}/{receptor_id}")
            record[receptor_id] = statistics.median(values) if aggregation == "median" else min(values)
        output.append(record)
    return output


def finalize(root: Path, config_path: Path, config: dict[str, Any], receptors: list[dict[str, str]], ligands: list[dict[str, str]], input_audit: dict[str, Any], executable: dict[str, Any] | None, executed: int, resumed: int, elapsed: float) -> dict[str, Any]:
    config_hash = common.file_sha256(config_path)
    rows, summaries, missing = collect(root, config, receptors, ligands, config_hash)
    outputs = config["outputs"]
    progress_path = common.rooted_path(root, outputs["progress_json"])
    progress = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage32b_production_complete" if not missing else "stage32b_partial_ok",
        "completed_batch_count": len(summaries),
        "missing_batch_count": len(missing),
        "missing_batches": missing,
        "completed_score_row_count": len(rows),
        "expected_batch_count": 6,
        "expected_score_row_count": 9456,
        "executed_batches_this_invocation": executed,
        "resumed_batches_this_invocation": resumed,
        "current_invocation_elapsed_seconds": elapsed,
        "locked_test_rows_read": 0,
    }
    common.write_json(progress_path, progress)
    if missing:
        print(json.dumps(progress, indent=2, sort_keys=True))
        return progress
    if len(rows) != 9456 or len({(row["seed_id"], row["receptor_id"], row["ligand_id"]) for row in rows}) != 9456:
        raise ValueError("complete Stage32b score coverage differs")
    if any(not math.isfinite(float(row["gpu_score"])) or abs(float(row["gpu_score"])) > 1000 for row in rows):
        raise ValueError("Stage32b contains an invalid score")
    seed_order = {value: index for index, (value, _) in enumerate(FROZEN_SEEDS)}
    receptor_ids = [row["conformer_id"] for row in receptors]
    receptor_order = {value: index for index, value in enumerate(receptor_ids)}
    ligand_order = {row["ligand_id"]: index for index, row in enumerate(ligands)}
    rows.sort(key=lambda row: (seed_order[row["seed_id"]], receptor_order[row["receptor_id"]], ligand_order[row["ligand_id"]]))
    scores_path = common.rooted_path(root, outputs["scores_csv"])
    batches_path = common.rooted_path(root, outputs["batch_runs_csv"])
    median_path = common.rooted_path(root, outputs["median_matrix_csv"])
    minimum_path = common.rooted_path(root, outputs["minimum_matrix_csv"])
    summary_path = common.rooted_path(root, outputs["summary_json"])
    common.write_csv(scores_path, rows)
    batch_rows = [{
        "seed_id": value["seed_id"], "base_seed": value["base_seed"], "receptor_id": value["receptor_id"],
        "ligand_count": value["ligand_count"], "elapsed_seconds": value["elapsed_seconds"],
        "known_warning_event_count": value["warning_adjudication"]["known_warning_event_count"],
        "unresolved_warning_event_count": value["warning_adjudication"]["unresolved_warning_event_count"],
        "pose_integrity_failure_count": value["pose_integrity_audit"]["failure_count"], "signature": value["signature"], "status": value["status"],
    } for value in summaries]
    batch_rows.sort(key=lambda row: (seed_order[row["seed_id"]], receptor_order[row["receptor_id"]]))
    common.write_csv(batches_path, batch_rows)
    common.write_csv(median_path, matrix_rows(rows, ligands, receptor_ids, "median"))
    common.write_csv(minimum_path, matrix_rows(rows, ligands, receptor_ids, "minimum"))
    result = {
        "schema_version": "1.0", "status": "stage32b_pparg_md_pair_fresh_validation_matrix_ok", "experiment_id": config["experiment_id"],
        "operation": "prospective PPARG fresh-validation Uni-Dock matrix generation for the frozen MD pair",
        "config": {"path": common.relative_path(root, config_path), "sha256": config_hash},
        "unidock_executable": executable, "input_audit": input_audit,
        "batch_count": 6, "score_row_count": 9456,
        "known_warning_event_count": sum(int(row["known_warning_event_count"]) for row in batch_rows),
        "unresolved_warning_event_count": 0, "pose_integrity_failure_count": 0,
        "batch_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in batch_rows),
        "data_boundary": {"fresh_validation_rows_read": 1576, "locked_test_rows_read": 0},
        "outputs": {key: common.output_descriptor(root, path) for key, path in {"scores_csv": scores_path, "batch_runs_csv": batches_path, "median_matrix_csv": median_path, "minimum_matrix_csv": minimum_path, "progress_json": progress_path}.items()},
        "next_gate": "apply the frozen Stage32b confirmation analysis without changing its thresholds",
        "decision_boundary": config["decision_boundary"],
    }
    common.write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run(config_path: Path, root: Path, unidock: str | None, audit_only: bool, resume: bool, finalize_only: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = common.read_json(config_path)
    validate_config(config)
    receptors, ligands, input_audit = validate_inputs(root, config)
    if audit_only:
        result = {"schema_version": "1.0", "experiment_id": config["experiment_id"], **input_audit, "operation": "input audit only; no docking started"}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    executable = None
    executed = 0
    resumed = 0
    started = time.perf_counter()
    if not finalize_only:
        executable = common.executable_evidence(unidock or config["unidock"]["executable"], config["unidock"]["required_package_version"])
        run_directory = common.rooted_path(root, config["outputs"]["run_directory"])
        config_hash = common.file_sha256(config_path)
        ligand_ids = {row["ligand_id"] for row in ligands}
        for seed in config["seeds"]:
            for receptor in receptors:
                paths = common.batch_paths(run_directory, seed["seed_id"], receptor["conformer_id"])
                paths["directory"].mkdir(parents=True, exist_ok=True)
                signature = common.protocol_signature(config_hash, seed["seed_id"], int(seed["base_seed"]), receptor, ligands, config["unidock"])
                existing = common.checkpoint(root, paths, signature, ligand_ids) if resume else None
                if existing is not None:
                    resumed += 1
                    print(f"resume ok: {seed['seed_id']}/{receptor['conformer_id']}", flush=True)
                    continue
                print(f"running: {seed['seed_id']}/{receptor['conformer_id']}", flush=True)
                batch_rows, summary = run_batch(root, paths, executable["resolved_executable"], receptor, ligands, config["unidock"], seed["seed_id"], int(seed["base_seed"]), signature)
                batch_rows, pose_audit = common.audit_batch_poses(root, ligands, batch_rows)
                warning = common.classify_warning_log(paths["log"], pose_audit)
                summary["pose_integrity_audit"] = pose_audit
                summary["warning_adjudication"] = warning
                summary["status"] = "ok" if int(pose_audit["failure_count"]) == 0 and int(warning["unresolved_warning_event_count"]) == 0 else "technical_integrity_failed"
                common.write_csv(paths["scores"], batch_rows)
                summary["scores_sha256"] = common.file_sha256(paths["scores"])
                common.write_json(paths["summary"], summary)
                if summary["status"] != "ok":
                    raise ValueError(f"Stage32b technical gate failed: {seed['seed_id']}/{receptor['conformer_id']}")
                executed += 1
                print(f"completed: {seed['seed_id']}/{receptor['conformer_id']} in {float(summary['elapsed_seconds']):.3f} s", flush=True)
    return finalize(root, config_path, config, receptors, ligands, input_audit, executable, executed, resumed, time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32b_pparg_md_pair_fresh_validation.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--unidock")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.root, args.unidock, args.audit_only, args.resume, args.finalize_only)
    return 0 if result["status"] in {"audit_only_ok", "stage32b_partial_ok", "stage32b_pparg_md_pair_fresh_validation_matrix_ok"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
