"""Run the Stage43 PPARG MD-96 matrix, reusing 16 audited Stage32 receptors."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.experimental.unidock import run_stage09_mk14_train696_production as common
from scripts.experimental.unidock.run_unidock_batch_targeted import run_batch


FROZEN_SEEDS = (("seed0", 20260801), ("seed1", 20260802), ("seed2", 20260803))
FROZEN_PROFILE = ("enhanced", 1024, 80)
RESCUE_AMENDMENT = Path("configs/stage43_pparg_md96_technical_rescue_amendment01.json")


def load_rescue_amendment(root: Path) -> dict[str, Any]:
    path = root / RESCUE_AMENDMENT
    if not path.is_file():
        return {"enabled": False}
    value = common.read_json(path)
    if value.get("status") != "frozen_technical_rescue_amendment":
        raise ValueError("Stage43 technical rescue amendment status differs")
    return value


def batch_signatures(
    config_hash: str, seed: dict[str, Any], receptor: dict[str, str],
    ligands: list[dict[str, str]], protocol: dict[str, Any], amendment: dict[str, Any],
) -> list[tuple[str, int, bool]]:
    primary_seed = int(seed["base_seed"])
    output = [(
        common.protocol_signature(config_hash, seed["seed_id"], primary_seed, receptor, ligands, protocol),
        primary_seed,
        False,
    )]
    trigger = amendment.get("trigger", {})
    if amendment.get("enabled") and seed["seed_id"] == trigger.get("seed_id") and receptor["conformer_id"] == trigger.get("receptor_id"):
        rescue_seed = primary_seed + int(amendment["rescue"]["base_seed_increment"])
        output.append((
            common.protocol_signature(config_hash, seed["seed_id"], rescue_seed, receptor, ligands, protocol),
            rescue_seed,
            True,
        ))
    return output


def rescue_checkpoint(
    root: Path, paths: dict[str, Path], signatures: list[tuple[str, int, bool]],
    ligand_ids: set[str],
) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    for signature, _base_seed, _rescued in signatures:
        value = common.checkpoint(root, paths, signature, ligand_ids)
        if value is not None:
            return value
    return None


def validate_config(config: dict[str, Any]) -> None:
    expected = config["expected"]
    fixed = {
        "receptor_count": 96, "historical_receptor_count": 16,
        "new_receptor_count": 80, "ligand_count": 160,
        "seed_count": 3, "combined_batch_count": 288,
        "new_batch_count": 240, "combined_pair_count": 46080,
        "historical_pair_count": 7680, "new_pair_count": 38400,
        "fresh_validation_rows": 0, "test_rows": 0,
    }
    for key, value in fixed.items():
        if int(expected[key]) != value:
            raise ValueError(f"Stage43 expected count differs: {key}")
    seeds = tuple((row["seed_id"], int(row["base_seed"])) for row in config["inputs"]["seeds"])
    if seeds != FROZEN_SEEDS:
        raise ValueError("Stage43 seed ledger differs")
    protocol = config["unidock"]
    if (protocol["profile_id"], int(protocol["exhaustiveness"]), int(protocol["max_step"])) != FROZEN_PROFILE:
        raise ValueError("Stage43 Uni-Dock profile differs")
    timing = config["evidence_timing"]
    if timing["fresh_validation_rows_permitted"] or timing["test_rows_permitted"] or timing["same_target_objective_weight_search_permitted"]:
        raise ValueError("Stage43 evidence boundary differs")


def validate_inputs(
    root: Path, config: dict[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    inputs = config["inputs"]
    outputs = config["outputs"]
    if common.read_json(common.rooted_path(root, inputs["stage42f_audit"])).get("status") != "stage42d_f_bace1_qubo_independent_audit_ok":
        raise ValueError("Stage42f independent audit did not pass")
    if common.read_json(common.rooted_path(root, inputs["stage32_matrix_audit"])).get("status") != "stage32_pparg_md_functional_pilot_matrix_audit_ok":
        raise ValueError("Stage32 historical matrix audit did not pass")
    if common.read_json(common.rooted_path(root, inputs["stage32_summary"])).get("status") != "stage32_pparg_md_functional_pilot_matrix_ok":
        raise ValueError("Stage32 historical matrix is incomplete")
    preparation = common.read_json(common.rooted_path(root, outputs["preparation_result"]))
    if preparation.get("status") != "stage43_pparg_md96_inputs_ok":
        raise ValueError("Stage43 receptor materialization did not pass")
    receptors = common.read_csv(common.rooted_path(root, outputs["prepared_receptor_manifest"]))
    ligands = common.read_csv(common.rooted_path(root, inputs["stage32_ligand_manifest"]))
    historical_rows = [dict(row) for row in common.read_csv(common.rooted_path(root, inputs["stage32_scores"]))]
    historical_batches = common.read_csv(common.rooted_path(root, inputs["stage32_batch_runs"]))
    if len(receptors) != 96 or Counter(int(row["start_index"]) for row in receptors) != Counter({index: 12 for index in range(8)}):
        raise ValueError("Stage43 receptor panel differs")
    historical_receptors = [row for row in receptors if row["evidence_role"] == "historical_stage32_reuse"]
    new_receptors = [row for row in receptors if row["evidence_role"] == "new_stage43_docking"]
    if len(historical_receptors) != 16 or len(new_receptors) != 80:
        raise ValueError("Stage43 historical/new receptor partition differs")
    if any(row["status"] != "historical_stage32_reuse" for row in historical_receptors):
        raise ValueError("Stage43 historical receptor status differs")
    if any(row["status"] != "ok" for row in new_receptors):
        raise ValueError("Stage43 contains an unprepared new receptor")
    for row in new_receptors:
        path = common.rooted_path(root, row["receptor_pdbqt"])
        if not path.is_file() or common.file_sha256(path) != row["receptor_pdbqt_sha256"].upper():
            raise ValueError(f"Stage43 receptor PDBQT differs: {row['conformer_id']}")
    if len(ligands) != 160 or Counter(row["label"] for row in ligands) != Counter({"active": 80, "decoy": 80}):
        raise ValueError("Stage43 ligand panel differs")
    for row in ligands:
        path = common.rooted_path(root, row["pdbqt_path"])
        if not path.is_file() or common.file_sha256(path) != row["pdbqt_sha256"].upper():
            raise ValueError(f"Stage43 ligand PDBQT differs: {row['ligand_id']}")
    historical_ids = {row["conformer_id"] for row in historical_receptors}
    ligand_ids = {row["ligand_id"] for row in ligands}
    expected_history = {(seed, receptor, ligand) for seed, _ in FROZEN_SEEDS for receptor in historical_ids for ligand in ligand_ids}
    observed_history = {(row["seed_id"], row["receptor_id"], row["ligand_id"]) for row in historical_rows}
    if observed_history != expected_history or len(historical_rows) != 7680:
        raise ValueError("Stage43 historical Stage32 score coverage differs")
    if any(row.get("status") != "ok" or row.get("pose_integrity_status") != "ok" for row in historical_rows):
        raise ValueError("Stage43 historical Stage32 score integrity differs")
    if len(historical_batches) != 48 or any(row["status"] != "ok" for row in historical_batches):
        raise ValueError("Stage43 historical Stage32 batch ledger differs")
    return receptors, new_receptors, historical_rows, ligands, {
        "status": "audit_only_ok", "target_id": "PPARG",
        "receptor_count": 96, "historical_receptor_count": 16,
        "new_receptor_count": 80, "ligand_count": 160,
        "label_counts": {"active": 80, "decoy": 80},
        "historical_pair_count": 7680, "new_pair_count": 38400,
        "combined_pair_count": 46080, "fresh_validation_rows": 0,
        "test_rows": 0,
    }


def collect_new_batches(
    root: Path, config: dict[str, Any], receptors: list[dict[str, str]],
    ligands: list[dict[str, str]], config_hash: str, amendment: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    run_directory = common.rooted_path(root, config["outputs"]["run_directory"])
    protocol = config["unidock"]
    ligand_ids = {row["ligand_id"] for row in ligands}
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for seed in config["inputs"]["seeds"]:
        for receptor in receptors:
            paths = common.batch_paths(run_directory, seed["seed_id"], receptor["conformer_id"])
            signatures = batch_signatures(config_hash, seed, receptor, ligands, protocol, amendment)
            value = rescue_checkpoint(root, paths, signatures, ligand_ids)
            if value is None:
                missing.append({"seed_id": seed["seed_id"], "receptor_id": receptor["conformer_id"]})
            else:
                batch_rows, summary = value
                rows.extend(dict(row) for row in batch_rows)
                summaries.append(dict(summary))
    return rows, summaries, missing


def matrix_rows(rows: list[dict[str, Any]], ligands: list[dict[str, str]], receptor_ids: list[str], aggregation: str) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        values.setdefault((row["ligand_id"], row["receptor_id"]), []).append(float(row["gpu_score"]))
    output: list[dict[str, Any]] = []
    for ligand in ligands:
        record: dict[str, Any] = {
            "ligand_id": ligand["ligand_id"], "label": ligand["label"],
            "selection_role": ligand["selection_role"], "split_group_id": ligand["split_group_id"],
        }
        for receptor_id in receptor_ids:
            scores = values.get((ligand["ligand_id"], receptor_id), [])
            if len(scores) != 3:
                raise ValueError(f"incomplete Stage43 seed values: {ligand['ligand_id']}/{receptor_id}")
            record[receptor_id] = statistics.median(scores) if aggregation == "median" else min(scores)
        output.append(record)
    return output


def finalize(
    root: Path, config_path: Path, config: dict[str, Any],
    all_receptors: list[dict[str, str]], new_receptors: list[dict[str, str]],
    historical_rows: list[dict[str, Any]], ligands: list[dict[str, str]],
    input_audit: dict[str, Any], executable_info: dict[str, Any] | None,
    executed: int, resumed: int, elapsed: float,
    selected_seed_ids: list[str], selected_receptor_ids: list[str],
) -> dict[str, Any]:
    config_hash = common.file_sha256(config_path)
    amendment = load_rescue_amendment(root)
    new_rows, new_summaries, missing = collect_new_batches(root, config, new_receptors, ligands, config_hash, amendment)
    outputs = config["outputs"]
    progress_path = common.rooted_path(root, outputs["progress_json"])
    progress = {
        "schema_version": "1.0", "experiment_id": config["experiment_id"],
        "status": "stage43_production_complete" if not missing else "stage43_partial_ok",
        "selected_seed_ids": selected_seed_ids, "selected_receptor_ids": selected_receptor_ids,
        "historical_completed_batch_count": 48,
        "new_completed_batch_count": len(new_summaries), "new_missing_batch_count": len(missing),
        "new_missing_batches": missing,
        "combined_completed_pair_count": len(historical_rows) + len(new_rows),
        "expected_combined_pair_count": 46080,
        "executed_batches_this_invocation": executed,
        "resumed_batches_this_invocation": resumed,
        "current_invocation_elapsed_seconds": elapsed,
        "fresh_validation_rows_read": 0, "test_rows_read": 0,
    }
    common.write_json(progress_path, progress)
    if missing:
        print(json.dumps(progress, indent=2, sort_keys=True))
        return progress
    rows = [dict(row, evidence_role="historical_stage32_reuse") for row in historical_rows]
    # Older completed checkpoints used the legacy helper's historical MK14
    # metadata label. Repair metadata only; docking values and hashes stay fixed.
    rows.extend(dict(row, target_id="PPARG", evidence_role="new_stage43_docking") for row in new_rows)
    unique = {(row["seed_id"], row["receptor_id"], row["ligand_id"]) for row in rows}
    if len(rows) != 46080 or len(unique) != 46080:
        raise ValueError("complete Stage43 pair coverage differs")
    seed_order = {seed_id: index for index, (seed_id, _) in enumerate(FROZEN_SEEDS)}
    receptor_ids = [row["conformer_id"] for row in all_receptors]
    receptor_order = {value: index for index, value in enumerate(receptor_ids)}
    ligand_order = {row["ligand_id"]: index for index, row in enumerate(ligands)}
    rows.sort(key=lambda row: (seed_order[row["seed_id"]], receptor_order[row["receptor_id"]], ligand_order[row["ligand_id"]]))
    if any(not math.isfinite(float(row["gpu_score"])) or abs(float(row["gpu_score"])) > 1000 for row in rows):
        raise ValueError("Stage43 contains an invalid score")
    scores_path = common.rooted_path(root, outputs["scores_csv"])
    batch_path = common.rooted_path(root, outputs["batch_runs_csv"])
    median_path = common.rooted_path(root, outputs["median_matrix_csv"])
    minimum_path = common.rooted_path(root, outputs["minimum_matrix_csv"])
    summary_path = common.rooted_path(root, outputs["summary_json"])
    common.write_csv(scores_path, rows)
    historical_batches = common.read_csv(common.rooted_path(root, config["inputs"]["stage32_batch_runs"]))
    batch_rows = [dict(
        row,
        evidence_role="historical_stage32_reuse",
        technical_rescue_applied=False,
        primary_base_seed=row["base_seed"],
        effective_base_seed=row["base_seed"],
    ) for row in historical_batches]
    batch_rows.extend({
        "seed_id": value["seed_id"], "base_seed": value["base_seed"],
        "receptor_id": value["receptor_id"], "ligand_count": value["ligand_count"],
        "elapsed_seconds": value["elapsed_seconds"], "score_minimum": value["score_minimum"],
        "score_maximum": value["score_maximum"],
        "known_warning_event_count": value["warning_adjudication"]["known_warning_event_count"],
        "unresolved_warning_event_count": value["warning_adjudication"]["unresolved_warning_event_count"],
        "pose_integrity_failure_count": value["pose_integrity_audit"]["failure_count"],
        "signature": value["signature"], "status": value["status"],
        "evidence_role": "new_stage43_docking",
        "technical_rescue_applied": bool(value.get("technical_rescue")),
        "primary_base_seed": value.get("technical_rescue", {}).get("primary_base_seed", value["base_seed"]),
        "effective_base_seed": value["base_seed"],
    } for value in new_summaries)
    batch_rows.sort(key=lambda row: (seed_order[row["seed_id"]], receptor_order[row["receptor_id"]]))
    common.write_csv(batch_path, batch_rows)
    common.write_csv(median_path, matrix_rows(rows, ligands, receptor_ids, "median"))
    common.write_csv(minimum_path, matrix_rows(rows, ligands, receptor_ids, "minimum"))
    new_elapsed = sum(float(value["elapsed_seconds"]) for value in new_summaries)
    result = {
        "schema_version": "1.0", "experiment_id": config["experiment_id"],
        "status": "stage43_pparg_md96_unidock_matrix_ok",
        "operation": "post-hoc PPARG Train-160 MD-96 matrix with 16 audited historical receptors and 80 newly docked receptors",
        "config": {"path": common.relative_path(root, config_path), "sha256": config_hash},
        "unidock_executable": executable_info, "input_audit": input_audit,
        "batch_count": 288, "historical_batch_count": 48, "new_batch_count": 240,
        "pair_count": 46080, "historical_pair_count": 7680, "new_pair_count": 38400,
        "new_batch_elapsed_seconds": new_elapsed,
        "new_pairs_per_batch_second": 38400 / new_elapsed,
        "technical_rescue_batch_count": sum(bool(value.get("technical_rescue")) for value in new_summaries),
        "technical_rescue_amendment": amendment if amendment.get("enabled") else None,
        "unresolved_warning_event_count": 0, "pose_integrity_failure_count": 0,
        "data_boundary": {"train_rows_read": 160, "fresh_validation_rows_read": 0, "test_rows_read": 0},
        "outputs": {
            "scores_csv": common.output_descriptor(root, scores_path),
            "batch_runs_csv": common.output_descriptor(root, batch_path),
            "median_matrix_csv": common.output_descriptor(root, median_path),
            "minimum_matrix_csv": common.output_descriptor(root, minimum_path),
            "progress_json": common.output_descriptor(root, progress_path),
        },
        "next_gate": "independently audit the matrix, then apply the unchanged Stage42f objective and frozen solver comparison",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    common.write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run(
    config_path: Path, root: Path, unidock: str | None, audit_only: bool,
    resume: bool, seed_ids: list[str] | None, receptor_ids: list[str] | None,
    finalize_only: bool,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = common.read_json(config_path)
    validate_config(config)
    amendment = load_rescue_amendment(root)
    all_receptors, new_receptors, historical_rows, ligands, input_audit = validate_inputs(root, config)
    seeds = common.selected_records([dict(row) for row in config["inputs"]["seeds"]], "seed_id", seed_ids)
    receptors = common.selected_records([dict(row) for row in new_receptors], "conformer_id", receptor_ids)
    selected_seed_ids = [row["seed_id"] for row in seeds]
    selected_receptor_ids = [row["conformer_id"] for row in receptors]
    if audit_only:
        result = {
            "schema_version": "1.0", "experiment_id": config["experiment_id"], **input_audit,
            "selected_seed_ids": selected_seed_ids, "selected_receptor_ids": selected_receptor_ids,
            "selected_new_batch_count": len(seeds) * len(receptors),
            "selected_new_pair_count": len(seeds) * len(receptors) * len(ligands),
            "operation": "input audit only; no GPU docking was started",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    executable_info = None
    executed = resumed = 0
    started = time.perf_counter()
    if not finalize_only:
        protocol = config["unidock"]
        executable_info = common.executable_evidence(unidock or protocol["executable"], protocol["required_package_version"])
        run_directory = common.rooted_path(root, config["outputs"]["run_directory"])
        config_hash = common.file_sha256(config_path)
        ligand_ids = {row["ligand_id"] for row in ligands}
        for seed in seeds:
            for receptor in receptors:
                paths = common.batch_paths(run_directory, seed["seed_id"], receptor["conformer_id"])
                paths["directory"].mkdir(parents=True, exist_ok=True)
                signatures = batch_signatures(config_hash, seed, receptor, ligands, protocol, amendment)
                checkpoint = rescue_checkpoint(root, paths, signatures, ligand_ids) if resume else None
                if checkpoint is not None:
                    resumed += 1
                    print(f"resume ok: {seed['seed_id']}/{receptor['conformer_id']}", flush=True)
                    continue
                print(f"running: {seed['seed_id']}/{receptor['conformer_id']}", flush=True)
                primary_signature, primary_seed, _ = signatures[0]
                if len(signatures) == 2:
                    rescue_signature, rescue_seed, _ = signatures[1]
                    print(
                        f"technical rescue: {seed['seed_id']}/{receptor['conformer_id']} "
                        f"with frozen fallback seed {rescue_seed}",
                        flush=True,
                    )
                    rows, summary = run_batch(root, paths, executable_info["resolved_executable"], receptor, ligands, protocol, seed["seed_id"], rescue_seed, rescue_signature)
                    summary["technical_rescue"] = {
                        "amendment_id": amendment["amendment_id"],
                        "trigger": amendment["reason"],
                        "primary_base_seed": primary_seed,
                        "effective_base_seed": rescue_seed,
                        "entire_batch_rerun": True,
                    }
                else:
                    rows, summary = run_batch(root, paths, executable_info["resolved_executable"], receptor, ligands, protocol, seed["seed_id"], primary_seed, primary_signature)
                rows, pose_audit = common.audit_batch_poses(root, ligands, rows)
                warning = common.classify_warning_log(paths["log"], pose_audit)
                summary["pose_integrity_audit"] = pose_audit
                summary["warning_adjudication"] = warning
                summary["status"] = "ok" if int(pose_audit["failure_count"]) == 0 and int(warning["unresolved_warning_event_count"]) == 0 else "technical_integrity_failed"
                common.write_csv(paths["scores"], rows)
                summary["scores_sha256"] = common.file_sha256(paths["scores"])
                common.write_json(paths["summary"], summary)
                if summary["status"] != "ok":
                    raise ValueError(f"Stage43 technical gate failed: {seed['seed_id']}/{receptor['conformer_id']}")
                executed += 1
                print(f"completed: {seed['seed_id']}/{receptor['conformer_id']} in {float(summary['elapsed_seconds']):.3f} s", flush=True)
    return finalize(root, config_path, config, all_receptors, new_receptors, historical_rows, ligands, input_audit, executable_info, executed, resumed, time.perf_counter() - started, selected_seed_ids, selected_receptor_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage43_pparg_md96_rank_sensitive_replication.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--unidock")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-id", action="append")
    parser.add_argument("--receptor-id", action="append")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    result = run(config_path, root, args.unidock, args.audit_only, args.resume, args.seed_id, args.receptor_id, args.finalize_only)
    return 0 if result["status"] in {"audit_only_ok", "stage43_partial_ok", "stage43_pparg_md96_unidock_matrix_ok"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
