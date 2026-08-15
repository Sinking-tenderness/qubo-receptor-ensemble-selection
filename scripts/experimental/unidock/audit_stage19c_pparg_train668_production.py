"""Independently audit the complete Stage 19c PPARG Uni-Dock matrix."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from scripts.experimental.unidock import audit_stage09_mk14_train696_production as prior
from scripts.experimental.unidock import run_stage19c_pparg_train668_production as runner


def run_audit(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = runner.common.read_json(config_path)
    descriptor = dict(config["implementation"])["independent_auditor"]
    auditor_path = runner.common.rooted_path(root, str(descriptor["path"]))
    if auditor_path.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 19c independent auditor path differs")
    if runner.common.file_sha256(auditor_path) != str(descriptor["sha256"]).upper():
        raise ValueError("Stage 19c independent auditor hash differs")

    outputs = dict(config["outputs"])
    summary_path = runner.common.rooted_path(root, str(outputs["summary_json"]))
    summary = runner.common.read_json(summary_path)
    if summary.get("status") != "stage19c_pparg_train668_unidock_matrix_ok":
        raise ValueError("Stage 19c source matrix did not pass")
    if str(summary["config"]["sha256"]).upper() != runner.common.file_sha256(config_path):
        raise ValueError("Stage 19c source config hash differs")
    if any(int(value) != 0 for value in dict(summary["data_boundary"]).values()):
        raise ValueError("Stage 19c source crossed a data boundary")

    source_outputs = dict(summary["outputs"])
    scores_path = prior.checked_output(root, dict(source_outputs["scores_csv"]))
    batches_path = prior.checked_output(root, dict(source_outputs["batch_runs_csv"]))
    median_path = prior.checked_output(root, dict(source_outputs["median_matrix_csv"]))
    minimum_path = prior.checked_output(root, dict(source_outputs["minimum_matrix_csv"]))
    prior.checked_output(root, dict(source_outputs["progress_json"]))
    receptors, ligands, input_audit = runner.validate_inputs(root, config)
    receptor_ids = [row["conformer_id"] for row in receptors]
    ligand_ids = {row["ligand_id"] for row in ligands}
    labels = Counter(row["label"] for row in ligands)
    expected = dict(config["expected"])
    run_directory = runner.common.rooted_path(root, str(outputs["run_directory"]))
    protocol = dict(config["unidock"])
    config_sha256 = runner.common.file_sha256(config_path)
    recomputed_rows: list[dict[str, str]] = []
    known_warnings = 0
    batch_count = 0
    for seed in dict(config["inputs"])["seeds"]:
        seed_id = str(seed["seed_id"])
        base_seed = int(seed["base_seed"])
        for receptor in receptors:
            receptor_id = receptor["conformer_id"]
            paths = runner.common.batch_paths(run_directory, seed_id, receptor_id)
            batch_summary = runner.common.read_json(paths["summary"])
            rows = runner.common.read_csv(paths["scores"])
            if batch_summary.get("status") != "ok":
                raise ValueError(f"Stage 19c batch failed: {seed_id}/{receptor_id}")
            signature = runner.common.protocol_signature(
                config_sha256, seed_id, base_seed, receptor, ligands, protocol
            )
            if batch_summary.get("signature") != signature:
                raise ValueError(f"Stage 19c signature differs: {seed_id}/{receptor_id}")
            if runner.common.file_sha256(paths["scores"]) != str(batch_summary["scores_sha256"]).upper():
                raise ValueError(f"Stage 19c score hash differs: {seed_id}/{receptor_id}")
            if runner.common.file_sha256(paths["log"]) != str(batch_summary["log_sha256"]).upper():
                raise ValueError(f"Stage 19c log hash differs: {seed_id}/{receptor_id}")
            if len(rows) != len(ligands) or {row["ligand_id"] for row in rows} != ligand_ids:
                raise ValueError(f"Stage 19c batch ligand grid differs: {seed_id}/{receptor_id}")
            audited_rows, pose_audit = runner.common.audit_batch_poses(root, ligands, rows)
            if pose_audit != dict(batch_summary["pose_integrity_audit"]):
                raise ValueError(f"Stage 19c pose audit differs: {seed_id}/{receptor_id}")
            warning = runner.common.classify_warning_log(paths["log"], pose_audit)
            if warning != dict(batch_summary["warning_adjudication"]):
                raise ValueError(f"Stage 19c warning audit differs: {seed_id}/{receptor_id}")
            if int(warning["unresolved_warning_event_count"]) != 0:
                raise ValueError(f"Stage 19c unresolved warning: {seed_id}/{receptor_id}")
            known_warnings += int(warning["known_warning_event_count"])
            for source, audited in zip(rows, audited_rows, strict=True):
                output = runner.common.rooted_path(root, source["output_pose_path"])
                if runner.common.file_sha256(output) != source["output_pose_sha256"].upper():
                    raise ValueError(f"Stage 19c pose hash differs: {source['ligand_id']}")
                score, pose_count = prior.parse_vina_pose(output)
                if not math.isfinite(score) or abs(score - float(source["gpu_score"])) > 1e-9:
                    raise ValueError(f"Stage 19c pose score differs: {source['ligand_id']}")
                if pose_count != 1 or int(source["pose_count"]) != 1:
                    raise ValueError(f"Stage 19c pose count differs: {source['ligand_id']}")
                if audited["pose_integrity_status"] != "ok":
                    raise ValueError(f"Stage 19c pose integrity failed: {source['ligand_id']}")
            recomputed_rows.extend(rows)
            batch_count += 1

    if batch_count != int(expected["batch_count"]) or len(recomputed_rows) != int(expected["pair_count"]):
        raise ValueError("Stage 19c independent matrix dimensions differ")
    aggregate_rows = runner.common.read_csv(scores_path)
    prior.compare_aggregate_rows(recomputed_rows, aggregate_rows)
    batch_rows = runner.common.read_csv(batches_path)
    if len(batch_rows) != batch_count:
        raise ValueError("Stage 19c batch ledger count differs")
    if any(
        row["status"] != "ok"
        or int(row["unresolved_warning_event_count"]) != 0
        or int(row["pose_integrity_failure_count"]) != 0
        for row in batch_rows
    ):
        raise ValueError("Stage 19c batch ledger technical gate differs")
    prior.compare_matrix(
        runner.common.read_csv(median_path), aggregate_rows, ligands, receptor_ids, "median"
    )
    prior.compare_matrix(
        runner.common.read_csv(minimum_path), aggregate_rows, ligands, receptor_ids, "minimum"
    )
    if int(summary["known_warning_event_count"]) != known_warnings:
        raise ValueError("Stage 19c known-warning total differs")
    if int(summary["unresolved_warning_event_count"]) != 0 or int(summary["pose_integrity_failure_count"]) != 0:
        raise ValueError("Stage 19c source technical totals differ")

    result = {
        "schema_version": "1.0",
        "audit_id": "stage19c-pparg-train668-unidock113-independent-matrix-audit-v1",
        "status": "independent_stage19c_pparg_train668_unidock_matrix_audit_ok",
        "experiment_class": "posthoc_exploratory_train_only",
        "stage18e_confirmatory_gate": "closed_failed_14_of_24",
        "config": {"path": runner.common.relative_path(root, config_path), "sha256": config_sha256},
        "source_summary": runner.common.output_descriptor(root, summary_path),
        "receptor_count": len(receptors),
        "ligand_count": len(ligands),
        "seed_count": len(dict(config["inputs"])["seeds"]),
        "batch_count": batch_count,
        "pair_count": len(recomputed_rows),
        "label_counts": dict(sorted(labels.items())),
        "known_warning_event_count": known_warnings,
        "unresolved_warning_event_count": 0,
        "pose_integrity_failure_count": 0,
        "aggregate_score_rows_exact": True,
        "median_matrix_exact": True,
        "minimum_matrix_exact": True,
        "input_audit": input_audit,
        "data_boundary": {"validation_rows_read": 0, "test_rows_read": 0},
        "outputs": {
            "scores_csv": runner.common.output_descriptor(root, scores_path),
            "median_matrix_csv": runner.common.output_descriptor(root, median_path),
            "minimum_matrix_csv": runner.common.output_descriptor(root, minimum_path),
        },
        "next_gate": "perform frozen train-only QUBO, greedy, exact, and literature-baseline analysis",
        "decision_boundary": "This audit establishes only the exploratory PPARG Train-668 Uni-Dock matrix. It does not establish validation performance, QUBO benefit, or quantum advantage.",
    }
    output_path = runner.common.rooted_path(root, str(outputs["audit_json"]))
    runner.common.write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_audit(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
