"""Independently audit the complete Stage 09 Uni-Dock production matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

try:
    from .run_stage07b_unidock_enhanced_confirmation import audit_batch_poses
    from .run_stage07c_unidock_warning_adjudication import classify_warning_log
    from .run_unidock_gpu_equivalence import (
        batch_paths,
        file_sha256,
        output_descriptor,
        parse_vina_pose,
        protocol_signature,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        verified_path,
        write_json,
    )
except ImportError:
    from run_stage07b_unidock_enhanced_confirmation import audit_batch_poses
    from run_stage07c_unidock_warning_adjudication import classify_warning_log
    from run_unidock_gpu_equivalence import (
        batch_paths,
        file_sha256,
        output_descriptor,
        parse_vina_pose,
        protocol_signature,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        verified_path,
        write_json,
    )


def checked_output(root: Path, descriptor: dict[str, object]) -> Path:
    return verified_path(root, descriptor)


def compare_aggregate_rows(
    batch_rows: list[dict[str, str]], aggregate_rows: list[dict[str, str]]
) -> None:
    keys = ("seed_id", "receptor_id", "ligand_id")
    batch_by_key = {tuple(row[key] for key in keys): row for row in batch_rows}
    aggregate_by_key = {
        tuple(row[key] for key in keys): row for row in aggregate_rows
    }
    if len(batch_by_key) != len(batch_rows) or len(aggregate_by_key) != len(
        aggregate_rows
    ):
        raise ValueError("Stage 09 aggregate contains duplicate keys")
    if set(batch_by_key) != set(aggregate_by_key):
        raise ValueError("Stage 09 aggregate key grid differs")
    compared = (
        "base_seed",
        "label",
        "selection_role",
        "gpu_score",
        "pose_count",
        "status",
        "output_pose_path",
        "output_pose_sha256",
        "input_atom_count",
        "output_atom_count",
        "atom_count_match",
        "atom_types_match",
        "single_pose_match",
        "pose_integrity_status",
    )
    for key, batch in batch_by_key.items():
        aggregate = aggregate_by_key[key]
        if any(batch[field] != aggregate[field] for field in compared):
            raise ValueError(f"Stage 09 aggregate row differs: {key}")


def compare_matrix(
    matrix_rows: list[dict[str, str]],
    score_rows: list[dict[str, str]],
    ligands: list[dict[str, str]],
    receptor_ids: list[str],
    aggregation: str,
) -> None:
    if [row["ligand_id"] for row in matrix_rows] != [
        row["ligand_id"] for row in ligands
    ]:
        raise ValueError(f"Stage 09 {aggregation} matrix ligand order differs")
    scores: dict[tuple[str, str], list[float]] = {}
    for row in score_rows:
        key = (row["ligand_id"], row["receptor_id"])
        scores.setdefault(key, []).append(float(row["gpu_score"]))
    ligand_by_id = {row["ligand_id"]: row for row in ligands}
    for row in matrix_rows:
        ligand = ligand_by_id[row["ligand_id"]]
        if row["label"] != ligand["label"] or row["selection_role"] != ligand[
            "selection_role"
        ]:
            raise ValueError(f"Stage 09 {aggregation} matrix labels differ")
        for receptor_id in receptor_ids:
            values = scores.get((row["ligand_id"], receptor_id), [])
            if len(values) != 3:
                raise ValueError("Stage 09 matrix seed count differs")
            expected = (
                statistics.median(values) if aggregation == "median" else min(values)
            )
            if abs(float(row[receptor_id]) - expected) > 1e-12:
                raise ValueError(
                    f"Stage 09 {aggregation} value differs: "
                    f"{row['ligand_id']}/{receptor_id}"
                )


def run_audit(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    auditor_descriptor = dict(config["implementation"])["independent_auditor"]
    auditor_path = rooted_path(root, str(auditor_descriptor["path"]))
    if auditor_path.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 09 independent auditor path differs")
    if file_sha256(auditor_path) != str(auditor_descriptor["sha256"]).upper():
        raise ValueError("Stage 09 independent auditor hash differs")

    outputs = dict(config["outputs"])
    summary_path = rooted_path(root, str(outputs["summary_json"]))
    summary = read_json(summary_path)
    if summary.get("status") != "stage09_train696_unidock_matrix_ok":
        raise ValueError("Stage 09 source matrix did not pass")
    if str(summary["config"]["sha256"]).upper() != file_sha256(config_path):
        raise ValueError("Stage 09 source config hash differs")
    if any(int(value) != 0 for value in dict(summary["data_boundary"]).values()):
        raise ValueError("Stage 09 source crossed a data boundary")

    source_outputs = dict(summary["outputs"])
    scores_path = checked_output(root, dict(source_outputs["scores_csv"]))
    batches_path = checked_output(root, dict(source_outputs["batch_runs_csv"]))
    median_path = checked_output(root, dict(source_outputs["median_matrix_csv"]))
    minimum_path = checked_output(root, dict(source_outputs["minimum_matrix_csv"]))
    checked_output(root, dict(source_outputs["progress_json"]))
    inputs = dict(config["inputs"])
    receptor_path = verified_path(root, dict(inputs["receptor_manifest"]))
    ligand_path = verified_path(root, dict(inputs["ligand_manifest"]))
    receptors = read_csv(receptor_path)
    ligands = read_csv(ligand_path)
    receptor_ids = [row["conformer_id"] for row in receptors]
    ligand_ids = {row["ligand_id"] for row in ligands}
    expected = dict(config["expected"])
    if len(receptors) != int(expected["receptor_count"]) or len(ligands) != int(
        expected["ligand_count"]
    ):
        raise ValueError("Stage 09 independent input counts differ")
    labels = Counter(row["label"] for row in ligands)
    if labels != Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    ):
        raise ValueError("Stage 09 independent labels differ")

    run_directory = rooted_path(root, str(outputs["run_directory"]))
    protocol = dict(config["unidock"])
    config_sha256 = file_sha256(config_path)
    recomputed_rows: list[dict[str, str]] = []
    known_warnings = 0
    batch_count = 0
    for seed in inputs["seeds"]:
        seed_id = str(seed["seed_id"])
        base_seed = int(seed["base_seed"])
        for receptor in receptors:
            receptor_id = receptor["conformer_id"]
            paths = batch_paths(run_directory, seed_id, receptor_id)
            batch_summary = read_json(paths["summary"])
            rows = read_csv(paths["scores"])
            if batch_summary.get("status") != "ok":
                raise ValueError(f"Stage 09 batch failed: {seed_id}/{receptor_id}")
            signature = protocol_signature(
                config_sha256,
                seed_id,
                base_seed,
                receptor,
                ligands,
                protocol,
            )
            if batch_summary.get("signature") != signature:
                raise ValueError(f"Stage 09 signature differs: {seed_id}/{receptor_id}")
            if file_sha256(paths["scores"]) != str(
                batch_summary["scores_sha256"]
            ).upper():
                raise ValueError(f"Stage 09 batch score hash differs: {seed_id}/{receptor_id}")
            if file_sha256(paths["log"]) != str(batch_summary["log_sha256"]).upper():
                raise ValueError(f"Stage 09 batch log hash differs: {seed_id}/{receptor_id}")
            if len(rows) != len(ligands) or {row["ligand_id"] for row in rows} != ligand_ids:
                raise ValueError(f"Stage 09 batch ligand grid differs: {seed_id}/{receptor_id}")

            audited_rows, pose_audit = audit_batch_poses(root, ligands, rows)
            if pose_audit != dict(batch_summary["pose_integrity_audit"]):
                raise ValueError(f"Stage 09 pose audit differs: {seed_id}/{receptor_id}")
            warning = classify_warning_log(paths["log"], pose_audit)
            if warning != dict(batch_summary["warning_adjudication"]):
                raise ValueError(f"Stage 09 warning audit differs: {seed_id}/{receptor_id}")
            if int(warning["unresolved_warning_event_count"]) != 0:
                raise ValueError(f"Stage 09 unresolved warning: {seed_id}/{receptor_id}")
            known_warnings += int(warning["known_warning_event_count"])
            for source, audited in zip(rows, audited_rows, strict=True):
                output = rooted_path(root, source["output_pose_path"])
                if file_sha256(output) != source["output_pose_sha256"].upper():
                    raise ValueError(f"Stage 09 pose hash differs: {source['ligand_id']}")
                score, pose_count = parse_vina_pose(output)
                if not math.isfinite(score) or abs(score - float(source["gpu_score"])) > 1e-9:
                    raise ValueError(f"Stage 09 pose score differs: {source['ligand_id']}")
                if pose_count != 1 or int(source["pose_count"]) != 1:
                    raise ValueError(f"Stage 09 pose count differs: {source['ligand_id']}")
                if audited["pose_integrity_status"] != "ok":
                    raise ValueError(f"Stage 09 pose integrity failed: {source['ligand_id']}")
            recomputed_rows.extend(rows)
            batch_count += 1

    if batch_count != int(expected["batch_count"]) or len(recomputed_rows) != int(
        expected["pair_count"]
    ):
        raise ValueError("Stage 09 independent matrix dimensions differ")
    aggregate_rows = read_csv(scores_path)
    compare_aggregate_rows(recomputed_rows, aggregate_rows)
    batch_rows = read_csv(batches_path)
    if len(batch_rows) != batch_count:
        raise ValueError("Stage 09 batch ledger count differs")
    if any(
        row["status"] != "ok"
        or int(row["unresolved_warning_event_count"]) != 0
        or int(row["pose_integrity_failure_count"]) != 0
        for row in batch_rows
    ):
        raise ValueError("Stage 09 batch ledger technical gate differs")
    compare_matrix(
        read_csv(median_path), aggregate_rows, ligands, receptor_ids, "median"
    )
    compare_matrix(
        read_csv(minimum_path), aggregate_rows, ligands, receptor_ids, "minimum"
    )
    if int(summary["known_warning_event_count"]) != known_warnings:
        raise ValueError("Stage 09 known-warning total differs")
    if int(summary["unresolved_warning_event_count"]) != 0 or int(
        summary["pose_integrity_failure_count"]
    ) != 0:
        raise ValueError("Stage 09 source technical totals differ")

    result = {
        "schema_version": "1.0",
        "audit_id": "stage09-mk14-train696-unidock113-independent-matrix-audit-v1",
        "status": "independent_stage09_train696_unidock_matrix_audit_ok",
        "config": {
            "path": relative_path(root, config_path),
            "sha256": file_sha256(config_path),
        },
        "source_summary": output_descriptor(root, summary_path),
        "receptor_count": len(receptors),
        "ligand_count": len(ligands),
        "seed_count": len(inputs["seeds"]),
        "batch_count": batch_count,
        "pair_count": len(recomputed_rows),
        "label_counts": dict(sorted(labels.items())),
        "known_warning_event_count": known_warnings,
        "unresolved_warning_event_count": 0,
        "pose_integrity_failure_count": 0,
        "aggregate_score_rows_exact": True,
        "median_matrix_exact": True,
        "minimum_matrix_exact": True,
        "data_boundary": {
            "validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            "scores_csv": output_descriptor(root, scores_path),
            "median_matrix_csv": output_descriptor(root, median_path),
            "minimum_matrix_csv": output_descriptor(root, minimum_path),
        },
        "next_gate": "perform preregistered train-only receptor-subset and QUBO-versus-classical optimization analysis",
        "decision_boundary": "This audit establishes only the complete Train-696 Uni-Dock score matrix. It does not establish validation performance, QUBO benefit, or quantum advantage.",
    }
    output_path = rooted_path(root, str(outputs["audit_json"]))
    write_json(output_path, result)
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
