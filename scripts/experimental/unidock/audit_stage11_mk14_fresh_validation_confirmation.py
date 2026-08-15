"""Independently audit the complete Stage 11 fresh-validation matrix."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

try:
    from . import audit_stage09_mk14_train696_production as stage09_audit
    from .run_stage07b_unidock_enhanced_confirmation import audit_batch_poses
    from .run_stage07c_unidock_warning_adjudication import classify_warning_log
    from .run_stage11_mk14_fresh_validation_confirmation import validate_inputs
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
    import audit_stage09_mk14_train696_production as stage09_audit
    from run_stage07b_unidock_enhanced_confirmation import audit_batch_poses
    from run_stage07c_unidock_warning_adjudication import classify_warning_log
    from run_stage11_mk14_fresh_validation_confirmation import validate_inputs
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


def run_audit(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    descriptor = dict(config["implementation"])["independent_auditor"]
    auditor_path = rooted_path(root, str(descriptor["path"]))
    if auditor_path.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 11 independent auditor path differs")
    if file_sha256(auditor_path) != str(descriptor["sha256"]).upper():
        raise ValueError("Stage 11 independent auditor hash differs")
    helper_descriptor = dict(config["implementation"])["stage09_auditor_helper"]
    helper_path = rooted_path(root, str(helper_descriptor["path"]))
    expected_helper = Path(__file__).with_name(
        "audit_stage09_mk14_train696_production.py"
    )
    if helper_path.resolve() != expected_helper.resolve() or file_sha256(
        helper_path
    ) != str(helper_descriptor["sha256"]).upper():
        raise ValueError("Stage 11 Stage 09 auditor helper identity differs")

    outputs = dict(config["outputs"])
    summary_path = rooted_path(root, str(outputs["summary_json"]))
    summary = read_json(summary_path)
    if summary.get("status") != "stage11_fresh_validation_unidock_matrix_ok":
        raise ValueError("Stage 11 source matrix did not pass")
    if str(summary["config"]["sha256"]).upper() != file_sha256(config_path):
        raise ValueError("Stage 11 source config hash differs")
    boundary = dict(summary["data_boundary"])
    if int(boundary["validation_rows_read"]) != 1576:
        raise ValueError("Stage 11 validation boundary differs")
    if int(boundary["train_score_rows_read"]) != 0 or int(
        boundary["test_rows_read"]
    ) != 0:
        raise ValueError("Stage 11 source crossed a frozen data boundary")

    source_outputs = dict(summary["outputs"])
    scores_path = verified_path(root, dict(source_outputs["scores_csv"]))
    batches_path = verified_path(root, dict(source_outputs["batch_runs_csv"]))
    median_path = verified_path(root, dict(source_outputs["median_matrix_csv"]))
    minimum_path = verified_path(root, dict(source_outputs["minimum_matrix_csv"]))
    verified_path(root, dict(source_outputs["progress_json"]))
    receptors, ligands, _ = validate_inputs(root, config_path, config)
    receptor_ids = [row["conformer_id"] for row in receptors]
    ligand_ids = {row["ligand_id"] for row in ligands}
    expected = dict(config["expected"])
    labels = Counter(row["label"] for row in ligands)
    if labels != Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    ):
        raise ValueError("Stage 11 independent labels differ")

    run_directory = rooted_path(root, str(outputs["run_directory"]))
    protocol = dict(config["unidock"])
    config_sha256 = file_sha256(config_path)
    recomputed_rows: list[dict[str, str]] = []
    known_warnings = 0
    batch_count = 0
    for seed in dict(config["inputs"])["seeds"]:
        seed_id = str(seed["seed_id"])
        base_seed = int(seed["base_seed"])
        for receptor in receptors:
            receptor_id = receptor["conformer_id"]
            paths = batch_paths(run_directory, seed_id, receptor_id)
            batch_summary = read_json(paths["summary"])
            rows = read_csv(paths["scores"])
            if batch_summary.get("status") != "ok":
                raise ValueError(f"Stage 11 batch failed: {seed_id}/{receptor_id}")
            signature = protocol_signature(
                config_sha256,
                seed_id,
                base_seed,
                receptor,
                ligands,
                protocol,
            )
            if batch_summary.get("signature") != signature:
                raise ValueError(
                    f"Stage 11 signature differs: {seed_id}/{receptor_id}"
                )
            if file_sha256(paths["scores"]) != str(
                batch_summary["scores_sha256"]
            ).upper():
                raise ValueError(
                    f"Stage 11 batch score hash differs: {seed_id}/{receptor_id}"
                )
            if file_sha256(paths["log"]) != str(batch_summary["log_sha256"]).upper():
                raise ValueError(
                    f"Stage 11 batch log hash differs: {seed_id}/{receptor_id}"
                )
            if len(rows) != len(ligands) or {
                row["ligand_id"] for row in rows
            } != ligand_ids:
                raise ValueError(
                    f"Stage 11 batch ligand grid differs: {seed_id}/{receptor_id}"
                )

            audited_rows, pose_audit = audit_batch_poses(root, ligands, rows)
            if pose_audit != dict(batch_summary["pose_integrity_audit"]):
                raise ValueError(
                    f"Stage 11 pose audit differs: {seed_id}/{receptor_id}"
                )
            warning = classify_warning_log(paths["log"], pose_audit)
            if warning != dict(batch_summary["warning_adjudication"]):
                raise ValueError(
                    f"Stage 11 warning audit differs: {seed_id}/{receptor_id}"
                )
            if int(warning["unresolved_warning_event_count"]) != 0:
                raise ValueError(
                    f"Stage 11 unresolved warning: {seed_id}/{receptor_id}"
                )
            known_warnings += int(warning["known_warning_event_count"])
            for source, audited in zip(rows, audited_rows, strict=True):
                output = rooted_path(root, source["output_pose_path"])
                if file_sha256(output) != source["output_pose_sha256"].upper():
                    raise ValueError(
                        f"Stage 11 pose hash differs: {source['ligand_id']}"
                    )
                score, pose_count = parse_vina_pose(output)
                if not math.isfinite(score) or abs(
                    score - float(source["gpu_score"])
                ) > 1e-9:
                    raise ValueError(
                        f"Stage 11 pose score differs: {source['ligand_id']}"
                    )
                if pose_count != 1 or int(source["pose_count"]) != 1:
                    raise ValueError(
                        f"Stage 11 pose count differs: {source['ligand_id']}"
                    )
                if audited["pose_integrity_status"] != "ok":
                    raise ValueError(
                        f"Stage 11 pose integrity failed: {source['ligand_id']}"
                    )
            recomputed_rows.extend(rows)
            batch_count += 1

    if batch_count != int(expected["batch_count"]) or len(recomputed_rows) != int(
        expected["pair_count"]
    ):
        raise ValueError("Stage 11 independent matrix dimensions differ")
    aggregate_rows = read_csv(scores_path)
    stage09_audit.compare_aggregate_rows(recomputed_rows, aggregate_rows)
    batch_rows = read_csv(batches_path)
    if len(batch_rows) != batch_count:
        raise ValueError("Stage 11 batch ledger count differs")
    if any(
        row["status"] != "ok"
        or int(row["unresolved_warning_event_count"]) != 0
        or int(row["pose_integrity_failure_count"]) != 0
        for row in batch_rows
    ):
        raise ValueError("Stage 11 batch ledger technical gate differs")
    stage09_audit.compare_matrix(
        read_csv(median_path), aggregate_rows, ligands, receptor_ids, "median"
    )
    stage09_audit.compare_matrix(
        read_csv(minimum_path), aggregate_rows, ligands, receptor_ids, "minimum"
    )
    if int(summary["known_warning_event_count"]) != known_warnings:
        raise ValueError("Stage 11 known-warning total differs")
    if int(summary["unresolved_warning_event_count"]) != 0 or int(
        summary["pose_integrity_failure_count"]
    ) != 0:
        raise ValueError("Stage 11 source technical totals differ")

    result = {
        "schema_version": "1.0",
        "audit_id": "stage11-mk14-fresh-validation-unidock113-independent-audit-v1",
        "status": "independent_stage11_fresh_validation_unidock_matrix_audit_ok",
        "config": {
            "path": relative_path(root, config_path),
            "sha256": file_sha256(config_path),
        },
        "source_summary": output_descriptor(root, summary_path),
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
        "data_boundary": {
            "validation_rows_read": len(ligands),
            "train_score_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            "scores_csv": output_descriptor(root, scores_path),
            "median_matrix_csv": output_descriptor(root, median_path),
            "minimum_matrix_csv": output_descriptor(root, minimum_path),
        },
        "next_gate": "evaluate only the four candidate subsets frozen before these validation scores existed",
        "decision_boundary": (
            "This audit establishes the Stage 11 validation matrix only. It does "
            "not itself establish QUBO benefit or quantum advantage."
        ),
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
