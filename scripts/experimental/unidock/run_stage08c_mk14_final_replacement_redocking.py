"""Run the final one-receptor Stage 08c Uni-Dock admission gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.prepare_receptor import file_sha256
    from scripts.run_mk14_expanded_redocking_gate import run_checked
    from .run_stage07b_unidock_enhanced_confirmation import audit_batch_poses
    from .run_stage07c_unidock_warning_adjudication import classify_warning_log
    from .run_stage08_mk14_expanded16_redocking import (
        merge_receptor_manifests,
        prepare_case,
        runtime_evidence,
        summarize_receptor_redocking_gate,
        validate_protocol,
    )
    from .run_unidock_gpu_equivalence import (
        batch_paths,
        executable_evidence,
        protocol_signature,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        run_batch,
        validate_checkpoint,
        write_csv,
        write_json,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.prepare_receptor import file_sha256
    from scripts.run_mk14_expanded_redocking_gate import run_checked
    from scripts.experimental.unidock.run_stage07b_unidock_enhanced_confirmation import (
        audit_batch_poses,
    )
    from scripts.experimental.unidock.run_stage07c_unidock_warning_adjudication import (
        classify_warning_log,
    )
    from scripts.experimental.unidock.run_stage08_mk14_expanded16_redocking import (
        merge_receptor_manifests,
        prepare_case,
        runtime_evidence,
        summarize_receptor_redocking_gate,
        validate_protocol,
    )
    from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
        batch_paths,
        executable_evidence,
        protocol_signature,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        run_batch,
        validate_checkpoint,
        write_csv,
        write_json,
    )


def checked_record(root: Path, record: dict[str, object]) -> Path:
    path = rooted_path(root, str(record["path"]))
    if not path.is_file() or file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"input identity differs: {path}")
    return path


def verify_implementation(
    root: Path, config: dict[str, object], key: str, expected_path: Path
) -> None:
    descriptor = dict(config["implementation"])[key]
    path = rooted_path(root, str(descriptor["path"]))
    if path.resolve() != expected_path.resolve():
        raise ValueError(f"implementation path differs: {key}")
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"implementation hash differs: {key}")


def validate_inputs(
    root: Path, config: dict[str, object]
) -> tuple[
    dict[str, Path],
    list[dict[str, str]],
    dict[str, str],
    dict[str, object],
    dict[str, object],
]:
    boundary = dict(config["data_boundary"])
    if any(int(value) != 0 for value in boundary.values()):
        raise ValueError("Stage 08c redocking crossed a data boundary")
    validate_protocol(dict(config["unidock"]))
    implementation = {
        "runner": Path(__file__),
        "stage08_preparation_helper": Path(__file__).with_name(
            "run_stage08_mk14_expanded16_redocking.py"
        ),
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
    for key, path in implementation.items():
        verify_implementation(root, config, key, path)
    inputs = dict(config["inputs"])
    paths = {
        key: checked_record(root, value)
        for key, value in inputs.items()
        if isinstance(value, dict)
    }
    required = {
        "selection_preregistration",
        "selection_summary",
        "replacement_selection",
        "current15_summary",
        "current15_manifest",
        "reference_pdb",
        "prepare_receptor_script",
        "transform_sdf_script",
        "evaluate_rmsd_script",
    }
    if set(paths) != required:
        raise ValueError("Stage 08c input paths differ")
    if read_json(paths["selection_summary"]).get("status") != "stage08c_final_replacement_selection_ok":
        raise ValueError("Stage 08c structural selection did not pass")
    current_summary = read_json(paths["current15_summary"])
    if current_summary.get("status") != "stage08c_current15_manifest_ok":
        raise ValueError("Stage 08c current manifest did not pass")
    current_rows = read_csv(paths["current15_manifest"])
    expected = dict(config["expected"])
    if len(current_rows) != int(expected["current_receptor_count"]):
        raise ValueError("Stage 08c current receptor count differs")
    if [row["conformer_id"] for row in current_rows] != [
        str(value) for value in expected["current_receptor_ids"]
    ]:
        raise ValueError("Stage 08c current receptor order differs")
    for row in current_rows:
        receptor = rooted_path(root, row["receptor_pdbqt"])
        if row["status"] != "ok" or file_sha256(receptor) != row[
            "receptor_pdbqt_sha256"
        ].upper():
            raise ValueError(f"current receptor identity differs: {row['conformer_id']}")
    selected_rows = read_csv(paths["replacement_selection"])
    if len(selected_rows) != 1:
        raise ValueError("Stage 08c requires exactly one replacement")
    selected = selected_rows[0]
    case = dict(config["case"])
    if selected["conformer_id"] != str(case["conformer_id"]):
        raise ValueError("Stage 08c replacement case differs")
    for key in (
        "pdb_id",
        "chain",
        "selected_ligand_resname",
        "selected_ligand_resseq",
        "selected_ligand_icode",
        "selected_ligand_heavy_atom_count",
    ):
        if str(selected.get(key, "")) != str(case.get(key, "")):
            raise ValueError(f"Stage 08c replacement field differs: {key}")
    for path_key, hash_key in (
        ("pdb_path", "pdb_sha256"),
        ("aligned_pdb_path", "aligned_pdb_sha256"),
    ):
        path = rooted_path(root, selected[path_key])
        if not path.is_file() or file_sha256(path) != selected[hash_key].upper():
            raise ValueError(f"Stage 08c replacement coordinate differs: {path_key}")
    seeds = [
        (str(row["seed_id"]), int(row["base_seed"])) for row in inputs["seeds"]
    ]
    if seeds != [
        ("seed0", 20260801),
        ("seed1", 20260802),
        ("seed2", 20260803),
    ]:
        raise ValueError("Stage 08c paired seeds differ")
    audit = {
        "status": "audit_only_ok",
        "current_receptor_count": len(current_rows),
        "replacement_receptor_id": selected["conformer_id"],
        "replacement_receptor_count": 1,
        "expected_redocking_pair_count": 3,
        "final_receptor_count_if_pass": len(current_rows) + 1,
        "validation_rows": 0,
        "test_rows": 0,
    }
    return paths, current_rows, selected, case, audit


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
    paths, current_rows, selected, case, input_audit = validate_inputs(root, config)
    if audit_only:
        result = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "config": {
                "path": relative_path(root, config_path),
                "sha256": file_sha256(config_path),
            },
            **input_audit,
            "operation": "input audit only; no preparation or GPU docking was started",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    runtime = runtime_evidence(config)
    protocol = dict(config["unidock"])
    executable_info = executable_evidence(
        unidock or str(protocol["executable"]),
        str(protocol["required_package_version"]),
    )
    outputs = dict(config["outputs"])
    run_root = rooted_path(root, str(outputs["run_directory"]))
    run_root.mkdir(parents=True, exist_ok=True)
    receptor_row, prepared = prepare_case(
        root, config, paths, selected, case, run_root
    )
    receptor_row["source_pool"] = "stage08c_final_structural_replacement"
    replacement_manifest = rooted_path(
        root, str(outputs["replacement_receptor_manifest_csv"])
    )
    write_csv(replacement_manifest, [receptor_row])
    minimum_margin = float(dict(config["common_box_gate"])["minimum_crystal_pose_margin_angstrom"])
    if float(prepared["box_audit"]["minimum_margin_angstrom"]) < minimum_margin:
        raise ValueError("Stage 08c common-box gate failed")

    config_sha256 = file_sha256(config_path)
    threshold = float(dict(config["redocking_gate"])["maximum_rmsd_angstrom"])
    ligand = {
        "ligand_id": str(prepared["case_id"]),
        "label": "cognate_control",
        "selection_role": "technical_final_replacement_redocking",
        "pdbqt_path": str(prepared["ligand_pdbqt"]),
        "pdbqt_sha256": str(prepared["ligand_pdbqt_sha256"]),
    }
    all_rows: list[dict[str, object]] = []
    executed = 0
    resumed = 0
    for seed in dict(config["inputs"])["seeds"]:
        seed_id = str(seed["seed_id"])
        base_seed = int(seed["base_seed"])
        batch = batch_paths(
            run_root / "redocking", seed_id, str(receptor_row["conformer_id"])
        )
        batch["directory"].mkdir(parents=True, exist_ok=True)
        signature = protocol_signature(
            config_sha256,
            seed_id,
            base_seed,
            receptor_row,
            [ligand],
            protocol,
        )
        checkpoint = (
            validate_checkpoint(root, batch, signature, {str(ligand["ligand_id"])})
            if resume
            else None
        )
        if checkpoint is not None:
            rows, batch_summary = checkpoint
            if "warning_adjudication" not in batch_summary or not all(
                "pose_integrity_status" in row for row in rows
            ):
                checkpoint = None
        if checkpoint is None:
            print(f"running: {seed_id}/{receptor_row['conformer_id']}", flush=True)
            rows, batch_summary = run_batch(
                root,
                batch,
                str(executable_info["resolved_executable"]),
                receptor_row,
                [ligand],
                protocol,
                seed_id,
                base_seed,
                signature,
            )
            rows, pose_audit = audit_batch_poses(root, [ligand], rows)
            warning = classify_warning_log(batch["log"], pose_audit)
            batch_summary["pose_integrity"] = pose_audit
            batch_summary["warning_adjudication"] = warning
            write_csv(batch["scores"], rows)
            batch_summary["scores_sha256"] = file_sha256(batch["scores"])
            write_json(batch["summary"], batch_summary)
            executed += 1
        else:
            rows, batch_summary = checkpoint
            resumed += 1
            print(f"resume ok: {seed_id}/{receptor_row['conformer_id']}", flush=True)
        output_pose = rooted_path(root, str(rows[0]["output_pose_path"]))
        evaluation_dir = batch["directory"] / "rmsd"
        evaluation_path = evaluation_dir / "summary.json"
        pose_table = evaluation_dir / "poses.csv"
        run_checked(
            [
                sys.executable,
                str(paths["evaluate_rmsd_script"]),
                "--case-id",
                f"{prepared['case_id']}__{seed_id}",
                "--reference-sdf",
                str(rooted_path(root, str(prepared["reference_sdf"]))),
                "--docked-pdbqt",
                str(output_pose),
                "--pose-table-output",
                str(pose_table),
                "--summary-output",
                str(evaluation_path),
                "--success-threshold",
                str(threshold),
            ],
            f"Stage 08c RMSD evaluation for {seed_id}",
        )
        evaluation = read_json(evaluation_path)
        warning = dict(batch_summary["warning_adjudication"])
        pose_audit = dict(batch_summary["pose_integrity"])
        all_rows.append(
            {
                "case_id": prepared["case_id"],
                "conformer_id": receptor_row["conformer_id"],
                "seed_id": seed_id,
                "base_seed": base_seed,
                "reference_ligand": prepared["reference_ligand"],
                "reference_sdf": prepared["reference_sdf"],
                "reference_sdf_sha256": prepared["reference_sdf_sha256"],
                "receptor_pdbqt": receptor_row["receptor_pdbqt"],
                "receptor_pdbqt_sha256": receptor_row["receptor_pdbqt_sha256"],
                "docked_pdbqt": relative_path(root, output_pose),
                "docked_pdbqt_sha256": file_sha256(output_pose),
                "top_ranked_affinity_kcal_per_mol": evaluation[
                    "top_ranked_affinity_kcal_per_mol"
                ],
                "top_ranked_rmsd_angstrom": evaluation[
                    "top_ranked_rmsd_angstrom"
                ],
                "top_ranked_pose_success": evaluation["top_ranked_pose_success"],
                "pose_integrity_failure_count": pose_audit["failure_count"],
                "known_warning_event_count": warning["known_warning_event_count"],
                "unresolved_warning_event_count": warning[
                    "unresolved_warning_event_count"
                ],
                "status": "ok",
            }
        )
    if len(all_rows) != 3:
        raise ValueError("Stage 08c redocking pair count differs")
    gate = dict(config["redocking_gate"])
    gate_rows = summarize_receptor_redocking_gate(
        all_rows,
        [str(receptor_row["conformer_id"])],
        threshold,
        int(gate["minimum_successful_seeds_per_receptor"]),
    )
    unresolved = sum(int(row["unresolved_warning_event_count"]) for row in all_rows)
    pose_failures = sum(int(row["pose_integrity_failure_count"]) for row in all_rows)
    passed = bool(gate_rows[0]["gate_pass"]) and unresolved == 0 and pose_failures == 0
    status = (
        "stage08c_final_replacement_redocking_gate_ok"
        if passed
        else "stage08c_final_replacement_redocking_gate_failed"
    )
    results_path = rooted_path(root, str(outputs["redocking_results_csv"]))
    write_csv(results_path, all_rows)
    final_manifest = rooted_path(root, str(outputs["final_receptor_manifest_csv"]))
    final_rows = merge_receptor_manifests(
        [dict(row) for row in current_rows], [receptor_row]
    )
    if passed:
        write_csv(final_manifest, final_rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {
            "path": relative_path(root, config_path),
            "sha256": config_sha256,
        },
        "runtime": runtime,
        "unidock_executable": executable_info,
        "frozen_protocol": protocol,
        "input_audit": input_audit,
        "case_preparation": prepared,
        "redocking_pair_count": len(all_rows),
        "executed_batches_this_invocation": executed,
        "resumed_batches_this_invocation": resumed,
        "receptor_gate_results": gate_rows,
        "replacement_pass": passed,
        "unresolved_warning_event_count": unresolved,
        "pose_integrity_failure_count": pose_failures,
        "final_receptor_ids": (
            [str(row["conformer_id"]) for row in final_rows]
            if final_manifest.is_file()
            else []
        ),
        "data_boundary": {
            "benchmark_ligand_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "previous_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            "replacement_receptor_manifest_csv": {
                "path": relative_path(root, replacement_manifest),
                "sha256": file_sha256(replacement_manifest),
            },
            "redocking_results_csv": {
                "path": relative_path(root, results_path),
                "sha256": file_sha256(results_path),
            },
            "final_receptor_manifest_csv": (
                {
                    "path": relative_path(root, final_manifest),
                    "sha256": file_sha256(final_manifest),
                }
                if final_manifest.is_file()
                else None
            ),
        },
        "next_gate": (
            "run the independent Stage 08c audit before creating the Train-696 production bundle"
            if passed
            else "stop and continue the preregistered nonredundant replacement cascade"
        ),
        "decision_boundary": config["decision_boundary"],
    }
    summary_path = rooted_path(root, str(outputs["summary_json"]))
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--unidock", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(
        args.config,
        args.root,
        args.unidock,
        args.audit_only,
        args.resume,
    )
    return 0 if result["status"] in {
        "audit_only_ok",
        "stage08c_final_replacement_redocking_gate_ok",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
